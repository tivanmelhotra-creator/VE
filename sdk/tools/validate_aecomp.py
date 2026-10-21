"""
DEV TOOL: measure how faithfully mograph.aecomp reproduces a reference After Effects element.

    python3 tools/validate_aecomp.py comp.json preview.mp4 [--sheet out.jpg] [--n 8]

Samples n instants, renders the comp (white-on-black silhouettes) and the vendor preview at the same size,
thresholds both and reports IoU per instant + mean.  The sheet shows  reference | ours | difference.
Only numbers and our own renders are kept; the vendor preview is read, never copied into the repo.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mograph.aecomp.render import render_bgra  # noqa: E402


def ref_frame(mp4, t, W, H):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.4f}", "-i", mp4, "-frames:v", "1", "-vf",
                        f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    if len(r.stdout) < W * H:
        return None
    return np.frombuffer(r.stdout[:W * H], np.uint8).reshape(H, W)


def bg_level(mp4, W, H):
    f = ref_frame(mp4, 0.0, W, H)
    return int(np.median(f)) if f is not None else 0


def probe(mp4):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", mp4], capture_output=True, text=True)
    w, h = (int(x) for x in r.stdout.strip().split(",")[:2])
    return w, h


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def fscore(a, b, tol=3):
    """line-art F-score: share of our pixels within tol px of the reference and vice versa (chamfer tolerance)."""
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    prec = float((db[a] <= tol).mean())
    rec = float((da[b] <= tol).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def _masks(comp, ts, W, H, k, dx, dy, dt):
    """our silhouettes for a framing (scale k of the comp height, centre offset dx, dy) and time shift dt."""
    out = []
    cw, ch = float(comp["w"]), float(comp["h"])
    s = k * H / ch
    for t in ts:
        S = render_bgra(comp, max(0.0, float(t) + dt), int(round(cw * s)), int(round(ch * s)))[..., 3] > 60
        m = np.zeros((H, W), bool)
        x0 = int(round(W / 2 + dx - S.shape[1] / 2))
        y0 = int(round(H / 2 + dy - S.shape[0] / 2))
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W, x0 + S.shape[1]), min(H, y0 + S.shape[0])
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = S[ya - y0:yb - y0, xa - x0:xb - x0]
        out.append(m)
    return out


def measure(comp, mp4, n=8, W=None, H=None, sheet=None, register=True):
    """IoU of our render vs the reference.  The vendor previews frame/offset each element differently, so the
    framing (scale + centre) and a global time shift are REGISTERED first (grid search), then reported."""
    if W is None:
        W, H = probe(mp4)
    dur = float(comp.get("dur", 1.0))
    bg = bg_level(mp4, W, H)
    ts = [float(t) for t in np.linspace(0.08 * dur, 0.92 * dur, n)]
    refs = [ref_frame(mp4, t, W, H) for t in ts]
    keep = [i for i, r in enumerate(refs) if r is not None]
    ts = [ts[i] for i in keep]
    refs = [refs[i] for i in keep]
    rms = [np.abs(r.astype(int) - bg) > 40 for r in refs]
    best = (1.0, 0.0, 0.0, 0.0)
    if register:
        # reference silhouette bbox centre → initial offset
        allm = np.any(np.stack(rms), 0)
        ys, xs = np.nonzero(allm)
        cx, cy = (xs.mean() - W / 2, ys.mean() - H / 2) if len(xs) else (0.0, 0.0)
        score = -1.0

        def grid():
            for k in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.78):
                for dt in (0.0, 1 / 30.0, 2 / 30.0):
                    yield k, dt
            k0, dt0 = best[0], best[3]                     # refine around the coarse optimum
            for k in np.linspace(k0 * 0.9, k0 * 1.1, 9):
                for dt in (dt0 - 0.033, dt0, dt0 + 0.033):
                    yield float(k), dt
        def _sc(ms):
            return float(np.mean([_iou(cv2.dilate(a.astype(np.uint8), np.ones((5, 5))) > 0,
                                       cv2.dilate(b.astype(np.uint8), np.ones((5, 5))) > 0) for a, b in zip(ms, rms)]))
        for k, dt in grid():
            ms0 = _masks(comp, ts, W, H, k, 0.0, 0.0, dt)
            ours = np.any(np.stack(ms0), 0)
            oy, ox = np.nonzero(ours)
            dx, dy = (cx - (ox.mean() - W / 2), cy - (oy.mean() - H / 2)) if len(ox) else (0.0, 0.0)
            # candidates: centroid-registered AND un-shifted (centroids mislead when the elements fade unevenly)
            for ddx, ddy, ms in ((dx, dy, _masks(comp, ts, W, H, k, dx, dy, dt)), (0.0, 0.0, ms0)):
                sc = _sc(ms)
                if sc > score:
                    score, best = sc, (k, ddx, ddy, dt)
    k, dx, dy, dt = best
    ms = _masks(comp, ts, W, H, k, dx, dy, dt)
    ious, rows, fs = [], [], []
    for t, ref, rm, ours in zip(ts, refs, rms, ms):
        fs.append(fscore(ours, rm))
        d3 = lambda m: (cv2.dilate(m.astype(np.uint8), np.ones((3, 3))) > 0)  # noqa: E731  (1-px AA tolerance)
        iou = _iou(d3(ours), d3(rm))
        ious.append(iou)
        if sheet:
            o = (ours * 255).astype(np.uint8)
            r = (rm * 255).astype(np.uint8)
            d = np.stack([r, o, np.zeros_like(r)], -1)
            cell = np.concatenate([cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR), d], 0)
            cv2.putText(cell, f"{t:.2f}s {iou:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            rows.append(cell)
    if sheet and rows:
        cv2.imwrite(sheet, np.concatenate(rows, 1))
    return ious, {"scale": round(k, 3), "dx": round(dx, 1), "dy": round(dy, 1), "dt": round(dt, 3),
                  "f3": round(float(np.mean(fs)) if fs else 0.0, 3)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("comp")
    ap.add_argument("mp4")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sheet")
    a = ap.parse_args(argv)
    comp = json.load(open(a.comp))
    ious, reg = measure(comp, a.mp4, a.n, sheet=a.sheet)
    print(json.dumps({"iou": [round(x, 3) for x in ious], "mean": round(float(np.mean(ious)) if ious else 0, 3),
                      "framing": reg}))


if __name__ == "__main__":
    main()
