"""
mograph.pro.stabilize — v22 RANGE-AWARE VIDEO STABILISATION of the A-roll (idea: Concat concat-host/stabilize.rs).

Hand-held talking heads shake; world-locked titles (follow 0.12–0.18) then look glued to a moving world.  This removes
the shake from the plate itself:

  1. ANALYSIS (once, cached in <video>.v14/stab.json): only the source stretches the edit USES
     (source.segments ± 1.5 s margin for the smoothing) are decoded small + grey (360 px wide).  Corners are tracked
     frame → frame (Shi-Tomasi + pyramidal LK), the SPEAKER IS MASKED OUT with the cached v14 matte (her gestures
     are not camera motion), and a similarity transform (dx, dy, rotation, scale) is fitted with RANSAC.
     Making a segment longer only analyses the new frames.
  2. CORRECTION (every render, free): the cumulative path is smoothed with a centred Gaussian (zero lag, `smooth`
     seconds) per stretch; correction = smooth − raw, × `strength`, plus a fixed `crop` zoom that hides the moving
     borders.  It is folded into the virtual-camera matrix (Renderer.matrix), so the plate, the matte, every anchor
     and every tracked layer move together — nothing drifts.

Spec:   "source": {"stabilize": true}
        "source": {"stabilize": {"strength": 0.85, "smooth": 0.8, "crop": 1.05, "mask_speaker": true}}
CLI:    python3 -m mograph.pro stabilize edit.json        (analyse now + print shake before / after)
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

MARGIN = 1.5
WORK_W = 360
DEF = {"strength": 0.85, "smooth": 0.8, "crop": 1.05, "mask_speaker": True}


def ranges_for(segments, sfps, n, margin=MARGIN):
    """source segments [[a, b], …] (seconds) → merged frame ranges [f0, f1) including the smoothing margin."""
    rs = sorted((max(0, int((a - margin) * sfps)), min(n, int(math.ceil((b + margin) * sfps)) + 1))
                for a, b in segments)
    out = []
    for a, b in rs:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _sim(prev, cur, mask=None):
    """similarity prev→cur in work px: (dx, dy, d_angle, d_log_scale) or None."""
    p0 = cv2.goodFeaturesToTrack(prev, 220, 0.01, 8, mask=mask, blockSize=7)
    if p0 is None or len(p0) < 12:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
    good = st.reshape(-1) == 1
    if good.sum() < 10:
        return None
    M, _ = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        return None
    return (float(M[0, 2]), float(M[1, 2]), float(math.atan2(M[1, 0], M[0, 0])),
            float(math.log(max(1e-6, math.hypot(M[0, 0], M[1, 0])))))


def analyse(video, segments=None, mask_speaker=True, log=print):
    """→ ({frame i: [dx, dy, da, dls]} motion i-1 → i in SOURCE px, fps, n, w, h) — cached + incremental."""
    cap = cv2.VideoCapture(video)
    sfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw, sh = int(cap.get(3)), int(cap.get(4))
    segments = segments or [[0, n / sfps]]
    d = video + ".v14"
    os.makedirs(d, exist_ok=True)
    cp = os.path.join(d, "stab.json")
    cache = {}
    if os.path.exists(cp):
        try:
            c = json.load(open(cp))
            if c.get("n") == n and c.get("mask") == bool(mask_speaker):
                cache = {int(k): v for k, v in c["motion"].items()}
        except Exception:  # noqa: BLE001
            cache = {}
    k = WORK_W / sw
    ww, wh = WORK_W, int(round(sh * k))
    todo = [(a, b) for a, b in ranges_for(segments, sfps, n) if any(i not in cache for i in range(a + 1, b))]
    new = 0
    for a, b in todo:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        prev = None
        for i in range(a, b):
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(fr, (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if prev is not None and i not in cache:
                mask = None
                if mask_speaker:
                    mp = os.path.join(d, "matte", f"{i - 1:05d}.png")
                    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.exists(mp) else None
                    if m is not None:
                        m = cv2.resize(m, (ww, wh), interpolation=cv2.INTER_AREA)
                        mask = np.where(cv2.dilate(m, np.ones((9, 9), np.uint8)) > 40, 0, 255).astype(np.uint8)
                        if (mask > 0).mean() < 0.12:     # speaker fills the frame → track everything
                            mask = None
                r = _sim(prev, g, mask)
                cache[i] = [0.0, 0.0, 0.0, 0.0] if r is None else [r[0] / k, r[1] / k, r[2], r[3]]
                new += 1
            prev = g
    if new:
        json.dump({"n": n, "fps": sfps, "w": sw, "h": sh, "mask": bool(mask_speaker),
                   "motion": {str(i): [round(x, 5) for x in v] for i, v in sorted(cache.items())}}, open(cp, "w"))
        log(f"[stabilize] analysed {new} new frame(s) in {len(todo)} stretch(es) → {cp}")
    return cache, sfps, n, sw, sh


def _gauss(a, sigma):
    if sigma <= 0 or len(a) < 3:
        return a.copy()
    r = max(1, int(sigma * 3))
    kk = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    kk /= kk.sum()
    pad = np.pad(a, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], kk, mode="valid") for c in range(a.shape[1])], 1)


class Stabilizer:
    """per-source-frame correction (source px) — identity outside the analysed stretches."""

    def __init__(self, video, segments=None, cfg=None, log=print):
        cfg = dict(DEF, **(cfg if isinstance(cfg, dict) else {}))
        self.cfg = cfg
        self.motion, self.sfps, self.n, self.sw, self.sh = analyse(video, segments, bool(cfg["mask_speaker"]), log)
        self.C = {}
        sig, stg = float(cfg["smooth"]) * self.sfps, float(cfg["strength"])
        for a, b in ranges_for(segments or [[0, self.n / self.sfps]], self.sfps, self.n):
            idx = list(range(a, b))
            steps = np.array([self.motion.get(i, [0, 0, 0, 0]) if i > a else [0, 0, 0, 0] for i in idx], np.float64)
            traj = np.cumsum(steps, axis=0)                      # camera path (dx, dy, angle, log-scale)
            corr = (_gauss(traj, sig) - traj) * stg
            for j, i in enumerate(idx):
                self.C[i] = corr[j]
        self.crop = float(cfg["crop"])

    def matrix(self, i):
        """3×3 (source px): undo the shake around the source centre, then the fixed crop zoom."""
        c = self.C.get(int(i))
        cx, cy = self.sw / 2, self.sh / 2
        dx, dy, da, dl = (0.0, 0.0, 0.0, 0.0) if c is None else c
        s = math.exp(dl) * self.crop
        ca, sa = math.cos(da) * s, math.sin(da) * s
        return np.array([[ca, -sa, cx - ca * cx + sa * cy + dx],
                         [sa, ca, cy - sa * cx - ca * cy + dy],
                         [0, 0, 1]], np.float64)

    def shake(self):
        """(before, after) mean |Δ step| in px over the analysed frames — a rough shake measure."""
        ks = sorted(k for k in self.motion if k in self.C and k - 1 in self.C)
        if len(ks) < 3:
            return 0.0, 0.0
        raw = np.array([self.motion[k] for k in ks])
        after = raw + np.array([np.asarray(self.C[k]) - np.asarray(self.C[k - 1]) for k in ks])
        return jitter(raw), jitter(after)


def jitter(steps):
    s = np.asarray(steps, np.float64)
    return float(np.abs(np.diff(s[:, :2], axis=0)).mean()) if len(s) > 2 else 0.0


def install(R):
    """called by the v14 Renderer when source.stabilize is set."""
    cfg = R.spec["source"].get("stabilize")
    if not cfg:
        return None
    segs = R.spec["source"].get("segments") or [[0.0, R.sn / R.sfps]]
    R.stab = Stabilizer(R.src, segs, cfg if isinstance(cfg, dict) else {})
    return R.stab


def report(R, log=print):
    st = getattr(R, "stab", None)
    if st is None:
        log("source.stabilize is not set in this spec")
        return 1
    b, a = st.shake()
    log(f"[stabilize] {len(st.motion)} frames analysed · shake {b:.2f} px → {a:.2f} px "
        f"({(1 - a / max(1e-6, b)) * 100:.0f} % less) · crop {st.crop:.2f}")
    return 0


def selftest():
    """synthetic hand-held clip (slow pan + random shake) → the correction removes the shake, keeps the pan."""
    rng = np.random.default_rng(5)
    H, W, n = 240, 360, 60
    base = cv2.GaussianBlur((rng.random((H * 2, W * 2)) * 255).astype(np.uint8), (0, 0), 1.5)
    d = os.path.join(os.path.expanduser("~/.cache"), "mograph_selftest")
    os.makedirs(d, exist_ok=True)
    vid = os.path.join(d, "shaky.mp4")
    pos = np.stack([np.linspace(0, 40, n), np.zeros(n)], 1) + rng.normal(0, 3.0, (n, 2))
    if not os.path.exists(vid):
        vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for i in range(n):
            x, y = int(W / 2 + pos[i, 0]), int(H / 2 + pos[i, 1])
            vw.write(cv2.cvtColor(base[y:y + H, x:x + W], cv2.COLOR_GRAY2BGR))
        vw.release()
    cp = os.path.join(vid + ".v14", "stab.json")
    if os.path.exists(cp):
        os.remove(cp)
    st = Stabilizer(vid, [[0, n / 30]], {"strength": 1.0, "smooth": 0.3, "crop": 1.0, "mask_speaker": False},
                    log=lambda *a, **k: None)
    b, a = st.shake()
    raw = np.array([st.motion.get(i, [0, 0, 0, 0]) for i in range(1, n)])
    after = raw + np.diff(np.array([st.C[i] for i in range(n)]), axis=0)
    pan_kept = abs(float(after[:, 0].sum()) - float(raw[:, 0].sum())) < 8.0
    rng_ok = (ranges_for([[1.0, 2.0], [2.5, 4.0]], 30, 10_000) == [(0, 166)] and
              ranges_for([[1.0, 2.0], [10, 11]], 30, 10_000) == [(0, 106), (255, 376)])
    return a < b * 0.5 and pan_kept and rng_ok, \
        f"synthetic hand-held clip: shake {b:.2f} → {a:.2f} px, intended pan kept {pan_kept}, ranges merge {rng_ok}"
