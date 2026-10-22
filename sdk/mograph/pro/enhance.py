"""
mograph.pro.enhance — v22 TILED REAL-ESRGAN restore / upscale with a natural-skin MIX (idea: Concat enhance.rs).

Model: realesr-general-x4v3.onnx (4.9 MB, BSD-3, fetched once to ~/.cache/mograph_models — never in git).
The frame is cut into CORE×CORE tiles read with a PAD margin (window = CORE + 2·PAD = 288 px), each window runs
through the model alone and only the tile's own (×4) pixels are pasted back → seamless, and the RAM peak is one
288² window (~60 MB) whatever the frame size: safe in the 1 GB sandbox.  `mix` blends the restored picture with a
bicubic enlargement of the original, so skin keeps its pores instead of turning waxy (default 0.6).

    python3 -m mograph.pro enhance in.jpg|in.mp4 [-o out] [--factor 2] [--mix 0.6] [--start s --end e]

Spec (v14):
    broll layer:  "enhance": {"mix": 0.6, "factor": 2}      # low-res stock / logo restored before compositing
    "source": {"enhance": {"mix": 0.5, "punch_in": 1.3}}    # restore the plate ONLY while the camera zoom ≥ punch_in
                                                            # (sharp face on punch-ins), cached per frame on disk
Fallback without the model: unsharp-mask detail boost (info["fallback"] = True), never a crash.
Speed: ≈ 1.3 s per 256² tile on 2 CPU cores → meant for stills, logos, short punch-in windows — not whole videos.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

SCALE, CORE, PAD = 4, 256, 16
WINDOW = CORE + 2 * PAD


def tiles(w, h, core=CORE):
    return [(x, y, min(core, w - x), min(core, h - y)) for y in range(0, h, core) for x in range(0, w, core)]


def _window(img, x, y):
    """the WINDOW×WINDOW square around a tile origin, edges replicated (fixed size → one compiled shape)."""
    h, w = img.shape[:2]
    ys = np.clip(np.arange(y - PAD, y - PAD + WINDOW), 0, h - 1)
    xs = np.clip(np.arange(x - PAD, x - PAD + WINDOW), 0, w - 1)
    return img[ys][:, xs]


def _session():
    from ..models_hub import session
    return session("esrgan")


def upscale4(bgr, log=None):
    """×4 restore with the model (tiled).  → uint8 BGR (4h×4w) or None when the model is unavailable."""
    s = _session()
    if s is None:
        return None
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = np.zeros((h * SCALE, w * SCALE, 3), np.float32)
    T = tiles(w, h)
    t0 = time.time()
    o = PAD * SCALE
    for k, (x, y, tw, th) in enumerate(T):
        win = np.ascontiguousarray(_window(rgb, x, y).transpose(2, 0, 1)[None])
        y4 = s.run(None, {"input": win})[0][0].transpose(1, 2, 0)
        out[y * SCALE:(y + th) * SCALE, x * SCALE:(x + tw) * SCALE] = y4[o:o + th * SCALE, o:o + tw * SCALE]
        if log and (k + 1) % 8 == 0:
            log(f"[enhance] tile {k + 1}/{len(T)}  {(time.time() - t0) / (k + 1):.2f}s/tile")
    return cv2.cvtColor((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _fallback(bgr, tw, th):
    big = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(big, 1.6, cv2.GaussianBlur(big, (0, 0), 1.2), -0.6, 0)


def enhance(bgr, factor=2, mix=0.6, max_in=1100, log=None):
    """restore + upscale by `factor` (1, 2 or 4) and blend `mix` of it over a bicubic copy of the original.

    factor 1 = same size, restored detail (the model runs ×4, the result is area-downscaled back).
    Inputs whose long side > max_in are downscaled first (the model is meant for low-res material).
    → (uint8 BGR, info dict)."""
    factor = int(factor) if int(factor) in (1, 2, 4) else 2
    h, w = bgr.shape[:2]
    tw, th = w * factor, h * factor
    work = bgr
    if max(h, w) > max_in:
        sc = max_in / max(h, w)
        work = cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    t0 = time.time()
    up = upscale4(work, log)
    if up is None:
        return _fallback(bgr, tw, th), {"fallback": True, "secs": time.time() - t0}
    plain = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_CUBIC)
    interp = cv2.INTER_AREA if up.shape[1] >= tw else cv2.INTER_CUBIC
    rest = cv2.resize(up, (tw, th), interpolation=interp)
    m = float(np.clip(mix, 0, 1))
    return cv2.addWeighted(rest, m, plain, 1 - m, 0), {
        "fallback": False, "tiles": len(tiles(work.shape[1], work.shape[0])), "secs": time.time() - t0,
        "mix": m, "factor": factor}


class FrameEnhancer:
    """per-frame disk cache (JPEG 95) for spec use — source punch-ins / broll stills; nothing is kept in RAM."""

    def __init__(self, cache_dir, factor=1, mix=0.6):
        self.dir, self.factor, self.mix = cache_dir, factor, mix
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key, bgr):
        p = os.path.join(self.dir, f"{key}_{self.factor}_{int(self.mix * 100)}.jpg")
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                return im
        out, _ = enhance(bgr, self.factor, self.mix)
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return out


def selftest():
    """tiling geometry exact (pure numpy) + model path if already cached (the selftest never downloads)."""
    w, h = 600, 333
    T = tiles(w, h)
    cov = np.zeros((h, w), np.int32)
    for x, y, tw, th in T:
        cov[y:y + th, x:x + tw] += 1
    geom = bool((cov == 1).all())
    wsz = _window(np.zeros((h, w, 3), np.float32), 512, 256).shape[:2] == (WINDOW, WINDOW)
    from ..models_hub import have
    msg = f"{len(T)} tiles cover 600×333 exactly once {geom}, window {WINDOW}² fixed {wsz}"
    if not have("esrgan"):
        return geom and wsz, msg + " (model not cached → geometry only)"
    img = np.zeros((40, 52, 3), np.uint8)
    cv2.putText(img, "VE", (4, 32), 0, 1.1, (255, 255, 255), 2)
    out, info = enhance(img, factor=4, mix=1.0)
    g = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()  # noqa: E731
    sharp, soft = g(out), g(cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
    ok_m = out.shape[:2] == (160, 208) and not info["fallback"] and sharp > soft
    return geom and wsz and ok_m, msg + f", model ×4 sharper than bicubic ({sharp:.0f} > {soft:.0f}) {ok_m}"


def cli(src, out=None, factor=2, mix=0.6, start=None, end=None, log=print):
    ext = os.path.splitext(src)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        res, info = enhance(cv2.imread(src), factor, mix, log=log)
        out = out or os.path.splitext(src)[0] + f"_enh{factor}x.png"
        cv2.imwrite(out, res)
        log(f"{out}  {res.shape[1]}×{res.shape[0]}  {info}")
        return out
    import subprocess
    from .rtime import ff_rate
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int((start or 0) * fps)
    f1 = min(n, int(end * fps)) if end else n
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out = out or os.path.splitext(src)[0] + f"_enh{factor}x.mp4"
    p = None
    for i in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        res, info = enhance(fr, factor, mix)
        hh, ww = res.shape[0] // 2 * 2, res.shape[1] // 2 * 2
        if p is None:
            p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                                  f"{ww}x{hh}", "-r", ff_rate(fps), "-i", "-", "-c:v", "libx264", "-crf", "17",
                                  "-preset", "superfast", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        p.stdin.write(np.ascontiguousarray(res[:hh, :ww]).tobytes())
        if (i - f0) % 10 == 0:
            log(f"[enhance] frame {i - f0 + 1}/{f1 - f0}  {info['secs']:.1f}s")
    if p:
        p.stdin.close()
        p.wait()
    log(out)
    return out
