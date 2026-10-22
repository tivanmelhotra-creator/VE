"""
mograph.pro.cutout — v22 OBJECT / PRODUCT BACKGROUND REMOVAL (idea: Concat concat-vision "object" = IS-Net).

A user hands over a plain product photo / logo / object → a clean RGBA sticker for `puppet_sticker`, `broll`
(PNG alpha is honoured), `object_3d` sprites …  Models (models_hub, outside git):
    quality "fast" (default)  u2netp.onnx            4.6 MB  Apache-2.0   ~0.3 s / image on 2 cores
    quality "high" (opt-in)   isnet-general-use.onnx 178 MB  Apache-2.0   (only when explicitly asked: size)
Fallback offline: OpenCV GrabCut seeded with a centre rectangle.

    python3 -m mograph.pro cutout product.jpg [-o product_cut.png] [--quality fast|high] [--feather 1.2]

Spec: any layer that takes an image "src" accepts "cutout": true → the cut PNG is made once (cached next to the
source as <name>.cut.png) and used instead (broll / puppet_sticker).
Edge clean-up: alpha refined by a guided upsample + optional feather, colour decontaminated at the edge
(foreground colour pulled from the inner neighbourhood so no background fringe is left).
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def _model(quality):
    from ..models_hub import session
    if quality == "high":
        s = session("isnet")
        if s is not None:
            return s, 1024, ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    s = session("u2netp")
    return (s, 320, ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])) if s is not None else (None, 0, None)


def _grabcut(bgr):
    h, w = bgr.shape[:2]
    sc = 480 / max(h, w)
    sm = cv2.resize(bgr, (int(w * sc), int(h * sc))) if sc < 1 else bgr
    m = np.zeros(sm.shape[:2], np.uint8)
    hh, ww = m.shape
    rect = (int(ww * 0.06), int(hh * 0.06), int(ww * 0.88), int(hh * 0.88))
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(sm, m, rect, bg, fg, 4, cv2.GC_INIT_WITH_RECT)
    a = np.where((m == 1) | (m == 3), 1.0, 0.0).astype(np.float32)
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)


def alpha(bgr, quality="fast"):
    """→ (alpha float32 h×w 0..1, engine name)."""
    s, size, norm = _model(quality)
    h, w = bgr.shape[:2]
    if s is None:
        return _grabcut(bgr), "grabcut"
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    x /= 255.0
    mean, std = np.array(norm[0], np.float32), np.array(norm[1], np.float32)
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    out = s.run(None, {s.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0, 0]
    lo, hi = float(out.min()), float(out.max())
    out = (out - lo) / max(1e-6, hi - lo)
    a = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(a, 0, 1).astype(np.float32), ("isnet" if size == 1024 else "u2netp")


def refine(bgr, a, feather=1.0):
    """contrast the soft edge, feather, guided by the image (keeps hair / fine edges, kills haze)."""
    a = np.clip((a - 0.08) / 0.84, 0, 1)
    try:
        g = cv2.ximgproc.guidedFilter(bgr, (a * 255).astype(np.uint8), 4, 1e-2 * 255 * 255) / 255.0  # noqa
        a = np.clip(g, 0, 1).astype(np.float32)
    except Exception:  # noqa: BLE001  (opencv-contrib not installed → plain feather)
        pass
    if feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather)
    return a.astype(np.float32)


def decontaminate(bgr, a, r=6):
    """replace edge colours by the colour of nearby solid foreground (no background fringe)."""
    solid = (a > 0.92).astype(np.float32)
    if solid.sum() < 16:
        return bgr
    k = (0, 0)
    num = cv2.GaussianBlur(bgr.astype(np.float32) * solid[..., None], k, r)
    den = cv2.GaussianBlur(solid, k, r)[..., None] + 1e-4
    fg = num / den
    edge = ((a > 0.02) & (a < 0.92)).astype(np.float32)[..., None]
    w = edge * (1 - a[..., None]) * 0.9
    return np.clip(bgr * (1 - w) + fg * w, 0, 255).astype(np.uint8)


def cutout(bgr, quality="fast", feather=1.0):
    """→ (BGRA uint8, engine)."""
    a, eng = alpha(bgr, quality)
    a = refine(bgr, a, feather)
    rgb = decontaminate(bgr, a)
    return np.dstack([rgb, (a * 255 + 0.5).astype(np.uint8)]), eng


def cutout_file(src, out=None, quality="fast", feather=1.0, log=print):
    out = out or os.path.splitext(src)[0] + ".cut.png"
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"cutout: cannot read {src}")
    rgba, eng = cutout(im, quality, feather)
    cv2.imwrite(out, rgba)
    log(f"[cutout] {eng}: {src} → {out}  (subject {float((rgba[..., 3] > 127).mean()) * 100:.0f} % of the frame)")
    return out


def selftest():
    """synthetic product shot: textured orange object on a light studio gradient."""
    h, w = 320, 320
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    bg = (np.ones((h, w, 3), np.float32) * (225 - 25 * y)).astype(np.uint8)
    im = bg.copy()
    cv2.ellipse(im, (160, 175), (78, 105), 0, 0, 360, (30, 120, 235), -1, cv2.LINE_AA)
    cv2.rectangle(im, (140, 50), (180, 80), (40, 40, 40), -1)
    gt = np.zeros((h, w), np.uint8)
    cv2.ellipse(gt, (160, 175), (78, 105), 0, 0, 360, 1, -1)
    cv2.rectangle(gt, (140, 50), (180, 80), 1, -1)
    rgba, eng = cutout(im)
    pred = rgba[..., 3] > 127
    iou = float((pred & (gt > 0)).sum()) / max(1, float((pred | (gt > 0)).sum()))
    return iou > 0.80, f"{eng}: product IoU {iou:.2f} on a synthetic studio shot"
