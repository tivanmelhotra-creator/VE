"""
mograph.v14.motionblur — velocity-driven 180° shutter DIRECTIONAL motion blur (v14.8).

WHY: at 30 fps a spring entrance (drop_bounce, swing_3d, flip_3d, object_3d spin-in) moves 40-120 px between two
frames.  Without blur the eye sees discrete copies ("stroboscopic stepping").  A real camera with a 180° shutter
integrates the motion over half the frame interval → a 1-D streak along the velocity vector.

HOW (exact, cheap, local):
  1  the caller knows the element at time t and at t − Δ (Δ = shutter/360 / fps) → screen displacement vector d
     (glyphs: projected contour points; object_3d: rim speed of the yaw spin + spring translation)
  2  if |d| < threshold (default 2.5 px) nothing happens — static / settled frames cost zero
  3  otherwise the element is painted into a small OFFSCREEN premultiplied-ARGB surface (its device bbox + |d|),
     convolved with a normalised LINE kernel of length |d| along d (cv2.filter2D — premultiplied alpha makes the
     streak's edges correct), and painted back with the context's clip intact.
Spec:  per layer "motion_blur": true | false | {"shutter": 180, "threshold": 2.5, "max": 48}
       global  "post": {"motion_blur": {"shutter": 180}}  or false (disables everywhere).
"""
from __future__ import annotations

import math

import cairo
import cv2
import numpy as np

DEFAULTS = {"shutter": 180.0, "threshold": 2.5, "max": 48.0}
AUTO_ENTERS = ("drop_bounce", "swing_3d", "flip_3d")
AUTO_EXITS = ("fall", "flip_out")


def config(L, F):
    """Resolved motion-blur config for a layer, or None when disabled."""
    g = getattr(F.R, "post", {}).get("motion_blur", {}) if getattr(F, "R", None) is not None else {}
    if g is False:
        return None
    v = L.get("motion_blur", "auto")
    if v is False:
        return None
    cfg = dict(DEFAULTS, **(g if isinstance(g, dict) else {}))
    if isinstance(v, dict):
        cfg.update(v)
    cfg["auto"] = (v == "auto")
    return cfg


def shutter_dt(cfg, fps):
    return float(cfg["shutter"]) / 360.0 / max(1.0, float(fps))


def line_kernel(dx, dy):
    """Normalised 1-D line kernel of length |d| along (dx, dy) (anti-aliased endpoints)."""
    L = math.hypot(dx, dy)
    n = int(math.ceil(L)) | 1
    n = max(3, n)
    k = np.zeros((n, n), np.float32)
    c = (n - 1) / 2.0
    steps = max(2, int(L * 2) + 1)
    for i in range(steps):
        f = i / (steps - 1) - 0.5
        x, y = c + dx * f, c + dy * f
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for (xx, yy, ww) in ((x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)),
                             (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy)):
            if 0 <= xx < n and 0 <= yy < n:
                k[yy, xx] += ww
    s = k.sum()
    return k / s if s > 0 else k


def principal_displacement(P0, P1):
    """Dominant screen displacement (dx, dy) between two (N,2) point sets (translation, or the main axis of a
    rotation/flip when the mean translation is small)."""
    D = (P1 - P0).astype(np.float64)
    if len(D) == 0:
        return 0.0, 0.0
    mean = D.mean(axis=0)
    rms = math.sqrt(float((D ** 2).sum(axis=1).mean()))
    if rms < 1e-6:
        return 0.0, 0.0
    if math.hypot(*mean) >= 0.6 * rms:
        return float(mean[0]), float(mean[1])
    C = D.T @ D
    w, V = np.linalg.eigh(C)
    ax = V[:, int(np.argmax(w))]
    return float(ax[0] * rms), float(ax[1] * rms)


def paint_blurred(ctx, draw_fn, bbox, dx, dy, max_len=48.0):
    """Draw `draw_fn(c)` (user-space coords of ctx) into an offscreen surface, directional-blur by (dx, dy)
    (device px), and paint it onto ctx.  bbox = user-space (x0, y0, x1, y1) that contains the drawing."""
    L = math.hypot(dx, dy)
    if L > max_len:
        dx, dy = dx * max_len / L, dy * max_len / L
        L = max_len
    pts = [ctx.user_to_device(bbox[0], bbox[1]), ctx.user_to_device(bbox[2], bbox[1]),
           ctx.user_to_device(bbox[0], bbox[3]), ctx.user_to_device(bbox[2], bbox[3])]
    pad = int(L / 2) + 4
    X0 = int(math.floor(min(p[0] for p in pts))) - pad
    Y0 = int(math.floor(min(p[1] for p in pts))) - pad
    X1 = int(math.ceil(max(p[0] for p in pts))) + pad
    Y1 = int(math.ceil(max(p[1] for p in pts))) + pad
    tgt = ctx.get_target()
    try:
        TW, TH = tgt.get_width(), tgt.get_height()
        X0, Y0, X1, Y1 = max(0, X0), max(0, Y0), min(TW, X1), min(TH, Y1)
    except Exception:  # noqa: BLE001 — non-image targets (recording / probe)
        pass
    w, h = X1 - X0, Y1 - Y0
    if w < 2 or h < 2 or w * h > 1400 * 1400:
        draw_fn(ctx)
        return False
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    c = cairo.Context(s)
    m = ctx.get_matrix()
    c.set_matrix(cairo.Matrix(m.xx, m.yx, m.xy, m.yy, m.x0 - X0, m.y0 - Y0))
    draw_fn(c)
    s.flush()
    st = s.get_stride()
    a = np.ndarray((h, st // 4, 4), np.uint8, s.get_data())[:, :w]
    k = line_kernel(dx, dy)
    a[:] = cv2.filter2D(a, -1, k, borderType=cv2.BORDER_CONSTANT)
    s.mark_dirty()
    ctx.save()
    ctx.identity_matrix()
    ctx.set_source_surface(s, X0, Y0)
    ctx.paint()
    ctx.restore()
    del c
    return True
