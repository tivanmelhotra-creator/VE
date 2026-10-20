"""
mograph.transfx — v12 edit-point transitions applied to the composited frame (graphics included, HUD excluded).

Events come from the EDL (spec._cutfx, see mograph.cut) or are authored:  "cut_fx": [{"t": 7.8, "type": "zoom_blur"}]
Each event is centred on its cut time t and spans dur (default 0.3 s): the first half accelerates the outgoing shot,
the second half decelerates the incoming one (frame-exact: a pure function of t).

types
  zoom_blur   radial zoom-through (scale 1 → 1.35 → 1 + radial blur)            the classic reel "zoom transition"
  whip        horizontal whip pan (offset + directional motion blur), "dir": 1|-1
  spin        roll ±18° + zoom + blur
  punch       hard cut with a 1.12 → 1 scale settle (jump-cut cover)
  flash       white exposure flash peaking at the cut
  glitch      RGB split + sliced offsets + scanline noise (deterministic)
  shake       short impact shake
keys: dur, strength (multiplies the effect), dir, color (flash)
"""
from __future__ import annotations

import math

import cairo
import cv2
import numpy as np

from .core import clamp, color as _color


def _bump(t, t0, d):
    """0 → 1 (at cut) → 0, smooth, over [t0-d/2, t0+d/2]."""
    h = d / 2
    if t < t0 - h or t > t0 + h:
        return 0.0
    x = 1 - abs(t - t0) / h
    return x * x * (3 - 2 * x)


def active(scene, t):
    out = []
    for ev in scene.spec.get("_cutfx") or []:
        d = float(ev.get("dur", 0.3))
        if ev.get("type") == "punch":
            pd = float(ev.get("dur", 0.22))
            if ev["t"] <= t <= ev["t"] + pd:
                out.append((ev, (t - ev["t"]) / pd))
        elif abs(t - float(ev["t"])) <= d / 2:
            out.append((ev, _bump(t, float(ev["t"]), d)))
    return out


def _arr(surf):
    surf.flush()
    W, H = surf.get_width(), surf.get_height()
    return np.ndarray((H, surf.get_stride() // 4, 4), np.uint8, buffer=surf.get_data())[:, :W]


def _zoom(img, z, cx, cy, ang=0.0):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), ang, z)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply(scene, surf, t, scale=1.0):
    evs = active(scene, t)
    if not evs:
        return
    a = _arr(surf)
    img = a.copy()
    H, W = img.shape[:2]
    cx, cy = W / 2, H / 2
    for ev, p in evs:
        typ = ev.get("type", "zoom_blur")
        k = float(ev.get("strength", 1.0))
        side = -1 if t < float(ev["t"]) else 1          # outgoing (-1) / incoming (+1)
        if typ == "zoom_blur" and p > 0.01:
            z = 1 + 0.35 * k * p
            n = 6
            acc = np.zeros_like(img, np.float32)
            for i in range(n):
                acc += _zoom(img, z * (1 + 0.06 * k * p * i / (n - 1)), cx, cy)
            img = (acc / n).astype(np.uint8)
        elif typ == "spin" and p > 0.01:
            ang = 18 * k * p * side
            n = 5
            acc = np.zeros_like(img, np.float32)
            for i in range(n):
                acc += _zoom(img, 1 + 0.25 * k * p, cx, cy, ang * (1 + 0.15 * i / n))
            img = (acc / n).astype(np.uint8)
        elif typ == "whip" and p > 0.01:
            d = int(ev.get("dir", 1))
            off = int(W * 0.35 * k * p * (-side) * d)
            img = np.roll(img, off, axis=1)
            L = max(3, int(W * 0.12 * k * p)) | 1
            ker = np.zeros((1, L), np.float32); ker[0, :] = 1.0 / L
            img = cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT)
        elif typ == "punch":
            amt = float(ev.get("amount", 0.14)) * k
            z = 1 + amt * (1 - (1 - (1 - clamp(p)) ** 3))
            img = _zoom(img, z, cx, cy)
        elif typ == "glitch" and p > 0.02:
            rng = np.random.default_rng(int(t * 30) + 7)
            sh = int(18 * k * p * scale) + 1
            b, g, r, al = cv2.split(img)
            img = cv2.merge([np.roll(b, -sh, axis=1), g, np.roll(r, sh, axis=1), al])
            for _ in range(int(6 * p) + 1):
                y0 = int(rng.integers(0, H - 10)); hh = int(rng.integers(8, max(9, int(H * 0.08))))
                img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], int(rng.integers(-60, 60) * k * p * scale), axis=1)
        elif typ == "shake" and p > 0.01:
            dx = int(20 * k * p * math.sin(t * 90) * scale); dy = int(14 * k * p * math.cos(t * 77) * scale)
            img = np.roll(np.roll(img, dx, axis=1), dy, axis=0)
    a[...] = img
    for ev, p in evs:                                     # additive flashes last
        if ev.get("type") in ("flash", "zoom_blur", "whip") and p > 0.01:
            amt = (0.9 if ev["type"] == "flash" else float(ev.get("flash", 0.25))) * p * float(ev.get("strength", 1.0))
            if amt > 0.01:
                ctx = cairo.Context(surf)
                surf.mark_dirty()
                ctx.set_source_rgba(*_color(ev.get("color", "#ffffff"), amt))
                ctx.set_operator(cairo.OPERATOR_ADD)
                ctx.paint()
                surf.flush()
    surf.mark_dirty()
