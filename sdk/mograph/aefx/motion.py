"""
mograph.aefx.motion — our own implementation of the 5 effects the Premiere "Motion Presets" are built from + the
preset player.  A preset = ordered STACK of adjustment-layer effects (each applied to the result below it).

    from mograph.aefx import motion as AM
    img = AM.apply_preset(img, "scale_01_in", t)              # t = s since the preset started (BGR or BGRA premult.)
    AM.preset_names(cat="Glitch", dir="in")

EFFECTS (frame-exact pure functions of t; px units at the 1920-wide reference → resolution independent):
  transform  anchor/position [u,v] (normalised), scale %, scale_w % (uniform 0), skew + skew_axis deg, rotation deg,
             opacity %, shutter deg (true sub-frame motion blur along the keyed path; presets 0 → pass 180)
  basic3d    swivel (Y) / tilt (X) deg with perspective, distance (+ further)
  fast_blur  blur, dims 0 both / 1 H / 2 V
  wave_warp  type 1 sine 2 square 3 triangle 4 saw 5 circle 6 semicircle 7 uncircle 8 noise 9 smooth-noise,
             height px, width px, direction deg, speed waves/s, phase deg
  twirl      angle deg, radius % (of half diagonal), center [u,v]
"""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from .keyframes import evaluate, scale_time, last_time

_LIB = None
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
REF_W = 1920.0


def library():
    global _LIB
    if _LIB is None:
        with open(os.path.join(_DIR, "motion_presets.json"), encoding="utf-8") as f:
            _LIB = json.load(f)
    return _LIB


def preset_names(cat=None, dir=None):  # noqa: A002
    return [k for k, v in library().items() if (cat is None or v["cat"] == cat) and (dir is None or v["dir"] == dir)]


def get(name):
    if isinstance(name, dict):
        return name
    lib = library()
    if name in lib:
        return lib[name]
    key = "_".join(t for t in "".join(c if c.isalnum() else " " for c in name.lower()).split())
    c = [k for k in lib if k == key] or [k for k in lib if k.startswith(key)]
    if not c:
        raise KeyError(f"unknown motion preset '{name}'  (python3 -m mograph.aefx list motion)")
    return lib[c[0]]


def retimed(p, dur=None, speed=1.0):
    k = (float(dur) / p["dur"]) if dur else 1.0 / float(speed)
    if abs(k - 1) < 1e-6:
        return p
    return dict(p, dur=p["dur"] * k,
                fx=[{"fx": f["fx"], "p": {n: scale_time(v, k) for n, v in f["p"].items()}} for f in p["fx"]])


# ----------------------------------------------------------------------------------------- matrices
def transform_matrix(P, t, W, H):
    ax, ay = evaluate(P.get("anchor", [0.5, 0.5]), t)
    px, py = evaluate(P.get("position", [0.5, 0.5]), t)
    sc = float(evaluate(P.get("scale", 100.0), t)) / 100.0
    uni = float(evaluate(P.get("uniform", 1.0), t)) > 0.5
    sw = sc if uni else float(evaluate(P.get("scale_w", 100.0), t)) / 100.0
    rot = math.radians(float(evaluate(P.get("rotation", 0.0), t)))
    skew = math.radians(float(evaluate(P.get("skew", 0.0), t)))
    sax = math.radians(float(evaluate(P.get("skew_axis", 0.0), t)))
    T1 = np.array([[1, 0, -ax * W], [0, 1, -ay * H], [0, 0, 1]], np.float64)
    S = np.diag([sw, sc, 1.0])
    if abs(skew) > 1e-6:
        Ra = np.array([[math.cos(sax), -math.sin(sax), 0], [math.sin(sax), math.cos(sax), 0], [0, 0, 1]])
        S = S @ Ra @ np.array([[1, math.tan(skew), 0], [0, 1, 0], [0, 0, 1]]) @ Ra.T
    R = np.array([[math.cos(rot), -math.sin(rot), 0], [math.sin(rot), math.cos(rot), 0], [0, 0, 1]])
    T2 = np.array([[1, 0, px * W], [0, 1, py * H], [0, 0, 1]], np.float64)
    return T2 @ R @ S @ T1


def basic3d_matrix(P, t, W, H):
    sw = math.radians(float(evaluate(P.get("swivel", 0.0), t)))
    tl = math.radians(float(evaluate(P.get("tilt", 0.0), t)))
    dist = float(evaluate(P.get("distance", 0.0), t)) * W / REF_W * 10.0
    f = W * 1.39                                              # ≈ AE default 50 mm camera (zoom 2666 px @1920)
    Ry = np.array([[math.cos(sw), 0, math.sin(sw)], [0, 1, 0], [-math.sin(sw), 0, math.cos(sw)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(tl), -math.sin(tl)], [0, math.sin(tl), math.cos(tl)]])
    Rm = Rx @ Ry
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    dst = []
    for x, y in src:
        p = Rm @ np.array([x - W / 2, y - H / 2, 0.0])
        z = max(f + dist + p[2], f * 0.05)
        dst.append([W / 2 + p[0] * f / z, H / 2 + p[1] * f / z])
    return cv2.getPerspectiveTransform(src, np.array(dst, np.float32)).astype(np.float64)


def _warp(img, M, W, H):
    if abs(M[2, 0]) + abs(M[2, 1]) > 1e-12:
        return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(img, M[:2], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _opacity(img, a):
    if a >= 0.999:
        return img
    return (img.astype(np.float32) * max(0.0, a)).astype(img.dtype)


# ----------------------------------------------------------------------------------------- effects
def fx_transform(img, P, t, fps=30.0):
    H, W = img.shape[:2]
    shutter = float(evaluate(P.get("shutter", 0.0), t))
    op = float(evaluate(P.get("opacity", 100.0), t)) / 100.0
    if op <= 0.001:
        return np.zeros_like(img)
    if shutter > 0.5:
        span = shutter / 360.0 / fps
        c = np.array([W / 2, H / 2, 1.0])
        corners = [np.array([0, 0, 1.0]), np.array([W, H, 1.0]), c]
        M0, M1 = transform_matrix(P, t - span / 2, W, H), transform_matrix(P, t + span / 2, W, H)
        mv = max(float(np.hypot(*((M1 @ q) - (M0 @ q))[:2])) for q in corners)
        n = int(min(16, mv / 2.0))
        if n >= 2:
            acc = np.zeros(img.shape, np.float32)
            for i in range(n):
                acc += _warp(img, transform_matrix(P, t - span / 2 + span * (i + 0.5) / n, W, H), W, H)
            return _opacity((acc / n).astype(img.dtype), op)
    return _opacity(_warp(img, transform_matrix(P, t, W, H), W, H), op)


def fx_basic3d(img, P, t, fps=30.0):
    H, W = img.shape[:2]
    return _warp(img, basic3d_matrix(P, t, W, H), W, H)


def fx_fast_blur(img, P, t, fps=30.0):
    H, W = img.shape[:2]
    b = float(evaluate(P.get("blur", 0.0), t)) * W / REF_W
    if b < 0.3:
        return img
    dims = int(evaluate(P.get("dims", 0.0), t))
    sig = b * 0.5
    q = 1 if sig < 6 else (2 if sig < 18 else 4)
    small = cv2.resize(img, (max(1, W // q), max(1, H // q)), interpolation=cv2.INTER_AREA) if q > 1 else img
    sx = sig / q if dims in (0, 1) else 0.01
    sy = sig / q if dims in (0, 2) else 0.01
    out = cv2.GaussianBlur(small, (0, 0), sigmaX=max(0.01, sx), sigmaY=max(0.01, sy))
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR) if q > 1 else out


def wave_shape(kind, x):
    u = (x / (2 * math.pi)) % 1.0
    if kind == 2:
        return np.where(u < 0.5, 1.0, -1.0)
    if kind == 3:
        return 1 - 4 * np.abs(u - 0.5)
    if kind == 4:
        return 2 * u - 1
    if kind == 5:
        return np.sqrt(np.clip(1 - (4 * (u % 0.5) - 1) ** 2, 0, 1)) * np.where(u < 0.5, 1, -1)
    if kind == 6:
        return np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1))
    if kind == 7:
        return 1 - np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1))
    if kind in (8, 9):
        c = x / (2 * math.pi) * 4
        i = np.floor(c)
        h = np.sin(i * 12.9898) * 43758.5453
        h = (h - np.floor(h)) * 2 - 1
        if kind == 8:
            return h
        h2 = np.sin((i + 1) * 12.9898) * 43758.5453
        h2 = (h2 - np.floor(h2)) * 2 - 1
        f = c - i
        f = f * f * (3 - 2 * f)
        return h * (1 - f) + h2 * f
    return np.sin(x)


_GRID = {}


def _grid(W, H):
    if (W, H) not in _GRID:
        _GRID.clear()
        _GRID[(W, H)] = np.mgrid[0:H, 0:W].astype(np.float32)
    return _GRID[(W, H)]


def fx_wave_warp(img, P, t, fps=30.0):
    H, W = img.shape[:2]
    k = W / REF_W
    amp = float(evaluate(P.get("height", 10.0), t)) * k
    if abs(amp) < 0.2:
        return img
    wl = max(1.0, float(evaluate(P.get("width", 40.0), t)) * k)
    d = math.radians(float(evaluate(P.get("direction", 90.0), t)))
    sp = float(evaluate(P.get("speed", 1.0), t))
    ph = math.radians(float(evaluate(P.get("phase", 0.0), t)))
    kind = int(evaluate(P.get("type", 1.0), t))
    dx, dy = math.sin(d), -math.cos(d)                        # displacement direction (AE 90° = horizontal)
    yy, xx = _grid(W, H)
    along = -xx * dy + yy * dx                                # wave runs perpendicular to the displacement
    w = wave_shape(kind, along / wl * 2 * math.pi - sp * t * 2 * math.pi + ph).astype(np.float32) * amp
    return cv2.remap(img, xx - w * dx, yy - w * dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def fx_twirl(img, P, t, fps=30.0):
    H, W = img.shape[:2]
    ang = math.radians(float(evaluate(P.get("angle", 0.0), t)))
    if abs(ang) < 1e-3:
        return img
    rad = max(1.0, float(evaluate(P.get("radius", 30.0), t)) / 100.0 * math.hypot(W, H) / 2)
    cx, cy = evaluate(P.get("center", [0.5, 0.5]), t)
    yy, xx = _grid(W, H)
    ddx, ddy = xx - cx * W, yy - cy * H
    f = np.clip(1 - np.sqrt(ddx * ddx + ddy * ddy) / rad, 0, 1)
    a = -ang * f * f
    ca, sa = np.cos(a), np.sin(a)
    return cv2.remap(img, (cx * W + ddx * ca - ddy * sa).astype(np.float32),
                     (cy * H + ddx * sa + ddy * ca).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


FX = {"transform": fx_transform, "basic3d": fx_basic3d, "fast_blur": fx_fast_blur, "wave_warp": fx_wave_warp,
      "twirl": fx_twirl}


def apply_stack(img, fx, t, fps=30.0, motion_blur=None):
    out = img
    for f in fx:
        P = f["p"] if motion_blur is None or f["fx"] != "transform" else dict(f["p"], shutter=float(motion_blur))
        out = FX[f["fx"]](out, P, t, fps)
    return out


def apply_preset(img, name, t, fps=30.0, dur=None, speed=1.0, motion_blur=None):
    """Play a preset on img at local time t.  Holds the first/last state outside [0, dur]."""
    p = retimed(get(name), dur, speed)
    return apply_stack(img, p["fx"], max(0.0, t), fps, motion_blur)


def settled(p, t):
    """True once every curve has finished (IN presets are then identity → callers can skip the work)."""
    return all(t >= last_time(v) for f in p["fx"] for v in f["p"].values())
