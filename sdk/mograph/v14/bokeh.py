"""
mograph.v14.bokeh — depth-sandwich 3D BOKEH DUST (v14.8, CPU port of EffectCraft three_d/bokeh.rs).

Out-of-focus floating dust in a real lens: every point light far from the focal plane becomes a disc shaped like the
iris (octagonal aperture blades), slightly brighter at the rim (spherical aberration "soap bubble"), and moves with
a parallax that depends on its distance to the camera.  We render it as TWO layers of the depth sandwich:

  BACK   12 small/medium octagonal discs BEHIND the speaker (drawn on the background before the speaker composite,
         so she occludes them).  Tinted by @accent mixed with warm white.  Parallax 0.7× of the camera.
  FRONT   6 LARGE very soft discs IN FRONT of everything (drawn after the front pass, before bloom), parallax
         1.3× of the camera → they slide faster than the speaker on every push-in / pan = real depth.
         Front discs fade out over the protected face ellipse (never fog the eyes / mouth).

Motion: slow Brownian-like drift (sum of incommensurate sines per particle, pure function of t → frames render in
any order), gentle breathing of size/alpha ("twinkle").  Deterministic layout from "seed".

Enable with either
    "post":   {"bokeh_dust": true | {"back": 12, "front": 6, "alpha": 1.0, "color": "@accent", "seed": 7}}
    a layer:  {"type": "ambient_bokeh", "t": 0, "out": "we:…", "back": 12, "front": 6, "alpha": 1.0}
              (the layer form adds a 0.6 s fade-in / fade-out window; several layers may overlap → strongest wins)
Cost: ≈4-7 ms / frame (cached sprites, additive blend on small ROIs).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

DEFAULTS = {"back": 12, "front": 6, "alpha": 1.0, "color": "@accent", "seed": 7, "par_back": 0.7,
            "par_front": 1.3, "back_size": [0.035, 0.085], "front_size": [0.14, 0.26]}
_SPRITES = {}


def _sprite(d, soft):
    """Octagonal bokeh disc, float32 (d×d) in 0..1 with a bright rim (cached by quantized (d_q, soft))."""
    d_q = (max(8, int(d)) // 8) * 8 + 1
    key = (d_q, round(float(soft), 2))
    if key in _SPRITES:
        return _SPRITES[key]
    d = d_q
    r = d / 2.0
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32) - (d - 1) / 2.0
    ang = np.arctan2(yy, xx) + math.pi / 8
    sector = (ang % (math.pi / 4)) - math.pi / 8                    # regular octagon: radius(θ) = a / cos(θ')
    rad_oct = (r * 0.86) * math.cos(math.pi / 8) / np.cos(sector)
    dist = np.sqrt(xx * xx + yy * yy)
    edge = np.clip((rad_oct - dist) / max(1.0, r * soft) + 0.5, 0.0, 1.0)
    rim = np.clip(1.0 - np.abs(dist - rad_oct * 0.88) / max(1.0, r * 0.18), 0.0, 1.0) * 0.45
    s = edge * (0.62 + rim)
    s = cv2.GaussianBlur(s, (0, 0), max(0.6, r * soft * 0.35))
    s = (s / max(1e-6, s.max())).astype(np.float32)
    _SPRITES[key] = s
    return s


def _particles(cfg, kind, W, H):
    n = int(cfg[kind])
    rng = np.random.default_rng(int(cfg["seed"]) * 131 + (1 if kind == "front" else 0))
    lo, hi = cfg[f"{kind}_size"]
    out = []
    for i in range(n):
        if kind == "front":                                         # front discs live near the frame edges
            side = i % 2
            u = rng.uniform(0.02, 0.26) if side == 0 else rng.uniform(0.74, 0.98)
            v = rng.uniform(0.08, 0.95)
        else:
            u, v = rng.uniform(0.04, 0.96), rng.uniform(0.06, 0.80)
        out.append({"u": u, "v": v, "s": rng.uniform(lo, hi), "ph": rng.uniform(0, 6.283), "sp": rng.uniform(0.6, 1.4),
                    "a": rng.uniform(0.45, 1.0), "hue": rng.uniform(-1, 1)})
    return out


def resolve(R, t):
    """Active config at time t (merging post.bokeh_dust and ambient_bokeh layers) or None."""
    best, gain = None, 0.0
    pb = R.post.get("bokeh_dust")
    if pb:
        best, gain = dict(DEFAULTS, **(pb if isinstance(pb, dict) else {})), 1.0
    for L in R.layers:
        if L["type"] != "ambient_bokeh":
            continue
        t0, t1 = L["_t0"], L["_t1"]
        if not (t0 <= t <= t1 + 0.6):
            continue
        g = min(1.0, (t - t0) / 0.6, max(0.0, (t1 + 0.6 - t) / 0.6))
        if g > gain:
            best = dict(DEFAULTS, **{k: v for k, v in L.items() if not k.startswith("_") and k in DEFAULTS})
            gain = g
    if best is None or gain <= 0.003:
        return None
    best["_gain"] = gain
    return best


def _screen(F, R, p, par, t):
    """Particle → screen px with camera parallax `par` (1 = locked to the background plate)."""
    dx = math.sin(t * 0.21 * p["sp"] + p["ph"]) * 0.018 + math.sin(t * 0.47 * p["sp"] + p["ph"] * 1.7) * 0.007
    dy = math.cos(t * 0.17 * p["sp"] + p["ph"] * 0.6) * 0.014 - math.sin(t * 0.09 * p["sp"] + p["ph"]) * 0.01
    sx, sy = (p["u"] + dx) * R.sw, (p["v"] + dy) * R.sh
    M = F.M
    if getattr(R, "_bokeh_M0", None) is None:
        R._bokeh_M0 = R.matrix(0.0)[0]
    M0 = R._bokeh_M0
    q = (M[0, 0] * sx + M[0, 1] * sy + M[0, 2], M[1, 0] * sx + M[1, 1] * sy + M[1, 2])
    q0 = (M0[0, 0] * sx + M0[0, 1] * sy + M0[0, 2], M0[1, 0] * sx + M0[1, 1] * sy + M0[1, 2])
    return q0[0] + par * (q[0] - q0[0]), q0[1] + par * (q[1] - q0[1]), M[0, 0] / max(1e-6, M0[0, 0])


def _add_sprite(arr, spr, cx, cy, col, a):
    """Additive (screen-like) blend of a sprite onto a BGR uint8 array (in place)."""
    H, W = arr.shape[:2]
    d = spr.shape[0]
    x0, y0 = int(round(cx - d / 2)), int(round(cy - d / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + d), min(H, y0 + d)
    if X1 <= X0 or Y1 <= Y0:
        return
    s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None] * a
    roi = arr[Y0:Y1, X0:X1].astype(np.float32)
    c = np.array(col, np.float32)[None, None, :] * 255.0
    roi = roi + (c - roi * (c / 255.0) * 0.55) * s                 # soft screen: bright areas saturate gently
    arr[Y0:Y1, X0:X1] = np.clip(roi, 0, 255).astype(np.uint8)


def _colors(F, cfg):
    from . import glyphs as G
    base = G.rgba(cfg.get("color", "@accent"), F.R.palette)
    bgr = (base[2], base[1], base[0])
    warm = (0.80, 0.90, 1.0)                                        # warm white (BGR)
    return bgr, warm


def draw(F, arr, kind, face_zone=None):
    """Draw the `kind` ("back" | "front") dust onto BGR uint8 `arr` (in place)."""
    R = F.R
    cfg = resolve(R, F.t)
    if cfg is None or int(cfg[kind]) <= 0:
        return
    key = (kind, int(cfg[kind]), int(cfg["seed"]), tuple(cfg[f"{kind}_size"]))
    cache = R.__dict__.setdefault("_bokeh_p", {})
    if key not in cache:
        cache[key] = _particles(cfg, kind, R.W, R.H)
    par = float(cfg["par_back" if kind == "back" else "par_front"])
    acc, warm = _colors(F, cfg)
    gain = float(cfg["_gain"]) * float(cfg.get("alpha", 1.0))
    for p in cache[key]:
        x, y, zk = _screen(F, R, p, par, F.t)
        tw = 0.75 + 0.25 * math.sin(F.t * 1.3 * p["sp"] + p["ph"] * 2.0)
        d = p["s"] * R.W * (zk ** (par - 0.2)) * (0.94 + 0.06 * tw)
        if kind == "back":
            mix = 0.5 + 0.5 * p["hue"]
            col = tuple(acc[c] * mix + warm[c] * (1 - mix) for c in range(3))
            a = 0.30 * p["a"] * tw * gain
            spr = _sprite(min(260, d), 0.10)
        else:
            col = tuple(acc[c] * 0.35 + warm[c] * 0.65 for c in range(3))
            a = 0.16 * p["a"] * tw * gain
            spr = _sprite(min(620, d), 0.22)
            if face_zone is not None:                               # never fog the face
                zx, zy, rx, ry = face_zone
                q = ((x - zx) / (rx + d * 0.5)) ** 2 + ((y - zy) / (ry + d * 0.5)) ** 2
                a *= float(np.clip((q - 0.8) / 0.8, 0.0, 1.0))
        if a > 0.004:
            _add_sprite(arr, spr, x, y, col, a)
