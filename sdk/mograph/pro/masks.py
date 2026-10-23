"""
mograph.pro.masks — v22 ANALYTIC SHAPE MASKS (signed distance fields), resolution-independent.

Idea taken from Concat (concat-core/src/mask.rs), re-implemented in numpy for VE: every shape is a signed distance
d(x, y) in units of the frame's SHORT side (so feather 0.03 = 3 % of the short side at 540p, 1080p or 4K alike);
alpha = smoothstep over [-feather/2, +feather/2].  Edges are perfectly sharp at any size (no raster upscaling).

    "mask": {"shape": "linear" | "mirror" | "circle" | "rectangle" | "heart" | "star",
             "cx": 0.5, "cy": 0.5,          # centre, fraction of W / H
             "w": 0.6,  "h": 0.4,           # size, fraction of the SHORT side (circle → ellipse radii = w/2, h/2)
             "angle": 0,                    # degrees (linear / mirror: direction of the edge line)
             "feather": 0.02,               # soft edge, fraction of the short side
             "roundness": 0.0,              # rectangle corner radius 0..1 (1 = pill); star: fatter inner points
             "width": 0.3,                  # mirror: band width (fraction of short side)
             "points": 5,                   # star
             "invert": false, "opacity": 1.0}

Any numeric value may be a pair [from, to] → eased (smoothstep) over the layer's life: animated wipes / iris.
Used by `broll` ("mask": {…}; mode "split_diag" = full + an angled linear mask) and available to any layer via
`mask_alpha(cfg, W, H, u)`.
"""
from __future__ import annotations

import math

import numpy as np

SHAPES = ("linear", "mirror", "circle", "rectangle", "heart", "star")
_DEF = {"cx": 0.5, "cy": 0.5, "w": 0.6, "h": 0.4, "angle": 0.0, "feather": 0.02, "roundness": 0.0, "width": 0.3,
        "points": 5, "invert": False, "opacity": 1.0}


def _val(v, u):
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        e = u * u * (3 - 2 * u)
        return float(v[0] + (v[1] - v[0]) * e)
    return v


def _grid(W, H, cx, cy, angle):
    """pixel centres → local coords (units of the short side), rotated by -angle around (cx, cy)."""
    s = float(min(W, H))
    xs = (np.arange(W, dtype=np.float32) + 0.5 - cx * W) / s
    ys = (np.arange(H, dtype=np.float32) + 0.5 - cy * H) / s
    X, Y = np.meshgrid(xs, ys)
    if angle:
        a = math.radians(angle)
        c, si = math.cos(a), math.sin(a)
        X, Y = X * c + Y * si, -X * si + Y * c
    return X, Y


def sdf(shape, W, H, P):
    """signed distance (negative inside) in short-side units."""
    X, Y = _grid(W, H, float(P["cx"]), float(P["cy"]), float(P["angle"]))
    if shape == "linear":                    # inside = above the line through the centre (y > 0 → outside)
        return Y
    if shape == "mirror":                    # band of half-width width/2 around the line
        return np.abs(Y) - float(P["width"]) / 2
    if shape == "circle":                    # ellipse (approximate SDF, exact for circles)
        rx, ry = max(1e-4, float(P["w"]) / 2), max(1e-4, float(P["h"]) / 2)
        k = np.sqrt((X / rx) ** 2 + (Y / ry) ** 2)
        return (k - 1.0) * min(rx, ry)
    if shape == "rectangle":
        hx, hy = float(P["w"]) / 2, float(P["h"]) / 2
        r = float(np.clip(P["roundness"], 0, 1)) * min(hx, hy)
        qx, qy = np.abs(X) - (hx - r), np.abs(Y) - (hy - r)
        out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
        return out + np.minimum(np.maximum(qx, qy), 0) - r
    if shape == "heart":                     # Inigo Quilez heart SDF, scaled to w
        sc = max(1e-4, float(P["w"]))
        x, y = np.abs(X) / sc, -Y / sc + 0.55          # y up, origin at the bottom tip
        d1 = np.sqrt((x - 0.25) ** 2 + (y - 0.75) ** 2) - math.sqrt(2) / 4
        t = np.maximum(x + y, 0) * 0.5
        d2 = np.sqrt((x - t) ** 2 + (y - t) ** 2)
        d3 = np.sqrt(x ** 2 + (y - 1) ** 2)
        d = np.where(x + y > 1.0, d1, np.minimum(d3, d2) * np.sign(x - y))
        return d * sc
    if shape == "star":                      # regular n-point star, outer radius w/2
        n = max(3, int(P["points"]))
        r = max(1e-4, float(P["w"]) / 2)
        inner = 1.0 - 0.6 * (1.0 - float(np.clip(P["roundness"], 0, 1)) * 0.5)
        ang = np.arctan2(X, -Y)
        seg = 2 * math.pi / n
        a = np.mod(ang + seg / 2, seg) - seg / 2
        rad = np.sqrt(X ** 2 + Y ** 2)
        px, py = np.abs(np.sin(a)) * rad, np.cos(a) * rad    # point folded into the first wedge, tip on +y
        tip = np.array([0.0, r])
        val = np.array([r * inner * math.sin(seg / 2), r * inner * math.cos(seg / 2)])
        e = val - tip
        h = np.clip(((px - tip[0]) * e[0] + (py - tip[1]) * e[1]) / float(e @ e), 0, 1)
        dx, dy = px - tip[0] - e[0] * h, py - tip[1] - e[1] * h
        dist = np.sqrt(dx * dx + dy * dy)
        side = (px - tip[0]) * e[1] - (py - tip[1]) * e[0]
        return np.where(side < 0, dist, -dist)
    raise KeyError(f"mask: unknown shape '{shape}'.  Use: {', '.join(SHAPES)}")


def mask_alpha(cfg, W, H, u=0.0):
    """mask config → alpha (H×W float32 0..1).  u = layer progress 0..1 for animated [from, to] values."""
    if cfg is None:
        return None
    if isinstance(cfg, str):
        cfg = {"shape": cfg}
    shape = cfg.get("shape", "rectangle")
    P = {k: _val(cfg.get(k, d), u) for k, d in _DEF.items()}
    d = sdf(shape, W, H, P)
    f = max(1.0 / min(W, H), float(P["feather"]))        # ≥ 1 px → antialiased
    x = np.clip(0.5 - d / f, 0, 1)
    a = (x * x * (3 - 2 * x)).astype(np.float32)
    if P["invert"]:
        a = 1.0 - a
    return a * float(P["opacity"])


def selftest():
    """resolution independence + shape sanity → (ok, msg)."""
    res = []
    for W, H in ((270, 480), (1080, 1920)):
        res.append({s: float(mask_alpha({"shape": s, "w": 0.5, "h": 0.5, "feather": 0.01}, W, H).mean())
                    for s in SHAPES})
    drift = max(abs(res[0][s] - res[1][s]) for s in SHAPES)
    a, b = res[1]["circle"], res[1]["rectangle"]
    ci = abs(a - math.pi * 0.25 ** 2 * 1080 ** 2 / (1080 * 1920)) < 0.01
    lin = abs(res[1]["linear"] - 0.5) < 0.01
    inv = mask_alpha({"shape": "circle", "invert": True}, 64, 64).mean() > 0.5
    anim = mask_alpha({"shape": "circle", "w": [0.0, 2.5], "h": [0.0, 2.5]}, 64, 64, 0.0).mean() < 0.01 < \
        mask_alpha({"shape": "circle", "w": [0.0, 2.5], "h": [0.0, 2.5]}, 64, 64, 1.0).mean()
    ok = (drift < 0.01 and ci and lin and b > a and inv and anim and 0.02 < res[1]["star"] < 0.2
          and 0.03 < res[1]["heart"] < 0.3)
    return ok, (f"{len(SHAPES)} SDF shapes, 480p↔1920p coverage drift {drift:.4f}, circle area exact {ci}, "
                f"linear half {lin}, invert {inv}, animated iris {anim}")
