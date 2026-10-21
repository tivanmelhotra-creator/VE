"""
mograph.aecomp.values — animatable property values for AE-style shape compositions.

A value is one of
    3 · [x, y] · "#ff3366"                        static (colours: hex or [r,g,b(,a)] 0..1)
    {"k": [[t, v, interp, si, ii, so, io, (to, ti)], ...]}   keyframes (t in comp seconds)
        interp "b" bezier (AE speed/influence) | "l" linear | "h" hold
        si/ii/so/io may be scalars or per-dimension lists (AE separate-dimension ease, e.g. Scale)
        to/ti (optional) spatial out/in tangents (relative, px) → curved motion paths with arc-length timing
        extra keys on the dict: "loop": "cycle" | "pingpong" | "offset" (AE loopOut),  "delay": s
    {"ctl": "stroke", "mul": 1, "add": 0}         linked to a comp CONTROL (AE "FX Control" slider/colour)
    {"rate": 180, "v": 0}                         v + rate·t  (AE  time*k  expression)
    {"wiggle": [freq, amp], "v": base, "seed": 1} AE wiggle()
Path values: {"v": [[x,y]..], "i": [[dx,dy]..], "o": [[dx,dy]..], "c": true}; keyframed paths interpolate vertices.
"""
from __future__ import annotations

import math

import numpy as np

from ..aefx.keyframes import _bez_y


def color(c):
    """hex / list → (r, g, b, a) floats."""
    if c is None:
        return (1.0, 1.0, 1.0, 1.0)
    if isinstance(c, str):
        h = c.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        a = int(h[6:8], 16) / 255.0 if len(h) >= 8 else 1.0
        return (r, g, b, a)
    c = [float(x) for x in c]
    return (c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0)


def _num(x):
    return isinstance(x, (int, float))


def _frac(t, k0, k1, d=None):
    """eased fraction 0..1 of the k0→k1 segment at t (dimension d for per-dim eases)."""
    t0, t1 = float(k0[0]), float(k1[0])
    dt = max(1e-9, t1 - t0)
    mode = k0[2] if len(k0) > 2 else "l"
    if mode == "h":
        return 0.0
    u = (t - t0) / dt
    in_mode = k1[2] if len(k1) > 2 else "l"
    # AE: a segment is linear only when BOTH the out of k0 and the in of k1 are linear; otherwise it is a bezier
    # whose linear side carries speed = average slope (py_aep exports that speed, so the bezier maths is exact)
    if len(k0) < 7 or len(k1) < 7 or (mode == "l" and in_mode == "l"):
        return u
    def pick(q):
        if isinstance(q, (list, tuple)):
            return float(q[d if d is not None and d < len(q) else 0])
        return float(q)
    so, io = pick(k0[5]), max(1e-4, min(1.0, pick(k0[6])))
    si, ii = pick(k1[3]), max(1e-4, min(1.0, pick(k1[4])))
    return so, io, si, ii, t0, t1


def _seg_scalar(t, k0, k1, a, b, d=None):
    r = _frac(t, k0, k1, d)
    if not isinstance(r, tuple):
        return a + (b - a) * r
    so, io, si, ii, t0, t1 = r
    dt = t1 - t0
    return _bez_y(t0, t0 + io * dt, t1 - ii * dt, t1, a, a + so * io * dt, b - si * ii * dt, b, t)


def _cubic(p0, p1, p2, p3, u):
    m = 1 - u
    return m * m * m * p0 + 3 * m * m * u * p1 + 3 * m * u * u * p2 + u * u * u * p3


def _spatial(t, k0, k1):
    """position with spatial tangents: arc-length parametrised cubic, timing from the temporal ease."""
    a, b = np.array(k0[1], float), np.array(k1[1], float)
    to = np.array(k0[7], float) if len(k0) > 7 and k0[7] is not None else np.zeros_like(a)
    ti = np.array(k1[8], float) if len(k1) > 8 and k1[8] is not None else np.zeros_like(a)
    n = min(len(a), len(to), len(ti))
    a2, b2 = a[:n], b[:n]
    P = [a2, a2 + to[:n], b2 + ti[:n], b2]
    us = np.linspace(0, 1, 33)
    pts = np.array([_cubic(*P, u) for u in us])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    L = seg[-1]
    if L < 1e-9:
        return list(a)
    s = _seg_scalar(t, k0, k1, 0.0, L)
    u = float(np.interp(s, seg, us))
    out = list(_cubic(*P, u))
    return out + list(a[n:])


def _path_lerp(a, b, f):
    out = {"c": a.get("c", True)}
    for key in ("v", "i", "o"):
        A, B = np.array(a[key], float), np.array(b[key], float)
        if A.shape != B.shape:
            return a if f < 0.5 else b
        out[key] = (A + (B - A) * f).tolist()
    return out


def eval_keys(ks, t):
    if t <= ks[0][0]:
        return ks[0][1]
    if t >= ks[-1][0]:
        return ks[-1][1]
    i = 1
    while ks[i][0] < t:
        i += 1
    k0, k1 = ks[i - 1], ks[i]
    a, b = k0[1], k1[1]
    if (k0[2] if len(k0) > 2 else "l") == "h":
        return a
    if isinstance(a, dict):                                   # path key
        r = _frac(t, k0, k1)
        f = r if not isinstance(r, tuple) else _seg_scalar(t, k0, k1, 0.0, 1.0)
        return _path_lerp(a, b, f)
    if isinstance(a, str) or isinstance(b, str):              # colour keys
        ca, cb = color(a), color(b)
        f = _seg_scalar(t, k0, k1, 0.0, 1.0)
        return [x + (y - x) * f for x, y in zip(ca, cb)]
    if _num(a):
        return _seg_scalar(t, k0, k1, float(a), float(b))
    perdim = len(k0) >= 7 and (isinstance(k0[5], (list, tuple)) or isinstance(k1[3], (list, tuple)))
    if len(k0) > 7 and (k0[7] is not None or (len(k1) > 8 and k1[8] is not None)):
        return _spatial(t, k0, k1)
    if perdim:
        return [_seg_scalar(t, k0, k1, float(x), float(y), d) for d, (x, y) in enumerate(zip(a, b))]
    # spatial straight segment: one speed for the arc length
    A, B = np.array(a, float), np.array(b, float)
    L = float(np.linalg.norm(B - A))
    if L < 1e-12:
        return list(a)
    s = _seg_scalar(t, k0, k1, 0.0, L) / L
    return list(A + (B - A) * s)


def _noise(x, seed):
    i = math.floor(x)
    f = x - i
    h = lambda n: (math.sin(n * 127.1 + seed * 311.7) * 43758.5453) % 1.0 * 2 - 1  # noqa: E731
    u = f * f * (3 - 2 * f)
    return h(i) * (1 - u) + h(i + 1) * u


def ev(v, t, C=None):
    """evaluate a value at comp time t; C = controls dict."""
    if isinstance(v, dict):
        if "k" in v:
            ks = v["k"]
            if not ks:
                return 0.0
            tt = t - float(v.get("delay", 0.0))
            lp = v.get("loop")
            if lp and len(ks) > 1 and tt > ks[-1][0]:
                a0, a1 = ks[0][0], ks[-1][0]
                span = max(1e-6, a1 - a0)
                n, r = divmod(tt - a0, span)
                if lp == "pingpong" and int(n) % 2 == 1:
                    tt = a1 - r
                elif lp == "offset":
                    base = eval_keys(ks, a0 + r)
                    d0, d1 = ks[0][1], ks[-1][1]
                    if _num(base):
                        return base + n * (d1 - d0)
                    return [b + n * (y - x) for b, x, y in zip(base, d0, d1)]
                else:
                    tt = a0 + r
            return eval_keys(ks, tt)
        if "ctl" in v:
            base = (C or {}).get(v["ctl"], v.get("default", 0.0))
            base = ev(base, t, C)
            if isinstance(base, str):
                return base
            m, a = v.get("mul", 1.0), v.get("add", 0.0)
            if _num(base):
                return base * m + a
            if isinstance(base, (list, tuple)) and v.get("vec"):
                return [base * m + a] * int(v["vec"]) if _num(base) else base
            return base
        if "sep" in v:                                       # separated dimensions (AE X/Y Position)
            return [num(q, t, C) for q in v["sep"]]
        if "rate_ctl" in v:
            return float(ev((C or {}).get(v["rate_ctl"], 0.0), t, C)) * t + float(ev(v.get("v", 0.0), t, C))
        if "trace" in v:                                     # resolved by the renderer (needs the other layer)
            fn = (C or {}).get("__trace__")
            if fn is not None:
                r = fn(v, t)
                if r is not None:
                    return r
            return ev(v.get("v", 0.0), t, C)
        if "rate" in v:
            base = ev(v.get("v", 0.0), t, C)
            if _num(base):
                return base + float(v["rate"]) * t
            return base
        if "wiggle" in v:
            base = ev(v.get("v", 0.0), t, C)
            fq, amp = v["wiggle"]
            sd = float(v.get("seed", 1))
            if _num(base):
                return base + amp * _noise(t * fq, sd)
            return [b + amp * _noise(t * fq, sd + 17 * j) for j, b in enumerate(base)]
        if "v" in v and ("i" in v or "c" in v):            # static path
            return v
        if "v" in v:
            return ev(v["v"], t, C)
    return v


def vec(v, t, C=None, n=2, default=0.0):
    x = ev(v, t, C)
    if x is None:
        return [default] * n
    if _num(x):
        return [float(x)] * n
    x = [float(q) for q in x]
    while len(x) < n:
        x.append(default)
    return x[:n]


def num(v, t, C=None, default=0.0):
    x = ev(v, t, C)
    if x is None:
        return default
    if _num(x):
        return float(x)
    return float(x[0])


def col(v, t, C=None):
    return color(ev(v, t, C))
