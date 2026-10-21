"""
mograph.aefx.keyframes — After-Effects / Premiere temporal keyframe engine.

A curve is a plain number / vector (static) or {"k": [[t, v, interp, si, ii, so, io], ...]}:
    t       time (s, relative to the preset start)          v   float or [x, y]
    interp  "b" bezier (AE speed/influence) | "l" linear | "h" hold
    si, ii  INCOMING speed (units/s) + influence (0..1)  → used by the segment that ENDS at this key
    so, io  OUTGOING speed + influence                    → used by the segment that STARTS at this key
AE temporal bezier for k0 → k1 (dt = t1 - t0):
    P0 = (t0, v0)  P1 = (t0 + io0·dt, v0 + so0·io0·dt)  P2 = (t1 - ii1·dt, v1 - si1·ii1·dt)  P3 = (t1, v1)
Validated on the Premiere preview of "Position L to R 01": 2.4 px RMS on a 320 px frame (the swapped field order
gives 11.7 px → the order above is the real one).  Vector keys move along the straight segment, speed = arc length.
"""
from __future__ import annotations

import math

TICKS = 254016000000.0                     # Premiere ticks per second


def _bez_y(x0, x1, x2, x3, y0, y1, y2, y3, x):
    lo, hi = 0.0, 1.0
    for _ in range(22):
        u = (lo + hi) * 0.5
        m = 1.0 - u
        if m * m * m * x0 + 3 * m * m * u * x1 + 3 * m * u * u * x2 + u * u * u * x3 < x:
            lo = u
        else:
            hi = u
    u = (lo + hi) * 0.5
    m = 1.0 - u
    return m * m * m * y0 + 3 * m * m * u * y1 + 3 * m * u * u * y2 + u * u * u * y3


def _seg1(t, k0, k1, v0, v1):
    t0, t1 = k0[0], k1[0]
    dt = max(1e-9, t1 - t0)
    mode = k0[2] if len(k0) > 2 else "l"
    if mode == "h":
        return v0
    if mode == "l" or len(k0) < 7 or len(k1) < 7:
        return v0 + (v1 - v0) * (t - t0) / dt
    so, io = float(k0[5]), max(1e-4, min(1.0, float(k0[6])))
    si, ii = float(k1[3]), max(1e-4, min(1.0, float(k1[4])))
    return _bez_y(t0, t0 + io * dt, t1 - ii * dt, t1, v0, v0 + so * io * dt, v1 - si * ii * dt, v1, t)


def evaluate(curve, t, default=0.0):
    """Value of a curve at time t (seconds).  Static values are returned as-is; ends are held."""
    if curve is None:
        return default
    if not isinstance(curve, dict):
        return curve
    ks = curve["k"]
    if not ks:
        return default
    if t <= ks[0][0]:
        return ks[0][1]
    if t >= ks[-1][0]:
        return ks[-1][1]
    i = 1
    while ks[i][0] < t:
        i += 1
    k0, k1 = ks[i - 1], ks[i]
    a, b = k0[1], k1[1]
    if isinstance(a, (list, tuple)):
        L = math.sqrt(sum((bb - aa) ** 2 for aa, bb in zip(a, b)))
        if L < 1e-12:
            return list(a)
        s = _seg1(t, k0, k1, 0.0, L) / L
        return [aa + (bb - aa) * s for aa, bb in zip(a, b)]
    return _seg1(t, k0, k1, float(a), float(b))


def last_time(curve):
    return curve["k"][-1][0] if isinstance(curve, dict) and curve.get("k") else 0.0


def easy_ease(keys, influence=1 / 3):
    """[[t, v], ...] → AE 'Easy Ease' curve (speed 0, influence 33 %)."""
    return {"k": [[float(t), v, "b", 0.0, influence, 0.0, influence] for t, v in keys]}


def linear(keys):
    return {"k": [[float(t), v, "l", 0, 0, 0, 0] for t, v in keys]}


def scale_time(curve, k, t0=0.0):
    """Stretch a curve by k around t0 (speeds ÷ k so the SHAPE is preserved)."""
    if not isinstance(curve, dict):
        return curve
    out = []
    for row in curve["k"]:
        r = list(row)
        r[0] = t0 + (r[0] - t0) * k
        if len(r) >= 7:
            r[3] = r[3] / k
            r[5] = r[5] / k
        out.append(r)
    return {"k": out}


def parse_prproj_keys(s, t_origin_ticks, point=False, ndp=5):
    """Premiere '<Keyframes>' text → curve.  Row: time,value,interp(5 bez/0 lin/4 hold),?,in_spd,in_inf,out_spd,out_inf."""
    rows = []
    for item in s.strip().strip(";").split(";"):
        p = item.split(",")
        if len(p) < 8:
            continue
        t = (int(p[0]) - t_origin_ticks) / TICKS
        if point:
            v = [round(float(c), ndp) for c in p[1].split(":")]
        else:
            vv = p[1].strip()
            v = (1.0 if vv == "true" else 0.0) if vv in ("true", "false") else round(float(vv), ndp)
        mode = "l" if p[2] == "0" else ("h" if p[2] == "4" else "b")
        rows.append([round(t, 5), v, mode, round(float(p[4]), ndp), round(float(p[5]), ndp),
                     round(float(p[6]), ndp), round(float(p[7]), ndp)])
    rows.sort(key=lambda r: r[0])
    return {"k": rows} if rows else None
