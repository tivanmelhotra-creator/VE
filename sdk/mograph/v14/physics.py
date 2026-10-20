"""
mograph.v14.physics — motion that feels ALIVE (the cure for "text moving like dry bread").

v13.9 audit: every animation used one scalar p (0→1) driving opacity/scale.  Real motion designers animate with
physics.  This module gives every layer:
  * closed-form damped spring with initial VELOCITY  (overshoot, settle)
  * squash & stretch from the motion derivative (volume preserved: sx·sy = 1)
  * secondary motion: per-glyph delay → wave / ripple / follow-through
  * jelly wobble: decaying scale + shear oscillation after an impact
All functions are PURE functions of time → frames render in any order / in parallel.
"""
from __future__ import annotations

import math


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def spring(t, k=170.0, c=12.0, m=1.0, v0=0.0):
    """Damped spring 0→1 for t ≥ 0 (initial velocity v0).  Overshoots when damping c is small."""
    if t <= 0:
        return 0.0
    w0 = math.sqrt(k / m)
    z = c / (2 * math.sqrt(k * m))
    x0 = -1.0
    if z < 1:
        wd = w0 * math.sqrt(1 - z * z)
        x = math.exp(-z * w0 * t) * (x0 * math.cos(wd * t) + ((v0 + z * w0 * x0) / wd) * math.sin(wd * t))
    else:
        x = math.exp(-w0 * t) * (x0 + (v0 + w0 * x0) * t)
    return 1.0 + x


PRESETS = {
    # name: (stiffness, damping)
    "jelly": (210.0, 7.5),     # big wobble ≈45 % overshoot — the @novadent jelly
    "bouncy": (260.0, 11.0),   # ≈30 %
    "snappy": (420.0, 24.0),   # ≈10 %, fast
    "smooth": (170.0, 22.0),   # ≈3 %, premium UI
    "heavy": (120.0, 16.0),    # slow weighty slam
    "stiff": (600.0, 42.0),    # no visible overshoot
}


def sp(t, preset="bouncy", v0=0.0):
    k, c = PRESETS.get(preset, PRESETS["bouncy"]) if isinstance(preset, str) else preset
    return spring(t, k, c, 1.0, v0)


def sp_vel(t, preset="bouncy", dt=1 / 240):
    return (sp(t + dt, preset) - sp(max(0.0, t - dt), preset)) / (2 * dt)


def bezier(x1, y1, x2, y2):
    def s(t, a, b):
        return 3 * a * (1 - t) ** 2 * t + 3 * b * (1 - t) * t * t + t ** 3

    def f(x):
        x = clamp(x)
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = (lo + hi) / 2
            if s(mid, x1, x2) < x:
                lo = mid
            else:
                hi = mid
        return s((lo + hi) / 2, y1, y2)
    return f


SILK = bezier(0.65, 0.0, 0.35, 1.0)      # the @novadent ease-in-out
EXPO_OUT = bezier(0.16, 1.0, 0.3, 1.0)
BACK_OUT = bezier(0.34, 1.56, 0.64, 1.0)


def ease_in_out(p):
    p = clamp(p)
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out(p):
    return 1 - (1 - clamp(p)) ** 3


def ease_in(p):
    return clamp(p) ** 3


def jelly(t, amp=1.0, freq=3.2, decay=4.2):
    """Decaying wobble after an impact at t=0 → (sx, sy, shear); volume preserved."""
    if t <= 0:
        return 1.0, 1.0, 0.0
    e = math.exp(-decay * t)
    w = 2 * math.pi * freq
    s = amp * 0.22 * e * math.sin(w * t + 0.35)
    sh = amp * 0.10 * e * math.sin(w * 0.83 * t + 1.4)
    sx = 1.0 + s
    return sx, 1.0 / max(0.5, sx), sh


def osc(t, freq=0.5, amp=1.0, seed=0.0):
    s = seed * 1.618
    return amp * (0.6 * math.sin(2 * math.pi * freq * t + s) + 0.4 * math.sin(2 * math.pi * freq * 1.73 * t + 2.1 * s))


def env(t, t_in, t_out, fin=0.25, fout=0.25):
    a = clamp((t - t_in) / max(1e-6, fin))
    if t_out is None:
        return a
    return min(a, clamp((t_out - t) / max(1e-6, fout)))
