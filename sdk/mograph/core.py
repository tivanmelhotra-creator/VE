"""
mograph.core — time, easing, springs, keyframe tracks, camera.

Everything here is pure-python and deterministic: a frame at time t is a pure
function of t. That property is what lets the renderer split frames across
CPU cores and lets QA render any single frame instantly.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------

def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else float(v)


def lerp(a, b, p):
    return a + (b - a) * p


def lerp_tuple(a, b, p):
    return tuple(x + (y - x) * p for x, y in zip(a, b))


def prog(t, t0, dur):
    """0..1 progress of a window that starts at t0 and lasts dur seconds."""
    if dur <= 1e-6:
        return 1.0 if t >= t0 else 0.0
    return clamp((t - t0) / dur)


def window(t, t_in, t_out, fade_in=0.35, fade_out=0.35):
    """Envelope 0→1→0 : rises after t_in, falls before t_out (t_out=None → never)."""
    a = prog(t, t_in, fade_in)
    if t_out is None:
        return a
    b = 1.0 - prog(t, t_out - fade_out, fade_out)
    return min(a, b)


# ---------------------------------------------------------------------------
# easing library (all map 0..1 → 0..1)
# ---------------------------------------------------------------------------

def linear(p): return clamp(p)
def ease_in(p): p = clamp(p); return p * p * p
def ease_out(p): p = clamp(p); return 1 - (1 - p) ** 3
def ease_in_out(p):
    p = clamp(p)
    return 4 * p * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2
def ease_out_quart(p): p = clamp(p); return 1 - (1 - p) ** 4
def ease_out_expo(p): p = clamp(p); return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)
def ease_in_out_expo(p):
    p = clamp(p)
    if p in (0.0, 1.0):
        return p
    return 2 ** (20 * p - 10) / 2 if p < 0.5 else (2 - 2 ** (-20 * p + 10)) / 2
def ease_out_back(p, s=1.70158):
    p = clamp(p) - 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2
def ease_in_out_sine(p): p = clamp(p); return -(math.cos(math.pi * p) - 1) / 2


def bezier_ease(x1, y1, x2, y2):
    """CSS cubic-bezier(x1,y1,x2,y2) — e.g. bezier_ease(.2,.8,.2,1) (Motion/Framer default-ish)."""
    def sample(t, a1, a2):
        return 3 * a1 * (1 - t) ** 2 * t + 3 * a2 * (1 - t) * t * t + t ** 3

    def f(x):
        x = clamp(x)
        lo, hi = 0.0, 1.0
        for _ in range(22):  # bisection: robust & plenty precise for 60fps
            mid = (lo + hi) / 2
            if sample(mid, x1, x2) < x:
                lo = mid
            else:
                hi = mid
        return sample((lo + hi) / 2, y1, y2)
    return f


EASINGS: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "in": ease_in,
    "out": ease_out,
    "in_out": ease_in_out,
    "out_quart": ease_out_quart,
    "in_expo": lambda p: 0.0 if p <= 0 else 2 ** (10 * (min(p, 1.0) - 1)),
    "out_expo": ease_out_expo,
    "in_out_expo": ease_in_out_expo,
    "out_back": ease_out_back,
    "in_out_quart": lambda p: 8 * clamp(p) ** 4 if p < 0.5 else 1 - (-2 * clamp(p) + 2) ** 4 / 2,
    "sine": ease_in_out_sine,
    "smooth": bezier_ease(0.22, 1.0, 0.36, 1.0),      # premium UI ease-out
    "snappy": bezier_ease(0.16, 1.0, 0.3, 1.0),
    "anticipate": bezier_ease(0.68, -0.35, 0.27, 1.35),
    # v12 — full Penner set (cubic / quint / circ / back / elastic / bounce)
    "in_cubic": lambda p: clamp(p) ** 3,
    "out_cubic": lambda p: 1 - (1 - clamp(p)) ** 3,
    "in_out_cubic": lambda p: 4 * clamp(p) ** 3 if p < 0.5 else 1 - (-2 * clamp(p) + 2) ** 3 / 2,
    "in_quart": lambda p: clamp(p) ** 4,
    "out_quint": lambda p: 1 - (1 - clamp(p)) ** 5,
    "in_out_quint": lambda p: 16 * clamp(p) ** 5 if p < 0.5 else 1 - (-2 * clamp(p) + 2) ** 5 / 2,
    "out_circ": lambda p: math.sqrt(max(0.0, 1 - (clamp(p) - 1) ** 2)),
    "in_back": lambda p: 2.70158 * clamp(p) ** 3 - 1.70158 * clamp(p) ** 2,
    "out_elastic": lambda p: 0.0 if p <= 0 else 1.0 if p >= 1 else
        2 ** (-10 * p) * math.sin((p * 10 - 0.75) * (2 * math.pi) / 3) + 1,
    "out_bounce": lambda p: _bounce(clamp(p)),
}


def _bounce(p):
    n, d = 7.5625, 2.75
    if p < 1 / d:
        return n * p * p
    if p < 2 / d:
        p -= 1.5 / d; return n * p * p + 0.75
    if p < 2.5 / d:
        p -= 2.25 / d; return n * p * p + 0.9375
    p -= 2.625 / d
    return n * p * p + 0.984375


def get_ease(e) -> Callable[[float], float]:
    if callable(e):
        return e
    if e is None:
        return ease_in_out
    if e not in EASINGS:                     # v12: tolerant (unknown name → smooth in/out, QA lints it)
        return ease_in_out
    return EASINGS[e]


# ---------------------------------------------------------------------------
# springs (closed form → deterministic per frame, no state needed)
# ---------------------------------------------------------------------------

def spring(t, t0=0.0, stiffness=180.0, damping=14.0, mass=1.0):
    """Damped harmonic spring from 0 → 1 starting at t0 (physical params, like Motion/Framer)."""
    if t <= t0:
        return 0.0
    x = t - t0
    w0 = math.sqrt(stiffness / mass)
    zeta = damping / (2 * math.sqrt(stiffness * mass))
    if zeta < 1:
        wd = w0 * math.sqrt(1 - zeta * zeta)
        v = math.exp(-zeta * w0 * x) * (math.cos(wd * x) + (zeta * w0 / wd) * math.sin(wd * x))
    else:  # critically / over damped
        v = math.exp(-w0 * x) * (1 + w0 * x)
    return 1.0 - v


SPRINGS = {
    "gentle": dict(stiffness=120, damping=14),
    "default": dict(stiffness=180, damping=14),
    "bouncy": dict(stiffness=260, damping=11),
    "stamp": dict(stiffness=420, damping=20),
    "stiff": dict(stiffness=380, damping=30),
}


def spring_preset(t, t0, name="default"):
    return spring(t, t0, **SPRINGS[name])


def stagger(t, t0, idx, step=0.08, dur=0.5, ease="out"):
    return get_ease(ease)(prog(t, t0 + idx * step, dur))


def wiggle(t, freq=0.6, amp=1.0, seed=0):
    """Smooth, deterministic idle motion (sum of incommensurate sines)."""
    s = seed * 1.618
    return amp * (0.6 * math.sin(2 * math.pi * freq * t + s)
                  + 0.4 * math.sin(2 * math.pi * freq * 1.73 * t + 2.1 * s))


# ---------------------------------------------------------------------------
# keyframe track
# ---------------------------------------------------------------------------

class Track:
    """
    Keyframe track: Track([(t, value), (t, value, "ease_name"), ...])
    The ease on a key controls the segment that ENDS at that key.
    Values can be floats or tuples (positions, colors, camera views).
    """

    def __init__(self, keys: Sequence[tuple]):
        ks = []
        for k in keys:
            if len(k) == 2:
                ks.append((float(k[0]), k[1], "in_out"))
            else:
                ks.append((float(k[0]), k[1], k[2]))
        ks.sort(key=lambda k: k[0])
        if not ks:
            raise ValueError("Track needs at least one key")
        self.keys = ks

    def __call__(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1]
        if t >= ks[-1][0]:
            return ks[-1][1]
        for i in range(1, len(ks)):
            if t <= ks[i][0]:
                t0, v0, _ = ks[i - 1]
                t1, v1, e = ks[i]
                p = get_ease(e)((t - t0) / max(1e-6, t1 - t0))
                if isinstance(v0, (tuple, list)):
                    return lerp_tuple(v0, v1, p)
                return lerp(v0, v1, p)
        return ks[-1][1]


# ---------------------------------------------------------------------------
# 2D camera with keyframes (+ optional subtle handheld drift)
# ---------------------------------------------------------------------------

class Camera:
    """
    World-space camera. View = (cx, cy, zoom[, rot_deg]).
    cam = Camera(1080, 1920, keys=[(0,(540,960,1)), (4,(540,700,1.4),"smooth")])
    cam.apply(ctx, t)  # call before drawing world-space content
    """

    def __init__(self, width, height, keys=None, drift=0.0):
        self.w, self.h = width, height
        self.track = Track(keys or [(0.0, (width / 2, height / 2, 1.0))])
        self.drift = drift

    def view(self, t):
        v = self.track(t)
        cx, cy, z = v[0], v[1], v[2]
        rot = v[3] if len(v) > 3 else 0.0
        if self.drift:
            cx += wiggle(t, 0.07, self.drift, 1)
            cy += wiggle(t, 0.05, self.drift, 2)
        return cx, cy, z, rot

    def apply(self, ctx, t):
        cx, cy, z, rot = self.view(t)
        ctx.translate(self.w / 2, self.h / 2)
        if rot:
            ctx.rotate(math.radians(rot))
        ctx.scale(z, z)
        ctx.translate(-cx, -cy)

    def world_to_screen(self, t, x, y):
        cx, cy, z, rot = self.view(t)
        dx, dy = (x - cx) * z, (y - cy) * z
        if rot:
            r = math.radians(rot)
            dx, dy = dx * math.cos(r) - dy * math.sin(r), dx * math.sin(r) + dy * math.cos(r)
        return self.w / 2 + dx, self.h / 2 + dy


# ---------------------------------------------------------------------------
# colors
# ---------------------------------------------------------------------------

def hex_rgba(h: str, a: float = 1.0):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    if len(h) == 8:
        a = int(h[6:8], 16) / 255 * a
    return (r, g, b, a)


def color(c, alpha_mul=1.0):
    """Accept '#fff', '#ffffffcc', (r,g,b), (r,g,b,a) with 0..1 floats; multiply alpha."""
    if isinstance(c, str):
        r, g, b, a = hex_rgba(c)
    elif len(c) == 3:
        r, g, b = c
        a = 1.0
    else:
        r, g, b, a = c
    return (r, g, b, a * alpha_mul)


def with_alpha(c, a):
    r, g, b, a0 = color(c)
    return (r, g, b, a0 * a)
