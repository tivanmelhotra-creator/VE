"""Keyframes, easing and AE-style expressions for vector animation.

Track: keyframed scalar/vector with easing per segment (`linear`, `smooth`, `io`, `back`
overshoot, `out`, `in`, `hold`).  Expressions (After Effects names): `wiggle`, `lag` (delayed
follow-through, like `valueAtTime(time - d)`), `spring` (damped overshoot after a change),
`loop_pingpong`.  Times can be snapped to audio onsets (`snap`).
"""
import numpy as np


def _c(t):
    return float(np.clip(t, 0.0, 1.0))


def ease_io(t, p=3.0):
    t = _c(t)
    return 0.5 * (2 * t) ** p if t < 0.5 else 1 - 0.5 * (2 - 2 * t) ** p


def back(t, s=1.4):
    t = _c(t) - 1
    return t * t * ((s + 1) * t + s) + 1


EASE = {
    'linear': _c,
    'smooth': lambda t: _c(t) ** 2 * (3 - 2 * _c(t)),
    'io': ease_io,
    'out': lambda t: 1 - (1 - _c(t)) ** 3,
    'in': lambda t: _c(t) ** 3,
    'back': back,
    'hold': lambda t: 0.0 if t < 1 else 1.0,
}


class Track:
    """keys: [(t, value, ease_into_this_key)]"""
    def __init__(self, keys):
        self.keys = sorted([(float(k[0]), np.asarray(k[1], float), k[2] if len(k) > 2 else 'io') for k in keys],
                           key=lambda k: k[0])

    def __call__(self, t):
        K = self.keys
        if t <= K[0][0]:
            return K[0][1]
        for (t0, v0, _), (t1, v1, e) in zip(K, K[1:]):
            if t <= t1:
                return v0 + (v1 - v0) * EASE[e]((t - t0) / max(t1 - t0, 1e-6))
        return K[-1][1]

    def times(self):
        return [k[0] for k in self.keys]


def const(v):
    return Track([(0, v)])


# --------------------------------------------------------------------------- expressions
def _noise1(x, seed):
    """smooth 1-D value noise in [-1, 1]"""
    i = np.floor(x); f = x - i
    def h(n):
        return (np.sin((n + seed * 17.13) * 127.1) * 43758.5453) % 1.0 * 2 - 1
    u = f * f * (3 - 2 * f)
    return h(i) * (1 - u) + h(i + 1) * u


def wiggle(freq, amp, seed=0, octaves=2):
    """AE wiggle(freq, amp): returns f(t)."""
    def f(t):
        v, a, fr = 0.0, 1.0, freq
        for o in range(octaves):
            v += a * _noise1(t * fr, seed + o * 7)
            a *= 0.5; fr *= 2
        return amp * v / 1.5
    return f


def lag(track, delay):
    """follow-through: value at (t - delay)"""
    return lambda t: track(t - delay)


def spring(track, freq=3.0, decay=5.0, amount=0.25):
    """Overshoot after every key (AE 'inertial bounce' expression): adds a damped sine
    proportional to the velocity the track had arriving at its last key."""
    T = track.times()

    def f(t):
        v = track(t)
        past = [k for k in T if k <= t]
        if len(past) == 0 or len(T) < 2:
            return v
        k = past[-1]
        dt = 1e-2
        vel = (np.asarray(track(k)) - np.asarray(track(k - dt))) / dt
        e = t - k
        return v + amount * vel * np.sin(freq * e * 2 * np.pi) / np.exp(decay * e) / (freq * 2 * np.pi)
    return f


def snap(t, onsets, window=0.25):
    """Move a key time to the nearest audio onset within `window` seconds."""
    if onsets is None or len(onsets) == 0:
        return t
    o = np.asarray(onsets)
    i = int(np.argmin(np.abs(o - t)))
    return float(o[i]) if abs(o[i] - t) <= window else t
