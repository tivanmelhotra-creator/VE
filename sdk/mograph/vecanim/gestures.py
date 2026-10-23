"""Gesture library: reusable acting blocks.  `char.do(name, t=..., dur=..., **params)`.

Each gesture is built ONCE from the character's landmarks / pin groups (rig JSON), so the
same call works on any rigged character in any scene:

    tilt_head(angle=deg, pivot='neck')        head pins rotate about the neck (ARAP bends the rest)
    turn_head(angle=deg)                      alias with a softer ease
    nod(angle=deg)                            down-up with spring follow-through
    smile(amount=1, from_=0)                  mouth corners up/out (MLS-rigid local warp)
    blink(close=0.13)                         eyelids squash to the lash line and back
    look(dx, dy)                              pupils shift (small local shift)
    breathe(period, amount)                   chest/shoulders rise (continuous)
    lift_shoulder(side, dy)                   shoulder pin offset
    hand_on_shoulder(target, side='far')      own shoulder lifts, a vector hand settles on the
                                              target's shoulder from behind (clip = shoulder line)
    sway(amp, freq)                           idle wiggle of the head pins (AE wiggle)
"""
import numpy as np
from . import anim as A
from .rig import rot, local_warp, squash, shift
from .bezier import Style, fit_closed, fit_open, chain_to_points
from .scene import Fill, Stroke


class Gesture:
    def __init__(self, fn):
        self.fn = fn

    def apply(self, st, t):
        self.fn(st, t)


def _env(t0, dur, ease='io'):
    return A.Track([(t0, 0.0), (t0 + dur, 1.0, ease)])


# ------------------------------------------------------------------ head
def tilt_head(ch, t, dur=None, angle=4.0, pivot='neck', ease='io', hold=None, back=None):
    dur = dur or 1.0
    keys = [(t, 0.0), (t + dur, np.radians(angle), ease)]
    if back is not None:                       # return at time `back`
        keys += [(back, np.radians(angle), 'hold'), (back + dur, 0.0, ease)]
    tr = A.Track(keys)
    sp = A.spring(tr, freq=1.6, decay=4.0, amount=0.18)
    piv = ch.L[pivot]
    names = set(ch.groups['head'])

    def f(st, tt):
        a = float(sp(tt))
        if abs(a) > 1e-5:
            st.rotations.append((piv, a, names))
    return Gesture(f)


def turn_head(ch, t, dur=None, angle=-6.0, from_=None, **kw):
    """from_: starting angle (the drawing's pose = 0).  e.g. from_=7 -> starts turned away."""
    dur = dur or 1.2
    piv = ch.L['neck']; names = set(ch.groups['head'])
    a0 = np.radians(from_ or 0.0)
    tr = A.Track([(0, a0), (t, a0), (t + dur, np.radians(angle), 'io'), (t + dur * 1.6, np.radians(angle) * 0.6, 'io')])

    def f(st, tt):
        a = float(tr(tt))
        if abs(a) > 1e-5:
            st.rotations.append((piv, a, names))
    return Gesture(f)


def nod(ch, t, dur=None, angle=3.0):
    dur = dur or 0.7
    tr = A.Track([(t, 0.0), (t + dur * 0.4, np.radians(angle), 'io'), (t + dur, 0.0, 'io')])
    sp = A.spring(tr, freq=2.2, decay=5, amount=0.3)
    piv = ch.L['neck']; names = set(ch.groups['head'])
    return Gesture(lambda st, tt: st.rotations.append((piv, float(sp(tt)), names)))


def sway(ch, t, dur=None, amp=0.6, freq=0.35, seed=1):
    w = A.wiggle(freq, np.radians(amp), seed=seed)
    piv = ch.L['neck']; names = set(ch.groups['head'])
    env = _env(t, 1.0)
    return Gesture(lambda st, tt: st.rotations.append((piv, float(w(tt)) * float(env(tt)), names)))


# ------------------------------------------------------------------ face
def smile(ch, t, dur=None, amount=1.0, from_=0.0, lift=3.2, pull=1.4):
    """Corner landmarks: mouth_back (the visible corner) and mouth_front."""
    dur = dur or 0.8
    tr = A.Track([(0, from_), (t, from_), (t + dur, amount, 'io')])
    mb, mf = ch.L['mouth_back'], ch.L['mouth_front']
    c = (mb + mf) / 2
    r = np.linalg.norm(mb - mf) * 1.25 + 14
    up = ch.spec.get('smile_dir', [0, -1])
    up = np.asarray(up, float)
    out = (mb - mf) / max(np.linalg.norm(mb - mf), 1e-6)

    def f(st, tt):
        k = float(tr(tt))
        if abs(k) < 1e-3:
            return
        src = np.array([mb, mf, c + up * 9, c - up * 9])
        dst = np.array([mb + k * (up * lift + out * pull), mf + k * up * lift * 0.35, c + up * 9, c - up * 9])
        st.local.append(lambda P, s=src, d=dst: local_warp(P, c, r, s, d))
    return Gesture(f)


def blink(ch, t, dur=None, close=None):
    dur = dur or 0.26
    eyes = [ch.L[k] for k in ('eye_l', 'eye_r') if k in ch.L]
    rx, ry = ch.spec.get('eye_size', [15, 10])

    def f(st, tt):
        b = max(0.0, 1 - abs(tt - (t + dur / 2)) / (dur / 2))
        b = b * b * (3 - 2 * b)
        if b < 1e-3:
            return
        for e in eyes:
            st.local.append(lambda P, e=e, b=b: squash(P, e, rx, ry, 0.85 * b, e[1] + ry * 0.45))
    return Gesture(f)


def look(ch, t, dur=None, dx=0.0, dy=0.0, from_=(0, 0)):
    dur = dur or 0.6
    tr = A.Track([(0, from_), (t, from_), (t + dur, (dx, dy), 'io')])
    eyes = [ch.L[k] for k in ('eye_l', 'eye_r') if k in ch.L]
    r = ch.spec.get('eye_size', [15, 10])[1] * 0.9

    def f(st, tt):
        d = np.asarray(tr(tt), float)
        if np.abs(d).max() < 0.05:
            return
        for e in eyes:
            st.local.append(lambda P, e=e, d=d: shift(P, e, r, d))
    return Gesture(f)


# ------------------------------------------------------------------ body
def breathe(ch, t, dur=None, period=3.4, amount=1.6):
    names = ch.groups.get('chest', [])

    def f(st, tt):
        k = np.sin((tt - t) * 2 * np.pi / period) * amount * min(1.0, max(0.0, tt - t))
        for n in names:
            st.add_offset(n, (0, -k))
    return Gesture(f)


def lift_shoulder(ch, t, dur=None, side='r', dy=-6.0, dx=3.0):
    dur = dur or 0.8
    tr = _env(t, dur)
    names = ch.groups.get('shoulder_' + side, [])

    def f(st, tt):
        k = float(tr(tt))
        for i, n in enumerate(names):
            s = 1.0 / (1 + i)
            st.add_offset(n, (dx * k * s, dy * k * s))
    return Gesture(f)


# ------------------------------------------------------------------ hand asset (vector)
_HAND = None


def hand_asset():
    """A foreshortened hand seen from behind a shoulder, as vector items in a local frame
    (x along the knuckle line, y down the fingers, origin at the knuckles).  Built from
    capsule outlines fitted with the same Bézier fitter as the traced art."""
    global _HAND
    if _HAND is not None:
        return _HAND
    st = Style(step=0.5, sigma=0.8, corner_deg=80, corner_win=2.0, tol=0.25, line_tol=0.2)
    fingers = [(-19, 24, 0.20, 10.0), (-6.5, 32, 0.07, 11.4), (6.5, 30, -0.04, 11.0), (18, 21, -0.16, 9.2)]
    fills, inks, outl = [], [], []

    def capsule(x, L, sp, wd, n=40):
        d = np.array([np.sin(sp), np.cos(sp)]); nr = np.array([d[1], -d[0]])
        r = wd / 2; tip = np.array([x, 0]) + d * L
        side1 = [np.array([x, -2]) + nr * r + d * L * u for u in np.linspace(0, 1, n // 2)]
        a0 = np.arctan2(nr[1], nr[0])
        arc = [tip + r * 0.8 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(a0, a0 + np.pi, n // 2)]
        side2 = [np.array([x, -2]) - nr * r + d * L * u for u in np.linspace(1, 0, n // 2)]
        return np.array(side1 + arc + side2)

    back = np.array([(-30, 2)] + [(-30 + 58 * u, -26 * np.sin(np.pi * u) - 2 * (1 - u) + 2 * u) for u in np.linspace(0, 1, 30)] + [(28, 2)])
    back = np.vstack([back, [(0, 6)]])
    for poly in [back] + [capsule(*f) for f in fingers][::-1]:
        segs = fit_closed(poly, st)
        P = chain_to_points(segs, True)
        fills.append(Fill([P], 0.86, group='hand_fill'))
        inks.append(Stroke(P, True, 0.9, None, 0.06, group='hand_ink'))
    # creases: knuckle line + finger joints
    for x, L, sp, wd in fingers:
        d = np.array([np.sin(sp), np.cos(sp)]); nr = np.array([d[1], -d[0]])
        for k, ln in ((0.42, 3.0), (0.72, 2.4)):
            c = np.array([x, 0]) + d * L * k
            seg = np.array([c - nr * ln, c, c + nr * ln * 0.7])
            inks.append(Stroke(chain_to_points(fit_open(seg, st), False), False, 0.8, None, 0.06, group='hand_ink'))
    ridge = np.array([(-24, -4), (-8, -8), (8, -8), (24, -4)])
    inks.append(Stroke(chain_to_points(fit_open(ridge, st), False), False, 0.8, None, 0.35, group='hand_ink'))
    _HAND = (fills, inks)
    return _HAND


def _xf(items, M):
    out = []
    for it in items:
        c = it.copy()
        P = it.points()
        c.set_points(P @ M[:, :2].T + M[:, 2])
        if c.kind == 'stroke':
            sc = np.sqrt(abs(np.linalg.det(M[:, :2])))
            c.width *= sc; c.aw = c.aw * sc
        out.append(c)
    return out


def hand_on_shoulder(ch, t, dur=None, target=None, side='far', size=1.6, angle=0.40, drop=55.0, slide=14.0):
    """The character lifts his shoulder; his hand lands on `target`'s shoulder from behind.
    Target landmarks: shoulder_<side> (knuckle anchor) and shoulder_<side>_line (contour)."""
    dur = dur or 1.1
    lift = lift_shoulder(ch, t - 0.4, dur * 0.9, side=ch.spec.get('reach_side', 'r'))
    k_tr = A.Track([(t, 0.0), (t + dur, 1.0, 'back')])
    fade = A.Track([(t, 0.0), (t + dur * 0.55, 1.0, 'smooth')])
    anchor = target.L['shoulder_' + side]
    line = target.L['shoulder_' + side + '_line']
    fills, inks = hand_asset()
    c, s = np.cos(angle) * size, np.sin(angle) * size

    def f(st, tt):
        lift.apply(st, tt)
        k = float(k_tr(tt))
        if k <= 1e-3:
            return
        dx, dy = (1 - k) * slide, (1 - k) * -drop
        M = np.array([[c, -s, anchor[0] + dx], [s, c, anchor[1] + dy]])
        items = _xf(fills, M) + _xf(inks, M)
        lift_px = 4 * min(1.0, k * 1.3)
        clip = [(x, y - lift_px) for x, y in line] + [(line[-1][0], line[-1][1] + 150), (line[0][0], line[0][1] + 150)]
        for it in items:
            it.clip = clip
            it.opacity = float(fade(tt))
        st.overlay.append((target.z + 0.5, items))
    return Gesture(f)


GESTURES = dict(tilt_head=tilt_head, turn_head=turn_head, nod=nod, sway=sway, smile=smile, blink=blink,
                look=look, breathe=breathe, lift_shoulder=lift_shoulder, hand_on_shoulder=hand_on_shoulder)


def _wrap(fn):
    def g(ch, t, dur=None, **kw):
        return fn(ch, t, dur, **kw)
    return g


GESTURES = {k: _wrap(v) for k, v in GESTURES.items()}
