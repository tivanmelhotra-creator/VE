"""
mograph.pro.rtime — v22 EXACT RATIONAL TIME (idea from Concat concat-core/src/time.rs, Rational(i64, i64)).

Floats drift at fractional frame rates: 29.97 is really 30000/1001, and `int(t * 29.97)` vs `round(...)` disagree on
some frames, which shifts chunk borders in a parallel render by one frame.  Every frame ↔ time conversion in the
pro renderer goes through these helpers:

    fps_q(29.97)        → Fraction(30000, 1001)    (NTSC rates snapped: 23.976, 29.97, 59.94, 47.952, 119.88)
    fps_q("30000/1001") → Fraction(30000, 1001)
    frame_at(t, fps)    → nearest frame index for a time in seconds (half-up, exact)
    time_of(i, fps)     → exact Fraction seconds of frame i   (float(time_of(...)) for the renderer)
    frames_in(dur, fps) → frame count of a duration
    chunks(a, b, step)  → [(f0, f1), …] contiguous, non-overlapping, covering [a, b) exactly
    ff_rate(fps)        → "30000/1001" for ffmpeg -r (never "29.97")
"""
from __future__ import annotations

from fractions import Fraction

_NTSC = {23.976: Fraction(24000, 1001), 29.97: Fraction(30000, 1001), 47.952: Fraction(48000, 1001),
         59.94: Fraction(60000, 1001), 119.88: Fraction(120000, 1001)}


def fps_q(fps):
    if isinstance(fps, Fraction):
        return fps
    if isinstance(fps, str) and "/" in fps:
        n, d = fps.split("/")
        return Fraction(int(n), int(d))
    f = float(fps)
    for k, v in _NTSC.items():
        if abs(f - k) < 0.006 or abs(f - float(v)) < 1e-6:
            return v
    q = Fraction(f).limit_denominator(1001)
    return q if abs(float(q) - f) < 1e-9 else Fraction(round(f * 1000), 1000)


def to_q(t):
    """seconds (float / str / Fraction) → Fraction, using the shortest decimal (0.1 → 1/10, not the binary float)."""
    if isinstance(t, Fraction):
        return t
    return Fraction(t) if isinstance(t, str) else Fraction(repr(float(t)))


def frame_at(t, fps):
    """nearest frame for time t (exact half-up rounding, no float error)."""
    x = to_q(t) * fps_q(fps)
    return int((x + Fraction(1, 2)) // 1)


def time_of(i, fps):
    return Fraction(int(i)) / fps_q(fps)


def frames_in(dur, fps):
    return frame_at(dur, fps)


def chunks(a, b, step):
    return [(f, min(b, f + step)) for f in range(int(a), int(b), max(1, int(step)))]


def ff_rate(fps):
    q = fps_q(fps)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def selftest():
    q = fps_q(29.97)
    ok = q == Fraction(30000, 1001) and fps_q("24000/1001") == fps_q(23.976) and fps_q(30) == 30
    rt = all(frame_at(time_of(i, q), q) == i for i in range(0, 216000, 7))        # 2 h of NTSC
    flt = int(33.4 * 29.97)                    # the float bug: frame 1001 lies at exactly 33.4 s
    ex = frame_at("33.4", q) == 1001
    cs = chunks(0, 1000, 120)
    cov = cs[0][0] == 0 and cs[-1][1] == 1000 and all(cs[i][1] == cs[i + 1][0] for i in range(len(cs) - 1))
    ff = ff_rate(29.97) == "30000/1001" and ff_rate(30.0) == "30"
    return ok and rt and ex and cov and ff, (f"NTSC snapped {q}, 2 h frame↔time round-trip exact {rt}, "
                                             f"33.4 s → frame 1001 (float int() gave {flt}), chunks contiguous {cov}")
