"""
toonface.kit.visemes — 14 hand-tuned mouth key-shapes (one topology) + Persian G2P → timed viseme track.

Params: open, upper (upper-lip lift), width, round, teeth_up, teeth_lo, tongue, tongue_up, press, fv, corner, skew,
        tongue_out (0..1 stuck out over the lower lip), tongue_side (-1..1 lateral slide)
"""
from __future__ import annotations


VISEMES = {
    "X":    dict(open=0.0, width=0.92),
    "MBP":  dict(open=0.0, width=0.86, press=1.0),
    "AA":   dict(open=0.85, upper=0.10, width=0.98, teeth_up=0.45, tongue=0.55),
    "AH":   dict(open=0.62, upper=0.08, width=0.95, teeth_up=0.40, tongue=0.40),
    "EH":   dict(open=0.42, upper=0.08, width=1.02, teeth_up=0.50, tongue=0.25),
    "EE":   dict(open=0.26, upper=0.06, width=1.10, teeth_up=0.70, teeth_lo=0.45),
    "OH":   dict(open=0.62, round=0.55, width=0.95, teeth_up=0.25, tongue=0.30),
    "OO":   dict(open=0.32, round=0.95, width=0.95, teeth_up=0.0),
    "FV":   dict(open=0.16, width=0.98, fv=1.0, teeth_up=0.95),
    "L":    dict(open=0.48, width=0.95, teeth_up=0.40, tongue=0.75, tongue_up=1.0),
    "DTN":  dict(open=0.24, width=0.96, teeth_up=0.55, tongue=0.45, tongue_up=0.6),
    "SZ":   dict(open=0.17, width=1.04, teeth_up=0.80, teeth_lo=0.70),
    "SHCH": dict(open=0.26, round=0.50, width=0.95, teeth_up=0.70, teeth_lo=0.55),
    "KG":   dict(open=0.42, width=0.93, teeth_up=0.35, tongue=0.50),
}
KEYS = ["open", "upper", "width", "round", "teeth_up", "teeth_lo", "tongue", "tongue_up", "press", "fv", "corner",
        "skew", "shift", "tongue_out", "tongue_side"]
DEFAULT = dict(open=0.0, upper=0.0, width=1.0, round=0.0, teeth_up=0.0, teeth_lo=0.0, tongue=0.0, tongue_up=0.0,
               press=0.0, fv=0.0, corner=0.0, skew=0.0, shift=0.0,
               tongue_out=0.0, tongue_side=0.0)


def vis(name):
    d = dict(DEFAULT)
    d.update(VISEMES[name])
    return d


def mix(a: dict, b: dict, t: float) -> dict:
    return {k: a.get(k, DEFAULT[k]) + (b.get(k, DEFAULT[k]) - a.get(k, DEFAULT[k])) * t for k in KEYS}


# ---------------------------------------------------------------------------------------------- Persian G2P
from .g2p_fa import VOWELS, g2p  # noqa: E402  (universal rule-based Persian G2P + lexicon)

P2V = {"b": "MBP", "p": "MBP", "m": "MBP", "f": "FV", "v": "FV", "t": "DTN", "d": "DTN", "n": "DTN", "r": "DTN",
       "s": "SZ", "z": "SZ", "sh": "SHCH", "ch": "SHCH", "j": "SHCH", "zh": "SHCH", "k": "KG", "g": "KG", "q": "KG",
       "x": "KG", "h": "KG", "'": "KG", "l": "L", "y": "EE", "w": "OO",
       "a": "AH", "A": "AA", "e": "EH", "i": "EE", "o": "OH", "u": "OO", "ow": "OH", "ey": "EH"}
# relative durations: long vowels > short vowels > consonants; labials keep a visible closure
DUR = {"A": 1.9, "i": 1.6, "u": 1.7, "a": 1.4, "e": 1.25, "o": 1.35, "ow": 1.8, "ey": 1.7,
       "b": 1.05, "p": 1.05, "m": 1.1, "f": 1.0, "v": 0.95, "h": 0.6, "'": 0.5, "y": 0.7, "w": 0.7}
MIN_LABIAL = 0.055          # s — a bilabial closure must be on screen ≥ 1 drawing on twos (≈ 1/12 s with lead)


def phonemes(words):
    """[(text, t0, t1)] → [(text, t0, t1, [phonemes])] (merges a word-final -e with the next ezafe if written)."""
    return [(w, t0, t1, g2p(w)) for w, t0, t1 in words]


def viseme_track(words: list) -> list:
    """words = [(text, start, end)] → [(t0, t1, viseme)].
    Geminates (tashdid) are one longer segment; labial closures are guaranteed a minimum hold."""
    track = []
    for text, t0, t1, ph in phonemes(words):
        if not ph:
            continue
        segs = []                                           # merge identical neighbours (gemination)
        for p in ph:
            if segs and segs[-1][0] == p and p not in VOWELS:
                segs[-1][1] += 0.8
            else:
                segs.append([p, DUR.get(p, 1.0 if p not in VOWELS else 1.4)])
        tot = sum(d for _, d in segs)
        span = t1 - t0
        durs = [span * d / tot for _, d in segs]
        lab = [i for i, (p, _) in enumerate(segs) if P2V.get(p) == "MBP"]
        for i in lab:                                       # steal time from neighbours for a readable closure
            need = MIN_LABIAL - durs[i]
            if need > 0:
                durs[i] += need
                others = [j for j in range(len(durs)) if j != i]
                for j in others:
                    durs[j] = max(0.01, durs[j] - need / max(1, len(others)))
        t = t0
        for (p, _), d in zip(segs, durs):
            track.append((t, t + d, P2V.get(p, "EH")))
            t += d
    return track


def mouth_at(track: list, t: float, lead: float = 0.04) -> dict:
    """co-articulated mouth params at time t (shape anticipates the sound by `lead` s)."""
    t = t + lead
    for i, (a, b, v) in enumerate(track):
        if a <= t < b:
            cur = vis(v)
            # blend into the next viseme over the last 35 % of the segment
            k = (t - a) / max(1e-6, b - a)
            if i + 1 < len(track) and track[i + 1][0] - b < 0.08 and k > 0.65:
                return mix(cur, vis(track[i + 1][2]), (k - 0.65) / 0.35 * 0.5)
            return cur
        if t < a:
            # gap before the next word: ease from rest
            gap = a - t
            return mix(vis("X"), vis(v), max(0.0, 1 - gap / 0.10) * 0.6)
    return vis("X")
