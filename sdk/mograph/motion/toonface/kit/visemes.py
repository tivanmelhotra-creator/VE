"""
toonface.kit.visemes — 14 hand-tuned mouth key-shapes (one topology) + Persian G2P → timed viseme track.

Params: open, upper (upper-lip lift), width, round, teeth_up, teeth_lo, tongue, tongue_up, press, fv, corner, skew
"""
from __future__ import annotations

import re

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
        "skew", "shift"]
DEFAULT = dict(open=0.0, upper=0.0, width=1.0, round=0.0, teeth_up=0.0, teeth_lo=0.0, tongue=0.0, tongue_up=0.0,
               press=0.0, fv=0.0, corner=0.0, skew=0.0, shift=0.0)


def vis(name):
    d = dict(DEFAULT)
    d.update(VISEMES[name])
    return d


def mix(a: dict, b: dict, t: float) -> dict:
    return {k: a.get(k, DEFAULT[k]) + (b.get(k, DEFAULT[k]) - a.get(k, DEFAULT[k])) * t for k in KEYS}


# ---------------------------------------------------------------------------------------------- Persian G2P
# phoneme symbols: consonants b p m f v t d n s z sh ch j k g q x h r l y ' ; vowels a A e i o u
LEXICON = {
    "ببین": "b e b i n", "من": "m a n", "که": "k e", "بهت": "b e h e t", "گفتم": "g o f t a m",
    "دیگه": "d i g e", "اصلا": "A s l a n", "اصلاً": "A s l a n", "حوصله": "h o w s e l e", "حوصله‌ی": "h o w s e l e y e",
    "این": "i n", "بازیا": "b A z i y A", "بازی‌ها": "b A z i h A", "رو": "r o", "ندارم": "n a d A r a m",
}
_LET = {"ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "x", "د": "d", "ذ": "z", "ر": "r",
        "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "'", "غ": "q",
        "ف": "f", "ق": "q", "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ه": "h", "ء": "'"}


def g2p(word: str) -> list:
    w = re.sub(r"[\u064B-\u0652،.!؟?…\"«»]", "", word).strip()
    if w in LEXICON:
        return LEXICON[w].split()
    out = []
    for i, ch in enumerate(w):
        if ch in "اآ":
            out.append("A")
        elif ch == "و":
            out.append("v" if i == 0 else "u")
        elif ch in "یي":
            out.append("y" if i == 0 else "i")
        elif ch == "ه" and i == len(w) - 1 and i > 0:
            out.append("e")
        elif ch in _LET:
            if out and out[-1] not in "aAeiou" and _LET[ch] not in "'":
                out.append("a")                      # unwritten short vowel between consonants
            out.append(_LET[ch])
    return out


P2V = {"b": "MBP", "p": "MBP", "m": "MBP", "f": "FV", "v": "FV", "t": "DTN", "d": "DTN", "n": "DTN", "r": "DTN",
       "s": "SZ", "z": "SZ", "sh": "SHCH", "ch": "SHCH", "j": "SHCH", "zh": "SHCH", "k": "KG", "g": "KG", "q": "KG",
       "x": "KG", "h": "KG", "'": "KG", "l": "L", "y": "EE", "w": "OO",
       "a": "AH", "A": "AA", "e": "EH", "i": "EE", "o": "OH", "u": "OO"}


def viseme_track(words: list) -> list:
    """words = [(text, start, end)] → [(t0, t1, viseme)] with vowels weighted 1.6× consonants."""
    track = []
    for text, t0, t1 in words:
        ph = g2p(text)
        if not ph:
            continue
        wts = [1.6 if p in "aAeiou" else 1.0 for p in ph]
        tot = sum(wts)
        t = t0
        for p, wgt in zip(ph, wts):
            d = (t1 - t0) * wgt / tot
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
