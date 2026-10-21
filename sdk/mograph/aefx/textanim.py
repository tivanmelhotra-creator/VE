"""
mograph.aefx.textanim — After-Effects TEXT ANIMATORS (range + wiggly selectors) for our own glyph engine.

Faithful re-implementation of the AE selector maths (same formulas as Lottie/bodymovin `TextSelectorProp`, which
matches AE): shapes square / ramp_up / ramp_down / triangle / round / smooth, Ease High / Ease Low, smoothness,
units percent / index, based on characters / words / lines, randomize order, selector modes and the wiggly selector.
Works on RTL Persian: "reading order" is the glyph `order` produced by v14.glyphs (right → left).

    from mograph.aefx import textanim as TA
    anim = TA.Animator("atp_23", dur=0.9)                  # any of the 351 presets (text_presets.json) or a dict
    st = anim.frame(block, t)                              # per-glyph deltas for the whole block at local time t
    gs = st.apply(g, gs)                                   # add them to a v14 glyphs.GS

Spec (v14 kinetic_title / any glyph layer):
    "preset": "atp_23"            entrance (local time from the layer "t")       "preset_dur": 0.9  (default 1.0 s)
    "preset_out": "atp_100"       exit = that preset played BACKWARDS, ending at "out"   "preset_out_dur": 0.6
    "preset_px": 100              AE px reference size (px values in presets are relative to a 100 px font)
    "animators": [...]            your own animators in the same JSON format (see tools/import_text_presets.py)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .keyframes import evaluate, scale_time

_LIB = None
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


def library():
    global _LIB
    if _LIB is None:
        with open(os.path.join(_DIR, "text_presets.json"), encoding="utf-8") as f:
            _LIB = json.load(f)
    return _LIB


def preset_names(cat=None, sub=None):
    return [k for k, v in library().items() if (cat is None or v["cat"] == cat) and (sub is None or v["sub"] == sub)]


def get(name):
    if isinstance(name, dict):
        return name
    lib = library()
    k = str(name).lower().replace(" ", "_")
    if k in lib:
        return lib[k]
    if k.isdigit() and f"atp_{k}" in lib:
        return lib[f"atp_{k}"]
    by_name = [v for v in lib.values() if v["name"].lower() == str(name).lower()]
    if by_name:
        return by_name[0]
    raise KeyError(f"unknown text preset '{name}'  (python3 -m mograph.aefx list text)")


# ------------------------------------------------------------------------------------------ bezier easer
def _easer(ease_hi, ease_lo):
    """AE 'Ease High' / 'Ease Low' → cubic easing (Lottie convention: ne = low, xe = high)."""
    x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
    ne, xe = float(ease_lo) / 100.0, float(ease_hi) / 100.0
    if ne > 0:
        x1 = ne
    else:
        y1 = -ne
    if xe > 0:
        x2 = 1.0 - xe
    else:
        y2 = 1.0 + xe
    if x1 == 0 and y1 == 0 and x2 == 1 and y2 == 1:
        return lambda m: m

    def f(m):
        m = np.asarray(m, np.float64)
        lo, hi = np.zeros_like(m), np.ones_like(m)
        for _ in range(24):
            u = (lo + hi) * 0.5
            xu = 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u * u * x2 + u ** 3
            lo = np.where(xu < m, u, lo)
            hi = np.where(xu < m, hi, u)
        u = (lo + hi) * 0.5
        return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u * u * y2 + u ** 3
    return f


def range_weights(sel, n, t):
    """AE range selector: weights (len n, may be negative via amount) for units 0..n-1 at time t."""
    ev = lambda k, d: evaluate(sel.get(k, d), t, d)  # noqa: E731
    start, end, off = float(ev("start", 0.0)), float(ev("end", 100.0)), float(ev("offset", 0.0))
    if sel.get("units", "pct") == "index":
        s, e = start + off, end + off
    else:
        s, e = (start + off) * n / 100.0, (end + off) * n / 100.0
    if s > e:
        s, e = e, s
    ind = np.arange(n, dtype=np.float64)
    if sel.get("random"):
        rng = np.random.default_rng(int(ev("seed", 0.0)) + 7919)
        ind = rng.permutation(n).astype(np.float64)
    shape = sel.get("shape", "square")
    easer = _easer(ev("ease_hi", 0.0), ev("ease_lo", 0.0))
    if shape in ("ramp_up", "ramp_down", "triangle"):
        if e == s:
            m = (ind >= e).astype(np.float64)
        else:
            m = np.clip(0.5 / (e - s) + (ind - s) / (e - s), 0, 1)
        if shape == "triangle":
            m = np.where(m < 0.5, m * 2, 1 - 2 * (m - 0.5)) if e != s else np.zeros(n)
        m = easer(m)
        if shape == "ramp_down":
            m = 1 - m
    elif shape == "round":
        if e == s:
            m = np.zeros(n)
        else:
            tot = e - s
            i2 = np.clip(ind + 0.5 - s, 0, tot)
            x = -tot / 2 + i2
            a = tot / 2
            m = easer(np.sqrt(np.clip(1 - (x * x) / (a * a), 0, 1)))
    elif shape == "smooth":
        if e == s:
            m = np.zeros(n)
        else:
            i2 = np.clip(ind + 0.5 - s, 0, e - s)
            m = easer((1 + np.cos(math.pi + 2 * math.pi * i2 / (e - s))) / 2)
    else:                                                          # square (default)
        m = np.where(ind >= math.floor(s),
                     np.where(ind - s < 0, np.clip(np.minimum(e, 1) - (s - ind), 0, 1), np.clip(e - ind, 0, 1)), 0.0)
        m = easer(m)
        sm = float(ev("smooth", 100.0))
        if sm < 100:
            k = max(1e-8, sm / 100.0)
            th = 0.5 - k * 0.5
            m = np.where(m < th, 0.0, np.clip((m - th) / k, 0, 1))
    return m * float(ev("amount", 100.0)) / 100.0


def _noise1(x):
    """smooth value noise in [-1, 1]."""
    i = np.floor(x)
    f = x - i
    h = lambda q: (np.sin(q * 127.1 + 311.7) * 43758.5453) % 1.0 * 2 - 1  # noqa: E731
    u = f * f * (3 - 2 * f)
    return h(i) * (1 - u) + h(i + 1) * u


def wiggly_weights(sel, n, t):
    ev = lambda k, d: float(evaluate(sel.get(k, d), t, d))  # noqa: E731
    lo, hi = ev("min", -100.0) / 100.0, ev("max", 100.0) / 100.0
    freq, corr = ev("freq", 2.0), ev("corr", 50.0) / 100.0
    seed = ev("seed", 0.0)
    tp, sp = ev("tphase", 0.0) / 360.0, ev("sphase", 0.0) / 360.0
    ind = np.arange(n, dtype=np.float64)
    x = t * freq + tp + seed * 13.37 + ind * (1 - corr) * 3.1 + sp
    v = (_noise1(x) + 0.5 * _noise1(x * 2.03 + 17.0)) / 1.5            # [-1, 1]
    return lo + (v * 0.5 + 0.5) * (hi - lo)


def combine(prev, m, mode):
    if prev is None:
        return m if mode != "subtract" else 1 - m
    if mode == "subtract":
        return prev - m
    if mode == "intersect":
        return prev * m
    if mode == "min":
        return np.minimum(prev, m)
    if mode == "max":
        return np.maximum(prev, m)
    if mode == "difference":
        return np.abs(prev - m)
    return prev + m


# ------------------------------------------------------------------------------------------- frame state
class FrameState:
    """Per-glyph deltas for one block at one instant (arrays indexed by glyph.order)."""

    def __init__(self, n):
        self.n = n
        z = lambda: np.zeros(n)  # noqa: E731
        self.tx, self.ty, self.tz = z(), z(), z()
        self.rx, self.ry, self.rz = z(), z(), z()
        self.sx, self.sy = np.ones(n), np.ones(n)
        self.alpha, self.fill = np.ones(n), np.ones(n)
        self.shear, self.blur, self.track, self.scramble = z(), z(), z(), z()
        self.ax, self.ay = z(), z()

    def apply(self, g, gs):
        i = min(self.n - 1, g.order)
        sx, sy, rz = self.sx[i], self.sy[i], self.rz[i]
        gs.sx *= sx
        gs.sy *= sy
        gs.rx += self.rx[i]
        gs.ry += self.ry[i]
        gs.rz += rz
        gs.shear += self.shear[i]
        ax, ay = self.ax[i], self.ay[i]
        if ax or ay:                                  # pivot around the AE anchor: offset = a - R·S·a
            c, s = math.cos(math.radians(rz)), math.sin(math.radians(rz))
            qx, qy = ax * sx, ay * sy
            gs.tx += ax - (c * qx - s * qy)
            gs.ty += ay - (s * qx + c * qy)
        gs.tx += self.tx[i] + self.track[i]
        gs.ty += self.ty[i]
        gs.tz += self.tz[i]
        gs.alpha *= max(0.0, min(1.0, self.alpha[i]))
        gs.fill_alpha *= max(0.0, min(1.0, self.fill[i]))
        if self.blur[i] > 0.4:
            try:
                gs.blur = (gs.blur or 0.0) + float(self.blur[i])
            except AttributeError:                    # older GS without a blur slot → fade instead
                gs.alpha *= 1.0 / (1.0 + self.blur[i] / 12.0)
        if self.scramble[i] > 0.5:
            gs.alpha *= 0.35 + 0.65 * ((hash((g.order, int(self.scramble[i] * 7))) % 100) / 100.0)
        return gs


NON_JOIN_NEXT = set("اآأإٱدذرزژوؤةۀء ‌")      # letters that never connect to the FOLLOWING letter (+ ZWNJ / space)


def _pieces(gl, block=None):
    """cursive CONNECTED pieces: joined Persian/Arabic letters share a unit, a non-joining letter (ا د ذ ر ز ژ و …)
    ends its piece — so 'characters' presets keep the script legible instead of tearing joined letters apart.
    Uses the real joining rules when the word text maps 1:1 to glyphs, else outline overlap."""
    words = {}
    if block is not None:
        for w in getattr(block, "words", []) or []:
            if isinstance(w, dict) and "i" in w:
                words[w["i"]] = w.get("text", "")
    out, cur, prev, k_in_word = [], -1, None, 0
    for g in gl:
        new_word = prev is None or g.word != prev.word or g.line != prev.line
        if new_word:
            cur += 1
            k_in_word = 0
        else:
            txt = "".join(ch for ch in words.get(g.word, "") if not ("\u064b" <= ch <= "\u065f"))
            nglyph = sum(1 for q in gl if q.word == g.word and q.line == g.line)
            if txt and len(txt) == nglyph:
                if txt[k_in_word - 1] in NON_JOIN_NEXT:
                    cur += 1
            else:
                gap = max(prev.x0, g.x0) - min(prev.x1, g.x1)
                if gap > 0.015 * max(1.0, g.size):
                    cur += 1
        out.append(cur)
        prev = g
        k_in_word += 1
    return out


def _is_cursive(block):
    gl = block.glyphs
    if len(gl) < 2 or not hasattr(gl[0], "x0"):
        return False
    try:
        return any(any(0x0600 <= ord(ch) <= 0x06FF or 0xFB50 <= ord(ch) <= 0xFEFF for ch in w.get("text", ""))
                   for w in getattr(block, "words", []) if isinstance(w, dict))
    except Exception:  # noqa: BLE001
        return False


def _units(block, based, cursive=None):
    """map glyph.order → unit index for chars / pieces / words / lines, plus unit count.
    cursive=None → auto: AE 'characters' become connected 'pieces' on Persian/Arabic text."""
    gl = sorted(block.glyphs, key=lambda g: g.order)
    if based in ("chars", "chars_ns") and (cursive if cursive is not None else _is_cursive(block)):
        based = "pieces"
    if based == "pieces":
        idx = _pieces(gl, block)
    elif based == "words":
        idx = [g.word for g in gl]
    elif based == "lines":
        idx = [g.line for g in gl]
    else:
        idx = [g.order for g in gl]
    u = sorted(set(idx))
    remap = {v: k for k, v in enumerate(u)}
    return np.array([remap[v] for v in idx]), len(u), gl


class Animator:
    """A text preset (or list of animators) evaluated on a glyph block."""

    def __init__(self, preset, dur=None, px=100.0, reverse=False):
        p = get(preset) if not isinstance(preset, list) else {"animators": preset, "dur": 1.0}
        self.base_dur = float(p.get("dur") or 1.0) or 1.0
        self.dur = float(dur) if dur else self.base_dur
        k = self.dur / self.base_dur if self.base_dur > 0 else 1.0
        self.anims = [{"sel": [{kk: scale_time(vv, k) for kk, vv in s.items()} for s in a["sel"]],
                       "props": {kk: scale_time(vv, k) for kk, vv in a["props"].items()}} for a in p["animators"]]
        self.px, self.reverse = float(px), reverse
        self._cache = (None, None)

    def frame(self, block, t, size=None):
        """FrameState at local time t (seconds since the animation starts; reverse → played backwards)."""
        key = (id(block), round(t, 5), size)
        if self._cache[0] == key:
            return self._cache[1]
        tt = (self.dur - t) if self.reverse else t
        tt = min(max(tt, 0.0), self.dur + 1e-6)
        gl = sorted(block.glyphs, key=lambda g: g.order)
        n = len(gl)
        st = FrameState(max(1, n))
        if n == 0:
            return st
        k = (size or (gl[0].size if gl else self.px)) / self.px
        for a in self.anims:
            w = None
            based = "chars"
            for s in a["sel"]:
                based = s.get("based", based) or "chars"
                ui, nu, _ = _units(block, based)
                m = wiggly_weights(s, nu, tt) if s["type"] == "wiggly" else range_weights(s, nu, tt)
                w = combine(w, m[ui], s.get("mode", "add"))
            if w is None:
                w = np.ones(n)
            w = np.clip(w, -1.0, 1.0)
            P = a["props"]
            ev = lambda nm: evaluate(P[nm], tt) if nm in P else None  # noqa: E731
            v = ev("position")
            if v is not None:
                st.tx += w * v[0] * k
                st.ty += w * v[1] * k
                st.tz += w * (v[2] if len(v) > 2 else 0.0) * k
            v = ev("anchor")
            if v is not None:
                st.ax += w * v[0] * k
                st.ay += w * v[1] * k
            v = ev("scale")
            if v is not None:
                v = v if isinstance(v, list) else [v, v, 100]
                st.sx *= 1 + w * (v[0] - 100) / 100.0
                st.sy *= 1 + w * (v[1] - 100) / 100.0
            for nm, arr in (("rotation", st.rz), ("rx", st.rx), ("ry", st.ry)):
                v = ev(nm)
                if v is not None:
                    arr += w * float(v)
            v = ev("opacity")
            if v is not None:
                st.alpha *= 1 + w * (float(v) - 100) / 100.0
            v = ev("fill_opacity")
            if v is not None:
                st.fill *= 1 + w * (float(v) - 100) / 100.0
            v = ev("blur")
            if v is not None:
                b = v if isinstance(v, list) else [v, v]
                st.blur += np.abs(w) * max(b) * k
            v = ev("skew")
            if v is not None:
                st.shear += w * math.tan(math.radians(max(-80, min(80, float(v)))))
            v = ev("tracking")
            if v is not None:                          # AE tracking: cumulative space AFTER each selected char
                sp = w * float(v) * k * 0.1            # AE tracking units = 1/1000 em → px at font size
                cum = np.concatenate([[0.0], np.cumsum(sp)[:-1]])
                cum -= cum.mean() if n > 1 else 0      # centre-aligned text grows both ways
                st.track += -cum                       # RTL reading order → advancing = −x
            for nm in ("char_offset", "char_replace"):
                v = ev(nm)
                if v is not None:
                    st.scramble += np.abs(w) * (1 + abs(float(v)) * (tt * 30 % 7))
        self._cache = (key, st)
        return st

    def motion(self, block, t_fn):
        """→ fn(g, gs) for v14.glyphs.draw_block-style motion callbacks (t_fn() = local time)."""
        def m(g, gs):
            return self.frame(block, t_fn()).apply(g, gs)
        return m
