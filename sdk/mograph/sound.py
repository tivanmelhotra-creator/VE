"""
mograph.sound — EVENT-LOCKED sound design (v7).

Old engine: one fixed sample at the element's start time → sounds felt random (a counter "rolled"
1.4 s before the digits moved; a 1.1 s drawing got a 0.5 s scribble).
New engine: every element type has a *score* that reads the SAME timing maths the renderer uses and
returns events exactly where the eye sees something happen:
  * pen strokes last as long as the stroke and follow its speed (ease derivative)
  * counters tick on every frame the SHOWN number changes, pitch rises, lock click when it settles
  * stamps hit at the CONTACT frame (spring reaches 1) with a short air pre-roll
  * grids tick per item at their stagger time; bars/rings sweep with their growth
  * transitions whoosh with the camera (peak at max velocity)
  * kinetic words hit on their word; emphasised words hit harder
Stereo pan by x; small shared room; soft-clip bus.

Palettes: meta.sound = "paper" (line-art) | "ui" (swiss) | "punchy" (kinetic); default from meta.style.
Per element: "sfx": "none" | "auto" | <kind>; "sfx_gain"; "cues": [[t, kind, gain, dur]].
CLI:  python -m mograph cues spec.json   → readable score (time, kind, why) — check sync without listening
"""
from __future__ import annotations

import math

import numpy as np

from .core import clamp, get_ease, prog, spring_preset

SR = 44100
KINDS = ("pen", "tick", "click", "type", "pop", "blip", "whoosh", "swipe", "impact", "hit", "soft", "rise",
         "scratch", "lock", "hum", "rumble", "drop", "jelly")


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------
def _t(d):
    return np.arange(max(1, int(d * SR))) / SR


def _env(n, a=0.003, decay=10.0):
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(a, 1e-4)) * np.exp(-decay * t)


def _lp(x, k):
    k = max(1, int(k))
    return np.convolve(x, np.ones(k) / k, "same")


def _bp(x, lo_k, hi_k):
    return _lp(x, lo_k) - _lp(x, hi_k)


def _noise(n, seed):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def _curve(curve, n):
    if curve is None or len(curve) < 2:
        return np.sin(np.pi * np.linspace(0, 1, n)) ** 0.8
    c = np.interp(np.linspace(0, len(curve) - 1, n), np.arange(len(curve)), np.asarray(curve, float))
    return c / (c.max() + 1e-9)


def _edges(x, ms=12):
    f = min(len(x) // 6, int(ms / 1000 * SR))
    if f > 1:
        x[:f] *= np.linspace(0, 1, f); x[-f:] *= np.linspace(1, 0, f)
    return x


def syn(kind, dur=0.3, seed=0, curve=None, pitch=1.0, palette="paper"):
    d = max(0.02, dur)
    n = int(d * SR)
    if kind == "pen":            # graphite on paper: band noise gated by stroke speed + grain
        x = _bp(_noise(n, seed), 2, 9) * 0.9 + _bp(_noise(n, seed + 1), 1, 3) * 0.25
        grain = 0.75 + 0.25 * np.sign(np.sin(2 * np.pi * (38 + 9 * (seed % 5)) * _t(d)))
        return _edges(x * _curve(curve, n) * grain * 0.32).astype(np.float32)
    if kind in ("tick", "click"):
        m = int(0.035 * SR); tt = _t(0.035)
        f = (3400 if (palette == "ui" or kind == "click") else 2600) * pitch
        s = np.sin(2 * np.pi * f * tt) * _env(m, 0.0004, 140) * 0.30
        s += _bp(_noise(m, seed), 1, 4) * _env(m, 0.0002, 260) * (0.18 if kind == "click" else 0.10)
        return s.astype(np.float32)
    if kind == "type":           # keyboard / typewriter key
        m = int(0.06 * SR); tt = _t(0.06)
        body = np.sin(2 * np.pi * (520 + 140 * (seed % 4)) * pitch * tt) * _env(m, 0.0005, 70) * 0.18
        clack = _bp(_noise(m, seed), 1, 3) * _env(m, 0.0002, 180) * 0.30
        return (body + clack).astype(np.float32)
    if kind == "pop":
        m = int(0.16 * SR); tt = _t(0.16)
        return (np.sin(2 * np.pi * (480 + 820 * np.exp(-tt * 38)) * pitch * tt) * _env(m, 0.001, 24) * 0.34
                ).astype(np.float32)
    if kind == "blip":
        m = int(0.09 * SR); tt = _t(0.09)
        return (np.sin(2 * np.pi * 880 * pitch * tt) * _env(m, 0.002, 38) * 0.22).astype(np.float32)
    if kind == "whoosh":         # peak follows the curve (camera / morph velocity)
        env = _curve(curve, n) ** 1.6
        x = _bp(_noise(n, seed + 7), 3, 14) * 0.55 + _bp(_noise(n, seed + 8), 1, 3) * env * 0.35
        return _edges(x * env * 0.55).astype(np.float32)
    if kind == "swipe":          # tight filtered wipe (swiss)
        return _edges(_bp(_noise(n, seed + 3), 2, 6) * _curve(curve, n) ** 2.2 * 0.5).astype(np.float32)
    if kind == "impact":
        m = int(0.7 * SR); tt = _t(0.7)
        sub = np.sin(2 * np.pi * (62 - 24 * tt) * tt) * _env(m, 0.0015, 7.5) * 0.95
        crack = _bp(_noise(m, seed), 1, 4) * _env(m, 0.0003, 55) * 0.55
        paper = _bp(_noise(m, seed + 2), 2, 8) * _env(m, 0.001, 18) * 0.25
        return (sub + crack + paper).astype(np.float32)
    if kind == "hit":            # kinetic word hit
        m = int(0.32 * SR); tt = _t(0.32)
        sub = np.sin(2 * np.pi * (95 * pitch - 40 * tt) * tt) * _env(m, 0.001, 14) * 0.7
        snap = _bp(_noise(m, seed), 1, 3) * _env(m, 0.0002, 90) * 0.45
        return (sub + snap).astype(np.float32)
    if kind == "soft":
        x = _bp(_noise(n, seed + 11), 4, 20)
        return (x * np.sin(np.pi * np.linspace(0, 1, n)) ** 2 * 0.12).astype(np.float32)
    if kind == "rise":           # growth sweep — pitch follows the growth
        env = _curve(curve, n)
        pr = np.cumsum(env); pr /= pr[-1] + 1e-9
        ph = 2 * np.pi * np.cumsum((180 + 520 * pr) * pitch) / SR
        return _edges((np.sin(ph) + 0.3 * np.sin(2 * ph)) * (0.25 + 0.75 * env) * 0.10, 30).astype(np.float32)
    if kind == "scratch":        # marker strike
        tt = _t(d)
        sq = np.sin(2 * np.pi * (1400 + 300 * np.sin(2 * np.pi * 7 * tt)) * tt) * 0.08
        return _edges((_bp(_noise(n, seed + 3), 1, 3) * 0.3 + sq) * _curve(curve, n)).astype(np.float32)
    if kind == "lock":
        m = int(0.22 * SR); s = np.zeros(m, np.float32); k = int(0.1 * SR); a = _t(0.1)
        for d0, f in ((0.0, 1500), (0.07, 980)):
            i0 = int(d0 * SR)
            seg = (np.sin(2 * np.pi * f * a) + 0.5 * np.sin(2 * np.pi * f * 2.7 * a)) * _env(k, 0.0004, 55) * 0.26
            s[i0:i0 + k] += seg[: m - i0]
        return s
    if kind == "hum":
        tt = _t(d)
        s = (np.sin(2 * np.pi * 220 * tt) * 0.5 + np.sin(2 * np.pi * 331 * tt) * 0.3
             + _bp(_noise(n, seed), 1, 2) * 0.15) * 0.12
        return _edges(s, 60).astype(np.float32)
    if kind == "rumble":
        tt = _t(d)
        s = (_lp(_noise(n, seed), 40) * 3 * (0.6 + 0.4 * np.sin(2 * np.pi * 24 * tt))
             + np.sin(2 * np.pi * 48 * tt) * 0.4) * _curve(curve, n) * 0.35
        return _edges(s).astype(np.float32)
    if kind == "drop":
        tt = _t(d)
        f = 900 - 500 * tt / d
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * tt / d) ** 2 * 0.06).astype(np.float32)
    if kind == "jelly":          # v13.9 @novadent elastic jelly squash-stretch sound
        m = int(0.22 * SR); tt = _t(0.22)
        # Frequency-modulated elastic wobble (squish + harmonic bubble overtone)
        f_mod = (280 + 340 * np.sin(2 * np.pi * 11.5 * tt) * np.exp(-tt * 14.0)) * pitch
        ph = 2 * np.pi * np.cumsum(f_mod) / SR
        wobble = (np.sin(ph) * 0.65 + np.sin(2.0 * ph + 0.4) * 0.28) * _env(m, 0.004, 16.0)
        water = _bp(_noise(m, seed + 19), 3, 12) * _env(m, 0.002, 28.0) * 0.14
        return _edges((wobble + water) * 0.36, 8).astype(np.float32)
    return np.zeros(1, np.float32)


# ---------------------------------------------------------------------------
# timing helpers (read the renderer's maths)
# ---------------------------------------------------------------------------
def speed_curve(ease_name, n=48):
    e = get_ease(ease_name)
    p = np.array([e(i / n) for i in range(n + 1)])
    v = np.abs(np.diff(p))
    return v / (v.max() + 1e-9)


def first_time(f, t0, t1, step=1 / 240):
    t = t0
    while t <= t1:
        if f(t):
            return t
        t += step
    return None


class Ev:
    __slots__ = ("t", "kind", "gain", "dur", "curve", "pitch", "x", "why", "seed", "pal")

    def __init__(self, t, kind, gain=1.0, dur=0.3, curve=None, pitch=1.0, x=None, why="", pal=None):
        self.t, self.kind, self.gain, self.dur, self.curve = float(t), kind, float(gain), float(dur), curve
        self.pitch, self.x, self.why, self.seed = pitch, x, why, 0
        self.pal = pal


SCORES = {}


def score(*types):
    def deco(fn):
        for t in types:
            SCORES[t] = fn
        return fn
    return deco


def _dur(el, default):
    from .elements import REGISTRY
    return float(el.get("dur", REGISTRY[el["type"]]["dur"] if el["type"] in REGISTRY else default))


def _ease_of(el, enter):
    return el.get("ease", "smooth" if enter in ("pop", "rise", "fade") else "in_out")


def _pen(el, t0, dur, gain=1.0, why="draw", ease=None):
    return Ev(t0, "pen", gain, dur, speed_curve(ease or _ease_of(el, "draw")), x=el.get("x"), why=why)


def _slam(el, t):
    """scale = 1.8 - 0.8*spring → contact when the spring first reaches ~1."""
    tc = first_time(lambda tt: spring_preset(tt, t, "stamp") >= 0.995, t, t + 0.6) or (t + 0.12)
    return [Ev(max(0.0, tc - 0.16), "whoosh", 0.35, 0.18, np.linspace(0, 1, 24) ** 3, x=el.get("x"), why="slam air"),
            Ev(tc, "impact", 1.0, x=el.get("x"), why=f"slam contact (+{tc - t:.2f}s)")]


# ---------------------------------------------------------------------------
# element scores
# ---------------------------------------------------------------------------
@score("icon", "box", "path", "circle", "divider", "line", "timeline", "image", "video")
def _s_generic(el, sc, pal):
    from .elements import REGISTRY
    enter = el.get("enter", REGISTRY[el["type"]]["enter"])
    t, d = float(el["t"]), _dur(el, 1.0)
    size = el.get("size", el.get("w", 200))
    size = float(size) if isinstance(size, (int, float)) else 200.0
    g = clamp(0.45 + size / 500, 0.45, 1.0)
    ev = []
    if enter == "draw":
        ev.append(_pen(el, t, d, g, f"{el['type']} stroke {d:.2f}s"))
    elif enter == "pop":
        ev.append(Ev(t + 0.06, "pop" if pal != "ui" else "blip", 0.8, x=el.get("x"), why="pop"))
    elif enter == "slam":
        ev += _slam(el, t)
    elif enter in ("fade", "rise"):
        ev.append(Ev(t, "soft", 0.6, max(0.3, d), x=el.get("x"), why=enter))
    if el["type"] == "timeline":
        n = len(el.get("items", []))
        e = get_ease(_ease_of(el, "draw"))
        for i in range(n):   # tick appears when trim*(n+1) - i - 0.6 > 0
            ti = first_time(lambda tt: e(prog(tt, t, d)) * (n + 1) - i - 0.6 > 0, t, t + d)
            if ti is not None:
                ev.append(Ev(ti, "tick" if pal != "ui" else "click", 0.7, x=el.get("x"), why=f"timeline item {i}"))
    if el.get("morph_to") and el.get("morph_at") is not None:
        md = float(el.get("morph_dur", 0.9))
        ev.append(Ev(float(el["morph_at"]), "whoosh", 0.6, md, speed_curve(el.get("morph_ease", "in_out_expo")),
                     x=el.get("x"), why="morph"))
    return ev


@score("morph")
def _s_morph(el, sc, pal):
    times = el.get("_chain_t") or [float(el["t"])]
    d = float(el.get("dur", 0.9))
    ev = []
    if el.get("draw_in"):
        di = float(el["draw_in"])
        ev.append(_pen(el, times[0] - di, di * 0.9, 0.8, "morph draw-in", "in_out"))
    chain = el.get("chain") or [el.get("from"), el.get("to")]
    for k, tk in enumerate(times[: max(1, len(chain) - 1)]):
        ev.append(Ev(tk, "whoosh", 0.65, d, speed_curve(el.get("ease", "in_out_expo")), x=el.get("x"),
                     why=f"morph {chain[k]}→{chain[k + 1] if k + 1 < len(chain) else ''}"))
    return ev


@score("text")
def _s_text(el, sc, pal):
    enter = el.get("enter", "rise")
    t, d = float(el["t"]), _dur(el, 0.6)
    size = float(el.get("size", 52))
    if enter == "words":
        n = max(1, len(str(el.get("text", "")).replace("**", "").split()))
        e = get_ease(_ease_of(el, "words"))
        ev = []
        for k in range(n):   # renderer: wp = clamp(rp*(n+1.5) - k) → word half-visible at wp = .5
            tk = first_time(lambda tt: e(prog(tt, t, d)) * (n + 1.5) - k >= 0.5, t, t + d + 0.01)
            if tk is not None:
                kind = {"paper": "type", "ui": "click"}.get(pal, "hit")
                ev.append(Ev(tk, kind, 0.5, x=el.get("x"), pitch=1 + 0.03 * k, why=f"word {k + 1}/{n}"))
        return ev
    if enter == "slam":
        return _slam(el, t)
    if enter == "wipe":
        return [Ev(t, "pen" if pal == "paper" else "swipe", 0.45, d, speed_curve(_ease_of(el, "wipe")),
                   x=el.get("x"), why="text wipe")]
    if size >= 70:   # titles get a soft air; body text stays silent (no clutter)
        return [Ev(t, "soft", 0.9, max(0.35, d), x=el.get("x"), why="title rise")]
    return []


@score("callout")
def _s_callout(el, sc, pal):
    t, d = float(el["t"]), _dur(el, 0.8)
    e = get_ease(_ease_of(el, "draw"))
    tline = first_time(lambda tt: e(prog(tt, t, d)) * 1.25 - 0.1 > 0, t, t + d) or t
    tlab = first_time(lambda tt: (prog(tt, t, d) - 0.5) * 2.5 > 0.3, t, t + d) or t + d * 0.6
    return [Ev(t, "tick", 0.55, why="anchor dot"),
            Ev(tline, "pen", 0.45, max(0.2, d * 0.75), speed_curve("in_out"), why="leader line"),
            Ev(tlab, "type" if pal == "paper" else "click", 0.4, why="label")]


@score("arrow")
def _s_arrow(el, sc, pal):
    t, d = float(el["t"]), _dur(el, 0.6)
    return [Ev(t, "pen" if pal == "paper" else "swipe", 0.55, d, speed_curve(_ease_of(el, "draw")), why="arrow")]


@score("stamp")
def _s_stamp(el, sc, pal):
    return _slam(el, float(el["t"]))


@score("pill")
def _s_pill(el, sc, pal):
    return [Ev(float(el["t"]) + 0.05, "pop" if pal != "ui" else "blip", 0.7, x=el.get("x"), why="pill pop")]


@score("strike")
def _s_strike(el, sc, pal):
    t = float(el["t"])
    ev = [Ev(t, "scratch", 0.8, 0.55, speed_curve("in_out"), why="strike line")]
    if el.get("true"):
        ev.append(Ev(t + 0.62, "pop", 0.55, why="true word"))
        ev.append(Ev(t + 0.78, "tick", 0.5, why="check"))
    return ev


@score("counter", "ring")
def _s_counter(el, sc, pal):
    """Ticks on the exact frames where the DISPLAYED digits change; lock when the number settles."""
    from .elements import compute_state, value_at
    t0 = float(el["t"])
    t1 = float(el["out"]) if el.get("out") is not None else (sc.duration if sc else t0 + 6)
    fps = sc.fps if sc else 30.0
    is_ring = el["type"] == "ring"
    fmt = el.get("fmt", "{:,.0f}")
    prev, changes = None, []
    for i in range(int(t0 * fps), int(min(t1, t0 + 30) * fps) + 1):
        tt = i / fps
        st = compute_state(el, tt, sc)
        v = value_at(el, st, default_key="value", default_from=0.0) if is_ring else value_at(el, st)
        shown = round(v * 100) if is_ring else fmt.format(v)
        if prev is not None and shown != prev:
            changes.append(tt)
        prev = shown
    ev = []
    if is_ring:
        ev.append(Ev(t0, "rise", 0.7, _dur(el, 1.4), speed_curve(el.get("ease", "out_quart")), why="ring sweep"))
    if not changes:
        return ev
    runs, cur = [], [changes[0]]
    for c in changes[1:]:
        if c - cur[-1] > 0.2:
            runs.append(cur); cur = [c]
        else:
            cur.append(c)
    runs.append(cur)
    last = -1.0
    for r in runs:
        span = max(1e-3, r[-1] - r[0])
        for c in r:
            if c - last < 1 / 20:      # max ~20 ticks/s (a real mechanical counter)
                continue
            last = c
            ev.append(Ev(c, "tick" if pal != "ui" else "click", 0.4, pitch=1 + 0.35 * (c - r[0]) / span,
                         x=el.get("x"), why="digit change"))
        ev.append(Ev(r[-1] + 1 / fps, "lock", 0.85, x=el.get("x"), why=f"settles ({len(r)} changes)"))
    return ev


@score("bars")
def _s_bars(el, sc, pal):
    t = float(el["t"])
    ev = []
    for i in range(len(el.get("values", []))):   # gp = out_quart(prog(lt, .25 + i*.12, .9))
        ts = t + 0.25 + i * 0.12
        ev.append(Ev(ts, "rise", 0.5, 0.9, speed_curve("out_quart"), pitch=1 + 0.08 * i, why=f"bar {i}"))
    return ev


@score("dot_grid")
def _s_dots(el, sc, pal):
    t = float(el["t"])
    rows, cols = int(el.get("rows", 20)), int(el.get("cols", 14))
    ts = sorted(t + math.hypot(c - cols / 2, r - rows / 2) / math.hypot(cols / 2, rows / 2) * 0.7
                for r in range(rows) for c in range(cols))
    ev, last = [], -1.0
    for x in ts:
        if x - last >= 0.05:
            ev.append(Ev(x, "tick", 0.2, pitch=1.2, why="dots")); last = x
    if el.get("highlight_at") is not None:
        ev.append(Ev(float(el["highlight_at"]), "soft", 0.9, 0.6, why="dim others"))
    return ev


@score("icon_grid")
def _s_igrid(el, sc, pal):
    t = float(el["t"]); n = int(el.get("rows", 1)) * int(el.get("cols", 7))
    step = float(el.get("stagger", 0.06))
    ev, last = [], -1.0
    for i in range(n):
        ti = t + i * step + 0.1
        if ti - last >= 0.05:
            ev.append(Ev(ti, "pen", 0.25, 0.5, speed_curve("in_out"), pitch=1 + 0.02 * i, why=f"item {i}")); last = ti
    if el.get("highlight_at") is not None:
        ev.append(Ev(float(el["highlight_at"]), "soft", 0.9, 0.6, why="dim others"))
    if el.get("cross_at") is not None:
        ev.append(Ev(float(el["cross_at"]), "scratch", 0.7, 0.3 + n * 0.012, speed_curve("out"), why="cross all"))
    return ev


# --- widgets (timings mirror widgets.py) ------------------------------------
@score("quote_rig")
def _s_quote(el, sc, pal):
    t = float(el["t"])
    ev = [_pen(el, t, 0.8, 0.55, "card", "in_out")]
    for hl in el.get("highlights", []):
        ht = float(hl["t"])
        ev.append(Ev(ht, "scratch" if hl.get("style") == "underline" else "pen", 0.7, 0.55, speed_curve("in_out"),
                     why=f"highlight {hl.get('word', '')}"))
        if hl.get("num") is not None:
            ev.append(Ev(ht + 0.35, "pop", 0.5, why="badge"))
    return ev


@score("flow_loop")
def _s_flow(el, sc, pal):
    t = float(el["t"])
    ev = [_pen(el, t + i * 0.25, 0.9, 0.5, f"node {i}", "in_out") for i in range(3)]
    ev.append(Ev(t + 0.7, "pen", 0.45, 0.9, speed_curve("in_out"), why="arrows"))
    for f in (el.get("flips") or []):
        if float(f[0]) > t + 0.05:
            ev.append(Ev(float(f[0]), "pop", 0.7, why=f"flip → {f[1]}"))
    return ev


@score("laser_scan")
def _s_scan(el, sc, pal):
    t = float(el["t"])
    ev = [_pen(el, t, _dur(el, 0.9), 0.55, "document")]
    if el.get("scan_at") is not None:
        sa, sd = float(el["scan_at"]), float(el.get("scan_dur", 1.8))
        h = float(el.get("h", 820)); n = int(el.get("lines", 8)); row = int(el.get("clause_row", n // 2))
        top_y = -h / 2 + 170; bot_y = h / 2 - 90; gap = (bot_y - top_y) / max(1, n - 1)
        cy_ = top_y + row * gap
        e = get_ease("in_out")
        tf = first_time(lambda tt: (e(prog(tt, sa, sd)) * (bot_y + 20 - (top_y - 30)) + top_y - 30) >= cy_, sa, sa + sd)
        ev.append(Ev(sa, "hum", 0.9, sd, why="laser"))
        ev.append(Ev(tf or sa + sd * 0.55, "lock", 0.9, why="clause found"))
    return ev


@score("invoice")
def _s_invoice(el, sc, pal):
    t = float(el["t"])
    ev = [_pen(el, t, _dur(el, 0.8), 0.5, "paper")]
    for i, r in enumerate(el.get("rows", [])):
        rt = float(r.get("t", t + 0.6 + i * 0.4))
        ev.append(Ev(rt + 0.04, "type", 0.6, why=f"row {r.get('item', '')}"))
        ev.append(Ev(rt + 0.16, "type", 0.45, pitch=1.15, why="row value"))
    tot = el.get("total")
    if isinstance(tot, dict) and "t" in tot:
        ev.append(Ev(float(tot["t"]) + 0.08, "lock", 0.8, why="total"))
    st = el.get("stamp")
    if isinstance(st, dict) and "t" in st:
        ev += _slam(el, float(st["t"]))
    return ev


@score("strata")
def _s_strata(el, sc, pal):
    t = float(el["t"])
    ev = [_pen(el, t + 0.3, 1.2, 0.5, "ground")]
    if el.get("drill_at") is not None:
        da = float(el["drill_at"])
        ev.append(Ev(da, "rumble", 0.85, 1.4, speed_curve("in_out"), why="drill down"))
        ev.append(Ev(da + 1.5, "rise", 0.6, 1.0, speed_curve("in_out"), why="flow up"))
        ev.append(Ev(da + 2.5, "pop", 0.8, why="question"))
    return ev


@score("scale_rig")
def _s_scale(el, sc, pal):
    t = float(el["t"])
    ev = [_pen(el, t, _dur(el, 1.0), 0.5, "scale")]
    for dr in el.get("drops", []):
        t0 = float(dr["t"])
        ev.append(Ev(t0, "drop", 0.8, 0.55, why=f"drop {dr.get('label', '')}"))
        ev.append(Ev(t0 + 0.55, "lock", 0.95, why="lands"))
    return ev


# ---------------------------------------------------------------------------
# collect + mix
# ---------------------------------------------------------------------------
def palette_of(scene):
    m = scene.spec.get("meta", {})
    if m.get("sound"):
        return m["sound"]
    return {"swiss": "ui", "kinetic": "punchy"}.get(m.get("style", ""), "paper")


def chapter_palette(scene, ch):
    """v8: sound palette of ONE chapter (its style world), meta.sound / element "sound" still override."""
    w = ch.get("_world") or {}
    return w.get("sound") or palette_of(scene)


def collect(scene):
    out = []
    for ci, ch in enumerate(scene.chapters):
        pal = chapter_palette(scene, ch)
        for el in ch["elements"]:
            if el.get("_bad"):
                continue
            mode = el.get("sfx", "auto")
            if mode in (None, False, "none"):
                continue
            fn = SCORES.get(el["type"])
            if mode == "auto":
                evs = fn(el, scene, el.get("sound", pal)) if fn else []
            else:
                evs = [Ev(float(el["t"]), mode, 1.0, _dur(el, 0.4), speed_curve("in_out"), why="manual")]
            for c in el.get("cues", []) or []:
                evs.append(Ev(scene.resolve_time(c[0], ch["t"], ch["end"]), c[1], c[2] if len(c) > 2 else 1.0,
                              c[3] if len(c) > 3 else 0.4, speed_curve("in_out"), why="cue"))
            g = float(el.get("sfx_gain", 1.0))
            for e in evs:
                e.gain *= g
                e.t += float(el.get("sfx_offset", 0.0))
                if not isinstance(e.x, (int, float)):
                    e.x = el.get("x") if isinstance(el.get("x"), (int, float)) else scene.W / 2
                e.why = f"{el['id']}: {e.why}"
                e.pal = e.pal or el.get("sound", pal)
                out.append(e)
        if ci > 0:
            tr = ch["transition"]
            if tr["type"] in ("pan", "thread", "wipe", "slide", "push"):
                kind = "swipe" if (tr["type"] in ("wipe", "slide", "push") or pal == "ui") else "whoosh"
                out.append(Ev(ch["t"], kind, 0.7, float(tr["dur"]), speed_curve("in_out"), x=scene.W / 2,
                              why=f"chapter {ci + 1} {tr['type']}", pal=pal))
            elif tr["type"] == "portal":     # iris opens: rising air, then a soft hit when it covers the frame
                d = float(tr["dur"])
                fx = tr.get("x", scene.W / 2) if isinstance(tr.get("x"), (int, float)) else scene.W / 2
                out.append(Ev(ch["t"], "rise", 0.55, d, speed_curve(tr.get("ease", "in_out_quart")), x=fx,
                              why=f"chapter {ci + 1} portal opens", pal=pal))
                out.append(Ev(ch["t"] + d * 0.92, "impact" if pal == "punchy" else "soft", 0.6, x=fx,
                              why=f"chapter {ci + 1} portal lands", pal=pal))
    for e in getattr(scene, "extra_sound_events", []):
        out.append(e)
    for e in cutfx_events(scene):                # v12
        e.pal = e.pal or palette_of(scene)
        out.append(e)
    out = [e for e in out if e.kind in KINDS]
    out.sort(key=lambda e: e.t)
    out = _declutter(out, scene)
    for i, e in enumerate(out):
        e.seed = i
    return out


_PERC = ("hit", "impact", "click", "pop", "rumble", "lock", "drop", "blip", "tick")


def _declutter(evs, scene, win=0.09):
    """v12: SFX collision limiter — percussive events closer than `win` s are not stacked (that is what makes a
    pop/click): only the strongest keeps full gain, the others are attenuated 12 dB. When the music bed has a drop
    (audio.beat section 'drop'), percussive SFX on that exact moment are attenuated too (the beat already hits)."""
    drops = []
    b = (scene.spec.get("audio") or {}).get("beat")
    if isinstance(b, dict):
        for sct in b.get("sections") or []:
            if sct.get("part") in ("drop", "outro") and isinstance(sct.get("t"), (int, float)):
                drops.append(float(sct["t"]))
    perc = [e for e in evs if e.kind in _PERC]
    for i, e in enumerate(perc):
        rivals = [o for o in perc if o is not e and abs(o.t - e.t) < win]
        if rivals and any(o.gain > e.gain or (o.gain == e.gain and id(o) < id(e)) for o in rivals):
            e.gain *= 0.25
        if any(abs(e.t - d) < 0.06 for d in drops):
            e.gain *= 0.35
    return evs


def render_bus(scene, n):
    """→ (L, R) float32 stereo SFX bus of n samples."""
    pal = palette_of(scene)
    L = np.zeros(n, np.float32); R = np.zeros(n, np.float32)
    cache = {}
    for e in collect(scene):
        ck = None if e.curve is None else np.round(np.asarray(e.curve, float), 2).tobytes()
        ep = e.pal or pal
        key = (e.kind, round(e.dur, 3), e.seed % 6, round(e.pitch, 2), ck, ep)
        s = cache.get(key)
        if s is None:
            s = cache[key] = syn(e.kind, e.dur, e.seed % 6, e.curve, e.pitch, ep)
        i0 = int(round(e.t * SR))
        if i0 >= n or i0 + len(s) <= 0:
            continue
        a = max(0, -i0); i0 = max(0, i0)
        m = min(len(s) - a, n - i0)
        pan = clamp((float(e.x) / scene.W - 0.5) * 1.2, -0.6, 0.6)
        gl, gr = math.cos((pan + 1) * math.pi / 4) * 1.41, math.sin((pan + 1) * math.pi / 4) * 1.41
        L[i0:i0 + m] += s[a:a + m] * e.gain * gl
        R[i0:i0 + m] += s[a:a + m] * e.gain * gr
    try:   # one shared small room → sounds belong to the same space
        from scipy.signal import fftconvolve
        ir_n = int(0.28 * SR)
        decay = np.exp(-np.arange(ir_n) / SR * 14)
        wet_g = {"paper": 0.6, "ui": 0.25, "punchy": 0.4}.get(pal, 0.5)
        for ch_, seed in ((L, 91), (R, 92)):
            ir = _noise(ir_n, seed) * decay * 0.012
            ir[0] = 0
            ch_ += fftconvolve(ch_, ir)[:n].astype(np.float32) * wet_g
    except Exception:
        pass
    return (np.tanh(L * 1.1) / 1.1).astype(np.float32), (np.tanh(R * 1.1) / 1.1).astype(np.float32)


def score_report(scene):
    lines = [f"palette: {palette_of(scene)}"]
    for e in collect(scene):
        lines.append(f"{e.t:7.2f}s  {e.kind:7s} {e.dur:4.2f}s  g{e.gain:.2f}  [{e.pal or '-':6s}] {e.why}")
    return "\n".join(lines)


# --- v7 swiss / kinetic ------------------------------------------------------
@score("block")
def _s_block(el, sc, pal):
    t, d = float(el["t"]), _dur(el, 0.6)
    ev = [Ev(t, "swipe", 0.7, d, speed_curve(el.get("ease", "in_out_expo")), x=el.get("x"), why="block wipe")]
    if el.get("out") is not None:
        ev.append(Ev(float(el["out"]), "swipe", 0.5, float(el.get("out_dur", 0.45)), speed_curve("in_out_expo"),
                     why="block out"))
    return ev


@score("headline")
def _s_headline(el, sc, pal):
    t = float(el["t"]); stg = float(el.get("stagger", 0.12)); d = _dur(el, 0.9)
    n = len(parse_lines(el.get("text", "")))
    e = get_ease(el.get("ease", "out_expo"))
    ev = []
    for i in range(n):   # line lands when its rise reaches ~90%
        tl = first_time(lambda tt: e(prog(tt, t + i * stg, d)) >= 0.9, t + i * stg, t + i * stg + d) or t + i * stg
        ev.append(Ev(t + i * stg, "swipe", 0.35, max(0.12, tl - t - i * stg), speed_curve(el.get("ease", "out_expo")),
                     why=f"line {i + 1} rise"))
        ev.append(Ev(tl, "click" if pal != "punchy" else "hit", 0.6, pitch=1 - 0.05 * i, why=f"line {i + 1} lands"))
    return ev


def parse_lines(text):
    return [ln for ln in str(text).replace("**", "").split("\n") if ln.strip()]


@score("rule")
def _s_rule(el, sc, pal):
    t, d = float(el["t"]), _dur(el, 0.5)
    return [Ev(t, "swipe", 0.45, d, speed_curve(el.get("ease", "in_out_expo")), why="rule")]


@score("big_number")
def _s_bignum(el, sc, pal):
    ev = _s_counter(el, sc, pal)
    ev.insert(0, Ev(float(el["t"]), "swipe", 0.4, 0.4, speed_curve("out_expo"), why="numeral rise"))
    return ev


@score("index")
def _s_index(el, sc, pal):
    t = float(el["t"]); stg = float(el.get("stagger", 0.18))
    ev = [Ev(t + i * stg + 0.18, "click", 0.55, pitch=1 + 0.06 * i, why=f"item {i + 1}")
          for i in range(len(el.get("items", [])))]
    if el.get("highlight_at") is not None:
        ev.append(Ev(float(el["highlight_at"]), "blip", 0.7, why="highlight"))
    return ev


@score("disc")
def _s_disc(el, sc, pal):
    return [Ev(float(el["t"]) + 0.05, "pop" if pal != "ui" else "blip", 0.7, x=el.get("x"), why="disc")]


@score("kinetic")
def _s_kinetic(el, sc, pal):
    from .typo import kinetic_words
    mode = el.get("mode", "slam")
    ev = []
    for i, (w, wt, we, em) in enumerate(kinetic_words(el, sc)):
        if mode == "type":
            n = max(1, len(w))
            for k in range(0, n, 2):
                ev.append(Ev(wt + (we - wt) * k / n, "type", 0.35, pitch=1 + 0.02 * (k % 5), why=f"type {w}"))
            continue
        if mode == "scroll":
            ev.append(Ev(wt, "tick", 0.35 if not em else 0.7, why=f"scroll {w}"))
            continue
        if em:
            ev.append(Ev(wt, "hit", 1.0, pitch=0.9, why=f"EMPH {w}"))
        else:
            ev.append(Ev(wt, "hit" if pal == "punchy" else "click", 0.5 if pal == "punchy" else 0.55,
                         pitch=1 + 0.04 * (i % 4), why=f"word {w}"))
    return ev


# --- v8 Studio Horizon --------------------------------------------------------
@score("waffle_grid")
def _s_waffle(el, sc, pal):
    t = float(el["t"]); fat = float(el.get("fill_at") if el.get("fill_at") is not None else t + 1.2)
    n_side = int(el.get("n", 10))
    k = int(round(float(el.get("value", 1)) / float(el.get("total", 100)) * n_side * n_side))
    ev = [Ev(t, "rise" if pal != "punchy" else "swipe", 0.35, float(el.get("build", 0.7)) + 0.25,
             speed_curve("out"), x=el.get("x"), why="grid builds")]
    for i in range(min(k, 12)):                       # each accent cell pop (capped: a run, not a machine gun)
        ev.append(Ev(fat + i * 0.03 + 0.05, "blip" if pal == "ui" else "pop", 0.75 if i == 0 else 0.3,
                     pitch=1 + 0.03 * i, x=el.get("x"), why=f"accent cell {i + 1}"))
    ev.append(Ev(fat + 0.05, "click", 0.5, why="callout bars"))
    return ev


@score("swiss_bars")
def _s_sbars(el, sc, pal):
    ev = []
    stg = float(el.get("stagger", 0.22)); d = _dur(el, 0.8)
    for i, it in enumerate(el.get("items") or []):
        ti = float(it["t"]) if isinstance(it, dict) and isinstance(it.get("t"), (int, float)) else float(el["t"]) + i * stg
        ev.append(Ev(ti, "swipe", 0.5, d * 0.6, speed_curve(el.get("ease", "out_expo")), x=el.get("x"),
                     why=f"bar {i + 1} slab"))
        ev.append(Ev(ti + d * 0.55, "click", 0.55, pitch=1 + 0.05 * i, why=f"bar {i + 1} lands"))
    return ev


@score("versus_table")
def _s_versus(el, sc, pal):
    t = float(el["t"])
    ev = [Ev(t, "swipe", 0.5, 0.5, speed_curve("in_out_expo"), why="top rule"),
          Ev(t + 0.3, "click", 0.55, why="header slab")]
    for i, r in enumerate(el.get("rows") or []):
        ti = float(r["t"]) if isinstance(r.get("t"), (int, float)) else t + 0.6 + i * 0.5
        ev.append(Ev(ti, "swipe", 0.35, 0.4, speed_curve("out_expo"), why=f"row {i + 1} claim"))
        ev.append(Ev(ti + 0.16, "hit" if pal == "punchy" else "click", 0.6, pitch=0.95, why=f"row {i + 1} reality"))
    return ev


@score("waveform")
def _s_wave(el, sc, pal):
    return []        # it IS the voice


def _s_kinetic_v8(el, sc, pal):
    from .typo import kinetic_words
    ev = []
    for i, (w, wt, we, em) in enumerate(kinetic_words(el, sc)):
        if el.get("mode") == "highlight_sweep":
            if em:
                ev.append(Ev(wt, "swipe", 0.7, float(el.get("sweep_dur", 0.32)), speed_curve("in_out_expo"),
                             why=f"marker sweep {w}"))
                ev.append(Ev(wt + float(el.get("sweep_dur", 0.32)) * 0.9, "hit", 0.6, pitch=1.1, why=f"sweep lands {w}"))
            else:
                ev.append(Ev(wt, "tick", 0.25, why=f"word {w}"))
        else:   # masonry: each word slams into its slot
            ev.append(Ev(wt + 0.02, "hit" if (em or pal == "punchy") else "click", 1.0 if em else 0.45,
                         pitch=(0.85 if em else 1 + 0.04 * (i % 4)), why=f"{'EMPH ' if em else ''}slot {w}"))
    return ev


_s_kinetic_v7 = SCORES.get("kinetic")


@score("kinetic")
def _s_kinetic_dispatch(el, sc, pal):
    if el.get("mode") in ("masonry", "highlight_sweep"):
        return _s_kinetic_v8(el, sc, pal)
    return _s_kinetic_v7(el, sc, pal)


# --- v10 Studio Omni (ui_physics event-locked scores) -------------------------
@score("ui_toggle")
def _s_ui_toggle(el, sc, pal):
    t = float(el["t"])
    ct = float(el.get("click_at", t + 0.10))
    ev = [Ev(ct, "click", 0.85, pitch=1.1, x=el.get("x"), why="toggle click"),
          Ev(ct + 0.08, "pop", 0.7, pitch=1.25, x=el.get("x"), why="toggle flip")]
    if el.get("reset_at") is not None and float(el["reset_at"]) < 900:
        ev.append(Ev(float(el["reset_at"]), "lock", 0.75, x=el.get("x"), why="toggle reset"))
    return ev


@score("ui_slider")
def _s_ui_slider(el, sc, pal):
    t = float(el["t"])
    rt = float(el.get("roll_at", t + 0.2))
    rd = float(el.get("roll_dur", 1.6))
    nt = int(el.get("ticks", 11))
    ev = [Ev(rt, "drop", 0.65, x=el.get("x"), why="slider land")]
    for i in range(1, nt):
        u = 1.0 - (1.0 - i / max(1, nt - 1)) ** (1.0 / 1.85)
        ev.append(Ev(rt + u * rd, "tick", 0.45, pitch=0.85 + 0.05 * i, x=el.get("x"), why=f"slider tick {i}"))
    return ev


@score("ui_notification_stack")
def _s_ui_notif(el, sc, pal):
    t0 = float(el["t"])
    stg = float(el.get("stagger", 0.25))
    ev = []
    for i, c in enumerate(el.get("cards") or [{}, {}, {}]):
        ap = float(c.get("appear_at", t0 + i * stg))
        ev.append(Ev(ap, "blip", 0.6, pitch=1.0 + 0.08 * i, x=el.get("x"), why=f"notif {i+1} appear"))
        if c.get("hit_at") is not None and float(c["hit_at"]) < 900:
            ev.append(Ev(float(c["hit_at"]), "pop", 0.75, pitch=0.95 + 0.06 * i, x=el.get("x"), why=f"notif {i+1} hit"))
    return ev


@score("ui_keyboard")
def _s_ui_kb(el, sc, pal):
    ev = []
    for i, h in enumerate(el.get("hits") or []):
        ht = float(h.get("t", 0.0))
        k = str(h.get("key", h.get("id", "")))
        if k == "return":
            ev.append(Ev(ht, "lock", 0.85, pitch=0.9, x=el.get("x"), why="return key"))
        else:
            ev.append(Ev(ht, "type", 0.65, pitch=0.95 + 0.04 * (i % 5), x=el.get("x"), why=f"key {k}"))
    return ev


@score("ui_film_world")
def _s_ui_film_world(el, sc, pal):
    t0 = float(el["t"])
    ev = [
        Ev(t0 + 0.10, "click", 0.85, pitch=1.1, why="toggle click"),
        Ev(t0 + 0.18, "pop", 0.70, pitch=1.25, why="ball launch"),
        Ev(t0 + 0.50, "drop", 0.65, why="slider land"),
    ]
    for i in range(1, 11):
        u = 1.0 - (1.0 - i / 10.0) ** (1.0 / 1.85)
        ev.append(Ev(t0 + 0.54 + u * 1.56, "tick", 0.42, pitch=0.85 + 0.05 * i, why=f"slider tick {i}"))
    for i, ht in enumerate((2.50, 3.00, 3.50)):
        ev.append(Ev(t0 + ht, "pop", 0.75, pitch=0.95 + 0.06 * i, why=f"card {i+1} bounce"))
    for i, ht in enumerate((4.00, 4.50, 5.00, 5.50, 6.00)):
        ev.append(Ev(t0 + ht, "type", 0.68, pitch=0.95 + 0.04 * i, why=f"key hit {i+1}"))
    ev.append(Ev(t0 + 7.00, "lock", 0.85, pitch=0.9, why="ISO return press"))
    ev.append(Ev(t0 + 7.85, "blip", 0.75, pitch=1.3, why="message sent"))
    ev.append(Ev(t0 + 8.80, "lock", 0.80, pitch=1.0, why="toggle loop lock"))
    return ev


# --- v11 Studio Pantheon (footage edit events + light FX + showcase widgets) -----------------------------
@score("footage")
def _s_footage(el, sc, pal):
    """Every editorial event inside a clip gets its matching SFX: speed ramp = riser into a whoosh out,
    punch / zoom = hit, flash = impact, hue shift / look = swipe, shake = rumble."""
    t0 = float(el["t"])
    ev = []
    for r in el.get("ramps") or []:
        a = t0 + float(r["t"]); d = float(r.get("dur", 0.35)); h = float(r.get("hold", 0.6))
        ev.append(Ev(a, "soft", 0.55, d + h, speed_curve("in"), why="ramp slow-down"))
        ev.append(Ev(a + d + h, "whoosh", 0.7, float(r.get("back_dur", d)), speed_curve("out"), why="ramp back"))
    for z in el.get("zooms") or []:
        ev.append(Ev(t0 + float(z["t"]), "hit", 0.6, pitch=0.9, why="zoom punch"))
    if isinstance(el.get("punch"), list):
        ev += [Ev(t0 + float(p), "click", 0.45, why="punch-in cut") for p in el["punch"]]
    for f in el.get("flash") or []:
        ft = f if isinstance(f, (int, float)) else f.get("t", 0)
        ev.append(Ev(t0 + float(ft), "impact", 0.8, why="flash"))
    for hs in el.get("hue_shift") or []:
        ev.append(Ev(t0 + float(hs["t"]), "swipe", 0.55, float(hs.get("dur", 0.25)), speed_curve("out"), why="hue shift"))
    for lk in el.get("looks") or []:
        ev.append(Ev(t0 + float(lk["t"]), "hit", 0.7, pitch=1.2, why=f"look {lk.get('look', '')}"))
    for sh in el.get("shake") or []:
        ev.append(Ev(t0 + float(sh["t"]), "rumble", 0.7, float(sh.get("dur", 0.3)), why="shake"))
    eg = el.get("edge_glow")
    if eg:
        ev.append(Ev(t0 + float(eg.get("t", 0)), "hum", 0.45, 0.8, speed_curve("in"), why="edge glow on"))
    return ev


@score("saber")
def _s_saber(el, sc, pal):
    return [Ev(float(el["t"]), "whoosh", 0.65, _dur(el, 0.6), speed_curve(el.get("ease", "in_out")), x=el.get("x"),
               why="saber stroke"), Ev(float(el["t"]), "hum", 0.3, _dur(el, 0.6) + 0.3, why="saber hum")]


@score("flash_pulse")
def _s_flash(el, sc, pal):
    return [Ev(float(el["t"]), "impact", 0.9, x=el.get("x"), why="flash pulse")]


@score("glow_ring")
def _s_glow_ring(el, sc, pal):
    return [Ev(float(el["t"]), "blip", 0.5, pitch=1.3, x=el.get("x"), why="glow ring")]


@score("tracked_callout")
def _s_tcallout(el, sc, pal):
    d = _dur(el, 0.7)
    return [Ev(float(el["t"]), "pen", 0.5, d * 0.6, speed_curve("out"), why="callout line"),
            Ev(float(el["t"]) + d * 0.6, "pop", 0.6, why="callout box")]


@score("emitter")
def _s_emitter(el, sc, pal):
    if el.get("kind", "fall") in ("burst", "radial"):
        return [Ev(float(el["t"]), "pop", 0.7, why="burst")]
    return [Ev(float(el["t"]), "soft", 0.35, 1.0, why="particles")]


@score("radial_formula")
def _s_formula(el, sc, pal):
    ev = [Ev(float(el["t"]), "pop", 0.8, why="formula hub")]
    for i, nd in enumerate(el.get("nodes") or []):
        if isinstance(nd.get("t"), (int, float)):
            ev.append(Ev(float(nd["t"]), "swipe", 0.5, 0.6, speed_curve("in_out"), why=f"rotate to node {i + 1}"))
            ev.append(Ev(float(nd["t"]) + 0.6, "click", 0.6, pitch=1 + 0.04 * i, why=f"node {i + 1} lock"))
    return ev


@score("nle_timeline")
def _s_nle(el, sc, pal):
    ev = []
    for i, it in enumerate(el.get("items") or []):
        if i and isinstance(it.get("t"), (int, float)):
            ev.append(Ev(float(it["t"]), "click" if el.get("mode", "clips") == "segments" else "swipe", 0.5, 0.4,
                         speed_curve("in_out"), why=f"timeline step {i + 1}"))
    return ev


@score("badge_button")
def _s_badge(el, sc, pal):
    ev = [Ev(float(el["t"]), "pop", 0.55, why="button")]
    if el.get("press_at") is not None:
        ev.append(Ev(float(el["t"]) + float(el["press_at"]), "click", 0.8, why="button press"))
    return ev


@score("social_outro")
def _s_outro(el, sc, pal):
    return [Ev(float(el["t"]) + 0.15, "pop", 0.8, why="logo"), Ev(float(el["t"]) + 0.35, "type", 0.4, why="handle")]


@score("redact_text")
def _s_redact(el, sc, pal):
    return [Ev(float(el["t"]) + float(el.get("reveal_at", 0.6)), "swipe", 0.55, 0.35, speed_curve("in_out"),
               why="redaction lifts")]


@score("auto_trace")
def _s_trace(el, sc, pal):
    return [Ev(float(el["t"]), "pen", 0.6, _dur(el, 2.5), speed_curve(el.get("ease", "in_out")), x=el.get("x"),
               why="line art draws on")]


@score("sprite_rig")
def _s_rig(el, sc, pal):
    return [Ev(float(el["t"]), "pop", 0.5, why="puppet")]


# --- v12 Studio Cut ----------------------------------------------------------
@score("title_card")
def _s_title_card(el, sc, pal):
    t0 = float(el["t"])
    n = max(1, len(str(el.get("text", "")).split()))
    wd = float(el.get("wipe_dur", 0.32))
    ev = [Ev(t0, "swipe", 0.8, wd, speed_curve("out_expo"), why="slab wipe")]
    for k in range(n):
        ev.append(Ev(t0 + wd * 0.6 + k * float(el.get("word_gap", 0.07)), "hit", 0.5, pitch=1.0 + 0.05 * k,
                     why=f"word {k + 1}"))
    return ev


@score("logo_chip")
def _s_logo_chip(el, sc, pal):
    out = []
    for i, it in enumerate(el.get("items") or []):
        ti = float(it["t"]) if isinstance(it.get("t"), (int, float)) else float(el["t"]) + 0.12 * i
        out.append(Ev(ti, "pop", 0.7, pitch=1.0 + 0.08 * i, why=f"chip {it.get('text')}"))
    return out


@score("route_arc")
def _s_route(el, sc, pal):
    t0 = float(el["t"]); d = float(el.get("dur", 1.4))
    return [Ev(t0, "pop", 0.6, why="pin A"), Ev(t0 + 0.25, "whoosh", 0.55, d, speed_curve("in_out"), why="flight"),
            Ev(t0 + 0.25 + d * 0.95, "pop", 0.8, pitch=1.2, why="pin B lands")]


@score("price_tag")
def _s_price(el, sc, pal):
    t0 = float(el["t"]); d = float(el.get("count_dur", 0.9))
    return [Ev(t0, "pop", 0.8, why="tag pops"), Ev(t0 + 0.15, "rise", 0.45, d, speed_curve("out_expo"), why="count up"),
            Ev(t0 + 0.15 + d, "lock", 0.8, why="value lands")]


@score("grade_scale")
def _s_grade(el, sc, pal):
    t0 = float(el["t"])
    n = len(el.get("grades") or [1, 2, 3])
    ev = [Ev(t0 + 0.15 + 0.12 * i, "pop", 0.55, pitch=1 + 0.1 * i, why="grade cell") for i in range(n)]
    ev.append(Ev(t0 + float(el.get("at", 0.9)), "click", 0.8, why="grade picked"))
    return ev


@score("sticker")
def _s_sticker(el, sc, pal):
    return [Ev(float(el["t"]), "pop", 0.85, pitch=0.9, why="sticker")]


@score("cta_arrow")
def _s_cta(el, sc, pal):
    return [Ev(float(el["t"]), "pen", 0.6, float(el.get("dur", 0.5)), speed_curve("out_cubic"), why="arrow draw")]


def cutfx_events(scene):
    """v12: edit-point transitions get their sound (zoom/whip = whoosh into the cut, glitch = blip, flash = hit)."""
    out = []
    for c in scene.spec.get("_cutfx") or []:
        t, typ, d = float(c["t"]), c.get("type"), float(c.get("dur", 0.3))
        if typ in ("zoom_blur", "whip", "spin"):
            out.append(Ev(t - d / 2, "whoosh", 0.5, d, speed_curve("in"), x=scene.W / 2, why=f"cut {typ}"))
        elif typ == "glitch":
            out.append(Ev(t - d / 2, "blip", 0.6, d, x=scene.W / 2, why="glitch"))
        elif typ in ("flash", "punch"):
            out.append(Ev(t, "hit", 0.5, x=scene.W / 2, why=f"cut {typ}"))
    return out
