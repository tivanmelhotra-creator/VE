"""
mograph.aecomp.generators — OUR OWN procedural shape-element library (no vendor data is shipped).

Each generator returns an aecomp JSON comp (render with mograph.aecomp.render).  Timing/easing constants are the
MEASURED medians of 447 studied reference elements (stats in presets/aecomp_stats.json):
    element length 1.0 s (bursts / blow-ups / simple), 1.33 s (arrows / slides / transitions), 1.53 s (arcs)
    trim draw  : End leads, Start follows ≈ 0.15–0.18 s later, each ≈ 0.6–1.1 s long
    ease       : AE influence 33 % out / 33 % in (Easy Ease) on most props; bursts use 17 % out (fast attack)
    strokes    : round caps (≈ 60 %), width slider 5–15 px at a 1100 px comp (→ scale with size)
    repeaters  : 2–8 copies (4 most common), polystar 3/4/30 points, twist ±26°
Controls (AE "FX Control" equivalents) every generator understands: c1, c2, c3 (colours), stroke (px), size (%).

    from mograph.aecomp import generators as GEN
    comp = GEN.make("burst", seed=3, c1="#ffd36b")
    GEN.names()            # all generator names
"""
from __future__ import annotations

import math
import random

E33 = 1 / 3          # AE Easy Ease influence
E17 = 1 / 6


def kf(*rows, ease=(E33, E33)):
    """kf((t, v), (t, v), ...) → AE bezier keys with zero speed and the given influences (out, in)."""
    io, ii = ease
    return {"k": [[float(t), v, "b", 0.0, ii, 0.0, io] for t, v in rows]}


def kfl(*rows):
    return {"k": [[float(t), v, "l", 0, 0, 0, 0] for t, v in rows]}


def _stroke(color="c1", width=1.0, cap="round", op=None, dash=None):
    s = {"ty": "st", "color": {"ctl": color}, "width": {"ctl": "stroke", "mul": width}, "cap": cap, "join": "round"}
    if op is not None:
        s["opacity"] = op
    if dash:
        s["dash"] = dash
    return s


def _fill(color="c1", op=None):
    f = {"ty": "fl", "color": {"ctl": color}}
    if op is not None:
        f["opacity"] = op
    return f


def _trim(t0, d, lag=0.16, ease=(0.7, 0.33), s_end=100, rev=False):
    a, b = (0, 100) if not rev else (100, 0)
    return {"ty": "tm", "end": kf((t0, a), (t0 + d, b), ease=ease),
            "start": kf((t0 + lag, a), (t0 + lag + d, b), ease=ease), "mode": 1}


def _fade(t_in, t_out, d=0.1):
    return kf((t_in, 0), (t_in + d, 100), (t_out - d, 100), (t_out, 0))


def comp(layers, dur, size=1100, controls=None, name="element"):
    C = {"c1": "#ffffff", "c2": "#ffd36b", "c3": "#2dd4bf", "stroke": 12, "size": 100}
    C.update(controls or {})
    return {"name": name, "w": size, "h": size, "dur": dur, "fps": 30, "controls": C,
            "layers": [{"name": "fx", "type": "shape", "in": 0, "out": dur,
                        "tr": {"pos": [size / 2, size / 2], "scale": {"ctl": "size", "vec": 2}},
                        "shapes": layers}]}


# ===================================================================================== generators
def burst(seed=0, rays=8, dur=1.0, radius=420, inner=90, dots=True, dash=False):
    """Explosion: radial strokes shoot out (trim), optional dots on alternate rays (measured median 1.0 s)."""
    r = random.Random(seed)
    rays = rays or r.choice([6, 8, 10, 12])
    L = []
    ray = {"ty": "gr", "it": [
        {"ty": "sh", "path": {"v": [[0, -inner], [0, -radius]], "i": [[0, 0], [0, 0]], "o": [[0, 0], [0, 0]], "c": False}},
        _trim(0.0, 0.55 * dur, lag=0.18 * dur, ease=(0.83, 0.33)),
        _stroke("c1", 1.0, dash={"d": [30, 22], "offset": 0} if dash else None),
        {"ty": "rp", "copies": rays, "tr": {"rot": 360.0 / rays, "pos": [0, 0]}}]}
    L.append(ray)
    if dots:
        dot = {"ty": "gr", "it": [
            {"ty": "el", "size": kf((0.15 * dur, [0, 0]), (0.45 * dur, [34, 34]), (0.95 * dur, [0, 0])),
             "pos": kf((0.15 * dur, [0, -inner]), (0.8 * dur, [0, -radius * 1.12]), ease=(E17, E33))},
            _fill("c2"),
            {"ty": "rp", "copies": rays, "tr": {"rot": 360.0 / rays, "pos": [0, 0]}}],
            "tr": {"rot": 180.0 / rays}}
        L.append(dot)
    return comp(L, dur, name=f"burst_{seed}")


def ring_pop(seed=0, dur=1.0, radius=380, rings=2):
    """Blow-up: concentric rings expand and thin out (measured 1.0 s, repeater copies 2–3)."""
    L = []
    for k in range(rings):
        d0 = 0.08 * k * dur
        L.append({"ty": "gr", "it": [
            {"ty": "el", "size": kf((d0, [0, 0]), (d0 + 0.75 * dur, [2 * radius * (1 - 0.18 * k)] * 2), ease=(E17, E33))},
            {"ty": "st", "color": {"ctl": "c1" if k == 0 else "c3"}, "cap": "round",
             "width": kf((d0, 0.0), (d0 + 0.12 * dur, 20.0), (d0 + 0.8 * dur, 0.0))}]})
    return comp(L, dur, name=f"ring_pop_{seed}")


def arc_swirl(seed=0, dur=1.53, radius=380, arcs=3, spin=120):
    """Arcs: several concentric arcs draw on & off with offset rotation (measured 1.53 s, trim ≈ 1.09 s)."""
    r = random.Random(seed)
    L = []
    for k in range(arcs):
        rr = radius * (1 - 0.2 * k)
        d0 = 0.06 * k
        L.append({"ty": "gr", "it": [
            {"ty": "el", "size": [2 * rr, 2 * rr]},
            {"ty": "tm", "end": kf((d0, 0), (d0 + 0.72 * dur, 100), ease=(0.52, 0.87)),
             "start": kf((d0 + 0.1 * dur, 0), (d0 + 0.82 * dur, 100), ease=(0.52, 0.87)),
             "offset": kf((0, r.uniform(0, 360)), (dur, r.uniform(0, 360) + spin * (1 if k % 2 == 0 else -1)))},
            _stroke(["c1", "c2", "c3"][k % 3], [1.0, 0.45, 0.45][k % 3])]})
    return comp(L, dur, name=f"arc_swirl_{seed}")


def arrow_trace(seed=0, dur=1.33, radius=420):
    """Arrows: a circular line is traced and an arrow-head rides its end (measured 1.33 s)."""
    L = [{"ty": "gr", "it": [
        {"ty": "el", "size": [2 * radius, 2 * radius]},
        {"ty": "tm", "end": kf((0.1, 0), (0.85 * dur, 100), ease=(0.6, 0.6)),
         "start": kf((0.25, 0), (dur, 100), ease=(0.6, 0.6)), "offset": -90},
        _stroke("c1", 1.0)]}]
    # head: rides the end point = angle of trim end around the circle (ellipse starts at the top)
    ks = []
    import numpy as np
    from .values import ev
    endk = L[0]["it"][1]["end"]
    for t in np.linspace(0.1, dur, 24):
        e = ev(endk, float(t)) / 100.0
        a = math.radians(-90 + 360 * e - 90)
        ks.append([float(t), [radius * math.cos(a), radius * math.sin(a)], "l", 0, 0, 0, 0])
    rot = [[r[0], math.degrees(math.atan2(r[1][1], r[1][0])) + 180, "l", 0, 0, 0, 0] for r in ks]
    head = {"ty": "gr", "it": [{"ty": "sr", "kind": "polygon", "points": 3, "outer": 46, "rot": 0},
                               {"ty": "rd", "radius": 8}, _fill("c2")],
            "tr": {"pos": {"k": ks}, "rot": {"k": rot},
                   "scale": kf((0.1, [0, 0]), (0.3, [100, 100]), (0.85 * dur, [100, 100]), (dur, [0, 0]))}}
    L.insert(0, head)
    return comp(L, dur, name=f"arrow_trace_{seed}")


def slide_lines(seed=0, dur=1.33, length=820, lines=3, gap=46):
    """Slides: parallel strokes whoosh across (End leads, Start follows; measured trim ≈ 0.77 s)."""
    r = random.Random(seed)
    L = []
    for k in range(lines):
        y = (k - (lines - 1) / 2) * gap
        d0 = r.uniform(0, 0.12)
        w = 1.0 if k == lines // 2 else 0.4
        L.append({"ty": "gr", "it": [
            {"ty": "sh", "path": {"v": [[-length / 2, y], [length / 2, y]], "i": [[0, 0], [0, 0]], "o": [[0, 0], [0, 0]], "c": False}},
            {"ty": "tm", "end": kf((d0, 0), (d0 + 0.6 * dur, 100), ease=(0.7, 0.97)),
             "start": kf((d0 + 0.13 * dur, 0), (d0 + 0.75 * dur, 100), ease=(0.7, 0.97))},
            _stroke("c1" if k == lines // 2 else "c3", w)]})
    return comp(L, dur, name=f"slide_lines_{seed}")


def wipe_circle(seed=0, dur=1.27, size=1100):
    """Transition: an outer disc grows to cover the frame, an inner hole follows and reveals it again
    (even-odd doughnut; measured 1.27 s, strong 90 % influence)."""
    big = size * 1.5
    outer = {"ty": "el", "size": kf((0.0, [0, 0]), (0.55 * dur, [big, big]), ease=(0.9, 0.9))}
    inner = {"ty": "el", "size": kf((0.35 * dur, [0, 0]), (dur, [big, big]), ease=(0.9, 0.9))}
    return comp([{"ty": "gr", "it": [outer, inner, {"ty": "fl", "color": {"ctl": "c1"}, "rule": "eo"}]}], dur,
                size=size, name=f"wipe_circle_{seed}")


def bars_wipe(seed=0, dur=1.27, size=1100, bars=5, angle=0):
    """Transition: staggered bars sweep across and leave (rect size keys, 33 % ease)."""
    L = []
    h = size / bars
    for k in range(bars):
        d = 0.05 * k * dur
        L.append({"ty": "gr", "it": [
            {"ty": "rc", "size": kf((d, [0, h + 2]), (d + 0.45 * dur, [size, h + 2]), (d + 0.55 * dur, [size, h + 2]),
                                    (d + 0.9 * dur, [0, h + 2]), ease=(0.8, 0.8)),
             "pos": kf((d, [-size / 2, 0]), (d + 0.45 * dur, [0, 0]), (d + 0.55 * dur, [0, 0]),
                       (d + 0.9 * dur, [size / 2, 0]), ease=(0.8, 0.8))},
            _fill("c1" if k % 2 == 0 else "c2")],
            "tr": {"pos": [0, (k - (bars - 1) / 2) * h]}})
    return comp([{"ty": "gr", "it": L, "tr": {"rot": angle}}], min(dur + 0.05 * bars, 2.0), size=size,
                name=f"bars_wipe_{seed}")


def star_blink(seed=0, dur=1.0, points=4, outer=220):
    """Simple element: a sparkle star pops (scale overshoot) and spins out (measured 1.0 s)."""
    L = [{"ty": "gr", "it": [{"ty": "sr", "kind": "star", "points": points, "outer": outer, "inner": outer * 0.28,
                              "inner_round": 30, "outer_round": 0}, _fill("c2")],
          "tr": {"scale": kf((0, [0, 0]), (0.35 * dur, [115, 115]), (0.5 * dur, [100, 100]), (dur, [0, 0])),
                 "rot": kf((0, -45), (dur, 90))}}]
    return comp(L, dur, name=f"star_blink_{seed}")


def zigzag_line(seed=0, dur=1.0, length=800, ridges=8, amp=40):
    """Simple element: a zig-zag stroke draws through (zigzag modifier + trim)."""
    L = [{"ty": "gr", "it": [
        {"ty": "sh", "path": {"v": [[-length / 2, 0], [length / 2, 0]], "i": [[0, 0], [0, 0]], "o": [[0, 0], [0, 0]], "c": False}},
        {"ty": "zz", "size": amp, "ridges": ridges, "smooth": seed % 2 == 1},
        _trim(0, 0.6 * dur, lag=0.2 * dur), _stroke("c1", 0.8)]}]
    return comp(L, dur, name=f"zigzag_line_{seed}")


def spinner_loop(seed=0, dur=2.0, radius=300, dots=12):
    """Looped element: dots on a ring pulse in sequence, the ring rotates (loops seamlessly; Looped median 8 s)."""
    L = [{"ty": "gr", "it": [
        {"ty": "el", "size": {"k": [[0, [18, 18], "b", 0, E33, 0, E33], [dur / 2, [44, 44], "b", 0, E33, 0, E33],
                                    [dur, [18, 18], "b", 0, E33, 0, E33]], "loop": "cycle"}, "pos": [0, -radius]},
        _fill("c1"), {"ty": "rp", "copies": dots, "offset": 0, "tr": {"rot": 360 / dots, "pos": [0, 0], "start_op": 25, "end_op": 100}}],
        "tr": {"rot": {"rate": 360 / (dur * 2), "v": 0}}}]
    return comp(L, dur, name=f"spinner_loop_{seed}")


def confetti(seed=0, dur=1.0, n=14, radius=440):
    """Explosion variant: mixed triangles / diamonds / circles fly out, spin and shrink (Explosions sheet look)."""
    r = random.Random(seed)
    L = []
    for i in range(n):
        a = 2 * math.pi * i / n + r.uniform(-0.2, 0.2)
        d = radius * r.uniform(0.55, 1.0)
        kind = r.choice(["tri", "dia", "dot"])
        sz = r.uniform(18, 34)
        shape = ({"ty": "sr", "kind": "polygon", "points": 3, "outer": sz} if kind == "tri" else
                 {"ty": "sr", "kind": "polygon", "points": 4, "outer": sz} if kind == "dia" else
                 {"ty": "el", "size": [sz * 1.3, sz * 1.3]})
        L.append({"ty": "gr", "it": [shape, _stroke(r.choice(["c1", "c2", "c3"]), 0.45)],
                  "tr": {"pos": kf((0, [0, 0]), (0.75 * dur, [d * math.cos(a), d * math.sin(a)]), ease=(E17, E33)),
                         "rot": kf((0, 0), (dur, r.uniform(-240, 240))),
                         "scale": kf((0, [0, 0]), (0.2 * dur, [100, 100]), (0.6 * dur, [100, 100]), (dur, [0, 0]))}})
    return comp(L, dur, name=f"confetti_{seed}")


GENERATORS = {
    "burst": (burst, "explosion", "radial rays shoot out + dots"),
    "burst_dash": (lambda seed=0, **k: burst(seed, dash=True, dots=False, **k), "explosion", "dashed rays"),
    "confetti": (confetti, "explosion", "mixed outline shapes fly out & spin"),
    "ring_pop": (ring_pop, "blow_up", "concentric rings expand and thin out"),
    "arc_swirl": (arc_swirl, "arc", "concentric arcs draw on/off while rotating"),
    "arrow_trace": (arrow_trace, "arrow", "arrow head rides a traced circle"),
    "slide_lines": (slide_lines, "slide", "parallel lines whoosh across"),
    "wipe_circle": (wipe_circle, "transition", "doughnut disc wipe (cover + reveal)"),
    "bars_wipe": (bars_wipe, "transition", "staggered bars sweep across"),
    "star_blink": (star_blink, "simple", "sparkle star pops with overshoot"),
    "zigzag_line": (zigzag_line, "simple", "zig-zag stroke draws through"),
    "spinner_loop": (spinner_loop, "loop", "seamless loop: dots pulse around a turning ring"),
}


def names(cat=None):
    return [k for k, v in GENERATORS.items() if cat is None or v[1] == cat]


def make(name, seed=0, **controls):
    """build a comp; keyword controls (c1/c2/c3/stroke/size) override defaults, other kwargs go to the generator."""
    fn = GENERATORS[name][0]
    ctl = {k: v for k, v in controls.items() if k in ("c1", "c2", "c3", "stroke", "size")}
    gk = {k: v for k, v in controls.items() if k not in ctl}
    c = fn(seed=seed, **gk)
    c["controls"].update(ctl)
    c["gen"] = {"name": name, "seed": seed}
    return c
