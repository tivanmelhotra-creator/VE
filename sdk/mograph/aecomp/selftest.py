"""numeric regression tests for mograph.aecomp (no vendor files needed).  python3 -m mograph.aecomp selftest"""
from __future__ import annotations

import math
import time

import numpy as np

from . import geom as Gm
from . import generators as GEN
from . import lowerthirds as LT
from . import titles as TT
from .render import render_bgra
from .values import ev


def _alpha(comp, t, W=200, H=200):
    return render_bgra(comp, t, W, H)[..., 3].astype(np.float32) / 255.0


def _one(shapes, dur=1.0, size=200, controls=None):
    return {"w": size, "h": size, "dur": dur, "controls": controls or {},
            "layers": [{"name": "s", "type": "shape", "in": 0, "out": dur, "tr": {"pos": [size / 2, size / 2]},
                        "shapes": shapes}]}


def t_keys():
    k = {"k": [[0, 0, "b", 0, 1 / 3, 0, 1 / 3], [1, 100, "b", 0, 1 / 3, 0, 1 / 3]]}
    mid, q1, q3 = ev(k, 0.5), ev(k, 0.25), ev(k, 0.75)
    loop = {"k": [[0, 0, "l"], [1, 10, "l"]], "loop": "cycle"}
    ok = abs(mid - 50) < 0.01 and abs(q1 + q3 - 100) < 0.01 and q1 < 25 and abs(ev(loop, 1.5) - 5) < 1e-6
    return ok, f"easy-ease mid {mid:.2f}, q1 {q1:.2f}, loopOut cycle {ev(loop, 1.5):.1f}"


def t_ellipse():
    p = Gm.ellipse([200, 200])
    L = Gm.path_length(p)
    return abs(L - 2 * math.pi * 100) / (2 * math.pi * 100) < 0.002, f"ellipse perimeter err {abs(L / (200 * math.pi) - 1) * 100:.3f} %"


def t_trim():
    line = Gm.Path([[0, 0], [100, 0]], None, None, False)
    a = Gm.trim([line], 25, 75, 0)
    circ = Gm.trim([Gm.ellipse([200, 200])], 0, 50, 0)
    ok = abs(Gm.path_length(a[0]) - 50) < 0.2 and abs(Gm.path_length(circ[0]) - math.pi * 100) < 1.0
    wrap = Gm.trim([Gm.ellipse([200, 200])], 0, 25, 340)            # crosses the seam → still one piece
    return ok and len(wrap) == 1, f"line 25-75 → {Gm.path_length(a[0]):.2f}, half circle {Gm.path_length(circ[0]):.1f}, wrap pieces {len(wrap)}"


def t_render_fill():
    c = _one([{"ty": "rc", "size": [100, 100]}, {"ty": "fl", "color": "#ffffff"}])
    a = _alpha(c, 0.0)
    cov = float(a.sum()) / 10000.0
    return abs(cov - 1.0) < 0.02, f"100x100 rect coverage {cov:.3f}"


def t_merge():
    c = _one([{"ty": "gr", "it": [{"ty": "el", "size": [160, 160]}, {"ty": "fl", "color": "#fff"}]},
              {"ty": "gr", "it": [{"ty": "el", "size": [80, 80]}, {"ty": "fl", "color": "#fff"}]},
              {"ty": "mm", "mode": "sub"}, {"ty": "fl", "color": "#ffffff"}])
    a = _alpha(c, 0.0)
    ring = math.pi * (80 ** 2 - 40 ** 2)
    return abs(a.sum() / ring - 1) < 0.03 and a[100, 100] < 0.05, f"merge subtract ring area ratio {a.sum() / ring:.3f}"


def t_merge_outline():
    # union of two overlapping circles, then ONE stroke: AE strokes the union outline (no inner arcs)
    c = _one([{"ty": "el", "size": [100, 100], "pos": [-30, 0]}, {"ty": "el", "size": [100, 100], "pos": [30, 0]},
              {"ty": "mm", "mode": "add"}, {"ty": "st", "color": "#fff", "width": 4}])
    a = _alpha(c, 0.0)
    inner = a[100, 100 - 20] + a[100, 100 + 20]           # the two inner arcs pass through x = ±20 at y = 0
    outer = a[100, 100 - 80] > 0.5 and a[100, 100 + 80] > 0.5
    return inner < 0.05 and outer, f"union outline inner {inner:.2f} outer {outer}"


def t_wipes():
    # full-frame white square; radial wipe 25 % (clockwise from 12) removes the top-right quadrant,
    # linear wipe 50 % at 90° removes the left half
    base = {"w": 200, "h": 200, "dur": 1, "layers": [{"name": "s", "type": "shape", "in": 0, "out": 1,
            "tr": {"pos": [100, 100]}, "shapes": [{"ty": "rc", "size": [200, 200]}, {"ty": "fl", "color": "#fff"}]}]}
    import copy
    r = copy.deepcopy(base)
    r["layers"][0]["fx"] = [{"ty": "radial_wipe", "completion": 25, "center": [100, 100], "dir": "cw"}]
    a = render_bgra(r, 0.0)[..., 3] / 255.0
    ok_r = a[40, 160] < 0.05 and a[40, 40] > 0.95 and a[160, 160] > 0.95
    li = copy.deepcopy(base)
    li["layers"][0]["fx"] = [{"ty": "linear_wipe", "completion": 50, "angle": 90}]
    b = render_bgra(li, 0.0)[..., 3] / 255.0
    ok_l = b[100, 40] < 0.05 and b[100, 160] > 0.95
    td = copy.deepcopy(base)
    td["layers"][0]["shapes"] = [{"ty": "sh", "path": {"v": [[-90, 0], [90, 0]], "i": [[0, 0], [0, 0]], "o": [[0, 0], [0, 0]],
                                  "c": False}}, {"ty": "st", "color": "#fff", "width": 4}]
    plain = render_bgra(td, 0.0)[..., 3]
    td["layers"][0]["fx"] = [{"ty": "turbulent_displace", "amount": 60, "size": 4, "complexity": 2, "evolution": 30}]
    d1, d2 = render_bgra(td, 0.0)[..., 3], render_bgra(td, 0.0)[..., 3]
    moved = float(((d1 > 128) & (plain < 128)).sum()) / max(1.0, float((plain > 128).sum()))
    ok_t = (d1 == d2).all() and moved > 0.1
    return ok_r and ok_l and ok_t, f"radial quadrant {ok_r}, linear half {ok_l}, turbulent moved {moved:.2f}"


def t_repeater():
    c = _one([{"ty": "el", "size": [20, 20], "pos": [0, -60]}, {"ty": "fl", "color": "#fff"},
              {"ty": "rp", "copies": 6, "tr": {"rot": 60, "pos": [0, 0]}}])
    a = _alpha(c, 0.0)
    blobs = 0
    for k in range(6):
        ang = math.radians(-90 + 60 * k)
        x, y = int(100 + 60 * math.cos(ang)), int(100 + 60 * math.sin(ang))
        blobs += a[y, x] > 0.5
    return blobs == 6 and a[100, 100] < 0.01, f"repeater 6 copies found {blobs}"


def t_stroke_trim_anim():
    c = _one([{"ty": "sh", "path": {"v": [[-80, 0], [80, 0]], "i": [[0, 0], [0, 0]], "o": [[0, 0], [0, 0]], "c": False}},
              {"ty": "tm", "end": {"k": [[0, 0, "l"], [1, 100, "l"]]}}, {"ty": "st", "color": "#fff", "width": 6, "cap": "butt"}])
    w = [float(_alpha(c, t).sum() / 6.0) for t in (0.25, 0.5, 0.999)]
    return abs(w[0] - 40) < 3 and abs(w[1] - 80) < 3 and abs(w[2] - 160) < 4, f"trim-on length {w[0]:.0f}/{w[1]:.0f}/{w[2]:.0f} px"


def t_matte():
    c = {"w": 200, "h": 200, "dur": 1, "layers": [
        {"name": "fill", "type": "shape", "in": 0, "out": 1, "tr": {"pos": [100, 100]}, "matte": {"layer": "m", "mode": "alpha"},
         "shapes": [{"ty": "rc", "size": [200, 200]}, {"ty": "fl", "color": "#fff"}]},
        {"name": "m", "type": "shape", "in": 0, "out": 1, "tr": {"pos": [50, 100]},
         "shapes": [{"ty": "rc", "size": [100, 200]}, {"ty": "fl", "color": "#fff"}]}]}
    a = _alpha(c, 0.0)
    return a[100, 40] > 0.95 and a[100, 160] < 0.05, f"alpha matte left {a[100, 40]:.2f} right {a[100, 160]:.2f}"


def t_generators():
    bad = []
    t0 = time.time()
    for n in GEN.names():
        c = GEN.make(n, seed=1)
        vis = [float(_alpha(c, f * c["dur"], 120, 120).sum()) for f in (0.3, 0.5)]
        if max(vis) < 5:
            bad.append(n)
    return not bad, f"{len(GEN.names())} generators visible mid-animation ({(time.time() - t0):.1f}s)" + (f" EMPTY: {bad}" if bad else "")


def t_lower_thirds():
    bad = []
    for st in LT.styles():
        c = LT.make(st, "دکتر سارا", "متخصص")
        a0, a1, a2 = (float(_alpha(c, t, 270, 480).sum()) for t in (0.0, 2.5, c["dur"] - 0.01))
        if not (a1 > 30 and a0 < a1 and a2 < a1 * 0.2):
            bad.append((st, round(a0), round(a1), round(a2)))
    return not bad, f"{len(LT.styles())} lower thirds: hidden → visible → hidden" + (f" BAD {bad}" if bad else "")


def t_titles():
    bad = []
    for st in TT.styles():
        c = TT.make(st, "مقدمه را آخر بنویس", "نکته")
        a1 = _alpha(c, 1.6, 270, 480)
        cols = np.nonzero(a1.sum(0) > 0.5)[0]
        width_ok = len(cols) and cols.min() > 3 and cols.max() < 266          # auto-fit: inside the frame
        a2 = float(_alpha(c, c["dur"] - 0.01, 270, 480).sum())
        if not (a1.sum() > 50 and width_ok and a2 < a1.sum() * 0.25):
            bad.append(st)
    return not bad, f"{len(TT.styles())} titles fit the frame and exit" + (f" BAD {bad}" if bad else "")


TESTS = [("keyframes", t_keys), ("ellipse", t_ellipse), ("trim", t_trim), ("fill", t_render_fill),
         ("merge", t_merge), ("merge_line", t_merge_outline), ("wipes", t_wipes), ("repeater", t_repeater), ("trim_anim", t_stroke_trim_anim), ("matte", t_matte),
         ("generators", t_generators), ("lower3rds", t_lower_thirds),
         ("titles", t_titles)]


def run():
    ok_n = 0
    T = time.time()
    for name, fn in TESTS:
        t0 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        ok_n += bool(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:10s} {msg}   ({time.time() - t0:.1f}s)")
    print(f"aecomp selftest: {ok_n}/{len(TESTS)} pass  ({time.time() - T:.1f}s)")
    return ok_n == len(TESTS)
