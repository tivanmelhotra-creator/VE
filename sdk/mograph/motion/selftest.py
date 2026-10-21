"""numeric regression tests of the v18 motion kit (< 20 s, no footage).  python3 -m mograph.motion selftest"""
from __future__ import annotations

import math
import time

import numpy as np


def t_glossary():
    from .glossary import resolve
    r = {d["canonical"]: d for d in resolve("یه اسلوموشن نرم با ease out و افکت نئون آبی روی متن، بعد لیپ سینک کاراکتر خرفه‌ای")}
    ok = (r.get("retime", {}).get("op") == "slow_motion" and r.get("easing", {}).get("type") == "ease_out"
          and r.get("saber", {}).get("preset") == "neon" and r["saber"].get("target") == "text_layer"
          and r["saber"].get("color") == "#2a7dff" and "lip_sync" in r
          and r.get("character_animation", {}).get("style") == "rubber_hose")
    r2 = resolve("تایم‌لپس")[0]
    return ok and r2.get("speed") == 8.0, f"{len(r)} concepts, timelapse → ×{r2.get('speed')}"


def t_easing():
    from ..core import EASINGS
    from .easing import cubic_bezier, ease, measure
    err = max(max(abs(ease(n)(i / 400) - EASINGS[n](i / 400)) for i in range(401))
              for n in ("out_back", "out_elastic", "out_bounce", "out_expo", "in_out_cubic", "out_quart"))
    css = cubic_bezier(0.42, 0, 0.58, 1)
    sym = abs(css(0.5) - 0.5)
    m = measure("out_back")
    same = abs(ease("easeOutBack")(0.3) - ease("back.out")(0.3)) + abs(ease("کاهش سرعت")(0.4) - ease("ease_out")(0.4))
    return err < 1e-9 and sym < 1e-6 and abs(m["overshoot"] - 10.0) < 0.3 and same < 1e-12, \
        f"Penner err {err:.1e}, css sym {sym:.1e}, out_back overshoot {m['overshoot']} %"


def t_retime():
    from .retime import Retime
    rt = Retime([{"op": "slow_motion", "t": 1, "dur": 1, "speed": 0.25, "ramp": 0.2},
                 {"op": "freeze_frame", "t": 3, "dur": 0.5},
                 {"op": "speed_ramp", "t": 4, "dur": 0.5, "from": 1, "to": 0.2, "hold": 0.4, "back": 0.4},
                 {"op": "reverse", "t": 6, "dur": 1, "mode": "replace"}], 8.0)
    us = np.linspace(0, rt.out(6.0) - 1e-3, 8000)              # continuity up to the reverse (a jump by design)
    es = np.array([rt.edited(u) for u in us])
    jump = float(np.abs(np.diff(es)).max())
    inv = max(abs(rt.edited(rt.out(e)) - e) for e in (0.5, 1.5, 3.5, 4.2, 5.5, 7.5))
    # slow motion 1 s at 0.25 with 0.2 s ramps: 0.6/0.25 + 2·∫ramps → exact expectation
    ramp = 2 * 0.2 * np.mean([1 / (1 + (0.25 - 1) * (0.5 - 0.5 * math.cos(math.pi * x))) for x in np.linspace(0, 1, 2001)])
    exp_slow = 0.6 / 0.25 + ramp
    got = rt.out(2.0) - rt.out(1.0)
    return jump < 0.01 and inv < 1e-3 and abs(got - exp_slow) < 0.01, \
        f"out {rt.duration:.2f}s from 8 s, continuity {jump:.4f}, inverse err {inv:.1e}, slow span {got:.3f}≈{exp_slow:.3f}"


def t_flow():
    import cv2
    from .retime import FlowInterp
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur((rng.random((240, 340, 3)) * 255).astype(np.uint8), (0, 0), 2.5)
    a, b = np.ascontiguousarray(base[:, 20:300]), np.ascontiguousarray(base[:, 8:288])   # 12 px pan
    ref = base[:, 14:294]                                                                  # true middle frame
    m = FlowInterp("flow", 280).frame(a, b, 0.5)
    ef = float(np.abs(m.astype(int) - ref)[:, 20:-20].mean())
    eb = float(np.abs(cv2.addWeighted(a, .5, b, .5, 0).astype(int) - ref)[:, 20:-20].mean())
    return ef < eb * 0.2, f"in-between error flow {ef:.3f} vs cross-fade {eb:.2f} (12 px pan)"


def t_saber():
    from . import saber as SB
    W, H = 320, 320
    fx = SB.render({"preset": "neon", "core": "shape", "shape": "circle", "w": 160, "x": 160, "y": 160}, 0.5, W, H)
    ring = int(fx[160, 80, :3].max())           # on the ring
    centre = int(fx[160, 160, :3].max())        # inside
    far = int(fx[5, 5, :3].max())
    pre = bool((fx[..., :3].max(axis=2).astype(int) <= fx[..., 3].astype(int) + 1).all())
    fx2 = SB.render({"preset": "default", "start": [40, 160], "end": [280, 160], "core_end": 50}, 0.5, W, H)
    trim = int(fx2[160, 100, :3].max()) > 200 and int(fx2[160, 250, :3].max()) < 60
    n = len(SB.preset_names())
    return ring > 230 and centre < ring * 0.6 and far < 20 and pre and trim and n >= 51, \
        f"{n} presets, ring {ring} centre {centre} far {far}, premultiplied {pre}, trim {trim}"


def t_saber_time():
    from . import saber as SB
    sp = {"preset": "fire", "core": "shape", "shape": "circle", "w": 300, "x": 270, "y": 480}
    a = SB.render(sp, 1.0, 540, 960).astype(np.int16)
    b = SB.render(sp, 1.0 + 1 / 30, 540, 960).astype(np.int16)
    c = SB.render(sp, 1.6, 540, 960).astype(np.int16)
    d1, d2 = float(np.abs(a - b).mean()), float(np.abs(a - c).mean())
    t0 = time.time()
    SB.render({"preset": "electric", "core": "text", "text": "سلام", "size": 200, "x": 540, "y": 960}, 0.3, 1080, 1920)
    dt = time.time() - t0
    return d1 < d2 and d1 > 0.01, f"animated: 1-frame Δ {d1:.2f} < 0.6 s Δ {d2:.2f} (continuous), 1080×1920 text {dt:.2f}s"


def t_lipsync():
    from .lipsync import at, g2v, track
    ok_g = g2v("مامان")[0] == "MBP" and "AI" in g2v("مامان") and g2v("فیل")[0] == "FV" and g2v("mama") == ["MBP", "AI", "MBP", "AI"]
    k = track([(0.4, 0.9, "مامان"), (1.4, 1.8, "بابا")], fps=30)
    vs = [x["v"] for x in k]
    closed_gap = at(k, 1.2)[0] == "rest"
    closures = [k[i + 1]["t"] - k[i]["t"] for i in range(len(k) - 1) if k[i]["v"] == "MBP"]
    return ok_g and closed_gap and min(closures) >= 2 / 30 - 1e-3, f"{len(k)} keys {vs[:6]}…, MBP ≥ {min(closures) * 30:.1f} frames"


def t_character():
    from . import character as CH
    C = CH.prepare({"rig": "bean", "actions": [{"do": "hop", "t": 0.2}, {"do": "wave", "t": 1.4}], "timing": "twos"})
    r = CH.principles_report(C, 0, 2.8)
    q = CH.quantise(0.05, "twos") == 0.0 and abs(CH.quantise(0.07, "twos") - 2 / 30) < 1e-9
    ok = (r["squash_range"][0] < 0.85 and r["squash_range"][1] > 1.08 and r["volume_error_max"] < 0.08
          and r["anticipated_jumps"] == 1 and r["follow_through_deg"] > 15 and q)
    import cairo
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 300, 400)
    CH.draw(cairo.Context(s), C, 0.6, 150, 380, 200)
    return ok, f"squash {r['squash_range']}, volume err {r['volume_error_max']}, follow-through {r['follow_through_deg']}°"


def t_bridges():
    from ..aefx import bridge as AB
    from ..elements import REGISTRY
    from ..v14 import layers as LY
    img = np.zeros((200, 200, 4), np.uint8)
    img[90:110, 40:160] = 255
    it = AB.resolve_items([{"saber": {"preset": "neon", "color": "#ff2a6d"}}], float, 0.0, 5.0)
    out = AB.apply_items(img, it, 1.0)
    glow = int(out[80, 100, 3])
    return all(k in LY.DRAW for k in ("saber", "character", "mouth")) and "saber" in REGISTRY and glow > 20, \
        f"v14 DRAW + v5 REGISTRY + fx item (glow alpha 10 px above the layer = {glow})"


TESTS = [t_glossary, t_easing, t_retime, t_flow, t_saber, t_saber_time, t_lipsync, t_character, t_bridges]


def main():
    t0 = time.time()
    n = 0
    for f in TESTS:
        t1 = time.time()
        try:
            ok, msg = f()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, msg = False, f"{type(e).__name__}: {e}"
        n += bool(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {f.__name__[2:]:<11} {msg}   ({time.time() - t1:.1f}s)")
    print(f"motion selftest: {n}/{len(TESTS)} pass  ({time.time() - t0:.1f}s)")
    return 0 if n == len(TESTS) else 1
