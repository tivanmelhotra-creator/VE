"""regression tests of the v19 PRO EDITOR kit (< 5 s, no footage needed).   python3 -m mograph.pro selftest"""
from __future__ import annotations

import json
import os
import tempfile
import time

import cv2
import numpy as np

TMP = tempfile.gettempdir()


def test_frame(W=360, H=640):
    """synthetic plate: warm→cool gradient, grey ramp (bottom half), skin patch."""
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    img = np.zeros((H, W, 3), np.float32)
    img[..., 0] = 0.85 - 0.4 * y
    img[..., 1] = 0.60 - 0.2 * y
    img[..., 2] = 0.35 + 0.1 * y
    img[H // 2:, :, :] = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    img[H // 4 - 60:H // 4 + 60, W // 2 - 60:W // 2 + 60] = [0.50, 0.62, 0.85]     # skin (BGR)
    return (img * 255).astype(np.uint8)


class _F:
    def __init__(self, t, W=360, H=640):
        self.t, self.W, self.H, self.M = t, W, H, None


def t_cube():
    from .color import identity, write_cube, read_cube, sample, resample, look_lut
    p = os.path.join(TMP, "_mg_t.cube")
    lut = look_lut("teal_orange", 17)
    write_cube(p, lut)
    L2, _, _ = read_cube(p)
    rt = float(np.abs(L2 - lut).max())
    write_cube(p, identity(17))
    I, _, _ = read_cube(p)
    rgb = np.random.default_rng(3).random((2000, 3)).astype(np.float32)
    err = float(np.abs(sample(resample(I), rgb) - rgb).max())
    return rt < 1e-5 and err < 1e-5, f".cube write→read {rt:.1e}, identity 17³→33³ trilinear err {err:.1e}"


def t_looks():
    from .color import LOOKS, Grader
    img = test_frame()
    d = {nm: float(np.abs(Grader({"look": nm, "protect_skin": 0}).apply_lut(img).astype(np.float32) - img).mean())
         for nm in LOOKS}
    mono = Grader({"look": "mono_film", "protect_skin": 0}).apply_lut(img).astype(np.float32)
    chroma = float(np.abs(mono[..., 0] - mono[..., 2]).mean())
    ident = Grader({"contrast": 1.0})
    same = np.array_equal(ident.apply_lut(img), img)
    return all(v > 1.0 for v in d.values()) and chroma < 6 and same, \
        f"{len(LOOKS)} looks change the plate (min Δ {min(d.values()):.1f}), mono chroma {chroma:.1f}, identity no-op"


def t_skin():
    from .color import Grader
    img = test_frame()
    H, W = img.shape[:2]
    a = Grader({"look": "night_neon", "protect_skin": 0.0}).apply_lut(img).astype(np.float32)
    b = Grader({"look": "night_neon", "protect_skin": 1.0}).apply_lut(img).astype(np.float32)
    sk = (slice(H // 4 - 40, H // 4 + 40), slice(W // 2 - 40, W // 2 + 40))
    bg = (slice(10, 60), slice(10, 60))
    da, db = float(np.abs(a[sk] - img[sk]).mean()), float(np.abs(b[sk] - img[sk]).mean())
    bgk = float(np.abs(b[bg] - img[bg]).mean())
    return db < da * 0.5 and bgk > 1.0, f"skin shift {da:.1f} → {db:.1f} with protect_skin 1 (background still graded {bgk:.1f})"


def t_finish():
    from .color import Grader, add_grain, letterbox
    img = np.full((640, 360, 3), 128, np.uint8)
    g1, g2, g3 = add_grain(img, 0.5, 1.4, 7), add_grain(img, 0.5, 1.4, 7), add_grain(img, 0.5, 1.4, 8)
    det = np.array_equal(g1, g2) and not np.array_equal(g1, g3)
    lb = letterbox(img, 2.39)
    bars = int(lb[:5].max()) == 0 and int(lb[320].min()) == 128
    gr = Grader({"grain": 0.3, "sharpen": 0.4, "halation": 0.3, "letterbox": 1.0})
    out = gr.finish(img, 3)
    return det and bars and out.shape == img.shape and gr.active, \
        f"grain deterministic per frame (chunk-safe) {det}, std {float(g1.std()):.1f}, letterbox {bars}"


def t_transitions():
    from .broll import TRANSITIONS, transition_alpha
    bad = []
    for k in TRANSITIONS:
        op1, m1, dx1, dy1, z1, _ = transition_alpha(k, 1.0, 64, 96)
        if not (op1 >= 0.999 and (m1 is None or float(m1.mean()) > 0.97) and abs(dx1) < 1e-6 and abs(dy1) < 1e-6
                and abs(z1 - 1) < 1e-6):
            bad.append(k)
    return not bad, f"{len(TRANSITIONS)} transitions land exactly on the picture (bad: {bad or 'none'})"


def _broll_png():
    p = os.path.join(TMP, "_mg_broll.png")
    im = np.zeros((400, 600, 3), np.uint8)
    im[:] = (40, 160, 250)
    cv2.circle(im, (300, 200), 120, (255, 255, 255), -1)
    cv2.imwrite(p, im)
    return p


def t_broll_modes():
    from . import broll as B
    p = _broll_png()
    res = {}
    for mode in ("full", "pip", "split_top", "split_bottom"):
        L = {"type": "broll", "src": p, "_src_path": p, "mode": mode, "_t0": 1.0, "_t1": 3.0, "in": "fade",
             "out_fx": "fade", "kenburns": True, "_grader": None}
        mid, fin, out = B.render_layer(L, _F(2.0)), B.render_layer(L, _F(1.1)), B.render_layer(L, _F(3.5))
        res[mode] = (round(float(mid[1].mean()), 2), float(fin[1].max()) < float(mid[1].max()), out is None)
    ok = (res["full"][0] > 0.99 and 0.04 < res["pip"][0] < 0.5 and 0.45 < res["split_top"][0] < 0.55
          and all(v[1] and v[2] for v in res.values()))
    return ok, "coverage " + " ".join(f"{k}:{v[0]}" for k, v in res.items()) + ", fade-in + out-of-range ok"


def t_kenburns():
    from .broll import kenburns
    L = {"kenburns": {"from": [1.0, 0.5, 0.5], "to": [1.2, 0.6, 0.4]}}
    a, m, b = kenburns(L, 0.0), kenburns(L, 0.5), kenburns(L, 1.0)
    return abs(a[0] - 1.0) < 1e-9 and abs(b[0] - 1.2) < 1e-9 and abs(m[0] - 1.1) < 1e-9 and abs(b[1] - 0.6) < 1e-9, \
        f"zoom {a[0]:.2f} → {m[0]:.2f} → {b[0]:.2f}, centre {a[1]:.2f} → {b[1]:.2f}"


def t_chroma():
    from .broll import chroma_key
    im = np.zeros((100, 100, 3), np.uint8)
    im[:] = (0, 255, 0)
    cv2.circle(im, (50, 50), 25, (60, 90, 200), -1)
    im[:, :8] = (60, 200, 70)
    a, out = chroma_key(im, {"color": "#00ff00", "tol": 0.15, "soft": 0.1, "spill": 1.0})
    bg, fg = float(a[5:15, 40:60].mean()), float(a[45:55, 45:55].mean())
    spill = int(out[50, 3, 1]) <= int(max(out[50, 3, 0], out[50, 3, 2])) + 1
    return bg < 0.05 and fg > 0.95 and spill, f"bg alpha {bg:.2f}, subject alpha {fg:.2f}, despill {spill}"


def t_time_anchors():
    from ..v14.compositor import Timeline
    from ..motion.retime import RetimedTimeline
    tl = Timeline([[0, 2], [3, 5]], [{"w": "سلام", "t": 0.5, "end": 0.9}, {"w": "دنیا", "t": 3.2, "end": 3.6}])
    r = RetimedTimeline(tl, [{"op": "freeze_frame", "t": 1.0, "dur": 0.5}])
    ok = (tl.resolve("src:3.5") == 2.5 and tl.resolve("e:1.5") == 1.5 and r.resolve(1.5) == 2.0 and
          abs(r.resolve(4.0) - 4.5) < 1e-9 and r.resolve("s:1.2") == 1.2 and abs(r.resolve("w:دنیا") - 2.7) < 1e-6
          and r.resolve("2") == 2.5 and r.resolve(0.5) == 0.5 and abs(r.resolve(4.2) - 4.7) < 1e-9)
    return ok, "plain seconds / e: / s: / src: / w: all consistent through a retime (freeze +0.5 s)"


def t_captions():
    from . import export as EX
    ph = [(0.0, 1.25, "سلام دنیا"), (1.3, 2.0, "Hello 12")]
    s = open(EX.write_srt(ph, os.path.join(TMP, "_mg.srt")), encoding="utf-8").read()
    v = open(EX.write_vtt(ph, os.path.join(TMP, "_mg.vtt")), encoding="utf-8").read()
    a = open(EX.write_ass(ph, os.path.join(TMP, "_mg.ass")), encoding="utf-8").read()
    ok = ("00:00:01,250" in s and "\u202b" in s and v.startswith("WEBVTT") and "00:00:01.300" in v and
          "Dialogue: 0,0:00:00.00,0:00:01.25" in a and EX.tc(61.5, 30) == "00:01:01:15" and EX.ts_srt(59.9996) == "00:01:00,000")
    return ok, "SRT / VTT / ASS + SMPTE timecode, rounding carry, RTL marks on Persian lines"


def t_lint():
    from .lint import known_keys, layer_types, _sugg, lint
    ks, ty = known_keys(), layer_types()
    ok = ("broll" in ty and "saber" in ty and "kinetic_title" in ty and "offset" in ks and "kenburns" in ks
          and "ofset" not in ks and "'offset'" in _sugg("ofset", ks) and "'kinetic_title'" in _sugg("kinetc_title", ty))
    p = os.path.join(TMP, "_mg_bad.json")
    json.dump({"source": {"src": "nope.mp4"}, "grade": {"look": "teal_orang"}, "layers": [{"type": "kinetc_title"}]},
              open(p, "w"))
    E, _ = lint(p)
    ok = ok and any("nope.mp4" in e for e in E) and any("'kinetic_title'" in e for e in E) and \
        any("'teal_orange'" in e for e in E)
    return ok, f"{len(ty)} layer types, {len(ks)} keys read from the engine source, typo suggestions ok"


def t_registry():
    from ..v14 import layers as LY
    ok = LY.DRAW.get("broll") is not None and "broll" in LY.PREPARE
    from ..motion import bridge as MB
    return ok and hasattr(MB, "_anchor_blade"), "broll registered in v14 DRAW + PREPARE, saber blade anchors present"


TESTS = [("cube", t_cube), ("looks", t_looks), ("skin", t_skin), ("finish", t_finish), ("transitions", t_transitions),
         ("broll", t_broll_modes), ("kenburns", t_kenburns), ("chroma", t_chroma), ("time", t_time_anchors),
         ("captions", t_captions), ("lint", t_lint), ("registry", t_registry)]


def main():
    t0 = time.time()
    n = 0
    for name, fn in TESTS:
        t1 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, msg = False, f"{type(e).__name__}: {e}"
        n += bool(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<11} {msg}   ({time.time() - t1:.1f}s)")
    print(f"pro selftest: {n}/{len(TESTS)} pass  ({time.time() - t0:.1f}s)")
    return 0 if n == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
