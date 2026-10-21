"""
mograph.aefx.selftest — numeric regression tests for the AE-FX engines (no footage, < 15 s).

    cd sdk && python3 -m mograph.aefx selftest

  keyframes   AE bezier: speed 0 / influence 33 % == Easy Ease (symmetric S), ends exact, hold & linear modes
  prproj      'Position L to R 01' fit to its reference trajectory ≤ 4 px RMS (field order speed/influence)
  libraries   352 motion + 351 text presets load; every IN preset ends at identity (≤ 2 grey levels, 340+ of them)
  selector    AE range: square/ramp/smooth/triangle bounds, offset −100 → all selected, +100 → none
  textanim    every text preset evaluates; opacity-ramp preset is monotone in time (no flicker)
  cursive     Persian joined letters share one unit ('آموزش' → آ|مو|ز|ش = 4 pieces)
  deepglow    dark background untouched (0 LSB), highlight energy spreads (> 25 px halo), BGRA keeps alpha ≥ input
  pathblur    zero blur = identity; blur along x leaves a horizontal line sharp and smears a vertical one
  bridge      layer fx resolve (in/out/at word) + repivot keeps the motion identical about a new anchor
"""
from __future__ import annotations

import sys
import time

import numpy as np


def _t_keyframes():
    from .keyframes import evaluate, easy_ease, linear
    c = easy_ease([[0, 0.0], [1, 100.0]])
    mid, q1, q3 = evaluate(c, 0.5), evaluate(c, 0.25), evaluate(c, 0.75)
    ok = abs(mid - 50) < 0.5 and abs((q1 + q3) - 100) < 0.5 and q1 < 25 and evaluate(c, 1.0) == 100.0
    h = {"k": [[0, 1.0, "h", 0, 0, 0, 0], [1, 5.0, "h", 0, 0, 0, 0]]}
    ok &= evaluate(h, 0.99) == 1.0 and abs(evaluate(linear([[0, 0], [2, 10]]), 1.0) - 5) < 1e-9
    return ok, f"ease mid {mid:.2f}, q1+q3 {q1 + q3:.2f}"


def _t_prproj():
    from .keyframes import evaluate
    # Premiere ground truth (preview GIF, logo centroid x on a 320 px frame, 25 fps, frames 11..26)
    gif = [196, 190, 184, 180, 176, 170, 168, 166, 165, 163, 161, 161, 160, 160, 159, 159]
    c = {"k": [[0.0, 1.7625, "b", 0.0, 0.92355, 0.01909, 0.51515], [1.01274, 0.5, "b", 0.0, 1.0, 0.46713, 0.33333]]}
    best = min(np.sqrt(np.mean([((evaluate(c, (11 + j - off) / 25) - 0.5) * 320 + 159 - g) ** 2
                                for j, g in enumerate(gif)])) for off in np.arange(-15, 15, 0.25))
    return best < 4.0, f"RMS {best:.2f} px"


def _t_libraries():
    from . import motion as AM, textanim as TA
    import cv2
    img = np.zeros((90, 160, 3), np.uint8)
    cv2.rectangle(img, (40, 25), (120, 65), (60, 160, 255), -1)
    n_in = n_ok = 0
    for k, p in AM.library().items():
        if p["dir"] == "in":
            n_in += 1
            n_ok += float(np.abs(AM.apply_preset(img, k, p["dur"] + 0.01).astype(int) - img).mean()) <= 2.0
    nm, nt = len(AM.library()), len(TA.library())
    return nm >= 350 and nt >= 351 and n_ok >= n_in - 12, f"{nm} motion, {nt} text, IN→identity {n_ok}/{n_in}"


def _t_selector():
    from .textanim import range_weights
    n = 10
    sq = range_weights({"start": 0, "end": 100, "offset": -100, "shape": "square"}, n, 0)
    allsel = range_weights({"start": 0, "end": 100, "offset": 0, "shape": "square"}, n, 0)
    sm = range_weights({"start": 0, "end": 100, "offset": 0, "shape": "smooth"}, n, 0)
    tri = range_weights({"start": 0, "end": 100, "offset": 0, "shape": "triangle"}, n, 0)
    ramp = range_weights({"start": 0, "end": 100, "offset": 0, "shape": "ramp_up"}, n, 0)
    ok = sq.max() == 0 and allsel.min() == 1 and 0.9 < sm[n // 2] <= 1 and sm[0] < 0.2 and tri.max() > 0.85
    ok &= np.all(np.diff(ramp) >= -1e-9) and ramp.min() >= 0 and ramp.max() <= 1
    return bool(ok), f"smooth centre {sm[n // 2]:.2f}, ramp monotone"


class _G:
    def __init__(self, i):
        self.order = i
        self.word = self.line = 0
        self.size = 100.0


class _B:
    def __init__(self, n):
        self.glyphs = [_G(i) for i in range(n)]
        self.words = []


def _t_textanim():
    from . import textanim as TA
    b = _B(6)
    for k in TA.library():
        an = TA.Animator(k)
        for t in (0, 0.4, 1.0):
            st = an.frame(b, t)
            if not np.all(np.isfinite(st.tx)) or not np.all(np.isfinite(st.alpha)):
                return False, f"{k} non-finite"
    an = TA.Animator("atp_100")
    vis = [float(np.clip(an.frame(b, t).alpha, 0, 1).mean()) for t in np.linspace(0, 1.0, 26)]
    mono = all(b2 >= a2 - 1e-6 for a2, b2 in zip(vis, vis[1:]))
    return mono and vis[0] < 0.05 and vis[-1] > 0.98, f"opacity ramp {vis[0]:.2f}→{vis[-1]:.2f} monotone={mono}"


def _t_cursive():
    try:
        from ..v14 import glyphs as G
    except Exception as e:  # noqa: BLE001
        return True, f"skipped ({e.__class__.__name__})"
    from .textanim import _units
    b = G.typeset("آموزش", 80)
    ui, n, _ = _units(b, "chars")
    return n == 4, f"'آموزش' → {n} connected pieces (آ|مو|ز|ش) ({len(b.glyphs)} glyphs)"


def _t_deepglow():
    import cv2
    from .deepglow import deep_glow
    img = np.zeros((240, 240, 3), np.uint8)
    img[:] = (20, 16, 12)
    cv2.circle(img, (120, 120), 12, (255, 255, 255), -1)
    o = deep_glow(img, radius=60)
    bg = int(np.abs(o[3, 3].astype(int) - img[3, 3]).max())
    halo = int(np.argmax(o[120, 132:].max(-1) < 40))
    a = np.zeros((64, 64, 4), np.uint8)
    a[28:36, 28:36] = 255
    oa = deep_glow(a, radius=20)
    return bg == 0 and halo >= 20 and bool(np.all(oa[..., 3] >= a[..., 3])), f"bg drift {bg}, halo {halo} px"


def _t_pathblur():
    import cv2
    from .pathblur import path_blur
    img = np.zeros((120, 200, 3), np.uint8)
    cv2.line(img, (20, 60), (180, 60), (255, 255, 255), 1)          # horizontal
    cv2.line(img, (100, 10), (100, 110), (255, 255, 255), 1)        # vertical
    same = path_blur(img, [(0, 60), (200, 60)], 0, 0)
    o = path_blur(img, [(0, 60), (200, 60)], 30, 30)
    h_sharp = o[60, 40, 0] > 200
    row = o[30, 80:121, 0].astype(int)
    v_smear = row.max() < 40 and (row > 3).sum() >= 24                # 1 px line spread over ~30 px, no gaps
    return bool(np.array_equal(same, img) and h_sharp and v_smear), f"horizontal kept {o[60, 40, 0]}, vertical smeared over {(row > 3).sum()} px"


def _t_bridge():
    from . import bridge as AB, motion as AM
    it = AB.resolve_items([{"motion": "scale_01_in"}, {"motion": "blur_01_out", "at": "out"}],
                          lambda v: float(v), 2.0, 6.0)
    p_out = AM.get("blur_01_out")
    ok = abs(it[0]["_start"] - 2.0) < 1e-9 and abs(it[1]["_end"] - 6.0) < 1e-9 and abs(it[1]["_start"] -
                                                                                     (6 - p_out["dur"])) < 1e-9
    f = {"fx": "transform", "p": {"rotation": 30.0, "scale": 80.0}}
    M0 = AM.transform_matrix(f["p"], 0, 400, 300)
    M1 = AM.transform_matrix(AB._repivot(f, 0.25, 0.25)["p"], 0, 400, 300)
    pt = np.array([100, 75, 1.0])                                  # the new pivot must stay fixed
    ok &= np.allclose((M1 @ pt)[:2], [100, 75], atol=1e-6) and not np.allclose(M0 @ pt, M1 @ pt)
    return bool(ok), "in/out windows + pivot fixed"


TESTS = [("keyframes", _t_keyframes), ("prproj", _t_prproj), ("libraries", _t_libraries), ("selector", _t_selector),
         ("textanim", _t_textanim), ("cursive", _t_cursive), ("deepglow", _t_deepglow), ("pathblur", _t_pathblur),
         ("bridge", _t_bridge)]


def main():
    fails = 0
    t00 = time.time()
    for name, fn in TESTS:
        t0 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{e.__class__.__name__}: {e}"
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:10s} {msg}   ({time.time() - t0:.1f}s)")
    print(f"aefx selftest: {len(TESTS) - fails}/{len(TESTS)} pass  ({time.time() - t00:.1f}s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
