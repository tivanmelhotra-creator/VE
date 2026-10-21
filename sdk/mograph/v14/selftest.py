"""
mograph.v14.selftest — numeric regression tests for every v15 Craft engine (no footage needed, < 15 s).

    cd sdk && python3 -m mograph.v14 selftest          # exit 0 = all pass

Each test measures the property that v15.0 got wrong (see sdk/V15_REVIEW.md), so a regression is caught at once:
  multiband   Σ bands == input (perfect reconstruction) and flat response at low level (±2.5 dB)
  remix       beat grid preserved through every splice (click track, 92 & 120 BPM, shorten + extend)
  arap        a pure 90° rotation of two pins is reproduced exactly (rigid) ; pins hit their targets
  morph       rotated rectangle: area constant (±1 %) and orientation linear in t (±2°)
  oklab       round trip exact (≤ 1 LSB) ; grade match on a strong cast keeps skin hue within 8°
  decontam    composite error on a synthetic green-spill edge reduced ≥ 3×
  pathtext    glyph spacing across a path knot stays uniform (±15 %)
  varstroke   miter ribbon keeps width through a 90° corner (ink present at the outer corner)
  selector    range / wiggly weights in range and smooth shape = 1 at centre
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np


def _t_multiband():
    from .audio import split_bands, multiband_compress_4band, SR
    x = np.random.default_rng(0).standard_normal(SR * 2).astype(np.float32) * 0.01
    rec = np.sum(split_bands(x, SR), axis=0)
    err = float(np.abs(rec - x).max())
    y = multiband_compress_4band(x, SR, bands_cfg=[(1.0, 2.0, 1.0)] * 4)       # no compression, unity makeup
    X, Y = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    dev = max(abs(20 * np.log10(Y[(f > a) & (f < b)].mean() / X[(f > a) & (f < b)].mean()))
              for a, b in [(60, 140), (150, 170), (700, 900), (4000, 5000), (9000, 12000)])
    return err < 1e-4 and dev < 0.5, f"recon err {err:.1e}, crossover dev {dev:.2f} dB"


def _t_remix():
    from .remix import remix_music_to_duration
    sr = 44100
    worst = 0.0
    for bpm in (92, 120):
        beat = 60 / bpm
        y = np.zeros(int(sr * 14), np.float32)
        for k in range(int(13.5 / beat)):
            i = int((0.3 + k * beat) * sr)
            y[i:i + 200] += 1.0 if k % 4 == 0 else 0.6
        for tgt in (7.3, 9.1, 31.0):
            o = remix_music_to_duration(y, sr, tgt, None)
            if len(o) != int(round(tgt * sr)):
                return False, f"length {len(o)} != {tgt}s"
            idx = np.where((o[1:] > 0.3) & (o[:-1] <= 0.3))[0]
            d = np.diff(idx) / sr
            worst = max(worst, float(np.abs(d - beat).max()))
    return worst < 0.006, f"max beat-interval error {worst * 1000:.1f} ms"


def _t_arap():
    from ..puppet_arap import ArapPuppetMesh
    M = ArapPuppetMesh(np.zeros((120, 120, 4), np.uint8), 8, 8)
    c = M.V_rest.mean(0)
    rot = lambda p: np.array([[0, -1], [1, 0]]) @ (np.asarray(p) - c) + c  # noqa: E731
    pins = [(0, tuple(rot(M.V_rest[0])), 1e3), (63, tuple(rot(M.V_rest[63])), 1e3)]
    e_rot = float(np.abs(M.solve(pins) - np.array([rot(p) for p in M.V_rest])).max())
    pins = [(i, tuple(M.V_rest[i]), 1e4) for i in range(56, 64)] + [(7, (M.V_rest[7][0] + 40, M.V_rest[7][1] + 10), 1e4)]
    V = M.solve(pins, iterations=20)
    e_pin = float(np.abs(V[7] - np.array(pins[-1][1])).max())
    return e_rot < 1e-3 and e_pin < 0.5, f"rigid err {e_rot:.1e} px, pin err {e_pin:.2f} px"


def _t_morph():
    import cv2
    from ..autotrace import shape_context_procrustes_morph
    r = np.array([[0, 0], [200, 0], [200, 50], [0, 50]], np.float32)
    worst_a, worst_ang = 0.0, 0.0
    for deg in (60, -80):
        th = math.radians(deg)
        Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        r2 = ((r - [100, 25]) @ Rm.T + [300, 25]).astype(np.float32)
        a0 = abs(cv2.contourArea(shape_context_procrustes_morph(r, r2, 0.0, bending_resistance=1.0)))
        for t in (0.25, 0.5, 0.75):
            m = shape_context_procrustes_morph(r, r2, t, bending_resistance=1.0)
            worst_a = max(worst_a, abs(abs(cv2.contourArea(m)) / a0 - 1))
            m0 = m - m.mean(0)
            w, v = np.linalg.eigh(np.cov(m0.T))
            ang = math.degrees(math.atan2(v[1, 1], v[0, 1])) % 180
            exp = (deg * t) % 180
            worst_ang = max(worst_ang, min(abs(ang - exp), 180 - abs(ang - exp)))
    return worst_a < 0.01 and worst_ang < 2.0, f"area dev {worst_a * 100:.2f} %, angle dev {worst_ang:.2f}°"


def _t_oklab():
    from .palette import bgr_to_oklab, oklab_to_bgr, match_reference_grade, apply_grade_match
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    rt = int(np.abs(oklab_to_bgr(bgr_to_oklab(img)).astype(int) - img).max())
    # skin-ish frame + a strong green reference → skin hue must move < 8°
    skin = np.zeros((120, 90, 3), np.uint8)
    skin[...] = (120, 150, 205)
    skin[:40] = (90, 110, 70)
    ref = skin.copy()
    ref[..., 1] = np.clip(ref[..., 1] * 1.5, 0, 255)
    ref[..., 2] = (ref[..., 2] * 0.6).astype(np.uint8)
    out = apply_grade_match(skin, match_reference_grade(skin, ref), 1.0, work=1.0)

    def hue(px):
        lab = bgr_to_oklab(np.asarray(px, np.uint8).reshape(1, 1, 3))[0, 0]
        return math.degrees(math.atan2(lab[2], lab[1])) % 360
    dh = abs(hue(out[80, 45]) - hue(skin[80, 45]))
    dh = min(dh, 360 - dh)
    return rt <= 1 and dh <= 8.5, f"round-trip {rt} LSB, skin hue shift {dh:.1f}°"


def _t_decontam():
    from .compositor import decontaminate_foreground_edge
    H = W = 400
    yy, xx = np.mgrid[0:H, 0:W]
    A = np.clip((120 - np.hypot(xx - 200, yy - 200)) / 20 + 0.5, 0, 1).astype(np.float32)
    fg = np.zeros((H, W, 3), np.float32)
    fg[...] = [30, 30, 220]
    bg = np.zeros_like(fg)
    bg[...] = [30, 220, 30]
    plate = (fg * A[..., None] + bg * (1 - A[..., None])).astype(np.uint8)
    out = decontaminate_foreground_edge(plate, A, 1.0)
    nb = np.full_like(fg, [200, 40, 40])
    band = (A > 0.2) & (A < 0.8)
    ideal = fg * A[..., None] + nb * (1 - A[..., None])
    e0 = float(np.abs(plate * A[..., None] + nb * (1 - A[..., None]) - ideal)[band].mean())
    e1 = float(np.abs(out * A[..., None] + nb * (1 - A[..., None]) - ideal)[band].mean())
    return e1 * 3 < e0, f"edge composite error {e0:.1f} → {e1:.1f}"


def _t_pathtext():
    from . import glyphs as G
    F = G.Field("bezier_path", points=[[-300, 0], [0, -90], [300, 0]])
    x = np.linspace(-20, 20, 5).astype(np.float32)
    o = F(np.stack([x, np.full_like(x, -30), np.zeros_like(x)], 1))[:, :2]
    g = np.hypot(*np.diff(o, axis=0).T)
    dev = float(g.max() / g.min() - 1)
    return dev < 0.15, f"spacing variation across knot {dev * 100:.1f} %"


def _t_varstroke():
    import cairo
    from ..varstroke import draw_tapered_stroke, Taper
    s = cairo.ImageSurface(cairo.FORMAT_A8, 200, 200)
    c = cairo.Context(s)
    draw_tapered_stroke(c, np.array([[20, 100], [100, 100], [100, 180]], np.float32), 30, (1, 1, 1, 1),
                        taper=Taper(0, 0, 1, 1))
    s.flush()
    a = np.ndarray((200, s.get_stride()), np.uint8, s.get_data())[:, :200]
    ok = a[88, 112] > 128            # outer corner of the elbow (miter keeps full width)
    return bool(ok), f"outer-corner ink {int(a[88, 112])}"


def _t_selector():
    from .glyphs import eval_selector
    v = [eval_selector(u, 0.3, {"shape": "smooth"}) for u in np.linspace(0, 1, 11)]
    w = [eval_selector(u, t, {"shape": "wiggly"}) for u in np.linspace(0, 1, 7) for t in (0, .5, 1.3)]
    ok = abs(v[5] - 1) < 1e-6 and min(v) >= 0 and max(abs(q) for q in w) <= 1.0
    return ok, f"smooth centre {v[5]:.2f}, wiggly range ±{max(abs(q) for q in w):.2f}"


TESTS = [("multiband", _t_multiband), ("remix", _t_remix), ("arap", _t_arap), ("morph", _t_morph),
         ("oklab", _t_oklab), ("decontam", _t_decontam), ("pathtext", _t_pathtext), ("varstroke", _t_varstroke),
         ("selector", _t_selector)]


def main():
    fails = 0
    t_all = time.time()
    for name, fn in TESTS:
        t0 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"EXCEPTION {type(e).__name__}: {e}"
        fails += not ok
        print(f"  {'✓' if ok else '✗'} {name:<10} {msg}   ({(time.time() - t0) * 1000:.0f} ms)")
    print(f"selftest: {len(TESTS) - fails}/{len(TESTS)} pass  ({time.time() - t_all:.1f} s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
