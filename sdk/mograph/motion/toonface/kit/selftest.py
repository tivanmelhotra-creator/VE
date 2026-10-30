"""
toonface.kit.selftest — v31.4 Automated Quality & Stability Self-Test Suite.

Run:
    PYTHONPATH=sdk python3 -m mograph.motion.toonface.kit.selftest
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .g2p_fa import g2p
from .rig25 import Face, Kit, brow_line, mouth_geom, tongue_inside_poly, tongue_out_poly
from .sheet25 import EXPRESSIONS_18
from .turn25 import Character
from .videotrack import TrackedHeadFrame, composite_face_on_tracked_frame, detect_moving_head_pose
from .visemes import VISEMES, vis, viseme_track


def test_g2p_and_visemes():
    words = [("ببین", 0.0, 0.4), ("من", 0.4, 0.6), ("بهت", 0.6, 0.9), ("گفتم", 0.9, 1.4),
             ("دیگه", 1.5, 1.8), ("اصلاً", 1.8, 2.2), ("حوصله‌ی", 2.2, 2.7), ("ندارم", 2.7, 3.2)]
    tr = viseme_track(words)
    assert len(tr) >= 18, f"Expected >=18 viseme segments, got {len(tr)}"
    has_mbp = any(v == "MBP" and (t1 - t0) >= 0.05 for t0, t1, v in tr)
    assert has_mbp, "Bilabial closure MBP must hold >= 50ms"
    assert len(VISEMES) == 14


def test_rig25_aesthetics():
    kit = Kit("girl")
    v34, _ = kit.view(25.0)
    # 1. Verify open mouth arches upper lip upward relative to closed sag
    up_closed, lo_closed, *_ = mouth_geom(v34["mouth"], dict(open=0.0, corner=-0.2))
    up_open, lo_open, *_ = mouth_geom(v34["mouth"], dict(open=0.75, corner=-0.2, teeth_up=0.6))
    mid = len(up_open) // 2
    assert up_open[mid, 1] < up_closed[mid, 1] - 4.0, "Open upper lip must arch upward (D-bean shape)"
    # 2. Verify angry brow preserves organic curve camber (not a straight stick)
    c_angry, _ = brow_line(v34["brows"]["L"], v34["eyes"]["L"], Face(brow_anger=1.0))
    chord_mid = 0.5 * (c_angry[0] + c_angry[-1])
    camber_dev = float(np.linalg.norm(c_angry[len(c_angry) // 2] - chord_mid))
    assert camber_dev > 1.8, f"Angry brow should preserve organic arch camber (>1.8px), got {camber_dev:.2f}px"


def test_tongue_v314():
    """Tongue rests on the mouth floor (never floats), L/DTN raise the tip, tongue_out clears the lower lip."""
    kit = Kit("girl")
    v34, _ = kit.view(25.0)
    res = {}
    for name in ("AA", "AH", "L"):
        m = vis(name)
        up, lo, base, hw, u, nv, tp = mouth_geom(v34["mouth"], m)
        gap = float(np.max(np.linalg.norm(lo - up, axis=1)))
        k = int(len(lo) * tp)
        _, top, (b0, half_w, height, tip) = tongue_inside_poly(lo, k, gap, hw, u, nv, m)
        # root sits exactly on the lower lip at the lowest point (no floating egg)
        assert np.linalg.norm(b0 - lo[k]) < 1e-6
        # wide enough to read as a tongue bed (≥ 55% of the mouth half-width)
        assert half_w > 0.55 * hw, f"{name}: tongue too narrow ({half_w / hw:.2f} hw)"
        # never reaches the upper lip
        peak = float(np.max((b0 - top) @ nv))
        assert peak < 0.95 * gap, f"{name}: tongue touches upper lip ({peak / gap:.2f} gap)"
        res[name] = peak / gap
    assert res["L"] > res["AH"] + 0.10, f"L must raise the tongue tip (L={res['L']:.2f}, AH={res['AH']:.2f})"
    m = dict(vis("AH"), tongue_out=0.85, tongue_side=0.35)
    up, lo, base, hw, u, nv, tp = mouth_geom(v34["mouth"], m)
    gap = float(np.max(np.linalg.norm(lo - up, axis=1)))
    k = int(len(lo) * tp)
    _, lobe, (b0, half_w, length, lean, lip) = tongue_out_poly(lo, k, gap, hw, u, nv, m, kit.sw)
    below = float(np.max((lobe - b0) @ nv))
    assert below > 0.35 * hw, f"tongue_out must hang over the lower lip ({below / hw:.2f} hw)"
    return res


def test_turn25_subpixel_stability():
    for nm in ("girl", "boy"):
        ch = Character(nm)
        max_med = 0.0
        max_pt = 0.0
        for yaw in (-35.0, -25.0, -12.5, 0.0, 12.5, 25.0, 35.0):
            rgba = ch.render(yaw, Face(lid_open=0.85, brow_up=0.2, mouth=dict(open=0.4, teeth_up=0.5)))
            assert rgba.shape == (1180, 860, 4)
            err, _ = ch.registration()
            dn = np.linalg.norm(err, axis=1)
            max_med = max(max_med, float(np.median(dn)))
            max_pt = max(max_pt, float(np.max(dn)))
        assert max_med < 0.65, f"{nm} median registration error {max_med:.3f}px exceeds 0.65px"
        assert max_pt < 4.5, f"{nm} max outlier landmark error {max_pt:.3f}px exceeds 4.5px"
    return max_med, max_pt


def test_videotrack_and_18_presets():
    assert len(EXPRESSIONS_18) == 18
    kit = Kit("girl")
    dummy = np.full((720, 720, 3), (145, 175, 210), dtype=np.uint8)
    cv2.circle(dummy, (360, 300), 140, (175, 215, 243), -1)
    tr = detect_moving_head_pose(dummy)
    out = composite_face_on_tracked_frame(dummy, tr, kit, EXPRESSIONS_18[0][4], acting_yaw_delta=20.0)
    assert out.shape == dummy.shape


def main():
    t0 = time.time()
    test_g2p_and_visemes()
    print("  PASS  g2p_fa & 14 visemes (bilabial closure >= 50ms)")
    test_rig25_aesthetics()
    print("  PASS  rig25 v31.3 aesthetics (D-bean arched upper lip + organic curved angry brow)")
    r = test_tongue_v314()
    print(f"  PASS  tongue v31.4 (floor-rested bean, L tip {r['L']:.2f} > AH {r['AH']:.2f} gap, tongue_out over lip)")
    max_med, max_pt = test_turn25_subpixel_stability()
    print(f"  PASS  turn25 stability across [-35..+35 deg] (max median={max_med:.3f}px, max point={max_pt:.3f}px)")
    test_videotrack_and_18_presets()
    print("  PASS  videotrack moving-head clamp & 18 master expressions")
    print(f"toonface.kit selftest: 5/5 pass ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
