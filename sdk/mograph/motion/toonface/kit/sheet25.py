"""
toonface.kit.sheet25 — v31.3 Complete 18-Expression Master Sheet + Legacy Compose Helpers.

Renders all 18 reference expressions on both characters (`girl` and `boy`) across Front (0°), 3/4 Right (+25°),
and 3/4 Left (−25°) views using `turn25.Character` and `rig25.Face`.

CLI:
    python3 -m mograph.motion.toonface.kit.sheet25 --out-dir /tmp
"""
from __future__ import annotations

import argparse
import os

import cairo
import cv2
import numpy as np

from .acting import HeadPose
from .blankplate import load_blank
from .draw25 import FaceState, draw_face
from .rig25 import Face
from .turn25 import Character
from .visemes import mix, vis


def bgra_from_surface(surf):
    w, h = surf.get_width(), surf.get_height()
    a = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    return a.copy()


def over(dst, src_pm):
    a = src_pm[:, :, 3:4].astype(np.float32) / 255.0
    return (dst.astype(np.float32) * (1 - a) + src_pm[:, :, :3].astype(np.float32)).clip(0, 255).astype(np.uint8)


def over_straight(dst, bgra):
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    return (dst.astype(np.float32) * (1 - a) + bgra[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def render_face_layer(shape_hw, kit, st):
    h, w = shape_hw
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    draw_face(ctx, kit, st)
    surf.flush()
    return bgra_from_surface(surf)


def compose(bp, kit, st):
    face = render_face_layer(bp.base.shape[:2], kit, st)
    out = over(bp.base, face)
    return over_straight(out, bp.hair_fg)


# 18 Master Expressions (6x3 Grid) across Front (0°), 3/4 Right (+25°), and 3/4 Left (-25°)
EXPRESSIONS_18: list[tuple[str, float, float, float, Face]] = [
    ("1_friendly_ah",     0.0,   0.0,  0.0, Face(lid_open=0.88, brow_up=0.35, mouth=dict(open=0.68, width=1.02, corner=0.25, teeth_up=0.55, tongue=0.55))),
    ("2_bright_ee_smile", 25.0,  0.0,  1.2, Face(lid_open=0.90, lower=0.18, brow_up=0.45, mouth=dict(open=0.52, width=1.12, corner=0.45, teeth_up=0.75, tongue=0.45))),
    ("3_curious_oo",      -25.0, 2.0, -1.0, Face(lid_open=0.92, look=(0.18, -0.08), brow_up=0.55, mouth=dict(open=0.38, round=0.92, width=0.92))),
    ("4_sassy_side_eh",   25.0,  0.0, -1.5, Face(lid_open=0.64, look=(0.32, 0.0), brow_up=0.25, mouth=dict(open=0.44, width=0.96, corner=-0.12, skew=0.25, teeth_up=0.50, tongue=0.35))),
    ("5_warm_smile",      0.0,   0.0,  0.0, Face(lid_open=0.84, lower=0.12, brow_up=0.20, mouth=dict(open=0.0, width=1.02, corner=0.38, skew=0.12))),
    ("6_playful_tongue",  25.0, -3.0,  1.5, Face(lid_open=0.90, look=(-0.12, 0.0), brow_up=0.40, mouth=dict(open=0.56, width=1.00, corner=0.30, teeth_up=0.50, tongue=0.85, tongue_up=0.95))),
    ("7_lip_bite_fv",     0.0,   2.0,  0.0, Face(lid_open=0.82, look=(0.15, 0.08), brow_worry=0.35, mouth=dict(open=0.20, width=0.98, fv=1.0, teeth_up=0.95))),
    ("8_clenched_grit",   25.0,  0.0,  0.0, Face(lid_open=0.80, lid_tilt=0.22, brow_anger=0.55, furrow=0.50, mouth=dict(open=0.32, width=1.12, teeth_up=0.85, teeth_lo=0.75))),
    ("9_smug_half_lid",   -25.0, 0.0, -2.0, Face(lid_open=0.52, lower=0.18, look=(0.30, -0.05), brow_up=0.30, mouth=dict(open=0.0, width=0.96, corner=0.32, skew=0.35))),
    ("10_cocked_talk",    25.0,  0.0,  1.0, Face(lid_open=0.66, look=(-0.18, 0.05), brow_up=0.50, mouth=dict(open=0.48, width=0.96, corner=0.15, skew=0.28, teeth_up=0.55, tongue=0.45))),
    ("11_happy_blink",    0.0,  -4.0,  1.5, Face(lid_open=0.0, brow_up=0.55, mouth=dict(open=0.72, width=1.14, corner=0.50, teeth_up=0.65, tongue=0.65))),
    ("12_fierce_anger_v", 25.0,  3.0,  1.0, Face(lid_open=0.72, lid_tilt=0.32, brow_anger=1.0, furrow=1.0, mouth=dict(open=0.62, width=1.04, corner=-0.30, teeth_up=0.60, tongue=0.55))),
    ("13_annoyed_sigh",   25.0, -2.0, -2.2, Face(lid_open=0.48, lower=0.25, look=(0.35, -0.55), brow_worry=0.60, mouth=dict(open=0.0, width=0.92, corner=-0.25, skew=0.22))),
    ("14_shocked_gasp",   0.0,  -5.0,  0.0, Face(lid_open=1.18, pupil=0.62, brow_up=0.95, mouth=dict(open=0.76, round=0.55, width=0.94, teeth_up=0.35, tongue=0.40))),
    ("15_sleepy_yawn",    -25.0, 5.0, -1.5, Face(lid_open=0.38, lower=0.22, look=(0.0, 0.20), brow_worry=0.50, bags=0.65, mouth=dict(open=0.78, round=0.65, width=0.92, tongue=0.50))),
    ("16_pleading_worry", 25.0,  4.0, -1.5, Face(lid_open=0.94, lid_tilt=-0.25, pupil=1.12, look=(-0.15, -0.12), brow_worry=0.95, furrow=0.45, mouth=dict(open=0.38, width=0.90, corner=-0.35, teeth_up=0.35, tongue=0.45))),
    ("17_determined_nod", -25.0, 4.0,  1.0, Face(lid_open=0.76, lid_tilt=0.26, brow_anger=0.85, furrow=0.80, mouth=dict(open=0.0, width=0.94, press=1.0, corner=-0.20))),
    ("18_cheerful_laugh", 25.0, -4.0,  2.0, Face(lid_open=0.92, lower=0.28, brow_up=0.65, mouth=dict(open=0.82, width=1.14, corner=0.55, teeth_up=0.70, tongue=0.70))),
]


def render_18_sheet(character: str, out_jpg: str, cols: int = 6) -> str:
    """Render all 18 expressions on `character` ('girl' or 'boy') into a 6x3 master showcase sheet."""
    ch = Character(character)
    tw, th = 320, 340
    tiles = []
    for label, yaw, pitch, rot, face_st in EXPRESSIONS_18:
        head = HeadPose(yaw=yaw, pitch=pitch, rot=rot, dy=0.0, breath=0.0)
        rgba = ch.render(yaw, face_st, head=head, pitch=pitch)
        # Composite onto warm studio card background
        H, W = rgba.shape[:2]
        bg = np.full((H, W, 3), (142, 168, 206), dtype=np.uint8)
        al = rgba[:, :, 3:4].astype(np.float32) / 255.0
        comp = (bg.astype(np.float32) * (1.0 - al) + rgba[:, :, :3].astype(np.float32) * al).astype(np.uint8)
        # Crop head box (y: 120..760, x: 110..750)
        crop = comp[120:760, 110:750]
        tile = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tw - 1, 32), (24, 20, 18), -1)
        tag = f"{label} ({yaw:+.0f}d)"
        cv2.putText(tile, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 248, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)]
    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_jpg)), exist_ok=True)
    cv2.imwrite(out_jpg, sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return out_jpg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp")
    a = ap.parse_args()
    for nm in ("girl", "boy"):
        p = os.path.join(a.out_dir, f"v313_{nm}_18_expressions.jpg")
        render_18_sheet(nm, p)
        print("[sheet25] saved", p)


if __name__ == "__main__":
    main()
