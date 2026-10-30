"""quick still sheet of kit states over the blank plate (visual QC before animating)."""
from __future__ import annotations

import cairo
import cv2
import numpy as np

from .blankplate import load_blank
from .draw25 import FaceState, draw_face
from .visemes import vis, mix


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
