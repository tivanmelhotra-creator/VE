"""
toonface.compose — put a vector face on a character plate.

    frame = render_face(rig, controls, t=0.0, head_override=None)   → BGR image

Layers (bottom → top)
  1. rig.clean          the plate with its original drawn face erased
  2. face layer         draw_face() on a transparent cairo surface, then masked:
                          skin features (mouth / nose / cheeks)  → face fill mask
                          eyes + brows                          → face fill mask dilated (toon far-eye may kiss the
                                                                  contour)
  3. occluder           hair curls that hang over the forehead are copied back from the ORIGINAL plate on top, so the
                        brows tuck under the fringe exactly like in the drawing
  4. outer FX           sweat, tears, stress marks, bulb … never masked
Head acting (controls.head_yaw / pitch / roll) only rotates the FACE FRAME on the head sphere (the plate is static),
clamped to small angles so features never slide off the skull.
"""
from __future__ import annotations

import math

import cairo
import cv2
import numpy as np

from .controls import FaceControls
from .face import draw_face, _outer_fx
from .geom import HeadFrame
from .plate import PlateRig


def _surface_to_bgra(surf: cairo.ImageSurface) -> np.ndarray:
    surf.flush()
    h, w = surf.get_height(), surf.get_width()
    buf = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    return buf.copy()                     # premultiplied BGRA


def _over(dst_bgr: np.ndarray, src_pm_bgra: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    a = src_pm_bgra[:, :, 3:4].astype(np.float32) / 255.0
    rgb = src_pm_bgra[:, :, :3].astype(np.float32)
    if mask is not None:
        m = mask.astype(np.float32)[:, :, None]
        a = a * m
        rgb = rgb * m
    out = dst_bgr.astype(np.float32) * (1.0 - a) + rgb
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def acting_head(rig: PlateRig, c: FaceControls, max_yaw_deg: float = 9.0, max_pitch_deg: float = 7.0,
                max_roll_deg: float = 6.0) -> HeadFrame:
    h = rig.head
    dy = math.radians(float(np.clip(c.head_yaw, -max_yaw_deg, max_yaw_deg)))
    dp = math.radians(float(np.clip(c.head_pitch, -max_pitch_deg, max_pitch_deg)))
    dr = math.radians(float(np.clip(c.head_roll, -max_roll_deg, max_roll_deg))) * 0.5
    return h.replace(yaw=h.yaw + dy, pitch=h.pitch + dp, roll=h.roll + dr)


class FaceCompositor:
    """Caches the masks of one rig so per-frame rendering is cheap."""

    def __init__(self, rig: PlateRig, original: np.ndarray | None = None, supersample: int = 1):
        self.rig = rig
        H, W = rig.clean.shape[:2]
        self.H, self.W = H, W
        fill = (rig.fill > 0).astype(np.uint8)
        k = max(3, int(round(rig.head.R * 0.02)) | 1)
        # skin mask: shrink by ~the contour thickness so nothing is drawn on top of the outline ink
        inner = cv2.erode(fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        self.skin_mask = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), 0.7)
        self.eye_mask = cv2.GaussianBlur(fill.astype(np.float32), (0, 0), 0.7)
        self.occ = None
        if original is not None and rig.occluder is not None and rig.occluder.any():
            occ = rig.occluder.astype(np.float32)
            self.occ = cv2.GaussianBlur(occ, (0, 0), 0.6)[:, :, None]
            self.orig = original

    def render(self, c: FaceControls, t: float = 0.0, head: HeadFrame | None = None,
               base: np.ndarray | None = None) -> np.ndarray:
        rig = self.rig
        head = head or acting_head(rig, c)
        W, H = self.W, self.H
        # pass 1: skin features (mouth, nose, cheek fx) / pass 2: eyes + brows, no outer fx
        s1 = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        ctx = cairo.Context(s1)
        from .face import _cheek_fx, draw_nose, draw_mouth, draw_eyes, draw_brows
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        dv_e, dv_m = -0.04 * c.squash, 0.05 * c.squash
        _cheek_fx(ctx, head, rig.style, c, t)
        draw_nose(ctx, head, rig.style, c)
        draw_mouth(ctx, head, rig.style, c, dv_m)
        s2 = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        ctx2 = cairo.Context(s2)
        ctx2.set_antialias(cairo.ANTIALIAS_BEST)
        draw_eyes(ctx2, head, rig.style, c, t, dv_e)
        draw_brows(ctx2, head, rig.style, c, dv_e)
        s3 = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        ctx3 = cairo.Context(s3)
        ctx3.set_antialias(cairo.ANTIALIAS_BEST)
        _outer_fx(ctx3, head, rig.style, c, t)
        out = (base if base is not None else rig.clean).copy()
        out = _over(out, _surface_to_bgra(s1), self.skin_mask)
        out = _over(out, _surface_to_bgra(s2), self.eye_mask)
        if self.occ is not None:
            out = (out.astype(np.float32) * (1 - self.occ) + self.orig.astype(np.float32) * self.occ).astype(np.uint8)
        out = _over(out, _surface_to_bgra(s3))
        return out


def render_face(rig: PlateRig, c: FaceControls, t: float = 0.0, original: np.ndarray | None = None) -> np.ndarray:
    return FaceCompositor(rig, original).render(c, t)
