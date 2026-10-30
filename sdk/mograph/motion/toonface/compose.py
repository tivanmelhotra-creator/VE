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
    """Caches the masks of one rig; per frame only a head-sized ROI is drawn (fast)."""

    def __init__(self, rig: PlateRig, original: np.ndarray | None = None):
        self.rig = rig
        H, W = rig.clean.shape[:2]
        self.H, self.W = H, W
        R = rig.head.R
        fill = (rig.fill > 0).astype(np.uint8)
        k = max(3, int(round(R * 0.03)) | 1)
        inner = cv2.erode(fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        self.skin_mask = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), 0.8)
        # toon cheat: eyes may spill past the cheek contour onto the hair (never onto the background far away)
        ke = max(5, int(round(R * 0.35)) | 1)
        self.eye_mask = cv2.GaussianBlur(cv2.dilate(fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke)))
                                         .astype(np.float32), (0, 0), 0.8)
        self.occ = None
        if original is not None and rig.occluder is not None and rig.occluder.any():
            self.occ = cv2.GaussianBlur(rig.occluder.astype(np.float32), (0, 0), 0.6)[:, :, None]
            self.orig = original
        h = rig.head
        m = 1.75 * R
        self.roi = (max(0, int(h.cx - m)), max(0, int(h.cy - m)), min(W, int(h.cx + m)), min(H, int(h.cy + m)))

    def _layer(self, draw, x0, y0, w, h) -> np.ndarray:
        s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(s)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.translate(-x0, -y0)
        draw(ctx)
        return _surface_to_bgra(s)

    def render(self, c: FaceControls, t: float = 0.0, head: HeadFrame | None = None,
               base: np.ndarray | None = None) -> np.ndarray:
        from .face import _cheek_fx, draw_nose, draw_mouth, draw_eyes, draw_brows
        rig, st = self.rig, self.rig.style
        head = head or acting_head(rig, c)
        x0, y0, x1, y1 = self.roi
        w, h = x1 - x0, y1 - y0
        dv_e, dv_m = -0.04 * c.squash, 0.05 * c.squash

        def skin(ctx):
            _cheek_fx(ctx, head, st, c, t)
            draw_nose(ctx, head, st, c)
            draw_mouth(ctx, head, st, c, dv_m)

        def eyes(ctx):
            draw_eyes(ctx, head, st, c, t, dv_e)
            draw_brows(ctx, head, st, c, dv_e)

        out = (base if base is not None else rig.clean).copy()
        sub = out[y0:y1, x0:x1]
        sub = _over(sub, self._layer(skin, x0, y0, w, h), self.skin_mask[y0:y1, x0:x1])
        sub = _over(sub, self._layer(eyes, x0, y0, w, h), self.eye_mask[y0:y1, x0:x1])
        if self.occ is not None:
            o = self.occ[y0:y1, x0:x1]
            sub = (sub.astype(np.float32) * (1 - o) + self.orig[y0:y1, x0:x1].astype(np.float32) * o).astype(np.uint8)
        out[y0:y1, x0:x1] = sub
        # outer FX may leave the ROI (bulb, zzz, tears) → full-frame transparent layer, cheap when empty
        if c.fx:
            out = _over(out, self._layer(lambda ctx: _outer_fx(ctx, head, st, c, t), 0, 0, self.W, self.H))
        return out


def render_face(rig: PlateRig, c: FaceControls, t: float = 0.0, original: np.ndarray | None = None) -> np.ndarray:
    return FaceCompositor(rig, original).render(c, t)
