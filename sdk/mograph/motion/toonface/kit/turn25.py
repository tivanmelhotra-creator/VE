"""
toonface.kit.turn25 — 2.5D head-turn compositor: blank plates (front / 3/4 / mirrored 3/4) + rig25 face, all locked
to ONE head anchor frame.

    ch = Character("girl")                       # loads kit + layered plates
    rgba = ch.render(yaw, face, head)            # plate-space RGBA (H×W×4), face already composited + forelock on top

Head anchor lock: the vector face is drawn in PLATE coordinates, composited onto the plate, and only THEN the
head transform (roll / nod / bob / breathing) is applied to the whole image → the face can never float or jitter
relative to the skull.  Gaze changes move pupils/lids only.

Turn: the face features morph continuously (fixed topology).  The plate drawing swaps front ↔ 3/4 at the middle of
the turn (Moho "switch layer"); before/after the swap each plate's head region is slid toward the other key's head
position so the skull never pops, and the face morph tracks it exactly.
"""
from __future__ import annotations

import math
import os

import cairo
import cv2
import numpy as np

from .blankplate import load_blank
from .rig25 import YAW_KEY, Face, Kit, draw_face

HERE = os.path.dirname(os.path.abspath(__file__))
PLATES = os.path.normpath(os.path.join(HERE, "../../../../tests/fixtures/toonface/plates"))


def _surface_bgra(w, h, fn):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    fn(ctx)
    surf.flush()
    a = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    return a.copy()


def _over_pm(dst, src):
    a = src[:, :, 3:4].astype(np.float32) / 255.0
    return (dst.astype(np.float32) * (1 - a) + src[:, :, :3].astype(np.float32)).clip(0, 255).astype(np.uint8)


def _over(dst, bgra):
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    return (dst.astype(np.float32) * (1 - a) + bgra[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def head_slide(img, dx, y_head_end=700, y_body=800):
    """shift the head region horizontally by dx with a smooth falloff over the neck (body stays)."""
    if abs(dx) < 0.05:
        return img
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    w = np.clip((y_body - yy) / float(y_body - y_head_end), 0, 1)
    w = w * w * (3 - 2 * w)
    return cv2.remap(img, xx - dx * w, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def head_warp(img, pivot, rot_deg, dy, breath, y_head_end=660, y_body=790):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    w = np.clip((y_body - yy) / float(y_body - y_head_end), 0, 1)
    w = w * w * (3 - 2 * w)
    a = -math.radians(rot_deg) * w
    px, py = pivot
    ca, sa = np.cos(a), np.sin(a)
    X = xx - px
    Y = yy - py - dy * w
    sx = px + ca * X - sa * Y
    sy = py + sa * X + ca * Y
    s = 1.0 + 0.005 * breath
    tb = 1 - w
    sy = sy * (1 - tb) + (H - (H - sy) / s) * tb
    return cv2.remap(img, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


class Character:
    def __init__(self, name: str):
        self.kit = Kit(name)
        d = self.kit.d
        self.layers = {}
        for v in ("front", "34"):
            img = cv2.imread(os.path.join(PLATES, d["plates"][v]))
            bp = load_blank(img, skin_seed=tuple(d["skin_seed"][v]), contour_cut_y=d["contour_cut_y"][v])
            self.layers[v] = bp
        q = self.layers["34"]
        W = q.base.shape[1]
        from .blankplate import BlankPlate
        self.layers["34m"] = BlankPlate(base=q.base[:, ::-1].copy(), face=q.face[:, ::-1].copy(),
                                        hair_fg=q.hair_fg[:, ::-1].copy(), alpha=q.alpha[:, ::-1].copy(),
                                        skin_bgr=q.skin_bgr)
        self.H, self.W = q.base.shape[:2]
        cf = np.array(d["chin"]["front"], float)
        cq = np.array(d["chin"]["34"], float)
        self.chin_dx = float(cq[0] - cf[0])                 # head offset of the 3/4 drawing vs the front drawing
        self.pivot = (float(np.mean([cf[0], cq[0]])), 735.0)

    def plate_for(self, yaw):
        t = min(1.0, abs(yaw) / YAW_KEY)
        sgn = 1.0 if yaw >= 0 else -1.0
        key = "front" if t < 0.5 else ("34" if sgn > 0 else "34m")
        bp = self.layers[key]
        # slide the chosen drawing's head toward the interpolated head position
        if key == "front":
            dx = sgn * self.chin_dx * t
        else:
            dx = -sgn * self.chin_dx * (1 - t)
        return bp, dx, t

    def render(self, yaw: float, face: Face, head=None):
        view, t = self.kit.view(yaw)
        bp, dx, t = self.plate_for(yaw)
        # face is drawn in the interpolated geometry; plate head slid to match → identical anchor
        fl = _surface_bgra(self.W, self.H, lambda ctx: draw_face(ctx, self.kit, view, t, face,
                                                                parts=("brows", "eyes", "nose")))
        ml = _surface_bgra(self.W, self.H, lambda ctx: draw_face(ctx, self.kit, view, t, face, parts=("mouth",)))
        fm = head_slide(bp.face.astype(np.uint8) * 255, dx)
        fm = cv2.erode(fm, np.ones((int(self.kit.sw) + 1, int(self.kit.sw) + 1), np.uint8))
        fm = cv2.GaussianBlur(fm, (3, 3), 0).astype(np.float32)[:, :, None] / 255.0
        ml = (ml.astype(np.float32) * fm).astype(np.uint8)          # premultiplied → scale all 4 channels
        base = head_slide(bp.base, dx)
        hf = head_slide(bp.hair_fg, dx)
        al = head_slide((bp.alpha * 255).astype(np.uint8), dx)
        out = _over_pm(base, fl)
        out = _over_pm(out, ml)
        out = _over(out, hf)
        rgba = np.dstack([out, cv2.GaussianBlur(al, (3, 3), 0)])
        if head is not None:
            rgba = head_warp(rgba, self.pivot, head.rot, head.dy, head.breath)
        return rgba
