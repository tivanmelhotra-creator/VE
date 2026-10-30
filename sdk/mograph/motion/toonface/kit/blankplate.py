"""
toonface.kit.blankplate — layers from an art-team BLANK-FACE plate (eyes/brows/nose/mouth already removed by the
image-edit model).  No pixel surgery on the face: we only split it into layers.

    bp = load_blank(img_bgr, skin_seed)
      bp.base      BGR plate (body + blank face)
      bp.face      bool, face skin region (filled)  → the vector face is clipped to it (+ a small cheat margin)
      bp.hair_fg   BGRA forelock layer: hair-coloured pixels + their ink rim that overlap the face hull, composited
                   AFTER brows / eyes so brows always slide UNDER the curls
      bp.alpha     float matte of the character (border-flooded paper removed)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class BlankPlate:
    base: np.ndarray
    face: np.ndarray
    hair_fg: np.ndarray
    alpha: np.ndarray
    skin_bgr: np.ndarray


def load_blank(img: np.ndarray, skin_seed=(470, 560), tol: float = 22.0, contour_cut_y: int | None = None) -> BlankPlate:
    """contour_cut_y: below this row the face contour is NOT part of the forelock layer (far eye may spill past it)."""
    H, W = img.shape[:2]
    sm = cv2.medianBlur(img, 3)
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    dark = g < 110
    n, lbl = cv2.connectedComponents((~dark).astype(np.uint8), connectivity=4)
    border = np.unique(np.r_[lbl[0], lbl[:, 0], lbl[:, -1]])
    bg = np.isin(lbl, border[border > 0])
    n, l2, st, _ = cv2.connectedComponentsWithStats((~bg).astype(np.uint8), connectivity=8)
    alpha = ndi.binary_fill_holes(l2 == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    sx, sy = skin_seed
    skin = np.median(sm[sy - 6:sy + 7, sx - 12:sx + 13].reshape(-1, 3), axis=0)
    ds = np.linalg.norm(sm.astype(np.float32) - skin.astype(np.float32), axis=2)
    n, ls = cv2.connectedComponents((ds < tol).astype(np.uint8), connectivity=4)
    face = ndi.binary_fill_holes(ls == ls[sy, sx])
    # stroke width of the face contour
    edge = cv2.morphologyEx(face.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    near = cv2.dilate(edge, np.ones((17, 17), np.uint8)) > 0
    dt = cv2.distanceTransform((dark & near).astype(np.uint8), cv2.DIST_L2, 3)
    th = int(round(float(np.clip(2.0 * np.percentile(dt[dark & near], 85), 2, 16))))
    # forelock: non-skin, non-paper pixels inside the face's convex hull, in the upper half of the face
    ys, xs = np.where(face)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    hm = np.zeros((H, W), np.uint8)
    cv2.drawContours(hm, [hull], -1, 1, -1)
    hm = cv2.dilate(hm, np.ones((2 * th + 3, 2 * th + 3), np.uint8)) > 0
    ymid = int(ys.min() + 0.55 * (ys.max() - ys.min()))
    fore = hm & ~face & alpha
    fore[ymid:] = False
    # keep only the parts connected to the hair mass ABOVE the face (not the ear / the cheek contour)
    contour_band = cv2.dilate(edge, np.ones((2 * th + 3, 2 * th + 3), np.uint8)) > 0
    light = (g > 170) & fore
    nl, ll, sl, _ = cv2.connectedComponentsWithStats(light.astype(np.uint8), connectivity=4)
    keep = np.zeros((H, W), bool)
    for i in range(1, nl):
        if sl[i, cv2.CC_STAT_AREA] > 60:
            keep |= ll == i
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((2 * th + 3, 2 * th + 3), np.uint8)) > 0
    fore = keep & (hm) & alpha & ~(face & ~contour_band)
    fore[ymid:] = False
    if contour_cut_y is not None:
        fore[contour_cut_y:] &= ~contour_band[contour_cut_y:]
        fore[contour_cut_y:] &= ~(g[contour_cut_y:] < 110)
    a = cv2.GaussianBlur(fore.astype(np.float32), (3, 3), 0)
    hair_fg = np.dstack([img, (np.clip(a, 0, 1) * 255).astype(np.uint8)])
    return BlankPlate(base=img, face=face, hair_fg=hair_fg, alpha=alpha.astype(np.float32), skin_bgr=skin)
