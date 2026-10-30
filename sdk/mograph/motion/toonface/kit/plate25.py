"""
toonface.kit.plate25 — clean plate for a flat 3/4 toon frame (no PSD needed).

Law 3 of the v31 master prompt:
  * far eye: sclera, pupil, lid wedge and lash wings are removed; pixels on the face side of the cheek contour become
    skin, outside become hair; the cheek contour is RE-STROKED (circle fitted on the pure cheek arc below the eye)
    with the character's own stroke width and ink colour → no white / ink remnant, contour stays continuous.
  * forelock: hair-coloured pixels (+ ink rim) hanging over the forehead → independent RGBA layer composited AFTER
    brows and eyes.  Taken from the CLEAN plate so the old brows never come back.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..geom import fit_circle

INK_BGR = (18, 29, 53)


@dataclass
class Plate25:
    src: np.ndarray
    clean: np.ndarray
    alpha: np.ndarray
    face: np.ndarray
    hair_fg: np.ndarray
    skin_bgr: np.ndarray
    hair_bgr: np.ndarray
    stroke_px: float
    eyes: list = field(default_factory=list)       # [near(x,y,w,h), far(x,y,w,h)]
    contour_far: np.ndarray | None = None
    cheek_circle: tuple = (0.0, 0.0, 0.0)
    info: dict = field(default_factory=dict)


def _white_blobs(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 235)).astype(np.uint8)


def _stroke_width(dark, near):
    d = dark & near
    if d.sum() < 30:
        return 6.0
    dt = cv2.distanceTransform(d.astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.clip(2.0 * np.percentile(dt[d], 85), 2.0, 16.0))


def analyze25(img: np.ndarray, skin_seed=None, far_side: int = +1) -> Plate25:
    H, W = img.shape[:2]
    sm = cv2.medianBlur(img, 3)
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    dark = g < 110
    # ---- matte
    n, lbl = cv2.connectedComponents((~dark).astype(np.uint8), connectivity=4)
    border = np.unique(np.r_[lbl[0], lbl[:, 0], lbl[:, -1]])
    bg = np.isin(lbl, border[border > 0])
    n, l2, st, _ = cv2.connectedComponentsWithStats((~bg).astype(np.uint8), connectivity=8)
    alpha = ndi.binary_fill_holes(l2 == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    # ---- eyes
    wb = _white_blobs(sm) & alpha.astype(np.uint8)
    n, lw, sw, _ = cv2.connectedComponentsWithStats(wb, connectivity=4)
    cands = sorted([(i, sw[i]) for i in range(1, n) if 300 < sw[i][4] < 0.05 * alpha.sum()], key=lambda c: -c[1][4])
    eyes = sorted([(int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(i)) for i, s in cands[:2]], key=lambda e: e[0])
    if len(eyes) < 2:
        raise ValueError("need both eyes for a 3/4 plate")
    near, far = (eyes[0], eyes[1]) if far_side > 0 else (eyes[1], eyes[0])
    # ---- skin
    if skin_seed is None:
        skin_seed = (int(near[0] + 0.6 * near[2]), int(near[1] + 1.45 * near[3]))
    sx, sy = skin_seed
    skin = np.median(sm[sy - 5:sy + 6, sx - 12:sx + 13].reshape(-1, 3), axis=0)
    ds = np.linalg.norm(sm.astype(np.float32) - skin.astype(np.float32), axis=2)
    n, lsk = cv2.connectedComponents((ds < 20).astype(np.uint8), connectivity=4)
    face_skin = lsk == lsk[sy, sx]
    ys, xs = np.where(face_skin)
    # ---- hair colour
    top = ys.min()
    band = sm[max(0, top - 60):top + 40, xs.min():xs.max()].reshape(-1, 3).astype(np.float32)
    hb = band[(np.linalg.norm(band - skin, axis=1) > 14) & (band.mean(1) > 185)]
    hair = np.median(hb, axis=0) if len(hb) > 50 else np.array([198, 232, 250], np.float32)
    hair_m = (np.linalg.norm(sm.astype(np.float32) - hair.astype(np.float32), axis=2) < 16) & alpha
    stroke = _stroke_width(dark, cv2.dilate(face_skin.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    th = int(round(stroke))
    # ---- brows: hair-coloured islands just above the near eye (they touch the hairline → not holes)
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(hair_m.astype(np.uint8), connectivity=4)
    brow_m = np.zeros((H, W), bool)
    for i in range(1, nb):
        x, y, w, h, a = sb[i]
        cyb = y + 0.5 * h
        if 150 < a < 0.2 * face_skin.sum() and near[1] - 0.8 * near[3] < cyb < near[1] and w > 0.35 * near[2]:
            brow_m |= lb == i
    brow_m = cv2.dilate(brow_m.astype(np.uint8), np.ones((2 * th + 5, 2 * th + 5), np.uint8)) > 0
    Fh = ndi.binary_fill_holes(face_skin) | (brow_m & alpha)
    holes = Fh & ~face_skin
    clean = img.copy()
    er = (cv2.dilate(holes.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    er &= cv2.dilate(Fh.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    er &= ~(hair_m & ~brow_m)                         # never paint skin over real hair
    clean[er] = skin.astype(np.uint8)
    inner = cv2.erode(Fh.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    clean[inner & (g > 150) & ~(hair_m & ~brow_m)] = skin.astype(np.uint8)   # flat cel → no upscale halos
    # ---- far eye
    fx, fy, fw, fh, fi = far
    sclera = (lw == fi).astype(np.uint8)
    zone = np.zeros((H, W), np.uint8)
    pad = int(0.55 * fw)
    x0 = fx - int(0.25 * fw) if far_side > 0 else fx - pad
    x1 = fx + fw + pad if far_side > 0 else fx + fw + int(0.25 * fw)
    cv2.rectangle(zone, (x0, fy - int(0.32 * fh)), (x1, fy + fh + int(0.12 * fh)), 1, -1)
    scl_d = cv2.dilate(sclera, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * th + 7, 2 * th + 7)))
    dist_s = cv2.distanceTransform((1 - sclera).astype(np.uint8), cv2.DIST_L2, 5)
    lashm = dark & (zone > 0) & (dist_s < 0.42 * fw)
    lash = ndi.binary_propagation(lashm & (scl_d > 0), mask=lashm)
    old_eye = cv2.dilate(((scl_d > 0) | lash).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # cheek arc below the far eye → circle → extrapolate through the eye zone
    rows = np.arange(fy + fh - int(0.05 * fh), min(H - 1, fy + fh + int(0.75 * fh)))
    pts = []
    for y in rows:
        r = np.where(face_skin[y])[0]
        if len(r):
            pts.append((float(r.max() if far_side > 0 else r.min()), float(y)))
    pts = np.array(pts)
    ccx, ccy, cR = fit_circle(pts)
    ys_c = np.arange(fy - int(0.30 * fh), rows[-1] + 1).astype(float)
    sgn = 1.0 if far_side > 0 else -1.0
    xs_c = ccx + sgn * np.sqrt(np.maximum(cR ** 2 - (ys_c - ccy) ** 2, 0.0))
    contour = np.c_[xs_c, ys_c]
    side = np.zeros((H, W), bool)
    for x, y in contour.astype(int):
        if 0 <= y < H:
            if far_side > 0:
                side[y, :max(0, x)] = True
            else:
                side[y, min(W, x):] = True
    side[:int(ys_c[0])] = False
    clean[old_eye & side] = skin.astype(np.uint8)
    clean[old_eye & ~side] = hair.astype(np.uint8)
    face = (Fh | (old_eye & side)) & alpha
    seg = contour[(contour[:, 1] >= fy + 0.05 * fh) & (contour[:, 1] <= fy + fh + 0.25 * fh)]
    tmp = clean.copy()
    if len(seg) > 2:
        cv2.polylines(tmp, [np.round(seg * 4).astype(np.int32).reshape(-1, 1, 2)], False, INK_BGR, max(2, th),
                      cv2.LINE_AA, shift=2)
    weld = cv2.dilate(old_eye.astype(np.uint8), np.ones((2 * th + 5, 2 * th + 5), np.uint8)) > 0
    clean[weld] = tmp[weld]
    # ---- forelock layer
    eye_top = min(near[1], far[1])
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    hullm = np.zeros((H, W), np.uint8)
    cv2.drawContours(hullm, [hull], -1, 1, -1)
    cut = eye_top + int(0.2 * near[3])
    fore = hair_m & (hullm > 0) & ~brow_m
    fore[cut:] = False
    fore = cv2.dilate(fore.astype(np.uint8), np.ones((2 * th + 3, 2 * th + 3), np.uint8)) > 0
    fore &= ~face_skin & (hullm > 0) & ~old_eye & ~brow_m
    fore[cut:] = False
    a = cv2.GaussianBlur(fore.astype(np.float32), (3, 3), 0)
    hair_fg = np.dstack([clean, (a * 255).astype(np.uint8)])
    info = {"skin_bgr": skin.round().tolist(), "hair_bgr": hair.round().tolist(), "stroke_px": round(stroke, 2),
            "near_eye": list(near[:4]), "far_eye": list(far[:4]),
            "cheek_circle": [round(float(v), 1) for v in (ccx, ccy, cR)]}
    return Plate25(src=img, clean=clean, alpha=alpha.astype(np.float32), face=face, hair_fg=hair_fg, skin_bgr=skin,
                   hair_bgr=hair, stroke_px=stroke, eyes=[near[:4], far[:4]], contour_far=seg,
                   cheek_circle=(ccx, ccy, cR), info=info)
