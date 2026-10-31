"""
toonface.kit.facefind — v33 skin-tone-agnostic, clutter-robust face finding + style measurement.

v32 assumed peach skin on a flat amber wall.  Production scenes have any skin colour (pale, brown, green orc, blue
alien) and busy backgrounds.  A blank cartoon face is still very recognisable by STRUCTURE, not colour:
    flat (texture-free) region · closed by dark line-art · egg-shaped (solid, aspect 0.7–1.5) · upper part of the
    frame · a textured ring around it (hair / ears).
find_face() scores every flat region on those cues (pattern from classical cartoon-face detection papers and the
SkyTNT/anime-segmentation idea of character-first parsing), so it works for any palette.  A hint (point or box)
picks one character in multi-character shots.

measure_style() reads the drawing's own style so new features match it automatically:
    skin colour · ink colour · outline weight (px) · hair colour (for brows) · face width/chin/centre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class FaceFind:
    mask: np.ndarray            # bool filled face mask
    skin_bgr: np.ndarray
    score: float
    bbox: tuple


@dataclass
class FaceStyle:
    skin_bgr: np.ndarray
    ink_bgr: np.ndarray
    hair_bgr: np.ndarray
    line_px: float              # outline stroke width in px
    cx: float                   # cheek-centre x
    chin_y: float
    top_y: float
    width: float                # widest row (px) = 1 face unit

    def unit_to_frame(self) -> np.ndarray:
        """2×3 affine  face units (x right, y UP from chin, 1 = face width) → frame px."""
        W = self.width
        return np.array([[W, 0.0, self.cx], [0.0, -W, self.chin_y]], np.float64)


def ink_mask(gray: np.ndarray, k: int = 11, contrast: float = 38.0) -> np.ndarray:
    """line-art by LOCAL contrast (morphological black-hat), not absolute darkness: works for dark-skinned
    characters (brown skin is darker than a fixed 'ink' threshold) and for coloured line-art."""
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return (bh > contrast) | (gray < 40)


def _flat_regions(img: np.ndarray, ink_thr: int):
    """label map of flat colour regions bounded by ink / strong edges (half-res for speed)."""
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.bilateralFilter(small, 7, 30, 7), cv2.COLOR_BGR2LAB)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1) ** 2
    for c in (1, 2):
        gx += cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1) ** 2
    edge = (np.sqrt(gx) > 40) | ink_mask(g, k=7)
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    n, lbl, st, ce = cv2.connectedComponentsWithStats((~edge).astype(np.uint8), connectivity=4)
    return small, lab, g, lbl, st, n


def find_face(img: np.ndarray, hint=None, ink_thr: int = 100, min_frac: float = 0.006,
              max_frac: float = 0.25) -> FaceFind:
    """hint: None | (x, y) point inside the face | (x0, y0, x1, y1) box."""
    H, W = img.shape[:2]
    small, lab, g, lbl, st, n = _flat_regions(img, ink_thr)
    h2, w2 = g.shape
    A = h2 * w2
    dark = ink_mask(g, k=7)
    best, bi = -1.0, -1
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_frac * A or a > max_frac * A:
            continue
        if hint is not None:
            hx = np.asarray(hint, float) * 0.5
            if len(hx) == 2:
                if lbl[int(np.clip(hx[1], 0, h2 - 1)), int(np.clip(hx[0], 0, w2 - 1))] != i:
                    continue
            else:
                cx, cy = x + w / 2, y + h / 2
                if not (hx[0] <= cx <= hx[2] and hx[1] <= cy <= hx[3]):
                    continue
        comp = (lbl[y:y + h, x:x + w] == i)
        filled = ndi.binary_fill_holes(comp)
        cs, _ = cv2.findContours(filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        ca = cv2.contourArea(c)
        sol = ca / max(1.0, cv2.contourArea(cv2.convexHull(c)))
        asp = w / max(1.0, h)
        # ink enclosure: fraction of the boundary ring that is dark line-art
        pad = 14
        y0, y1, x0, x1 = max(0, y - pad), min(h2, y + h + pad), max(0, x - pad), min(w2, x + w + pad)
        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        sub[y - y0:y - y0 + h, x - x0:x - x0 + w] = filled
        ring = (cv2.dilate(sub, np.ones((9, 9), np.uint8)) > 0) & (sub == 0)   # 4 px: past the AA edge band
        enc = float(dark[y0:y1, x0:x1][ring].mean()) if ring.any() else 0.0
        # flatness: low colour variance inside
        var = float(lab[y:y + h, x:x + w][comp].astype(np.float32).std(0).mean())
        # textured surroundings (hair / ears) just outside the region
        big = (cv2.dilate(sub, np.ones((25, 25), np.uint8)) > 0) & (sub == 0)
        tex = float(cv2.Laplacian(g[y0:y1, x0:x1], cv2.CV_32F)[big].std()) if big.any() else 0.0
        if enc < 0.30:                      # a face is drawn: it is closed by line-art
            continue
        # curved, not boxy: extent of an ellipse = 0.785, of a rectangle = 1.0; a polygon has few vertices
        extent = ca / max(1.0, float(w * h))
        nvert = len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True))
        curved = math.exp(-max(0.0, extent - 0.84) / 0.04) * min(1.0, nvert / 8.0)
        # neck / jaw cue: directly below the region's bottom-centre there is a narrower body part (not open space)
        bx = int(x + w / 2)
        by0, by1 = min(h2 - 1, y + h + 2), min(h2, y + h + max(6, h // 4))
        col = lbl[by0:by1, max(0, bx - w // 8):bx + w // 8 + 1]
        neck = 1.0 if col.size and (np.mean(col != lbl[min(h2 - 1, by1 - 1), 0]) > 0.5) else 0.6
        s = (min(sol, 0.98) ** 6) * math.exp(-((math.log(asp)) / 0.35) ** 2) * enc ** 2 * curved * neck \
            * math.exp(-var / 9.0) * (0.5 + min(tex, 40) / 40.0) * (1.0 - 0.5 * (y + h / 2) / h2) * math.sqrt(a / A)
        if s > best:
            best, bi = s, i
    if bi < 0:
        raise RuntimeError("find_face: no face-like region (pass hint=(x, y) on the face)")
    m = cv2.resize((lbl == bi).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    m = ndi.binary_fill_holes(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0)
    skin = np.median(img[cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0], axis=0).astype(np.float32)
    ys, xs = np.nonzero(m)
    return FaceFind(mask=m, skin_bgr=skin, score=float(best), bbox=(int(xs.min()), int(ys.min()), int(xs.max()),
                                                                     int(ys.max())))


def face_frame_metrics(mask: np.ndarray):
    rows = np.where(mask.any(1))[0]
    x0 = np.array([np.argmax(mask[y]) for y in rows])
    x1 = np.array([mask.shape[1] - 1 - np.argmax(mask[y, ::-1]) for y in rows])
    w = x1 - x0
    iy = int(np.argmax(w))
    band = slice(max(0, iy - len(rows) // 10), iy + len(rows) // 10 + 1)
    cx = float(np.median(0.5 * (x0[band] + x1[band])))
    return cx, float(rows[-1]), float(rows[0]), float(w.max())


def measure_style(img: np.ndarray, face_mask: np.ndarray, ink_thr: int = 100) -> FaceStyle:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = face_mask.astype(np.uint8)
    skin = np.median(img[cv2.erode(fm, np.ones((9, 9), np.uint8)) > 0], axis=0).astype(np.float32)
    ring = (cv2.dilate(fm, np.ones((9, 9), np.uint8)) > 0) & (fm == 0)
    inkm = ink_mask(g)
    dark = ring & inkm
    ink = np.median(img[dark], axis=0).astype(np.float32) if dark.sum() > 30 else np.array([18, 29, 53], np.float32)
    # outline weight: 2 × distance-transform peak of the dark band around the face
    band = (cv2.dilate(fm, np.ones((21, 21), np.uint8)) > 0) & inkm & (fm == 0)
    dt = cv2.distanceTransform(band.astype(np.uint8), cv2.DIST_L2, 3)
    line_px = float(np.clip(2.0 * np.percentile(dt[band], 85), 1.5, 14.0)) if band.any() else 4.0
    cx, chin, top, W = face_frame_metrics(face_mask)
    # hair: pixels just above the face that are neither skin nor ink nor background-like grey
    yy = np.arange(img.shape[0])[:, None]
    above = (cv2.dilate(fm, np.ones((61, 61), np.uint8)) > 0) & (fm == 0) & ~inkm & (yy < top + 0.35 * W)
    if above.sum() > 50:
        px = img[above].astype(np.float32)
        far = np.linalg.norm(px - skin, axis=1) > 25
        hair = np.median(px[far], axis=0) if far.sum() > 30 else np.median(px, axis=0)
    else:
        hair = np.array([200, 230, 250], np.float32)
    return FaceStyle(skin_bgr=skin, ink_bgr=ink, hair_bgr=hair.astype(np.float32), line_px=line_px, cx=cx,
                     chin_y=chin, top_y=top, width=W)


def skin_mask(img: np.ndarray, skin_bgr: np.ndarray, tol: float | None = None, ink_thr: int = 100) -> np.ndarray:
    """skin-coloured pixels (Lab distance, tolerance adapts to the skin's own brightness), ink excluded."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(skin_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    d = np.sqrt(0.35 * (lab[..., 0] - ref[0]) ** 2 + (lab[..., 1] - ref[1]) ** 2 + (lab[..., 2] - ref[2]) ** 2)
    tol = tol if tol is not None else 14.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = cv2.dilate(ink_mask(g).astype(np.uint8), np.ones((3, 3), np.uint8))
    return (d < tol) & (ink == 0)
