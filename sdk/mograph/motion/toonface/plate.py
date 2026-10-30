"""
toonface.plate — turn ANY flat-colour toon character image into a face rig.  No per-character templates.

    rig = analyze(img_bgr, style=None, search_box=None)   →  PlateRig(clean, fill, head, style, eyes, info)

Pipeline (eye-seeded, character-agnostic)
  1. eyes        white blobs (low saturation, very bright) that are compact AND ringed by ink → best horizontal pair
                 (one eye is accepted too: profile views)
  2. skin        colour sampled on the cheek just below the eye pair; face skin = the connected region of that colour
                 (hair of a similar cream colour is rejected by the tight colour distance + connectivity)
  3. face fill   skin closed + holes filled + eye blobs inside the skin hull (in 3/4 the far eye touches the contour)
  4. clean plate every interior feature (eyes, brows, nose, mouth) repainted with skin; ink that is connected to the
                 eye outlines but escapes the face (lash wings, eye outline merged with the contour) is inpainted and
                 the face contour is re-inked from the skin hull — hair curls over the forehead survive untouched
  5. head sphere Kåsa circle fit of the cheek / chin contour → cx, cy, R
  6. yaw         least-squares fit of the projected canonical eye pair to the detected eye centres (grid search),
                 roll from the eye line (damped)
  7. style       palette + proportions in head radii, lash detection (front-ish views only, or when no style given)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from .geom import HeadFrame, fit_circle
from .style import ToonStyle, bgr2rgb01


@dataclass
class PlateRig:
    clean: np.ndarray
    fill: np.ndarray                  # uint8 0/1 face area (skin + feature holes)
    head: HeadFrame
    style: ToonStyle
    eyes: list = field(default_factory=list)       # [(x, y, w, h, cx, cy)]
    info: dict = field(default_factory=dict)


def _white(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] < 28) & (hsv[:, :, 2] > 232)).astype(np.uint8)


def find_eyes(img: np.ndarray, region: np.ndarray) -> list:
    sm = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    wm = _white(sm) & region
    n, lbl, st, ce = cv2.connectedComponentsWithStats(wm, connectivity=4)
    area_r = float(region.sum())
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 40 or a > 0.06 * area_r:
            continue
        if a / float(w * h) < 0.42 or not (0.35 < h / float(w) < 3.2):
            continue
        blob = (lbl[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6] == i).astype(np.uint8)
        ring = cv2.dilate(blob, np.ones((7, 7), np.uint8)) - blob
        g = gray[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6]
        if ring.sum() == 0 or (g[ring > 0] < 130).mean() < 0.15:
            continue
        cands.append((int(x), int(y), int(w), int(h), float(ce[i][0]), float(ce[i][1]), int(a)))
    best, best_s = None, -1.0
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            mh, mw = max(a[3], b[3]), 0.5 * (a[2] + b[2])
            dy, dx = abs(a[5] - b[5]), abs(a[4] - b[4])
            if dy > 0.6 * mh or not (0.8 * mw < dx < 4.5 * mw):
                continue
            s = a[6] + b[6] - 4.0 * dy
            if s > best_s:
                best, best_s = (a, b), s
    if best:
        return sorted([e[:6] for e in best], key=lambda e: e[4])
    if cands:
        return [max(cands, key=lambda e: e[6])[:6]]
    return []


def _fit_yaw(eyes, cx, R, depth, eye_u=None, eye_v=0.0):
    xs = np.array([e[4] for e in eyes])
    us = np.arange(0.26, 0.60, 0.01) if eye_u is None else np.array([eye_u])
    best = (1e18, 0.0, us[0])
    for u in us:
        z = math.sqrt(max(0.0, 1 - u * u - eye_v * eye_v)) * depth
        for yaw in np.arange(-1.3, 1.3001, 0.01):
            cy_, sy_ = math.cos(yaw), math.sin(yaw)
            if len(xs) == 2:
                pred = np.array([cx + R * (-u * cy_ + z * sy_), cx + R * (u * cy_ + z * sy_)])
            else:                                    # single eye: assume it is the near eye
                side = 1 if yaw < 0 else -1
                pred = np.array([cx + R * (side * u * cy_ + z * sy_)]) if False else \
                    np.array([cx + R * ((u if xs[0] > cx else -u) * cy_ + z * sy_)])
            err = float(((pred - xs) ** 2).sum()) + (abs(yaw) * 2.0 if len(xs) == 2 else 0.0)
            if err < best[0]:
                best = (err, float(yaw), float(u))
    return best[1], best[2]


def analyze(img: np.ndarray, style: ToonStyle | None = None, search_box=None, measure_style: bool | None = None,
            skin_tol: float = 22.0) -> PlateRig:
    H, W = img.shape[:2]
    if search_box is None:
        search_box = (0, 0, W, int(H * 0.65))
    x0, y0, x1, y1 = [int(v) for v in search_box]
    region = np.zeros((H, W), np.uint8)
    region[y0:y1, x0:x1] = 1
    sm = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    eyes = find_eyes(img, region)
    if not eyes:
        raise ValueError("no eyes found in the search box")
    # ---- 2. skin from the cheek below the eyes
    ey_bot = max(e[1] + e[3] for e in eyes)
    eh = float(np.mean([e[3] for e in eyes]))
    sx = int(np.mean([e[4] for e in eyes]))
    sy = int(min(H - 1, ey_bot + 0.45 * eh))
    patch = sm[max(0, sy - 6):sy + 7, max(0, sx - 14):sx + 15].reshape(-1, 3)
    patch = patch[cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).ravel() > 150]
    if len(patch) == 0:
        raise ValueError("cheek sample failed")
    skin = np.median(patch, axis=0)
    dist = np.linalg.norm(sm.astype(np.float32) - skin[None, None].astype(np.float32), axis=2)
    skm = ((dist < skin_tol) & (region > 0)).astype(np.uint8)
    n, lbl = cv2.connectedComponents(skm, connectivity=4)
    ys_s, xs_s = np.where(skm[max(0, sy - 10):sy + 11, max(0, sx - 30):sx + 31] > 0)
    if len(ys_s) == 0:
        raise ValueError("skin component not found")
    k = np.argmin((ys_s - 10) ** 2 + (xs_s - 30) ** 2)
    face_skin = lbl == lbl[max(0, sy - 10) + ys_s[k], max(0, sx - 30) + xs_s[k]]
    # ---- 3. face fill
    sealed = cv2.morphologyEx(face_skin.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cs, _ = cv2.findContours(face_skin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull_pts = cv2.convexHull(max(cs, key=cv2.contourArea))
    hull = np.zeros((H, W), np.uint8)
    cv2.drawContours(hull, [hull_pts], -1, 1, -1)
    eye_m = np.zeros((H, W), np.uint8)
    for e in eyes:
        x, y, w, h = e[:4]
        cv2.ellipse(eye_m, (int(e[4]), int(e[5])), (int(w * 0.5) + 3, int(h * 0.5) + 3), 0, 0, 360, 1, -1)
    fill = ndi.binary_fill_holes((sealed > 0) | ((eye_m > 0) & (hull > 0)))
    holes = fill & ~face_skin
    # ---- 4. clean plate
    guard = cv2.dilate((~fill).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    erase = (cv2.dilate(holes.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & fill & ~guard
    clean = img.copy()
    clean[erase] = skin.astype(np.uint8)
    # ink escaping the face but connected to an eye outline (lash wings, merged outlines)
    dark = (gray < 125).astype(np.uint8)
    zone = np.zeros((H, W), np.uint8)
    for e in eyes:
        x, y, w, h = e[:4]
        cv2.rectangle(zone, (int(x - 0.5 * w), int(y - 0.6 * h)), (int(x + 1.5 * w), int(y + 1.15 * h)), 1, -1)
    eye_ring = cv2.dilate(eye_m, np.ones((5, 5), np.uint8))
    nd, ld = cv2.connectedComponents(dark & zone, connectivity=8)
    touch = np.unique(ld[(eye_ring > 0) & (ld > 0)])
    rem = np.isin(ld, touch[touch > 0]) & ~erase
    # never touch the face contour itself: protect a band around the face boundary
    th = max(2, int(round(_contour_thickness(gray, fill.astype(np.uint8)))))
    edge = cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    band = cv2.dilate(edge, np.ones((2 * th + 3, 2 * th + 3), np.uint8)) > 0
    rem &= ~band
    if rem.any():
        rem_d = cv2.dilate(rem.astype(np.uint8), np.ones((3, 3), np.uint8)) & (~band).astype(np.uint8)
        inside = (rem_d > 0) & fill
        clean[inside] = skin.astype(np.uint8)
        outside = ((rem_d > 0) & ~fill).astype(np.uint8)
        if outside.any():
            clean = cv2.inpaint(clean, outside, 4, cv2.INPAINT_TELEA)
    # ---- 5. head sphere
    cnt = hull_pts.reshape(-1, 2)
    ccs, _ = cv2.findContours(fill.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(ccs, key=cv2.contourArea).reshape(-1, 2)
    eye_y = float(np.mean([e[5] for e in eyes]))
    low = cnt[cnt[:, 1] > eye_y + 0.25 * (cnt[:, 1].max() - eye_y)]
    cx, cy, R = fit_circle(low if len(low) > 20 else cnt)
    fw = float(np.ptp(cnt[:, 0]))
    R = float(np.clip(R, 0.42 * fw, 0.72 * fw))
    st0 = style or ToonStyle()
    if measure_style is None:
        measure_style = style is None
    eye_v = (eye_y - cy) / R
    yaw, eye_u = _fit_yaw(eyes, cx, R, st0.face_depth, None if measure_style else st0.eye_u, eye_v)
    roll = 0.0
    if len(eyes) == 2:
        ang = math.atan2(eyes[1][5] - eyes[0][5], eyes[1][4] - eyes[0][4])
        roll = float(np.clip(0.5 * ang, -0.15, 0.15))
    info = {"skin_bgr": skin.tolist(), "yaw_deg": round(math.degrees(yaw), 1), "roll_deg": round(math.degrees(roll), 1),
            "R": round(R, 1)}
    st_out = st0.copy()
    if measure_style:
        st_out = _measure(img, gray, st0, eyes, holes, fill, skin, cx, cy, R, yaw, eye_u, eye_v, info)
    head = HeadFrame(cx, cy, R, yaw=yaw, roll=roll, depth=st_out.face_depth, cheat=st_out.cheat)
    return PlateRig(clean=clean, fill=fill.astype(np.uint8), head=head, style=st_out, eyes=eyes, info=info)


def _contour_thickness(gray, hull) -> float:
    edge = cv2.morphologyEx(hull, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    band = cv2.dilate(edge, np.ones((9, 9), np.uint8)) > 0
    d = (gray < 110) & band
    if d.sum() == 0:
        return 2.0
    dt = cv2.distanceTransform(d.astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.clip(2.0 * np.percentile(dt[d], 90), 1.5, 8.0))


def _measure(img, gray, st, eyes, holes, fill, skin, cx, cy, R, yaw, eye_u, eye_v, info):
    kw: dict = {"skin": bgr2rgb01(skin)}
    ink_px = img[(gray < 70) & (cv2.dilate(fill.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)]
    if len(ink_px):
        kw["ink"] = bgr2rgb01(np.median(ink_px, axis=0))
        kw["pupil"] = bgr2rgb01(np.median(ink_px, axis=0) * 0.85)
    ys_f, xs_f = np.where(fill)
    top = int(ys_f.min())
    band = img[max(0, top - int(0.45 * R)):top + 2, int(xs_f.min()):int(xs_f.max())].reshape(-1, 3)
    hsv = cv2.cvtColor(band.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hair = band[(hsv[:, 2] > 200) & (hsv[:, 1] < 100) & (np.linalg.norm(band.astype(float) - skin, axis=1) > 12)]
    if len(hair) > 40:
        kw["hair"] = bgr2rgb01(np.median(hair, axis=0))
    kw["line"] = float(np.clip(_contour_thickness(gray, fill.astype(np.uint8)) / R * 0.95, 0.018, 0.045))
    kw["line_fine"] = kw["line"] * 0.58
    kw["eye_v"] = float(np.clip(eye_v, -0.25, 0.25))
    if len(eyes) == 2 and abs(yaw) < 0.3:
        cosy = max(0.5, math.cos(yaw))
        kw["eye_u"] = float(np.clip(eye_u, 0.26, 0.58))
        kw["eye_rx"] = float(np.clip(0.5 * np.mean([e[2] for e in eyes]) / R / cosy + 0.012, 0.14, 0.44))
        kw["eye_ry"] = float(np.clip(0.5 * np.mean([e[3] for e in eyes]) / R + 0.012, 0.12, 0.44))
        lash = []
        for e in eyes:
            x, y, w, h = e[:4]
            roi = gray[max(0, y - int(0.4 * h)):y + int(0.1 * h), max(0, x - int(0.25 * w)):x + int(1.25 * w)]
            lash.append(float((roi < 90).mean()) if roi.size else 0.0)
        info["lash_score"] = round(float(np.mean(lash)), 3)
        kw["lashes"] = bool(np.mean(lash) > 0.20)
        hl, nh = ndi.label(holes)
        feats = []
        for i in range(1, nh + 1):
            m = hl == i
            if m.sum() < 20:
                continue
            ys, xs = np.where(m)
            feats.append((float(xs.mean()), float(ys.mean()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)))
        eye_y = cy + eye_v * R
        ehp = 0.5 * np.mean([e[3] for e in eyes])
        mid_x = 0.5 * (eyes[0][4] + eyes[1][4])
        brows = [f for f in feats if f[1] < eye_y - 1.05 * ehp and f[2] > 0.15 * R]
        below = [f for f in feats if f[1] > eye_y + 1.05 * ehp]
        if brows:
            kw["brow_v"] = float(np.clip((np.mean([b[1] for b in brows]) - cy) / R, -0.75, -0.15))
            kw["brow_thick"] = float(np.clip(np.mean([b[3] for b in brows]) / R * 0.6, 0.03, 0.08))
            kw["brow_len"] = float(np.clip(np.mean([b[2] for b in brows]) / R / (2 * kw["eye_rx"] * 0.55) * 0.95,
                                           0.8, 1.6))
        if below:
            mouth = max(below, key=lambda f: f[1])
            if mouth[2] > 0.12 * R:
                kw["mouth_v"] = float(np.clip((mouth[1] - cy) / R, 0.25, 0.75))
                kw["mouth_w"] = float(np.clip(0.5 * mouth[2] / R * 0.9, 0.12, 0.30))
            noses = [f for f in below if f is not mouth and abs(f[0] - mid_x) < 0.3 * R]
            if noses:
                nose = min(noses, key=lambda f: f[1])
                kw["nose_v"] = float(np.clip((nose[1] - cy) / R, 0.05, 0.45))
                kw["nose_size"] = float(np.clip(0.5 * max(nose[2], nose[3]) / R, 0.035, 0.09))
    info["measured"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()
                        if not isinstance(v, tuple)}
    return st.copy(**kw)


def mirror(rig: PlateRig) -> PlateRig:
    """Mirrored rig (3/4 facing the other way).  The vector face is REDRAWN for the new orientation, never flipped."""
    W = rig.clean.shape[1]
    h = rig.head
    eyes = sorted([(W - e[0] - e[2], e[1], e[2], e[3], W - 1 - e[4], e[5]) for e in rig.eyes], key=lambda e: e[4])
    return PlateRig(clean=rig.clean[:, ::-1].copy(), fill=rig.fill[:, ::-1].copy(),
                    head=h.replace(cx=W - 1 - h.cx, yaw=-h.yaw, roll=-h.roll),
                    style=rig.style.copy(nose_dir=-rig.style.nose_dir), eyes=eyes, info=dict(rig.info, mirrored=True))
