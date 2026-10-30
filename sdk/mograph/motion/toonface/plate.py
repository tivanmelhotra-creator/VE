"""
toonface.plate — turn ANY flat-colour toon character image into a rig: no per-character templates.

    rig = analyze(img_bgr, style=None)      →  PlateRig(clean, face_mask, head, style, ...)

1. skin colour       = dominant warm mid-tone inside the upper half of the character (mode of a coarse histogram)
2. face_skin         = largest connected component of skin-coloured pixels
3. face_fill         = face_skin with its interior holes filled (eyes, brows, nose, mouth are holes in the skin)
4. clean plate       = every hole pixel repainted with skin (feathered), so the original drawn features vanish
                       while hair curls overlapping the forehead (connected to the OUTSIDE, not holes) survive
5. head sphere       = Kåsa circle fit to the cheek / chin contour (lower half of face_fill) → cx, cy, R
6. yaw               = horizontal offset of the eye-pair midpoint from the sphere centre, asin(dx / R·depth);
                       roll = eye-line angle
7. style             = measured eye spacing / size, brow height, nose height, mouth height & width, lash
                       detection (dark mass above the eyes), palette (skin, hair, ink, pupil) — in head radii

Everything later draws with HeadFrame + ToonStyle, so a profile, a mirrored 3/4 or a brand-new character uses the
same expression and viseme library.
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
    clean: np.ndarray                     # BGR plate with the original face features erased
    face_mask: np.ndarray                 # uint8 0/255, face skin incl. feature holes
    head: HeadFrame
    style: ToonStyle
    eyes: list = field(default_factory=list)     # [(x, y, w, h)] detected scleras (screen)
    info: dict = field(default_factory=dict)


def _skin_color(img: np.ndarray, region: np.ndarray) -> np.ndarray:
    px = img[region > 0].reshape(-1, 3).astype(int)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ok = (hsv[:, 1] > 35) & (hsv[:, 1] < 120) & (hsv[:, 2] > 170)          # warm, mid-saturated, light
    px = px[ok]
    if len(px) == 0:
        raise ValueError("no skin-like colour found")
    q = (px // 8)
    keys, counts = np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2], return_counts=True)
    k = keys[np.argmax(counts)]
    sel = (q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]) == k
    return np.median(px[sel], axis=0)


def _color_mask(img: np.ndarray, col: np.ndarray, tol: float) -> np.ndarray:
    d = np.linalg.norm(img.astype(np.float32) - col[None, None, :].astype(np.float32), axis=2)
    return (d < tol).astype(np.uint8)


def analyze(img: np.ndarray, style: ToonStyle | None = None, skin_bgr=None, search_box=None,
            mirror_style: bool = False, measure_style: bool | None = None) -> PlateRig:
    """search_box = (x0, y0, x1, y1) restricts the face search (useful for multi-character frames)."""
    H, W = img.shape[:2]
    region = np.zeros((H, W), np.uint8)
    if search_box is None:
        search_box = (0, 0, W, int(H * 0.6))
    x0, y0, x1, y1 = search_box
    region[y0:y1, x0:x1] = 1
    skin = np.asarray(skin_bgr, float) if skin_bgr is not None else _skin_color(img, region)
    sk = _color_mask(img, skin, 26.0) & region
    sk = cv2.morphologyEx(sk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(sk, connectivity=4)
    if n < 2:
        raise ValueError("face skin not found")
    # prefer big components that contain holes (eyes); the neck has none
    best, best_score = 1, -1.0
    for i in range(1, n):
        a = st[i, 4]
        if a < 400:
            continue
        comp = lbl == i
        holes = ndi.binary_fill_holes(comp) & ~comp
        score = a + 40.0 * holes.sum()
        if score > best_score:
            best, best_score = i, score
    face_skin = (lbl == best)
    # seal tiny gaps at the contour (far eye / lashes touching the outline) before filling holes
    sealed = cv2.morphologyEx(face_skin.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    fill = ndi.binary_fill_holes(sealed)
    holes = fill & ~face_skin
    # ---------------- features (holes) ----------------
    hl, nh = ndi.label(holes)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 225)
    eyes = []
    for i in range(1, nh + 1):
        m = hl == i
        if m.sum() < 60:
            continue
        wf = (white & m).sum() / m.sum()
        ys, xs = np.where(m)
        if wf > 0.30:
            eyes.append((int(xs.min()), int(ys.min()), int(xs.ptp() + 1), int(ys.ptp() + 1), float(xs.mean()),
                         float(ys.mean())))
    eyes = sorted(eyes, key=lambda e: -e[2] * e[3])[:2]
    eyes = sorted(eyes, key=lambda e: e[4])
    # ---------------- clean plate ----------------
    erase = cv2.dilate(holes.astype(np.uint8), np.ones((3, 3), np.uint8)) & fill.astype(np.uint8)
    # keep the outer contour ink: never erase within 2 px of the outside
    outside_guard = cv2.dilate((~fill).astype(np.uint8), np.ones((5, 5), np.uint8))
    erase = erase & (1 - outside_guard)
    clean = img.copy()
    clean[erase > 0] = skin.astype(np.uint8)
    soft = cv2.GaussianBlur(erase.astype(np.float32), (0, 0), 1.0)[:, :, None]
    clean = (clean.astype(np.float32) * (1 - soft * 0.0) + 0).astype(np.uint8)
    # ---------------- head sphere ----------------
    face_u8 = fill.astype(np.uint8) * 255
    cs, _ = cv2.findContours(face_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cs, key=cv2.contourArea).reshape(-1, 2)
    eye_y = float(np.mean([e[5] for e in eyes])) if eyes else float(cnt[:, 1].mean())
    low = cnt[cnt[:, 1] > eye_y + 0.15 * (cnt[:, 1].max() - eye_y)]
    cx, cy, R = fit_circle(low if len(low) > 20 else cnt)
    ys_f, xs_f = np.where(fill)
    R = float(np.clip(R, 0.40 * xs_f.ptp(), 0.75 * xs_f.ptp()))
    st0 = style or ToonStyle()
    roll = 0.0
    yaw = 0.0
    if len(eyes) == 2:
        (ax, ay), (bx, by) = (eyes[0][4], eyes[0][5]), (eyes[1][4], eyes[1][5])
        roll = math.atan2(by - ay, bx - ax)
        emx = 0.5 * (ax + bx)
        # weight by eye size: the far eye is smaller → its centre is closer to the contour
        wa, wb = eyes[0][2], eyes[1][2]
        ratio = min(wa, wb) / max(wa, wb)
        dx = (emx - cx) / (R * st0.face_depth)
        yaw = math.asin(float(np.clip(dx * 1.25, -0.95, 0.95)))
        if ratio > 0.86:
            yaw *= 0.5 * (1 - (ratio - 0.86) / 0.14) + 0.0 if abs(dx) < 0.05 else 1.0
    elif len(eyes) == 1:
        dx = (eyes[0][4] - cx) / R
        yaw = math.asin(float(np.clip(dx * 1.1, -0.95, 0.95)))
    if measure_style is None:
        measure_style = style is None
    # ---------------- style measurement (front-ish views) ----------------
    st_out = st0.copy()
    info: dict = {"skin_bgr": skin.tolist(), "yaw_deg": math.degrees(yaw), "roll_deg": math.degrees(roll)}
    if measure_style:
        st_out = _measure_style(img, st_out, eyes, holes, hl, fill, face_skin, skin, cx, cy, R, yaw, roll, info)
    # place the sphere centre so the canonical eye line projects onto the drawn eye line
    head = HeadFrame(cx, cy, R, yaw=yaw, pitch=0.0, roll=roll, depth=st_out.face_depth, cheat=st_out.cheat)
    if eyes:
        tgt_y = float(np.mean([e[5] for e in eyes]))
        p, _ = head.wrap(np.array([[0.0, st_out.eye_v]]))
        head = head.replace(cy=cy + (tgt_y - p[0, 1]))
        if len(eyes) == 2:
            pm, _ = head.wrap(np.array([[0.0, st_out.eye_v]]))
            emx = 0.5 * (eyes[0][4] + eyes[1][4])
            head = head.replace(cx=head.cx + 0.6 * (emx - pm[0, 0]) * (1 if abs(yaw) > 0.1 else 0.0))
    return PlateRig(clean=clean, face_mask=face_u8, head=head, style=st_out, eyes=eyes, info=info)


def _measure_style(img, st: ToonStyle, eyes, holes, hl, fill, face_skin, skin, cx, cy, R, yaw, roll, info):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink_px = img[(gray < 70) & (cv2.dilate(fill.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)]
    kw = {}
    if len(ink_px):
        kw["ink"] = bgr2rgb01(np.median(ink_px, axis=0))
        kw["pupil"] = bgr2rgb01(np.median(ink_px, axis=0) * 0.9)
    kw["skin"] = bgr2rgb01(skin)
    # hair: light cream pixels directly above the face
    ys_f, xs_f = np.where(fill)
    top = int(ys_f.min())
    band = img[max(0, top - int(0.5 * R)):top, int(xs_f.min()):int(xs_f.max())].reshape(-1, 3)
    hsv = cv2.cvtColor(band.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hair = band[(hsv[:, 2] > 200) & (hsv[:, 1] < 90) & (np.linalg.norm(band.astype(float) - skin, axis=1) > 14)]
    if len(hair) > 50:
        kw["hair"] = bgr2rgb01(np.median(hair, axis=0))
    cos_y = max(0.4, math.cos(yaw))
    if len(eyes) == 2 and abs(yaw) < 0.25:
        e0, e1 = eyes
        kw["eye_u"] = float(np.clip(0.5 * (e1[4] - e0[4]) / R / cos_y, 0.25, 0.48))
        eh = 0.5 * float(np.mean([e0[3], e1[3]])) / R
        ew = 0.5 * float(np.mean([e0[2], e1[2]])) / R / cos_y
        kw["eye_rx"] = float(np.clip(ew * 1.0, 0.15, 0.32))
        kw["eye_ry"] = float(np.clip(eh * 1.0, 0.13, 0.32))
        eye_y = 0.5 * (e0[5] + e1[5])
        # lashes = lots of ink in a band just above the scleras
        lash = 0
        for e in eyes:
            x, y, w, h = e[:4]
            y0 = max(0, y - int(0.35 * h))
            roi = gray[y0:y + int(0.15 * h), max(0, x - int(0.2 * w)):x + int(1.2 * w)]
            lash += float((roi < 80).mean())
        kw["lashes"] = bool(lash / 2 > 0.22)
        info["lash_score"] = lash / 2
        # other holes: brows (above eyes), nose / mouth (below)
        feats = []
        for i in range(1, hl.max() + 1):
            m = hl == i
            if m.sum() < 25:
                continue
            ys, xs = np.where(m)
            feats.append((float(xs.mean()), float(ys.mean()), int(xs.ptp() + 1), int(ys.ptp() + 1), m.sum()))
        brows = [f for f in feats if f[1] < eye_y - 0.5 * eh * R and f[2] > 0.15 * R]
        below = [f for f in feats if f[1] > eye_y + 0.6 * eh * R]
        mid_x = 0.5 * (e0[4] + e1[4])
        if brows:
            kw["brow_v"] = float(np.clip(st.eye_v + (np.mean([b[1] for b in brows]) - eye_y) / R, -0.6, -0.2))
            kw["brow_thick"] = float(np.clip(np.mean([b[3] for b in brows]) / R * 0.55, 0.03, 0.08))
        if below:
            mouth = max(below, key=lambda f: f[1])
            if mouth[2] > 0.12 * R:
                kw["mouth_v"] = float(np.clip(st.eye_v + (mouth[1] - eye_y) / R, 0.35, 0.7))
                kw["mouth_w"] = float(np.clip(0.5 * mouth[2] / R * 0.95, 0.12, 0.3))
            noses = [f for f in below if f is not mouth and abs(f[0] - mid_x) < 0.3 * R]
            if noses:
                nose = min(noses, key=lambda f: f[1])
                kw["nose_v"] = float(np.clip(st.eye_v + (nose[1] - eye_y) / R, 0.12, 0.42))
                kw["nose_size"] = float(np.clip(0.5 * max(nose[2], nose[3]) / R, 0.035, 0.09))
                kw["nose_dir"] = -1 if nose[0] > mid_x else 1
    info["measured"] = {k: (v if not isinstance(v, float) else round(v, 3)) for k, v in kw.items()}
    return st.copy(**kw)


def mirror(rig: PlateRig) -> PlateRig:
    """Horizontally mirrored rig (a 3/4 view facing the other way) — the vector face is redrawn, never flipped."""
    W = rig.clean.shape[1]
    h = rig.head
    st = rig.style.copy(nose_dir=-rig.style.nose_dir)
    eyes = [(W - e[0] - e[2], e[1], e[2], e[3], W - 1 - e[4], e[5]) for e in rig.eyes][::-1]
    return PlateRig(clean=rig.clean[:, ::-1].copy(), face_mask=rig.face_mask[:, ::-1].copy(),
                    head=h.replace(cx=W - 1 - h.cx, yaw=-h.yaw, roll=-h.roll), style=st, eyes=eyes,
                    info=dict(rig.info, mirrored=True))
