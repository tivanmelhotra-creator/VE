"""
Head / eye-pair / mouth landmarks for flat 2D cartoon characters (white or flat-colour heads, ink outlines).

Why it is robust (v29.3 rewrite of the v29.2 detector — every rule fixes a failure seen on real panels):
  * the head score uses the CONVEX HULL of each fill region: eyes / open mouths / raised fingers cut notches into
    the raw contour (raw circularity of a screaming head = 0.15 → v29.2 picked the trousers instead).
  * score = hull_circularity^6 · ellipse_ratio^0.5 · feature_bonus · upper_bonus · sqrt(area) · border_penalty;
    feature_bonus rewards enclosed holes (eyes / mouth) — a sweater or trouser blob has none.
  * eyes = sclera blobs lying inside the DILATED head hull (catches the far eye of a 3/4 view whose ring touches
    the head outline) → best PAIR by size similarity + horizontality; 1 eye = profile; 0 sclera = ink-eye
    fallback (closed / squinted / `><`).
  * mouth = the non-fill component inside the eroded head, below the eyes, nearest the anatomical expectation
    (eyes_mid + 0.62·inter_ocular along the face's down axis) — not "union of all ink" (blush, hands, wrinkles).
  * stroke width is measured (distance transform of the head outline) so erosion and line weights scale.
  * `hint=(x, y)` biases the head choice (temporal tracking in video mode); ROI crops remain supported.

API: find_heads(img) → ranked candidates · detect_face_anatomy(img) → dict · anatomy_summary(info) → JSON-safe ·
     debug_anatomy_image(img) → (vis, info)
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

VIEW_MODES = ("front", "three_quarter_left", "three_quarter_right", "profile_left", "profile_right")


def fill_mask(img_bgr: np.ndarray, fill: tuple[int, int, int] | None = None, tol: float = 40.0) -> np.ndarray:
    """uint8 0/1 mask of the character fill colour. None → 'white' (low saturation, high value; beige bg excluded)."""
    if fill is None:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return ((hsv[:, :, 1] < 48) & (hsv[:, :, 2] > 168)).astype(np.uint8)
    d = np.linalg.norm(img_bgr.astype(np.float32) - np.asarray(fill, np.float32)[None, None, :], axis=2)
    return (d < tol).astype(np.uint8)


def kernel(k: float) -> np.ndarray:
    """odd elliptical structuring element of size ≈ k."""
    k = max(1, int(round(k))) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


class _Components:
    """connected components of the fill mask, computed once and shared by head / eye search."""

    def __init__(self, img_bgr: np.ndarray, fill):
        self.fm = fill_mask(img_bgr, fill)
        self.n, self.lbl, self.st, self.ce = cv2.connectedComponentsWithStats(self.fm, connectivity=4)


def find_heads(
    img_bgr: np.ndarray,
    fill: tuple[int, int, int] | None = None,
    hint: tuple[float, float] | None = None,
    min_area_frac: float = 0.0012,
    max_candidates: int = 6,
    _cc: _Components | None = None,
) -> list[dict[str, Any]]:
    """Ranked head candidates [{score, label, bbox, hull, circularity, ellipse, n_features, centroid, ...}]."""
    H, W = img_bgr.shape[:2]
    cc = _cc or _Components(img_bgr, fill)
    min_area = max(120, int(H * W * min_area_frac))
    out = []
    for i in range(1, cc.n):
        x, y, bw, bh, a = (int(v) for v in cc.st[i])
        if a < min_area or not (0.4 <= bw / float(max(1, bh)) <= 2.2):
            continue
        comp = (cc.lbl[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        ha, hp = float(cv2.contourArea(hull)), float(cv2.arcLength(hull, True))
        if ha < min_area or len(hull) < 5 or hp <= 0:
            continue
        hc = min(1.0, 4.0 * np.pi * ha / (hp * hp))
        (_, _), (ea, eb), _ = cv2.fitEllipse(hull)
        ell = min(ea, eb) / max(ea, eb, 1e-6)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, [c], -1, 1, -1)
        holes = ((filled > 0) & (comp == 0)).astype(np.uint8)
        holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel(max(3, 0.03 * np.sqrt(ha))))
        hn, _, hs, hcent = cv2.connectedComponentsWithStats(holes, connectivity=8)
        n_feat = sum(1 for j in range(1, hn)
                     if 0.002 * ha <= hs[j, 4] <= 0.35 * ha and 0.05 <= hcent[j][1] / max(1, bh) <= 0.85)
        # enclosed white blobs (sclera inside the head) also count as features
        hull_l = hull - 0
        n_white_inside = 0
        for j in range(1, cc.n):
            if j == i:
                continue
            jx, jy = cc.ce[j]
            ja = cc.st[j, 4]
            if 0.0025 * ha <= ja <= 0.2 * ha and x <= jx < x + bw and y <= jy < y + bh:
                if cv2.pointPolygonTest(hull_l, (float(jx - x), float(jy - y)), False) >= 0:
                    n_white_inside += 1
        n_feat += n_white_inside
        feat_bonus = 1.0 if n_feat >= 1 else 0.35
        border = x <= 1 or y <= 1 or x + bw >= W - 1 or y + bh >= H - 1
        cx, cy = float(cc.ce[i][0]), float(cc.ce[i][1])
        upper = 1.0 - 0.45 * (cy / float(H))
        score = (hc ** 6) * (ell ** 0.5) * feat_bonus * upper * np.sqrt(ha) * (0.5 if border else 1.0)
        if hint is not None:
            d = np.hypot(cx - hint[0], cy - hint[1]) / max(1.0, np.sqrt(ha))
            score *= float(np.exp(-0.5 * (d / 0.9) ** 2)) + 0.05
        out.append({
            "score": float(score), "label": i, "bbox": (x, y, bw, bh), "hull": hull + np.array([[[x, y]]]),
            "circularity": float(hc), "ellipse": float(ell), "n_features": int(n_feat), "centroid": (cx, cy),
            "area": int(a), "hull_area": ha,
        })
    out.sort(key=lambda d: -d["score"])
    return out[:max_candidates]


def _stroke_width(gray: np.ndarray, head_mask: np.ndarray, head_scale: float) -> float:
    ring = cv2.dilate(head_mask, kernel(max(5, head_scale * 0.08))) & (1 - head_mask)
    ink = ((gray < 120) & (ring > 0)).astype(np.uint8)
    if ink.sum() < 10:
        return float(np.clip(head_scale * 0.025, 1.5, 12.0))
    v = cv2.distanceTransform(ink, cv2.DIST_L2, 3)[ink > 0]
    return float(np.clip(2.0 * np.percentile(v, 90), 1.5, max(2.0, head_scale * 0.08)))


def _pick_pair(cands: list[tuple], hw: float, hh: float) -> list[tuple]:
    """cands: (cx, cy, bx, by, bw, bh, area, label). Best horizontal pair by size similarity + alignment."""
    best, best_s = None, -1.0
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            if dx < 0.12 * hw or dy > 0.5 * dx + 0.06 * hh:
                continue
            size_sim = min(a[6], b[6]) / float(max(a[6], b[6]))
            if size_sim < 0.12:
                continue
            s = (size_sim ** 0.5) * (1.0 - dy / (dx + 1.0)) * np.sqrt(a[6] + b[6])
            if s > best_s:
                best, best_s = (a, b), s
    return [] if best is None else sorted(best, key=lambda t: t[0])


def detect_face_anatomy(
    img_bgr: np.ndarray,
    head_roi_y: tuple[int | None, int | None] | None = None,
    head_roi_x: tuple[int | None, int | None] | None = None,
    fill: tuple[int, int, int] | None = None,
    hint: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """
    Universal head / eye-pair / mouth landmarks (front, 3/4, profile, tilted, closed eyes). Zero manual ROI needed.
    All coordinates are FULL-image pixels. Keys: view_mode, head_box, head_scale, num_eyes, eye_kind, eyes,
    eyes_mid, eyes_bottom_y, inter_ocular, eye_angle_deg, mouth_box, mouth_center, mouth_found, stroke_px,
    face_color, confidence, eye_fills, *_full masks (head, inner_head, eye_mask, mouth_sprite_mask).
    Raises ValueError if no head-like fill region exists.
    """
    H, W = img_bgr.shape[:2]
    y0 = 0 if head_roi_y is None or head_roi_y[0] is None else max(0, int(head_roi_y[0]))
    y1 = H if head_roi_y is None or head_roi_y[1] is None else min(H, int(head_roi_y[1]))
    x0 = 0 if head_roi_x is None or head_roi_x[0] is None else max(0, int(head_roi_x[0]))
    x1 = W if head_roi_x is None or head_roi_x[1] is None else min(W, int(head_roi_x[1]))
    sub = np.ascontiguousarray(img_bgr[y0:y1, x0:x1])
    sh, sw = sub.shape[:2]
    if sh < 16 or sw < 16:
        raise ValueError("image / ROI too small for head detection")
    cc = _Components(sub, fill)
    hint_l = None if hint is None else (hint[0] - x0, hint[1] - y0)
    cands = find_heads(sub, fill=fill, hint=hint_l, _cc=cc)
    if not cands:
        raise ValueError("No head-like fill region detected in image")
    best = cands[0]
    head_id = best["label"]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    fm = cc.fm

    head_mask = np.zeros((sh, sw), np.uint8)
    cv2.drawContours(head_mask, [best["hull"]], -1, 1, -1)
    hx, hy, hw, hh = cv2.boundingRect(best["hull"])
    head_scale = float(np.sqrt(hw * hh))
    stroke = _stroke_width(gray, head_mask, head_scale)
    head_dil = cv2.dilate(head_mask, kernel(2 * stroke + 3))
    ha = float(best["hull_area"])

    # ---- 2A: sclera eyes = fill blobs (≠ head) inside the dilated head hull
    eye_cands = []
    for i in range(1, cc.n):
        if i == head_id:
            continue
        bx, by, bw_, bh_, a = (int(v) for v in cc.st[i])
        if not (0.0025 * ha <= a <= 0.2 * ha):
            continue
        if bx + bw_ < hx or bx > hx + hw or by + bh_ < hy or by > hy + hh:
            continue
        comp = cc.lbl[by:by + bh_, bx:bx + bw_] == i
        if head_dil[by:by + bh_, bx:bx + bw_][comp].mean() < 0.6:
            continue
        cx, cy = float(cc.ce[i][0]), float(cc.ce[i][1])
        if not (0.10 <= (cy - hy) / float(max(1, hh)) <= 0.70):
            continue
        if bw_ < 0.05 * hw or bh_ < 0.07 * hh:
            continue
        eye_cands.append((cx, cy, bx, by, bw_, bh_, a, i))
    eyes = _pick_pair(eye_cands, hw, hh)
    eye_kind = "sclera"
    if not eyes and eye_cands:
        upper = [e for e in eye_cands if (e[1] - hy) / max(1, hh) <= 0.62]
        if upper:
            eyes = [max(upper, key=lambda t: t[6])]
    # ---- 2B: ink eyes (closed / squinted / ><)
    if not eyes:
        eye_kind = "ink"
        inner_band = cv2.erode(head_mask, kernel(stroke + 3))
        band = np.zeros_like(head_mask)
        ya, yb = int(hy + 0.12 * hh), int(hy + 0.60 * hh)
        band[ya:yb] = inner_band[ya:yb]
        dark = cv2.morphologyEx(((gray < 130) & (band > 0)).astype(np.uint8), cv2.MORPH_CLOSE, kernel(3))
        en, _, es, ec = cv2.connectedComponentsWithStats(dark, connectivity=8)
        ink_c = [(float(ec[i][0]), float(ec[i][1]), *(int(v) for v in es[i][:4]), int(es[i][4]), -1)
                 for i in range(1, en) if 0.0015 * ha <= es[i][4] <= 0.1 * ha and es[i][2] >= 0.06 * hw]
        eyes = _pick_pair(ink_c, hw, hh) or ([max(ink_c, key=lambda t: t[6])] if ink_c else [])
    if not eyes:
        eye_kind = "none"

    head_cx = hx + hw / 2.0
    if len(eyes) >= 2:
        (ax, ay), (bx_, by_) = eyes[0][:2], eyes[1][:2]
        eyes_mid_x, eyes_mid_y = 0.5 * (ax + bx_), 0.5 * (ay + by_)
        inter_ocular = float(np.hypot(bx_ - ax, by_ - ay))
        eye_angle_deg = float(np.degrees(np.arctan2(by_ - ay, bx_ - ax)))
        off = (eyes_mid_x - head_cx) / float(max(1, hw))
        size_ratio = min(eyes[0][6], eyes[1][6]) / float(max(eyes[0][6], eyes[1][6]))
        if off < -0.06 or (off < -0.03 and size_ratio < 0.6):
            view_mode = "three_quarter_left"
        elif off > 0.06 or (off > 0.03 and size_ratio < 0.6):
            view_mode = "three_quarter_right"
        else:
            view_mode = "front"
    elif len(eyes) == 1:
        eyes_mid_x, eyes_mid_y = eyes[0][0], eyes[0][1]
        inter_ocular, eye_angle_deg = head_scale * 0.42, 0.0
        view_mode = "profile_left" if eyes_mid_x < head_cx else "profile_right"
    else:
        eyes_mid_x, eyes_mid_y = head_cx, hy + 0.45 * hh
        inter_ocular, eye_angle_deg, view_mode = head_scale * 0.42, 0.0, "front"

    # eye protection mask (sclera + ring) and eye fills (for blinking)
    eye_mask = np.zeros_like(head_mask)
    eye_fills = []
    for (cx, cy, bx, by, bw_, bh_, a, lab) in eyes:
        ef = np.zeros_like(head_mask)
        if lab >= 0:
            em = (cc.lbl == lab).astype(np.uint8)
            cs, _ = cv2.findContours(em, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ef, cs, -1, 1, -1)
        else:
            cv2.ellipse(ef, (int(cx), int(cy)), (max(2, bw_ // 2), max(2, bh_ // 2)), 0, 0, 360, 1, -1)
        eye_mask |= cv2.dilate(ef, kernel(2 * stroke + 3))
        ys, xs = np.nonzero(ef)
        if len(xs):
            ex0, ey0, ex1, ey1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            eye_fills.append({"box": (int(ex0 + x0), int(ey0 + y0), int(ex1 - ex0), int(ey1 - ey0)),
                              "mask": ef[ey0:ey1, ex0:ex1].copy(), "kind": "sclera" if lab >= 0 else "ink"})
    eyes_bottom_y = int(max((e[3] + e[5] for e in eyes), default=int(hy + 0.52 * hh)))

    # ---- 3: mouth = best non-fill component in the lower face near the anatomical expectation
    inner_head = cv2.erode(head_mask, kernel(max(3, stroke)))
    zone = ((cv2.erode(inner_head, kernel(3)) > 0) & (eye_mask == 0)).astype(np.uint8)
    zone[:int(eyes_bottom_y + max(2, 0.02 * hh)), :] = 0
    if view_mode == "profile_left":
        mx_min, mx_max = hx, int(eyes_mid_x + 0.45 * hw)
    elif view_mode == "profile_right":
        mx_min, mx_max = int(eyes_mid_x - 0.45 * hw), hx + hw
    else:
        mx_min, mx_max = int(eyes_mid_x - 0.85 * inter_ocular), int(eyes_mid_x + 0.85 * inter_ocular)
    zone[:, :max(0, mx_min)] = 0
    zone[:, max(0, mx_max):] = 0
    ang = np.radians(eye_angle_deg)
    down = np.array([-np.sin(ang), np.cos(ang)])
    exp_pt = np.array([eyes_mid_x, eyes_mid_y]) + down * (0.62 * inter_ocular)
    if "profile" in view_mode:
        exp_pt[0] = eyes_mid_x + (-0.15 if view_mode == "profile_left" else 0.15) * hw
    mouth_ink = ((fm == 0) & (zone > 0)).astype(np.uint8)
    mouth_ink = cv2.morphologyEx(mouth_ink, cv2.MORPH_CLOSE, kernel(stroke * 1.5 + 2))
    mn, ml, ms, mc = cv2.connectedComponentsWithStats(mouth_ink, connectivity=8)
    mouth_sprite_mask = np.zeros_like(head_mask)
    mouth_found = False
    if mn > 1:
        scores = []
        for i in range(1, mn):
            if ms[i, 4] < max(4, 0.0006 * ha):
                scores.append(-1.0)
                continue
            d = np.hypot(mc[i][0] - exp_pt[0], mc[i][1] - exp_pt[1]) / max(1.0, inter_ocular)
            scores.append(float(np.sqrt(ms[i, 4]) / (1.0 + 4.0 * d * d)))
        k = int(np.argmax(scores)) + 1
        if scores[k - 1] > 0:
            mouth_found = True
            kx, ky, kw, kh = ms[k, :4]
            reach = 0.3 * inter_ocular
            for i in range(1, mn):
                bx_, by_, bw_, bh_ = ms[i, :4]
                if scores[i - 1] > 0 and (bx_ <= kx + kw + reach and bx_ + bw_ >= kx - reach and
                                          by_ <= ky + kh + reach and by_ + bh_ >= ky - reach):
                    mouth_sprite_mask[ml == i] = 255
            mouth_sprite_mask = cv2.dilate(mouth_sprite_mask, kernel(3))
    if mouth_found:
        mx, my, mw, mh = cv2.boundingRect(mouth_sprite_mask)
        m_cx, m_cy = mx + mw / 2.0, my + mh / 2.0
    else:
        mw, mh = max(8, int(0.5 * inter_ocular)), max(3, int(stroke * 2))
        m_cx, m_cy = float(exp_pt[0]), float(max(exp_pt[1], eyes_bottom_y + 0.15 * inter_ocular))
        mx, my = int(m_cx - mw / 2), int(m_cy - mh / 2)

    skin = (inner_head > 0) & (eye_mask == 0) & (mouth_sprite_mask == 0) & (fm > 0)
    face_color = (np.median(sub[skin], axis=0) if skin.sum() > 20 else np.array([250, 250, 250])).astype(np.uint8)
    conf = {2: 1.0, 1: 0.8, 0: 0.45}[min(2, len(eyes))] * (0.85 if eye_kind == "ink" else 1.0)
    conf *= best["circularity"] * (1.0 if mouth_found else 0.8)

    pad = ((y0, H - y1), (x0, W - x1))
    return {
        "view_mode": view_mode,
        "offset": (x0, y0),
        "head_box": (int(hx + x0), int(hy + y0), int(hw), int(hh)),
        "head_scale": head_scale,
        "num_eyes": len(eyes),
        "eye_kind": eye_kind,
        "eyes": [(float(e[0] + x0), float(e[1] + y0), int(e[2] + x0), int(e[3] + y0), int(e[4]), int(e[5]))
                 for e in eyes],
        "eyes_mid": (float(eyes_mid_x + x0), float(eyes_mid_y + y0)),
        "eyes_bottom_y": int(eyes_bottom_y + y0),
        "inter_ocular": float(inter_ocular),
        "eye_angle_deg": float(eye_angle_deg),
        "mouth_box": (int(mx + x0), int(my + y0), int(mw), int(mh)),
        "mouth_center": (float(m_cx + x0), float(m_cy + y0)),
        "mouth_found": bool(mouth_found),
        "stroke_px": float(stroke),
        "face_color": face_color,
        "confidence": float(round(conf, 3)),
        "head_mask_full": np.pad(head_mask * 255, pad),
        "inner_head_full": np.pad(inner_head * 255, pad),
        "eye_mask_full": np.pad(eye_mask * 255, pad),
        "mouth_sprite_mask_full": np.pad(mouth_sprite_mask, pad),
        "eye_fills": eye_fills,
    }


def anatomy_summary(info: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe subset of a detect_face_anatomy() result."""
    keys = ("view_mode", "head_box", "head_scale", "num_eyes", "eye_kind", "eyes_mid", "eyes_bottom_y",
            "inter_ocular", "eye_angle_deg", "mouth_box", "mouth_center", "mouth_found", "stroke_px", "confidence")
    out: dict[str, Any] = {}
    for k in keys:
        v = info[k]
        if isinstance(v, tuple):
            v = [round(float(x), 2) for x in v]
        elif isinstance(v, (float, np.floating)):
            v = round(float(v), 2)
        elif isinstance(v, np.integer):
            v = int(v)
        out[k] = v
    out["face_color"] = [int(c) for c in info["face_color"]]
    return out


def debug_anatomy_image(
    img_bgr: np.ndarray,
    head_roi_y: tuple[int | None, int | None] | None = None,
    head_roi_x: tuple[int | None, int | None] | None = None,
    fill: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draws head box, eyes, eye-bottom ruler, mouth box + crosshair and a view/confidence label."""
    info = detect_face_anatomy(img_bgr, head_roi_y, head_roi_x, fill=fill)
    vis = img_bgr.copy()
    lw = max(1, int(round(info["head_scale"] / 90)))
    hx, hy, hw, hh = info["head_box"]
    cv2.rectangle(vis, (hx, hy), (hx + hw, hy + hh), (255, 140, 0), lw)
    for (cx, cy, bx, by, bw, bh) in info["eyes"]:
        cv2.rectangle(vis, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 200, 0), lw)
        cv2.circle(vis, (int(cx), int(cy)), 2 + lw, (0, 200, 0), -1)
    eb = info["eyes_bottom_y"]
    cv2.line(vis, (hx, eb), (hx + hw, eb), (0, 180, 255), lw)
    mx, my, mw, mh = info["mouth_box"]
    cv2.rectangle(vis, (mx, my), (mx + mw, my + mh), (0, 0, 255), lw)
    mcx, mcy = info["mouth_center"]
    cv2.drawMarker(vis, (int(mcx), int(mcy)), (0, 0, 255), cv2.MARKER_CROSS, 10 + 4 * lw, lw)
    label = f"{info['view_mode']} eyes={info['num_eyes']}({info['eye_kind']}) conf={info['confidence']:.2f}"
    fs = max(0.33, hw / 420)
    org = (max(0, hx - hw // 3), max(12, hy - 6))
    cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lw + 2, cv2.LINE_AA)
    cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), lw, cv2.LINE_AA)
    return vis, info
