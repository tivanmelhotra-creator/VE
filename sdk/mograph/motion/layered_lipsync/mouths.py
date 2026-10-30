"""
Mouth sources + placement for the layered lip-sync engine.

Two interchangeable sources, both keyed by the Preston Blair visemes of `mograph.motion.lipsync`
(rest MBP etc E AI O U FV L WQ):
  * GridMouths(grid_img, rows, cols)   cut real drawn mouths from an R×C viseme sheet (each cell = the character's
                                       head with one mouth). Each sprite is AUTO-ASSIGNED to a viseme by its measured
                                       geometry (width / opening relative to the inter-ocular ruler) — no hand-made
                                       index table (v29.2 hard-coded cells 0/2/4/10/5 for 5 loudness bands).
  * VectorMouths(style)                procedural, resolution-independent mouths drawn with OpenCV at 4× supersampling
                                       from `lipsync.MOUTHS` geometry; they BLEND between shapes (shape_at), stroke
                                       width follows the character's measured line weight.

place_mouth(frame, anatomy, sprite) scales by the pose ruler, rotates with head tilt, clips to the inner head
(minus eyes), clamps below the eyes / above the chin and returns a QA dict.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .. import lipsync as LS
from .anatomy import detect_face_anatomy

VISEMES = list(LS.VISEMES)


# --------------------------------------------------------------------------------------------- vector mouths
def _bez(p0, p1, p2, p3, n=28):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3


DEFAULT_STYLE = {
    "ink": (12, 10, 10), "inner": (20, 20, 44), "teeth": (248, 250, 252), "tongue": (102, 118, 214),
    "lip": None,                       # None → face colour (FV lower lip)
    "width_io": 0.64,                  # mouth width (open AI) as a fraction of the inter-ocular distance
    "stroke_mult": 0.92,               # mouth line weight vs measured character stroke
    "cheek_skew": 0.18,                # asymmetric 3/4 bean hook toward the back cheek
    "buccal_gap": True,                # dark corner corridor beside clenched teeth (E / etc)
    "pucker_arc": True,                # subtle lip-cushion arc on small U / WQ / O mouths
}


def draw_vector_mouth(g: dict[str, float], width_px: float, stroke_px: float, style: dict | None = None,
                      face_color=(250, 250, 250), view_mode: str = "front", ss: int = 4) -> dict[str, Any]:
    """render one mouth (geometry g from lipsync.MOUTHS / shape_at) → {bgr, alpha, anchor} (anchor = lip line centre)."""
    st = dict(DEFAULT_STYLE, **(style or {}))
    W0 = float(width_px) * ss
    mw = W0 * g.get("w", 0.8)
    mh = W0 * 0.52 * g.get("h", 0.0)
    lw = max(1.0, stroke_px * st["stroke_mult"]) * ss
    r = g.get("round", 0.0)
    vname = g.get("v", "rest")
    # Cheek direction: in three_quarter_left / profile_left, the back cheek is on the right (+1); else left (-1)
    cheek_dir = 1.0 if "left" in str(view_mode) else (-1.0 if "right" in str(view_mode) else 0.25)
    skew = float(st.get("cheek_skew", 0.18)) * cheek_dir
    half_w = mw / 2 + 3 * lw + 4 * ss
    half_h = max(mh * 1.45, W0 * 0.10) + 3 * lw + 4 * ss
    Wc, Hc = int(np.ceil(2 * half_w)), int(np.ceil(2 * half_h))
    cx, cy = Wc / 2.0, Hc / 2.0
    bgr = np.zeros((Hc, Wc, 3), np.uint8)
    alpha = np.zeros((Hc, Wc), np.uint8)
    ink = tuple(int(c) for c in st["ink"])
    off = np.array([cx, cy])

    def P(x, y):
        return np.array([x, y], np.float64)

    if mh < W0 * 0.02:                                       # closed — asymmetric smirk/bean line (MBP presses)
        press = g.get("press", 0.0)
        sag = W0 * 0.03 * (1 + press)
        c_lift = -W0 * 0.022 * skew
        pts = _bez(P(-mw / 2, -c_lift * 0.4), P(-mw / 4, sag), P(mw / 4, sag + c_lift * 0.5), P(mw / 2, c_lift)) + off
        th = max(1, int(round(lw * (1 + 0.45 * press))))
        cv2.polylines(bgr, [pts.astype(np.int32)], False, ink, th, cv2.LINE_AA)
        cv2.polylines(alpha, [pts.astype(np.int32)], False, 255, th, cv2.LINE_AA)
        if press > 0.5 or st.get("pucker_arc", True):
            # Subtle corner dimple / cushion tick at the cheek end
            corner_x = (mw / 2 + lw * 0.7) * (1 if cheek_dir >= 0 else -1)
            corner_y = c_lift if cheek_dir >= 0 else -c_lift * 0.4
            arc_pts = _bez(
                P(corner_x - cheek_dir * lw * 0.5, corner_y - lw * 1.3),
                P(corner_x + cheek_dir * lw * 0.8, corner_y),
                P(corner_x + cheek_dir * lw * 0.8, corner_y + lw * 0.8),
                P(corner_x - cheek_dir * lw * 0.5, corner_y + lw * 1.6),
                n=14
            ) + off
            cv2.polylines(bgr, [arc_pts.astype(np.int32)], False, ink, max(1, int(lw * 0.65)), cv2.LINE_AA)
            cv2.polylines(alpha, [arc_pts.astype(np.int32)], False, 255, max(1, int(lw * 0.65)), cv2.LINE_AA)
    else:
        # Asymmetric bean-shaped mouth outline: wider/hooked on the back-cheek side
        top_k = 0.38 + 0.35 * r
        l_lift = -mh * 0.22 * (-skew)
        r_lift = -mh * 0.22 * skew
        l_bulge = 1.0 + 0.18 * (-skew)
        r_bulge = 1.0 + 0.18 * skew
        top_l = _bez(
            P(-mw / 2, l_lift),
            P(-mw / 2, -mh * top_k * 1.65 * l_bulge + l_lift),
            P(-mw * 0.25 * (1 - r * 0.28), -mh * l_bulge),
            P(0, -mh * (0.86 + 0.14 * r)),
        )
        top_r = _bez(
            P(0, -mh * (0.86 + 0.14 * r)),
            P(mw * 0.25 * (1 - r * 0.28), -mh * r_bulge),
            P(mw / 2, -mh * top_k * 1.65 * r_bulge + r_lift),
            P(mw / 2, r_lift),
        )
        bot = _bez(
            P(mw / 2, r_lift),
            P(mw * 0.46, mh * 1.52 * r_bulge),
            P(-mw * 0.46, mh * 1.52 * l_bulge),
            P(-mw / 2, l_lift),
        )
        poly = (np.vstack([top_l, top_r[1:], bot[1:]]) + off).astype(np.int32)
        inner_m = np.zeros((Hc, Wc), np.uint8)
        cv2.fillPoly(inner_m, [poly], 255, cv2.LINE_AA)
        layer = np.zeros_like(bgr)
        layer[:] = st["inner"]
        th = max(1, int(round(lw)))

        # Tongue first (so clenched teeth or upper teeth band cleanly sit in appropriate order)
        if g.get("tongue", 0) > 0.05:
            ty = mh * (0.72 - 0.85 * g.get("tongue_up", 0.0))
            tx_off = -cheek_dir * mw * 0.08
            rad_x = max(1, int(min(mw * 0.36, mh * 1.05)))
            rad_y = max(1, int(min(mw * 0.28, mh * 0.85)))
            t_center = (int(cx + tx_off), int(cy + ty + mh * 0.58))
            cv2.ellipse(layer, t_center, (rad_x, rad_y), 0, 175, 365,
                        tuple(int(c) for c in st["tongue"]), -1, cv2.LINE_AA)
            # Subtle dark crease line on tongue
            cv2.ellipse(layer, t_center, (rad_x, rad_y), 0, 190, 350,
                        ink, max(1, int(lw * 0.65)), cv2.LINE_AA)

        # Teeth with AL-style dark buccal corner corridor on E / etc / clenched shapes
        if g.get("teeth", 0) > 0.05:
            is_clenched = (vname in ("E", "etc") or (g.get("teeth", 0) >= 0.65 and r < 0.35 and mh < W0 * 0.22))
            if is_clenched and st.get("buccal_gap", True):
                # Fill teeth across 80% of mouth width, leaving a dark buccal corner on the cheek side!
                t_mask = np.zeros((Hc, Wc), np.uint8)
                if cheek_dir >= 0:
                    x_cut = int(cx + mw * 0.26)
                    t_mask[:, :x_cut] = 255
                else:
                    x_cut = int(cx - mw * 0.26)
                    t_mask[:, x_cut:] = 255
                y_bot_teeth = int(cy + mh * 0.95)
                t_mask[y_bot_teeth:, :] = 0
                layer[t_mask > 0] = st["teeth"]
                # Draw vertical ink edge of the teeth block and horizontal bite line
                cv2.line(layer, (x_cut, int(cy - mh * 1.2)), (x_cut, y_bot_teeth), ink, max(1, int(lw * 0.85)), cv2.LINE_AA)
                cv2.line(layer, (int(cx - mw * 0.5), y_bot_teeth), (int(cx + mw * 0.5), y_bot_teeth), ink, max(1, int(lw * 0.75)), cv2.LINE_AA)
            else:
                tb = mh * (0.46 + 0.34 * g["teeth"])
                y_a, y_b = int(cy - mh * 1.25), int(cy - mh * 1.25 + tb)
                layer[max(0, y_a):max(0, y_b)] = st["teeth"]
                cv2.line(layer, (int(cx - mw * 0.5), max(0, y_b)), (int(cx + mw * 0.5), max(0, y_b)), ink, max(1, int(lw * 0.7)), cv2.LINE_AA)
                if g.get("bite", 0) > 0.5:
                    lip = st["lip"] if st["lip"] is not None else tuple(int(c) for c in face_color)
                    layer[max(0, y_b):] = lip

        a_in = inner_m.astype(np.float32)[:, :, None] / 255.0
        bgr = (layer * a_in + bgr * (1 - a_in)).astype(np.uint8)
        alpha = np.maximum(alpha, inner_m)
        cv2.polylines(bgr, [poly], True, ink, th, cv2.LINE_AA)
        cv2.polylines(alpha, [poly], True, 255, th, cv2.LINE_AA)

        # Pucker lip-cushion arc on small rounded vowels (U / WQ / O)
        if r >= 0.65 and st.get("pucker_arc", True):
            px_side = (mw / 2 + lw * 1.25) * (1 if cheek_dir >= 0 else -1)
            p_arc = _bez(
                P(px_side - cheek_dir * lw * 0.4, -mh * 0.55),
                P(px_side + cheek_dir * lw * 0.9, -mh * 0.1),
                P(px_side + cheek_dir * lw * 0.9, mh * 0.55),
                P(px_side - cheek_dir * lw * 0.4, mh * 0.95),
                n=16,
            ) + off
            cv2.polylines(bgr, [p_arc.astype(np.int32)], False, ink, max(1, int(lw * 0.7)), cv2.LINE_AA)
            cv2.polylines(alpha, [p_arc.astype(np.int32)], False, 255, max(1, int(lw * 0.7)), cv2.LINE_AA)
    # downsample (anti-aliasing); premultiplied to avoid dark fringes
    a = alpha.astype(np.float32) / 255.0
    pm = bgr.astype(np.float32) * a[:, :, None]
    dw, dh = max(2, Wc // ss), max(2, Hc // ss)
    pm = cv2.resize(pm, (dw, dh), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, (dw, dh), interpolation=cv2.INTER_AREA)
    col = np.where(a[:, :, None] > 1e-3, pm / np.maximum(a[:, :, None], 1e-3), 0)
    return {"bgr": np.clip(col, 0, 255).astype(np.uint8), "alpha": a.astype(np.float32),
            "anchor": (cx / ss, cy / ss)}


class VectorMouths:
    """procedural mouth source. get(geom, anatomy) → sprite sized for that pose (cached by quantised geometry)."""

    kind = "vector"

    def __init__(self, style: dict | None = None):
        self.style = dict(DEFAULT_STYLE, **(style or {}))
        self._cache: dict = {}

    def get(self, geom: dict[str, float], anatomy: dict[str, Any]) -> dict[str, Any]:
        width = self.style["width_io"] * anatomy["inter_ocular"]
        if "profile" in anatomy["view_mode"]:
            width *= 0.6
        elif "three_quarter" in anatomy["view_mode"]:
            width *= 0.88
        key = (geom.get("v", "rest"), anatomy.get("view_mode", "front"), tuple(sorted((k, round(float(v), 2)) for k, v in geom.items() if k != "v")),
               round(width, 1), round(anatomy["stroke_px"], 1), tuple(int(c) for c in anatomy["face_color"]))
        if key not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            sp = draw_vector_mouth(geom, width, anatomy["stroke_px"], self.style, anatomy["face_color"], view_mode=anatomy.get("view_mode", "front"))
            sp["scale_ready"] = True
            self._cache[key] = sp
        return self._cache[key]


# --------------------------------------------------------------------------------------------- grid mouths
def _grid_edges(gray: np.ndarray, n: int, axis: int) -> list[int]:
    size = gray.shape[1 - axis] if axis == 0 else gray.shape[0]
    prof = gray.mean(axis=axis)
    win = max(4, int(size * 0.04))
    edges = [0]
    for k in range(1, n):
        c = int(size * k / n)
        lo, hi = max(0, c - win), min(size, c + win)
        edges.append(lo + int(np.argmin(prof[lo:hi])))
    edges.append(size)
    return edges


class GridMouths:
    """real drawn mouths cut from a viseme sheet; sprites auto-assigned to Preston Blair visemes by geometry."""

    kind = "grid"

    def __init__(self, grid_bgr: np.ndarray, rows: int = 4, cols: int = 4, mapping: dict[str, int] | None = None,
                 fill=None):
        if grid_bgr is None or grid_bgr.size == 0:
            raise ValueError("empty viseme grid image")
        gray = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2GRAY)
        xs, ys = _grid_edges(gray, cols, 0), _grid_edges(gray, rows, 1)
        self.sprites: list[dict[str, Any] | None] = []
        self.failures: list[int] = []
        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            m = max(2, int(0.01 * min(grid_bgr.shape[:2])))
            cell = np.ascontiguousarray(grid_bgr[ys[r] + m:ys[r + 1] - m, xs[c] + m:xs[c + 1] - m])
            try:
                info = detect_face_anatomy(cell, fill=fill)
            except ValueError:
                info = None
            if info is None or not info["mouth_found"]:
                self.sprites.append(None)
                self.failures.append(idx)
                continue
            mx, my, mw, mh = info["mouth_box"]
            pad = max(3, int(info["stroke_px"] * 1.5))
            ya, yb = max(0, my - pad), min(cell.shape[0], my + mh + pad)
            xa, xb = max(0, mx - pad), min(cell.shape[1], mx + mw + pad)
            mask = info["mouth_sprite_mask_full"][ya:yb, xa:xb].astype(np.float32) / 255.0
            mask = cv2.GaussianBlur(mask, (0, 0), 0.7)
            io = max(1.0, info["inter_ocular"])
            crop = cell[ya:yb, xa:xb]
            dark = float((cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[mask > 0.5] < 90).mean()) if (mask > 0.5).any() else 0
            self.sprites.append({
                "index": idx, "bgr": crop.copy(), "alpha": mask, "inter_ocular": io,
                "eye_angle_deg": info["eye_angle_deg"],
                "anchor": (info["mouth_center"][0] - xa, info["mouth_center"][1] - ya),
                "w_io": mw / io, "h_io": mh / io, "dark": dark, "view_mode": info["view_mode"],
            })
        ok = [s for s in self.sprites if s is not None]
        if not ok:
            raise ValueError("no usable mouth found in any viseme-grid cell")
        self.mapping = dict(mapping) if mapping else self._auto_map(ok)

    @staticmethod
    def _auto_map(ok: list[dict[str, Any]]) -> dict[str, int]:
        """nearest sprite per viseme in normalised (width, opening) space; opening measured as box height."""
        W = np.array([s["w_io"] for s in ok])
        Hh = np.array([s["h_io"] * (0.5 + s["dark"]) for s in ok])
        wn = (W - W.min()) / max(1e-6, W.max() - W.min())
        hn = (Hh - Hh.min()) / max(1e-6, Hh.max() - Hh.min())
        gw = np.array([LS.MOUTHS[v]["w"] for v in VISEMES])
        gh = np.array([LS.MOUTHS[v]["h"] for v in VISEMES])
        gwn = (gw - gw.min()) / max(1e-6, gw.max() - gw.min())
        ghn = (gh - gh.min()) / max(1e-6, gh.max() - gh.min())
        out = {}
        for k, v in enumerate(VISEMES):
            d = 2.0 * (hn - ghn[k]) ** 2 + 0.6 * (wn - gwn[k]) ** 2
            out[v] = int(ok[int(np.argmin(d))]["index"])
        return out

    def get(self, geom: dict[str, float], anatomy: dict[str, Any]) -> dict[str, Any]:
        sp = self.sprites[self.mapping.get(geom.get("v", "rest"), self.mapping["rest"])]
        if sp is None:
            sp = next(s for s in self.sprites if s is not None)
        return sp


# --------------------------------------------------------------------------------------------- placement
def place_mouth(frame_bgr: np.ndarray, anatomy: dict[str, Any], sprite: dict[str, Any],
                scale_mult: float = 0.92, in_place: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """composite a mouth sprite on an (erased) pose at anatomy['mouth_center'] → (frame, qa)."""
    frm = frame_bgr if in_place else frame_bgr.copy()
    H, W = frm.shape[:2]
    if sprite.get("scale_ready"):
        s = 1.0
        d_ang = anatomy["eye_angle_deg"]
    else:
        s = anatomy["inter_ocular"] / max(1.0, sprite["inter_ocular"]) * scale_mult
        d_ang = anatomy["eye_angle_deg"] - sprite.get("eye_angle_deg", 0.0)
    sb, sa = sprite["bgr"], sprite["alpha"]
    ax, ay = sprite["anchor"]
    tx, ty = anatomy["mouth_center"]
    hx, hy, hw, hh = anatomy["head_box"]
    min_ty = anatomy["eyes_bottom_y"] + 0.5 * sb.shape[0] * s * 0.6 + 2
    ty = float(np.clip(ty, min_ty, hy + hh - 0.1 * hh))
    M = cv2.getRotationMatrix2D((ax, ay), -d_ang if abs(d_ang) > 0.5 else 0.0, s)
    M[0, 2] += tx - ax
    M[1, 2] += ty - ay
    # warp only inside a bounding box around the target (cheap at 1080p)
    rad = int(np.hypot(*sb.shape[:2]) * s / 2 + 4)
    x0, x1 = max(0, int(tx) - rad), min(W, int(tx) + rad)
    y0, y1 = max(0, int(ty) - rad), min(H, int(ty) + rad)
    if x1 <= x0 or y1 <= y0:
        return frm, {"ok": False, "reason": "outside frame"}
    Mb = M.copy()
    Mb[0, 2] -= x0
    Mb[1, 2] -= y0
    bw, bh = x1 - x0, y1 - y0
    wb = cv2.warpAffine(sb, Mb, (bw, bh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    wa = cv2.warpAffine(sa, Mb, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    clip = ((anatomy["inner_head_full"][y0:y1, x0:x1] > 0) &
            (anatomy["eye_mask_full"][y0:y1, x0:x1] == 0)).astype(np.float32)
    clip = cv2.GaussianBlur(clip, (0, 0), 0.6)
    a = np.clip(wa * clip, 0, 1)[:, :, None]
    roi = frm[y0:y1, x0:x1]
    roi[:] = (wb.astype(np.float32) * a + roi.astype(np.float32) * (1 - a) + 0.5).astype(np.uint8)
    vis = wa > 0.3
    ys = np.nonzero(vis.any(axis=1))[0]
    lost = float(1.0 - (a[:, :, 0] > 0.3).sum() / max(1, vis.sum()))
    top = (y0 + ys[0]) if len(ys) else int(ty)
    bot = (y0 + ys[-1]) if len(ys) else int(ty)
    qa = {
        "below_eyes": bool(top >= anatomy["eyes_bottom_y"] - 1),
        "above_chin": bool(bot <= hy + hh),
        "clipped_frac": round(lost, 3),
        "mouth_center": (round(tx, 1), round(ty, 1)),
    }
    qa["ok"] = qa["below_eyes"] and qa["above_chin"] and lost < 0.25
    return frm, qa


def erase_base_mouth(img_bgr: np.ndarray, anatomy: dict[str, Any]) -> np.ndarray:
    """paint the drawn mouth away with the face colour (eyes protected), feathered so no ghost line remains."""
    out = img_bgr.copy()
    if not anatomy.get("mouth_found", True):
        return out
    k = max(3, int(anatomy.get("stroke_px", 3) * 1.2)) | 1
    m = cv2.dilate(anatomy["mouth_sprite_mask_full"], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m[anatomy["eye_mask_full"] > 0] = 0
    m[anatomy["inner_head_full"] == 0] = 0
    a = cv2.GaussianBlur((m > 0).astype(np.float32), (0, 0), 1.0)[:, :, None]
    face = np.empty_like(out)
    face[:] = anatomy["face_color"]
    return (face * a + out * (1 - a) + 0.5).astype(np.uint8)
