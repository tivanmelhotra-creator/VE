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
    "ink": (12, 10, 10), "inner": (34, 22, 40), "teeth": (250, 250, 248), "tongue": (150, 140, 236),
    "lip": None,                       # None → face colour (FV lower lip)
    "width_io": 0.62,                  # mouth width (open AI) as a fraction of the inter-ocular distance
    "stroke_mult": 0.9,                # mouth line weight vs measured character stroke
}


def draw_vector_mouth(g: dict[str, float], width_px: float, stroke_px: float, style: dict | None = None,
                      face_color=(250, 250, 250), ss: int = 4) -> dict[str, Any]:
    """render one mouth (geometry g from lipsync.MOUTHS / shape_at) → {bgr, alpha, anchor} (anchor = lip line centre)."""
    st = dict(DEFAULT_STYLE, **(style or {}))
    W0 = float(width_px) * ss
    mw = W0 * g.get("w", 0.8)
    mh = W0 * 0.5 * g.get("h", 0.0)
    lw = max(1.0, stroke_px * st["stroke_mult"]) * ss
    r = g.get("round", 0.0)
    half_w = mw / 2 + 2 * lw + 2 * ss
    half_h = max(mh * 1.35, W0 * 0.08) + 2 * lw + 2 * ss
    Wc, Hc = int(np.ceil(2 * half_w)), int(np.ceil(2 * half_h))
    cx, cy = Wc / 2.0, Hc / 2.0
    bgr = np.zeros((Hc, Wc, 3), np.uint8)
    alpha = np.zeros((Hc, Wc), np.uint8)
    ink = tuple(int(c) for c in st["ink"])
    off = np.array([cx, cy])

    def P(x, y):
        return np.array([x, y], np.float64)

    if mh < W0 * 0.02:                                       # closed — a line (MBP presses + thickens)
        press = g.get("press", 0.0)
        sag = W0 * 0.03 * (1 + press)
        pts = _bez(P(-mw / 2, 0), P(-mw / 4, sag), P(mw / 4, sag), P(mw / 2, 0)) + off
        th = max(1, int(round(lw * (1 + 0.5 * press))))
        cv2.polylines(bgr, [pts.astype(np.int32)], False, ink, th, cv2.LINE_AA)
        cv2.polylines(alpha, [pts.astype(np.int32)], False, 255, th, cv2.LINE_AA)
        if press > 0.5:
            for sy in (-1, 1):
                a_ = (P(-mw * 0.18, sy * lw * 1.7) + off).astype(int)
                b_ = (P(mw * 0.18, sy * lw * 1.7) + off).astype(int)
                cv2.line(bgr, tuple(a_), tuple(b_), ink, max(1, int(lw * 0.5)), cv2.LINE_AA)
                cv2.line(alpha, tuple(a_), tuple(b_), 255, max(1, int(lw * 0.5)), cv2.LINE_AA)
    else:
        top_k = 0.35 + 0.35 * r
        top_l = _bez(P(-mw / 2, 0), P(-mw / 2, -mh * top_k * 1.6), P(-mw * 0.25 * (1 - r * 0.3), -mh),
                     P(0, -mh * (0.85 + 0.15 * r)))
        top_r = _bez(P(0, -mh * (0.85 + 0.15 * r)), P(mw * 0.25 * (1 - r * 0.3), -mh),
                     P(mw / 2, -mh * top_k * 1.6), P(mw / 2, 0))
        bot = _bez(P(mw / 2, 0), P(mw / 2, mh * 1.5), P(-mw / 2, mh * 1.5), P(-mw / 2, 0))
        poly = (np.vstack([top_l, top_r[1:], bot[1:]]) + off).astype(np.int32)
        inner_m = np.zeros((Hc, Wc), np.uint8)
        cv2.fillPoly(inner_m, [poly], 255, cv2.LINE_AA)
        layer = np.zeros_like(bgr)
        layer[:] = st["inner"]
        if g.get("teeth", 0) > 0.05:
            tb = mh * (0.42 + 0.33 * g["teeth"])
            y_a, y_b = int(cy - mh * 1.2), int(cy - mh * 1.2 + tb)
            layer[max(0, y_a):max(0, y_b)] = st["teeth"]
            if g.get("bite", 0) > 0.5:
                lip = st["lip"] if st["lip"] is not None else tuple(int(c) for c in face_color)
                layer[max(0, y_b):] = lip
        if g.get("tongue", 0) > 0.05:
            ty = mh * (0.75 - 0.9 * g.get("tongue_up", 0.0))
            rad = max(1, int(min(mw * 0.3, mh * 0.9)))
            cv2.ellipse(layer, (int(cx), int(cy + ty + mh * 0.62)), (rad, rad), 0, 180, 360,
                        tuple(int(c) for c in st["tongue"]), -1, cv2.LINE_AA)
        a_in = inner_m.astype(np.float32)[:, :, None] / 255.0
        bgr = (layer * a_in + bgr * (1 - a_in)).astype(np.uint8)
        alpha = np.maximum(alpha, inner_m)
        th = max(1, int(round(lw)))
        cv2.polylines(bgr, [poly], True, ink, th, cv2.LINE_AA)
        cv2.polylines(alpha, [poly], True, 255, th, cv2.LINE_AA)
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
        key = (tuple(sorted((k, round(float(v), 2)) for k, v in geom.items() if k != "v")),
               round(width, 1), round(anatomy["stroke_px"], 1), tuple(int(c) for c in anatomy["face_color"]))
        if key not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            sp = draw_vector_mouth(geom, width, anatomy["stroke_px"], self.style, anatomy["face_color"])
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
