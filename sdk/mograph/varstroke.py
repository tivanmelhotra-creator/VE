"""
mograph.varstroke — VARIABLE-WIDTH TAPERED & WAVE STROKES + PER-VERTEX VARIABLE MASK FEATHER
(Ported from EffectCraft `crates/path/src/varstroke.rs` & `crates/path/src/feather.rs`).

1. `Taper` & `Wave`:
   Builds a variable-width stroke along any polyline/curve as an exact union of positively-wound
   trapezoid segments and round-join discs, so hand-drawn underlines, strike-throughs, circles,
   and callout leaders taper smoothly to a calligraphic brush tip (`start_width -> 1.0 -> end_width`).
2. `variable_feather_mask`:
   Interpolates signed inner/outer feather radii (`FeatherPoint`) along a closed contour's arc length
   so one side of a mask can be razor-sharp (`0.5px`) while the opposite side feathers over `45px`.
"""
from __future__ import annotations

import math
import cairo
import cv2
import numpy as np


class Taper:
    """Ported from EffectCraft `varstroke::Taper`."""
    def __init__(self, start_len: float = 0.25, end_len: float = 0.35,
                 start_width: float = 0.05, end_width: float = 0.02,
                 start_ease: float = 0.5, end_ease: float = 0.5, percent: bool = True):
        self.start_len = float(start_len)
        self.end_len = float(end_len)
        self.start_width = float(start_width)
        self.end_width = float(end_width)
        self.start_ease = float(start_ease)
        self.end_ease = float(end_ease)
        self.percent = bool(percent)

    def width_factor(self, s: float, total_len: float) -> float:
        if total_len <= 1e-6:
            return 1.0
        sl = self.start_len * total_len if self.percent else self.start_len
        el = self.end_len * total_len if self.percent else self.end_len
        w = 1.0
        if sl > 1e-5 and s < sl:
            u = np.clip(s / sl, 0.0, 1.0)
            p = max(0.2, 1.0 + self.start_ease * 1.5)
            e = 1.0 - (1.0 - u) ** p
            w = self.start_width + (1.0 - self.start_width) * e
        if el > 1e-5 and (total_len - s) < el:
            u = np.clip((total_len - s) / el, 0.0, 1.0)
            p = max(0.2, 1.0 + self.end_ease * 1.5)
            e = 1.0 - (1.0 - u) ** p
            w = min(w, self.end_width + (1.0 - self.end_width) * e)
        return float(max(0.0, w))


class Wave:
    """Ported from EffectCraft `varstroke::Wave`."""
    def __init__(self, amount: float = 0.0, cycles: float = 3.0, phase_deg: float = 0.0):
        self.amount = float(amount)
        self.cycles = float(cycles)
        self.phase_deg = float(phase_deg)

    def width_factor(self, s: float, total_len: float) -> float:
        if self.amount <= 1e-4 or total_len <= 1e-6:
            return 1.0
        ang = 2.0 * math.pi * self.cycles * (s / total_len) + math.radians(self.phase_deg)
        return 1.0 - self.amount * 0.5 * (1.0 - math.cos(ang))


def draw_tapered_stroke(
    ctx: cairo.Context,
    pts: np.ndarray | list,
    base_width: float,
    color_rgba: tuple[float, float, float, float],
    taper: Taper | dict | None = None,
    wave: Wave | dict | None = None,
    progress: float = 1.0,
) -> None:
    """
    Render a polyline `pts` (N x 2) with variable-width `Taper` and `Wave` profile as a union of
    trapezoids and round-join discs in PyCairo.
    """
    arr = np.asarray(pts, dtype=np.float32)
    if len(arr) < 2 or progress <= 0.002 or base_width <= 0.05:
        return
    if isinstance(taper, dict):
        taper = Taper(**taper)
    elif taper is None:
        taper = Taper()
    if isinstance(wave, dict):
        wave = Wave(**wave)

    diffs = np.diff(arr, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    tot = float(cum[-1])
    if tot <= 1e-4:
        return

    vis_len = tot * min(1.0, max(0.0, float(progress)))
    # Collect visible vertices up to vis_len
    v_pts = [arr[0]]
    v_s = [0.0]
    for i in range(len(seg_lens)):
        if cum[i + 1] <= vis_len:
            v_pts.append(arr[i + 1])
            v_s.append(float(cum[i + 1]))
        else:
            rem = vis_len - cum[i]
            if rem > 1e-4 and seg_lens[i] > 1e-5:
                f = rem / seg_lens[i]
                v_pts.append(arr[i] + f * diffs[i])
                v_s.append(vis_len)
            break
    if len(v_pts) < 2:
        return

    # v15.1: RESAMPLE along arc length so the taper/wave profile is smooth even on coarse polylines
    # (v15.0 evaluated the width only at the input vertices → stepped tapers on a 7-vertex heptagon comet).
    vp = resample_polyline(np.asarray(v_pts, dtype=np.float32), step=max(1.5, base_width * 0.25))
    dd = np.hypot(*np.diff(vp, axis=0).T)
    v_s = np.concatenate([[0.0], np.cumsum(dd)])
    radii = []
    for s_val in v_s:
        wf = taper.width_factor(float(s_val), vis_len)
        if wave is not None:
            wf *= wave.width_factor(float(s_val), vis_len)
        radii.append(0.5 * base_width * wf)

    # per-vertex MITER normals: m = normalise(n_prev + n_next), offset r / cos(φ/2) (clamped at 2.2 r) → the ribbon
    # keeps its full width through sharp corners (v15.0 used a plain tangent average → pinched joints)
    seg = np.diff(vp, axis=0)
    seg /= (np.hypot(seg[:, 0], seg[:, 1])[:, None] + 1e-6)
    sn = np.column_stack([-seg[:, 1], seg[:, 0]])
    n_prev = np.vstack([sn[:1], sn])
    n_next = np.vstack([sn, sn[-1:]])
    normals = n_prev + n_next
    normals /= (np.hypot(normals[:, 0], normals[:, 1])[:, None] + 1e-6)
    cos_half = np.clip(np.sum(normals * n_next, axis=1), 1.0 / 2.2, 1.0)
    r_arr = (np.asarray(radii, dtype=np.float32) / cos_half)[:, None]

    left_side = vp + normals * r_arr
    right_side = vp - normals * r_arr

    ctx.save()
    ctx.new_path()
    ctx.move_to(float(left_side[0, 0]), float(left_side[0, 1]))
    for pt in left_side[1:]:
        ctx.line_to(float(pt[0]), float(pt[1]))
    for pt in right_side[::-1]:
        ctx.line_to(float(pt[0]), float(pt[1]))
    ctx.close_path()
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)      # self-overlapping loops stay solid (no even-odd holes)
    ctx.set_source_rgba(*color_rgba)
    ctx.fill()
    ctx.restore()


def resample_polyline(pts: np.ndarray, step: float = 2.0, max_pts: int = 1200) -> np.ndarray:
    """Uniform arc-length resampling of an open polyline (keeps both end points)."""
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return pts
    d = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    tot = float(cum[-1])
    if tot <= 1e-6:
        return pts[:1].repeat(2, 0)
    n = int(np.clip(math.ceil(tot / step) + 1, 2, max_pts))
    tq = np.linspace(0.0, tot, n)
    # keep every ORIGINAL vertex (corners stay sharp), drop samples closer than step/3 to a vertex
    tq = tq[np.min(np.abs(tq[:, None] - cum[None, :]), axis=1) > step / 3.0]
    tq = np.unique(np.concatenate([tq, cum]))
    tq = tq[np.concatenate([[True], np.diff(tq) > 1e-4])]
    return np.column_stack([np.interp(tq, cum, pts[:, 0]), np.interp(tq, cum, pts[:, 1])]).astype(np.float32)


def polyline_slice(pts: np.ndarray, s0: float, s1: float, closed: bool = True, step: float = 2.0) -> np.ndarray:
    """Sub-path between arc lengths s0 → s1 (wraps around a closed contour) — comets / trim-paths."""
    pts = np.asarray(pts, dtype=np.float32)
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-3:
        pts = np.vstack([pts, pts[:1]])
    d = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    tot = float(cum[-1])
    if tot <= 1e-6:
        return pts[:2]
    n = int(np.clip(math.ceil(abs(s1 - s0) / step) + 1, 2, 1200))
    sq = np.linspace(s0, s1, n)
    sq = np.mod(sq, tot) if closed else np.clip(sq, 0, tot)
    return np.column_stack([np.interp(sq, cum, pts[:, 0]), np.interp(sq, cum, pts[:, 1])]).astype(np.float32)


def variable_feather_mask(W: int, H: int, poly_pts: np.ndarray | list,
                          vertex_radii: list[float]) -> np.ndarray:
    """
    Ported from EffectCraft `crates/path/src/feather.rs`:
    Computes an anti-aliased float32 mask (H, W) in 0..1 for closed polygon `poly_pts` where each
    vertex `i` has its own feather radius `vertex_radii[i]` (px), smoothly interpolated along the contour.
    """
    pts = np.asarray(poly_pts, dtype=np.float32)
    mask_u8 = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask_u8, [np.round(pts).astype(np.int32)], 255)
    if not vertex_radii or max(vertex_radii) <= 1.0:
        return mask_u8.astype(np.float32) / 255.0

    # Distance from boundary (signed: positive outside, negative inside)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(255 - mask_u8, cv2.DIST_L2, 3)
    sdist = dist_out - dist_in  # < 0 inside, > 0 outside

    # Interpolate local feather radius at grid (1/4 resolution for speed)
    hs, ws = max(8, H // 4), max(8, W // 4)
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32) * 4.0
    r_arr = np.asarray(vertex_radii, dtype=np.float32)
    weights = np.zeros((hs, ws), dtype=np.float32)
    w_sum = np.zeros((hs, ws), dtype=np.float32) + 1e-6
    for (vx, vy), vr in zip(pts, r_arr):
        d2 = (xx - vx) ** 2 + (yy - vy) ** 2 + 64.0
        w = 1.0 / d2
        weights += w * max(0.5, float(vr))
        w_sum += w
    rad_map = cv2.resize(weights / w_sum, (W, H), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(0.5 - sdist / np.maximum(1.0, rad_map), 0.0, 1.0)
    return alpha.astype(np.float32)
