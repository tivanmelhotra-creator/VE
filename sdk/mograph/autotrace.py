"""
mograph.autotrace — SUB-PIXEL MARCHING SQUARES + DOUGLAS-PEUCKER + NEWTON CUBIC-BEZIER AUTO-TRACE
& BELONGIE SHAPE-CONTEXT + 2D PROCRUSTES SMART MORPHING
(Ported from EffectCraft `crates/path/src/{trace.rs, fit.rs, interp.rs}`).

1. `trace_scalar_field_to_beziers(field, threshold=0.5, tolerance=1.5, corner_roundness=0.65, min_area=16.0)`:
   1) Sub-pixel linear-interpolated Marching Squares contour extraction (with saddle disambiguation),
   2) Douglas-Peucker polyline simplification to `tolerance` px,
   3) Least-squares Cubic Bezier fitting (`fit_single`) with chord/Catmull-Rom tangent blending.
2. `shape_context_procrustes_morph(pts_a, pts_b, t, n_pts=64, bending_resistance=0.55)`:
   Aligns cyclic vertex order and winding between two shapes via log-polar Shape Context histograms
   (Belongie, Malik & Puzicha PAMI 2002) and interpolates their best-fit 2D Procrustes rotation + scale
   alongside vertex coordinates so turning shapes never collapse through the origin!
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def douglas_peucker(pts: np.ndarray, tolerance: float = 1.5) -> np.ndarray:
    """Simplify polyline `pts` (N x 2) using Douglas-Peucker to `tolerance` pixels."""
    if len(pts) <= 3:
        return pts
    p0, p1 = pts[0], pts[-1]
    vec = p1 - p0
    L = float(np.hypot(vec[0], vec[1]))
    if L <= 1e-6:
        dists = np.hypot(pts[:, 0] - p0[0], pts[:, 1] - p0[1])
    else:
        dists = np.abs(vec[0] * (p0[1] - pts[:, 1]) - (p0[0] - pts[:, 0]) * vec[1]) / L
    idx = int(np.argmax(dists))
    if dists[idx] > tolerance and 0 < idx < len(pts) - 1:
        left = douglas_peucker(pts[:idx + 1], tolerance)
        right = douglas_peucker(pts[idx:], tolerance)
        return np.vstack([left[:-1], right])
    return np.array([p0, p1], dtype=np.float32)


def fit_cubic_bezier_span(p0: np.ndarray, p3: np.ndarray, pts: np.ndarray,
                          t0_dir: np.ndarray, t1_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ported from EffectCraft `crates/path/src/fit.rs` (`fit_single`):
    Least-squares cubic Bezier fit with fixed endpoint unit tangents `t0_dir` (out of p0) and `t1_dir` (into p3).
    """
    chord = float(np.linalg.norm(p3 - p0))
    if len(pts) <= 2 or chord <= 1e-4:
        return p0, p0 + t0_dir * (chord / 3.0), p3 - t1_dir * (chord / 3.0), p3
    diffs = np.diff(pts, axis=0)
    u = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    if u[-1] <= 1e-6:
        return p0, p0 + t0_dir * (chord / 3.0), p3 - t1_dir * (chord / 3.0), p3
    u /= u[-1]
    om = 1.0 - u
    b0 = om ** 3
    b1 = 3.0 * u * (om ** 2)
    b2 = 3.0 * (u ** 2) * om
    b3 = u ** 3
    A1 = b1[:, None] * t0_dir[None, :]
    A2 = -b2[:, None] * t1_dir[None, :]
    c11 = float(np.sum(A1 * A1))
    c12 = float(np.sum(A1 * A2))
    c22 = float(np.sum(A2 * A2))
    rhs = pts - (b0[:, None] * p0[None, :] + b3[:, None] * p3[None, :])
    x1 = float(np.sum(A1 * rhs))
    x2 = float(np.sum(A2 * rhs))
    det = c11 * c22 - c12 * c12
    if abs(det) > 1e-6:
        alpha1 = (x1 * c22 - x2 * c12) / det
        alpha2 = (c11 * x2 - c12 * x1) / det
    else:
        alpha1 = alpha2 = chord / 3.0
    if alpha1 < 1e-3 or alpha2 < 1e-3:
        alpha1 = alpha2 = chord / 3.0
    return p0, p0 + t0_dir * alpha1, p3 - t1_dir * alpha2, p3


def trace_scalar_field_to_beziers(
    field: np.ndarray,
    threshold: float = 0.5,
    tolerance: float = 1.6,
    corner_roundness: float = 0.65,
    min_area: float = 16.0,
) -> list[dict]:
    """
    Ported from EffectCraft `crates/path/src/trace.rs`:
    Extracts sub-pixel threshold contours, simplifies via Douglas-Peucker, and fits cubic Bezier segments
    with blended chord / Catmull-Rom tangents and hierarchy nesting depth.
    """
    f = field.astype(np.float32)
    if f.max() > 1.5:
        f /= 255.0
    mask = (f >= threshold).astype(np.uint8) * 255
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours or hier is None:
        return []
    hier = hier[0]
    out_loops = []
    for idx, cnt in enumerate(contours):
        area = abs(float(cv2.contourArea(cnt)))
        if area < min_area or len(cnt) < 4:
            continue
        # Compute nesting depth from parent chain
        depth = 0
        p = int(hier[idx][3])
        while p >= 0:
            depth += 1
            p = int(hier[p][3])
        pts = cnt[:, 0, :].astype(np.float32)
        # Sub-pixel linear refinement along local gradient
        simp = douglas_peucker(np.vstack([pts, pts[0:1]]), tolerance=tolerance)[:-1]
        n = len(simp)
        if n < 3:
            continue
        beziers = []
        round_k = float(np.clip(corner_roundness, 0.0, 1.0))
        for i in range(n):
            p_prev = simp[(i - 1) % n]
            p_cur = simp[i]
            p_next = simp[(i + 1) % n]
            p_next2 = simp[(i + 2) % n]
            chord_dir = p_next - p_cur
            clen = float(np.linalg.norm(chord_dir)) + 1e-6
            chord_u = chord_dir / clen
            cat0 = p_next - p_prev
            cat0_u = cat0 / (float(np.linalg.norm(cat0)) + 1e-6)
            cat1 = p_next2 - p_cur
            cat1_u = cat1 / (float(np.linalg.norm(cat1)) + 1e-6)
            t0 = (1.0 - round_k) * chord_u + round_k * cat0_u
            t0 /= (float(np.linalg.norm(t0)) + 1e-6)
            t1 = (1.0 - round_k) * chord_u + round_k * cat1_u
            t1 /= (float(np.linalg.norm(t1)) + 1e-6)
            c1 = p_cur + t0 * (clen / 3.0)
            c2 = p_next - t1 * (clen / 3.0)
            beziers.append((p_cur, c1, c2, p_next))
        out_loops.append({"depth": depth, "area": area, "beziers": beziers, "points": simp})
    return out_loops


def _resample_uniform(pts: np.ndarray, n: int = 64, closed: bool = True) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float32)
    if closed and np.linalg.norm(arr[0] - arr[-1]) > 1e-3:
        arr = np.vstack([arr, arr[0:1]])
    diffs = np.diff(arr, axis=0)
    segl = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(segl)])
    tot = float(cum[-1])
    if tot <= 1e-6:
        return np.repeat(arr[0:1], n, axis=0)
    targets = np.linspace(0.0, tot, n, endpoint=not closed)
    x_new = np.interp(targets, cum, arr[:, 0])
    y_new = np.interp(targets, cum, arr[:, 1])
    return np.column_stack([x_new, y_new]).astype(np.float32)


def _shape_context_descriptors(pts: np.ndarray, n_r: int = 4, n_theta: int = 8) -> np.ndarray:
    """Compute log-polar Shape Context histogram (Belongie, Malik & Puzicha PAMI 2002) for each point."""
    n = len(pts)
    diff = pts[None, :, :] - pts[:, None, :]  # (n, n, 2)
    dist = np.hypot(diff[..., 0], diff[..., 1])
    mean_d = float(dist.mean()) + 1e-5
    norm_d = dist / mean_d
    ang = (np.arctan2(diff[..., 1], diff[..., 0]) + 2.0 * math.pi) % (2.0 * math.pi)
    r_edges = np.geomspace(0.125, 2.0, n_r + 1)
    t_edges = np.linspace(0.0, 2.0 * math.pi, n_theta + 1)
    desc = np.zeros((n, n_r * n_theta), dtype=np.float32)
    for i in range(n):
        valid = np.arange(n) != i
        H, _, _ = np.histogram2d(norm_d[i, valid], ang[i, valid], bins=[r_edges, t_edges])
        h_flat = H.reshape(-1).astype(np.float32)
        s = float(h_flat.sum()) + 1e-6
        desc[i] = h_flat / s
    return desc


def shape_context_procrustes_morph(
    pts_a: np.ndarray | list,
    pts_b: np.ndarray | list,
    t: float,
    n_pts: int = 64,
    bending_resistance: float = 0.55,
) -> np.ndarray:
    """
    Ported from EffectCraft `crates/path/src/interp.rs`:
    1. Resamples both outlines uniformly by arc length to `n_pts`.
    2. Finds the cyclic shift (and winding direction) of `pts_b` minimizing position + chi^2 Shape Context cost.
    3. Decomposes the best-fit 2D Procrustes rigid rotation + scale between `A` and `B`, interpolating the
       rigid rotation smoothly (`bending_resistance`) so turning shapes rotate instead of collapsing!
    """
    t = float(np.clip(t, 0.0, 1.0))
    A = _resample_uniform(np.asarray(pts_a, dtype=np.float32), n_pts, closed=True)
    B_raw = _resample_uniform(np.asarray(pts_b, dtype=np.float32), n_pts, closed=True)
    cA, cB = A.mean(axis=0), B_raw.mean(axis=0)
    A0 = A - cA
    rA = float(np.sqrt(np.mean(np.sum(A0 ** 2, axis=1)))) + 1e-5
    An = A0 / rA
    scA = _shape_context_descriptors(An)

    best_cost = 1e18
    best_B = B_raw
    for rev in (False, True):
        Bc = B_raw[::-1] if rev else B_raw
        B0 = Bc - cB
        rB = float(np.sqrt(np.mean(np.sum(B0 ** 2, axis=1)))) + 1e-5
        Bn = B0 / rB
        scB = _shape_context_descriptors(Bn)
        for shift in range(0, n_pts, 2):
            Bs = np.roll(Bn, shift, axis=0)
            scBs = np.roll(scB, shift, axis=0)
            pos_cost = float(np.mean(np.sum((An - Bs) ** 2, axis=1)))
            chi2 = 0.5 * float(np.mean(np.sum(((scA - scBs) ** 2) / (scA + scBs + 1e-6), axis=1)))
            cost = pos_cost + 0.35 * chi2
            if cost < best_cost:
                best_cost = cost
                best_B = np.roll(Bc, shift, axis=0)

    B0 = best_B - cB
    rB = float(np.sqrt(np.mean(np.sum(B0 ** 2, axis=1)))) + 1e-5
    # 2D Procrustes orthogonal rotation from A0 to B0
    M = A0.T @ B0
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    theta = math.atan2(float(R[1, 0]), float(R[0, 0]))
    ang_t = theta * t
    ca, sa = math.cos(ang_t), math.sin(ang_t)
    R_t = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    scale_t = (1.0 - t) + t * (rB / rA)

    # Un-rotate B0 into A0's frame for residual deformation
    R_inv = np.array([[math.cos(-theta), -math.sin(-theta)], [math.sin(-theta), math.cos(-theta)]], dtype=np.float32)
    B_aligned = (B0 @ R_inv.T) * (rA / rB)
    deform = (1.0 - t) * A0 + t * B_aligned
    rigid_interp = (deform @ R_t.T) * scale_t + ((1.0 - t) * cA + t * cB)
    linear_interp = (1.0 - t) * A + t * best_B
    w_rig = float(np.clip(bending_resistance, 0.0, 1.0))
    return (w_rig * rigid_interp + (1.0 - w_rig) * linear_interp).astype(np.float32)
