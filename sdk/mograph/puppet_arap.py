"""
mograph.puppet_arap — SIGGRAPH 2005 AS-RIGID-AS-POSSIBLE (ARAP) 2D PUPPET MESH DEFORMATION
(Ported from EffectCraft `crates/effects/src/puppet/{arap.rs, mesh.rs}`).

Implements Igarashi, Moscovich & Hughes, "As-Rigid-As-Possible Shape Manipulation" (SIGGRAPH 2005):
  1. Triangulates an RGBA image's alpha silhouette (or grid) into a 2D triangle mesh `(V_rest, tris)`.
  2. Step 1 (Laplacian / Similarity solve): solves the sparse symmetric positive-definite system with
     soft/hard positional pin constraints `(pin_idx, target_xy, weight)` and optional `Starch` triangle
     stiffness weights.
  3. Step 2 (2D Procrustes Scale-Adjustment step): fits each deformed triangle with a rigid rotated
     copy of its rest shape via 2D Procrustes rotation about its centroid, then solves the Poisson edge-vector
     system so triangles rotate rigidly without unnatural shearing or volume loss!
"""
from __future__ import annotations

import math
import cairo
import cv2
import numpy as np
from scipy import sparse


class ArapPuppetMesh:
    """Pre-built 2D triangle mesh with cached rest edge vectors for fast per-frame ARAP deformation."""
    def __init__(self, rgba: np.ndarray, grid_nx: int = 12, grid_ny: int = 12, starch_weights: np.ndarray | None = None):
        self.rgba = rgba
        self.H, self.W = rgba.shape[:2]
        xs = np.linspace(0.0, float(self.W - 1), grid_nx, dtype=np.float32)
        ys = np.linspace(0.0, float(self.H - 1), grid_ny, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        self.V_rest = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
        self.n_v = len(self.V_rest)

        tris = []
        for r in range(grid_ny - 1):
            for c in range(grid_nx - 1):
                i00 = r * grid_nx + c
                i10 = i00 + 1
                i01 = (r + 1) * grid_nx + c
                i11 = i01 + 1
                tris.append([i00, i10, i11])
                tris.append([i00, i11, i01])
        self.tris = np.asarray(tris, dtype=np.int32)
        self.n_t = len(self.tris)
        self.tw = np.ones(self.n_t, dtype=np.float64) if starch_weights is None else np.asarray(starch_weights, dtype=np.float64)

        # Build base graph Laplacian L (n_v x n_v) weighted by triangle stiffness
        rows, cols, vals = [], [], []
        for ti, (i, j, k) in enumerate(self.tris):
            w = float(self.tw[ti])
            for a, b in ((i, j), (j, k), (k, i)):
                rows += [a, b, a, b]
                cols += [a, b, b, a]
                vals += [w, w, -w, -w]
        self.L_base = sparse.coo_matrix((vals, (rows, cols)), shape=(self.n_v, self.n_v)).tocsr()

    def nearest_vertex(self, u_norm: float, v_norm: float) -> int:
        px, py = u_norm * (self.W - 1), v_norm * (self.H - 1)
        d2 = (self.V_rest[:, 0] - px) ** 2 + (self.V_rest[:, 1] - py) ** 2
        return int(np.argmin(d2))

    def solve(self, pins: list[tuple[int, tuple[float, float], float]], iterations: int = 8,
              tol: float = 0.05) -> np.ndarray:
        """
        Local/global ARAP for pin targets `[(vertex_idx, (tx, ty), weight), ...]` → deformed vertices (n_v x 2).

        v15.1 fixes (v15.0 did not converge: a pure 90° rotation of two pins left 78 px error after 4 iterations,
        103 px after 1 — the mesh sheared instead of turning):
          * WARM START with the best similarity transform (2D Procrustes rotation + uniform scale + translation)
            mapping the pinned rest vertices to their targets → rigid motions are exact at iteration 0;
          * the system matrix (L + P) is constant → factorised ONCE (splu) and re-used every iteration;
          * the per-triangle right-hand side is assembled vectorised (np.add.at) instead of a Python loop;
          * early exit when the largest vertex update < `tol` px.
        """
        if not pins:
            return self.V_rest.astype(np.float32)
        from scipy.sparse.linalg import splu
        p_idx = np.array([int(p[0]) for p in pins])
        p_xy = np.array([p[1] for p in pins], dtype=np.float64)
        p_w = np.array([float(p[2]) for p in pins], dtype=np.float64)

        P_mat = sparse.coo_matrix((p_w, (p_idx, p_idx)), shape=(self.n_v, self.n_v)).tocsc()
        reg = sparse.eye(self.n_v, format="csc") * 1e-6
        lu = splu((self.L_base.tocsc() + P_mat + reg).tocsc())

        # ---- warm start: similarity transform of the pins
        src = self.V_rest[p_idx]
        cs, ct = src.mean(0), p_xy.mean(0)
        if len(pins) >= 2:
            A0, B0 = src - cs, p_xy - ct
            num = float(np.sum(A0[:, 0] * B0[:, 1] - A0[:, 1] * B0[:, 0]))
            den = float(np.sum(A0[:, 0] * B0[:, 0] + A0[:, 1] * B0[:, 1]))
            th = math.atan2(num, den)
            sc = math.sqrt(float(np.sum(B0 ** 2)) / max(1e-9, float(np.sum(A0 ** 2))))
        else:
            th, sc = 0.0, 1.0
        c, s_ = math.cos(th) * sc, math.sin(th) * sc
        Rs = np.array([[c, -s_], [s_, c]])
        V_cur = (self.V_rest - cs) @ Rs.T + ct

        i_idx, j_idx, k_idx = self.tris[:, 0], self.tris[:, 1], self.tris[:, 2]
        rest_c = (self.V_rest[i_idx] + self.V_rest[j_idx] + self.V_rest[k_idx]) / 3.0
        P0 = np.stack([self.V_rest[i_idx] - rest_c, self.V_rest[j_idx] - rest_c, self.V_rest[k_idx] - rest_c], axis=1)
        w = self.tw[:, None]
        pin_rhs = np.zeros_like(self.V_rest)
        np.add.at(pin_rhs, p_idx, p_w[:, None] * p_xy)

        for _ in range(max(1, iterations)):
            # local step: best rotation per triangle (closed-form 2D Procrustes)
            cur_c = (V_cur[i_idx] + V_cur[j_idx] + V_cur[k_idx]) / 3.0
            Q0 = np.stack([V_cur[i_idx] - cur_c, V_cur[j_idx] - cur_c, V_cur[k_idx] - cur_c], axis=1)
            num = np.sum(P0[..., 0] * Q0[..., 1] - P0[..., 1] * Q0[..., 0], axis=1)
            den = np.sum(P0[..., 0] * Q0[..., 0] + P0[..., 1] * Q0[..., 1], axis=1) + 1e-12
            ang = np.arctan2(num, den)
            ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
            rot = lambda v: np.stack([ca[:, 0] * v[:, 0] - sa[:, 0] * v[:, 1], sa[:, 0] * v[:, 0] + ca[:, 0] * v[:, 1]], 1)  # noqa: E731
            ri, rj, rk = rot(P0[:, 0]), rot(P0[:, 1]), rot(P0[:, 2])
            # global step: Poisson solve with the rotated rest edges
            rhs = reg @ V_cur + pin_rhs
            np.add.at(rhs, i_idx, w * ((ri - rj) + (ri - rk)))
            np.add.at(rhs, j_idx, w * ((rj - ri) + (rj - rk)))
            np.add.at(rhs, k_idx, w * ((rk - ri) + (rk - rj)))
            V_new = np.column_stack([lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])])
            delta = float(np.abs(V_new - V_cur).max())
            V_cur = V_new
            if delta < tol:
                break
        return V_cur.astype(np.float32)

    def warp_image(self, V_def: np.ndarray) -> np.ndarray:
        """Piecewise-affine warp of `self.rgba` from `self.V_rest` to `V_def`."""
        out = np.zeros_like(self.rgba)
        V0 = self.V_rest.astype(np.float32)
        for i, j, k in self.tris:
            src_t = np.array([V0[i], V0[j], V0[k]], dtype=np.float32)
            dst_t = np.array([V_def[i], V_def[j], V_def[k]], dtype=np.float32)
            r_src = cv2.boundingRect(src_t)
            r_dst = cv2.boundingRect(dst_t)
            sx, sy, sw, sh = r_src
            dx, dy, dw, dh = r_dst
            if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
                continue
            sx0, sy0 = max(0, sx), max(0, sy)
            sx1, sy1 = min(self.W, sx + sw), min(self.H, sy + sh)
            dx0, dy0 = max(0, dx), max(0, dy)
            dx1, dy1 = min(self.W, dx + dw), min(self.H, dy + dh)
            if sx1 <= sx0 or sy1 <= sy0 or dx1 <= dx0 or dy1 <= dy0:
                continue
            M = cv2.getAffineTransform(src_t - np.array([sx0, sy0], np.float32),
                                       dst_t - np.array([dx0, dy0], np.float32))
            warped_patch = cv2.warpAffine(self.rgba[sy0:sy1, sx0:sx1], M, (dx1 - dx0, dy1 - dy0),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = np.zeros((dy1 - dy0, dx1 - dx0), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(dst_t - np.array([dx0, dy0], np.float32)).astype(np.int32), 255)
            roi = out[dy0:dy1, dx0:dx1]
            m_bool = mask > 0
            roi[m_bool] = warped_patch[m_bool]
        return out
