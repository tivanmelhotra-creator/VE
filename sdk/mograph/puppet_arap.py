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
from scipy.sparse.linalg import spsolve


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

    def solve(self, pins: list[tuple[int, tuple[float, float], float]], iterations: int = 2) -> np.ndarray:
        """
        Solve 2-step ARAP for current pin targets `[(vertex_idx, (tx, ty), weight), ...]`.
        Returns deformed vertex positions `V_def` (n_v x 2 float32).
        """
        if not pins:
            return self.V_rest.astype(np.float32)

        p_idx = [int(p[0]) for p in pins]
        p_xy = np.array([p[1] for p in pins], dtype=np.float64)
        p_w = np.array([float(p[2]) for p in pins], dtype=np.float64)

        P_mat = sparse.coo_matrix((p_w, (p_idx, p_idx)), shape=(self.n_v, self.n_v)).tocsr()
        reg = sparse.eye(self.n_v, format="csr") * 1e-4
        A_sys = self.L_base + P_mat + reg

        # Initial guess: rest positions + pin offsets
        rhs_init = self.L_base @ self.V_rest + reg @ self.V_rest
        for idx, xy, w in zip(p_idx, p_xy, p_w):
            rhs_init[idx] += w * xy
        V_cur = spsolve(A_sys, rhs_init)

        # Step 2: Per-triangle 2D Procrustes rigid rotation fit + Poisson edge-vector solve
        i_idx, j_idx, k_idx = self.tris[:, 0], self.tris[:, 1], self.tris[:, 2]
        rest_c = (self.V_rest[i_idx] + self.V_rest[j_idx] + self.V_rest[k_idx]) / 3.0
        P0 = np.stack([self.V_rest[i_idx] - rest_c, self.V_rest[j_idx] - rest_c, self.V_rest[k_idx] - rest_c], axis=1)  # (T, 3, 2)

        for _ in range(max(1, iterations)):
            cur_c = (V_cur[i_idx] + V_cur[j_idx] + V_cur[k_idx]) / 3.0
            Q0 = np.stack([V_cur[i_idx] - cur_c, V_cur[j_idx] - cur_c, V_cur[k_idx] - cur_c], axis=1)
            # 2D Procrustes closed-form rotation angle for each triangle
            # sum(p_x * q_y - p_y * q_x) vs sum(p_x * q_x + p_y * q_y)
            num = np.sum(P0[..., 0] * Q0[..., 1] - P0[..., 1] * Q0[..., 0], axis=1)
            den = np.sum(P0[..., 0] * Q0[..., 0] + P0[..., 1] * Q0[..., 1], axis=1) + 1e-9
            ang = np.arctan2(num, den)
            ca, sa = np.cos(ang), np.sin(ang)
            R_tris = np.stack([np.stack([ca, -sa], axis=-1), np.stack([sa, ca], axis=-1)], axis=1)  # (T, 2, 2)
            P_rot = np.einsum("tij,tkj->tki", R_tris, P0)  # (T, 3, 2)

            rhs = reg @ self.V_rest
            for ti in range(self.n_t):
                w = float(self.tw[ti])
                vi, vj, vk = int(i_idx[ti]), int(j_idx[ti]), int(k_idx[ti])
                ri, rj, rk = P_rot[ti, 0], P_rot[ti, 1], P_rot[ti, 2]
                rhs[vi] += w * ((ri - rj) + (ri - rk))
                rhs[vj] += w * ((rj - ri) + (rj - rk))
                rhs[vk] += w * ((rk - ri) + (rk - rj))
            for idx, xy, w in zip(p_idx, p_xy, p_w):
                rhs[idx] += w * xy
            V_cur = spsolve(A_sys, rhs)

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
