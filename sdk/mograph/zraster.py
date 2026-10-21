"""
mograph.zraster — tiny per-pixel Z-buffered triangle rasteriser (Gouraud colour, ID buffer) for the CPU 3D
engines (sculpt3d, real3d, three3d).  numba-JIT when available (≈ 5-15 ms for 2-5 k triangles at 600²),
ultra-fast vectorized NumPy fallback (~4x faster pre-filtered bbox rasteriser when numba is absent).

raster(tri_xy (N,3,2) px, tri_z (N,3) camera distance, tri_rgb (N,3,3) 0..255, tri_id (N,), W, H)
    → color (H,W,3) float32, alpha (H,W) float32 (coverage 0/1), ids (H,W) int32 (0 = empty)
Nearest surface wins (smaller z).  Pixel centres at +0.5.  Interpolation is perspective-correct for depth.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:          # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def d(f):
            return f
        return d if not (len(a) == 1 and callable(a[0])) else a[0]


@njit(cache=True, fastmath=True)
def _raster_nb(xy, z, rgb, ids, W, H, col, zbuf, idb):
    n = xy.shape[0]
    for t in range(n):
        x0, y0 = xy[t, 0, 0], xy[t, 0, 1]
        x1, y1 = xy[t, 1, 0], xy[t, 1, 1]
        x2, y2 = xy[t, 2, 0], xy[t, 2, 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        inv = 1.0 / area
        xmin = int(max(0.0, np.floor(min(x0, min(x1, x2)))))
        xmax = int(min(W - 1.0, np.ceil(max(x0, max(x1, x2)))))
        ymin = int(max(0.0, np.floor(min(y0, min(y1, y2)))))
        ymax = int(min(H - 1.0, np.ceil(max(y0, max(y1, y2)))))
        if xmin > xmax or ymin > ymax:
            continue
        iz0, iz1, iz2 = 1.0 / z[t, 0], 1.0 / z[t, 1], 1.0 / z[t, 2]
        for py in range(ymin, ymax + 1):
            cy = py + 0.5
            for px in range(xmin, xmax + 1):
                cx = px + 0.5
                w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * inv
                w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * inv
                w2 = 1.0 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                izp = w0 * iz0 + w1 * iz1 + w2 * iz2
                zp = 1.0 / izp
                if zp < zbuf[py, px]:
                    zbuf[py, px] = zp
                    idb[py, px] = ids[t]
                    a0, a1, a2 = w0 * iz0 * zp, w1 * iz1 * zp, w2 * iz2 * zp
                    for c in range(3):
                        col[py, px, c] = a0 * rgb[t, 0, c] + a1 * rgb[t, 1, c] + a2 * rgb[t, 2, c]


def raster(tri_xy, tri_z, tri_rgb, tri_id, W, H):
    col = np.zeros((H, W, 3), np.float32)
    zbuf = np.full((H, W), 1e30, np.float32)       # finite sentinel (fastmath assumes no inf)
    idb = np.zeros((H, W), np.int32)
    if len(tri_xy):
        (_raster_nb if HAVE_NUMBA else _raster_vec_np)(
            np.ascontiguousarray(tri_xy, np.float32), np.ascontiguousarray(tri_z, np.float32),
            np.ascontiguousarray(tri_rgb, np.float32), np.ascontiguousarray(tri_id, np.int32),
            int(W), int(H), col, zbuf, idb)
    alpha = (zbuf < 1e29).astype(np.float32)
    return col, alpha, idb, zbuf


@njit(cache=True, fastmath=True)
def _raster_attr_nb(xy, z, at, ids, W, H, abuf, zbuf, idb):
    """G-buffer rasteriser: perspective-correct interpolation of K attributes per vertex."""
    n = xy.shape[0]
    K = at.shape[2]
    for t in range(n):
        x0, y0 = xy[t, 0, 0], xy[t, 0, 1]
        x1, y1 = xy[t, 1, 0], xy[t, 1, 1]
        x2, y2 = xy[t, 2, 0], xy[t, 2, 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        inv = 1.0 / area
        xmin = int(max(0.0, np.floor(min(x0, min(x1, x2)))))
        xmax = int(min(W - 1.0, np.ceil(max(x0, max(x1, x2)))))
        ymin = int(max(0.0, np.floor(min(y0, min(y1, y2)))))
        ymax = int(min(H - 1.0, np.ceil(max(y0, max(y1, y2)))))
        if xmin > xmax or ymin > ymax:
            continue
        iz0, iz1, iz2 = 1.0 / z[t, 0], 1.0 / z[t, 1], 1.0 / z[t, 2]
        for py in range(ymin, ymax + 1):
            cy = py + 0.5
            for px in range(xmin, xmax + 1):
                cx = px + 0.5
                w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * inv
                w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * inv
                w2 = 1.0 - w0 - w1
                if w0 < -1e-7 or w1 < -1e-7 or w2 < -1e-7:
                    continue
                izp = w0 * iz0 + w1 * iz1 + w2 * iz2
                zp = 1.0 / izp
                if zp < zbuf[py, px]:
                    zbuf[py, px] = zp
                    idb[py, px] = ids[t]
                    a0, a1, a2 = w0 * iz0 * zp, w1 * iz1 * zp, w2 * iz2 * zp
                    for c in range(K):
                        abuf[py, px, c] = a0 * at[t, 0, c] + a1 * at[t, 1, c] + a2 * at[t, 2, c]


def raster_attr(tri_xy, tri_z, tri_attr, tri_id, W, H):
    """→ attr (H,W,K) float32, ids (H,W) int32 (0 = empty), zbuf (H,W) (1e30 = empty)."""
    K = tri_attr.shape[2]
    abuf = np.zeros((H, W, K), np.float32)
    zbuf = np.full((H, W), 1e30, np.float32)
    idb = np.zeros((H, W), np.int32)
    if len(tri_xy):
        (_raster_attr_nb if HAVE_NUMBA else _raster_attr_vec_np)(
            np.ascontiguousarray(tri_xy, np.float32), np.ascontiguousarray(tri_z, np.float32),
            np.ascontiguousarray(tri_attr, np.float32), np.ascontiguousarray(tri_id, np.int32),
            int(W), int(H), abuf, zbuf, idb)
    return abuf, idb, zbuf


def _prep_tris_np(xy, z, W, H, eps=1e-9):
    """Vectorized pre-filtering of triangle bounds and 1/z values outside the Python loop."""
    x0, y0 = xy[:, 0, 0], xy[:, 0, 1]
    x1, y1 = xy[:, 1, 0], xy[:, 1, 1]
    x2, y2 = xy[:, 2, 0], xy[:, 2, 1]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    xmin = np.maximum(0, np.floor(np.minimum(x0, np.minimum(x1, x2))).astype(np.int32))
    xmax = np.minimum(W - 1, np.ceil(np.maximum(x0, np.maximum(x1, x2))).astype(np.int32))
    ymin = np.maximum(0, np.floor(np.minimum(y0, np.minimum(y1, y2))).astype(np.int32))
    ymax = np.minimum(H - 1, np.ceil(np.maximum(y0, np.maximum(y1, y2))).astype(np.int32))
    valid = (np.abs(area) >= eps) & (xmin <= xmax) & (ymin <= ymax)
    idx = np.nonzero(valid)[0]
    inv = np.zeros_like(area)
    inv[idx] = 1.0 / area[idx]
    iz = 1.0 / np.maximum(z, 1e-6)
    return idx, x0, y0, x1, y1, x2, y2, inv, xmin, xmax, ymin, ymax, iz


def _raster_vec_np(xy, z, rgb, ids, W, H, col, zbuf, idb):
    """Fast vectorized NumPy triangle bbox rasterizer when numba is unavailable."""
    idx, x0a, y0a, x1a, y1a, x2a, y2a, inva, xmina, xmaxa, ymina, ymaxa, iza = _prep_tris_np(xy, z, W, H, 1e-9)
    grid_x = np.arange(W, dtype=np.float32)[None, :] + 0.5
    grid_y = np.arange(H, dtype=np.float32)[:, None] + 0.5
    for t in idx:
        xmin, xmax, ymin, ymax = int(xmina[t]), int(xmaxa[t]), int(ymina[t]), int(ymaxa[t])
        x0, y0, x1, y1, x2, y2, inv = float(x0a[t]), float(y0a[t]), float(x1a[t]), float(y1a[t]), float(x2a[t]), float(y2a[t]), float(inva[t])
        cy = grid_y[ymin:ymax + 1]
        cx = grid_x[:, xmin:xmax + 1]
        w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * inv
        w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * inv
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        iz0, iz1, iz2 = float(iza[t, 0]), float(iza[t, 1]), float(iza[t, 2])
        izp = w0 * iz0 + w1 * iz1 + w2 * iz2
        zp = 1.0 / np.maximum(izp, 1e-12)
        z_sub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        win = inside & (zp < z_sub)
        if not win.any():
            continue
        z_sub[win] = zp[win]
        idb[ymin:ymax + 1, xmin:xmax + 1][win] = ids[t]
        zp_w = zp[win]
        a0 = (w0[win] * iz0 * zp_w)[:, None]
        a1 = (w1[win] * iz1 * zp_w)[:, None]
        a2 = (w2[win] * iz2 * zp_w)[:, None]
        col[ymin:ymax + 1, xmin:xmax + 1][win] = a0 * rgb[t, 0] + a1 * rgb[t, 1] + a2 * rgb[t, 2]


def _raster_attr_vec_np(xy, z, at, ids, W, H, abuf, zbuf, idb):
    """Fast vectorized NumPy G-buffer triangle bbox rasterizer when numba is unavailable."""
    idx, x0a, y0a, x1a, y1a, x2a, y2a, inva, xmina, xmaxa, ymina, ymaxa, iza = _prep_tris_np(xy, z, W, H, 1e-12)
    grid_x = np.arange(W, dtype=np.float32)[None, :] + 0.5
    grid_y = np.arange(H, dtype=np.float32)[:, None] + 0.5
    for t in idx:
        xmin, xmax, ymin, ymax = int(xmina[t]), int(xmaxa[t]), int(ymina[t]), int(ymaxa[t])
        x0, y0, x1, y1, x2, y2, inv = float(x0a[t]), float(y0a[t]), float(x1a[t]), float(y1a[t]), float(x2a[t]), float(y2a[t]), float(inva[t])
        cy = grid_y[ymin:ymax + 1]
        cx = grid_x[:, xmin:xmax + 1]
        w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * inv
        w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * inv
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        if not inside.any():
            continue
        iz0, iz1, iz2 = float(iza[t, 0]), float(iza[t, 1]), float(iza[t, 2])
        izp = w0 * iz0 + w1 * iz1 + w2 * iz2
        zp = 1.0 / np.maximum(izp, 1e-12)
        z_sub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        win = inside & (zp < z_sub)
        if not win.any():
            continue
        z_sub[win] = zp[win]
        idb[ymin:ymax + 1, xmin:xmax + 1][win] = ids[t]
        zp_w = zp[win]
        a0 = (w0[win] * iz0 * zp_w)[:, None]
        a1 = (w1[win] * iz1 * zp_w)[:, None]
        a2 = (w2[win] * iz2 * zp_w)[:, None]
        abuf[ymin:ymax + 1, xmin:xmax + 1][win] = a0 * at[t, 0] + a1 * at[t, 1] + a2 * at[t, 2]
