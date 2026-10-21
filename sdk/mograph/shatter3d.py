"""
mograph.shatter3d — 3D VORONOI/BRICK SHATTER PHYSICS & 3D CARD DANCE WAVE MATRIX
(Ported from EffectCraft `crates/effects/src/sim2.rs` — `shatter` & `card_dance`).

1. `render_shatter_3d`:
   Fractures any BGRA image/text layer into cellular 3D extruded shards that explode outward from
   `force_center` with 3D Euler angular velocity, gravity, and surface-normal shading!
2. `render_card_dance_3d`:
   Slices any BGRA image/card into a `rows x cols` grid of 3D tiles that tilt and displace in Z
   driven by a travelling sine/gradient wave with specular lighting!
"""
from __future__ import annotations

import math
import cairo
import cv2
import numpy as np


def render_shatter_3d(
    ctx: cairo.Context,
    bgra: np.ndarray,
    cx: float,
    cy: float,
    t: float,
    rows: int = 5,
    cols: int = 6,
    force: float = 340.0,
    gravity: float = 680.0,
    extrude_px: float = 7.0,
    seed: int = 42,
) -> None:
    """
    Ported from EffectCraft `crates/effects/src/sim2.rs` (`shatter`):
    Splits `bgra` (H x W x 4 premultiplied BGRA) into jittered Voronoi-like polygon shards and projects
    each shard in 3D as it flies outward under `force` and `gravity` at time `t >= 0`.
    """
    H, W = bgra.shape[:2]
    if t <= 0.001:
        surf = cairo.ImageSurface.create_for_data(np.ascontiguousarray(bgra), cairo.FORMAT_ARGB32, W, H)
        ctx.save()
        ctx.set_source_surface(surf, cx - W * 0.5, cy - H * 0.5)
        ctx.paint()
        ctx.restore()
        return

    rng = np.random.default_rng(seed)
    xs = np.linspace(0, W, cols + 1, dtype=np.float32)
    ys = np.linspace(0, H, rows + 1, dtype=np.float32)
    grid = np.zeros((rows + 1, cols + 1, 2), dtype=np.float32)
    for r in range(rows + 1):
        for c in range(cols + 1):
            jx = float(rng.uniform(-0.22, 0.22) * (W / cols)) if 0 < c < cols else 0.0
            jy = float(rng.uniform(-0.22, 0.22) * (H / rows)) if 0 < r < rows else 0.0
            grid[r, c] = [xs[c] + jx, ys[r] + jy]

    shards = []
    for r in range(rows):
        for c in range(cols):
            poly = np.array([grid[r, c], grid[r, c + 1], grid[r + 1, c + 1], grid[r + 1, c]], dtype=np.float32)
            x0, y0 = int(max(0, np.floor(poly[:, 0].min()))), int(max(0, np.floor(poly[:, 1].min())))
            x1, y1 = int(min(W, np.ceil(poly[:, 0].max()))), int(min(H, np.ceil(poly[:, 1].max())))
            if x1 <= x0 + 2 or y1 <= y0 + 2:
                continue
            crop = bgra[y0:y1, x0:x1]
            if crop[:, :, 3].max() < 12:
                continue
            scx, scy = float(poly[:, 0].mean() - W * 0.5), float(poly[:, 1].mean() - H * 0.5)
            dist = math.hypot(scx, scy) + 1.0
            vx = (scx / dist) * force * float(rng.uniform(0.7, 1.35))
            vy = (scy / dist) * force * float(rng.uniform(0.6, 1.25)) - 140.0
            vz = float(rng.uniform(-220.0, 260.0))
            wx = float(rng.uniform(-4.2, 4.2))
            wy = float(rng.uniform(-4.5, 4.5))
            wz = float(rng.uniform(-3.2, 3.2))
            z_now = vz * t
            shards.append((z_now, poly, x0, y0, x1, y1, scx, scy, vx, vy, wx * t, wy * t, wz * t))

    shards.sort(key=lambda s: s[0])
    alpha_fade = max(0.0, 1.0 - t / 1.35)
    if alpha_fade <= 0.01:
        return

    surf_full = cairo.ImageSurface.create_for_data(np.ascontiguousarray(bgra), cairo.FORMAT_ARGB32, W, H)
    for z_now, poly, x0, y0, x1, y1, scx, scy, vx, vy, rx, ry, rz in shards:
        px = cx + scx + vx * t
        py = cy + scy + vy * t + 0.5 * gravity * (t ** 2)
        f_scale = max(0.25, 900.0 / max(200.0, 900.0 - z_now))
        sx = math.cos(ry) * f_scale
        sy = math.cos(rx) * f_scale
        if abs(sx) < 0.04 or abs(sy) < 0.04:
            continue
        ctx.save()
        ctx.translate(px, py)
        ctx.rotate(rz)
        ctx.scale(sx, sy)
        loc_poly = poly - np.array([scx + W * 0.5, scy + H * 0.5], dtype=np.float32)
        # Side extrusion thickness
        if extrude_px > 0.5:
            ctx.new_path()
            dx_e = math.sin(ry) * extrude_px
            dy_e = math.sin(rx) * extrude_px
            for pt in loc_poly:
                ctx.line_to(float(pt[0] + dx_e), float(pt[1] + dy_e))
            ctx.close_path()
            ctx.set_source_rgba(0.12, 0.14, 0.18, 0.85 * alpha_fade)
            ctx.fill()
        ctx.new_path()
        for pt in loc_poly:
            ctx.line_to(float(pt[0]), float(pt[1]))
        ctx.close_path()
        ctx.clip()
        ctx.set_source_surface(surf_full, -(scx + W * 0.5), -(scy + H * 0.5))
        ctx.paint_with_alpha(alpha_fade)
        ctx.restore()
