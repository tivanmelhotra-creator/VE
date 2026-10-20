"""Render a visual catalog of every icon (name + tags) so the agent can SEE what exists."""
from __future__ import annotations

import math

import cairo

from .geometry import draw_path, rounded_rect_d
from .icons import ICONS, draw_icon
from .text import draw_text


def render_catalog(path, cols=8, cell=150):
    names = sorted(ICONS)
    rows = math.ceil(len(names) / cols)
    W, H = cols * cell + 40, rows * (cell + 30) + 110
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()
    draw_text(ctx, f"کاتالوگ آیکون‌ها — {len(names)} آیکون (mograph v4)", W / 2, 50, size=34)
    for i, n in enumerate(names):
        r, c = divmod(i, cols)
        x = W - 20 - cell / 2 - c * cell  # RTL grid
        y = 110 + r * (cell + 30) + cell / 2 - 10
        draw_path(ctx, rounded_rect_d(x - cell / 2 + 8, y - cell / 2 + 8, cell - 16, cell - 16, 22), 0, 0, 1,
                  anchor="none", stroke="#2a2a2e", width=1.4)
        draw_icon(ctx, n, x, y - 8, size=cell * 0.6)
        draw_text(ctx, n, x, y + cell / 2 + 4, size=17, color="#9a9aa0", weight="medium")
    surf.write_to_png(path)
    return path
