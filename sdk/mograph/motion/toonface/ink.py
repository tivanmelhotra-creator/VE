"""
toonface.ink — brush primitives on cairo: pressure-tapered strokes, Catmull-Rom splines, fills, group alpha.
Every line of the face goes through here so the whole rig shares one "hand" (brush pen on a cel).
"""
from __future__ import annotations

import math

import cairo
import numpy as np


def resample(pts, n: int = 48) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] < 1e-9:
        return np.repeat(p[:1], n, axis=0)
    s = np.linspace(0, d[-1], n)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def catmull(pts, per_seg: int = 10, closed: bool = False) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 3:
        return p
    P = np.vstack([p[-1], p, p[0], p[1]]) if closed else np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(P[-2][None, :])
    return np.vstack(out)


def path(ctx, pts, closed: bool = False) -> None:
    p = np.asarray(pts, float).reshape(-1, 2)
    ctx.new_path()
    if len(p) == 0:
        return
    ctx.move_to(*p[0])
    for q in p[1:]:
        ctx.line_to(*q)
    if closed:
        ctx.close_path()


def fill(ctx, pts, rgb, alpha: float = 1.0) -> None:
    path(ctx, pts, True)
    ctx.set_source_rgba(*rgb, alpha)
    ctx.fill()


def stroke(ctx, pts, rgb, width, alpha: float = 1.0, closed: bool = False) -> None:
    path(ctx, pts, closed)
    ctx.set_source_rgba(*rgb, alpha)
    ctx.set_line_width(max(0.6, width))
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.stroke()


def taper_profile(n: int, kind: str = "both", tip: float = 0.18, power: float = 0.6) -> np.ndarray:
    s = np.linspace(0, 1, n)
    if kind == "both":
        f = np.sin(np.pi * s) ** power
    elif kind == "start":
        f = np.sin(0.5 * np.pi * s) ** power
    elif kind == "end":
        f = np.cos(0.5 * np.pi * s) ** power
    else:
        f = np.ones(n)
    return tip + (1 - tip) * f


def tapered(ctx, pts, rgb, width: float, kind: str = "both", tip: float = 0.18, power: float = 0.6,
            alpha: float = 1.0, widths=None, n: int | None = None) -> None:
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 2 or alpha <= 0.003:
        return
    span = float(np.hypot(*np.ptp(p, axis=0)))
    p = resample(p, n or max(10, min(180, int(span / 1.5) + 10)))
    if widths is None:
        w = taper_profile(len(p), kind, tip, power) * width
    else:
        w = np.interp(np.linspace(0, 1, len(p)), np.linspace(0, 1, len(widths)), np.asarray(widths, float))
    g = np.gradient(p, axis=0)
    g /= np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-9)[:, None]
    nrm = np.c_[-g[:, 1], g[:, 0]]
    L = p + nrm * (w[:, None] * 0.5)
    Rr = p - nrm * (w[:, None] * 0.5)
    ctx.set_source_rgba(*rgb, alpha)
    path(ctx, np.vstack([L, Rr[::-1]]), True)
    ctx.fill()
    for i in (0, -1):
        if w[i] > 0.9:
            ctx.new_path()
            ctx.arc(p[i, 0], p[i, 1], w[i] * 0.5, 0, 2 * math.pi)
            ctx.fill()


def circle(ctx, x, y, r, rgb=None, alpha=1.0, outline=None, lw=1.0) -> None:
    ctx.new_path()
    ctx.arc(x, y, max(0.3, r), 0, 2 * math.pi)
    if rgb is not None:
        ctx.set_source_rgba(*rgb, alpha)
        if outline is not None:
            ctx.fill_preserve()
        else:
            ctx.fill()
    if outline is not None:
        ctx.set_source_rgba(*outline, alpha)
        ctx.set_line_width(lw)
        ctx.stroke()


def ellipse_pts(cx, cy, rx, ry, n=64, rot=0.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = math.cos(rot), math.sin(rot)
    return np.c_[cx + c * x - s * y, cy + s * x + c * y]


def superellipse(rx, ry, n_exp=2.0, n=120) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / n_exp
    return np.c_[rx * np.sign(c) * np.abs(c) ** e, ry * np.sign(s) * np.abs(s) ** e]


def star_pts(cx, cy, r_out, r_in, n=5, rot=-math.pi / 2) -> np.ndarray:
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def push(ctx) -> None:
    ctx.push_group()


def pop(ctx, alpha: float) -> None:
    ctx.pop_group_to_source()
    ctx.paint_with_alpha(max(0.0, min(1.0, alpha)))
