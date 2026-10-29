"""
mograph.toon.fx — 2D cartoon effects anchored to a character's head / hands (screen space, hand-drawn look):
sweat drops, flying sweat, Zzz, shake lines, speed lines, sparkles, hearts, steam puffs, stink/heat waves,
idea bulb, question/exclamation marks, music notes, stomach-growl scribble, and a hand-lettered caption.

All functions: fx_x(ctx, x, y, s, t, ...) with (x, y) = anchor px, s = head radius px, t = seconds.
"""
from __future__ import annotations

import math

import numpy as np

INK = (0.1, 0.095, 0.09)
SWEAT = (0.62, 0.82, 0.96)


def _stroke(ctx, P, w, rgb=INK, a=1.0):
    ctx.set_source_rgba(*rgb, a)
    ctx.set_line_width(w)
    ctx.new_path()
    ctx.move_to(*P[0])
    for q in P[1:]:
        ctx.line_to(*q)
    ctx.stroke()


def _drop(ctx, x, y, r, ang=0.0, w=2.0, rgb=SWEAT):
    """teardrop pointing up (rotated by ang)."""
    t = np.linspace(0, 2 * math.pi, 40)
    px = r * np.sin(t) * (1 - np.cos(t)) * 0.55
    py = -r * (1 - np.cos(t)) * 0.0 + r * np.cos(t) * 1.15
    # simple drop: circle bottom + pointed top
    th = np.linspace(-math.pi * 0.15, math.pi * 1.15, 30)
    P = [(r * math.cos(a), r * math.sin(a)) for a in th] + [(0, -r * 2.1)]
    P = np.asarray(P)
    P[:, 1] *= -1
    P[:, 1] -= 0
    c, s = math.cos(ang), math.sin(ang)
    P = P @ np.array([[c, s], [-s, c]]) + [x, y]
    ctx.new_path()
    ctx.move_to(*P[0])
    for q in P[1:]:
        ctx.line_to(*q)
    ctx.close_path()
    ctx.set_source_rgba(*rgb, 1)
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK, 1)
    ctx.set_line_width(w)
    ctx.stroke()


def sweat(ctx, x, y, s, t, side=1, w=2.5):
    """one big drop sliding down the side of the head."""
    ph = (t * 0.6) % 1.0
    _drop(ctx, x + side * s * 0.85, y - s * 0.35 + ph * s * 0.35, s * 0.11, 0.0, w)


def fly_sweat(ctx, x, y, s, t, w=2.2, n=4):
    """drops flying off the head (scared / panicking), on twos."""
    tq = math.floor(t * 12) / 12
    for k in range(n):
        a = math.radians(-160 + k * (140 / max(1, n - 1)))
        ph = (tq * 1.8 + k * 0.27) % 1.0
        d = s * (1.05 + 0.45 * ph)
        _drop(ctx, x + math.cos(a) * d, y + math.sin(a) * d, s * 0.075 * (1 - 0.4 * ph), a + math.pi / 2, w)


def shake_lines(ctx, x, y, s, t, w=2.2, h=1.0):
    """( (  ) )  trembling marks either side of the body."""
    tq = math.floor(t * 12)
    j = (tq % 2) * s * 0.04
    for sd in (-1, 1):
        for k in range(2):
            cx = x + sd * (s * (1.25 + 0.22 * k) + j)
            th = np.linspace(-0.5, 0.5, 10)
            P = np.c_[cx + sd * np.cos(th * 2.2) * s * 0.1 - sd * s * 0.1, y + th * s * 0.9 * h]
            _stroke(ctx, P, w)


def speed_lines(ctx, x, y, s, t, direction=-1, w=2.0, n=4):
    tq = math.floor(t * 12)
    rng = np.random.default_rng(int(tq))
    for k in range(n):
        yy = y + (k - n / 2) * s * 0.35 + rng.uniform(-0.05, 0.05) * s
        x0 = x + direction * s * (1.2 + rng.uniform(0, 0.3))
        L = s * rng.uniform(0.6, 1.1)
        _stroke(ctx, np.array([[x0, yy], [x0 + direction * L, yy]]), w)


def zzz(ctx, x, y, s, t, w=2.6):
    import cairo
    ctx.select_font_face("Caveat", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    for k in range(3):
        ph = (t * 0.5 + k / 3) % 1.0
        ctx.set_font_size(s * (0.35 + 0.25 * ph))
        ctx.move_to(x + s * (0.9 + 0.6 * ph), y - s * (0.6 + 0.9 * ph))
        ctx.set_source_rgba(*INK, min(1, (1 - ph) * 3))
        ctx.show_text("Z")


def sparkles(ctx, x, y, s, t, n=5, w=2.2, rgb=INK):
    tq = math.floor(t * 8)
    rng = np.random.default_rng(7)
    for k in range(n):
        a = rng.uniform(0, 2 * math.pi)
        d = s * rng.uniform(1.2, 1.8)
        px, py = x + math.cos(a) * d, y + math.sin(a) * d
        r = s * 0.12 * (0.6 + 0.4 * ((tq + k) % 2))
        _stroke(ctx, np.array([[px - r, py], [px + r, py]]), w, rgb)
        _stroke(ctx, np.array([[px, py - r], [px, py + r]]), w, rgb)


def hearts(ctx, x, y, s, t, n=3, w=2.0):
    for k in range(n):
        ph = (t * 0.6 + k / n) % 1.0
        px = x + s * (0.9 * math.sin(k * 2.1 + t) * 0.6 + (k - 1) * 0.5)
        py = y - s * (1.2 + 1.2 * ph)
        r = s * 0.16 * (1 - 0.3 * ph)
        tt = np.linspace(0, 2 * math.pi, 40)
        hx = 16 * np.sin(tt) ** 3
        hy = -(13 * np.cos(tt) - 5 * np.cos(2 * tt) - 2 * np.cos(3 * tt) - np.cos(4 * tt))
        P = np.c_[px + hx / 17 * r, py + hy / 17 * r]
        ctx.new_path()
        ctx.move_to(*P[0])
        for q in P[1:]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.set_source_rgba(0.88, 0.25, 0.33, 1 - ph)
        ctx.fill()


def steam(ctx, x, y, s, t, w=2.2, side=1):
    """puff of breath / anger steam."""
    ph = (t * 1.2) % 1.0
    cx = x + side * s * (1.1 + 0.4 * ph)
    for k, (dx, dy, r) in enumerate(((0, 0, 0.16), (0.15, -0.08, 0.12), (0.12, 0.1, 0.1))):
        ctx.new_path()
        ctx.arc(cx + side * dx * s, y + dy * s, r * s * (0.7 + 0.5 * ph), 0, 2 * math.pi)
        ctx.set_source_rgba(1, 1, 1, 1 - ph)
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK, 1 - ph)
        ctx.set_line_width(w)
        ctx.stroke()


def waves(ctx, x, y, s, t, n=3, w=2.0, h=1.0):
    """wavy lines rising (heat / smell / steam from a pot)."""
    for k in range(n):
        ys = np.linspace(0, -s * 1.2 * h, 20)
        xs = x + (k - (n - 1) / 2) * s * 0.35 + np.sin(ys / (s * 0.18) + t * 5 + k) * s * 0.07
        _stroke(ctx, np.c_[xs, y + ys], w, INK, 0.8)


def mark(ctx, x, y, s, t, ch="!", rgb=INK, pop_t=0.0):
    import cairo
    k = min(1.0, max(0.0, (t - pop_t) / 0.12))
    sc = 1.0 + 0.35 * math.sin(math.pi * k) if k < 1 else 1.0
    ctx.select_font_face("Caveat", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(s * 0.9 * sc * k)
    e = ctx.text_extents(ch)
    ctx.move_to(x + s * 0.95 - e.width / 2, y - s * 0.9)
    ctx.set_source_rgba(*rgb, 1)
    ctx.show_text(ch)


def notes(ctx, x, y, s, t, w=2.2):
    import cairo
    ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    for k, c in enumerate("\u266a\u266b\u266a"):
        ph = (t * 0.45 + k / 3) % 1.0
        ctx.set_font_size(s * 0.45)
        ctx.move_to(x + s * (1.0 if k % 2 else -1.5) + math.sin(t * 2 + k) * s * 0.15, y - s * (0.4 + 1.0 * ph))
        ctx.set_source_rgba(*INK, min(1, (1 - ph) * 3))
        ctx.show_text(c)


def growl(ctx, x, y, s, t, w=2.0, text="grr"):
    """little zig-zag scribble near the belly + text (hungry)."""
    import cairo
    tq = math.floor(t * 12)
    j = (tq % 2) * s * 0.03
    for sd in (-1, 1):
        xs = x + sd * s * 0.9 + np.array([0, 0.1, 0, 0.1, 0]) * s * sd + j
        ys = y + np.linspace(-0.3, 0.3, 5) * s
        _stroke(ctx, np.c_[xs, ys], w)
    ctx.select_font_face("Caveat", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(s * 0.4)
    ctx.move_to(x - s * 1.4, y + s * 0.7)
    ctx.set_source_rgba(*INK, 1)
    ctx.show_text(text)


def bulb(ctx, x, y, s, t, w=2.4, pop_t=0.0):
    k = min(1.0, max(0.0, (t - pop_t) / 0.15))
    if k <= 0:
        return
    cx, cy, r = x + s * 1.0, y - s * 1.1, s * 0.25 * k
    ctx.new_path()
    ctx.arc(cx, cy, r, 0, 2 * math.pi)
    ctx.set_source_rgba(1.0, 0.92, 0.55, 1)
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK, 1)
    ctx.set_line_width(w)
    ctx.stroke()
    ctx.rectangle(cx - r * 0.45, cy + r * 0.9, r * 0.9, r * 0.5)
    ctx.stroke()
    for a in range(-2, 3):
        th = -math.pi / 2 + a * 0.55
        _stroke(ctx, np.array([[cx + math.cos(th) * r * 1.35, cy + math.sin(th) * r * 1.35],
                               [cx + math.cos(th) * r * 1.8, cy + math.sin(th) * r * 1.8]]), w)


def caption(ctx, text, x, y, size, t=1.0, t0=0.0, rgb=INK, font="Caveat", align="center", pop=True):
    """hand-lettered text with a little pop-in."""
    import cairo
    k = min(1.0, max(0.0, (t - t0) / 0.18)) if pop else 1.0
    if k <= 0:
        return
    sc = (0.6 + 0.4 * k) * (1.0 + 0.12 * math.sin(math.pi * k))
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size * sc)
    e = ctx.text_extents(text)
    dx = {"center": -e.width / 2 - e.x_bearing, "left": 0, "right": -e.width}[align]
    ctx.move_to(x + dx, y)
    ctx.set_source_rgba(*rgb, k)
    ctx.show_text(text)
