"""Small reusable props / helpers for vector scenes: text -> vector fills, draw-on schedule,
camera push, flame flicker on a region of static items."""
import numpy as np, cairo
from .scene import Fill
from . import anim as A


def text_fill(txt, size, x, y, font='Dancing Script', tone=1.0):  # noqa
    """Text as one compound Fill (cairo glyph outlines -> flat cubic control points)."""
    S = cairo.ImageSurface(cairo.FORMAT_A8, 4, 4); c = cairo.Context(S)
    c.select_font_face(font); c.set_font_size(size); c.move_to(x, y); c.text_path(txt)
    paths, cur, start = [], [], None
    for kind, pts in c.copy_path():
        if kind == cairo.PATH_MOVE_TO:
            cur = [np.array(pts)]; start = np.array(pts)
        elif kind == cairo.PATH_LINE_TO:
            a, b = cur[-1], np.array(pts); cur += [a + (b - a) / 3, a + (b - a) * 2 / 3, b]
        elif kind == cairo.PATH_CURVE_TO:
            cur += [np.array(pts[0:2]), np.array(pts[2:4]), np.array(pts[4:6])]
        elif kind == cairo.PATH_CLOSE_PATH and len(cur) > 1:
            if np.linalg.norm(cur[-1] - start) > 1e-6:
                a = cur[-1]; cur += [a + (start - a) / 3, a + (start - a) * 2 / 3, start.copy()]
            paths.append(np.array(cur[:-1])); cur = []
    return Fill(paths, tone, group='title')


def draw_on_schedule(items, seed, t0=0.15, spread=2.2):
    """Start time per item from its distance to `seed` (the 'one line' grows from there)."""
    d = np.array([np.linalg.norm(it.points() - seed, axis=1).min() if len(it.points()) else 0 for it in items])
    return t0 + spread * d / max(d.max(), 1e-6)


def apply_draw_on(items, starts, t, length=0.55):
    for it, s in zip(items, starts):
        u = (t - s) / length
        if it.kind == 'stroke':
            it.reveal = float(np.clip(u, 0, 1))
        elif it.group == 'silhouette':
            it.opacity = float(u > 0)
        else:
            it.opacity = float(np.clip((u - 0.6) / 0.8, 0, 1))


def push_camera(W, H, OW, OH, T, zoom=0.06, center=(0.47, 0.5), dy=40, breath=0.004):
    def cam(t):
        z = (1 + zoom * A.EASE['smooth'](t / T)) * (1 + breath * np.sin(t * 2 * np.pi / 3.4))
        s = z * OW / W
        return np.array([[s, 0, OW / 2 - W * center[0] * s], [0, s, OH / 2 - H * center[1] * s + dy]])
    return cam


def flame(box, start=2.0):
    """fx for Rig.fx: items fully inside `box` stretch/sway like a candle flame."""
    x0, y0, x1, y1 = box

    def fx(t, items):
        if t < start:
            return items
        fl = 1 + 0.07 * np.sin(t * 23) + 0.05 * np.sin(t * 37 + 1)
        sw = 1.2 * np.sin(t * 11)
        out = list(items)
        for i, it in enumerate(items):
            P = it.points()
            if not (np.all(P[:, 0] > x0) and np.all(P[:, 0] < x1) and np.all(P[:, 1] > y0) and np.all(P[:, 1] < y1)):
                continue
            c = it.copy(); Q = P.copy()
            k = (y1 - Q[:, 1]) / (y1 - y0)
            Q[:, 1] = y1 - (y1 - Q[:, 1]) * fl
            Q[:, 0] += sw * k
            c.set_points(Q); out[i] = c
        return out
    return fx
