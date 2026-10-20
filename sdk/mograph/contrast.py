"""
mograph.contrast — scene-aware colour: WCAG contrast maths + background sampling.

Used by the HUD (subtitles, header, dots, handle) every frame: the HUD is drawn AFTER the world, so the
renderer samples the REAL pixels under each HUD zone (bg colour, plates, blocks, footage …) and every
HUD colour is passed through `adapt()` against that sample.  A light word can never land on a light
background again, whatever the style, palette or footage.

    ratio(fg, bg)                       WCAG 2.x contrast ratio (1 … 21)
    ensure(fg, bg, min_ratio, prefer)   fg if it passes, else the first preferred colour that passes,
                                        else fg pushed toward black/white until it passes
    dim(fg, bg, target)                 a "quiet" colour (upcoming karaoke words): visible but recessive
    sample(surface, rect)               median RGB of a canvas region (BGRA premultiplied cairo surface)

Targets: body text 4.5:1, large/bold text 3:1 (WCAG AA); upcoming/inactive words ≈ 1.9:1.
"""
from __future__ import annotations

import numpy as np

from .core import color as _color


def _lin(c):
    c = np.clip(np.asarray(c, dtype=float), 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(rgb):
    r, g, b = _lin(rgb[:3])
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def ratio(fg, bg):
    a, b = luminance(_c(fg)), luminance(_c(bg))
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def is_light(bg):
    return luminance(_c(bg)) > 0.38


def _c(x):
    if isinstance(x, str):
        return _color(x)
    x = tuple(float(v) for v in x)
    return x if len(x) == 4 else x + (1.0,)


def _mix(a, b, k):
    return tuple(a[i] + (b[i] - a[i]) * k for i in range(3)) + (a[3] if len(a) > 3 else 1.0,)


def ensure(fg, bg, min_ratio=4.5, prefer=()):
    """Return a colour with contrast >= min_ratio against bg (keeps fg's alpha)."""
    fg, bg = _c(fg), _c(bg)
    if ratio(fg, bg) >= min_ratio:
        return fg
    for p in prefer:
        if p is None:
            continue
        pc = _c(p)
        if ratio(pc, bg) >= min_ratio:
            return pc[:3] + (fg[3],)
    target = (0.067, 0.067, 0.067, fg[3]) if is_light(bg) else (1.0, 1.0, 1.0, fg[3])
    sat = max(fg[:3]) - min(fg[:3])
    if sat < 0.15:                 # neutral (white/grey) that fails → go straight to bold ink, never muddy grey
        return target
    for k in np.linspace(0.1, 1.0, 10):     # coloured (brand accent) → keep the hue, deepen/lighten it
        c = tuple(float(v) for v in _mix(fg, target, k))
        if ratio(c, bg) >= min_ratio:
            return c
    return target


def dim(fg, bg, target=1.9):
    """Recessive colour between bg and fg whose contrast is ≈ target (never invisible)."""
    fg, bg = _c(fg), _c(bg)
    if ratio(fg, bg) <= target:
        return fg
    lo, hi = 0.0, 1.0                    # k = how far from bg toward fg
    for _ in range(14):
        m = (lo + hi) / 2
        if ratio(_mix(bg, fg, m), bg) < target:
            lo = m
        else:
            hi = m
    return tuple(float(v) for v in _mix(bg, fg, hi)[:3]) + (fg[3],)


def sample(surface, rect, scale=1.0):
    """Median RGB (0..1) of a canvas rect (x, y, w, h in canvas px before `scale`)."""
    W, H = surface.get_width(), surface.get_height()
    x, y, w, h = [int(round(v * scale)) for v in rect]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return (0.0, 0.0, 0.0, 1.0)
    surface.flush()
    stride = surface.get_stride()
    buf = np.frombuffer(surface.get_data(), np.uint8).reshape(H, stride // 4, 4)
    reg = buf[y0:y1:3, x0:x1:3, :3].reshape(-1, 3).astype(np.float32) / 255.0   # every 3rd px: cheap
    b, g, r = np.median(reg, axis=0)
    return (float(r), float(g), float(b), 1.0)


def hud_env(surface, W, H, hud, scale=1.0):
    """Background samples under each HUD zone (computed once per frame, before the HUD is drawn)."""
    subs = hud.get("subs", {})
    sy = H * subs.get("y", 0.885)
    sz = subs.get("size", 50)
    dots = hud.get("dots", {})
    return {
        "subs": sample(surface, (W * 0.07, sy - sz * 1.3, W * 0.86, sz * 2.6), scale),
        "header": sample(surface, (W * 0.05, 30, W * 0.9, 90), scale),
        "dots": sample(surface, (W * 0.3, H * dots.get("y", 0.82) - 24, W * 0.4, 48), scale),
        "handle": sample(surface, (W * 0.04, H * 0.79 - 20, W * 0.35, 40), scale),
    }


def mix(a, b, k):
    """v9: hex mix of two colours (k = 0 → a, 1 → b)."""
    ca, cb = _c(a), _c(b)
    m = [ca[i] + (cb[i] - ca[i]) * k for i in range(3)]
    return "#%02x%02x%02x" % tuple(int(round(max(0, min(1, v)) * 255)) for v in m)
