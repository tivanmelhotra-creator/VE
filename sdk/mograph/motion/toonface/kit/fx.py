"""
toonface.kit.fx — v33 graphic extras of exaggerated cartoon expressions (from the user's expression sheets).

Works for BOTH renderers (hand-authored kit and procedural): everything is placed from a small set of anchors in
the renderer's local space (eye centres/sizes, mouth corners, face unit U = face width, chin, head top).

    mouth_fx : wavy (worried squiggle) · grit (clenched teeth band) · grin (wide toothy crescent) · o (whistle) · cat (ω)
    eye_fx   : star · spiral · x · squeeze (> <) · heart        (drawn by proc.draw_eye_fx)
    overlay  : blush · sweat drops · tear streams · sparkle · cold tint
    lines    : shock (zig-zag bolts) · anger (vein cross + steam puffs) · joy (radiating ticks) · cold (shiver)
               · idea (light bulb)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import ink


@dataclass
class Anchors:
    eyes: dict            # side -> dict(c=np.array, w, h, outer)
    mouth_l: np.ndarray
    mouth_r: np.ndarray
    U: float              # face width in local units
    chin: np.ndarray
    top: np.ndarray       # top of the face (hairline)
    sw: float             # line weight in local units


# ================================================================================================ mouths
def draw_mouth_fx(ctx, kind: str, a: Anchors, open_: float, pal, phase: float = 0.0) -> bool:
    """returns True if it drew (and the regular mouth must be skipped)."""
    L, R = a.mouth_l, a.mouth_r
    c = 0.5 * (L + R)
    u = (R - L) / max(np.linalg.norm(R - L), 1e-6)
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n = -n
    hw = 0.5 * float(np.linalg.norm(R - L))
    sw = a.sw
    if kind == "wavy":
        s = np.linspace(-1, 1, 60)
        amp = 0.10 * hw * (1 + 0.6 * open_)
        pts = c[None] + np.outer(s * hw * 0.85, u) + np.outer(amp * np.sin(s * 3.2 * math.pi + phase * 0.0), n)
        ink.tapered(ctx, pts, pal.ink, 1.0 * sw, kind="both", tip=0.25)
        return True
    if kind == "grit":
        hh = hw * (0.30 + 0.35 * min(1.0, open_ + 0.3))
        w = hw * 1.05
        box = np.array([c - u * w - n * hh * 0.5, c + u * w - n * hh * 0.5, c + u * w + n * hh * 0.5,
                        c - u * w + n * hh * 0.5])
        poly = ink.catmull(np.vstack([box, box[:1]])[:-1], per_seg=6, closed=True)
        ink.fill(ctx, poly, (1.0, 0.995, 0.975))
        ctx.save()
        ink.path(ctx, poly, True)
        ctx.clip()
        ink.stroke(ctx, np.array([c - u * w * 1.1, c + u * w * 1.1]), pal.ink, 0.55 * sw)
        for k in np.linspace(-0.8, 0.8, 7):
            p = c + u * w * k
            ink.stroke(ctx, np.array([p - n * hh, p + n * hh]), pal.ink, 0.45 * sw)
        ctx.restore()
        ink.stroke(ctx, poly, pal.ink, 0.9 * sw, closed=True)
        return True
    if kind == "grin":
        w = hw * 1.25
        s = np.linspace(-1, 1, 40)
        top = c[None] + np.outer(s * w, u) - np.outer(0.10 * hw * (1 - s ** 2) + 0.05 * hw, n)
        bot = c[None] + np.outer(s * w * 0.98, u) + np.outer(hw * (0.45 + 0.25 * open_) * (1 - s ** 2) ** 0.9, n)
        poly = np.vstack([top, bot[::-1]])
        ink.fill(ctx, poly, (1.0, 0.995, 0.975))
        ctx.save()
        ink.path(ctx, poly, True)
        ctx.clip()
        mid = c[None] + np.outer(s * w, u) + np.outer(hw * 0.16 * (1 - s ** 2), n)
        ink.stroke(ctx, mid, pal.ink, 0.55 * sw)
        for k in np.linspace(-0.75, 0.75, 6):
            p = c + u * w * k
            ink.stroke(ctx, np.array([p - n * hw, p + n * hw]), pal.ink, 0.40 * sw)
        ctx.restore()
        ink.stroke(ctx, poly, pal.ink, 0.95 * sw, closed=True)
        return True
    if kind == "o":
        r = 0.26 * hw * (1 + 0.5 * open_)
        ell = ink.ellipse_pts(c[0] + 0.1 * hw * u[0], c[1] + 0.05 * hw, r * 0.85, r)
        ink.fill(ctx, ell, (0.42, 0.075, 0.115))
        ink.stroke(ctx, ell, pal.ink, 0.85 * sw, closed=True)
        return True
    if kind == "cat":
        s = np.linspace(0, 1, 40)
        for sgn in (-1, 1):
            q = c[None] + np.outer(sgn * s * hw * 0.55, u) + np.outer(0.18 * hw * np.sin(math.pi * s), n)
            ink.tapered(ctx, q, pal.ink, 0.9 * sw, kind="both", tip=0.3)
        return True
    return False


# ================================================================================================ overlays
def draw_overlay(ctx, fx: dict, a: Anchors, pal, phase: float = 0.0):
    """blush / sweat / tears / sparkle: on the face (clipped to the face by the caller)."""
    U, sw = a.U, a.sw
    b = float(fx.get("blush", 0.0))
    if b > 0.02:
        for side in ("L", "R"):
            e = a.eyes[side]
            cx = e["c"][0] + e["outer"] * 0.06 * U
            cy = e["c"][1] + 0.62 * e["h"] + 0.06 * U
            ell = ink.ellipse_pts(cx, cy, 0.12 * U, 0.055 * U)
            ink.fill(ctx, ell, pal.blush, alpha=0.55 * b)
            for k in (-1, 0, 1):                                          # hatch ticks
                p = np.array([cx + k * 0.05 * U - 0.015 * U, cy + 0.02 * U])
                ink.tapered(ctx, np.array([p, p + [0.03 * U, -0.035 * U]]), pal.blush_line, 0.35 * sw,
                            alpha=0.8 * b)
    t = float(fx.get("tears", 0.0))
    if t > 0.02:
        for side in ("L", "R"):
            e = a.eyes[side]
            x0 = e["c"][0] + e["outer"] * 0.30 * e["w"]
            y0 = e["c"][1] + 0.35 * e["h"]
            ln = (0.30 + 0.25 * t) * U
            wob = 0.02 * U * math.sin(phase * 2.0)
            s = np.linspace(0, 1, 24)
            mid = np.c_[x0 + e["outer"] * (0.03 * U * s + wob * s), y0 + ln * s]
            wd = 0.035 * U * (0.6 + 0.8 * s) * t
            poly = np.vstack([mid + np.c_[-wd, 0 * wd], (mid + np.c_[wd, 0 * wd])[::-1]])
            ink.fill(ctx, poly, (0.62, 0.84, 0.98), alpha=0.9)
            ink.stroke(ctx, poly, (0.30, 0.55, 0.80), 0.4 * sw, closed=True, alpha=0.9)
    sp = float(fx.get("sparkle", 0.0))
    if sp > 0.02:
        for side in ("L", "R"):
            e = a.eyes[side]
            c = e["c"] + [-0.18 * e["w"], -0.18 * e["h"]]
            ink.circle(ctx, c[0], c[1], 0.045 * e["w"] * sp, (1, 1, 1))


def draw_outside(ctx, fx: dict, a: Anchors, pal, phase: float = 0.0):
    """sweat drops + emotion lines: around the head (NOT clipped to the face)."""
    U, sw = a.U, a.sw
    sweat = float(fx.get("sweat", 0.0))
    if sweat > 0.05:
        e = a.eyes["R"]
        base = e["c"] + [0.26 * U, -0.30 * U + 0.03 * U * ((phase * 0.6) % 1.0)]
        for k, sc in ((0, 1.0), (1, 0.6)):
            p = base + [0.07 * U * k, 0.12 * U * k]
            r = 0.045 * U * sc * sweat
            t = np.linspace(0, 2 * math.pi, 40)
            drop = np.c_[p[0] + r * np.sin(t) * (1 - 0.0 * t), p[1] + r * (np.cos(t) - 1.6 * np.maximum(0, -np.cos(t)) ** 3)]
            drop[:, 1] = p[1] - r * np.cos(t)
            drop[:, 1] -= np.where(np.cos(t) > 0, r * 1.2 * np.cos(t) ** 3, 0)
            ink.fill(ctx, drop, (0.72, 0.90, 1.0))
            ink.stroke(ctx, drop, pal.ink, 0.45 * sw, closed=True)
    kind = fx.get("lines")
    if not kind:
        return
    top, chin = a.top, a.chin
    cx = 0.5 * (a.eyes["L"]["c"][0] + a.eyes["R"]["c"][0])
    cy = 0.5 * (top[1] + chin[1])
    R = 0.62 * U
    jitter = 0.01 * U * math.sin(phase * 13.0)
    if kind == "shock":
        for ang in (-150, -115, -65, -30, 20, 160):
            r0 = math.radians(ang)
            d = np.array([math.cos(r0), math.sin(r0)])
            p0 = np.array([cx, cy]) + d * (R + jitter)
            nrm = np.array([-d[1], d[0]])
            pts = np.array([p0, p0 + d * 0.06 * U + nrm * 0.03 * U, p0 + d * 0.09 * U - nrm * 0.02 * U,
                            p0 + d * 0.15 * U + nrm * 0.02 * U])
            ink.tapered(ctx, pts, pal.ink, 0.7 * sw, kind="both", tip=0.2)
    elif kind == "joy":
        for ang in (-160, -130, -50, -20):
            r0 = math.radians(ang)
            d = np.array([math.cos(r0), math.sin(r0)])
            p0 = np.array([cx, cy]) + d * R
            ink.tapered(ctx, np.array([p0, p0 + d * 0.10 * U]), pal.ink, 0.7 * sw, kind="both")
    elif kind == "anger":
        p = np.array([cx + 0.33 * U, top[1] + 0.10 * U])                  # vein cross
        for q in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            arc = p + np.array(q) * 0.035 * U
            s = np.linspace(0, math.pi / 2, 10)
            pts = np.c_[arc[0] - q[0] * 0.03 * U * np.cos(s), arc[1] - q[1] * 0.03 * U * np.sin(s)]
            ink.tapered(ctx, pts, (0.85, 0.15, 0.15), 0.8 * sw)
        for sgn in (-1, 1):                                               # steam puffs
            c0 = np.array([cx + sgn * (R + 0.05 * U), top[1] + 0.05 * U + jitter])
            for k, rr in enumerate((0.05, 0.035, 0.025)):
                cc = c0 + [sgn * 0.06 * U * k, -0.05 * U * k]
                ink.circle(ctx, cc[0], cc[1], rr * U, (1, 1, 1), outline=pal.ink, lw=0.5 * sw)
    elif kind == "cold":
        for sgn in (-1, 1):
            x = cx + sgn * (R + 0.04 * U) + jitter
            for k in range(3):
                y0 = cy - 0.15 * U + k * 0.12 * U
                pts = np.array([[x, y0], [x + sgn * 0.02 * U, y0 + 0.03 * U], [x, y0 + 0.06 * U]])
                ink.tapered(ctx, pts, pal.ink, 0.5 * sw)
    elif kind == "idea":
        c = np.array([cx + 0.55 * U, top[1] - 0.05 * U])
        ink.circle(ctx, c[0], c[1], 0.08 * U, (1.0, 0.92, 0.45), outline=pal.ink, lw=0.6 * sw)
        ink.stroke(ctx, np.array([c + [-0.035 * U, 0.08 * U], c + [0.035 * U, 0.08 * U]]), pal.ink, 0.8 * sw)
        for ang in (-150, -90, -30, 0, 180):
            r0 = math.radians(ang)
            d = np.array([math.cos(r0), math.sin(r0)])
            ink.tapered(ctx, np.array([c + d * 0.11 * U, c + d * 0.16 * U]), pal.ink, 0.5 * sw)
