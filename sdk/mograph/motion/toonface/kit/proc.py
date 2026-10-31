"""
toonface.kit.proc — v33 PROCEDURAL face renderer driven by CharacterProfile (any character, every slider live).

Design space "D": origin = chin centre, x right, y DOWN, 1000 = face width (the widest row of the blank face).
All geometry is generated in D, then pushed through a 2.5D head projection (ellipsoid, yaw) and drawn by cairo
with the frame transform  T = M_track ∘ A_design→frame.

Parts and styles (option + slider system, DiceBear/VRoid-style):
  eyes   round · almond · droopy · cat · button · sleepy        + width/height/roundness/tilt/pupil/lashes/wing
  brows  soft · thick · thin · angular · bushy · round · line    + length/arch/slant/thickness/gain/colour/layer
  nose   bump · hook · button · dots · line · none
  mouth  default · wide · small · cat (rig25 mouth engine: visemes, teeth, tongue) + wavy / grin / ω extras
  eye FX modes (from the expression sheets): star · spiral · x · squeeze (>  <) · heart
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import ink
from .profile import CharacterProfile
from .rig25 import Face, chaikin, draw_mouth

D = 1000.0


# ================================================================================================ projection
@dataclass
class Proj:
    """2.5D head: a point (X, Y) on the face sheet lies on an ellipsoid; yaw rotates it about the vertical axis."""
    yaw_deg: float = 0.0
    depth: float = 0.42
    nose_out: float = 0.10

    def z(self, X):
        u = np.clip(2.0 * np.asarray(X, float) / D, -1.0, 1.0)
        return self.depth * D * np.sqrt(np.maximum(0.04, 1.0 - u * u))

    def __call__(self, P, extra_z: float = 0.0):
        P = np.asarray(P, float)
        if abs(self.yaw_deg) < 1e-3:
            return P.copy()
        a = math.radians(self.yaw_deg)
        out = P.copy()
        out[..., 0] = P[..., 0] * math.cos(a) + (self.z(P[..., 0]) + extra_z) * math.sin(a)
        return out

    def sx(self, X) -> float:
        """local horizontal foreshortening at X (for circles / widths)."""
        e = 2.0
        return float(abs(self(np.array([[X + e, 0.0]]))[0, 0] - self(np.array([[X - e, 0.0]]))[0, 0]) / (2 * e))


# ================================================================================================ helpers
def hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def mix(a, b, t):
    return tuple(float(x + (y - x) * t) for x, y in zip(a, b))


@dataclass
class Palette:
    ink: tuple
    skin: tuple
    hair: tuple
    sclera: tuple
    pupil: tuple
    brow: tuple
    blush: tuple
    sw: float                     # line weight in D units


def make_palette(prof: CharacterProfile, style) -> Palette:
    """'auto' colours come from the video's own drawing → new characters are harmonised automatically."""
    c = prof.color

    def bgr2rgb(v):
        return (float(v[2]) / 255.0, float(v[1]) / 255.0, float(v[0]) / 255.0)

    inkc = bgr2rgb(style.ink_bgr) if c.ink == "auto" else hex_rgb(c.ink)
    skin = bgr2rgb(style.skin_bgr) if c.skin == "auto" else hex_rgb(c.skin)
    hair = bgr2rgb(style.hair_bgr)
    if prof.brow.color == "auto":
        # brows in hair colour, but never invisible on the skin (bald / blond-on-pale): darken toward ink
        lum = lambda q: 0.3 * q[0] + 0.59 * q[1] + 0.11 * q[2]
        brow = hair if abs(lum(hair) - lum(skin)) > 0.10 else mix(hair, inkc, 0.55)
    elif prof.brow.color == "ink":
        brow = inkc
    else:
        brow = hex_rgb(prof.brow.color)
    blush = mix(skin, (0.96, 0.42, 0.45), 0.55) if c.blush == "auto" else hex_rgb(c.blush)
    sw = (style.line_px / max(1.0, style.width) * D) if c.line_weight == "auto" else float(c.line_weight) * D
    return Palette(ink=inkc, skin=skin, hair=hair, sclera=hex_rgb(c.sclera), pupil=hex_rgb(c.pupil), brow=brow,
                   blush=blush, sw=float(np.clip(sw, 6.0, 40.0)))


class _KitShim:
    """the bits of rig25.Kit that draw_mouth needs."""

    def __init__(self, pal: Palette):
        self.sw = pal.sw
        self.d = {}
        self.skin = pal.skin
        self.hair = pal.hair


# ================================================================================================ layout
EYE_STYLE_MODS = {
    "round":  dict(n=2.0, lower=1.0, tilt=0.0),
    "almond": dict(n=1.75, lower=0.70, tilt=0.04),
    "droopy": dict(n=2.0, lower=0.85, tilt=-0.16),
    "cat":    dict(n=1.6, lower=0.65, tilt=0.24),
    "button": dict(n=2.0, lower=1.0, tilt=0.0),
    "sleepy": dict(n=2.2, lower=0.75, tilt=-0.05),
}


def layout(prof: CharacterProfile) -> dict:
    """neutral geometry of every part in D (before expression / projection)."""
    e, b, n, m = prof.eye, prof.brow, prof.nose, prof.mouth
    out = {"eyes": {}, "brows": {}}
    for side, sg in (("L", -1.0), ("R", 1.0)):
        out["eyes"][side] = dict(c=np.array([sg * e.spacing * D, -e.y * D]), w=e.width * D, h=e.height * D,
                                 outer=sg)
        xi, xo = sg * b.gap * D, sg * (b.gap + b.length) * D
        yb = -b.y * D
        out["brows"][side] = dict(inner=np.array([xi, yb - 0.5 * b.slant * D]),
                                  outer=np.array([xo, yb + 0.5 * b.slant * D]), arch=b.arch * D,
                                  th=b.thickness * D, sg=sg)
    out["nose"] = dict(c=np.array([n.x * D, -n.y * D]), s=n.size * D)
    hw = 0.5 * m.width * D
    a = math.radians(m.tilt)
    u = np.array([math.cos(a), -math.sin(a)])                   # y down: + tilt lifts the right corner
    c = np.array([m.x * D, -m.y * D])
    out["mouth"] = dict(left=c - u * hw, right=c + u * hw, c=c, hw=hw, u=u, curve=m.curve * D)
    return out


# ================================================================================================ eyes
def _superellipse(cx, cy, a, b, n, tilt, lower, N=48):
    """upper / lower lid curves of a superellipse eye.  t: 0 → 1 across the eye (screen left → right)."""
    t = np.linspace(0.0, 1.0, N)
    x = cx + a * (2 * t - 1)
    u = np.abs(2 * t - 1)
    k = np.power(np.maximum(0.0, 1.0 - u ** n), 1.0 / n)
    top = cy - b * k
    bot = cy + b * k * lower
    return t, x, top, bot


def eye_curves(geo, st: Face, mods, prof: CharacterProfile):
    e = prof.eye
    cx, cy = geo["c"]
    a, b = 0.5 * geo["w"], 0.5 * geo["h"]
    n = e.roundness if e.style in ("round", "button") else mods["n"] * e.roundness / 2.0
    t, x, top, bot = _superellipse(cx, cy, a, b, n, 0.0, mods["lower"] * e.lower_flat)
    so = t if geo["outer"] > 0 else 1.0 - t                      # 0 inner → 1 outer
    tilt = (e.tilt + mods["tilt"]) * b * 2.0                       # + outer corner up
    shift = -tilt * (so - 0.5)
    top, bot = top + shift, bot + shift
    h = bot - top
    op = float(np.clip(st.lid_open, 0.0, 1.25))
    lid = top + (1.0 - min(op, 1.0)) * h + st.lid_tilt * (0.55 - so) * 0.9 * h * (1 if op < 1.0 else 0.5)
    if op > 1.0:                                                    # wide-open: lid lifts above the rest shape
        lid = lid - (op - 1.0) * 0.6 * h
    lid = np.clip(lid, top - 0.35 * h, bot)
    low = bot - np.clip(st.lower, 0, 0.6) * h * (0.35 + 0.65 * np.sin(np.pi * t) ** 0.8)
    return t, x, top, bot, lid, low, so, a, b


def draw_eye(ctx, geo, st: Face, prof: CharacterProfile, pal: Palette, proj: Proj, fx: str | None = None):
    mods = EYE_STYLE_MODS[prof.eye.style]
    t, x, top, bot, lid, low, so, a, b = eye_curves(geo, st, mods, prof)
    sw = pal.sw
    P = lambda xs, ys: proj(np.c_[xs, ys])
    if fx:
        draw_eye_fx(ctx, fx, geo, pal, proj)
        return
    closed = float(np.max(low - lid)) < 0.07 * 2 * b
    if not closed:
        up = np.minimum(lid, low)
        poly = np.vstack([P(x, up), P(x[::-1], low[::-1])])
        ink.fill(ctx, poly, pal.sclera)
        ctx.save()
        ink.path(ctx, poly, True)
        ctx.clip()
        gx, gy = st.look
        pr = prof.eye.pupil * geo["w"] * st.pupil * (1.45 if prof.eye.style == "button" else 1.0)
        pcx = geo["c"][0] + gx * (a - pr * 1.05) * 0.8
        pcy = geo["c"][1] + gy * 0.35 * b
        pc = proj(np.array([[pcx, pcy]]))[0]
        k = proj.sx(pcx)
        rx, ry = pr * k, pr * prof.eye.pupil_aspect
        if prof.eye.style == "button":
            ry = rx = pr * 1.1 * k
        ctx.save()
        ctx.translate(*pc)
        ctx.scale(max(rx, 1e-3), max(ry, 1e-3))
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(*pal.pupil)
        ctx.fill()
        ink.circle(ctx, pc[0] - 0.36 * rx, pc[1] - 0.40 * ry, 0.30 * min(rx, ry * 0.8), (1, 1, 1))
        ink.circle(ctx, pc[0] + 0.30 * rx, pc[1] + 0.38 * ry, 0.12 * rx, (1, 1, 1), alpha=0.85)
        if st.pupil > 1.08:                                          # pleading: extra sparkle
            ink.circle(ctx, pc[0] + 0.05 * rx, pc[1] - 0.05 * ry, 0.10 * rx, (1, 1, 1), alpha=0.9)
        ctx.restore()
        # lower lid (thin, tapered), upper lid (weight from profile)
        ink.tapered(ctx, P(x, low), pal.ink, 1.0, widths=0.55 * sw * (0.3 + 0.7 * np.sin(np.pi * t) ** 0.4))
        lw = sw * prof.eye.lid_weight * (0.65 + 0.75 * np.sin(np.pi * t) ** 0.45) * (0.8 + 0.4 * so)
        ink.tapered(ctx, P(x, up), pal.ink, 1.0, widths=lw)
        if prof.eye.wing:
            o = -1 if geo["outer"] > 0 else 0
            p0 = P(np.array([x[o]]), np.array([up[o]]))[0]
            p1 = p0 + np.array([geo["outer"] * 0.12 * geo["w"], -0.10 * geo["h"]])
            ink.tapered(ctx, np.array([p0, p1]), pal.ink, 0.9 * lw.max(), kind="end", tip=0.1)
    else:                                                            # closed lid: soft arc (happy = up-arc)
        ink.fill(ctx, np.vstack([P(x, top), P(x[::-1], bot[::-1])]), pal.skin)
        happy = st.lower > 0.3
        yl = top + 0.62 * (bot - top) + (-1 if happy else 1) * 0.10 * b * np.sin(np.pi * t)
        ink.tapered(ctx, P(x, yl), pal.ink, 1.0, widths=sw * prof.eye.lid_weight * (0.5 + 0.9 * np.sin(np.pi * t) ** 0.6))
        up = yl
    if prof.eye.lashes:
        o = -1 if geo["outer"] > 0 else 0
        base = P(np.array([x[o]]), np.array([up[o] if not closed else up[o]]))[0]
        for ang, ln in ((-55, 0.22), (-20, 0.19)):
            r = math.radians(ang)
            d = np.array([math.cos(r) * geo["outer"], math.sin(r) * (1 if not closed else -0.4)])
            ink.tapered(ctx, np.array([base, base + d * geo["w"] * ln]), pal.ink, 0.9 * sw, kind="end", tip=0.05)


def draw_eye_fx(ctx, fx, geo, pal: Palette, proj: Proj):
    """special eye modes from the expression sheets."""
    c = proj(np.array([geo["c"]]))[0]
    r = 0.42 * min(geo["w"], geo["h"] * 1.2) * proj.sx(geo["c"][0]) ** 0.5
    sw = pal.sw
    if fx == "star":
        pts = ink.star_pts(c[0], c[1], r * 1.15, r * 0.48)
        ink.fill(ctx, ink.catmull(pts, 3, closed=True), (1.0, 0.86, 0.25))
        ink.stroke(ctx, ink.catmull(pts, 3, closed=True), pal.ink, 0.8 * sw, closed=True)
    elif fx == "spiral":
        th = np.linspace(0, 5.5 * math.pi, 120)
        rr = r * th / th[-1]
        ink.tapered(ctx, np.c_[c[0] + rr * np.cos(th), c[1] + rr * np.sin(th)], pal.ink, 0.75 * sw, kind="both")
    elif fx == "x":
        for s in (-1, 1):
            ink.tapered(ctx, np.array([c + [-0.7 * r, -0.7 * r * s], c + [0.7 * r, 0.7 * r * s]]), pal.ink, 1.1 * sw)
    elif fx == "squeeze":                                            # > <  (roar / weep / joy squint)
        sg = -geo["outer"]
        p = np.array([c + [-sg * 0.75 * r, -0.55 * r], c + [sg * 0.55 * r, 0.0], c + [-sg * 0.75 * r, 0.55 * r]])
        ink.tapered(ctx, ink.catmull(p, 8), pal.ink, 1.25 * sw, kind="both", tip=0.25)
    elif fx == "heart":
        t = np.linspace(0, 2 * math.pi, 80)
        hx = 16 * np.sin(t) ** 3
        hy = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        pts = np.c_[c[0] + hx * r / 17, c[1] + hy * r / 17]
        ink.fill(ctx, pts, (0.92, 0.22, 0.32))
        ink.stroke(ctx, pts, pal.ink, 0.7 * sw, closed=True)
