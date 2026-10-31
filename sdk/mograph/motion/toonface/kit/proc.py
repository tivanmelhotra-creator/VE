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
    blush_line: tuple = (0.85, 0.35, 0.40)


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
                   blush=blush, sw=float(np.clip(sw, 6.0, 40.0)), blush_line=mix(blush, inkc, 0.35))


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


# ================================================================================================ brows
BROW_STYLE_MODS = {     # thickness profile power, end taper, outline, hair-strand ticks, arch multiplier
    "soft":    dict(p=0.65, tip=0.22, bushy=0, arch=1.0, block=False),
    "thick":   dict(p=0.35, tip=0.45, bushy=0, arch=0.8, block=False),
    "thin":    dict(p=0.90, tip=0.10, bushy=0, arch=1.1, block=False),
    "angular": dict(p=0.50, tip=0.15, bushy=0, arch=0.0, block=False),
    "bushy":   dict(p=0.40, tip=0.40, bushy=7, arch=0.9, block=False),
    "round":   dict(p=0.30, tip=0.80, bushy=0, arch=0.6, block=True),
    "line":    dict(p=1.00, tip=0.05, bushy=0, arch=1.2, block=False),
}


def brow_points(geo, st: Face, prof: CharacterProfile, eye_h: float, N: int = 9):
    """centre-line outer → inner, posed by the expression.  brow.gain multiplies EVERY brow motion (v32 brows were
    too timid); worry lifts the inner head, anger dives it and kinks the arch, brow_up lifts with a slight arch."""
    b = prof.brow
    mods = BROW_STYLE_MODS[b.style]
    s = np.linspace(0.0, 1.0, N)                                   # 0 outer → 1 inner
    o, i = geo["outer"], geo["inner"]
    chord = o[None] + (i - o)[None] * s[:, None]
    L = float(np.linalg.norm(i - o))
    arch = geo["arch"] * mods["arch"]
    if b.style == "angular":                                       # peak at 1/3 from the outer end
        camber = -arch * 1.6 * np.where(s < 0.35, s / 0.35, (1 - s) / 0.65) * 0.0 - geo["arch"] * np.where(
            s < 0.35, s / 0.35, (1 - s) / 0.65)
    else:
        camber = -arch * np.sin(np.pi * s)
    c = chord.copy()
    c[:, 1] += camber
    g = b.gain
    up = st.brow_up * 0.42 * eye_h * g
    c[:, 1] -= up * (0.8 + 0.3 * np.sin(np.pi * s))
    a = float(np.clip(st.brow_anger, 0, 1.2)) * min(g, 1.6)
    if a > 0:                                                      # inner head dives, outer tail lifts → V
        c[:, 1] += a * eye_h * (0.42 * s ** 1.3 - 0.16 * (1 - s) ** 1.5)
        c[:, 0] += geo["sg"] * -a * 0.05 * L * s                    # inner heads pull together
        c[:, 1] -= a * 0.10 * eye_h * np.sin(np.pi * s ** 0.8)       # muscular kink
    w = float(np.clip(st.brow_worry, 0, 1.1)) * min(g, 1.6)
    if w > 0:                                                      # inner head lifts, outer drops → /\
        c[:, 1] -= w * eye_h * 0.42 * s ** 1.4
        c[:, 1] += w * eye_h * 0.12 * (1 - s) ** 1.3
    return c, mods


def draw_brow(ctx, geo, st: Face, prof: CharacterProfile, pal: Palette, proj: Proj, eye_h: float):
    c, mods = brow_points(geo, st, prof, eye_h)
    cc = proj(ink.catmull(c, per_seg=8))
    t = np.linspace(0, 1, len(cc))
    th = geo["th"] * (1.0 + 0.25 * float(np.clip(st.brow_anger, 0, 1)))     # anger thickens the inner head
    wprof = (mods["tip"] + (1 - mods["tip"]) * np.sin(np.pi * np.clip(t * 0.9 + 0.05, 0, 1)) ** mods["p"])
    wprof *= 1.0 + 0.30 * (t - 0.4)                                  # fuller inner head
    w = th * wprof
    if mods["block"]:
        w = th * (0.75 + 0.25 * np.sin(np.pi * t))
    gx = np.gradient(cc, axis=0)
    gx /= np.maximum(np.hypot(gx[:, 0], gx[:, 1]), 1e-9)[:, None]
    nrm = np.c_[-gx[:, 1], gx[:, 0]]
    poly = np.vstack([cc + nrm * (w[:, None] * 0.5), (cc - nrm * (w[:, None] * 0.5))[::-1]])
    if mods["block"]:
        poly = chaikin(poly, 2)
    ink.fill(ctx, poly, pal.brow)
    if prof.brow.outline and prof.brow.style != "line":
        ink.stroke(ctx, poly, pal.ink, 0.55 * pal.sw, closed=True)
    if prof.brow.style == "line":
        ink.tapered(ctx, cc, pal.ink, 1.0, widths=0.9 * w + 0.4 * pal.sw)
    for k in range(mods["bushy"]):                                   # hair strands sticking out of a bushy brow
        j = int((k + 0.5) / mods["bushy"] * (len(cc) - 1))
        p0 = cc[j] - nrm[j] * w[j] * 0.45
        p1 = p0 - nrm[j] * w[j] * 0.55 + gx[j] * w[j] * 0.35 * (1 if k % 2 else -1)
        ink.tapered(ctx, np.array([p0, p1]), pal.ink, 0.45 * pal.sw, kind="end", tip=0.1)
    return c


def draw_furrow(ctx, inner_l, inner_r, eh, amt, pal: Palette, proj: Proj):
    if amt < 0.05:
        return
    mid = 0.5 * (inner_l + inner_r)
    gap = abs(inner_r[0] - inner_l[0])
    for side in (-1, 1):
        x0 = mid[0] + side * 0.14 * gap
        p = np.array([(x0 + side * 0.05 * gap, mid[1] - 0.30 * eh), (x0, mid[1] - 0.12 * eh),
                      (x0 - side * 0.02 * gap, mid[1] + 0.03 * eh)])
        ink.tapered(ctx, proj(ink.catmull(p, 8)), pal.ink, 0.45 * pal.sw, alpha=min(1.0, amt))


# ================================================================================================ nose
def draw_nose(ctx, geo, prof: CharacterProfile, pal: Palette, proj: Proj, dy: float = 0.0):
    c = geo["c"] + [0.0, dy]
    s = geo["s"]
    st = prof.nose.style
    P = lambda p: proj(np.asarray(p, float), extra_z=proj.nose_out * D)
    sw = pal.sw
    if st == "none":
        return
    if st == "bump":                                                  # small rounded hook under the bridge
        p = np.array([c + [-0.45 * s, -0.10 * s], c + [-0.05 * s, 0.22 * s], c + [0.45 * s, -0.08 * s]])
        ink.tapered(ctx, P(ink.catmull(p, 10)), pal.ink, 0.95 * sw, kind="both", tip=0.12, power=0.55)
    elif st == "hook":                                                # side hook (3/4-ish nose of the girl kit)
        p = np.array([c + [0.05 * s, -0.75 * s], c + [-0.35 * s, 0.05 * s], c + [-0.05 * s, 0.25 * s],
                      c + [0.25 * s, 0.10 * s]])
        ink.tapered(ctx, P(ink.catmull(p, 10)), pal.ink, 0.95 * sw, kind="both", tip=0.12, power=0.55)
    elif st == "button":
        q = P(np.array([c]))[0]
        ink.circle(ctx, q[0], q[1], 0.30 * s, mix(pal.skin, pal.blush, 0.5))
        ink.circle(ctx, q[0] - 0.08 * s, q[1] - 0.08 * s, 0.09 * s, (1, 1, 1), alpha=0.6)
    elif st == "dots":
        for sx in (-1, 1):
            q = P(np.array([c + [sx * 0.22 * s, 0.0]]))[0]
            ink.circle(ctx, q[0], q[1], 0.09 * s, pal.ink)
    elif st == "line":
        p = np.array([c + [0.0, -0.70 * s], c + [-0.10 * s, 0.0], c + [0.15 * s, 0.12 * s]])
        ink.tapered(ctx, P(ink.catmull(p, 8)), pal.ink, 0.75 * sw, kind="both", tip=0.1)


# ================================================================================================ mouth
def mouth_view(geo) -> dict:
    """rig25 mouth topology (left / right corners + the 'low' sag point) from the profile layout."""
    L, R, c, u = geo["left"], geo["right"], geo["c"], geo["u"]
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n = -n
    low = L + (R - L) * 0.42 + n * max(2.0, geo["curve"])
    return {"left": L.copy(), "right": R.copy(), "low": low}


class _InkSwap:
    """rig25.draw_mouth uses the module constant INK; swap it to the character's measured ink colour."""

    def __init__(self, rgb):
        self.rgb = rgb

    def __enter__(self):
        from . import rig25
        self.mod, self.old = rig25, rig25.INK
        rig25.INK = self.rgb

    def __exit__(self, *a):
        self.mod.INK = self.old


def draw_mouth_proc(ctx, geo, m: dict, prof: CharacterProfile, pal: Palette, proj: Proj, near: float = 0.0):
    mv = {k: proj(np.array([v]))[0] for k, v in mouth_view(geo).items()}
    mm = dict(m)
    st = prof.mouth.style
    if st == "wide":
        mm["width"] = mm.get("width", 1.0) * 1.12
    elif st == "small":
        mm["width"] = mm.get("width", 1.0) * 0.80
    with _InkSwap(pal.ink):
        draw_mouth(ctx, mv, mm, _KitShim(pal), near=near)


# ================================================================================================ face
def anchors_for(lay, pal: Palette, proj: Proj):
    from .fx import Anchors
    eyes = {}
    for s, g in lay["eyes"].items():
        eyes[s] = dict(c=proj(np.array([g["c"]]))[0], w=g["w"] * proj.sx(g["c"][0]), h=g["h"], outer=g["outer"])
    return Anchors(eyes=eyes, mouth_l=proj(np.array([lay["mouth"]["left"]]))[0],
                   mouth_r=proj(np.array([lay["mouth"]["right"]]))[0], U=D, chin=np.array([0.0, 0.0]),
                   top=np.array([0.0, -1.15 * D]), sw=pal.sw)


def draw_face_proc(ctx, prof: CharacterProfile, pal: Palette, ff, yaw: float = 0.0,
                   parts=("brows", "eyes", "nose", "mouth", "overlay"), phase: float = 0.0):
    """everything in design space D; the caller sets the cairo matrix D → frame."""
    lay = layout(prof)
    proj = Proj(yaw_deg=yaw, depth=prof.head.depth, nose_out=prof.head.nose_out)
    fx = getattr(ff, "fx", {}) or {}
    eh = prof.eye.height * D
    sides = {"L": ff.left, "R": ff.right}
    # expressiveness gain from the personality (scales the AU-driven brow/lid deltas around rest)
    if "brows" in parts:
        cl = draw_brow(ctx, lay["brows"]["L"], ff.left, prof, pal, proj, eh)
        cr = draw_brow(ctx, lay["brows"]["R"], ff.right, prof, pal, proj, eh)
        draw_furrow(ctx, cl[-1], cr[-1], eh, max(ff.left.furrow, ff.right.furrow), pal, proj)
    if "eyes" in parts:
        efx = fx.get("eye_fx")
        for s in ("L", "R"):
            draw_eye(ctx, lay["eyes"][s], sides[s], prof, pal, proj, fx=efx)
    if "nose" in parts:
        draw_nose(ctx, lay["nose"], prof, pal, proj, dy=ff.nose_dy * eh)
    a = anchors_for(lay, pal, proj)
    if "mouth" in parts:
        from .fx import draw_mouth_fx
        mfx = fx.get("mouth_fx")
        if not (mfx and draw_mouth_fx(ctx, mfx, a, float(ff.mouth.get("open", 0.0)), pal, phase)):
            draw_mouth_proc(ctx, lay["mouth"], ff.mouth, prof, pal, proj, near=min(1.0, abs(yaw) / 25.0))
        for side, amt in (("left", ff.cheek_line[0]), ("right", ff.cheek_line[1])):
            if amt < 0.05:
                continue
            c = proj(np.array([lay["mouth"][side]]))[0]
            sx = -1 if side == "left" else 1
            p = np.array([c + [sx * 0.04 * D, -0.10 * D], c + [sx * 0.07 * D, -0.04 * D], c + [sx * 0.06 * D, 0.01 * D]])
            ink.tapered(ctx, ink.catmull(p, 6), pal.ink, 0.40 * pal.sw, alpha=0.75 * amt)
    if "overlay" in parts:
        from .fx import draw_overlay
        draw_overlay(ctx, fx, a, pal, phase)
    if "outside" in parts:
        from .fx import draw_outside
        draw_outside(ctx, fx, a, pal, phase)
    return a
