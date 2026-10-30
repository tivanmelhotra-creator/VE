"""
toonface.kit.draw25 — cairo renderer for the hand-authored 2.5D kit.

FaceState (all floats, 0 = neutral):
  lid_open   1 = authored open, 0 = shut              lid_tilt   + = inner corners lower (anger), - = sad
  lower      lower-lid raise (squint / smile)          look       (gx, gy) pupils, -1..1 of the free travel
  brow_up    + raises both brows (px of the plate)     brow_in    + inner ends down (anger) / - inner up (worry)
  furrow     glabella lines 0..1                       bags       under-eye fine line 0..1 (tired)
  yaw        small extra yaw in degrees → per-part parallax (depth weighted)
  mouth      dict of viseme params (see visemes.py) already mixed with mood
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cairo
import numpy as np

from .. import ink
from .shapes import CAVITY, CAVITY_DEEP, GLINT, INK, PUPIL, SCLERA, TEETH, TONGUE


@dataclass
class FaceState:
    lid_open: float = 0.80
    lid_tilt: float = 0.0
    lower: float = 0.0
    look: tuple = (0.0, 0.0)
    brow_up: float = 0.0
    brow_in: float = 0.0
    furrow: float = 0.0
    bags: float = 0.0
    yaw: float = 0.0
    mouth: dict = field(default_factory=dict)


def _smooth_closed(pts, per=10):
    return ink.catmull(np.asarray(pts, float), per_seg=per, closed=True)


def _top_bottom(P):
    """split a closed outline into top and bottom boundaries sampled on a common x grid."""
    i0, i1 = int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0]))
    n = len(P)
    a = [P[(i0 + k) % n] for k in range((i1 - i0) % n + 1)]
    b = [P[(i1 + k) % n] for k in range((i0 - i1) % n + 1)]
    a, b = np.array(a), np.array(b)
    top, bot = (a, b[::-1]) if a[:, 1].mean() < b[:, 1].mean() else (b[::-1], a)
    xs = np.linspace(P[i0, 0] + 0.5, P[i1, 0] - 0.5, 64)
    ty = np.interp(xs, *_mono(top))
    by = np.interp(xs, *_mono(bot))
    return xs, ty, by


def _mono(c):
    o = np.argsort(c[:, 0])
    return c[o, 0], c[o, 1]


def _parallax(depth, yaw_deg, R=230.0):
    return depth * R * math.sin(math.radians(yaw_deg)) * 0.55


# ------------------------------------------------------------------------------------------------------------ eyes
def draw_eye(ctx, spec, st: FaceState, kit, closed_shape: float = 0.0):
    sw = kit["stroke"]
    dx = _parallax(spec["depth"], st.yaw)
    P = _smooth_closed(spec["outline"]) + [dx, 0]
    xs, ty, by = _top_bottom(P)
    outer = spec["outer"]
    s = (xs - xs[0]) / (xs[-1] - xs[0])
    s_out = s if outer > 0 else 1 - s            # 0 inner corner … 1 outer corner
    h = by - ty
    op = float(np.clip(st.lid_open, 0.0, 1.15))
    u = (1.0 - min(op, 1.0)) + st.lid_tilt * (0.5 - s_out) * 0.9
    u = np.clip(u, 0.0, 1.0)
    if op > 1.0:
        u = u - (op - 1.0) * 0.6
    yl = ty + u * h
    yb = by - np.clip(st.lower, 0, 0.6) * h * (0.35 + 0.65 * np.sin(np.pi * s) ** 0.8)
    vis = yb - yl
    closed = float(np.max(vis)) < 0.06 * float(np.max(h))
    if not closed:
        poly = np.vstack([np.c_[xs, np.minimum(yl, yb)], np.c_[xs[::-1], yb[::-1]]])
        ink.fill(ctx, poly, SCLERA)
        # pupil (clipped to the visible sclera)
        ctx.save()
        ink.path(ctx, poly, True)
        ctx.clip()
        px, py = spec["pupil"]
        rx, ry = spec["pupil_r"]
        gx, gy = st.look
        trav_x = 0.5 * (xs[-1] - xs[0]) - rx * 1.05
        cx = px + dx + gx * trav_x * 0.75
        cy = py + gy * 0.22 * float(np.max(h))
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(rx, ry)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(*PUPIL)
        ctx.fill()
        ink.circle(ctx, cx - 0.38 * rx, cy - 0.42 * ry, 0.30 * rx, GLINT)
        ink.circle(ctx, cx + 0.30 * rx, cy + 0.38 * ry, 0.11 * rx, GLINT, alpha=0.85)
        ctx.restore()
        # thin outline of the lower sclera + short lower lid accent at the outer half
        ink.tapered(ctx, np.c_[xs, yb], INK, 0.62 * sw, kind="both", tip=0.25)
        k = s_out > 0.55
        seg = np.c_[xs[k], yb[k] + 0.25 * sw]
        if len(seg) > 3:
            ink.tapered(ctx, seg if outer > 0 else seg[::-1], INK, 1.15 * sw, kind="end" if outer > 0 else "start",
                        tip=0.2)
    # upper lid wedge: thick at the outer corner, flicked past it (wing)
    th_in, th_out = spec["lid_th"]
    th = th_in + (th_out - th_in) * s_out ** 1.4
    if closed:
        yl = np.minimum(yl, yb)
        th = th * 0.75
    ext = 0.10 * (xs[-1] - xs[0])
    if outer > 0:
        wx, wy = xs[-1] + ext, yl[-1] - 0.10 * float(np.max(h)) * (0.4 + 0.6 * min(op, 1))
    else:
        wx, wy = xs[0] - ext, yl[0] - 0.10 * float(np.max(h)) * (0.4 + 0.6 * min(op, 1))
    lower_edge = np.c_[xs, yl + 0.35 * sw]
    upper_edge = np.c_[xs, yl - th]
    if outer > 0:
        wedge = np.vstack([lower_edge, [[wx, wy]], upper_edge[::-1]])
    else:
        wedge = np.vstack([[[wx, wy]], lower_edge, upper_edge[::-1]])
    wedge = ink.catmull(wedge, per_seg=3, closed=True)
    ink.fill(ctx, wedge, INK)
    # lashes: two-prong fork out of the outer end of the wedge (girl)
    if kit.get("lashes"):
        L = xs[-1] - xs[0]
        ox, oy = (xs[-1], yl[-1] - 0.6 * th[-1]) if outer > 0 else (xs[0], yl[0] - 0.6 * th[0])
        lift = 0.55 + 0.45 * min(op, 1.0)
        for ang, ln, wd in ((-58, 0.30, 0.95), (-18, 0.26, 0.85)):
            a = math.radians(ang)
            d = np.array([math.cos(a) * outer, math.sin(a) * lift])
            p0 = np.array([ox, oy])
            p2 = p0 + d * L * ln
            p1 = p0 + d * L * ln * 0.5 + np.array([0.0, -0.06 * L])
            curve = np.array([(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2 for t in np.linspace(0, 1, 16)])
            ink.tapered(ctx, curve, INK, wd * th_out, kind="end", tip=0.05, power=0.9)
    if st.bags > 0.02 and not closed:
        k = (s_out > 0.25) & (s_out < 0.85)
        bag = np.c_[xs[k], by[k] + 0.16 * float(np.max(h))]
        ink.tapered(ctx, bag, INK, 0.45 * sw, alpha=min(1.0, st.bags))


# ----------------------------------------------------------------------------------------------------------- brows
def draw_brow(ctx, spec, st: FaceState, kit, side_depth):
    sw = kit["stroke"]
    c = np.asarray(spec["line"], float).copy()
    n = len(c)
    s = np.linspace(0, 1, n)                     # 0 outer → 1 inner
    c[:, 0] += _parallax(side_depth, st.yaw)
    c[:, 1] -= st.brow_up
    c[:, 1] += st.brow_in * 14.0 * s ** 1.6 - st.brow_in * 3.0 * (1 - s)
    # anger also pulls the inner end toward the nose a little
    vin = c[-1] - c[-2]
    vin /= np.linalg.norm(vin) + 1e-9
    c[-1] += vin * max(0.0, st.brow_in) * 4.0
    cc = ink.catmull(c, per_seg=10)
    m = len(cc)
    t = np.linspace(0, 1, m)
    w = spec["th"] * (0.18 + 0.82 * np.sin(np.pi * np.clip(t * 0.92 + 0.04, 0, 1)) ** 0.75)
    w *= 1.0 + 0.12 * (t - 0.5)                   # slightly heavier toward the inner end
    g = np.gradient(cc, axis=0)
    g /= np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-9)[:, None]
    nrm = np.c_[-g[:, 1], g[:, 0]]
    A = cc + nrm * (w[:, None] * 0.5)
    B = cc - nrm * (w[:, None] * 0.5)
    poly = np.vstack([A, B[::-1]])
    ink.fill(ctx, poly, kit["hair"])
    ink.stroke(ctx, poly, INK, 0.62 * sw, closed=True)


def draw_furrow(ctx, kit, st: FaceState):
    if st.furrow < 0.05:
        return
    sw = kit["stroke"]
    (x0, y0), (x1, y1) = kit["furrow"]
    dx = _parallax(0.95, st.yaw)
    for off, ln in ((0.0, 1.0), (9.0, 0.7)):
        p = np.array([(x0 + dx + off, y0 - st.brow_up * 0.5 + 6), (x0 + dx + off + 3, (y0 + y1) / 2),
                      (x1 + dx + off - 1, y1 * ln + y0 * (1 - ln))])
        ink.tapered(ctx, ink.catmull(p, 8), INK, 0.55 * sw, alpha=min(1.0, st.furrow))


# ------------------------------------------------------------------------------------------------------------ nose
def draw_nose(ctx, kit, st: FaceState):
    sw = kit["stroke"]
    p = np.asarray(kit["nose"]["line"], float)
    p[:, 0] += _parallax(1.25, st.yaw)
    ink.tapered(ctx, ink.catmull(p, 10), INK, 1.0 * sw, kind="both", tip=0.15, power=0.5)
    nx, ny = kit["nose"]["nostril"]
    nx += _parallax(1.2, st.yaw)
    ink.tapered(ctx, np.array([(nx - 3, ny - 2), (nx + 1, ny + 1), (nx + 5, ny + 1)]), INK, 0.45 * sw, alpha=0.8)


# ----------------------------------------------------------------------------------------------------------- mouth
def mouth_curves(kit, m: dict, yaw: float = 0.0, n: int = 41):
    mk = kit["mouth"]
    L = np.array(mk["left"], float)
    R = np.array(mk["right"], float)
    L[0] += _parallax(1.1, yaw)
    R[0] += _parallax(1.1, yaw)
    C = 0.5 * (L + R)
    u = (R - L) / np.linalg.norm(R - L)
    nv = np.array([-u[1], u[0]])                  # points DOWN-ish (screen)
    if nv[1] < 0:
        nv = -nv
    hw = 0.5 * np.linalg.norm(R - L) * mk["scale"]
    asym = mk["asym"]
    near_left = mk["near"] == "left"
    o = float(m.get("open", 0.0))
    rnd = float(m.get("round", 0.0))
    wid = float(m.get("width", 1.0)) * (1.0 - 0.42 * rnd)
    corner = float(m.get("corner", 0.0))
    skew = float(m.get("skew", 0.0))             # + lifts the near corner (smirk)
    wl = wid * (1.0 if near_left else asym)
    wr = wid * (asym if near_left else 1.0)
    cen = C + u * hw * float(m.get("shift", 0.0))
    s = np.linspace(0, 1, n)
    xloc = -wl + (wl + wr) * s                    # local x in hw units
    peak = 0.42 if near_left else 0.58
    def bump(pw):
        z = np.where(s < peak, s / peak, (1 - s) / (1 - peak))
        return np.sin(0.5 * np.pi * np.clip(z, 0, 1)) ** pw
    # neutral line keeps the authored smile curvature
    smile_curv = 0.10 + 0.20 * corner
    base = -smile_curv * (1 - (2 * s - 1) ** 2) * -1.0       # dips at the middle
    cy_l = -corner * 0.26 - skew * (0.18 if near_left else -0.10)
    cy_r = -corner * 0.20 + skew * (-0.10 if near_left else 0.18)
    ends = cy_l + (cy_r - cy_l) * s
    up = ends + base - float(m.get("upper", 0.0)) * bump(0.9) - 0.30 * rnd * o * bump(1.2)
    lo_depth = o * (1.05 + 0.25 * rnd) * bump(0.65 + 0.6 * rnd)
    lo = ends + base + lo_depth
    U = cen + np.outer(xloc * hw, u) + np.outer(up * hw, nv)
    Lw = cen + np.outer(xloc * hw, u) + np.outer(lo * hw, nv)
    return U, Lw, hw, u, nv


def draw_mouth(ctx, kit, st: FaceState):
    m = st.mouth or {}
    sw = kit["stroke"]
    U, Lw, hw, u, nv = mouth_curves(kit, m, st.yaw)
    o = float(m.get("open", 0.0))
    gap = np.linalg.norm(Lw - U, axis=1).max()
    if gap < 0.10 * hw or o < 0.04:
        w = 1.0 * sw * (1.25 if m.get("press", 0) > 0.5 else 1.0)
        ink.tapered(ctx, U, INK, w, kind="both", tip=0.25, power=0.55)
        # tiny corner accent on the near side (hand-drawn feel)
        c = U[0]
        ink.tapered(ctx, np.array([c + nv * -0.10 * hw - u * 0.05 * hw, c + nv * 0.06 * hw + u * 0.02 * hw]), INK,
                    0.55 * sw, alpha=0.9)
        return
    poly = np.vstack([U, Lw[::-1]])
    ink.fill(ctx, poly, CAVITY)
    ctx.save()
    ink.path(ctx, poly, True)
    ctx.clip()
    # deep throat shade
    mid = 0.5 * (U[len(U) // 2] + Lw[len(Lw) // 2])
    ink.circle(ctx, mid[0] + nv[0] * 0.15 * hw, mid[1] + nv[1] * 0.15 * hw, 0.38 * gap, CAVITY_DEEP, alpha=0.6)
    tu = float(m.get("teeth_up", 0.0))
    if tu > 0.02:
        band = min(0.42 * gap, tu * 0.36 * hw)
        T = np.vstack([U - nv * 0.2 * sw, (U + nv * band)[::-1]])
        ink.fill(ctx, T, TEETH)
    tl = float(m.get("teeth_lo", 0.0))
    if tl > 0.02:
        band = min(0.35 * gap, tl * 0.28 * hw)
        T = np.vstack([Lw + nv * 0.2 * sw, (Lw - nv * band)[::-1]])
        ink.fill(ctx, T, TEETH)
    tg = float(m.get("tongue", 0.0))
    if tg > 0.02:
        tup = float(m.get("tongue_up", 0.0))
        k = int(len(Lw) * (0.45 if kit["mouth"]["near"] == "left" else 0.55))
        base = Lw[k]
        ry = (0.22 + 0.25 * tg) * gap * (1 + 0.8 * tup)
        rx = 0.42 * hw * (0.7 + 0.3 * tg)
        cy = base - nv * ry * (0.55 + 0.5 * tup)
        ctx.save()
        ctx.translate(cy[0], cy[1])
        ctx.rotate(math.atan2(u[1], u[0]))
        ctx.scale(rx, ry)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(*TONGUE)
        ctx.fill()
        # tongue mid line
        ink.tapered(ctx, np.array([cy - u * rx * 0.05 - nv * ry * 0.1, cy + nv * ry * 0.55]), CAVITY, 0.35 * sw,
                    alpha=0.5)
    fv = float(m.get("fv", 0.0))
    ctx.restore()
    # outline: heavier upper lip, lighter lower lip, tapered into the corners
    ink.tapered(ctx, U, INK, 1.05 * sw, kind="both", tip=0.35, power=0.5)
    ink.tapered(ctx, Lw, INK, 0.92 * sw, kind="both", tip=0.35, power=0.5)
    if fv > 0.3:
        # lower lip tucked under the upper teeth: a soft lip line just under the teeth
        k = slice(len(U) // 5, 4 * len(U) // 5)
        ink.tapered(ctx, Lw[k] + nv * 0.12 * hw, INK, 0.6 * sw, alpha=0.8)
    if o > 0.45:
        k = slice(len(Lw) // 3, 2 * len(Lw) // 3)
        ink.tapered(ctx, Lw[k] + nv * (0.22 * hw + 0.6 * sw), INK, 0.5 * sw, alpha=0.75)


# ------------------------------------------------------------------------------------------------------------ face
def draw_face(ctx, kit, st: FaceState):
    e = kit["eyes"]
    b = kit["brows"]
    draw_brow(ctx, b["far"], st, kit, 0.8)
    draw_brow(ctx, b["near"], st, kit, 0.95)
    draw_furrow(ctx, kit, st)
    draw_eye(ctx, e["far"], st, kit)
    draw_eye(ctx, e["near"], st, kit)
    draw_nose(ctx, kit, st)
    draw_mouth(ctx, kit, st)
