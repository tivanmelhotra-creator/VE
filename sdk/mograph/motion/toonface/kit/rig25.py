"""
toonface.kit.rig25 — v31.2 multi-view hand-authored kit (front 0° ↔ 3/4 ±25°) with FIXED topology + renderer v2.

Topology (identical in every view key → continuous morph, CANDIDE-style fixed point count):
    eye outline 32 pts (canonical start = leftmost point, positive orientation) · brow centre line 7 pts (outer→inner)
    nose 7 pts · mouth = 3 anchors (left corner, lowest point, right corner) → quadratic neutral line
Negative yaw = mirrored 3/4 key (labels L/R stay screen-left / screen-right).

Face (all 0 = neutral):
    lid_open (1 authored, 0 shut, >1 wide)   lid_tilt (+ inner corners down)   lower (lower-lid raise)
    look (gx, gy) pupils only                pupil (scale, <1 surprise)
    brow_up   (+1 ≈ a third of an eye height up, arc grows; -1 down)
    brow_anger 0..1 (inner ends dive to the lid, V angle, outer tails flick up, thicker inner)
    brow_worry 0..1 (inner ends up, outer down)   furrow 0..1   bags 0..1 (only for real exhaustion)
    mouth dict (visemes params)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .. import ink
from .shapes import CAVITY, CAVITY_DEEP, GLINT, INK, PUPIL, SCLERA, TEETH, TONGUE

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N_EYE, N_BROW, N_NOSE = 32, 7, 7
YAW_KEY = 25.0


# ------------------------------------------------------------------------------------------------- topology helpers
def _arc_resample(p, n, closed=False):
    p = np.asarray(p, float)
    if closed:
        p = np.vstack([p, p[:1]])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    s = np.linspace(0, d[-1], n + 1 if closed else n)
    if closed:
        s = s[:-1]
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def canon_closed(p, n=N_EYE):
    p = np.asarray(p, float)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area < 0:
        p = p[::-1]
    p = np.roll(p, -int(np.argmin(p[:, 0])), axis=0)
    return _arc_resample(p, n, closed=True)


def _view_arrays(v, lid_inner_k=0.6):
    out = {"eyes": {}, "brows": {}}
    for s, e in v["eyes"].items():
        sm = ink.catmull(np.array(e["outline"], float), per_seg=6, closed=True)
        lt = np.array(e["lid_th"], float)
        out["eyes"][s] = {"outline": canon_closed(sm), "pupil": np.array(e["pupil"], float),
                          "pupil_r": np.array(e["pupil_r"], float), "lid_th": np.array([lt[0] * lid_inner_k, lt[1]]),
                          "outer": int(e["outer"])}
    for s, b in v["brows"].items():
        out["brows"][s] = {"line": _arc_resample(ink.catmull(np.array(b["line"], float), 8), N_BROW),
                           "th": float(b["th"])}
    out["nose"] = _arc_resample(ink.catmull(np.array(v["nose"], float), 8), N_NOSE)
    m = v["mouth"]
    out["mouth"] = {"left": np.array(m["left"], float), "right": np.array(m["right"], float),
                    "low": np.array(m["low"], float)}
    return out


def _mirror(va, W):
    def mx(p):
        q = np.array(p, float).copy()
        q[..., 0] = W - 1 - q[..., 0]
        return q
    out = {"eyes": {}, "brows": {}}
    for s, o in (("L", "R"), ("R", "L")):
        e = va["eyes"][s]
        out["eyes"][o] = dict(e, outline=canon_closed(mx(e["outline"])), pupil=mx(e["pupil"]), outer=-e["outer"])
        b = va["brows"][s]
        out["brows"][o] = dict(b, line=mx(b["line"]))
    out["nose"] = mx(va["nose"])
    m = va["mouth"]
    out["mouth"] = {"left": mx(m["right"]), "right": mx(m["left"]), "low": mx(m["low"])}
    return out


def _lerp(a, b, t):
    if isinstance(a, dict):
        return {k: _lerp(a[k], b[k], t) for k in a}
    if isinstance(a, int):
        return a
    return a + (b - a) * t


class Kit:
    """hand-authored multi-view kit of one character."""

    def __init__(self, name: str, W: int = 860):
        self.d = json.load(open(os.path.join(DATA, name + ".json")))
        self.name = name
        self.W = W
        self.front = _view_arrays(self.d["views"]["front"])
        self.q = _view_arrays(self.d["views"]["34"])
        self.qm = _mirror(self.q, W)
        self.sw = float(self.d["stroke"])
        sk = self.d.get("skin_rgb", [1.0, 0.765, 0.561])
        self.skin = tuple(sk)
        hb = self.d["hair_bgr"]
        self.hair = (hb[2] / 255.0, hb[1] / 255.0, hb[0] / 255.0)

    def view(self, yaw: float, pitch: float = 0.0):
        t = min(1.0, abs(yaw) / YAW_KEY)
        v = _lerp(self.front, self.q if yaw >= 0 else self.qm, t)
        if abs(pitch) > 0.01:
            v = _pitch(v, pitch)
        return v, t


def _pitch(v, pitch_deg):
    """head pitch on a blank face: features slide on the sphere (depth-weighted) and eyes foreshorten."""
    s = math.sin(math.radians(pitch_deg))
    k = 95.0 * s                                     # plate px at full ±90° (≈ 0.4 face radius)
    v = {**v, "eyes": {}, "brows": {}}
    for side in ("L", "R"):
        pass
    return v


# --------------------------------------------------------------------------------------------------------- state
@dataclass
class Face:
    lid_open: float = 0.85
    lid_tilt: float = 0.0
    lower: float = 0.0
    look: tuple = (0.0, 0.0)
    pupil: float = 1.0
    brow_up: float = 0.0
    brow_anger: float = 0.0
    brow_worry: float = 0.0
    furrow: float = 0.0
    bags: float = 0.0
    mouth: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------- utilities
def _mono(c):
    o = np.argsort(c[:, 0])
    return c[o, 0], c[o, 1]


def top_bottom(P, n=72):
    i0, i1 = int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0]))
    N = len(P)
    a = np.array([P[(i0 + k) % N] for k in range((i1 - i0) % N + 1)])
    b = np.array([P[(i1 + k) % N] for k in range((i0 - i1) % N + 1)])
    top, bot = (a, b) if a[:, 1].mean() < b[:, 1].mean() else (b, a)
    xs = np.linspace(P[i0, 0] + 0.4, P[i1, 0] - 0.4, n)
    return xs, np.interp(xs, *_mono(top)), np.interp(xs, *_mono(bot))


def chaikin(p, it=2):
    p = np.asarray(p, float)
    for _ in range(it):
        q = np.roll(p, -1, axis=0)
        p = np.vstack([np.c_[0.75 * p[:, 0] + 0.25 * q[:, 0], 0.75 * p[:, 1] + 0.25 * q[:, 1]],
                       np.c_[0.25 * p[:, 0] + 0.75 * q[:, 0], 0.25 * p[:, 1] + 0.75 * q[:, 1]]]
                      ).reshape(2, -1, 2).transpose(1, 0, 2).reshape(-1, 2)
    return p


def _quad(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


# ------------------------------------------------------------------------------------------------------------ eyes
def eye_geom(e, st: Face):
    xs, ty, by = top_bottom(e["outline"])
    outer = e["outer"]
    s = (xs - xs[0]) / (xs[-1] - xs[0])
    s_out = s if outer > 0 else 1 - s
    h = by - ty
    op = float(np.clip(st.lid_open, 0.0, 1.2))
    u = (1.0 - min(op, 1.0)) + st.lid_tilt * (0.55 - s_out) * 0.9
    u = np.clip(u, 0.0, 1.0)
    if op > 1.0:
        u = u - (op - 1.0) * 0.5
    yl = ty + u * h
    yb = by - np.clip(st.lower, 0, 0.6) * h * (0.35 + 0.65 * np.sin(np.pi * s) ** 0.8)
    return xs, ty, by, yl, yb, s, s_out, h


def draw_eye(ctx, e, st: Face, kit: Kit):
    sw = kit.sw
    xs, ty, by, yl, yb, s, s_out, h = eye_geom(e, st)
    outer = e["outer"]
    hmax = float(np.max(h))
    L = xs[-1] - xs[0]
    op = float(np.clip(st.lid_open, 0.0, 1.2))
    closed = float(np.max(yb - yl)) < 0.06 * hmax
    wing = kit.d.get("wing", True)
    th_in, th_out = e["lid_th"]
    th = th_in + (th_out - th_in) * s_out ** 1.4
    if not closed:
        top_vis = np.minimum(yl, yb)
        poly = np.vstack([np.c_[xs, top_vis], np.c_[xs[::-1], yb[::-1]]])
        ink.fill(ctx, poly, SCLERA)
        ctx.save()
        ink.path(ctx, poly, True)
        ctx.clip()
        px, py = e["pupil"]
        rx, ry = e["pupil_r"] * st.pupil
        gx, gy = st.look
        cx = px + gx * (0.5 * L - rx * 1.05) * 0.8
        cy = py + gy * 0.24 * hmax
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(rx, ry)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(*PUPIL)
        ctx.fill()
        ink.circle(ctx, cx - 0.36 * rx, cy - 0.42 * ry, 0.30 * min(rx, ry * 0.8), GLINT)
        ink.circle(ctx, cx + 0.30 * rx, cy + 0.40 * ry, 0.11 * rx, GLINT, alpha=0.85)
        ctx.restore()
        # lower lid: thin, a touch heavier toward the outer corner
        lw = (0.55 + 0.35 * s_out ** 1.5) * sw * kit.d.get("lower_w", 1.0)
        ink.tapered(ctx, np.c_[xs, yb], INK, 1.0, widths=lw * (0.35 + 0.65 * np.sin(np.pi * s) ** 0.35))
        # lower lashes (girl): two small strokes at the outer-lower corner, like the model sheet
        if kit.d.get("lower_lashes") and op > 0.3:
            for so, ln in ((0.80, 0.085), (0.92, 0.075)):
                k = int(np.argmin(np.abs(s_out - so)))
                p0 = np.array([xs[k], yb[k] + 0.2 * sw])
                d = np.array([outer * 0.55, 1.0])
                d /= np.linalg.norm(d)
                p2 = p0 + d * L * ln
                p1 = 0.5 * (p0 + p2) + np.array([outer * 0.02 * L, 0.0])
                ink.tapered(ctx, _quad(p0, p1, p2, 10), INK, 0.62 * sw, kind="end", tip=0.08, power=0.8)
        if st.bags > 0.35:
            k = (s_out > 0.25) & (s_out < 0.8)
            ink.tapered(ctx, np.c_[xs[k], by[k] + 0.17 * hmax], INK, 0.42 * sw, alpha=min(0.7, st.bags - 0.35))
    if closed and not wing:
        ink.fill(ctx, np.vstack([np.c_[xs, ty], np.c_[xs[::-1], by[::-1]]]), kit.skin)
    if closed:
        yl = ty + 0.70 * h + 0.06 * hmax * np.sin(np.pi * s)
        lwid = (0.55 + 0.9 * np.sin(np.pi * s) ** 0.6) * ((0.35 * th_out if wing else 0.0) + 0.75 * sw)
        ink.tapered(ctx, np.c_[xs, yl], INK, 1.0, widths=lwid)
        if wing:
            ox, oy = (xs[-1], yl[-1]) if outer > 0 else (xs[0], yl[0])
            ink.tapered(ctx, np.array([[ox, oy], [ox + outer * 0.07 * L, oy - 0.05 * hmax]]), INK, 0.55 * th_out,
                        kind="end", tip=0.1)
    elif wing:
        ext = 0.10 * L
        wy_off = 0.10 * hmax * (0.4 + 0.6 * min(op, 1))
        wx, wy = (xs[-1] + ext, yl[-1] - wy_off) if outer > 0 else (xs[0] - ext, yl[0] - wy_off)
        lower_edge = np.c_[xs, yl + 0.35 * sw]
        upper_edge = np.c_[xs, yl - th]
        if outer > 0:
            wedge = np.vstack([lower_edge, [[wx, wy]], upper_edge[::-1]])
        else:
            wedge = np.vstack([[[wx, wy]], lower_edge, upper_edge[::-1]])
        ink.fill(ctx, ink.catmull(wedge, per_seg=3, closed=True), INK)
    else:
        # round "boy" eye: one continuous outline (no corner gaps); a lowered lid = skin cap + its own lid line
        ring = np.vstack([np.c_[xs, ty], np.c_[xs[::-1], by[::-1]]])
        u_any = float(np.max(yl - ty))
        if u_any > 0.04 * hmax:
            cap = np.vstack([np.c_[xs, ty - 0.6 * sw], np.c_[xs[::-1], yl[::-1]]])
            ink.fill(ctx, cap, kit.skin)
        ink.stroke(ctx, ring, INK, 0.95 * sw, closed=True)
        if u_any > 0.04 * hmax:
            ink.tapered(ctx, np.c_[xs, yl], INK, 1.0, widths=sw * (0.55 + 0.75 * np.sin(np.pi * s) ** 0.4))
    if kit.d.get("lashes"):
        oy_off = 0.0 if closed else 0.6
        ox, oy = (xs[-1], yl[-1] - oy_off * th[-1]) if outer > 0 else (xs[0], yl[0] - oy_off * th[0])
        lift = (0.55 + 0.45 * min(op, 1.0)) if not closed else -0.35
        for ang, ln, wd in ((-58, 0.28, 0.9), (-18, 0.24, 0.8)):
            a = math.radians(ang)
            d = np.array([math.cos(a) * outer, math.sin(a) * lift])
            p0 = np.array([ox, oy])
            p2 = p0 + d * L * ln
            p1 = p0 + d * L * ln * 0.5 + np.array([0.0, -0.06 * L])
            ink.tapered(ctx, _quad(p0, p1, p2, 16), INK, wd * th_out, kind="end", tip=0.05, power=0.9)


# ----------------------------------------------------------------------------------------------------------- brows
def brow_line(b, e, st: Face):
    c = b["line"].copy()
    n = len(c)
    s = np.linspace(0, 1, n)                                   # 0 outer → 1 inner
    xs, ty, by, *_ = eye_geom(e, Face())
    eh = float(np.max(by - ty))
    L = float(np.hypot(*(c[-1] - c[0])))
    dirx = 1.0 if c[-1, 0] > c[0, 0] else -1.0                 # outer → inner direction on screen
    up = st.brow_up * 0.34 * eh
    c[:, 1] -= up * (0.85 + 0.35 * np.sin(np.pi * s))         # raising also lifts the arch
    a = float(np.clip(st.brow_anger, 0, 1.2))
    if a > 0:
        lid_top_inner = float(np.interp(c[-1, 0], xs, ty))
        gap = max(0.0, lid_top_inner - c[-1, 1])
        tin = c[-1] + np.array([dirx * 0.08 * L, 0.92 * gap])
        tout = c[0] + np.array([-dirx * 0.03 * L, -0.20 * eh])
        straight = tout[None] + (tin - tout)[None] * s[:, None]
        straight[:, 1] -= 0.06 * eh * np.sin(np.pi * np.clip(s * 1.15, 0, 1))   # slight convex bulge
        c = c * (1 - min(a, 1)) + straight * min(a, 1)
    w = float(np.clip(st.brow_worry, 0, 1))
    if w > 0:
        c[:, 1] -= w * 0.26 * eh * s ** 1.6
        c[:, 1] += w * 0.08 * eh * (1 - s) ** 1.5
        c[-1, 0] -= dirx * w * 0.04 * L
    return c, a


def draw_brow(ctx, b, e, st: Face, kit: Kit):
    sw = kit.sw
    c, a = brow_line(b, e, st)
    cc = ink.catmull(c, per_seg=10)
    t = np.linspace(0, 1, len(cc))
    w = b["th"] * (0.16 + 0.84 * np.sin(np.pi * np.clip(t * 0.9 + 0.06, 0, 1)) ** 0.7)
    w *= 1.0 + (0.10 + 0.30 * a) * (t - 0.45)
    g = np.gradient(cc, axis=0)
    g /= np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-9)[:, None]
    nrm = np.c_[-g[:, 1], g[:, 0]]
    poly = np.vstack([cc + nrm * (w[:, None] * 0.5), (cc - nrm * (w[:, None] * 0.5))[::-1]])
    ink.fill(ctx, poly, kit.hair)
    ink.stroke(ctx, poly, INK, 0.62 * sw, closed=True)
    return c


def draw_furrow(ctx, inner_l, inner_r, eh, amt, kit: Kit):
    if amt < 0.05:
        return
    sw = kit.sw
    mid = 0.5 * (inner_l + inner_r)
    gap = abs(inner_r[0] - inner_l[0])
    for side in (-1, 1):
        x0 = mid[0] + side * 0.12 * gap
        p = np.array([(x0 + side * 0.04 * gap, mid[1] - 0.30 * eh), (x0, mid[1] - 0.12 * eh),
                      (x0 - side * 0.02 * gap, mid[1] + 0.02 * eh)])
        ink.tapered(ctx, ink.catmull(p, 8), INK, 0.5 * sw, alpha=min(1.0, amt))


# ------------------------------------------------------------------------------------------------------------ nose
def draw_nose(ctx, nose, kit: Kit):
    ink.tapered(ctx, ink.catmull(nose, 10), INK, 0.95 * kit.sw, kind="both", tip=0.12, power=0.55)


# ----------------------------------------------------------------------------------------------------------- mouth
def mouth_geom(mv, m: dict, n: int = 49):
    Lc, Rc, Lo = mv["left"].copy(), mv["right"].copy(), mv["low"].copy()
    u = (Rc - Lc) / np.linalg.norm(Rc - Lc)
    nv = np.array([-u[1], u[0]])
    if nv[1] < 0:
        nv = -nv
    hw = 0.5 * np.linalg.norm(Rc - Lc)
    C = 0.5 * (Lc + Rc)
    tp = float(np.clip(np.dot(Lo - Lc, u) / (2 * hw), 0.2, 0.8))
    o = float(m.get("open", 0.0))
    rnd = float(m.get("round", 0.0))
    wid = float(m.get("width", 1.0)) * (1.0 - 0.45 * rnd) * (1.0 + 0.12 * o)
    corner = float(m.get("corner", 0.0))
    skew = float(m.get("skew", 0.0))
    # corners: width about the centre, smile lifts / frown drops them, skew lifts the screen-left (near) corner
    Lc2 = C + (Lc - C) * wid - nv * hw * (0.22 * corner + 0.16 * skew)
    Rc2 = C + (Rc - C) * wid - nv * hw * (0.20 * corner - 0.08 * skew)
    sag = float(np.dot(Lo - (Lc + (Rc - Lc) * tp), nv)) * (1.0 + 0.9 * max(0.0, corner))
    mid = Lc2 + (Rc2 - Lc2) * tp + nv * sag
    P = (mid - (1 - tp) ** 2 * Lc2 - tp ** 2 * Rc2) / (2 * (1 - tp) * tp)
    base = _quad(Lc2, P, Rc2, n)
    s = np.linspace(0, 1, n)
    z = np.where(s < tp, s / tp, (1 - s) / (1 - tp))
    def bump(pw):
        return np.sin(0.5 * np.pi * np.clip(z, 0, 1)) ** pw
    K = 1.25 * hw                                              # ≈ +40 % area over v31.0 (with wider anchors)
    lo = base + nv[None] * (o * K * (1.0 + 0.2 * rnd) * bump(0.45 + 0.35 * rnd))[:, None]
    up = base - nv[None] * ((float(m.get("upper", 0.0)) * 0.35 * hw * bump(0.8) + 0.25 * rnd * o * K * bump(0.9)))[:, None]
    return up, lo, base, hw, u, nv, tp


def draw_mouth(ctx, mv, m: dict, kit: Kit, near: float = 1.0):
    """near: 1 = 3/4 (dimple tick only on the near/screen-left corner), 0 = front (both ticks, lighter)."""
    sw = kit.sw
    up, lo, base, hw, u, nv, tp = mouth_geom(mv, m)
    o = float(m.get("open", 0.0))
    gap = float(np.max(np.linalg.norm(lo - up, axis=1)))
    if o < 0.05 or gap < 0.08 * hw:
        w = sw * (1.18 if m.get("press", 0) > 0.5 else 1.0)
        ink.tapered(ctx, base, INK, w, kind="both", tip=0.22, power=0.5)
        for idx, alpha in ((0, 1.0), (-1, (1 - near) * 0.75)):
            if alpha < 0.05:
                continue
            c = base[idx]
            tang = base[idx] - base[idx + (3 if idx == 0 else -4)]
            tang /= np.linalg.norm(tang) + 1e-9
            tick = np.array([c - nv * 0.10 * hw + tang * 0.02 * hw, c + tang * 0.05 * hw,
                             c + nv * 0.09 * hw + tang * 0.01 * hw])
            ink.tapered(ctx, ink.catmull(tick, 6), INK, 0.5 * sw, alpha=alpha)
        return
    poly = chaikin(np.vstack([up, lo[::-1]]), 2)
    ink.fill(ctx, poly, CAVITY)
    ctx.save()
    ink.path(ctx, poly, True)
    ctx.clip()
    k = int(len(lo) * tp)
    mid = 0.5 * (up[k] + lo[k])
    ink.circle(ctx, mid[0] + nv[0] * 0.2 * gap, mid[1] + nv[1] * 0.2 * gap, 0.42 * gap, CAVITY_DEEP, alpha=0.55)
    s = np.linspace(0, 1, len(up))
    zz = np.sin(np.pi * s) ** 0.5
    tu = float(m.get("teeth_up", 0.0))
    if tu > 0.02:
        band = min(0.46 * gap, tu * 0.42 * hw)
        ext = np.vstack([up[0] - u * hw * 0.2, up, up[-1] + u * hw * 0.2])
        low_edge = up + nv[None] * (band * (0.55 + 0.45 * zz))[:, None]
        low_edge = np.vstack([low_edge[0] - u * hw * 0.2, low_edge, low_edge[-1] + u * hw * 0.2])
        ink.fill(ctx, np.vstack([ext - nv * sw, low_edge[::-1]]), TEETH)
    tl = float(m.get("teeth_lo", 0.0))
    if tl > 0.02:
        band = min(0.36 * gap, tl * 0.30 * hw)
        hi_edge = lo - nv[None] * (band * (0.55 + 0.45 * zz))[:, None]
        ink.fill(ctx, np.vstack([lo + nv * sw, hi_edge[::-1]]), TEETH)
    tg = float(m.get("tongue", 0.0))
    if tg > 0.02:
        tup = float(m.get("tongue_up", 0.0))
        b0 = lo[k]
        ry = (0.24 + 0.24 * tg) * gap * (1 + 0.7 * tup)
        rx = 0.50 * hw * (0.7 + 0.3 * tg)
        c = b0 - nv * ry * (0.50 + 0.45 * tup)
        ctx.save()
        ctx.translate(c[0], c[1])
        ctx.rotate(math.atan2(u[1], u[0]))
        ctx.scale(rx, ry)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(*TONGUE)
        ctx.fill()
    ctx.restore()
    ink.stroke(ctx, poly, INK, 0.82 * sw, closed=True)
    ink.tapered(ctx, chaikin(up, 1)[2:-2] if len(up) > 8 else up, INK, 1.12 * sw, kind="both", tip=0.3, power=0.5)
    if float(m.get("fv", 0.0)) > 0.3:
        kk = slice(len(lo) // 5, 4 * len(lo) // 5)
        ink.tapered(ctx, lo[kk] + nv * 0.10 * hw, INK, 0.55 * sw, alpha=0.8)
    if o > 0.45:
        kk = slice(len(lo) // 3, 2 * len(lo) // 3)
        ink.tapered(ctx, lo[kk] + nv * (0.16 * hw + 0.7 * sw), INK, 0.45 * sw, alpha=0.7)


# ------------------------------------------------------------------------------------------------------------ face
def draw_face(ctx, kit: Kit, view, t_turn: float, st: Face, parts=("brows", "eyes", "nose", "mouth")):
    e, b = view["eyes"], view["brows"]
    if parts == ("mouth",):
        draw_mouth(ctx, view["mouth"], st.mouth or {}, kit, near=t_turn)
        return
    cl = draw_brow(ctx, b["L"], e["L"], st, kit)
    cr = draw_brow(ctx, b["R"], e["R"], st, kit)
    xs, ty, by, *_ = eye_geom(e["L"], Face())
    draw_furrow(ctx, cl[-1], cr[-1], float(np.max(by - ty)), st.furrow, kit)
    draw_eye(ctx, e["L"], st, kit)
    draw_eye(ctx, e["R"], st, kit)
    draw_nose(ctx, view["nose"], kit)
    if "mouth" in parts:
        draw_mouth(ctx, view["mouth"], st.mouth or {}, kit, near=t_turn)
