"""
toonface.kit.rig25 — v31.3 multi-view hand-authored kit (front 0° ↔ 3/4 ±25°) with FIXED topology + renderer v3.

Upgrades in v31.3:
  1. Organic S-Curved / Arched Angry & Surprised Eyebrows (`brow_line`):
     * Preserves the natural curved camber of the character's eyebrow instead of flattening it into a rigid straight
       line (`straight`), rotating and bending the arch organically into a fierce V-brow while tucking the far-eye
       outer tail inward in 3/4 view so it never pokes past the temple hairline.
  2. Hand-Crafted D-Bean / Kidney-Bean Open Mouths (`mouth_geom` & `draw_mouth`):
     * Decouples the upper-lip baseline (`base_up`) from the lower-lip sag when the mouth opens (`o > 0.05`) so the
       upper lip arches gracefully upward (`︵`) instead of sagging downward into the oral cavity, giving full,
       appealing visibility to the curved white upper teeth ribbon and rounded corners.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .. import ink
from .shapes import (CAVITY, CAVITY_DEEP, GLINT, INK, PUPIL, SCLERA, TEETH, TONGUE, TONGUE_HI, TONGUE_LINE,
                     TONGUE_SHADE)

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
    ns_m = mx(va["nose"])
    if abs(ns_m[-1, 0] - ns_m[0, 0]) > abs(ns_m[-1, 1] - ns_m[0, 1]) and ns_m[-1, 0] < ns_m[0, 0]:
        ns_m = ns_m[::-1].copy()
    out["nose"] = ns_m
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
        """continuous view.  |yaw| ≤ 25 interpolates front↔3/4; 25 < |yaw| ≤ 35 extrapolates the same (fixed-topology)
        deltas (max 1.4×).  Returns (view, t) with t ∈ [0, 1] the turn amount used for near-side styling."""
        tg = float(np.clip(abs(yaw) / YAW_KEY, 0.0, YAW_MAX / YAW_KEY))
        if tg > 1.0:
            tg = 1.0 + (tg - 1.0) / 0.4 * YAW_EXTRA
        v = _lerp(self.front, self.q if yaw >= 0 else self.qm, tg)
        if abs(pitch) > 0.01:
            v = _pitch(v, pitch)
        return v, min(1.0, tg)


YAW_MAX, PITCH_MAX = 35.0, 20.0
YAW_EXTRA = 0.15
PITCH_CY = 475.0
PITCH_SHIFT = 60.0
PITCH_SQUASH = 0.06


def pitch_map_y(y, pitch_deg, extra=0.0):
    """the SAME vertical map turn25.head_pitch applies to the plate (rigid lock) + optional sphere slide."""
    s = math.sin(math.radians(pitch_deg))
    k = 1.0 - PITCH_SQUASH * abs(s)
    return PITCH_CY + k * (np.asarray(y, float) - PITCH_CY + PITCH_SHIFT * s) + extra * s


def _pitch(v, pitch_deg):
    def P(p, extra):
        q = np.array(p, float).copy()
        q[..., 1] = pitch_map_y(q[..., 1], pitch_deg, extra)
        return q
    out = {"eyes": {}, "brows": {}}
    for sd, e in v["eyes"].items():
        out["eyes"][sd] = dict(e, outline=P(e["outline"], 10.0), pupil=P(e["pupil"], 10.0))
    for sd, b in v["brows"].items():
        out["brows"][sd] = dict(b, line=P(b["line"], 8.0))
    out["nose"] = P(v["nose"], 12.0)
    out["mouth"] = {k: P(m, 9.0) for k, m in v["mouth"].items()}
    return out


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
        lw = (0.55 + 0.35 * s_out ** 1.5) * sw * kit.d.get("lower_w", 1.0)
        ink.tapered(ctx, np.c_[xs, yb], INK, 1.0, widths=lw * (0.35 + 0.65 * np.sin(np.pi * s) ** 0.35))
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
        # Clean posed eye outline (follows top_vis and yb directly — zero ghost outer ring when half-lidded or squinting!)
        top_vis = np.minimum(yl, yb)
        posed_ring = np.vstack([np.c_[xs, top_vis], np.c_[xs[::-1], yb[::-1]]])
        ink.stroke(ctx, posed_ring, INK, 0.92 * sw, closed=True)
        ink.tapered(ctx, np.c_[xs, top_vis], INK, 1.0, widths=sw * (0.65 + 0.70 * np.sin(np.pi * s) ** 0.45))
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
    """
    v31.3 Organic Curved Brow Deformer:
      * Keeps the natural hand-drawn arch camber of the brow instead of flattening it into a straight stick (`straight`).
      * Scales vertical lift on foreshortened far eyes in 3/4 view (`far_k`) and tucks the outer tip inward so it
        never pokes past the temple hairline.
    """
    c = b["line"].copy()
    n = len(c)
    s = np.linspace(0, 1, n)                                   # 0 outer → 1 inner
    xs, ty, by, *_ = eye_geom(e, Face())
    eye_w = float(xs[-1] - xs[0])
    eh = float(np.max(by - ty))
    L = float(np.hypot(*(c[-1] - c[0])))
    dirx = 1.0 if c[-1, 0] > c[0, 0] else -1.0                 # outer → inner direction on screen

    # Detect foreshortened far eye in 3/4 view (`eye_w < 110 px`) to prevent over-lifting or temple poke-out
    far_k = float(np.clip((eye_w - 60.0) / 65.0, 0.68, 1.0))

    # Extract original organic arch camber relative to the outer→inner chord
    chord0 = c[0][None] + (c[-1] - c[0])[None] * s[:, None]
    camber0 = c - chord0

    up = st.brow_up * 0.30 * eh * far_k
    c[:, 1] -= up * (0.78 + 0.28 * np.sin(np.pi * s))

    a = float(np.clip(st.brow_anger, 0, 1.2))
    if a > 0:
        lid_top_inner = float(np.interp(c[-1, 0], xs, ty))
        gap = max(0.0, lid_top_inner - c[-1, 1])
        # Inner head dives toward the upper lid; outer tail lifts moderately while pulling inward on the far 3/4 side
        in_pull_x = 0.07 * L if far_k > 0.85 else 0.03 * L
        out_tuck_x = 0.02 * L if far_k > 0.85 else 0.14 * L
        tin = c[-1] + np.array([dirx * in_pull_x, 0.86 * gap])
        tout = c[0] + np.array([dirx * out_tuck_x, -0.15 * eh * far_k])
        new_chord = tout[None] + (tin - tout)[None] * s[:, None]
        # Preserve 55% of the natural hand-drawn camber + a muscular outer-third S-arch (zero straight-stick look!)
        s_arch = -0.11 * eh * far_k * (np.sin(np.pi * (s ** 0.82)) ** 1.15)
        posed = new_chord + 0.55 * camber0
        posed[:, 1] += s_arch
        k_a = min(a, 1.0)
        c = c * (1.0 - k_a) + posed * k_a

    w = float(np.clip(st.brow_worry, 0, 1))
    if w > 0:
        c[:, 1] -= w * 0.25 * eh * far_k * (s ** 1.5)
        c[:, 1] += w * 0.08 * eh * ((1 - s) ** 1.4)
        c[-1, 0] -= dirx * w * 0.04 * L
    return c, a


def draw_brow(ctx, b, e, st: Face, kit: Kit):
    sw = kit.sw
    c, a = brow_line(b, e, st)
    cc = ink.catmull(c, per_seg=10)
    t = np.linspace(0, 1, len(cc))
    # Plump, soft-tapered cream-blonde brow body (fuller inner head when angry, graceful outer taper)
    w = b["th"] * (0.20 + 0.84 * np.sin(np.pi * np.clip(t * 0.88 + 0.07, 0, 1)) ** 0.65)
    w *= 1.0 + (0.12 + 0.26 * a) * (t - 0.42)
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
        p = np.array([(x0 + side * 0.04 * gap, mid[1] - 0.28 * eh), (x0, mid[1] - 0.11 * eh),
                      (x0 - side * 0.02 * gap, mid[1] + 0.02 * eh)])
        ink.tapered(ctx, ink.catmull(p, 8), INK, 0.48 * sw, alpha=min(1.0, amt))


# ------------------------------------------------------------------------------------------------------------ nose
def draw_nose(ctx, nose, kit: Kit):
    ink.tapered(ctx, ink.catmull(nose, 10), INK, 0.95 * kit.sw, kind="both", tip=0.12, power=0.55)


# ----------------------------------------------------------------------------------------------------------- mouth
def mouth_geom(mv, m: dict, n: int = 49):
    """
    v31.3 Hand-Crafted D-Bean / Kidney-Bean Mouth Geometry:
      * Separates the upper-lip baseline (`base_up`) from the lower-lip baseline (`base_lo`) as soon as `open > 0`
        so the upper lip arches upward (`︵`) instead of sagging downward into the mouth cavity.
    """
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
    wid = float(m.get("width", 1.0)) * (1.0 - 0.45 * rnd) * (1.0 + 0.14 * o)
    corner = float(m.get("corner", 0.0))
    skew = float(m.get("skew", 0.0))

    # Lift corners gently when mouth opens so open mouths read as warm/expressive D-beans rather than droopy boats
    open_lift = 0.10 * min(1.0, o * 1.4)
    Lc2 = C + (Lc - C) * wid - nv * hw * (0.22 * corner + 0.16 * skew + open_lift)
    Rc2 = C + (Rc - C) * wid - nv * hw * (0.20 * corner - 0.08 * skew + open_lift * 0.7)

    sag_closed = float(np.dot(Lo - (Lc + (Rc - Lc) * tp), nv)) * (1.0 + 0.85 * max(0.0, corner))
    # Lower lip keeps the natural downward sag; upper lip flattens and arches upward (`sag_up <= 0`) when open!
    open_blend = min(1.0, o * 1.75)
    sag_up = sag_closed * (1.0 - 1.48 * open_blend) - (0.14 * hw * open_blend)

    mid_lo = Lc2 + (Rc2 - Lc2) * tp + nv * sag_closed
    P_lo = (mid_lo - (1 - tp) ** 2 * Lc2 - tp ** 2 * Rc2) / (2 * (1 - tp) * tp)
    base_lo = _quad(Lc2, P_lo, Rc2, n)

    mid_up = Lc2 + (Rc2 - Lc2) * tp + nv * sag_up
    P_up = (mid_up - (1 - tp) ** 2 * Lc2 - tp ** 2 * Rc2) / (2 * (1 - tp) * tp)
    base_up = _quad(Lc2, P_up, Rc2, n)

    s = np.linspace(0, 1, n)
    z = np.where(s < tp, s / tp, (1 - s) / (1 - tp))

    def bump(pw):
        return np.sin(0.5 * np.pi * np.clip(z, 0, 1)) ** pw

    K = 1.28 * hw
    lo = base_lo + nv[None] * (o * K * (0.96 + 0.22 * rnd) * bump(0.42 + 0.32 * rnd))[:, None]
    up = base_up - nv[None] * ((float(m.get("upper", 0.0)) * 0.38 * hw * bump(0.75) + 0.26 * rnd * o * K * bump(0.85)))[:, None]
    return up, lo, base_lo, hw, u, nv, tp


# ---------------------------------------------------------------------------------------------------------- tongue
def _tongue_frame(lo, k, hw, u, nv, m):
    """Tongue root sits on the floor of the mouth at the lowest lip point, optionally slid sideways (tongue_side)."""
    side = float(m.get("tongue_side", 0.0))
    return lo[k] + u * side * 0.32 * hw


def _to_world(b0, u, nv, xs, ys):
    return b0[None] + np.outer(xs, u) + np.outer(ys, nv)


def tongue_inside_poly(lo, k, gap, hw, u, nv, m):
    """
    v31.4 tongue body (inside the cavity, drawn UNDER the teeth):
      * wide bean resting on the floor of the mouth (fills the lower lip like a hand-drawn cel, not a floating egg);
      * soft median dip on the top edge, two lobes;
      * tongue_up raises a narrower tip toward the upper teeth (L / D / T / N);
      * the bottom closes far below the lip line → after clipping to the mouth it sits flush on the lower lip.
    Returns (polygon, top_edge, (b0, half_width, height, tip)).
    """
    tg = float(np.clip(m.get("tongue", 0.0), 0.0, 1.2))
    tup = float(np.clip(m.get("tongue_up", 0.0), 0.0, 1.0))
    tout = float(np.clip(m.get("tongue_out", 0.0), 0.0, 1.0))
    rnd = float(m.get("round", 0.0))
    tg = max(tg, 0.6 * tout)
    b0 = _tongue_frame(lo, k, hw, u, nv, m)
    half_w = hw * (0.62 + 0.20 * tg) * (1.0 - 0.32 * rnd) * (1.0 - 0.25 * tup)
    height = gap * (0.30 + 0.34 * tg) * (1.0 - 0.15 * rnd)
    tip = gap * 0.38 * tup
    x = np.linspace(-1.0, 1.0, 49)
    hump = np.clip(1.0 - x ** 2, 0.0, 1.0) ** 0.55
    dip = 1.0 - (0.06 * (1.0 - tup)) * np.exp(-(x / 0.22) ** 2)
    ys = -(height * hump * dip + tip * np.exp(-(x / 0.34) ** 2))
    top = _to_world(b0, u, nv, x * half_w, ys)
    deep = gap + 0.3 * hw
    bot = _to_world(b0, u, nv, np.array([1.3, 1.3, -1.3, -1.3]) * half_w, np.array([0.0, deep, deep, 0.0]))
    return np.vstack([top, bot]), top, (b0, half_w, height, tip)


def draw_tongue_inside(ctx, up, lo, k, gap, hw, u, nv, m, sw):
    tg = float(m.get("tongue", 0.0))
    tout = float(m.get("tongue_out", 0.0))
    if max(tg, tout) <= 0.02:
        return
    poly, top, (b0, half_w, height, tip) = tongue_inside_poly(lo, k, gap, hw, u, nv, m)
    tup = float(np.clip(m.get("tongue_up", 0.0), 0.0, 1.0))
    # 1) back of the tongue (throat side) = full silhouette in the shade colour
    ink.fill(ctx, poly, TONGUE_SHADE)
    # 2) lit front body: top edge pulled down & in → a thin cel-shade rim stays visible along the back
    rim = float(np.clip(0.12 * (height + tip), 0.7 * sw, 2.2 * sw))
    xx = np.linspace(-1, 1, len(top))
    lit_top = top + nv[None] * (rim * (0.7 + 0.6 * xx ** 2))[:, None] - u[None] * (xx * 0.05 * half_w)[:, None]
    ink.fill(ctx, np.vstack([lit_top, poly[len(top):]]), TONGUE)
    # 3) raised tip (L / DTN): darker underside wedge so the tip reads as curled up, not as a taller blob
    if tup > 0.15:
        a = float(np.clip((tup - 0.15) / 0.85, 0, 1))
        apex = b0 - nv * (height * 0.75 + tip * 0.70)
        under = np.array([b0 - u * 0.26 * half_w - nv * 0.30 * height, apex, b0 + u * 0.26 * half_w - nv * 0.30 * height,
                          b0 + nv * 0.1 * height])
        ink.fill(ctx, ink.catmull(np.vstack([under, under[:1]]), 8, closed=True), TONGUE_SHADE, alpha=0.6 * a)
    # 4) median groove — only when the tongue is big enough on screen to carry the detail
    vis_h = height + tip
    if vis_h > 7 * sw:
        g0 = b0 - nv * (vis_h * 0.80 - rim)
        g1 = b0 - nv * (vis_h * 0.38)
        ink.tapered(ctx, np.array([g0, 0.5 * (g0 + g1), g1]), TONGUE_LINE, 0.5 * sw, kind="both", tip=0.3,
                    alpha=0.45)
    # 5) small wet highlight on the near (screen-left) lobe
    hc = b0 - u * 0.45 * half_w - nv * (height * 0.60)
    ctx.save()
    ctx.translate(hc[0], hc[1])
    ctx.rotate(math.atan2(u[1], u[0]) - 0.30)
    ctx.scale(max(0.8 * sw, 0.13 * half_w), max(0.45 * sw, 0.07 * vis_h))
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()
    ctx.set_source_rgba(*TONGUE_HI, 0.65)
    ctx.fill()


def _lip_profile(lo, b0, u, nv):
    """Lower lip in tongue-local coordinates → callable x ↦ y (y along nv, + = down)."""
    rel = lo - b0[None]
    lx, ly = rel @ u, rel @ nv
    o = np.argsort(lx)
    return lambda xs: np.interp(xs, lx[o], ly[o])


def tongue_out_poly(lo, k, gap, hw, u, nv, m, sw):
    """
    Tongue stuck out over the lower lip (playful / raspberry). The root follows the lower-lip curve exactly (no notch,
    the lip ink is covered where the tongue crosses it), rounded U tip, optional lateral lean (tongue_side).
    """
    tout = float(np.clip(m.get("tongue_out", 0.0), 0.0, 1.0))
    side = float(m.get("tongue_side", 0.0))
    b0 = _tongue_frame(lo, k, hw, u, nv, m)
    lip = _lip_profile(lo, b0, u, nv)
    half_w = hw * (0.34 + 0.10 * tout)
    length = hw * (0.16 + 0.50 * tout)
    lean = side * 0.30 * length
    x = np.linspace(-1.0, 1.0, 49)
    prof = np.clip(1.0 - x ** 2, 0, 1) ** 0.5            # semicircle-ish U
    t = prof                                            # 0 at the root, 1 at the tip
    xs = x * half_w * (1.0 + 0.08 * t) + lean * t ** 2
    root_y = lip(x * half_w)
    ys = root_y + length * prof
    lobe = _to_world(b0, u, nv, xs, ys)
    # root edge: lip curve lifted a touch so the fill hides the lip ink under it
    xr = np.linspace(half_w, -half_w, 17)
    root = _to_world(b0, u, nv, xr, lip(xr) - 1.3 * sw)
    return np.vstack([lobe, root]), lobe, (b0, half_w, length, lean, lip)


def draw_tongue_out(ctx, lo, k, gap, hw, u, nv, m, sw):
    poly, lobe, (b0, half_w, length, lean, lip) = tongue_out_poly(lo, k, gap, hw, u, nv, m, sw)
    ink.fill(ctx, poly, TONGUE)
    ctx.save()
    ink.path(ctx, poly, True)
    ctx.clip()
    # contact shadow just under the lip (the tongue is pressed against it)
    xr = np.linspace(-1.4 * half_w, 1.4 * half_w, 21)
    sh = np.vstack([_to_world(b0, u, nv, xr, lip(xr) - 2 * sw),
                    _to_world(b0, u, nv, xr[::-1], lip(xr[::-1]) + 0.18 * length)])
    ink.fill(ctx, sh, TONGUE_SHADE, alpha=0.45)
    # form shade on the far (screen-right) side: the lobe offset toward the near side, inverted
    off = lobe - u[None] * (0.28 * half_w) - nv[None] * (0.10 * length)
    xr2 = np.linspace(-half_w, half_w, 17)
    shade = np.vstack([off, _to_world(b0, u, nv, xr2[::-1] * 1.5, lip(xr2[::-1]) - 3 * sw)])
    ink.path(ctx, np.vstack([lobe, _to_world(b0, u, nv, xr2[::-1] * 1.5, lip(xr2[::-1]) - 3 * sw)]), True)
    ink.path(ctx, shade, True)
    ctx.set_fill_rule(1)  # cairo.FILL_RULE_EVEN_ODD
    ctx.set_source_rgba(*TONGUE_SHADE, 0.32)
    ctx.fill()
    ctx.set_fill_rule(0)
    # wet highlight on the near lobe
    hc = b0 + u * (-0.42 * half_w + 0.4 * lean) + nv * (lip(-0.42 * half_w) + 0.45 * length)
    ink.circle(ctx, hc[0], hc[1], max(0.7 * sw, 0.10 * half_w), TONGUE_HI, alpha=0.8)
    ctx.restore()
    # median groove from the root toward the tip
    g = _to_world(b0, u, nv, np.array([0.0, 0.35 * lean, 0.65 * lean]),
                  lip(0.0) + np.array([0.12, 0.38, 0.62]) * length)
    ink.tapered(ctx, ink.catmull(g, 6), TONGUE_LINE, 0.55 * sw, kind="both", tip=0.25, alpha=0.65)
    # ink: sides + rounded tip only (root stays open — it disappears into the mouth)
    ink.tapered(ctx, lobe, INK, 0.82 * sw, kind="both", tip=0.12, power=0.5)


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

    # Smooth rounded corners on open mouths (3 Chaikin passes eliminate sharp triangular corners)
    poly = chaikin(np.vstack([up, lo[::-1]]), 3)
    rnd = float(m.get("round", 0.0))
    if rnd > 0.05:
        cen = poly.mean(axis=0)
        ang = np.arctan2(poly[:, 1] - cen[1], poly[:, 0] - cen[0])
        rel = poly - cen
        ax_u = np.abs(rel @ u).max()
        ax_n = np.abs(rel @ nv).max()
        oval = cen + np.outer(np.cos(ang) * ax_u * 0.92, u) + np.outer(np.sin(ang) * ax_n * 0.95, nv)
        k_ = min(1.0, rnd * 1.1)
        poly = poly * (1 - k_) + oval * k_

    ink.fill(ctx, poly, CAVITY)
    ctx.save()
    ink.path(ctx, poly, True)
    ctx.clip()
    k = int(len(lo) * tp)
    mid = 0.5 * (up[k] + lo[k])
    ink.circle(ctx, mid[0] + nv[0] * 0.22 * gap, mid[1] + nv[1] * 0.22 * gap, 0.44 * gap, CAVITY_DEEP, alpha=0.55)
    draw_tongue_inside(ctx, up, lo, k, gap, hw, u, nv, m, sw)
    s = np.linspace(0, 1, len(up))
    zz = np.sin(np.pi * s) ** 0.45
    tu = float(m.get("teeth_up", 0.0))
    if tu > 0.02:
        band = min(0.45 * gap, (0.14 + 0.86 * tu) * 0.46 * hw)
        ext = np.vstack([up[0] - u * hw * 0.2, up, up[-1] + u * hw * 0.2])
        low_edge = up + nv[None] * (band * (0.62 + 0.38 * zz))[:, None]
        low_edge = np.vstack([low_edge[0] - u * hw * 0.2, low_edge, low_edge[-1] + u * hw * 0.2])
        teeth_poly = np.vstack([ext - nv * sw, low_edge[::-1]])
        ink.fill(ctx, teeth_poly, TEETH)
        ink.stroke(ctx, low_edge[1:-1], INK, 0.46 * sw, closed=False)
    tl = float(m.get("teeth_lo", 0.0))
    if tl > 0.02:
        band = min(0.36 * gap, tl * 0.30 * hw)
        hi_edge = lo - nv[None] * (band * (0.55 + 0.45 * zz))[:, None]
        ink.fill(ctx, np.vstack([lo + nv * sw, hi_edge[::-1]]), TEETH)
        ink.stroke(ctx, hi_edge, INK, 0.42 * sw, closed=False)
    ctx.restore()
    ink.stroke(ctx, poly, INK, 0.84 * sw, closed=True)
    if rnd < 0.05:
        ink.tapered(ctx, chaikin(up, 1)[2:-2] if len(up) > 8 else up, INK, 1.14 * sw, kind="both", tip=0.28, power=0.5)
    if float(m.get("fv", 0.0)) > 0.3:
        kk = slice(len(lo) // 5, 4 * len(lo) // 5)
        ink.tapered(ctx, lo[kk] + nv * 0.10 * hw, INK, 0.55 * sw, alpha=0.8)
    if o > 0.45 and rnd < 0.05:
        kk = slice(len(lo) // 3, 2 * len(lo) // 3)
        ink.tapered(ctx, lo[kk] + nv * (0.16 * hw + 0.7 * sw), INK, 0.45 * sw, alpha=0.7)
    if float(m.get("tongue_out", 0.0)) > 0.02:
        draw_tongue_out(ctx, lo, k, gap, hw, u, nv, m, sw)


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
