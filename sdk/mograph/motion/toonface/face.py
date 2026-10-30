"""
toonface.face — draws a complete toon face from a FaceControls vector on a HeadFrame in a ToonStyle.
Pure vector construction — no reference pixels, no stamped drawings.

    draw_face(ctx, head, style, controls, t=0.0, clip_fn=None, occluder_fn=None)

  eye    superellipse socket on a decal (keeps a drawable shape at any yaw) · upper lid = dome that collapses into a
         "︶" line and tilts for anger / sadness · lower lid rises for cheek squints and meets the upper lid on a
         blink · pupil follows gaze + head yaw, clamped inside the visible aperture · glyph eyes (happy ∩, squeeze ><,
         spiral, star) swap THROUGH a blink, never by cross-dissolve
  brow   surface-wrapped crescent (hair-filled + fine ink, or solid ink brush) — raise / anger / worry / squeeze
  mouth  one continuous lip model: a smile line + upper / lower lip curves; jaw, width, corner lift, asymmetry,
         pucker roundness, teeth bands, tongue (+ tip-up), lip-bite, press, grit (clenched teeth) and wavy variants.
         A closed mouth is the zero-opening limit of the same shape, so every transition is continuous.
  nose   small hook with forward bulge — slides toward the cheek contour as the head turns
  fx     blush · puff cheeks · cold tint · sweat · tears fountains · stress marks · zzz · idea bulb · dizzy orbit ·
         music notes · shiver · sparkles · crumbs · anger vein
"""
from __future__ import annotations

import math

import cairo
import numpy as np

from . import ink
from .controls import FaceControls
from .geom import HeadFrame
from .style import ToonStyle, mix


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - e0) / max(1e-9, e1 - e0)))
    return t * t * (3 - 2 * t)


def _rot2(p: np.ndarray, th: float) -> np.ndarray:
    if not th:
        return p
    c, s = math.cos(th), math.sin(th)
    return np.c_[p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c]


# ================================================================================================== EYES
class EyeShape:
    """Visible aperture of one eye in LOCAL decal coordinates (head radii, a → screen-right, b ↓)."""

    def __init__(self, rx, ry, n_exp, openness, low, tilt, inner, n: int = 96):
        self.rx, self.ry, self.inner = rx, ry, inner
        a = np.linspace(-rx, rx, n)
        x = a / rx
        f = np.clip(1.0 - np.abs(x) ** n_exp, 0.0, 1.0) ** (1.0 / n_exp)
        top, bot = -ry * f, ry * f
        o = float(np.clip(openness, 0.0, 1.0))
        closed = ry * (0.20 - 0.26 * x * x)                      # "︶" line where the lids meet
        dome = -1.08 * ry + 0.30 * ry * x * x
        lid = closed + (dome - closed) * o
        lid = lid + tilt * 0.62 * ry * (x * inner + 0.15) * (0.35 + 0.65 * min(1.0, o + 0.2))
        lowc = closed + (1.08 * ry - closed) * min(1.0, o * 3.0)
        sq = float(np.clip(low, 0.0, 1.0))
        squint = (1.08 - 0.86 * sq) * ry - 0.30 * ry * sq * (1.0 - x * x)
        lowc = np.minimum(lowc, squint)
        self.a, self.lid, self.low, self.closed = a, lid, lowc, closed
        self.T = np.maximum(top, lid)
        self.B = np.minimum(bot, lowc)
        self.valid = (self.B - self.T) > ry * 0.012
        self.openness = o

    def _sl(self):
        idx = np.where(self.valid)[0]
        return None if len(idx) < 3 else slice(idx[0], idx[-1] + 1)

    def polygon(self):
        sl = self._sl()
        if sl is None:
            return None
        return np.vstack([np.c_[self.a[sl], self.T[sl]], np.c_[self.a[sl], self.B[sl]][::-1]])

    def upper(self):
        sl = self._sl()
        return None if sl is None else np.c_[self.a[sl], self.T[sl]]

    def lower(self):
        sl = self._sl()
        return None if sl is None else np.c_[self.a[sl], self.B[sl]]


def _lashes(ctx, P, eye: EyeShape, st: ToonStyle, out: int, lw: float, closed: bool, th: float) -> None:
    """Feminine lash band + outer wing flicks + lower ticks.  out = +1 if the outer corner is at +a.  P(local)→px."""
    w = st.lash_weight
    k = eye.rx / 0.22
    if closed:
        a = np.linspace(-eye.rx * 0.95, eye.rx * 0.95, 40)
        band = np.c_[a, eye.ry * (0.20 - 0.26 * (a / eye.rx) ** 2)]
        band = band if out > 0 else band[::-1]
        ink.tapered(ctx, P(_rot2(band, th)), st.ink, lw * 1.9 * w, kind="start", tip=0.35, power=0.8)
        end = band[-1]
        for j, (da, db) in enumerate(((0.05, 0.05), (0.03, 0.075), (0.0, 0.085))):
            p0 = end - np.array([out * 0.025 * j * k, 0.0])
            seg = np.array([p0, p0 + [out * da * k, db * k]])
            ink.tapered(ctx, P(_rot2(seg, th)), st.ink, lw * 1.1 * w, kind="end", tip=0.1)
        return
    up = eye.upper()
    if up is None:
        return
    band = (up if out > 0 else up[::-1]) + [0.0, -0.010 * k]
    prof = ink.taper_profile(len(band), "start", tip=0.40, power=0.7) * lw * 2.4 * w
    ink.tapered(ctx, P(_rot2(band, th)), st.ink, lw * 2.4 * w, widths=prof)
    end = band[-1]
    tang = band[-1] - band[-6]
    tang = tang / max(1e-9, float(np.hypot(*tang)))
    nrm = np.array([tang[1], -tang[0]])
    if nrm[1] > 0:
        nrm = -nrm
    for j, (ang, ln) in enumerate(((0.10, 0.085), (0.55, 0.075), (1.0, 0.06))):
        base = end - tang * (0.012 + 0.028 * j) * k
        d = tang * math.cos(ang) + nrm * math.sin(ang)
        seg = np.array([base, base + d * ln * 0.55 * k, base + (d * ln + nrm * 0.01) * k])
        ink.tapered(ctx, P(_rot2(seg, th)), st.ink, lw * 1.25 * w, kind="end", tip=0.06, power=0.9)
    lo = eye.lower()
    if lo is not None and len(lo) > 10 and eye.openness > 0.35:
        lb = lo if out > 0 else lo[::-1]
        for j in (0, 1):
            p0 = lb[int(len(lb) * (0.84 - 0.13 * j))]
            seg = np.array([p0, p0 + np.array([out * 0.35, 0.94]) * 0.035 * k])
            ink.tapered(ctx, P(_rot2(seg, th)), st.ink, lw * 0.8 * w, kind="end", tip=0.1)


def _normal_eye(ctx, head: HeadFrame, st: ToonStyle, c: FaceControls, anchor, side: int, openness: float,
                alpha: float) -> None:
    R = head.R
    lw = st.lw(R)
    inner = -side
    lid = c.lid_l if side < 0 else c.lid_r
    wide = max(0.0, lid - 1.0)
    rx = st.eye_rx * c.eye_scale * (1.0 + 0.10 * wide)
    ry = st.eye_ry * c.eye_scale * (1.0 + 0.25 * wide)
    low = c.low_l if side < 0 else c.low_r
    eye = EyeShape(rx, ry, st.eye_round, openness, low, c.lid_tilt, inner)
    th = -st.eye_tilt * side
    o, M = head.decal(anchor)
    P = lambda p: o[None, :] + np.asarray(p, float).reshape(-1, 2) @ M.T     # noqa: E731
    poly = eye.polygon()
    ink.push(ctx)
    up, lo = eye.upper(), eye.lower()
    if poly is not None:
        Pp = P(_rot2(poly, th))
        ink.fill(ctx, Pp, st.sclera)
        ctx.save()
        ink.path(ctx, Pp, True)
        ctx.clip()
        pr_x = st.pupil_rx * c.pupil * c.eye_scale
        pr_y = st.pupil_ry * c.pupil * c.eye_scale
        if c.pupil > 1.25:                                  # puppy pupils grow rounder
            kk = smoothstep(1.25, 1.8, c.pupil)
            m = 0.5 * (pr_x + pr_y) * 1.04
            pr_x, pr_y = pr_x + (m - pr_x) * kk, pr_y + (m - pr_y) * kk
        gx = c.gaze_x + 0.55 * math.sin(head.yaw)           # eyes lead the head turn
        gy = c.gaze_y + 0.5 * math.sin(head.pitch)
        ga = st.pupil_bias[0] * rx + gx * 0.55 * rx
        gb = st.pupil_bias[1] * ry + gy * 0.45 * ry
        ga = float(np.clip(ga, -(rx - pr_x * 1.1), rx - pr_x * 1.1))
        lim = max(0.0, ry - pr_y * 1.0)
        gb = float(np.clip(gb, -lim, lim))
        ink.fill(ctx, P(_rot2(ink.ellipse_pts(ga, gb, pr_x, pr_y, 48), th)), st.pupil)
        if c.sparkle > 0.02:
            s = c.sparkle
            ink.fill(ctx, P(_rot2(ink.ellipse_pts(ga + 0.33 * pr_x, gb - 0.36 * pr_y, 0.38 * pr_x, 0.34 * pr_y,
                                                  32), th)), (1, 1, 1), s)
            ink.fill(ctx, P(_rot2(ink.ellipse_pts(ga - 0.36 * pr_x, gb + 0.42 * pr_y, 0.16 * pr_x, 0.15 * pr_y,
                                                  24), th)), (1, 1, 1), s)
        if up is not None:                                   # soft cel shadow under the upper lid
            ink.tapered(ctx, P(_rot2(up + [0.0, 0.03 * ry / 0.22], th)), mix(st.sclera, st.skin, 0.6),
                        0.055 * R, kind="both", tip=0.3, alpha=0.5)
        ctx.restore()
        if lo is not None:
            ink.tapered(ctx, P(_rot2(lo, th)), st.ink, lw * 0.95, kind="both", tip=0.5, power=0.5)
        if up is not None:
            ink.tapered(ctx, P(_rot2(up, th)), st.ink, lw * 1.2, kind="both", tip=0.5, power=0.5)
    if openness < 0.2:
        kk = 1.0 - smoothstep(0.03, 0.2, openness)
        a = np.linspace(-rx * 0.94, rx * 0.94, 40)
        line = np.c_[a, np.interp(a, eye.a, np.maximum(eye.lid, eye.closed) if openness > 0 else eye.closed)]
        ink.tapered(ctx, P(_rot2(line, th)), st.ink, lw * 1.35, kind="both", tip=0.25, alpha=kk)
    if st.lashes:
        _lashes(ctx, P, eye, st, side, lw, openness < 0.06, th)
    ink.pop(ctx, alpha)


def _glyph_eye(ctx, head: HeadFrame, st: ToonStyle, c: FaceControls, anchor, side: int, alpha: float,
               t: float) -> None:
    R = head.R
    lw = st.lw(R)
    inner = -side
    rx, ry = st.eye_rx * c.eye_scale, st.eye_ry * c.eye_scale
    o, M = head.decal(anchor)
    P = lambda p: o[None, :] + np.asarray(p, float).reshape(-1, 2) @ M.T     # noqa: E731
    mode = c.eye_mode
    ink.push(ctx)
    if mode == "happy":                                      # ∩
        a = np.linspace(-rx * 0.8, rx * 0.8, 40)
        x = a / (rx * 0.8)
        b = ry * (0.22 - 0.62 * np.clip(1 - x * x, 0, 1) ** 0.85)
        ink.tapered(ctx, P(np.c_[a, b]), st.ink, lw * 1.55, kind="both", tip=0.3, power=0.6)
        if st.lashes:
            end = np.array([side * rx * 0.8, b[-1] if side > 0 else b[0]])
            for da, db in ((0.06, -0.035), (0.075, 0.005)):
                ink.tapered(ctx, P(np.array([end, end + [side * da, db]])), st.ink, lw * 1.1, kind="end", tip=0.1)
    elif mode == "squeeze":                                  # > <  (the point aims at the nose)
        out = side
        pts = np.array([[out * rx * 0.8, -ry * 0.55], [inner * rx * 0.5, -ry * 0.02], [out * rx * 0.8, ry * 0.5]])
        ink.tapered(ctx, P(pts[:2]), st.ink, lw * 1.6, kind="start", tip=0.35)
        ink.tapered(ctx, P(pts[1:]), st.ink, lw * 1.6, kind="end", tip=0.35)
        for k in (-1, 1):
            seg = np.array([[out * rx * 1.0, k * ry * 0.2], [out * rx * 1.2, k * ry * 0.36]])
            ink.tapered(ctx, P(seg), st.ink, lw * 0.7, kind="both", tip=0.2, alpha=0.8)
    elif mode == "spiral":
        sock = P(ink.superellipse(rx, ry, st.eye_round, 96))
        ink.fill(ctx, sock, st.sclera)
        ink.stroke(ctx, sock, st.ink, lw * 1.05, closed=True)
        th = np.linspace(0, 2.6 * 2 * math.pi, 170)
        r = np.linspace(0.03, 0.8, 170)
        spin = t * 4.0 * side
        ink.tapered(ctx, P(np.c_[r * np.cos(th + spin) * rx, r * np.sin(th + spin) * ry]), st.ink, lw * 1.0,
                    kind="start", tip=0.3)
    elif mode == "star":
        s = 1.0 + 0.06 * math.sin(t * 9.0)
        star = ink.star_pts(0.0, 0.0, max(rx, ry) * 1.02 * s, max(rx, ry) * 0.48 * s, 5, -math.pi / 2 + 0.08 * side)
        star = ink.catmull(star, 4, closed=True)
        Pp = P(star)
        ink.fill(ctx, Pp, (1.0, 1.0, 0.99))
        ink.stroke(ctx, Pp, st.ink, lw * 1.1, closed=True)
    ink.pop(ctx, alpha)


def draw_eyes(ctx, head: HeadFrame, st: ToonStyle, c: FaceControls, t: float = 0.0, dv: float = 0.0) -> None:
    g = c.eye_glyph if c.eye_mode != "normal" else 0.0
    for side in sorted((-1, 1), key=lambda s: head.normal_z((s * st.eye_u, st.eye_v))):   # far eye first
        anchor = (side * st.eye_u, st.eye_v + dv)
        vis = head.visibility(anchor, -0.10, 0.12)
        if vis <= 0.01:
            continue
        lid = c.lid_l if side < 0 else c.lid_r
        if c.eye_mode == "closed":
            _normal_eye(ctx, head, st, c, anchor, side, min(1.0, lid) * (1.0 - g), vis)
            continue
        through = 1.0 - smoothstep(0.0, 0.5, g)
        if through > 0.0:
            _normal_eye(ctx, head, st, c, anchor, side, min(1.0, lid) * through, vis)
        ga = smoothstep(0.5, 1.0, g)
        if ga > 0.0:
            _glyph_eye(ctx, head, st, c, anchor, side, vis * ga, t)


# ================================================================================================== BROWS
def draw_brows(ctx, head: HeadFrame, st: ToonStyle, c: FaceControls, dv: float = 0.0) -> None:
    R = head.R
    for side in (-1, 1):
        vis = head.visibility((side * st.eye_u, st.brow_v), -0.12, 0.10)
        if vis <= 0.01:
            continue
        raise_ = c.brow_l if side < 0 else c.brow_r
        hl = st.brow_len * st.eye_rx * 0.55
        s = np.linspace(-1, 1, 40)                          # −1 inner … +1 outer
        iw = ((1 - s) / 2) ** 1.25
        ow = (1 + s) / 2
        arch = st.brow_arch * (1.0 - 0.8 * c.brow_anger) * (1.0 - 0.3 * c.brow_worry)
        u = side * (st.eye_u + s * hl) - side * (0.06 * c.brow_squeeze + 0.03 * c.brow_anger) * (1 - s) / 2
        v = (st.brow_v + dv * 1.3 - 0.075 * raise_ - arch * (1 - s * s)
             + 0.12 * c.brow_anger * iw - 0.12 * c.brow_worry * iw + 0.04 * c.brow_worry * ow
             + 0.03 * c.brow_squeeze * (1 - s) / 2)
        th = st.brow_thick * (1.0 + 0.3 * c.brow_anger + 0.2 * c.brow_squeeze)
        prof = th * (0.28 + 0.72 * np.sin(np.pi * (s + 1) / 2) ** 0.6) * (1.0 + 0.2 * iw)
        poly = np.vstack([np.c_[u, v - prof * 0.5], np.c_[u, v + prof * 0.5][::-1]])
        Pp, _ = head.wrap(poly)
        ink.push(ctx)
        if st.brow_style == "hair":
            ink.fill(ctx, Pp, st.hair)
            ink.stroke(ctx, Pp, st.ink, st.lw(R, True) * 1.05, closed=True)
        else:
            mid, _ = head.wrap(np.c_[u, v])
            ink.tapered(ctx, mid, st.ink, th * R, kind="end" if side > 0 else "start", tip=0.25, power=0.5)
        ink.pop(ctx, vis)


# ================================================================================================== NOSE
def draw_nose(ctx, head: HeadFrame, st: ToonStyle, c: FaceControls) -> None:
    r = st.nose_size
    turn = smoothstep(0.05, 0.2, abs(head.yaw))
    d_front = st.nose_dir
    d_turn = 1 if head.yaw > 0 else -1          # in 3/4 the hook opens AWAY from the turn (cheek side)
    vis = head.visibility((0.0, st.nose_v), -0.35, 0.0)
    for d, w in ((d_front, 1.0 - turn), (d_turn, turn)):
        if w <= 0.01:
            continue
        base = np.array([[0.10, -1.0], [-0.42, -0.15], [-0.30, 0.50], [0.35, 0.62]]) * [d * r, r]
        pts = ink.catmull(base + [0.0, st.nose_v], 8)
        Pp, _ = head.wrap(pts, bulge=0.10)
        ink.tapered(ctx, Pp, st.ink, st.lw(head.R) * 0.95, kind="both", tip=0.3, power=0.6, alpha=w * vis)


# ================================================================================================== MOUTH
def mouth_curves(st: ToonStyle, c: FaceControls, n: int = 64, dv: float = 0.0):
    s = np.linspace(-1, 1, n)
    grit = c.mouth_glyph if c.mouth_mode == "grit" else 0.0
    hw = st.mouth_w * max(0.18, c.width) * (1.0 + 0.08 * grit)
    cx = c.shift_x + 0.015 * c.asym
    cy = st.mouth_v + dv
    lift_r = 0.11 * (c.smile + 0.8 * c.asym)
    lift_l = 0.11 * (c.smile - 0.4 * c.asym)
    lift = (lift_l + (lift_r - lift_l) * (s + 1) / 2) * (1.0 - 0.3 * c.press) * (1.0 - 0.5 * c.pucker)
    h = st.mouth_open * max(0.0, c.jaw) * (1.0 + 0.15 * c.pucker)
    h = max(h, 0.10 * grit)                                  # clenched teeth always show a full grid
    base = cy - lift * np.abs(s) ** 1.6 + 0.018 * c.smile * (1 - s * s) * (1 - c.pucker)
    if c.wobble > 0:
        base = base + 0.018 * c.wobble * np.sin(s * math.pi * 2.6 + 0.4) * np.clip(1 - s * s, 0, 1) ** 0.3
    a_up = float(np.clip(0.18 + 0.32 * c.pucker + 0.30 * c.upper_lip - 0.12 * max(c.smile, 0.0)
                         + 0.18 * max(-c.smile, 0.0) + 0.30 * grit, 0.05, 0.62))
    pu = max(0.15, 0.95 - 0.45 * c.pucker - 0.55 * grit)
    pd = max(0.15, 0.70 - 0.20 * c.pucker - 0.42 * grit)
    env = np.clip(1 - s * s, 0, 1)
    U = base - h * a_up * env ** pu - 0.02 * c.upper_lip * env ** pu
    D = base + h * (1.0 - a_up) * env ** pd
    return cx + hw * s, U, D, base, h, hw, cx, cy


def draw_mouth(ctx, head: HeadFrame, st: ToonStyle, c: FaceControls, dv: float = 0.0) -> None:
    R = head.R
    lw = st.lw(R)
    fine = st.lw(R, True)
    x, U, D, base, h, hw, cx, cy = mouth_curves(st, c, 64, dv)
    vis = head.visibility((cx, cy), -0.25, 0.05)
    if vis <= 0.01:
        return
    grit = c.mouth_glyph if c.mouth_mode == "grit" else 0.0
    W = lambda p: head.wrap(np.asarray(p, float).reshape(-1, 2))[0]          # noqa: E731
    env = np.clip(1 - np.linspace(-1, 1, len(x)) ** 2, 0, 1)
    ink.push(ctx)
    pu_px, pd_px = W(np.c_[x, U]), W(np.c_[x, D])
    open_px = h * R
    if open_px > 0.8:
        cav = np.vstack([pu_px, pd_px[::-1]])
        ink.fill(ctx, cav, st.cavity)
        ctx.save()
        ink.path(ctx, cav, True)
        ctx.clip()
        if c.tongue > 0.02:
            tw = hw * (0.62 + 0.1 * min(c.width, 1.3))
            th_ = h * (0.42 + 0.25 * c.tongue)
            tcy = float(np.interp(cx, x, D)) - h * 0.12 - h * 0.55 * c.tongue_up
            ta = min(1.0, c.tongue * 1.4)
            ink.fill(ctx, W(ink.ellipse_pts(cx, tcy + th_ * 0.35, tw, th_, 48)), st.tongue, ta)
            ax = np.linspace(-1, 1, 24)
            arc = np.c_[cx + tw * 0.8 * ax, tcy + th_ * 0.35 - th_ * np.sqrt(np.clip(1 - (0.8 * ax) ** 2, 0, 1))]
            ink.tapered(ctx, W(arc[6:-6]), mix(st.tongue, st.cavity, 0.5), fine * 0.9, kind="both", tip=0.2,
                        alpha=ta * 0.8)
        tu, tl = max(c.teeth_up, grit), max(c.teeth_low, grit)
        if grit > 0.02:
            mid = (U + D) * 0.5 + 0.008 * c.wobble * np.sin(np.linspace(0, 9, len(U)))
            for G in (np.vstack([np.c_[x, U - 0.01], np.c_[x, mid][::-1]]),
                      np.vstack([np.c_[x, mid], np.c_[x, D + 0.01][::-1]])):
                ink.fill(ctx, W(G), st.teeth, grit)
            ink.tapered(ctx, W(np.c_[x, mid]), st.ink, fine, kind="both", tip=0.4, alpha=grit)
            nseg = 6 if hw > 0.15 else 4
            for k in range(1, nseg):
                xx = cx + hw * (-1 + 2 * k / nseg)
                seg = np.array([[xx, float(np.interp(xx, x, U)) - 0.01], [xx, float(np.interp(xx, x, D)) + 0.01]])
                ink.stroke(ctx, W(seg), st.ink, fine * 0.85, alpha=grit)
        if tu > 0.02 and grit < 0.98:
            th_u = tu * min(h * 0.42, 0.05) + 0.004
            edge = np.c_[x, U + th_u * env ** 0.25]
            ink.fill(ctx, W(np.vstack([np.c_[x, U - 0.01], edge[::-1]])), st.teeth, 1.0 - grit)
            ink.tapered(ctx, W(edge[6:-6]), st.ink, fine * 0.8, kind="both", tip=0.2, alpha=(1 - grit) * 0.85)
        if tl > 0.02 and grit < 0.98:
            th_l = tl * min(h * 0.32, 0.04) + 0.003
            edge = np.c_[x, D - th_l * env ** 0.3]
            ink.fill(ctx, W(np.vstack([edge, np.c_[x, D + 0.01][::-1]])), st.teeth, 1.0 - grit)
            ink.tapered(ctx, W(edge[8:-8]), st.ink, fine * 0.8, kind="both", tip=0.2, alpha=(1 - grit) * 0.85)
        ctx.restore()
    if c.lip_bite > 0.02:                                   # F / V : upper teeth resting on the lower lip
        b = c.lip_bite
        sb = np.linspace(-0.55, 0.55, 24)
        xx = cx + hw * sb
        yy = np.interp(xx, x, base)
        tooth = np.vstack([np.c_[xx, yy - 0.003], np.c_[xx, yy + 0.034 * b * np.clip(1 - sb ** 2, 0, 1) ** 0.2][::-1]])
        Pt = W(tooth)
        ink.fill(ctx, Pt, st.teeth, b)
        ink.stroke(ctx, Pt, st.ink, fine * 0.85, alpha=b, closed=True)
        sl = np.linspace(-1, 1, 24)
        xl = cx + hw * 0.62 * sl
        lip = np.c_[xl, np.interp(xl, x, base) + 0.046 * b + 0.012 * b * (1 - sl ** 2)]
        ink.tapered(ctx, W(lip), st.ink, lw * 0.75, kind="both", tip=0.25, alpha=b)
    weight = lw * (1.0 + 0.25 * c.press + 0.4 * c.pucker)
    if open_px <= 0.8:
        ink.tapered(ctx, W(np.c_[x, base]), st.ink, weight * 1.05, kind="both", tip=0.28, power=0.55)
    else:
        k_open = smoothstep(0.8, 6.0, open_px)
        ink.tapered(ctx, pu_px, st.ink, weight * (1.0 - 0.12 * k_open), kind="both", tip=0.38, power=0.5)
        ink.tapered(ctx, pd_px, st.ink, weight * (1.0 + 0.12 * k_open), kind="both", tip=0.38, power=0.5)
    for sgn, idx in ((-1, 0), (1, -1)):
        lift_amt = cy - base[idx]
        p0 = np.array([x[idx], base[idx]])
        if lift_amt > 0.04 and c.pucker < 0.5:              # smile dimples
            k = smoothstep(0.04, 0.08, lift_amt)
            seg = np.array([p0 + [sgn * 0.004, -0.014], p0 + [sgn * 0.024, 0.004], p0 + [sgn * 0.015, 0.026]])
            ink.tapered(ctx, W(seg), st.ink, fine * 1.1, kind="both", tip=0.2, alpha=k)
        if c.press > 0.3:                                    # pressed-lip corner bumps
            seg = np.array([p0 + [sgn * 0.006, -0.016], p0 + [sgn * 0.013, 0.0], p0 + [sgn * 0.006, 0.016]])
            ink.tapered(ctx, W(seg), st.ink, fine, kind="both", tip=0.2, alpha=smoothstep(0.3, 0.8, c.press))
    ink.pop(ctx, vis)


# ================================================================================================== FX
def _cheek_fx(ctx, head: HeadFrame, st: ToonStyle, c: FaceControls, t: float) -> None:
    R = head.R
    blush = c.get_fx("blush")
    if blush > 0.01:
        for side in (-1, 1):
            anchor = (side * 0.43, 0.30)
            vis = head.visibility(anchor, -0.1, 0.15)
            if vis <= 0:
                continue
            o, M = head.decal(anchor, 0.5)
            ctx.save()
            ctx.transform(cairo.Matrix(M[0, 0] * 0.17, M[1, 0] * 0.17, M[0, 1] * 0.08, M[1, 1] * 0.08, o[0], o[1]))
            g = cairo.RadialGradient(0, 0, 0.0, 0, 0, 1.0)
            g.add_color_stop_rgba(0, *st.blush, 0.6 * blush * vis)
            g.add_color_stop_rgba(1, *st.blush, 0.0)
            ctx.set_source(g)
            ctx.arc(0, 0, 1.0, 0, 2 * math.pi)
            ctx.fill()
            ctx.restore()
            for k in range(3):
                a0 = -0.07 + 0.06 * k
                seg = np.array([[a0 + 0.02, -0.03], [a0 - 0.012, 0.03]])
                ink.tapered(ctx, o + seg @ M.T, mix(st.blush, st.ink, 0.35), st.lw(R, True) * 0.9, kind="both",
                            tip=0.25, alpha=0.75 * blush * vis)
    puff = c.get_fx("puff")
    if puff > 0.01:
        for side in (-1, 1):
            at = np.linspace(-0.9, 0.9, 20)
            Pp, _ = head.wrap(np.c_[side * (0.40 + 0.05 * np.cos(at)), 0.50 + 0.10 * np.sin(at)])
            ink.tapered(ctx, Pp, st.ink, st.lw(R, True) * 1.1, kind="both", tip=0.2,
                        alpha=puff * head.visibility((side * 0.4, 0.5), -0.1, 0.12))
    cold = c.get_fx("cold")
    if cold > 0.01:
        top, _ = head.wrap(np.array([[0.0, -0.7]]))
        mid, _ = head.wrap(np.array([[0.0, 0.3]]))
        g = cairo.LinearGradient(top[0, 0], top[0, 1], mid[0, 0], mid[0, 1])
        g.add_color_stop_rgba(0, *st.cold, 0.8 * cold)
        g.add_color_stop_rgba(0.55, *st.cold, 0.5 * cold)
        g.add_color_stop_rgba(1, *st.cold, 0.0)
        ctx.set_source(g)
        ctx.arc(head.cx, head.cy, head.R * 1.3, 0, 2 * math.pi)
        ctx.fill()


def _drop(ctx, x, y, size, rgb, ink_rgb, lw, alpha=1.0):
    t = np.linspace(0, 2 * math.pi, 48)
    pts = np.c_[x + size * 0.42 * np.sin(t) * (1 - np.cos(t)), y - size * 0.85 * np.cos(t)]
    ink.fill(ctx, pts, rgb, alpha)
    ink.stroke(ctx, pts, ink_rgb, lw, alpha=alpha, closed=True)
    ink.circle(ctx, x - size * 0.15, y + size * 0.25, size * 0.12, (1, 1, 1), alpha * 0.9)


def _star4(ctx, x, y, r, rgb, alpha=1.0, ink_rgb=None, lw=1.0):
    pts = []
    for i in range(8):
        rr = r if i % 2 == 0 else r * 0.3
        a = i * math.pi / 4 - math.pi / 2
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    ink.fill(ctx, np.array(pts), rgb, alpha)
    if ink_rgb is not None:
        ink.stroke(ctx, np.array(pts), ink_rgb, lw, alpha=alpha, closed=True)


def _outer_fx(ctx, head: HeadFrame, st: ToonStyle, c: FaceControls, t: float) -> None:
    R, cx, cy, roll = head.R, head.cx, head.cy, head.roll
    fine = st.lw(R, True)

    def hp(dx, dy):
        cr, sr = math.cos(roll), math.sin(roll)
        return cx + R * (dx * cr - dy * sr), cy + R * (dx * sr + dy * cr)

    sweat = c.get_fx("sweat")
    if sweat > 0.01:
        spots = [(0.62, -0.42), (-0.62, -0.15), (0.72, 0.12)]
        n = 1 + int(sweat > 0.5) + int(sweat > 0.85)
        for k, uv in enumerate(spots[:n]):
            Pp, z = head.wrap(np.array([uv]))
            if z[0] < -0.05:
                continue
            fall = (t * 0.35 + k * 0.37) % 1.0
            _drop(ctx, Pp[0, 0], Pp[0, 1] + fall * 0.12 * R, 0.09 * R * (1 - 0.2 * k), st.tear, st.ink, fine,
                  sweat * (1 - smoothstep(0.75, 1.0, fall)))
    tears = c.get_fx("tears")
    if tears > 0.01:
        for side in (-1, 1):
            Pp, z = head.wrap(np.array([[side * (st.eye_u + st.eye_rx * 0.75), st.eye_v + 0.04]]))
            if z[0] < -0.05:
                continue
            x0, y0 = Pp[0]
            s = np.linspace(0, 1, 32)
            amp = 0.55 * R * tears
            xs = x0 + side * amp * 0.9 * s
            ys = y0 - 0.10 * R * np.sin(s * math.pi) + 0.25 * R * s ** 2
            wob = 0.08 * R * (0.6 + 0.4 * np.sin(t * 12 + s * 6)) * np.sin(np.pi * np.clip(s * 1.1, 0, 1))
            poly = np.vstack([np.c_[xs, ys - wob * 0.6], np.c_[xs, ys + wob * 0.5][::-1]])
            ink.fill(ctx, poly, st.tear, 0.95 * tears)
            ink.stroke(ctx, poly, st.ink, fine, alpha=tears, closed=True)
            ink.stroke(ctx, np.c_[xs[4:20], ys[4:20] - 0.012 * R], (1, 1, 1), fine * 0.9, alpha=0.7 * tears)
            for k in range(3):
                ph = (t * 1.8 + k / 3.0) % 1.0
                ink.circle(ctx, x0 + side * amp * (0.95 + 0.25 * ph), y0 + 0.25 * R + 0.35 * R * ph ** 1.5,
                           0.032 * R * (1 - ph * 0.5), st.tear, tears * (1 - ph), st.ink, fine * 0.7)
    stress = c.get_fx("stress")
    if stress > 0.01:
        for k, a in enumerate((-150, -120, -60, -30, 35, 145)):
            a = math.radians(a + 4 * math.sin(t * 20 + k))
            for j in (-1, 0, 1):
                aa = a + j * 0.09
                r1 = 1.27 + 0.05 * (j == 0)
                ink.tapered(ctx, np.array([hp(1.12 * math.cos(aa), 1.12 * math.sin(aa)),
                                           hp(r1 * math.cos(aa), r1 * math.sin(aa))]),
                            st.ink, fine * 1.3, kind="end", tip=0.2, alpha=stress)
    zzz = c.get_fx("zzz")
    if zzz > 0.01:
        for k in range(3):
            ph = (t * 0.45 + k / 3.0) % 1.0
            size = 0.11 * R * (0.7 + 0.25 * k) * (0.8 + 0.4 * ph)
            x0, y0 = hp(0.85 + 0.25 * k + 0.1 * ph, -0.9 - 0.22 * k - 0.2 * ph)
            Z = np.array([[x0, y0], [x0 + size, y0], [x0, y0 + size], [x0 + size, y0 + size]])
            ink.stroke(ctx, Z, st.ink, fine * 1.3, alpha=zzz * (1 - smoothstep(0.8, 1.0, ph)))
    idea = c.get_fx("idea")
    if idea > 0.01:
        bx, by = hp(1.02, -1.02)
        r = 0.16 * R * (1 + 0.04 * math.sin(t * 10))
        ink.circle(ctx, bx, by, r * 1.9, (1.0, 0.95, 0.55), 0.25 * idea)
        ink.circle(ctx, bx, by, r, (1.0, 0.93, 0.45), idea, st.ink, fine * 1.2)
        base = np.array([[bx - r * 0.45, by + r * 0.85], [bx + r * 0.45, by + r * 0.85],
                         [bx + r * 0.38, by + r * 1.35], [bx - r * 0.38, by + r * 1.35]])
        ink.fill(ctx, base, (0.85, 0.85, 0.85), idea)
        ink.stroke(ctx, base, st.ink, fine, alpha=idea, closed=True)
        fil = np.array([[bx - r * 0.3, by + r * 0.4], [bx - r * 0.12, by - r * 0.15], [bx, by + r * 0.2],
                        [bx + r * 0.12, by - r * 0.15], [bx + r * 0.3, by + r * 0.4]])
        ink.stroke(ctx, fil, st.ink, fine * 0.8, alpha=idea)
        for k in range(7):
            a = math.radians(-200 + k * 36)
            ink.tapered(ctx, np.array([(bx + math.cos(a) * r * 1.35, by + math.sin(a) * r * 1.35),
                                       (bx + math.cos(a) * r * 1.85, by + math.sin(a) * r * 1.85)]),
                        st.ink, fine * 1.1, kind="both", tip=0.3, alpha=idea)
    dizzy = c.get_fx("dizzy")
    if dizzy > 0.01:
        ox, oy = hp(0.0, -1.0)
        th = np.linspace(0, 2 * math.pi, 80)
        ink.stroke(ctx, np.c_[ox + 0.95 * R * np.cos(th), oy + 0.2 * R * np.sin(th)], st.ink, fine * 1.1,
                   alpha=dizzy * 0.9, closed=True)
        ink.stroke(ctx, np.c_[ox + 1.05 * R * np.cos(th[10:40]), oy - 0.06 * R + 0.22 * R * np.sin(th[10:40])],
                   st.ink, fine, alpha=dizzy * 0.7)
        for k in range(3):
            a = t * 3.0 + k * 2 * math.pi / 3
            _star4(ctx, ox + 0.95 * R * math.cos(a), oy + 0.2 * R * math.sin(a), 0.09 * R, (1.0, 0.86, 0.3), dizzy,
                   st.ink, fine * 0.8)
    notes = c.get_fx("notes")
    if notes > 0.01:
        Pp, _ = head.wrap(np.array([[st.mouth_w + 0.15 + c.shift_x, st.mouth_v]]))
        for k in range(2):
            ph = (t * 0.6 + k * 0.5) % 1.0
            x0 = Pp[0, 0] + (0.25 + 0.35 * ph + 0.22 * k) * R
            y0 = Pp[0, 1] - (0.1 + 0.45 * ph) * R + 0.05 * R * math.sin(t * 6 + k)
            a = notes * (1 - smoothstep(0.75, 1.0, ph)) * smoothstep(0.0, 0.12, ph)
            hs = 0.05 * R
            ink.fill(ctx, ink.ellipse_pts(x0, y0, hs * 1.15, hs * 0.8, 24, -0.4), st.ink, a)
            ink.stroke(ctx, np.array([[x0 + hs, y0], [x0 + hs, y0 - hs * 3.2]]), st.ink, fine * 1.2, alpha=a)
            if k == 1:
                ink.fill(ctx, ink.ellipse_pts(x0 + hs * 2.6, y0 + hs * 0.3, hs * 1.15, hs * 0.8, 24, -0.4), st.ink, a)
                ink.stroke(ctx, np.array([[x0 + hs * 3.6, y0 + hs * 0.3], [x0 + hs * 3.6, y0 - hs * 2.9]]), st.ink,
                           fine * 1.2, alpha=a)
                ink.tapered(ctx, np.array([[x0 + hs, y0 - hs * 3.2], [x0 + hs * 3.6, y0 - hs * 2.9]]), st.ink,
                            fine * 2.2, kind="none", alpha=a)
            else:
                ink.tapered(ctx, np.array([[x0 + hs, y0 - hs * 3.2], [x0 + hs * 2.0, y0 - hs * 2.2]]), st.ink,
                            fine * 1.6, kind="end", alpha=a)
    shiver = c.get_fx("shiver")
    if shiver > 0.01:
        for sgn in (-1, 1):
            for k in range(3):
                j = 0.015 * R * math.sin(t * 40 + k + sgn)
                x0, y0 = hp(sgn * (1.13 + 0.07 * k), -0.15 + 0.18 * k)
                x1, y1 = hp(sgn * (1.13 + 0.07 * k), 0.05 + 0.18 * k)
                ink.tapered(ctx, np.array([[x0 + j, y0], [x0 - sgn * 0.03 * R + j, (y0 + y1) / 2], [x1 + j, y1]]),
                            st.ink, fine * 1.1, kind="both", tip=0.2, alpha=shiver)
    sparkle = c.get_fx("sparkle")
    if sparkle > 0.01:
        for k, (dx, dy) in enumerate(((-1.05, -0.65), (1.1, -0.45), (0.95, 0.75))):
            tw = 0.5 + 0.5 * math.sin(t * 7 + k * 2)
            x0, y0 = hp(dx, dy)
            _star4(ctx, x0, y0, 0.1 * R * (0.6 + 0.5 * tw), (1.0, 0.9, 0.4), sparkle)
    crumbs = c.get_fx("crumbs")
    if crumbs > 0.01:
        Pp, _ = head.wrap(np.array([[c.shift_x, st.mouth_v + 0.02]]))
        rng = np.random.default_rng(7)
        for k in range(7):
            ang = rng.uniform(0, 2 * math.pi)
            rr = rng.uniform(0.2, 0.33) * R
            x0 = Pp[0, 0] + math.cos(ang) * rr * 1.1
            y0 = Pp[0, 1] + abs(math.sin(ang)) * rr * 0.6 + 0.06 * R + 0.01 * R * math.sin(t * 9 + k)
            ink.circle(ctx, x0, y0, 0.022 * R * rng.uniform(0.7, 1.3), mix(st.skin, (0.85, 0.65, 0.4), 0.6),
                       crumbs, st.ink, fine * 0.8)
    vein = c.get_fx("anger_vein")
    if vein > 0.01:
        Pp, _ = head.wrap(np.array([[0.48, -0.55]]))
        x0, y0 = Pp[0]
        s = 0.08 * R * (1 + 0.1 * math.sin(t * 14))
        for a in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
            ca, sa = math.cos(a + 0.6), math.sin(a + 0.6)
            arc = np.array([[x0 + ca * s * 0.35 - sa * s * 0.5, y0 + sa * s * 0.35 + ca * s * 0.5],
                            [x0 + ca * s * 0.9, y0 + sa * s * 0.9],
                            [x0 + ca * s * 0.35 + sa * s * 0.5, y0 + sa * s * 0.35 - ca * s * 0.5]])
            ink.tapered(ctx, arc, (0.85, 0.15, 0.15), fine * 1.8, kind="both", tip=0.3, alpha=vein)


# ================================================================================================== ENTRY
def draw_face(ctx: cairo.Context, head: HeadFrame, st: ToonStyle, c: FaceControls, t: float = 0.0,
              clip_fn=None, eye_clip_fn=None, draw_fx: bool = True) -> None:
    """clip_fn(ctx): clip for skin features (mouth, nose, cheeks, brows).  eye_clip_fn(ctx): optional looser clip for
    eyes (toon far-eyes may overlap the cheek contour)."""
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    dv_eyes, dv_mouth = -0.04 * c.squash, 0.05 * c.squash
    ctx.save()
    if clip_fn is not None:
        clip_fn(ctx)
    _cheek_fx(ctx, head, st, c, t)
    draw_nose(ctx, head, st, c)
    draw_mouth(ctx, head, st, c, dv_mouth)
    ctx.restore()
    ctx.save()
    if eye_clip_fn is not None:
        eye_clip_fn(ctx)
    elif clip_fn is not None:
        clip_fn(ctx)
    draw_eyes(ctx, head, st, c, t, dv_eyes)
    draw_brows(ctx, head, st, c, dv_eyes)
    ctx.restore()
    if draw_fx:
        _outer_fx(ctx, head, st, c, t)
