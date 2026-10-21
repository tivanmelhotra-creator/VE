"""
mograph.v14.glassbadge — NESTED glass badge (the @novadent heptagon, measured from the reference frames).

Anatomy (outside → in):
  ① outline   thin ACCENT stroke (palette) (≈2.6 px) + soft orange emission that lights the speaker/clothes
  ② GAP       `gap` px (default 7) of UNTOUCHED footage between the outline and the glass body
  ③ body      inset copy of the shape: frosted blur of what is under it + DARK RADIAL CORE
              (centre ≈ 82 % dark → rim ≈ 15 %) so text reads while the rim stays glassy
  ④ hairline  1 px white @ 16 % on the body edge (glass thickness) + top-left sheen
  ⑤ specular  warm bloom at the head of the outline while it draws on, then slowly orbiting
Entrance: outline trims on (0.45 s silk) while the body jelly-springs from the centre; text pops last.
v14.8: the body is REFRACTIVE glass — lens bend at the rim (`refract` px), R/C dispersion (`chroma` px) and a
specular caustic (`caustic` 0..1) from compositor._glass_optics; set any of them to 0 per layer to disable.
Shapes: hept (default) · hex · pent · oct · circle · diamond · rsquare · pill
"""
from __future__ import annotations

import math

import cairo

from . import glyphs as G
from . import physics as ph
from .compositor import glass


def _poly_pts(shape, x, y, w, h, rot):
    n = {"pent": 5, "hex": 6, "hept": 7, "oct": 8, "diamond": 4}.get(shape)
    if n is None:
        return None
    off = -90.0 if shape != "diamond" else -90.0
    pts = []
    for i in range(n):
        a = math.radians(off + i * 360.0 / n + rot)
        pts.append((x + w / 2 * math.cos(a), y + h / 2 * math.sin(a)))
    return pts


def _rounded(c, pts, r):
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1 = (p0[0] - p1[0], p0[1] - p1[1]); v2 = (p2[0] - p1[0], p2[1] - p1[1])
        l1, l2 = math.hypot(*v1), math.hypot(*v2)
        rr = min(r, l1 / 2, l2 / 2)
        a = (p1[0] + v1[0] / l1 * rr, p1[1] + v1[1] / l1 * rr)
        b = (p1[0] + v2[0] / l2 * rr, p1[1] + v2[1] / l2 * rr)
        c.move_to(*a) if i == 0 else c.line_to(*a)
        c.curve_to(p1[0], p1[1], p1[0], p1[1], *b)
    c.close_path()


def shape_path(shape, x, y, w, h, rad=0.14, rot=0.0):
    def fn(c):
        if shape == "circle":
            c.new_sub_path(); c.arc(x, y, w / 2, 0, 2 * math.pi); c.close_path()
            return
        if shape in ("rsquare", "pill"):
            r = (h / 2) if shape == "pill" else min(w, h) * rad
            c.save(); c.translate(x, y); c.rotate(math.radians(rot))
            c.new_sub_path()
            c.arc(w / 2 - r, -h / 2 + r, r, -math.pi / 2, 0)
            c.arc(w / 2 - r, h / 2 - r, r, 0, math.pi / 2)
            c.arc(-w / 2 + r, h / 2 - r, r, math.pi / 2, math.pi)
            c.arc(-w / 2 + r, -h / 2 + r, r, math.pi, 3 * math.pi / 2)
            c.close_path(); c.restore()
            return
        _rounded(c, _poly_pts(shape, x, y, w, h, rot), min(w, h) * rad * 0.6)
    return fn


def _flat(path_fn):
    s = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
    c = cairo.Context(s)
    c.set_tolerance(0.3)
    path_fn(c)
    pts = [q[1] for q in c.copy_path_flat() if q[0] in (cairo.PATH_MOVE_TO, cairo.PATH_LINE_TO)]
    if pts:
        pts.append(pts[0])
    return pts


def _trace(pts, p):
    seg = [math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts))]
    tot = sum(seg) or 1.0
    lim, acc, out = tot * p, 0.0, [pts[0]]
    head = pts[0]
    for i, d in enumerate(seg):
        if acc + d >= lim:
            f = (lim - acc) / max(d, 1e-6)
            head = (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f)
            out.append(head)
            return out, head
        out.append(pts[i + 1])
        acc += d
    return out, pts[-1]


def draw_glass_badge(ctx, L, F, pos, glyph_motion, blk, out_lt):
    lt = F.t - L["_t0"]
    olt = out_lt(L, F)
    if olt is not None and olt > 0.5:
        return
    x, y, k = pos(L, F)
    pal = L["_pal"]
    hw = F.head_px()
    shape = L.get("shape", "hept")
    w = float(L.get("w", 1.6)) * hw
    h = w if shape == "circle" else float(L.get("h", 1.6)) * hw
    p_line = ph.SILK(min(1.0, max(0.0, lt) / float(L.get("draw_dur", 0.45))))
    p_body = ph.sp(lt - 0.08, L.get("spring", "jelly"))
    jx, jy, _ = ph.jelly(lt - 0.08, 0.8, 2.8, 3.8)
    pul = L.get("_pulse_at")
    for bt in (pul if isinstance(pul, list) else ([pul] if pul is not None else [])):
        if F.t > bt:
            a1, b1, _ = ph.jelly(F.t - bt, 0.9, 3.4, 4.5)
            jx *= a1; jy *= b1
    a = 1.0
    if olt is not None and olt > 0:
        q = ph.ease_in(min(1, olt / 0.32))
        p_body *= 1 - q; p_line *= 1 - q; a = 1 - q
    if p_line <= 0.01 and p_body <= 0.01:
        return
    rot = float(L.get("rot", 0.0)) + ph.osc(F.t, 0.17, 1.6, 4) + (1 - min(1.0, max(0.0, p_body))) * float(L.get("spin_in", -18))
    gap = float(L.get("gap", 9.0)) * k
    sw_ = float(L.get("stroke_w", 2.6)) * k
    ow, oh = w * jx, h * jy
    pb = max(0.0, p_body)
    pbs = min(1.0, pb)                                     # body never covers the gap, even at spring overshoot
    over = max(0.0, pb - 1.0)
    bw = max(1.0, (ow - 2 * (gap + sw_)) * pbs * (1 - 0.5 * over))
    bh = max(1.0, (oh - 2 * (gap + sw_)) * pbs * (1 + 0.0 * over))
    rad = float(L.get("round", 0.14))
    outline = shape_path(shape, x, y, ow, oh, rad, rot)
    body = shape_path(shape, x, y, bw, bh, rad, rot)
    sc = G.rgba(L.get("stroke", "@accent"), pal)
    # ③ glass body (only inside the inset shape → the gap stays clean footage)
    if pb > 0.02:
        if F.pass_ == "front":
            glass(ctx, F, body, (x - bw, y - bh, x + bw, y + bh), blur=float(L.get("blur", 14)),
                  tint=(0, 0, 0, 0.0), lift=0.0, sat=0.75, alpha=a,
                  refract=L.get("refract"), chroma=L.get("chroma"), caustic=L.get("caustic"))
        r = max(bw, bh) / 2
        core = float(L.get("core", 0.82))
        ctx.save(); body(ctx)
        rg = cairo.RadialGradient(x, y, 0, x, y, r)
        rg.add_color_stop_rgba(0.0, 0.02, 0.02, 0.025, core * a)
        rg.add_color_stop_rgba(0.55, 0.03, 0.03, 0.035, core * 0.78 * a)
        rg.add_color_stop_rgba(1.0, 0.05, 0.04, 0.03, 0.15 * a)
        ctx.set_source(rg); ctx.fill_preserve()
        rg2 = cairo.RadialGradient(x, y, r * 0.6, x, y, r * 1.02)
        rg2.add_color_stop_rgba(0, sc[0], sc[1], sc[2], 0.0)
        rg2.add_color_stop_rgba(1, sc[0], sc[1], sc[2], 0.10 * a)
        ctx.set_source(rg2); ctx.fill_preserve()
        hl = [0.55 + 0.45 * c for c in sc[:3]]
        ctx.set_source_rgba(hl[0], hl[1], hl[2], 0.26 * a); ctx.set_line_width(1.2); ctx.stroke()
        ctx.restore()
        ctx.save(); body(ctx); ctx.clip()
        lg = cairo.LinearGradient(x - bw / 2, y - bh / 2, x + bw * 0.1, y + bh * 0.1)
        lg.add_color_stop_rgba(0, 1, 1, 1, 0.10 * a)
        lg.add_color_stop_rgba(0.35, 1, 1, 1, 0.0)
        ctx.set_source(lg); ctx.paint(); ctx.restore()
    # ① outline (trim-path draw-on) + emission
    pts = _flat(outline)
    if len(pts) > 2 and p_line > 0.005:
        tr, head = _trace(pts, p_line)

        def tpath(c):
            c.move_to(*tr[0])
            for q in tr[1:]:
                c.line_to(*q)
        for lw, aa in ((sw_ * 3.2, 0.20), (sw_, 1.0)):
            ctx.save(); tpath(ctx)
            ctx.set_source_rgba(sc[0], sc[1], sc[2], aa * a)
            ctx.set_line_width(lw); ctx.set_line_join(cairo.LINE_JOIN_ROUND); ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            ctx.stroke(); ctx.restore()
        if F.ectx is not None:
            F.ectx.save(); tpath(F.ectx)
            F.ectx.set_source_rgba(sc[0], sc[1], sc[2], a * float(L.get("glow", 0.9)))
            F.ectx.set_line_width(sw_ * 4.5); F.ectx.set_line_join(cairo.LINE_JOIN_ROUND); F.ectx.stroke(); F.ectx.restore()
        # ⑤ specular
        if p_line >= 0.999:
            _, head = _trace(pts, (0.08 + 0.05 * F.t) % 1.0)
        sx_, sy_ = head
        R_ = 20 * k
        rg = cairo.RadialGradient(sx_, sy_, 0, sx_, sy_, R_)
        hl2 = [0.4 + 0.6 * c for c in sc[:3]]
        rg.add_color_stop_rgba(0, 1, 1, 1, 0.9 * a)
        rg.add_color_stop_rgba(0.35, hl2[0], hl2[1], hl2[2], 0.32 * a)
        rg.add_color_stop_rgba(1, sc[0], sc[1], sc[2], 0)
        ctx.set_source(rg); ctx.arc(sx_, sy_, R_, 0, 2 * math.pi); ctx.fill()
        if F.ectx is not None:
            F.ectx.set_source_rgba(0.5 + .5 * sc[0], 0.5 + .5 * sc[1], 0.5 + .5 * sc[2], 0.85 * a); F.ectx.arc(sx_, sy_, 7 * k, 0, 2 * math.pi); F.ectx.fill()
    # text fitted inside the BODY
    if L.get("text") and pb > 0.05:
        b = blk(L)
        fw, fh = ow - 2 * (gap + sw_), oh - 2 * (gap + sw_)
        fit = min(1.0, (fw * 0.70) / max(1.0, b.w * k), (fh * 0.56) / max(1.0, b.h * k))
        L2 = dict(L)
        L2.setdefault("enter", "jelly_pop")
        L2.setdefault("idle", "none")
        L2["_t0"] = L["_t0"] + float(L.get("text_delay", 0.22))
        mot = glyph_motion(L2, F, b, k, F.t - L2["_t0"], olt)
        G.draw_block(ctx, b, x, y + float(L.get("text_dy", 0)) * hw, motion=mot, pal=pal, ectx=F.ectx,
                     scale=k * fit * float(L.get("text_scale", 1.0)), alpha=a, plane={"rz": rot * 0.4},
                     shadow={"alpha": .45, "dy": 3})
