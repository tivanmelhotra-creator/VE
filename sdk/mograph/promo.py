"""
mograph.promo — v9 Studio Apex: kinetic-promo, 3D and SaaS-UI mockup widgets (pure JSON, zero scripting).
Colours default to the chapter world's palette tokens (@accent @ink @paper @mute @panel).

  liquid_blob     organic morphing blob r(θ,t)=R(1+a1·sin(3θ+ω1t)+a2·cos(4θ−ω2t)); optional "text": base text in
                  @ink outside, the overlapping part INVERTED inside (clip to the blob)
                  {"type":"liquid_blob","x":540,"y":900,"r":300,"text":"شکل‌های سیال","size":130,"drift":[120,0]}
  iso_cube        true 3D rotating cube / prism, back-face culled, Lambert-shaded (@accent lit top)
                  {"type":"iso_cube","x":540,"y":820,"s":300,"rx":-28,"ry":38,"wy":40,"shape":"cube|prism","label":"…"}
  starburst       rotating spiky impact badge {"type":"starburst","x":540,"y":800,"r":330,"points":16,"text":"شوخیت!"}
  prompt_box      glass AI prompt bar: label chip, neon border + bloom, RTL typewriter, caret, send button pulse
                  {"type":"prompt_box","x":540,"y":900,"w":900,"h":200,"label":"Claude","prompt":"…","type_at":"w:…"}
  timeline_ui     NLE window mock: traffic lights, layer bars, keyframes ◆, scrubbing playhead, cursor,
                  "strike_at" slams a red ✕   {"type":"timeline_ui","title":"After Effects","strike_at":"w:باز"}
  bezier_graph    grid card, animated cubic S-curve + tangent handles ●───◆ {"from":[.5,0,.5,1],"to":[.9,0,.1,1],"flex_at":…}
  viewfinder_hud  corner brackets, pulsing REC •, live Persian timecode, speed lines {"tc_start":21,"speed_lines":true}
  filmstrip       tilted scrolling 35mm ribbon behind a hero word {"angle":-8,"speed":220,"word":"فریم"}
  sine_wave       thick round-capped live-drawn squiggle {"w":900,"amp":34,"wavelength":180,"phase_speed":3}
"""
from __future__ import annotations

import math

import cairo
import numpy as np

from .core import clamp, color as _color, ease_in_out, ease_out, ease_out_expo, get_ease, lerp, prog, spring
from .elements import F, element, text_fx, xy
from .text import draw_text, text_bbox, to_fa_digits


def _c(el, key, s, token, default):
    from .typo import _col
    return _col(el, key, s, token, default)  # (el, key, state, palette-token, fallback)


def _rgba(ctx, c, a=1.0):
    ctx.set_source_rgba(c[0], c[1], c[2], c[3] * a)


def _shade(c, k):
    return (min(1, c[0] * k), min(1, c[1] * k), min(1, c[2] * k), c[3])


def _rrect(ctx, x0, y0, w, h, r):
    r = max(0.0, min(r, w / 2, h / 2))
    ctx.new_sub_path()
    ctx.arc(x0 + w - r, y0 + r, r, -math.pi / 2, 0)
    ctx.arc(x0 + w - r, y0 + h - r, r, 0, math.pi / 2)
    ctx.arc(x0 + r, y0 + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x0 + r, y0 + r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()


def _pop(s, el, stiff=320, damp=18):
    return spring(s.t, float(el["t"]), stiffness=stiff, damping=damp) if s.t >= float(el["t"]) else 0.0


def _xform(ctx, x, y, k, rot=0.0):
    ctx.translate(x, y)
    if rot:
        ctx.rotate(math.radians(rot))
    if k != 1.0:
        ctx.scale(k, k)
    ctx.translate(-x, -y)


def _glow_stroke(ctx, col, a, width, glow=True):
    path = ctx.copy_path()
    if glow:
        for wk, ak in ((width * 7, 0.06), (width * 4, 0.12), (width * 2.2, 0.22)):
            ctx.new_path(); ctx.append_path(path)
            _rgba(ctx, col, a * ak); ctx.set_line_width(wk); ctx.stroke()
    ctx.new_path(); ctx.append_path(path)
    _rgba(ctx, col, a); ctx.set_line_width(width); ctx.stroke()


def _on(fg, bg):
    from .contrast import ensure
    return ensure(fg, bg, 4.5, prefer=("#ffffff", "#000000"))


# ---------------------------------------------------------------------------
# 1. liquid_blob
# ---------------------------------------------------------------------------
def blob_points(x, y, R, t, a1=0.08, a2=0.06, w1=1.3, w2=0.9, n=72, seed=0.0):
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        r = R * (1 + a1 * math.sin(3 * th + w1 * t + seed) + a2 * math.cos(4 * th - w2 * t + seed * 0.7)
                 + 0.025 * math.sin(7 * th + 1.7 * t))
        pts.append((x + r * math.cos(th), y + r * math.sin(th)))
    return pts


def _smooth_closed(ctx, pts):
    n = len(pts)
    ctx.move_to(*pts[0])
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        ctx.curve_to(p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6,
                     p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6, p2[0], p2[1])
    ctx.close_path()


@element("liquid_blob", dur=0.9, enter="none", sfx="none")
def liquid_blob(ctx, el, s):
    x, y = xy(el, s.t)
    R = float(el.get("r", 300))
    acc = _c(el, "color", s, "accent", "#ff1e42")
    ink = _c(el, "text_color", s, "ink", "#ffffff")
    inv = _c(el, "invert_color", s, "paper", "#000000") if el.get("invert_color") else _on("#000000", acc)
    a = s.alpha * (1 - s.out_p)
    k = _pop(s, el, 200, 15) * (1 - 0.3 * s.out_p)
    if k <= 0.01 or a <= 0.01:
        return None
    lt = s.t - float(el["t"])
    dr = el.get("drift") or [0, 0]
    ds = float(el.get("drift_speed", 0.35))
    bx = x + float(dr[0]) * math.sin(ds * lt * math.pi)
    by = y + float(dr[1]) * math.sin(ds * lt * math.pi * 0.74 + 1.0)
    pts = blob_points(bx, by, R * k, lt, float(el.get("a1", 0.08)), float(el.get("a2", 0.06)),
                      float(el.get("w1", 1.3)), float(el.get("w2", 0.9)), seed=float(el.get("seed", 0)))
    txt = el.get("text")
    size = float(el.get("size", 130))
    fnt = F(el, "title")
    tx, ty = float(el.get("text_x", x)), float(el.get("text_y", y))
    fx = text_fx(el, s, size)
    mw = el.get("max_width")
    if txt:
        draw_text(ctx, txt, tx, ty, size=size, color=ink, alpha=a, weight=fnt, max_width=mw, line_height=1.15, **fx)
    ctx.save()
    if el.get("glow", True):
        g = cairo.RadialGradient(bx, by, R * 0.2 * k, bx, by, R * 1.5 * k)
        g.add_color_stop_rgba(0, acc[0], acc[1], acc[2], 0.35 * a)
        g.add_color_stop_rgba(1, acc[0], acc[1], acc[2], 0)
        ctx.set_source(g); ctx.arc(bx, by, R * 1.5 * k, 0, 2 * math.pi); ctx.fill()
    ctx.new_path(); _smooth_closed(ctx, pts)
    _rgba(ctx, acc, a)
    ctx.fill_preserve()
    if txt:
        ctx.clip()
        from . import text as _T
        _m = _T.AUDIT_MUTE; _T.AUDIT_MUTE = "all"          # intentional knockout inversion (not a layout)
        draw_text(ctx, txt, tx, ty, size=size, color=inv, alpha=a, weight=fnt, max_width=mw, line_height=1.15,
                  quiet=True, **fx)
        _T.AUDIT_MUTE = _m
    ctx.new_path()
    ctx.restore()
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    bb = (min(xs), min(ys), max(xs), max(ys))
    if txt:
        tb = text_bbox(txt, tx, ty, size, fnt, max_width=mw, line_height=1.15)
        bb = (min(bb[0], tb[0]), min(bb[1], tb[1]), max(bb[2], tb[2]), max(bb[3], tb[3]))
    return bb


# ---------------------------------------------------------------------------
# 2. iso_cube
# ---------------------------------------------------------------------------
def _rot(v, rx, ry, rz):
    x, y, z = v
    c, s_ = math.cos(rx), math.sin(rx); y, z = y * c - z * s_, y * s_ + z * c
    c, s_ = math.cos(ry), math.sin(ry); x, z = x * c + z * s_, -x * s_ + z * c
    c, s_ = math.cos(rz), math.sin(rz); x, y = x * c - y * s_, x * s_ + y * c
    return (x, y, z)


@element("iso_cube", dur=0.9, enter="none", sfx="none")
def iso_cube(ctx, el, s):
    x, y = xy(el, s.t)
    S = float(el.get("s", el.get("size", 300)))
    acc = _c(el, "color", s, "accent", "#ff1e42")
    edge = _c(el, "edge_color", s, "ink", "#ffffff")
    a = s.alpha * (1 - s.out_p)
    k = _pop(s, el, 220, 16) * (1 - 0.4 * s.out_p)
    if k <= 0.01 or a <= 0.01:
        return None
    lt = s.t - float(el["t"])
    spin_in = (1 - ease_out_expo(clamp(lt / 1.1))) * float(el.get("spin_in", 120))
    rx = math.radians(float(el.get("rx", -28)) + float(el.get("wx", 0)) * lt)
    ry = math.radians(float(el.get("ry", 38)) + float(el.get("wy", 40)) * lt + spin_in)
    rz = math.radians(float(el.get("rz", 0)) + float(el.get("wz", 0)) * lt)
    h = float(el.get("h", 1.0))
    hs = S / 2 * k
    if el.get("shape") == "prism":
        tri = [(0, -1), (0.866, 0.5), (-0.866, 0.5)]
        V = [(px * hs, py * hs, -hs * h) for px, py in tri] + [(px * hs, py * hs, hs * h) for px, py in tri]
        faces = [(0, 1, 2), (5, 4, 3), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
    else:
        V = [(sx * hs, sy * hs * h, sz * hs) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    P = [_rot(v, rx, ry, rz) for v in V]
    light = np.array(el.get("light", [-0.35, -0.85, -0.4]), float); light /= np.linalg.norm(light)
    fd = float(el.get("persp", 2600))
    cam = np.array([0, 0, -fd])

    def proj(p):
        q = fd / (fd + p[2])
        return (x + p[0] * q, y + p[1] * q)
    cen_all = np.mean(P, axis=0)
    vis = []
    for fi, f in enumerate(faces):
        a3, b3, c3 = np.array(P[f[0]]), np.array(P[f[1]]), np.array(P[f[2]])
        nrm = np.cross(b3 - a3, c3 - a3)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm /= ln
        cen = np.mean([P[i] for i in f], axis=0)
        if np.dot(nrm, cen - cen_all) < 0:          # make normals point outward
            nrm = -nrm
        if np.dot(nrm, cen - cam) >= 0:              # back-face cull
            continue
        vis.append((cen[2], fi, f, nrm))
    vis.sort(key=lambda v: -v[0])
    allp = []
    for _z, fi, f, nrm in vis:
        lam = max(0.0, float(np.dot(nrm, light)))
        col = acc if nrm[1] < -0.55 else _shade(acc, 0.3 + 0.55 * lam)
        if el.get("face_colors"):
            fc = el["face_colors"]; col = _shade(_color(fc[fi % len(fc)]), 0.55 + 0.45 * lam)
        pts = [proj(P[i]) for i in f]; allp += pts
        ctx.new_path(); ctx.move_to(*pts[0])
        for q in pts[1:]:
            ctx.line_to(*q)
        ctx.close_path()
        _rgba(ctx, col, a)
        if el.get("edges", True):
            ctx.fill_preserve(); _rgba(ctx, edge, a * 0.55); ctx.set_line_width(2.0)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND); ctx.stroke()
        else:
            ctx.fill()
    xs = [p[0] for p in allp] or [x]; ys = [p[1] for p in allp] or [y]
    bb = [min(xs), min(ys), max(xs), max(ys)]
    if el.get("label"):
        lsz = float(el.get("label_size", 64))
        lb = draw_text(ctx, el["label"], x, y + S * 0.95, size=lsz, color=_c(el, "label_color", s, "ink", "#fff"),
                       alpha=a * clamp((lt - 0.3) / 0.4), weight=F(el, "title"), **text_fx(el, s, lsz))
        bb = [min(bb[0], lb[0]), bb[1], max(bb[2], lb[2]), max(bb[3], lb[3])]
    return tuple(bb)


# ---------------------------------------------------------------------------
# 3. starburst
# ---------------------------------------------------------------------------
@element("starburst", dur=0.6, enter="none", sfx="none")
def starburst(ctx, el, s):
    x, y = xy(el, s.t)
    R = float(el.get("r", 330)); n = int(el.get("points", 16)); inner = float(el.get("inner", 0.78))
    acc = _c(el, "color", s, "accent", "#ff1e42")
    a = s.alpha * (1 - s.out_p)
    k = _pop(s, el, 420, 15) * (1 - 0.5 * s.out_p)
    if k <= 0.01 or a <= 0.01:
        return None
    lt = s.t - float(el["t"])
    rot = math.radians(float(el.get("angle", 0)) + float(el.get("spin", 14)) * lt - 40 * (1 - clamp(lt / 0.4)))
    ctx.save(); ctx.new_path()
    for i in range(2 * n):
        r = R * k * (1 if i % 2 == 0 else inner)
        th = rot + math.pi * i / n
        ctx.line_to(x + r * math.cos(th), y + r * math.sin(th))
    ctx.close_path(); _rgba(ctx, acc, a)
    if el.get("stroke"):
        ctx.fill_preserve(); _rgba(ctx, _color(el["stroke"]), a); ctx.set_line_width(float(el.get("stroke_w", 6)))
        ctx.stroke()
    else:
        ctx.fill()
    if el.get("rings", True):
        for j in range(2):
            q = clamp((lt - 0.06 * j) / 0.55)
            if 0 < q < 1:
                _rgba(ctx, acc, a * (1 - q) * 0.6); ctx.set_line_width(6 * (1 - q) + 1)
                ctx.arc(x, y, R * (1.0 + 0.5 * q), 0, 2 * math.pi); ctx.stroke()
    ctx.restore()
    if el.get("text"):
        size = float(el.get("size", 120)) * (0.6 + 0.4 * k)
        tc = _c(el, "text_color", s, "paper", "#000") if el.get("text_color") else _on("#000000", acc)
        ctx.save(); _xform(ctx, x, y, 1.0, float(el.get("text_rot", -6)))
        draw_text(ctx, el["text"], x, y, size=size, color=tc, alpha=a, weight=F(el, "title"), max_width=R * 1.5,
                  line_height=1.1, **text_fx(el, s, size))
        ctx.restore()
    return (x - R * k, y - R * k, x + R * k, y + R * k)


# ---------------------------------------------------------------------------
# 4. prompt_box
# ---------------------------------------------------------------------------
def prompt_done_time(el):
    t0 = float(el.get("type_at", float(el["t"]) + 0.4))
    return t0 + len(str(el.get("prompt", ""))) / float(el.get("cps", 16))


@element("prompt_box", dur=0.6, enter="none", sfx="none")
def prompt_box(ctx, el, s):
    from .text import _is_rtl_char
    x, y = xy(el, s.t)
    w, h = float(el.get("w", 900)), float(el.get("h", 200))
    acc = _c(el, "accent", s, "accent", "#ff1e42")
    ink = _c(el, "color", s, "ink", "#ffffff")
    mute = _c(el, "mute", s, "mute", "#8a8a90")
    panel = _c(el, "panel", s, "panel", "#140508")
    a0 = s.alpha * (1 - s.out_p)
    p_in = ease_out_expo(prog(s.t, float(el["t"]), 0.55))
    if a0 <= 0.01 or p_in <= 0.001:
        return None
    a = a0 * p_in
    dy = (1 - p_in) * 60
    x0, y0 = x - w / 2, y - h / 2 + dy
    r = float(el.get("radius", 36))
    ctx.save()
    _xform(ctx, x, y + dy, 0.92 + 0.08 * p_in)
    for pad, al in ((26, 0.05), (14, 0.09), (6, 0.16)):
        _rrect(ctx, x0 - pad, y0 - pad, w + 2 * pad, h + 2 * pad, r + pad)
        _rgba(ctx, acc, a * al * (0.8 + 0.2 * math.sin(s.t * 3))); ctx.fill()
    _rrect(ctx, x0, y0, w, h, r)
    g = cairo.LinearGradient(x0, y0, x0, y0 + h)
    hi = _shade(panel, 1.0)
    g.add_color_stop_rgba(0, min(1, hi[0] + 0.07), min(1, hi[1] + 0.07), min(1, hi[2] + 0.07), 0.94 * a)
    g.add_color_stop_rgba(1, panel[0], panel[1], panel[2], 0.97 * a)
    ctx.set_source(g); ctx.fill_preserve()
    _rgba(ctx, acc, a); ctx.set_line_width(3.0); ctx.stroke()
    _rrect(ctx, x0 + 6, y0 + 6, w - 12, h * 0.38, r - 6)
    sh = cairo.LinearGradient(x0, y0, x0, y0 + h * 0.4)
    sh.add_color_stop_rgba(0, 1, 1, 1, 0.07 * a); sh.add_color_stop_rgba(1, 1, 1, 1, 0)
    ctx.set_source(sh); ctx.fill()
    lab = el.get("label")
    if lab:
        lsz = float(el.get("label_size", 34))
        lb = text_bbox(lab, 0, 0, lsz, "ui")
        cw = (lb[2] - lb[0]) + lsz * 1.4
        cx0 = x0 + w - cw - 28
        _rrect(ctx, cx0, y0 - lsz * 0.75, cw, lsz * 1.5, lsz * 0.75); _rgba(ctx, acc, a); ctx.fill()
        draw_text(ctx, lab, cx0 + cw / 2, y0, size=lsz, color=_on("#ffffff", acc), alpha=a, weight="ui", quiet=True)
    full = str(el.get("prompt", ""))
    t0 = float(el.get("type_at", float(el["t"]) + 0.4))
    cps = float(el.get("cps", 16))
    n = int(max(0.0, s.t - t0) * cps)
    part, done, t_done = full[:n], n >= len(full), prompt_done_time(el)
    psz = float(el.get("size", 50))
    rtl = any(_is_rtl_char(c) for c in full)
    pad = 48
    bw = h * 0.42
    tx = (x0 + w - pad) if rtl else (x0 + pad + bw + 20)
    al = "right" if rtl else "left"
    maxw = w - 2 * pad - bw - 20
    if not part:
        draw_text(ctx, el.get("placeholder", "پیامت رو بنویس…" if rtl else "Ask anything…"), tx, y0 + h / 2,
                  size=psz, color=mute, alpha=a * 0.8, weight=F(el, "body"), align=al, quiet=True)
        cur_x = tx
    else:
        tb = text_bbox(part, 0, 0, psz, F(el, "body"))
        sz = psz * min(1.0, maxw / max(1.0, tb[2] - tb[0]))
        bb = draw_text(ctx, part, tx, y0 + h / 2, size=sz, color=ink, alpha=a, weight=F(el, "body"), align=al)
        cur_x = bb[0] - 10 if rtl else bb[2] + 10
    if (not done) or int(s.t * 2.4) % 2 == 0:
        _rgba(ctx, acc, a); ctx.rectangle(cur_x - 3, y0 + h / 2 - psz * 0.55, 6, psz * 1.05); ctx.fill()
    btn = el.get("button", "↑")
    if btn:
        bxc = (x0 + pad * 0.6 + bw / 2) if rtl else (x0 + w - pad * 0.6 - bw / 2)
        bxc = (x0 + pad * 0.6 + bw / 2)
        byc = y0 + h / 2
        sp = spring(s.t, t_done, stiffness=420, damping=12) if s.t >= t_done else 0.0
        bk = 1 + 0.22 * math.sin(math.pi * clamp(sp)) * (1 - clamp((s.t - t_done) / 0.6))
        bc = acc if done else mute
        ctx.save(); _xform(ctx, bxc, byc, bk)
        ctx.arc(bxc, byc, bw / 2, 0, 2 * math.pi); _rgba(ctx, bc, a); ctx.fill()
        if done and s.t - t_done < 0.8:
            q = clamp((s.t - t_done) / 0.8)
            _rgba(ctx, acc, a * (1 - q) * 0.7); ctx.set_line_width(4)
            ctx.arc(bxc, byc, bw / 2 + 30 * q, 0, 2 * math.pi); ctx.stroke()
        draw_text(ctx, btn, bxc, byc, size=bw * 0.62, color=_on("#ffffff", bc), alpha=a, weight="ui", quiet=True)
        ctx.restore()
    ctx.restore()
    return (x0 - 10, y0 - 40, x0 + w + 10, y0 + h + 10)


# ---------------------------------------------------------------------------
# 5a. timeline_ui
# ---------------------------------------------------------------------------
@element("timeline_ui", dur=0.6, enter="none", sfx="none")
def timeline_ui(ctx, el, s):
    x, y = xy(el, s.t)
    w, h = float(el.get("w", 920)), float(el.get("h", 560))
    ink = _c(el, "color", s, "ink", "#ffffff")
    mute = _c(el, "mute", s, "mute", "#8a8a90")
    panel = _color(el.get("panel", "#17171c"))
    a0 = s.alpha * (1 - s.out_p)
    p_in = ease_out_expo(prog(s.t, float(el["t"]), 0.6))
    if a0 <= 0.01 or p_in <= 0.001:
        return None
    a = a0 * p_in
    lt = s.t - float(el["t"])
    x0, y0 = x - w / 2, y - h / 2 + (1 - p_in) * 80
    ctx.save()
    _xform(ctx, x, y, 0.94 + 0.06 * p_in)
    _rrect(ctx, x0 + 10, y0 + 22, w, h, 26); ctx.set_source_rgba(0, 0, 0, 0.35 * a); ctx.fill()
    _rrect(ctx, x0, y0, w, h, 26); _rgba(ctx, panel, a); ctx.fill()
    tb = 64
    ctx.save(); _rrect(ctx, x0, y0, w, h, 26); ctx.clip()
    ctx.rectangle(x0, y0, w, tb); _rgba(ctx, _shade(panel, 1.6), a); ctx.fill(); ctx.restore()
    _rrect(ctx, x0, y0, w, h, 26); ctx.set_source_rgba(1, 1, 1, 0.12 * a); ctx.set_line_width(2); ctx.stroke()
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        _rgba(ctx, _color(c), a); ctx.arc(x0 + 38 + i * 32, y0 + tb / 2, 10, 0, 2 * math.pi); ctx.fill()
    tcol = (0.75, 0.75, 0.8, 1)
    draw_text(ctx, el.get("title", "After Effects"), x0 + w / 2, y0 + tb / 2, size=30, color=tcol, alpha=a,
              weight="ui", quiet=True)
    layers = el.get("layers") or ["متن", "شکل", "دوربین", "نور", "صدا"]
    cols = el.get("layer_colors") or ["@accent", "#7c5cff", "#21c7ff", "#ffb020", "#2bd67b"]
    from .typo import pal_of
    pal = pal_of(el, s)
    rows_top = y0 + tb + 70
    lh = (h - tb - 110) / max(1, len(layers))
    lab_w = w * 0.24
    tx0, tx1 = x0 + 30, x0 + w - lab_w - 20
    ctx.set_source_rgba(1, 1, 1, 0.3 * a); ctx.set_line_width(1.5)
    for j in range(13):
        xx = tx0 + (tx1 - tx0) * j / 12
        ctx.move_to(xx, rows_top - 34); ctx.line_to(xx, rows_top - (22 if j % 3 else 12))
    ctx.stroke()
    for i, nm in enumerate(layers):
        ly = rows_top + i * lh
        pi_ = ease_out_expo(prog(s.t, float(el["t"]) + 0.15 + i * 0.09, 0.5))
        cc = cols[i % len(cols)]
        cc = _color(pal.get(cc[1:], "#ff1e42") if isinstance(cc, str) and cc.startswith("@") else cc)
        draw_text(ctx, nm, x0 + w - 30, ly + lh / 2, size=min(34, lh * 0.42), color=(0.93, 0.93, 0.95, 1),
                  alpha=a * pi_, align="right", weight="ui", quiet=True)
        s0 = 0.05 + 0.12 * ((i * 37) % 5) / 5
        s1 = min(0.98, s0 + 0.45 + 0.1 * ((i * 53) % 4))
        bx0 = tx0 + (tx1 - tx0) * s0
        bx1 = tx0 + (tx1 - tx0) * lerp(s0, s1, pi_)
        _rrect(ctx, bx0, ly + lh * 0.22, max(4, bx1 - bx0), lh * 0.56, lh * 0.18); _rgba(ctx, cc, a * 0.9); ctx.fill()
        for kf in (s0 + 0.04, (s0 + s1) / 2, s1 - 0.04):
            kx = tx0 + (tx1 - tx0) * kf
            if kx > bx1:
                continue
            d = lh * 0.18
            ctx.move_to(kx, ly + lh / 2 - d); ctx.line_to(kx + d, ly + lh / 2); ctx.line_to(kx, ly + lh / 2 + d)
            ctx.line_to(kx - d, ly + lh / 2); ctx.close_path(); ctx.set_source_rgba(1, 1, 1, a * 0.95); ctx.fill()
    ph = 0.08 + 0.84 * (0.5 - 0.5 * math.cos(min(lt, 30) * float(el.get("scrub_speed", 0.9))))
    px = tx0 + (tx1 - tx0) * ph
    red = _color("#ff2d3a")
    _rgba(ctx, red, a); ctx.set_line_width(3)
    ctx.move_to(px, rows_top - 40); ctx.line_to(px, y0 + h - 26); ctx.stroke()
    ctx.move_to(px - 12, rows_top - 52); ctx.line_to(px + 12, rows_top - 52); ctx.line_to(px, rows_top - 36)
    ctx.close_path(); ctx.fill()
    cx_, cy_ = px + 26, rows_top + lh * (1.6 + 0.6 * math.sin(lt * 1.3))
    for dx_, dy_ in ((0, 0), (0, 44), (12, 33), (22, 54), (30, 50), (20, 30), (34, 30)):
        ctx.line_to(cx_ + dx_, cy_ + dy_)
    ctx.close_path(); ctx.set_source_rgba(1, 1, 1, a); ctx.fill_preserve()
    ctx.set_source_rgba(0, 0, 0, a); ctx.set_line_width(2); ctx.stroke()
    if el.get("strike_at") is not None and s.t >= float(el["strike_at"]):
        q = clamp((s.t - float(el["strike_at"])) / 0.28)
        sp = spring(s.t, float(el["strike_at"]), stiffness=480, damping=20)
        m = 40
        ctx.save(); _xform(ctx, x, y, 1.25 - 0.25 * sp); ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for (ax, ay, bx, by), qq in (((x0 + m, y0 + m, x0 + w - m, y0 + h - m), clamp(q * 2)),
                                     ((x0 + w - m, y0 + m, x0 + m, y0 + h - m), clamp(q * 2 - 1))):
            if qq > 0:
                ctx.new_path(); ctx.move_to(ax, ay); ctx.line_to(lerp(ax, bx, qq), lerp(ay, by, qq))
                _glow_stroke(ctx, red, a0, 34)
        ctx.restore()
    ctx.restore()
    return (x0, y0, x0 + w, y0 + h)


# ---------------------------------------------------------------------------
# 5b. bezier_graph
# ---------------------------------------------------------------------------
@element("bezier_graph", dur=0.6, enter="none", sfx="none")
def bezier_graph(ctx, el, s):
    x, y = xy(el, s.t)
    w, h = float(el.get("w", 760)), float(el.get("h", 560))
    acc = _c(el, "accent", s, "accent", "#ff1e42")
    panel = _color(el.get("panel", "#111114"))
    a0 = s.alpha * (1 - s.out_p)
    p_in = ease_out_expo(prog(s.t, float(el["t"]), 0.6))
    if a0 <= 0.01 or p_in <= 0.001:
        return None
    a = a0 * p_in
    x0, y0 = x - w / 2, y - h / 2 + (1 - p_in) * 60
    wh = (1, 1, 1, 1)
    ctx.save()
    _rrect(ctx, x0, y0, w, h, 28); _rgba(ctx, panel, a * 0.97); ctx.fill_preserve()
    ctx.set_source_rgba(1, 1, 1, 0.14 * a); ctx.set_line_width(2); ctx.stroke()
    pad = 70
    gx0, gy0, gx1, gy1 = x0 + pad, y0 + pad + 30, x0 + w - pad, y0 + h - pad
    ctx.set_source_rgba(1, 1, 1, 0.08 * a); ctx.set_line_width(1.2)
    for j in range(9):
        xx = gx0 + (gx1 - gx0) * j / 8; ctx.move_to(xx, gy0); ctx.line_to(xx, gy1)
        yy = gy0 + (gy1 - gy0) * j / 8; ctx.move_to(gx0, yy); ctx.line_to(gx1, yy)
    ctx.stroke()
    if el.get("title"):
        draw_text(ctx, el["title"], x0 + w / 2, y0 + 46, size=30, color=(0.7, 0.7, 0.75, 1), alpha=a, weight="ui",
                  quiet=True)
    cf, ct = el.get("from", [0.5, 0.0, 0.5, 1.0]), el.get("to", [0.9, 0.0, 0.1, 1.0])
    q = ease_in_out(prog(s.t, float(el.get("flex_at", float(el["t"]) + 0.8)), float(el.get("flex_dur", 1.0))))
    wob = 0.04 * math.sin(s.t * 2.2) * (1 - q * 0.5)
    c = [lerp(cf[i], ct[i], q) for i in range(4)]
    c[0] += wob; c[2] -= wob

    def P(u, v):
        return (gx0 + (gx1 - gx0) * u, gy1 - (gy1 - gy0) * v)
    p0, p3, p1, p2 = P(0, 0), P(1, 1), P(c[0], c[1]), P(c[2], c[3])
    dp = ease_out(prog(s.t, float(el["t"]) + 0.2, 0.9))
    for (ax, ay), (hx, hy) in ((p0, p1), (p3, p2)):
        hx, hy = lerp(ax, hx, dp), lerp(ay, hy, dp)
        ctx.set_source_rgba(1, 1, 1, 0.55 * a); ctx.set_line_width(2.4)
        ctx.move_to(ax, ay); ctx.line_to(hx, hy); ctx.stroke()
        d = 13
        ctx.move_to(hx, hy - d); ctx.line_to(hx + d, hy); ctx.line_to(hx, hy + d); ctx.line_to(hx - d, hy)
        ctx.close_path(); _rgba(ctx, wh, a); ctx.fill()
    N = 80
    m = max(2, int(N * dp))
    ctx.new_path()
    for i in range(m + 1):
        u = i / N
        bx = (1 - u) ** 3 * p0[0] + 3 * (1 - u) ** 2 * u * p1[0] + 3 * (1 - u) * u * u * p2[0] + u ** 3 * p3[0]
        by = (1 - u) ** 3 * p0[1] + 3 * (1 - u) ** 2 * u * p1[1] + 3 * (1 - u) * u * u * p2[1] + u ** 3 * p3[1]
        (ctx.move_to if i == 0 else ctx.line_to)(bx, by)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    _glow_stroke(ctx, acc, a, 7)
    for (ax, ay) in (p0, p3):
        _rgba(ctx, wh, a); ctx.arc(ax, ay, 11, 0, 2 * math.pi); ctx.fill()
    ctx.restore()
    bb = (x0, y0, x0 + w, y0 + h)
    if el.get("label"):
        lb = draw_text(ctx, el["label"], x, y0 + h + 80, size=float(el.get("label_size", 60)),
                       color=_c(el, "label_color", s, "ink", "#fff"), alpha=a, weight=F(el, "title"),
                       **text_fx(el, s, float(el.get("label_size", 60))))
        bb = (min(x0, lb[0]), y0, max(x0 + w, lb[2]), lb[3])
    return bb


# ---------------------------------------------------------------------------
# 6a. viewfinder_hud
# ---------------------------------------------------------------------------
@element("viewfinder_hud", dur=0.5, enter="none", sfx="none")
def viewfinder_hud(ctx, el, s):
    x, y = xy(el, s.t)
    w, h = float(el.get("w", 940)), float(el.get("h", 1500))
    ink = _c(el, "color", s, "ink", "#ffffff")
    red = _color(el.get("rec_color", "#ff2d3a"))
    a = s.alpha * (1 - s.out_p)
    p = ease_out_expo(prog(s.t, float(el["t"]), 0.55))
    if a <= 0.01 or p <= 0.001:
        return None
    a *= p
    lt = s.t - float(el["t"])
    k = 1.12 - 0.12 * p
    x0, y0, x1, y1 = x - w / 2 * k, y - h / 2 * k, x + w / 2 * k, y + h / 2 * k
    L = float(el.get("bracket", 110))
    ctx.save()
    ctx.set_line_width(float(el.get("width", 7))); ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
    _rgba(ctx, ink, a)
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        ctx.move_to(cx, cy + sy * L); ctx.line_to(cx, cy); ctx.line_to(cx + sx * L, cy)
    ctx.stroke()
    ctx.set_line_width(2.5); _rgba(ctx, ink, a * 0.7)
    ctx.move_to(x - 28, y); ctx.line_to(x + 28, y); ctx.move_to(x, y - 28); ctx.line_to(x, y + 28); ctx.stroke()
    blink = 0.35 + 0.65 * (0.5 + 0.5 * math.cos(lt * 2 * math.pi))
    rx, ry = x0 + 56, y0 + 70
    _rgba(ctx, red, a * blink); ctx.arc(rx, ry, 15, 0, 2 * math.pi); ctx.fill()
    draw_text(ctx, "REC", rx + 34, ry, size=40, color=ink, alpha=a, align="left", weight="ui", quiet=True)
    ts = float(el.get("tc_start", 0)) + max(0.0, lt)
    tc = f"{int(ts // 3600):02d}:{int(ts // 60) % 60:02d}:{int(ts) % 60:02d}:{int((ts % 1) * 24):02d}"
    if el.get("fa", True):
        tc = to_fa_digits(tc)
    draw_text(ctx, tc, x1 - 56, ry, size=40, color=ink, alpha=a, align="right", weight="counter", quiet=True)
    if el.get("label"):
        draw_text(ctx, el["label"], x1 - 56, y1 - 70, size=32, color=ink, alpha=a * 0.8, align="right", weight="ui",
                  quiet=True)
    for i in range(4):
        hh = 14 + i * 9
        _rgba(ctx, ink, a * (0.9 if i < 3 else 0.35)); ctx.rectangle(x0 + 56 + i * 18, y1 - 60 - hh, 10, hh); ctx.fill()
    if el.get("speed_lines"):
        n = int(el.get("lines", 48))
        rng = np.random.default_rng(int(el.get("seed", 5)))
        ang = rng.uniform(0, 2 * math.pi, n); off = rng.uniform(0, 1, n)
        R = math.hypot(w, h) * 0.55
        for i in range(n):
            f = (off[i] + lt * 0.9 * float(el.get("speed", 1.0))) % 1
            r0, ln = 120 + f ** 1.5 * R, 40 + 180 * f
            _rgba(ctx, ink, a * 0.35 * (1 - f) * clamp(f * 4)); ctx.set_line_width(2 + 2 * f)
            ctx.move_to(x + math.cos(ang[i]) * r0, y + math.sin(ang[i]) * r0)
            ctx.line_to(x + math.cos(ang[i]) * (r0 + ln), y + math.sin(ang[i]) * (r0 + ln)); ctx.stroke()
    ctx.restore()
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# 6b. filmstrip
# ---------------------------------------------------------------------------
@element("filmstrip", dur=0.6, enter="none", sfx="none")
def filmstrip(ctx, el, s):
    x, y = xy(el, s.t)
    w, h = float(el.get("w", 1700)), float(el.get("h", 300))
    film = _color(el.get("film_color", "#0c0c0e"))
    a = s.alpha * (1 - s.out_p)
    p = ease_out_expo(prog(s.t, float(el["t"]), 0.7))
    if a <= 0.01 or p <= 0.001:
        return None
    lt = s.t - float(el["t"])
    ang = float(el.get("angle", -8))
    nf = int(el.get("frames", 7))
    fw = w / nf
    sp_h = h * 0.13
    off = (lt * float(el.get("speed", 220))) % fw
    from .typo import pal_of
    pal = pal_of(el, s)
    ctx.save()
    ctx.translate(x - (1 - p) * 900, y); ctx.rotate(math.radians(ang))
    ctx.rectangle(-w / 2, -h / 2, w, h); _rgba(ctx, film, a); ctx.fill()
    ctx.save(); ctx.rectangle(-w / 2, -h / 2, w, h); ctx.clip()
    for j in range(-1, nf + 2):
        fx0 = -w / 2 + j * fw - off
        fc = (el.get("frame_colors") or ["@accent", "#2a2a30", "#3a3a42", "#1d1d22"])[j % 4]
        fc = _color(pal.get(fc[1:], "#333333") if isinstance(fc, str) and fc.startswith("@") else fc)
        g = cairo.LinearGradient(fx0, -h / 2, fx0 + fw, h / 2)
        g.add_color_stop_rgba(0, fc[0], fc[1], fc[2], 0.9 * a)
        g.add_color_stop_rgba(1, fc[0] * 0.55, fc[1] * 0.55, fc[2] * 0.55, 0.9 * a)
        ctx.rectangle(fx0 + 10, -h / 2 + sp_h + 8, fw - 20, h - 2 * sp_h - 16); ctx.set_source(g); ctx.fill()
    hole_w, hole_h, nh = fw / 8 * 0.55, sp_h * 0.55, 8
    hc = _color(el.get("hole_color", "#e8e8ea"))
    for side in (-1, 1):
        yy = side * (h / 2 - sp_h / 2) - hole_h / 2
        for j in range(-1, nf * nh + nh):
            hx = -w / 2 + j * (fw / nh) - off
            _rrect(ctx, hx, yy, hole_w, hole_h, 3)
        _rgba(ctx, hc, a * 0.9); ctx.fill()
    ctx.restore(); ctx.restore()
    dyb = abs(math.sin(math.radians(ang))) * w / 2
    bb = (x - w / 2, y - h / 2 - dyb, x + w / 2, y + h / 2 + dyb)
    if el.get("word"):
        size = float(el.get("size", 220))
        ws = spring(s.t, float(el["t"]) + float(el.get("word_delay", 0.25)), stiffness=320, damping=16)
        if ws > 0.01:
            ctx.save(); _xform(ctx, x, y, 0.5 + 0.5 * ws)
            st = el.get("text_style", {"kind": "extrude_3d", "depth": size * 0.1, "extrude": "#000000"})
            wc = _color(el["word_color"]) if el.get("word_color") else _on("#ffffff", film)   # film is dark
            draw_text(ctx, el["word"], x, y, size=size, color=wc,
                      alpha=a * clamp(ws * 2), weight=F(el, "title"), **text_fx(dict(el, text_style=st), s, size))
            ctx.restore()
    return bb


# ---------------------------------------------------------------------------
# 7. sine_wave
# ---------------------------------------------------------------------------
@element("sine_wave", dur=0.9, enter="none", sfx="none")
def sine_wave(ctx, el, s):
    x, y = xy(el, s.t)
    w, amp = float(el.get("w", 900)), float(el.get("amp", 34))
    wl, ps = float(el.get("wavelength", 180)), float(el.get("phase_speed", 3.0))
    col = _c(el, "color", s, "accent", "#ff1e42")
    a = s.alpha * (1 - s.out_p)
    draw = get_ease(el.get("ease", "in_out"))(prog(s.t, float(el["t"]), float(el.get("dur", 0.9))))
    if a <= 0.01 or draw <= 0.001:
        return None
    rtl = el.get("dir", "rtl") == "rtl"
    n = int(w / 6)
    m = max(2, int(n * draw))
    vo = (0.5 + 0.8 * s.scene.vo_env(s.t)) if (el.get("vo") and s.scene) else 1.0
    ctx.save(); ctx.new_path()
    for i in range(m + 1):
        u = i / n
        xx = (x + w / 2 - u * w) if rtl else (x - w / 2 + u * w)
        env = (math.sin(math.pi * u) ** 0.3 if el.get("taper", True) else 1.0) * vo
        (ctx.move_to if i == 0 else ctx.line_to)(xx, y + amp * env * math.sin(2 * math.pi * u * w / wl - ps * s.t))
    ctx.set_line_cap(cairo.LINE_CAP_ROUND); ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    _glow_stroke(ctx, col, a, float(el.get("width", 14)), glow=el.get("glow", True))
    ctx.restore()
    return (x - w / 2, y - amp * 1.4 - 10, x + w / 2, y + amp * 1.4 + 10)
