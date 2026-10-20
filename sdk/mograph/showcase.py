"""
mograph.showcase — v11 editor-showcase / social-reel layout widgets (refs 02–06, 08).

CONTAINER
  device_frame  rounded "phone" card that CONTAINS other elements:  "children": [ …any elements… ]
                Children are authored in a virtual screen vw×vh (default = scene size: author them exactly like a
                full-screen reel) and are scaled into the frame (w×h) and clipped to its rounded corners.
                keys: w h radius border border_w fill shadow(true|dict) label label_pos(top|bottom) label_size
                      label_color vw vh tilt glass  + normal enter/exit/tracks/follow.

BACKGROUND
  backdrop      full-frame procedural background (texture cached, ~1 ms/frame):
                kind: paper (teal crumpled paper, ref02-04) | tiles (dark diamond tiles, ref05) | smoke (ref06) |
                      streaks (blue light lines, ref06) | studio (light sweep, ref08) | dots | gradient
                color color2 seed drift alpha

UI PIECES (part 2, below)
  badge_button, handle_tag, social_outro, nle_timeline, image_card, radial_formula, redact_text
"""
from __future__ import annotations

import math
import os

import cairo
import cv2
import numpy as np

from .core import clamp, color as _color, ease_in_out, ease_out, ease_out_back, lerp, prog, spring
from .elements import element, xy, val
from .geometry import compile_path, rounded_rect_d
from .text import draw_text, resolve_font, text_bbox


def _rr(ctx, x0, y0, w, h, r):
    ctx.new_path()
    ctx.append_path(compile_path(rounded_rect_d(x0, y0, w, h, r)).whole)


def shape_path(ctx, shape, x, y, w, h, radius):
    if shape == "circle":
        ctx.new_path(); ctx.save(); ctx.translate(x, y); ctx.scale(w / 2, h / 2)
        ctx.arc(0, 0, 1, 0, 2 * math.pi); ctx.restore()
    elif shape in ("hex", "octagon"):
        n = 6 if shape == "hex" else 8
        off = math.pi / n if shape == "octagon" else 0
        ctx.new_path()
        for i in range(n):
            a = off + 2 * math.pi * i / n
            (ctx.move_to if i == 0 else ctx.line_to)(x + w / 2 * math.cos(a), y + h / 2 * math.sin(a))
        ctx.close_path()
    elif shape == "pill":
        _rr(ctx, x - w / 2, y - h / 2, w, h, min(w, h) / 2)
    elif shape == "rect":
        ctx.new_path(); ctx.rectangle(x - w / 2, y - h / 2, w, h)
    else:
        _rr(ctx, x - w / 2, y - h / 2, w, h, radius)


def soft_shadow(ctx, x, y, w, h, r, sh, a):
    from .fx import _soft_shadow
    sh = sh if isinstance(sh, dict) else {}
    _soft_shadow(ctx, x, y, w, h, r, float(sh.get("blur", 34)), float(sh.get("alpha", 0.4)) * a,
                 float(sh.get("dy", 20)))


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------
def expand_children(els):
    """Scene hook: flatten `children` of containers into the chapter list (child gets parent=<id>)."""
    out = []
    for el in els:
        if el.get("type") == "social_outro" and "layer" not in el:
            el = dict(el, layer=1000)            # end card always on top of everything
        out.append(el)
        kids = el.get("children")
        if not kids:
            continue
        pid = el.get("id")
        base_layer = float(el.get("layer", 0))
        for j, ch in enumerate(kids):
            ch = dict(ch)
            ch["parent"] = pid
            ch.setdefault("id", f"{pid}_k{j}")
            ch.setdefault("layer", base_layer)          # stable sort keeps children right after the frame
            if "t" not in ch and "t" in el:
                ch["t"] = el["t"]
            out.extend(expand_children([ch]))
        el["children"] = None
        el["_has_children"] = True
    return out


def apply_parent(ctx, xf):
    """xf = (cx, cy, k, tilt_deg, x0, y0, w, h, r, vw, vh, alpha) written by the parent this frame."""
    cx, cy, k, tilt, x0, y0, w, h, r, vw, vh, a = xf
    ctx.translate(cx, cy)
    if tilt:
        ctx.rotate(math.radians(tilt))
    ctx.translate(-cx, -cy)
    _rr(ctx, x0, y0, w, h, r)
    ctx.clip()
    ctx.translate(x0, y0)
    ctx.scale(k, k)


@element("device_frame", dur=0.5, enter="pop", sfx="pop")
def el_device_frame(ctx, el, s):
    sc = s.scene
    x, y = xy(el, s.t)
    w = float(val(el, "w", s.t, 520)); h = float(val(el, "h", s.t, w * 16 / 9))
    r = float(el.get("radius", w * 0.07))
    vw = float(el.get("vw", sc.W if sc else 1080)); vh = float(el.get("vh", vw * h / w))
    tilt = float(val(el, "tilt", s.t, 0.0))
    a = s.alpha * (1 - s.out_p) if s.exit == "fade" else s.alpha
    W, H, R = w * s.scale, h * s.scale, r * s.scale
    yy = y + s.dy
    x0, y0 = x - W / 2, yy - H / 2
    ctx.save()
    ctx.translate(x, yy)
    if tilt:
        ctx.rotate(math.radians(tilt))
    ctx.translate(-x, -yy)
    if el.get("shadow", True):
        soft_shadow(ctx, x, yy, W, H, R, el.get("shadow"), a)
    fill = el.get("fill", "#000000")
    if fill:
        _rr(ctx, x0, y0, W, H, R)
        ctx.set_source_rgba(*_color(fill, a)); ctx.fill()
    ctx.restore()
    el["_xf"] = (x, yy, W / vw, tilt, x0, y0, W, H, R, vw, vh, a) if a > 0.003 else None
    lab = el.get("label")
    if lab:
        ls = float(el.get("label_size", 34))
        ly = (y0 - ls * 0.9) if el.get("label_pos", "top") == "top" else (y0 + H + ls * 0.9)
        draw_text(ctx, lab, x, ly, size=ls, color=el.get("label_color", "#ffffff"), alpha=a,
                  weight=resolve_font(el.get("label_font", "title"), None, "title"))
    return (x0, y0, x0 + W, y0 + H)


def draw_frame_border(ctx, el):
    """Renderer calls this AFTER the children: border + glass sheen on top of the content."""
    xf = el.get("_xf")
    if not xf:
        return
    cx, cy, k, tilt, x0, y0, w, h, r, vw, vh, a = xf
    ctx.save()
    ctx.translate(cx, cy)
    if tilt:
        ctx.rotate(math.radians(tilt))
    ctx.translate(-cx, -cy)
    ga = float(el.get("glass", 0.07) or 0)
    if ga:
        g = cairo.LinearGradient(x0, y0, x0 + w, y0 + h)
        g.add_color_stop_rgba(0, 1, 1, 1, ga * a); g.add_color_stop_rgba(0.45, 1, 1, 1, 0)
        _rr(ctx, x0, y0, w, h, r); ctx.set_source(g); ctx.fill()
    b = el.get("border", "#ececec")
    if b:
        _rr(ctx, x0, y0, w, h, r)
        ctx.set_source_rgba(*_color(b, a)); ctx.set_line_width(float(el.get("border_w", 5))); ctx.stroke()
    ctx.restore()


# ---------------------------------------------------------------------------
# backdrop
# ---------------------------------------------------------------------------
_TEX = {}


def _bgr(c):
    r, g, b, _ = _color(c)
    return np.array([b * 255, g * 255, r * 255], np.float32)


def make_texture(kind, W, H, c1, c2, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == "paper":
        base = np.zeros((H, W, 3), np.float32) + _bgr(c1)
        low = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (H // 4 + 1, W // 4 + 1)).astype(np.float32), (0, 0), 6),
                         (W, H)) * 9
        cr = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (H // 16 + 1, W // 16 + 1)).astype(np.float32), (0, 0), 1.2),
                        (W, H), interpolation=cv2.INTER_CUBIC)
        cr = np.clip(cv2.Sobel(cr, cv2.CV_32F, 1, 0, ksize=3) * 40, -10, 10)
        fib = np.zeros((H, W), np.float32)
        for _ in range(int(W * H / 2500)):
            x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
            ang, L = rng.uniform(0, math.pi), rng.uniform(8, 40)
            cv2.line(fib, (x0, y0), (int(x0 + L * math.cos(ang)), int(y0 + L * math.sin(ang))),
                     float(rng.uniform(4, 12)), 1, cv2.LINE_AA)
        tex = base + (low + cr + fib * 0.8 + rng.normal(0, 3.0, (H, W)).astype(np.float32))[..., None]
    elif kind == "tiles":
        step = max(60, W // 6)
        u, v = (xx + yy) / step, (xx - yy) / step
        fu, fv = u - np.floor(u), v - np.floor(v)
        shade = (np.floor(u) + np.floor(v)) % 3
        facet = np.where(fu + fv < 1, 6, -6) * (shade - 1) + np.where(fu > fv, 4, -4)
        edge = np.minimum(np.minimum(fu, 1 - fu), np.minimum(fv, 1 - fv))
        line = np.clip(1 - edge * step / 2.0, 0, 1) * 14
        tex = _bgr(c1) + (facet + line)[..., None] + rng.normal(0, 1.5, (H, W, 1)).astype(np.float32)
        tex *= (1 - 0.45 * np.clip(np.hypot((xx - W / 2) / W, (yy - H / 2) / H) * 1.6, 0, 1) ** 2)[..., None]
    elif kind == "smoke":
        tex = np.zeros((H, W, 3), np.float32) + _bgr(c1)
        for k in range(7):
            blob = np.zeros((H // 8, W // 8), np.float32)
            cv2.circle(blob, (int(rng.uniform(0, W // 8)), int(rng.uniform(0, H // 8))),
                       int(rng.uniform(W / 40, W / 14)), 1.0, -1)
            blob = cv2.resize(cv2.GaussianBlur(blob, (0, 0), W / 60), (W, H))
            col = _bgr(c2) if k % 2 else _bgr(c2) * 0.6 + _bgr("#3a1d6e") * 0.4
            tex += blob[..., None] * col * 0.35
        tex += rng.normal(0, 2.0, (H, W, 1)).astype(np.float32)
    elif kind == "studio":
        k = np.clip(np.hypot((xx - W * 0.5) / W, (yy - H * 0.42) / H) * 1.5, 0, 1)[..., None] ** 1.4
        tex = _bgr(c1) * (1 - k) + _bgr(c2) * k + rng.normal(0, 1.2, (H, W, 1)).astype(np.float32)
    elif kind == "dots":
        tex = np.zeros((H, W, 3), np.float32) + _bgr(c1)
        step = max(24, W // 30)
        col = tuple(float(v) for v in _bgr(c2))
        for y_ in range(step // 2, H, step):
            for x_ in range(step // 2, W, step):
                cv2.circle(tex, (x_, y_), max(1, step // 14), col, -1, cv2.LINE_AA)
    else:  # gradient
        t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        tex = _bgr(c1) * (1 - t) + _bgr(c2) * t + np.zeros((H, W, 3), np.float32)
    return np.clip(tex, 0, 255).astype(np.uint8)


BACKDROP_DEFAULTS = {"paper": ("#9db8b8", "#7fa0a0"), "tiles": ("#2b2b2d", "#1c1c1e"), "smoke": ("#0d0b12", "#5b2a8a"),
                     "streaks": ("#05060c", "#1a4dff"), "studio": ("#f4f4f2", "#d9d9d6"), "dots": ("#0b0b0d", "#2a2a30"),
                     "gradient": ("#141418", "#050506")}


@element("backdrop", dur=0.4, enter="fade", sfx="none")
def el_backdrop(ctx, el, s):
    sc = s.scene
    W, H = (sc.W, sc.H) if sc else (1080, 1920)
    kind = el.get("kind", "paper")
    d = BACKDROP_DEFAULTS.get(kind, ("#202020", "#000000"))
    c1, c2 = el.get("color", d[0]), el.get("color2", d[1])
    a = s.alpha * (1 - s.out_p) * float(el.get("alpha", 1.0))
    if kind == "streaks":
        ctx.save()
        ctx.set_source_rgba(*_color(c1, a)); ctx.rectangle(-W, -H, 3 * W, 3 * H); ctx.fill()
        c = _color(c2)
        rng = np.random.default_rng(int(el.get("seed", 3)))
        ctx.set_operator(cairo.OPERATOR_ADD)
        for i in range(int(el.get("n", 9))):
            y0 = rng.uniform(-0.2, 1.2) * H; amp = rng.uniform(40, 160); ph = rng.uniform(0, 6)
            sp = rng.uniform(0.15, 0.45); br = rng.uniform(0.6, 1.0)
            for wmul, am in ((14, 0.03), (6, 0.07), (2, 0.25)):
                ctx.new_path()
                for k in range(41):
                    x_ = -0.1 * W + k / 40 * 1.2 * W
                    y_ = y0 + amp * math.sin(k / 40 * 3.0 + ph + s.t * sp) - k / 40 * H * 0.25
                    (ctx.move_to if k == 0 else ctx.line_to)(x_, y_)
                ctx.set_source_rgba(c[0], c[1], c[2], a * am * br)
                ctx.set_line_width(wmul); ctx.stroke()
        ctx.restore()
        return None
    scale = 0.5
    key = (kind, W, H, str(c1), str(c2), int(el.get("seed", 3)))
    if key not in _TEX:
        if len(_TEX) > 6:
            _TEX.clear()
        from .media import to_surface
        _TEX[key] = to_surface(make_texture(kind, int(W * scale), int(H * scale), c1, c2, int(el.get("seed", 3))))
    surf = _TEX[key][0]
    drift = float(el.get("drift", 0.0))
    ctx.save()
    ctx.translate(math.sin(s.t * 0.13) * drift - drift, math.cos(s.t * 0.11) * drift - drift)
    k = (1 + 2 * drift / W) / scale
    ctx.scale(k, k)
    ctx.set_source_surface(surf, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_BILINEAR)
    ctx.paint_with_alpha(a)
    ctx.restore()
    return None


# ---------------------------------------------------------------------------
# UI pieces
# ---------------------------------------------------------------------------
@element("badge_button", dur=0.45, enter="pop", sfx="pop")
def el_badge_button(ctx, el, s):
    """3D pill button with a bottom lip ("ادیت ویدیو"). press_at = seconds after t → button presses down."""
    x, y = xy(el, s.t)
    size = float(el.get("size", 44))
    txt = el.get("text", "ادیت ویدیو")
    fnt = resolve_font(el.get("font", "title"), None, "title")
    tb = text_bbox(txt, 0, 0, size, fnt)
    w = float(el.get("w", tb[2] - tb[0] + size * 1.6)); h = float(el.get("h", size * 1.9))
    r = float(el.get("radius", h * 0.32)); lip = float(el.get("lip", h * 0.12))
    press = 0.0
    if el.get("press_at") is not None:
        d = s.t - (float(el["t"]) + float(el["press_at"]))
        press = math.exp(-((d - 0.06) / 0.07) ** 2) if -0.1 < d < 0.4 else 0.0
    a = s.alpha * (1 - s.out_p)
    ctx.save()
    ctx.translate(x, y + s.dy); ctx.scale(s.scale, s.scale); ctx.translate(-x, -y)
    soft_shadow(ctx, x, y + lip, w, h, r, {"blur": 18, "alpha": 0.35, "dy": 8}, a)
    _rr(ctx, x - w / 2, y - h / 2 + lip, w, h, r)
    ctx.set_source_rgba(*_color(el.get("edge", "#b9b3a5"), a)); ctx.fill()
    oy = lip * press
    _rr(ctx, x - w / 2, y - h / 2 + oy, w, h, r)
    fc = _color(el.get("fill", "#f4efe4"))
    g = cairo.LinearGradient(0, y - h / 2, 0, y + h / 2)
    g.add_color_stop_rgba(0, min(1, fc[0] * 1.04), min(1, fc[1] * 1.04), min(1, fc[2] * 1.04), a)
    g.add_color_stop_rgba(1, fc[0] * 0.93, fc[1] * 0.93, fc[2] * 0.93, a)
    ctx.set_source(g); ctx.fill_preserve()
    ctx.set_source_rgba(1, 1, 1, a * 0.7); ctx.set_line_width(2); ctx.stroke()
    draw_text(ctx, txt, x, y + oy, size=size, color=el.get("text_color", "#1d1d1f"), alpha=a, weight=fnt)
    ctx.restore()
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2 + lip)


def draw_ig_glyph(ctx, x, y, s, color=None, alpha=1.0, gradient=False, width=None):
    """Instagram-style glyph: rounded square + lens + dot. gradient=True → brand-gradient tile + white glyph."""
    lw = width or s * 0.085
    s2 = s
    if gradient:
        g = cairo.LinearGradient(x - s / 2, y + s / 2, x + s / 2, y - s / 2)
        for off, c in ((0, "#feda75"), (0.3, "#fa7e1e"), (0.55, "#d62976"), (0.8, "#962fbf"), (1, "#4f5bd5")):
            cc = _color(c); g.add_color_stop_rgba(off, cc[0], cc[1], cc[2], alpha)
        _rr(ctx, x - s / 2, y - s / 2, s, s, s * 0.28); ctx.set_source(g); ctx.fill()
        color, s2 = "#ffffff", s * 0.66
    c = _color(color or "#ffffff")
    ctx.set_source_rgba(c[0], c[1], c[2], c[3] * alpha)
    ctx.set_line_width(lw)
    _rr(ctx, x - s2 / 2, y - s2 / 2, s2, s2, s2 * 0.3); ctx.stroke()
    ctx.new_path(); ctx.arc(x, y, s2 * 0.22, 0, 2 * math.pi); ctx.stroke()
    ctx.new_path(); ctx.arc(x + s2 * 0.26, y - s2 * 0.26, lw * 0.75, 0, 2 * math.pi); ctx.fill()


@element("handle_tag", dur=0.4, enter="fade", sfx="none")
def el_handle_tag(ctx, el, s):
    x, y = xy(el, s.t)
    size = float(el.get("size", 26))
    txt = el.get("handle", "@HANDLE")
    fnt = resolve_font(el.get("font", "ui"), None, "ui")
    tb = text_bbox(txt, 0, 0, size, fnt)
    tw = tb[2] - tb[0]
    gs = size * 1.25 if el.get("icon", True) else 0
    total = gs + (size * 0.4 if gs else 0) + tw
    x0 = {"left": x, "right": x - total}.get(el.get("align", "center"), x - total / 2)
    col = el.get("color", "#ffffff")
    a = s.alpha * (1 - s.out_p)
    if gs:
        draw_ig_glyph(ctx, x0 + gs / 2, y, gs, col, a, gradient=bool(el.get("gradient")))
    draw_text(ctx, txt, x0 + total - tw / 2, y, size=size, color=col, alpha=a, weight=fnt)
    return (x0, y - max(gs, size) / 2, x0 + total, y + max(gs, size) / 2)


@element("social_outro", dur=0.5, enter="none", sfx="pop")
def el_social_outro(ctx, el, s):
    """End card: dims everything drawn so far, gradient logo springs in, handle wipes in, logo turns white."""
    sc = s.scene
    W, H = (sc.W, sc.H) if sc else (1080, 1920)
    lt = s.lt
    x = float(el.get("x", W / 2)); y = float(el.get("y", H * 0.42))
    a = s.alpha * (1 - s.out_p)
    dim = float(el.get("dim", 0.62)) * ease_in_out(clamp(lt / 0.4)) * (1 - s.out_p)
    if dim > 0:
        ctx.set_source_rgba(*_color(el.get("dim_color", "#0a0f10"), dim))
        ctx.rectangle(-W, -H, 3 * W, 3 * H); ctx.fill()
    size = float(el.get("size", 120))
    p = spring(lt, 0.15, stiffness=220, damping=14) if lt > 0.15 else 0.0
    wp = ease_in_out(prog(lt, float(el.get("white_at", 1.2)), 0.35))
    if p > 0.01:
        ctx.save(); ctx.translate(x, y); ctx.scale(p, p); ctx.translate(-x, -y)
        if wp < 1:
            draw_ig_glyph(ctx, x, y, size, alpha=a * (1 - wp), gradient=True)
        if wp > 0:
            draw_ig_glyph(ctx, x, y, size * 0.8, "#ffffff", a * wp, width=size * 0.07)
        ctx.restore()
    hp = clamp((lt - 0.35) / 0.4)
    if hp > 0:
        draw_text(ctx, el.get("handle", "@HANDLE"), x, y + size * 0.85, size=float(el.get("handle_size", size * 0.22)),
                  color="#ffffff", alpha=a * hp, weight=resolve_font(el.get("font", "ui"), None, "ui"),
                  reveal=("wipe", hp))
    return None


@element("nle_timeline", dur=0.5, enter="rise", sfx="none")
def el_nle_timeline(ctx, el, s):
    """Editing timeline. mode clips (ref05): coloured clips slide under a fixed I-beam; mode segments (ref04):
    labelled segments + travelling red playhead. items: [{label, sub, color, t}] (t = chapter-local)."""
    x, y = xy(el, s.t)
    items = el.get("items") or []
    mode = el.get("mode", "clips")
    w = float(el.get("w", 1080)); h = float(el.get("h", 120))
    a = s.alpha * (1 - s.out_p)
    ls = float(el.get("label_size", 34)); ss_ = float(el.get("sub_size", 20))
    fnt = resolve_font(el.get("font", "title"), None, "title")
    sub_f = resolve_font(el.get("sub_font", "ui"), None, "ui")
    times = [float(it["t"]) if isinstance(it.get("t"), (int, float)) else float(el["t"]) + i
             for i, it in enumerate(items)]
    act = 0
    for i, ti in enumerate(times):
        if s.t >= ti:
            act = i
    ctx.save()
    if mode == "clips":
        cw = float(el.get("clip_w", 300)); gap = float(el.get("gap", 18)); step = cw + gap
        pos = float(act)
        if act > 0:
            pos = (act - 1) + ease_in_out(prog(s.t, times[act], float(el.get("slide_dur", 0.45))))
        ctx.set_source_rgba(*_color(el.get("bar_color", "#000000"), a * float(el.get("bar_alpha", 0.45))))
        ctx.rectangle(x - w / 2, y - h * 0.72, w, h * 1.44); ctx.fill()
        rtl = el.get("rtl", True)
        sgn = -1 if rtl else 1
        for i, it in enumerate(items):
            cx = x + (i - pos) * step * sgn - (cw / 2 + 14) * (1 if rtl else -1)
            if cx + cw / 2 < x - w / 2 - 10 or cx - cw / 2 > x + w / 2 + 10:
                continue
            c = _color(it.get("color", "#7f8ccf"))
            _rr(ctx, cx - cw / 2, y - h / 2, cw, h, 14)
            g = cairo.LinearGradient(0, y - h / 2, 0, y + h / 2)
            g.add_color_stop_rgba(0, min(1, c[0] * 1.08), min(1, c[1] * 1.08), min(1, c[2] * 1.08), a)
            g.add_color_stop_rgba(1, c[0] * 0.9, c[1] * 0.9, c[2] * 0.9, a)
            ctx.set_source(g); ctx.fill()
            tc = it.get("text_color", "#1c1c1e")
            draw_text(ctx, it.get("label", ""), cx, y - h * (0.14 if it.get("sub") else 0), size=ls, color=tc,
                      alpha=a, weight=fnt)
            if it.get("sub"):
                draw_text(ctx, it["sub"], cx, y + h * 0.27, size=ss_, color=tc, alpha=a * 0.75, weight=sub_f)
        pc = _color(el.get("playhead_color", "#ffffff"))
        ctx.set_source_rgba(pc[0], pc[1], pc[2], a); ctx.set_line_width(5)
        ctx.move_to(x, y - h * 0.62); ctx.line_to(x, y + h * 0.62); ctx.stroke()
        for yy in (y - h * 0.62, y + h * 0.62):
            ctx.move_to(x - 16, yy); ctx.line_to(x + 16, yy); ctx.stroke()
    else:
        n = max(1, len(items)); bh = float(el.get("bar_h", 34)); segw = w / n
        rtl = el.get("rtl", False)
        ctx.set_source_rgba(*_color(el.get("bar_color", "#2a2a2c"), a))
        ctx.rectangle(x - w / 2, y - bh / 2, w, bh); ctx.fill()
        for i, it in enumerate(items):
            sx0 = x - w / 2 + (n - 1 - i if rtl else i) * segw
            ctx.set_source_rgba(1, 1, 1, a * 0.55); ctx.set_line_width(2)
            ctx.move_to(sx0, y - bh / 2); ctx.line_to(sx0, y + bh / 2); ctx.stroke()
            on = i == act
            draw_text(ctx, it.get("label", ""), sx0 + segw / 2, y - bh / 2 - ls * 0.9, size=ls * (1.0 if on else 0.88),
                      color=it.get("text_color", "#ffffff"), alpha=a * (1 if on else 0.65), weight=sub_f,
                      max_width=segw * 0.95)
        t_end = times[act + 1] if act + 1 < len(times) else times[act] + float(el.get("last_dur", 3.0))
        q = clamp((s.t - times[act]) / max(1e-3, t_end - times[act]))
        ix = n - 1 - act if rtl else act
        px = x - w / 2 + (ix + (1 - q if rtl else q)) * segw
        pc = _color(el.get("playhead_color", "#c62828"))
        ctx.set_source_rgba(pc[0], pc[1], pc[2], a); ctx.set_line_width(5)
        ctx.move_to(px, y - bh); ctx.line_to(px, y + bh); ctx.stroke()
        for yy in (y - bh, y + bh):
            ctx.move_to(px - 12, yy); ctx.line_to(px + 12, yy); ctx.stroke()
    ctx.restore()
    return (x - w / 2, y - h, x + w / 2, y + h)


_IMGS = {}


def _img_bgr(path):
    if path not in _IMGS:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(path)
        _IMGS[path] = im
    return _IMGS[path]


@element("image_card", dur=0.5, enter="pop", sfx="pop")
def el_image_card(ctx, el, s):
    """Photo / clip card. shape rect|rounded|circle|hex|octagon|pill, kenburns, tilt, border, shadow, focus [u,v]."""
    sc = s.scene
    x, y = xy(el, s.t)
    w = float(val(el, "w", s.t, 360)); h = float(val(el, "h", s.t, w * 1.25))
    shape = el.get("shape", "rounded")
    r = float(el.get("radius", min(w, h) * 0.08))
    tilt = float(val(el, "tilt", s.t, 0.0))
    src = el.get("src", "")
    path = src if os.path.isabs(src) else os.path.join(sc.base_dir if sc else ".", src)
    if src.lower().endswith((".mp4", ".mov", ".webm")):
        from .media import reader
        fr = reader(path).frame(max(0.0, s.t - float(el["t"])) + float(el.get("clip_start", 0.0)), loop=True)
    elif src:
        fr = _img_bgr(path)
    else:
        fr = np.full((64, 64, 3), 128, np.uint8)
    if el.get("grade"):
        from .fx import apply_grade, grade_params
        fr = apply_grade(fr, grade_params(el["grade"]))
    fh, fw = fr.shape[:2]
    a = s.alpha * (1 - s.out_p)
    ctx.save()
    ctx.translate(x, y + s.dy)
    if tilt:
        ctx.rotate(math.radians(tilt))
    if s.scale != 1:
        ctx.scale(s.scale, s.scale)
    ctx.translate(-x, -y)
    if el.get("shadow", True):
        soft_shadow(ctx, x, y, w, h, min(w, h) / 2 if shape == "circle" else r, el.get("shadow"), a)
    ctx.save()
    shape_path(ctx, shape, x, y, w, h, r)
    ctx.clip()
    zk = 1 + float(el.get("kenburns", 0.06)) * clamp((s.t - float(el["t"])) / float(el.get("kb_dur", 6.0)))
    k = (max(w / fw, h / fh) if el.get("fit", "cover") == "cover" else min(w / fw, h / fh)) * zk
    from .media import to_surface
    surf, _keep = to_surface(fr)
    fu, fv = el.get("focus", [0.5, 0.5])
    ctx.translate(x, y); ctx.scale(k, k); ctx.translate(-fu * fw, -fv * fh)
    ctx.set_source_surface(surf, 0, 0); ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint_with_alpha(a)
    ctx.restore()
    if el.get("border"):
        shape_path(ctx, shape, x, y, w, h, r)
        ctx.set_source_rgba(*_color(el["border"] if isinstance(el["border"], str) else "#ffffff", a))
        ctx.set_line_width(float(el.get("border_w", 6))); ctx.stroke()
    ctx.restore()
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


@element("redact_text", dur=0.3, enter="fade", sfx="none")
def el_redact_text(ctx, el, s):
    """Text hidden under a censor bar that retracts at reveal_at (s after t)."""
    x, y = xy(el, s.t)
    size = float(el.get("size", 60))
    fnt = resolve_font(el.get("font", "title"), None, "title")
    txt = el.get("text", "")
    tb = text_bbox(txt, x, y, size, fnt)
    pad = float(el.get("pad", size * 0.15))
    a = s.alpha * (1 - s.out_p)
    p = ease_in_out(prog(s.t, float(el["t"]) + float(el.get("reveal_at", 0.6)), float(el.get("reveal_dur", 0.35))))
    draw_text(ctx, txt, x, y, size=size, color=el.get("color", "#111111"), alpha=a * clamp(p * 3), weight=fnt)
    if p < 1:
        x0, y0, x1, y1 = tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad
        bw = (x1 - x0) * (1 - p)
        ctx.set_source_rgba(*_color(el.get("bar_color", "#0b0b0b"), a))
        ctx.rectangle(x0 if el.get("dir", "rtl") == "rtl" else x1 - bw, y0, bw, y1 - y0); ctx.fill()
    return tb


@element("radial_formula", dur=0.8, enter="none", sfx="pop")
def el_radial_formula(ctx, el, s):
    """ref08 hub-and-nodes formula. nodes [{label, icon, t}] on a circle around an octagon hub; when a node's
    time comes the whole wheel rotates so it sits on top, a progress arc grows and the hub shows
    "مرحله‌ی X از N".  keys: r hub_r node_r rotate accent ink hub_title hub_sub step_fmt label_size show_labels"""
    from .icons import draw_icon
    from .text import to_fa_digits
    x, y = xy(el, s.t)
    nodes = el.get("nodes") or []
    n = max(1, len(nodes))
    R = float(el.get("r", 420)); hub = float(el.get("hub_r", 230))
    acc = el.get("accent", "#ff4b2b"); ink = el.get("ink", "#1b1b1d")
    a = s.alpha * (1 - s.out_p)
    times = [float(nd["t"]) if isinstance(nd.get("t"), (int, float)) else float(el["t"]) + 1 + i
             for i, nd in enumerate(nodes)]
    act = -1
    for i, ti in enumerate(times):
        if s.t >= ti:
            act = i
    step = 2 * math.pi / n
    rot = 0.0
    if el.get("rotate", True) and act >= 0:
        q = ease_in_out(prog(s.t, times[act], float(el.get("rot_dur", 0.7)))) if act > 0 else 1.0
        rot = -((act - 1) + q) * step if act > 0 else 0.0
    grow = ease_out_back(clamp(s.lt / 0.6))
    ctx.save()
    ctx.translate(x, y + s.dy); ctx.scale(s.scale * grow, s.scale * grow); ctx.translate(-x, -y)
    for rr, al in ((R * 1.18, 0.14), (R * 0.8, 0.2)):
        ctx.set_source_rgba(*_color(ink, a * al)); ctx.set_line_width(3)
        ctx.new_path(); ctx.arc(x, y, rr, 0, 2 * math.pi); ctx.stroke()
    if act >= 0:
        pn = act + (ease_in_out(prog(s.t, times[act], 0.7)) if act > 0 else 1.0)
        ctx.set_source_rgba(*_color(acc, a)); ctx.set_line_width(9); ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        a0 = -math.pi / 2 + rot - step * 0.4
        ctx.new_path(); ctx.arc(x, y, R * 0.8, a0, a0 + step * min(n, pn) * 0.999); ctx.stroke()
    lfont = resolve_font(el.get("font", "title"), None, "title")
    for i, nd in enumerate(nodes):
        ang = -math.pi / 2 + i * step + rot
        ca, sa = math.cos(ang), math.sin(ang)
        nx, ny = x + R * ca, y + R * sa
        on, done = i == act, i < act
        ctx.set_source_rgba(*_color(ink, a * 0.55)); ctx.set_line_width(3)
        ctx.new_path(); ctx.move_to(x + hub * 1.1 * ca, y + hub * 1.1 * sa); ctx.line_to(nx, ny); ctx.stroke()
        nr = float(el.get("node_r", 40)) * (1.18 if on else 1.0)
        if on:
            pul = ((s.t - times[i]) * 1.3) % 1.0
            ctx.set_source_rgba(*_color(acc, a * (1 - pul) * 0.5)); ctx.set_line_width(4)
            ctx.new_path(); ctx.arc(nx, ny, nr * (1.2 + pul * 0.8), 0, 2 * math.pi); ctx.stroke()
        ctx.new_path(); ctx.arc(nx, ny, nr, 0, 2 * math.pi)
        ctx.set_source_rgba(*_color(acc if (on or done) else ink, a)); ctx.fill()
        if nd.get("icon"):
            draw_icon(ctx, nd["icon"], nx, ny, nr * 1.25, 1.0, color="#ffffff", alpha=a, width=3.2)
        if nd.get("label") and (el.get("show_labels", "active") == "all" or on):
            lp = clamp((s.t - times[i]) / 0.35) if on else 1.0
            off = R + nr + float(el.get("label_gap", 60))
            draw_text(ctx, nd["label"], x + off * ca, y + off * sa, size=float(el.get("label_size", 52)), color=ink,
                      alpha=a * lp, weight=lfont)
    for gr, col in ((1.12, "#ffffff"), (1.0, ink)):
        ctx.new_path()
        for k in range(8):
            ang = math.pi / 8 + k * math.pi / 4
            (ctx.move_to if k == 0 else ctx.line_to)(x + hub * gr * math.cos(ang), y + hub * gr * math.sin(ang))
        ctx.close_path()
        if gr > 1:
            ctx.set_source_rgba(0, 0, 0, a * 0.16); ctx.set_line_width(16); ctx.stroke_preserve()
        cc = _color(col)
        g = cairo.LinearGradient(x, y - hub, x, y + hub)
        g.add_color_stop_rgba(0, min(1, cc[0] + 0.12), min(1, cc[1] + 0.12), min(1, cc[2] + 0.12), a)
        g.add_color_stop_rgba(1, cc[0], cc[1], cc[2], a)
        ctx.set_source(g); ctx.fill()
    if el.get("hub_sub"):
        draw_text(ctx, el["hub_sub"], x, y - hub * 0.3, size=hub * 0.15, color="#d0d0d0", alpha=a,
                  weight=resolve_font("body", None, "body"))
    draw_text(ctx, el.get("hub_title", "فرمول"), x, y, size=hub * 0.22, color="#ffffff", alpha=a, weight=lfont)
    fmt = el.get("step_fmt", "مرحله‌ی {i} از {n}")
    if act >= 0 and fmt:
        draw_text(ctx, to_fa_digits(fmt.format(i=act + 1, n=n)), x, y + hub * 0.34, size=hub * 0.1,
                  color="#cfcfcf", alpha=a, weight=resolve_font("body", None, "body"))
    ctx.restore()
    RR = (R + 160) * s.scale
    return (x - RR, y - RR, x + RR, y + RR)
