"""
mograph.glowfx — v11 light & energy effects + tracked annotation.

GENERIC KEYS (any element, applied by render.draw_world):
  "glow":   true | "#2ad1ff" | {"color": "#2ad1ff", "radius": 18, "strength": 1.4, "pulse": 0}   neon halo of the
            element's own pixels (coloured, additive) — glowing strokes / neon text (ref06, ref09)
  "shadow": true | {"blur": 18, "alpha": 0.45, "dx": 0, "dy": 10, "color": "#000"}   soft drop shadow (photo cards,
            stickers, Hormozi boxes)
  "shiver": 4 | {"amp": 4, "freq": 18, "t": .., "dur": ..}   high-frequency stress jitter (ref09 shaking icons)
  "follow": "<footage>.<tracker>"   (see fx.py)  glue any element to a tracked point.

SCENE / CHAPTER KEYS (render_frame):
  "shake": [{"t": "w:بوم", "dur": 0.35, "amp": 18, "freq": 22}]   camera shake (impact)
  "beat_pump": 0.02   whole-frame scale bump on every beat (spec.beats)
  "flashes": [{"t": 3.2, "dur": 0.2, "color": "#fff", "amount": 0.8}]   full-frame exposure flash

ELEMENTS:
  saber      neon light-saber stroke along a path / points / around a target box; trim-animated head, white core,
             coloured haze, optional travelling comet.  keys: points | d | target(id) + pad, color, core, width,
             haze, dur, travel (true → comet segment runs along), loop, flicker
  flash_pulse  flash / hit frame on a point: expanding ring(s) + star burst + optional full-frame white
             keys: x y | follow, color, r, rings, rays, screen (0..1 full-frame flash)
  glow_ring  pulsing concentric rings / halo around a point or a tracked point (product highlight, "chashmak")
             keys: r, color, rings, period, width, follow
  tracked_callout  leader line that stays attached to a moving point (tracker / element id) with a label box that
             keeps its own screen spot (or lags behind).  keys: anchor ([x,y] | id | "clip.tracker"), box [x,y],
             text, sub, box_style (pill|card|plain), color, accent, size, elbow ("h"|"v"|"none"), dot, lag
  light_rays divine rays / god-rays behind something: rotating soft beams.  keys: r, n, color, spin, alpha, width
  orbit_trail  a glowing dot travelling on a circle/ellipse with a fading trail (ref08 open-loop).  keys: r, ry,
             period, start_deg, trail (deg), color, size, ring (draw the faint track), dir (1/-1)
  emitter    simple particle system: hearts / dots / bubbles / icons falling, rising, bursting radially.
             keys: kind (fall|rise|burst|radial), shape (dot|heart|square|icon:<name>|text:<str>), n, rate, life,
             speed, spread, gravity, size, color(s), seed, area [w,h]
"""
from __future__ import annotations

import math

import cairo
import cv2
import numpy as np

from .core import clamp, color as _color, ease_in_out, ease_out, ease_out_expo, prog
from .elements import element, resolve_point, xy, val
from .text import draw_text, text_bbox


# ---------------------------------------------------------------------------
# generic offscreen layer effects (glow / shadow)
# ---------------------------------------------------------------------------
def _norm_glow(g):
    if g is True:
        return {"color": None}
    if isinstance(g, str):
        return {"color": g}
    return dict(g)


def _norm_shadow(sh):
    return {} if sh is True else dict(sh)


def layer_fx(ctx, draw_fn, glow=None, shadow=None, t=0.0, scale_hint=1.0):
    """Draw draw_fn(ctx2) into an offscreen layer (same transform), then composite shadow → glow → content.
    Blurs run on a cropped, 1/2-res ROI — cheap enough for every frame."""
    tgt = ctx.get_target()
    W, H = tgt.get_width(), tgt.get_height()
    layer = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    lc = cairo.Context(layer)
    lc.set_matrix(ctx.get_matrix())
    lc.set_antialias(ctx.get_antialias())
    res = draw_fn(lc)
    layer.flush()
    st = layer.get_stride()
    a = np.ndarray((H, st // 4, 4), np.uint8, buffer=layer.get_data())[:, :W]
    alpha = a[..., 3]
    ys, xs = np.nonzero(alpha > 2)
    if not len(ys):
        return res
    dev = abs(ctx.get_matrix()[0]) or 1.0                 # user → device scale (for radii in user px)
    ctx.save()
    ctx.identity_matrix()
    if shadow is not None:
        sh = _norm_shadow(shadow)
        blur = float(sh.get("blur", 18)) * dev
        pad = int(blur * 2.5) + 4
        dx, dy = float(sh.get("dx", 0)) * dev, float(sh.get("dy", 10)) * dev
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
        roi = alpha[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(0.6, blur / 4))
        m = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        m *= float(sh.get("alpha", 0.45))
        _paint_mask(ctx, m, x0 + dx, y0 + dy, sh.get("color", "#000000"), cairo.OPERATOR_OVER)
    ctx.set_source_surface(layer, 0, 0)
    ctx.paint()
    if glow is not None:
        g = _norm_glow(glow)
        rad = float(g.get("radius", 18)) * dev
        strength = float(g.get("strength", 1.3))
        if g.get("pulse"):
            strength *= 0.75 + 0.25 * math.sin(2 * math.pi * float(g["pulse"]) * t)
        pad = int(rad * 3) + 4
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
        roi = np.ascontiguousarray(a[y0:y1, x0:x1])
        sw, shh = max(1, (x1 - x0) // 3), max(1, (y1 - y0) // 3)
        small = cv2.resize(roi, (sw, shh), interpolation=cv2.INTER_AREA).astype(np.float32)
        b1 = cv2.GaussianBlur(small, (0, 0), max(0.6, rad / 6))
        b2 = cv2.GaussianBlur(small, (0, 0), max(0.6, rad / 2.2))
        bl = cv2.resize(b1 * 0.6 + b2 * 0.8, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        if g.get("color"):
            m = np.clip(bl[..., 3] / 255.0 * strength, 0, 1)
            _paint_mask(ctx, m, x0, y0, g["color"], cairo.OPERATOR_ADD)
        else:
            out = np.clip(bl * strength, 0, 255).astype(np.uint8)
            out[..., 3] = np.minimum(255, out[..., :3].max(axis=2))
            surf = cairo.ImageSurface.create_for_data(memoryview(np.ascontiguousarray(out)), cairo.FORMAT_ARGB32,
                                                      x1 - x0, y1 - y0)
            ctx.set_source_surface(surf, x0, y0)
            ctx.set_operator(cairo.OPERATOR_ADD)
            ctx.paint()
            ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.restore()
    return res


def _paint_mask(ctx, m, x0, y0, col, op):
    from .matte import premultiplied_surface
    c = _color(col)
    h, w = m.shape
    bgr = np.empty((h, w, 3), np.uint8)
    bgr[:] = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
    surf, _keep = premultiplied_surface(bgr, np.clip(m * c[3], 0, 1))
    ctx.save()
    ctx.set_source_surface(surf, x0, y0)
    ctx.set_operator(op)
    ctx.paint()
    ctx.restore()


def shiver_offset(el, t):
    sv = el.get("shiver")
    if not sv:
        return None
    sv = {"amp": float(sv)} if isinstance(sv, (int, float)) else sv
    if sv.get("t") is not None:
        t0 = float(sv["t"]); d = float(sv.get("dur", 1e9))
        if not (t0 <= t <= t0 + d):
            return None
    a, f = float(sv.get("amp", 4)), float(sv.get("freq", 18))
    seed = (hash(el.get("id", "")) % 97) * 0.37
    q = math.floor(t * f) + seed                      # stepped jitter (reads as "shaking", not floating)
    rnd = lambda k: math.sin(q * 12.9898 + k * 78.233) * 43758.5453 % 1.0
    return (a * (rnd(1) * 2 - 1), a * (rnd(2) * 2 - 1), a * 0.25 * (rnd(3) * 2 - 1))


# ---------------------------------------------------------------------------
# scene-level: shake, beat pump, flashes
# ---------------------------------------------------------------------------
def scene_shake(scene, t):
    dx = dy = 0.0
    for sh in scene.spec.get("_shakes", []):
        t0, d = sh["t"], float(sh.get("dur", 0.3))
        if t0 <= t < t0 + d:
            env = (1 - (t - t0) / d) ** 2
            f, a = float(sh.get("freq", 22)), float(sh.get("amp", 16))
            dx += a * env * math.sin(2 * math.pi * f * t + 1.3)
            dy += a * env * math.sin(2 * math.pi * f * 1.37 * t + 0.4)
    return dx, dy


def scene_pump(scene, t):
    amt = float(scene.spec.get("meta", {}).get("beat_pump", 0) or 0)
    if not amt or not scene.beats:
        return 1.0
    i = int(np.searchsorted(scene.beats, t)) - 1
    return 1.0 if i < 0 else 1 + amt * math.exp(-(t - scene.beats[i]) * 9.0)


def scene_flash(scene, t):
    best, col = 0.0, "#ffffff"
    for f in scene.spec.get("_flashes", []):
        t0, d = f["t"], float(f.get("dur", 0.2))
        if t0 - 0.03 <= t <= t0 + d:
            v = clamp((t - t0 + 0.03) / 0.03) if t < t0 else (1 - (t - t0) / d) ** 2
            v *= float(f.get("amount", 0.8))
            if v > best:
                best, col = v, f.get("color", "#ffffff")
    return best, col


def prepare_scene_fx(scene):
    """Resolve chapter/meta shake & flashes to absolute times (called once by Scene)."""
    sh, fl = [], []
    m = scene.spec.get("meta", {})
    for src, ct, ce in [(m, 0.0, None)] + [(c, c["t"], c["end"]) for c in scene.chapters]:
        for s in src.get("shake") or []:
            s = dict(s); s["t"] = scene.resolve_time(s["t"], ct, ce); sh.append(s)
        for f in src.get("flashes") or []:
            f = {"t": f} if isinstance(f, (int, float, str)) else dict(f)
            f["t"] = scene.resolve_time(f["t"], ct, ce); fl.append(f)
    scene.spec["_shakes"], scene.spec["_flashes"] = sh, fl


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _point(el, key, s, default=(540, 960)):
    """[x,y] | element id | 'clip.tracker' → chapter-local point."""
    p = el.get(key)
    if p is None:
        return default
    if isinstance(p, str) and "." in p.split("+")[0]:
        from .fx import resolve_follow
        base = p.split("+")[0]
        off = [float(v) for v in p.split("+")[1].split(",")] if "+" in p else [0, 0]
        r = resolve_follow(s.scene, base, s.t)
        if r is None:
            return default
        return r[0] + off[0], r[1] + off[1]
    try:
        return resolve_point(p, s.t, s.scene)
    except Exception:  # noqa: BLE001
        return default


def _glow_stroke(ctx, path_fn, color, width, core="#ffffff", haze=3.2, alpha=1.0, flick=1.0):
    """Cheap vector neon: 4 widening translucent strokes + white core (no blur needed)."""
    c = _color(color)
    for k, (wm, am) in enumerate(((haze * 2.2, 0.07), (haze * 1.4, 0.12), (haze, 0.2), (1.7, 0.55))):
        ctx.new_path(); path_fn()
        ctx.set_source_rgba(c[0], c[1], c[2], alpha * am * flick)
        ctx.set_line_width(width * wm)
        ctx.stroke()
    if core:
        cc = _color(core)
        ctx.new_path(); path_fn()
        ctx.set_source_rgba(cc[0], cc[1], cc[2], alpha * flick)
        ctx.set_line_width(width * 0.55)
        ctx.stroke()


def _poly_len(pts):
    return [0.0] + list(np.cumsum([math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]))


def _sub_poly(pts, cum, a, b):
    """Part of a polyline between arc lengths a..b."""
    out = []
    L = cum[-1]
    a, b = max(0, a), min(L, b)
    if b <= a:
        return out
    for i in range(len(pts) - 1):
        s0, s1 = cum[i], cum[i + 1]
        if s1 < a or s0 > b:
            continue
        seg = s1 - s0 or 1e-6
        p0 = max(a, s0); p1 = min(b, s1)
        f0, f1 = (p0 - s0) / seg, (p1 - s0) / seg
        P = lambda f: (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f)
        if not out:
            out.append(P(f0))
        out.append(P(f1))
    return out


def _target_rect(el, s, pad):
    from .fx import resolve_follow
    tg = el.get("target")
    if s.scene is not None and tg in s.scene.by_id:
        from .render import render_frame  # noqa: F401  (ensure module loaded)
        t_el = s.scene.by_id[tg]
        if t_el.get("_last_bb"):
            x0, y0, x1, y1 = t_el["_last_bb"]
            return x0 - pad, y0 - pad, x1 + pad, y1 + pad
        x, y = xy(t_el, s.t)
        w, h = float(t_el.get("w", 300)), float(t_el.get("h", 300))
        return x - w / 2 - pad, y - h / 2 - pad, x + w / 2 + pad, y + h / 2 + pad
    if isinstance(tg, str) and "." in tg:
        r = resolve_follow(s.scene, tg, s.t)
        if r and r[4]:
            q = np.array(r[4])
            return q[:, 0].min() - pad, q[:, 1].min() - pad, q[:, 0].max() + pad, q[:, 1].max() + pad
    return None


# ---------------------------------------------------------------------------
# elements
# ---------------------------------------------------------------------------
@element("saber", dur=0.6, enter="draw", sfx="whoosh")
def el_saber(ctx, el, s):
    pad = float(el.get("pad", 18))
    if el.get("points"):
        pts = [tuple(map(float, p)) if isinstance(p, (list, tuple)) else _point({"p": p}, "p", s) for p in el["points"]]
    elif el.get("target"):
        r = _target_rect(el, s, pad)
        if r is None:
            return None
        x0, y0, x1, y1 = r
        rr = float(el.get("radius", 26))
        pts = []
        for cx, cy, a0 in ((x1 - rr, y0 + rr, -90), (x1 - rr, y1 - rr, 0), (x0 + rr, y1 - rr, 90), (x0 + rr, y0 + rr, 180)):
            for k in range(7):
                a = math.radians(a0 + k * 15)
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        pts.append(pts[0])
    else:
        x, y = xy(el, s.t)
        r = float(el.get("r", 160))
        pts = [(x + r * math.cos(math.radians(a)), y + r * math.sin(math.radians(a))) for a in range(-90, 271, 6)]
    cum = _poly_len(pts)
    L = cum[-1]
    lt = s.lt
    dur = float(el.get("dur", 0.6))
    trim = s.trim if isinstance(s.trim, float) else 1.0
    if el.get("travel"):
        seg = float(el.get("seg", 0.28)) * L
        period = float(el.get("period", dur))
        ph = (lt / period) % 1.0 if el.get("loop") else clamp(lt / period)
        head = ph * (L + seg)
        a0, a1 = head - seg, head
    else:
        a0, a1 = 0.0, trim * L
        if isinstance(s.trim, tuple):
            a0, a1 = s.trim[0] * L, s.trim[1] * L
    sub = _sub_poly(pts, cum, a0, a1)
    if len(sub) < 2:
        return None
    fl = 1.0
    if el.get("flicker"):
        fl = 1 - float(el["flicker"]) * (0.5 + 0.5 * math.sin(s.t * 47) * math.sin(s.t * 13.1))

    def path():
        ctx.move_to(*sub[0])
        for p in sub[1:]:
            ctx.line_to(*p)

    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND); ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_operator(cairo.OPERATOR_ADD if el.get("additive", True) else cairo.OPERATOR_OVER)
    _glow_stroke(ctx, path, el.get("color", "#2ad1ff"), float(el.get("width", 8)), el.get("core", "#ffffff"),
                 float(el.get("haze", 3.2)), s.alpha, fl)
    hx, hy = sub[-1]
    hc = _color(el.get("core", "#ffffff"))
    g = cairo.RadialGradient(hx, hy, 0, hx, hy, float(el.get("width", 8)) * 4)
    g.add_color_stop_rgba(0, hc[0], hc[1], hc[2], s.alpha * 0.9 * fl)
    g.add_color_stop_rgba(1, hc[0], hc[1], hc[2], 0)
    ctx.set_source(g); ctx.arc(hx, hy, float(el.get("width", 8)) * 4, 0, 2 * math.pi); ctx.fill()
    ctx.restore()
    xs_, ys_ = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs_), min(ys_), max(xs_), max(ys_))


@element("flash_pulse", dur=0.5, enter="none", sfx="impact")
def el_flash_pulse(ctx, el, s):
    x, y = _point(el, "follow", s, xy(el, s.t)) if el.get("follow") else xy(el, s.t)
    lt = s.lt
    d = float(el.get("dur", 0.5))
    if lt > d + 0.05:
        return None
    p = clamp(lt / d)
    R = float(el.get("r", 140))
    c = _color(el.get("color", "#ffffff"))
    a = s.alpha
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    for k in range(int(el.get("rings", 2))):
        q = clamp(p * 1.3 - k * 0.18)
        if 0 < q < 1:
            ctx.set_source_rgba(c[0], c[1], c[2], a * (1 - q) ** 1.5)
            ctx.set_line_width(max(1.0, 10 * (1 - q)))
            ctx.arc(x, y, R * ease_out_expo(q), 0, 2 * math.pi); ctx.stroke()
    nr = int(el.get("rays", 8))
    if nr:
        q = ease_out(clamp(p * 1.6))
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for i in range(nr):
            ang = 2 * math.pi * i / nr + float(el.get("ray_rot", 0.3))
            r0, r1 = R * (0.35 + 0.5 * q), R * (0.55 + 0.75 * q)
            ctx.set_source_rgba(c[0], c[1], c[2], a * (1 - q))
            ctx.set_line_width(6 * (1 - q) + 1)
            ctx.move_to(x + r0 * math.cos(ang), y + r0 * math.sin(ang))
            ctx.line_to(x + r1 * math.cos(ang), y + r1 * math.sin(ang)); ctx.stroke()
    core = (1 - p) ** 3
    g = cairo.RadialGradient(x, y, 0, x, y, R * 0.6)
    g.add_color_stop_rgba(0, c[0], c[1], c[2], a * core)
    g.add_color_stop_rgba(1, c[0], c[1], c[2], 0)
    ctx.set_source(g); ctx.arc(x, y, R * 0.6, 0, 2 * math.pi); ctx.fill()
    if el.get("screen"):
        ctx.set_source_rgba(c[0], c[1], c[2], a * float(el["screen"]) * (1 - p) ** 2)
        ctx.paint()
    ctx.restore()
    return (x - R, y - R, x + R, y + R)


@element("glow_ring", dur=0.4, enter="fade", sfx="none")
def el_glow_ring(ctx, el, s):
    x, y = _point(el, "follow", s, xy(el, s.t)) if el.get("follow") else xy(el, s.t)
    r = float(val(el, "r", s.t, 90))
    c = _color(el.get("color", "#ffffff"))
    per = float(el.get("period", 1.2))
    n = int(el.get("rings", 3))
    a = s.alpha * (1 - s.out_p)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    for k in range(n):
        q = ((s.lt / per) + k / n) % 1.0
        ctx.set_source_rgba(c[0], c[1], c[2], a * (1 - q) ** 1.6 * 0.9)
        ctx.set_line_width(float(el.get("width", 4)) * (1 - q * 0.6))
        ctx.arc(x, y, r * (1 + q * float(el.get("spread", 0.9))), 0, 2 * math.pi); ctx.stroke()
    g = cairo.RadialGradient(x, y, r * 0.6, x, y, r * 1.5)
    g.add_color_stop_rgba(0, c[0], c[1], c[2], a * 0.18 * (0.8 + 0.2 * math.sin(s.lt * 6)))
    g.add_color_stop_rgba(1, c[0], c[1], c[2], 0)
    ctx.set_source(g); ctx.arc(x, y, r * 1.5, 0, 2 * math.pi); ctx.fill()
    ctx.restore()
    R = r * 1.9
    return (x - R, y - R, x + R, y + R)


@element("tracked_callout", dur=0.7, enter="draw", sfx="tick")
def el_tracked_callout(ctx, el, s):
    ax, ay = _point(el, "anchor", s)
    bx, by = el.get("box", [ax + 220, ay - 220])
    bx, by = float(bx), float(by)
    lag = float(el.get("lag", 0.0))
    if lag > 0 and isinstance(el.get("anchor"), str):         # label drifts after the anchor (spring feel)
        from .core import lerp
        pts = []
        for i in range(4):
            sub = type("S", (), {})(); sub.t = s.t - lag * i / 3; sub.scene = s.scene
            pts.append(_point(el, "anchor", sub))
        mx = sum(p[0] for p in pts) / 4 - ax; my = sum(p[1] for p in pts) / 4 - ay
        bx, by = bx + mx * float(el.get("box_follow", 1.0)), by + my * float(el.get("box_follow", 1.0))
    elif el.get("box_follow"):
        bx, by = bx + (ax - float(el.get("anchor0", [ax, ay])[0])) * float(el["box_follow"]), by
    col = _color(el.get("color", "#ffffff"))
    acc = _color(el.get("accent", el.get("color", "#ffffff")))
    trim = s.trim if isinstance(s.trim, float) else 1.0
    a = s.alpha
    size = float(el.get("size", 40))
    text = el.get("text", "")
    sub_t = el.get("sub")
    fnt = el.get("font", "title")
    from .text import resolve_font
    f_main = resolve_font(fnt, None, "title")
    tb = text_bbox(text, 0, 0, size, f_main)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    sb = text_bbox(sub_t, 0, 0, size * 0.6, resolve_font(el.get("sub_font", "body"), None, "body")) if sub_t else (0, 0, 0, 0)
    pw = max(tw, sb[2] - sb[0]) + size * 1.1
    ph = th + (sb[3] - sb[1] + size * 0.25 if sub_t else 0) + size * 0.75
    # elbow path anchor → box edge
    side = 1 if bx >= ax else -1
    ex = bx - side * pw / 2
    mode = el.get("elbow", "h")
    if mode == "h":
        pts = [(ax, ay), (ax + (ex - ax) * 0.45, by), (ex, by)] if abs(by - ay) > 4 else [(ax, ay), (ex, by)]
    elif mode == "v":
        pts = [(ax, ay), (ax, by), (ex, by)]
    else:
        pts = [(ax, ay), (ex, by)]
    cum = _poly_len(pts)
    line_p = clamp(trim * 1.6)
    sub = _sub_poly(pts, cum, 0, line_p * cum[-1])
    ctx.save()
    if el.get("dot", True):
        rp = clamp(trim * 5)
        ctx.set_source_rgba(acc[0], acc[1], acc[2], a * rp)
        ctx.arc(ax, ay, 7 * rp, 0, 2 * math.pi); ctx.fill()
        ctx.set_line_width(2)
        pulse = 1 + 0.35 * ((s.lt * 1.4) % 1.0)
        ctx.set_source_rgba(acc[0], acc[1], acc[2], a * rp * (1 - (s.lt * 1.4) % 1.0) * 0.8)
        ctx.arc(ax, ay, 14 * pulse, 0, 2 * math.pi); ctx.stroke()
    if len(sub) > 1:
        ctx.set_source_rgba(col[0], col[1], col[2], a)
        ctx.set_line_width(float(el.get("width", 3)))
        ctx.set_line_cap(cairo.LINE_CAP_ROUND); ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.move_to(*sub[0])
        for p in sub[1:]:
            ctx.line_to(*p)
        ctx.stroke()
    bp = ease_out(clamp((trim - 0.55) / 0.45))
    if bp > 0:
        style = el.get("box_style", "pill")
        ctx.save()
        ctx.translate(bx, by); ctx.scale(0.85 + 0.15 * bp, 0.85 + 0.15 * bp); ctx.translate(-bx, -by)
        if style != "plain":
            from .geometry import compile_path, rounded_rect_d
            rad = ph / 2 if style == "pill" else size * 0.3
            fill = _color(el.get("fill", "#ffffff" if style != "dark" else "#141416"))
            ctx.new_path(); ctx.append_path(compile_path(rounded_rect_d(bx - pw / 2, by - ph / 2, pw, ph, rad)).whole)
            ctx.set_source_rgba(fill[0], fill[1], fill[2], fill[3] * a * bp); ctx.fill_preserve()
            ctx.set_source_rgba(acc[0], acc[1], acc[2], a * bp * float(el.get("border_alpha", 0.0)))
            ctx.set_line_width(2.5); ctx.stroke()
        tcol = el.get("text_color", "#141416" if style not in ("plain", "dark") else "#ffffff")
        ty = by - (sb[3] - sb[1]) * 0.45 if sub_t else by
        draw_text(ctx, text, bx, ty, size=size, color=tcol, alpha=a * bp, weight=f_main)
        if sub_t:
            draw_text(ctx, sub_t, bx, ty + th * 0.5 + size * 0.45, size=size * 0.6,
                      color=el.get("sub_color", el.get("accent", tcol)), alpha=a * bp,
                      weight=resolve_font(el.get("sub_font", "body"), None, "body"))
        ctx.restore()
    ctx.restore()
    return (min(ax, bx - pw / 2) - 16, min(ay, by - ph / 2) - 16, max(ax, bx + pw / 2) + 16, max(ay, by + ph / 2) + 16)


@element("light_rays", dur=0.8, enter="fade", sfx="none")
def el_light_rays(ctx, el, s):
    x, y = xy(el, s.t)
    R = float(el.get("r", 700))
    n = int(el.get("n", 14))
    c = _color(el.get("color", "#fff3c4"))
    rot = math.radians(float(el.get("spin", 6)) * s.t)
    wdeg = float(el.get("width", 7))
    a = s.alpha * (1 - s.out_p) * float(el.get("alpha", 0.35))
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    for i in range(n):
        ang = rot + 2 * math.pi * i / n
        w = math.radians(wdeg * (0.6 + 0.4 * math.sin(i * 2.3 + s.t * 0.8)))
        g = cairo.RadialGradient(x, y, R * 0.05, x, y, R)
        g.add_color_stop_rgba(0, c[0], c[1], c[2], a)
        g.add_color_stop_rgba(1, c[0], c[1], c[2], 0)
        ctx.set_source(g)
        ctx.move_to(x, y)
        ctx.arc(x, y, R, ang - w / 2, ang + w / 2)
        ctx.close_path(); ctx.fill()
    ctx.restore()
    return None


@element("orbit_trail", dur=0.5, enter="fade", sfx="none")
def el_orbit_trail(ctx, el, s):
    x, y = xy(el, s.t)
    r = float(el.get("r", 300)); ry = float(el.get("ry", r))
    per = float(el.get("period", 4.0))
    d = float(el.get("dir", 1))
    a0 = math.radians(float(el.get("start_deg", -90)))
    trail = math.radians(float(el.get("trail", 70)))
    c = _color(el.get("color", "#ff4b2b"))
    a = s.alpha * (1 - s.out_p)
    spd = el.get("speed_keys")
    ang = a0 + d * 2 * math.pi * s.lt / per
    if spd:                     # [[t, revolutions], …] eased (accelerate etc.)
        from .core import Track
        ang = a0 + d * 2 * math.pi * float(Track([tuple(k) for k in spd])(s.lt))
    ctx.save()
    if el.get("ring", True):
        rc = _color(el.get("ring_color", "#ffffff"))
        ctx.set_source_rgba(rc[0], rc[1], rc[2], a * float(el.get("ring_alpha", 0.12)))
        ctx.set_line_width(2)
        ctx.save(); ctx.translate(x, y); ctx.scale(1, ry / r); ctx.arc(0, 0, r, 0, 2 * math.pi); ctx.restore(); ctx.stroke()
    ctx.set_operator(cairo.OPERATOR_ADD)
    steps = 28
    sz = float(el.get("size", 9))
    for i in range(steps):
        q = i / steps
        aa = ang - d * trail * q
        px, py = x + r * math.cos(aa), y + ry * math.sin(aa)
        ctx.set_source_rgba(c[0], c[1], c[2], a * (1 - q) ** 2 * 0.55)
        ctx.arc(px, py, sz * (1 - q * 0.7), 0, 2 * math.pi); ctx.fill()
    hx, hy = x + r * math.cos(ang), y + ry * math.sin(ang)
    g = cairo.RadialGradient(hx, hy, 0, hx, hy, sz * 4)
    g.add_color_stop_rgba(0, c[0], c[1], c[2], a)
    g.add_color_stop_rgba(1, c[0], c[1], c[2], 0)
    ctx.set_source(g); ctx.arc(hx, hy, sz * 4, 0, 2 * math.pi); ctx.fill()
    ctx.set_source_rgba(1, 1, 1, a); ctx.arc(hx, hy, sz * 0.55, 0, 2 * math.pi); ctx.fill()
    ctx.restore()
    return (x - r - sz * 4, y - ry - sz * 4, x + r + sz * 4, y + ry + sz * 4)


def _heart(ctx, x, y, s):
    ctx.move_to(x, y + s * 0.35)
    ctx.curve_to(x - s * 0.9, y - s * 0.25, x - s * 0.45, y - s * 0.95, x, y - s * 0.45)
    ctx.curve_to(x + s * 0.45, y - s * 0.95, x + s * 0.9, y - s * 0.25, x, y + s * 0.35)
    ctx.close_path()


@element("emitter", dur=0.3, enter="none", sfx="none")
def el_emitter(ctx, el, s):
    x, y = xy(el, s.t)
    kind = el.get("kind", "fall")
    n = int(el.get("n", 18))
    life = float(el.get("life", 2.2))
    rate = float(el.get("rate", n / max(0.1, life)))
    sp = float(el.get("speed", 380)); grav = float(el.get("gravity", 600 if kind == "fall" else 0))
    spread = math.radians(float(el.get("spread", 60)))
    aw, ah = el.get("area", [900, 0])
    cols = el.get("colors") or [el.get("color", "#d13b3b")]
    size = float(el.get("size", 34))
    shape = el.get("shape", "dot")
    rng = np.random.default_rng(int(el.get("seed", 7)))
    total = int(el.get("total", n if kind in ("burst", "radial") else int(rate * 30)))
    R = rng.random((total, 6))
    lt = s.lt
    a_el = s.alpha * (1 - s.out_p)
    bb = [x, y, x, y]
    ctx.save()
    for i in range(total):
        born = 0.0 if kind in ("burst", "radial") else i / rate
        if el.get("stagger") and kind in ("burst", "radial"):
            born = i * float(el["stagger"])
        age = lt - born
        if age < 0 or age > life:
            continue
        q = age / life
        if kind in ("burst", "radial"):
            ang = (2 * math.pi * i / total if kind == "radial" else -math.pi / 2 + (R[i, 0] - 0.5) * 2 * spread)
            ang += float(el.get("angle", 0)) * math.pi / 180
            v = sp * (0.6 + 0.6 * R[i, 1])
            dist = v * life * ease_out(q) * 0.5 if kind == "radial" else v * age
            px, py = x + math.cos(ang) * dist, y + math.sin(ang) * dist + 0.5 * grav * age * age
        elif kind == "rise":
            px = x + (R[i, 0] - 0.5) * aw + math.sin(age * 2 + i) * 18
            py = y - sp * age + (R[i, 1] - 0.5) * ah
        else:   # fall
            px = x + (R[i, 0] - 0.5) * aw + math.sin(age * 1.7 + i) * 26
            py = y + (R[i, 1] - 0.5) * ah + sp * 0.2 * age + 0.5 * grav * age * age * 0.25
        al = a_el * min(1.0, q * 6) * (1 - q) ** 0.7
        sz = size * (0.6 + 0.6 * R[i, 2])
        rot = (R[i, 3] - 0.5) * 2 + age * (R[i, 4] - 0.5) * 3
        c = _color(cols[i % len(cols)])
        ctx.save(); ctx.translate(px, py); ctx.rotate(rot)
        if shape == "heart":
            _heart(ctx, 0, 0, sz); ctx.set_source_rgba(c[0], c[1], c[2], c[3] * al); ctx.fill()
        elif shape == "square":
            ctx.rectangle(-sz / 2, -sz / 2, sz, sz); ctx.set_source_rgba(c[0], c[1], c[2], c[3] * al); ctx.fill()
        elif shape.startswith("icon:"):
            from .icons import draw_icon
            draw_icon(ctx, shape[5:], 0, 0, sz * 2, 1.0, color=cols[i % len(cols)], alpha=al)
        elif shape.startswith("text:"):
            words = shape[5:].split("|")
            from .text import resolve_font
            fill = el.get("bubble")
            w_ = words[i % len(words)]
            if fill:
                tb = text_bbox(w_, 0, 0, sz * 0.6, resolve_font("title", None, "title"))
                pw, ph = tb[2] - tb[0] + sz * 0.8, sz * 1.2
                from .geometry import compile_path, rounded_rect_d
                ctx.new_path(); ctx.append_path(compile_path(rounded_rect_d(-pw / 2, -ph / 2, pw, ph, ph * 0.4)).whole)
                fc = _color(fill); ctx.set_source_rgba(fc[0], fc[1], fc[2], fc[3] * al); ctx.fill()
            draw_text(ctx, w_, 0, 0, size=sz * 0.6, color=el.get("text_color", "#ffffff") if fill else cols[i % len(cols)],
                      alpha=al, weight=resolve_font("title", None, "title"))
        else:
            ctx.arc(0, 0, sz / 2, 0, 2 * math.pi); ctx.set_source_rgba(c[0], c[1], c[2], c[3] * al); ctx.fill()
        ctx.restore()
        bb = [min(bb[0], px - sz), min(bb[1], py - sz), max(bb[2], px + sz), max(bb[3], py + sz)]
    ctx.restore()
    return None          # particles never count for overlap QA
