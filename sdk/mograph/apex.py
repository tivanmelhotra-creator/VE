"""
mograph.apex — v9 Studio Apex world layer: procedural chapter backdrops, ambient depth layers, directional
motion blur (whip), aspect-ratio presets.

BACKDROPS  ("bg_mode" on a chapter, on meta, or from the style preset)
  solid             flat @paper (default, zero cost)
  radial_spotlight  deep edge (@edge, #070103) → glowing @glow spotlight (#5E0818) — crimson-black promo look
  studio_vignette   bright studio centre (#FAFAFA) → soft silver outer vignette (#B8B8C0)
  crimson_vivid     vibrant @accent centre (#FF1E42) → deep crimson edge (#9E0820)
  "bg_focus": [x, y] (chapter-local spotlight centre, default frame centre), "bg_radius": px (default 0.9·H)
  "bg_pulse": 0..1   spotlight breathes with the narrator (scene.vo_env)

AMBIENT LAYERS  ("ambient" on a chapter: "bokeh" | "warp_stars" | "grid" | list | {"kind": ..., options})
  bokeh       18–25 soft out-of-focus @accent discs drifting slowly in 2.5D depth (parallax with the camera)
              {"kind":"bokeh","n":22,"color":"@accent","size":[18,70],"alpha":0.22,"speed":14}
  warp_stars  radial speed lines flying outward from the focus; intensity follows camera ZOOM velocity
              (+ "warp": base 0..1)   {"kind":"warp_stars","n":90,"warp":0.2,"color":"@ink","center":[540,960]}
  grid        subtle architectural grid with parallax depth   {"kind":"grid","step":90,"z":-0.25,"color":"#ffffff12"}

WHIP
  "enter": "whip_blur" on text / headline / kinetic (every word) / any element: the element rockets in along
  "whip_dir" ("right" = from the right edge, RTL default | "left" | "up" | "down") over "whip_dur" (0.38 s)
  from "whip_dist" px, with a 1-D directional blur kernel whose length is proportional to the instantaneous
  velocity |v(t)| (cv2.filter2D on an offscreen layer).  Chapter transition {"type": "whip_pan"} slides
  old → new world with full-frame directional blur.

ASPECT  meta.aspect = "9:16" (1080×1920, default) | "16:9" (1920×1080) | "1:1" (1080×1080) | "4:5" (1080×1350)
"""
from __future__ import annotations

import math

import cairo
import cv2
import numpy as np

from .core import clamp, color as _color, ease_out_expo, ease_in_out, lerp, prog

ASPECTS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080), "4:5": (1080, 1350)}
BG_MODES = ("solid", "radial_spotlight", "studio_vignette", "crimson_vivid")


# ---------------------------------------------------------------------------
# backdrops
# ---------------------------------------------------------------------------
def backdrop_spec(ch):
    w = ch.get("_world") or {}
    return w.get("bg_mode") or "solid"


def _stops(mode, pal):
    if mode == "radial_spotlight":
        return pal.get("glow", "#5E0818"), pal.get("edge", "#070103"), pal.get("mid")
    if mode == "studio_vignette":
        return pal.get("studio", "#FAFAFA"), pal.get("vignette", "#B8B8C0"), None
    if mode == "crimson_vivid":
        return pal.get("vivid", pal.get("accent", "#FF1E42")), pal.get("vivid_edge", "#9E0820"), None
    return None


def paint_backdrop(ctx, scene, ch, t, alpha=1.0, solid_too=False):
    """Paint the chapter backdrop over the whole clip (world coords, chapter origin applied here)."""
    mode = backdrop_spec(ch)
    w = ch.get("_world") or {}
    if mode == "solid" or mode not in BG_MODES:
        if solid_too:
            c = _color(ch.get("bg", "#000000"))
            ctx.set_source_rgba(c[0], c[1], c[2], c[3] * alpha)
            ctx.paint()
        return
    pal = w.get("palette", {})
    a, b, mid = _stops(mode, pal)
    ox, oy = ch["origin"]
    fx, fy = ch.get("bg_focus") or (scene.W / 2, scene.H * (0.46 if mode == "radial_spotlight" else 0.5))
    R = float(ch.get("bg_radius", max(scene.W, scene.H) * (0.78 if mode == "radial_spotlight" else 0.95)))
    pulse = float(ch.get("bg_pulse", 0.0))
    if pulse:
        R *= 1 + 0.08 * pulse * scene.vo_env(t)
    g = cairo.RadialGradient(ox + fx, oy + fy, 0, ox + fx, oy + fy, R)
    ca, cb = _color(a), _color(b)
    g.add_color_stop_rgba(0.0, ca[0], ca[1], ca[2], alpha)
    if mode == "radial_spotlight":
        cm = _color(mid) if mid else tuple(lerp(ca[k], cb[k], 0.55) for k in range(3)) + (1,)
        g.add_color_stop_rgba(0.42, cm[0], cm[1], cm[2], alpha)
    g.add_color_stop_rgba(1.0, cb[0], cb[1], cb[2], alpha)
    g.set_extend(cairo.EXTEND_PAD)
    ctx.set_source(g)
    ctx.paint()


def backdrop_center_color(ch):
    """Representative colour (for contrast decisions) — the spotlight centre."""
    mode = backdrop_spec(ch)
    w = ch.get("_world") or {}
    st = _stops(mode, w.get("palette", {}))
    return st[0] if st else ch.get("bg")


# ---------------------------------------------------------------------------
# ambient layers
# ---------------------------------------------------------------------------
def _ambient_list(ch):
    a = ch.get("ambient")
    if a is None:
        a = (ch.get("_world") or {}).get("ambient")
    if not a:
        return []
    if isinstance(a, (str, dict)):
        a = [a]
    out = []
    for it in a:
        out.append({"kind": it} if isinstance(it, str) else dict(it))
    return out


def _rng(seed):
    return np.random.default_rng(seed)


def _resolve(c, pal, default):
    if c is None:
        c = default
    if isinstance(c, str) and c.startswith("@"):
        c = pal.get(c[1:], pal.get("accent", "#ffffff"))
    return _color(c)


def draw_ambient(ctx, scene, ch, t, alpha=1.0):
    layers = _ambient_list(ch)
    if not layers:
        return
    w = ch.get("_world") or {}
    pal = w.get("palette", {})
    ox, oy = ch["origin"]
    cx, cy, zm, _rot = scene.camera.view(t)
    from .render import chapter_rest_view
    rx, ry, rz = chapter_rest_view(scene, ch)
    dx, dy = cx - rx, cy - ry
    W, H = scene.W, scene.H
    ci = scene.chapters.index(ch)
    lt = t - ch["t"]
    for li, L in enumerate(layers):
        k = L.get("kind")
        seed = int(L.get("seed", 11 + ci * 7 + li))
        if k == "bokeh":
            n = int(L.get("n", 22))
            col = _resolve(L.get("color"), pal, "@accent")
            smin, smax = L.get("size", [16, 74])
            base_a = float(L.get("alpha", 0.24))
            sp = float(L.get("speed", 14))
            r = _rng(seed)
            px, py = r.uniform(-0.1, 1.1, n) * W, r.uniform(-0.05, 1.05, n) * H
            zz = r.uniform(-0.45, 0.35, n)
            rad = r.uniform(smin, smax, n)
            ph = r.uniform(0, 2 * math.pi, n)
            vx, vy = r.uniform(-0.6, 0.6, n), r.uniform(-1.0, -0.2, n)
            fade = clamp(lt / 0.8)
            for i in range(n):
                z = zz[i]
                x = px[i] + vx[i] * sp * t + 18 * math.sin(0.4 * t + ph[i]) - z * dx
                y = py[i] + vy[i] * sp * t + 12 * math.cos(0.33 * t + ph[i]) - z * dy
                x = (x + 0.15 * W) % (1.3 * W) - 0.15 * W
                y = (y + 0.1 * H) % (1.2 * H) - 0.1 * H
                rr = rad[i] * (1 + 0.6 * (z + 0.45))
                tw = 0.65 + 0.35 * math.sin(1.3 * t + ph[i] * 3)
                a = alpha * fade * base_a * tw * (0.55 + 0.9 * (z + 0.45))
                g = cairo.RadialGradient(ox + x, oy + y, 0, ox + x, oy + y, rr)
                g.add_color_stop_rgba(0, col[0], col[1], col[2], a)
                g.add_color_stop_rgba(0.55, col[0], col[1], col[2], a * 0.55)
                g.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
                ctx.set_source(g)
                ctx.arc(ox + x, oy + y, rr, 0, 2 * math.pi)
                ctx.fill()
        elif k == "warp_stars":
            n = int(L.get("n", 90))
            col = _resolve(L.get("color"), pal, "@ink")
            cxl, cyl = L.get("center") or (W / 2, H / 2)
            # zoom velocity drives the streak length (|d zoom / dt| / zoom)
            z0 = scene.camera.view(t - 1 / 30)[2]
            zv = abs(zm - z0) * 30 / max(0.2, zm)
            inten = clamp(float(L.get("warp", 0.15)) + zv * float(L.get("gain", 2.2)))
            if inten <= 0.01:
                continue
            r = _rng(seed)
            ang = r.uniform(0, 2 * math.pi, n)
            spd = r.uniform(0.35, 1.0, n)
            off = r.uniform(0, 1, n)
            Rm = math.hypot(W, H) * 0.62
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            for i in range(n):
                f = (off[i] + spd[i] * t * float(L.get("speed", 0.55))) % 1.0
                r0 = 40 + f ** 1.6 * Rm
                ln = (14 + 260 * inten * spd[i]) * f
                ca, sa = math.cos(ang[i]), math.sin(ang[i])
                a = alpha * clamp(f * 3) * (1 - f) * 1.6 * inten * float(L.get("alpha", 0.7))
                if a <= 0.01:
                    continue
                ctx.set_source_rgba(col[0], col[1], col[2], a)
                ctx.set_line_width(1.2 + 2.6 * f * spd[i])
                ctx.move_to(ox + cxl + ca * r0, oy + cyl + sa * r0)
                ctx.line_to(ox + cxl + ca * (r0 + ln), oy + cyl + sa * (r0 + ln))
                ctx.stroke()
        elif k == "dot_matrix":
            dflt = pal.get("dot", "#1e1e24")
            col = _resolve(L.get("color", dflt), pal, dflt)
            step = float(L.get("step", 32.0))
            dot_r = float(L.get("r", 1.5))
            view_hw = (W * 0.5) / max(0.25, zm) + step * 3
            view_hh = (H * 0.5) / max(0.25, zm) + step * 3
            sx0 = math.floor((cx - view_hw) / step) * step
            sx1 = math.ceil((cx + view_hw) / step) * step
            sy0 = math.floor((cy - view_hh) / step) * step
            sy1 = math.ceil((cy + view_hh) / step) * step
            ctx.set_source_rgba(col[0], col[1], col[2], col[3] * alpha)
            ctx.new_path()
            gx = sx0
            while gx <= sx1:
                gy = sy0
                while gy <= sy1:
                    ctx.move_to(gx + dot_r, gy)
                    ctx.arc(gx, gy, dot_r, 0, 2 * math.pi)
                    gy += step
                gx += step
            ctx.fill()
        elif k == "grid":
            from .contrast import is_light
            dflt = "#00000012" if is_light(ch.get("bg", "#000")) else "#ffffff14"
            col = _resolve(L.get("color", dflt), pal, dflt)
            step = float(L.get("step", 90))
            z = float(L.get("z", -0.25))
            sx, sy = -z * dx, -z * dy
            ctx.set_line_width(float(L.get("width", 1.2)))
            ctx.set_source_rgba(col[0], col[1], col[2], col[3] * alpha * clamp(lt / 0.6))
            x0 = ox + (sx % step) - step * 3
            while x0 < ox + W + step * 3:
                ctx.move_to(x0, oy - H); ctx.line_to(x0, oy + 2 * H); x0 += step
            y0 = oy + (sy % step) - step * 3
            while y0 < oy + H + step * 3:
                ctx.move_to(ox - W, y0); ctx.line_to(ox + 2 * W, y0); y0 += step
            ctx.stroke()
            if L.get("dots", True):                      # brighter crossings every 4 cells (architectural)
                ctx.set_source_rgba(col[0], col[1], col[2], min(1.0, col[3] * 3.2) * alpha * clamp(lt / 0.6))
                xi = ox + (sx % (step * 4)) - step * 4
                while xi < ox + W + step * 4:
                    yi = oy + (sy % (step * 4)) - step * 4
                    while yi < oy + H + step * 4:
                        ctx.rectangle(xi - 2, yi - 2, 4, 4); yi += step * 4
                    xi += step * 4
                ctx.fill()


# ---------------------------------------------------------------------------
# whip (directional motion blur)
# ---------------------------------------------------------------------------
_DIRS = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}


def _whip_x(lt, dur, dist):
    return dist * (1 - ease_out_expo(clamp(lt / max(1e-3, dur))))


def whip_offset(el, s):
    """Signed offset along the whip axis (scalar along x for text; generic elements use whip_vec)."""
    d = _DIRS.get(el.get("whip_dir", "right"), (1, 0))
    return d[0] * _whip_x(s.lt, float(el.get("whip_dur", 0.38)), float(el.get("whip_dist", 900)))


def whip_vec(el, lt):
    d = _DIRS.get(el.get("whip_dir", "right"), (1, 0))
    k = _whip_x(lt, float(el.get("whip_dur", 0.38)), float(el.get("whip_dist", 900)))
    return d[0] * k, d[1] * k


def whip_word(t, wt, size, dur=0.32):
    """Kinetic word offset (each word rockets in from the right, RTL reading direction)."""
    return _whip_x(t - wt, dur, size * 5.0)


def whip_velocity(el, s, scene):
    """Instantaneous whip velocity in px/frame (user units) along the whip axis at the element's state."""
    fps = scene.fps if scene is not None else 30.0
    if el.get("type") == "kinetic":
        from .typo import kinetic_words
        ws = kinetic_words(el, scene)
        cur = max([i for i, w in enumerate(ws) if s.t >= w[1]], default=-1)
        if cur < 0:
            return 0.0, (1, 0)
        wt = ws[cur][1]
        size = float(el.get("size", 150))
        v = abs(whip_word(s.t, wt, size) - whip_word(s.t - 1 / fps, wt, size))
        return v, (1, 0)
    d = _DIRS.get(el.get("whip_dir", "right"), (1, 0))
    dur, dist = float(el.get("whip_dur", 0.38)), float(el.get("whip_dist", 900))
    v = abs(_whip_x(s.lt, dur, dist) - _whip_x(s.lt - 1 / fps, dur, dist))
    return v, d


def wants_whip(el, s):
    if el.get("enter") != "whip_blur" and not el.get("whip"):
        return False
    if el.get("type") == "kinetic":
        return True
    return s.lt < float(el.get("whip_dur", 0.38)) + 0.05


def blur_layer(ctx, draw_fn, length_px, direction, scale=1.0):
    """Render draw_fn(ctx) into an offscreen layer (same device transform), apply a 1-D directional box
    kernel of `length_px` device pixels along `direction`, composite OVER.  Returns draw_fn's result."""
    tgt = ctx.get_target()
    W, H = tgt.get_width(), tgt.get_height()
    layer = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    lc = cairo.Context(layer)
    lc.set_matrix(ctx.get_matrix())
    lc.set_antialias(ctx.get_antialias())
    res = draw_fn(lc)
    layer.flush()
    k = int(round(length_px))
    if k >= 3:
        st = layer.get_stride()
        a = np.ndarray((H, st // 4, 4), np.uint8, buffer=layer.get_data())
        ys, xs = np.nonzero(a[:, :W, 3] if a.shape[1] >= W else a[:, :, 3])
        if len(ys):
            pad = k + 4
            y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
            roi = np.ascontiguousarray(a[y0:y1, x0:x1])
            dx, dy = direction
            if abs(dx) >= abs(dy):
                ker = np.ones((1, k), np.float32) / k
            else:
                ker = np.ones((k, 1), np.float32) / k
            out = cv2.filter2D(roi, -1, ker, borderType=cv2.BORDER_CONSTANT)
            a[y0:y1, x0:x1] = out
        layer.mark_dirty()
    ctx.save()
    ctx.identity_matrix()
    ctx.set_source_surface(layer, 0, 0)
    ctx.paint()
    ctx.restore()
    return res


def frame_blur(surf, length_px, horizontal=True):
    """Full-frame directional blur in place (whip_pan transition)."""
    k = int(round(length_px))
    if k < 3:
        return
    surf.flush()
    W, H, st = surf.get_width(), surf.get_height(), surf.get_stride()
    a = np.ndarray((H, st // 4, 4), np.uint8, buffer=surf.get_data())
    ker = np.ones((1, k), np.float32) / k if horizontal else np.ones((k, 1), np.float32) / k
    a[:] = cv2.filter2D(a, -1, ker, borderType=cv2.BORDER_REFLECT)
    surf.mark_dirty()


def whip_pan_blur(scene, t):
    """Blur length (px at 1080 wide) of an active whip_pan transition at t, else 0."""
    if not scene.chapters:
        return 0.0, True
    i = scene.chapter_index(t)
    ch = scene.chapters[i]
    tr = ch["transition"]
    if i == 0 or tr.get("type") != "whip_pan":
        return 0.0, True
    lt = t - ch["t"]
    d = float(tr["dur"])
    if lt >= d:
        return 0.0, True
    from .core import get_ease
    e = get_ease(tr.get("ease", "in_out_expo"))
    v = abs(e(clamp(lt / d)) - e(clamp((lt - 1 / scene.fps) / d))) * (scene.W if tr.get("dir", "left") in
                                                                      ("left", "right") else scene.H)
    return min(float(tr.get("blur_max", 220)), v * float(tr.get("blur", 1.4))), tr.get("dir", "left") in ("left", "right")
