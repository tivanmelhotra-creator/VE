"""
mograph.render — frame renderer + parallel segmented encoder.

render_frame(scene, t, scale=1.0) -> cairo.ImageSurface (BGRA premultiplied)
frame_to_rgb(surface) -> HxWx3 uint8 RGB numpy array
render_video(scene, out, preview=False, workers=None, start=0, end=None)

Speed design
------------
* every frame is a pure function of t  → frames are split into N contiguous
  chunks; each worker process renders its chunk and pipes BGRA straight into
  its OWN ffmpeg (no pickling of frames, no PNGs on disk), then segments are
  concatenated with `-c copy` and the audio is muxed once.
* preview=True renders at half resolution & 15 fps (≈8× cheaper) — use it for
  every iteration; render full quality only once at the end.
* bloom runs on a quarter-resolution threshold image (cheap, broadcast look).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time

import cairo
import cv2
import numpy as np

from .core import clamp, color as _color
from .elements import REGISTRY, compute_state
from .hud import draw_hud
from . import text as _text

# ---------------------------------------------------------------------------
# single frame
# ---------------------------------------------------------------------------


def _region_path(ctx, scene, reg, is_new, ox, oy):
    """Append the clip path of the OLD or NEW region of a wipe/slide/portal (in chapter-local coords
    offset by the chapter origin).  Returns nothing; caller clips."""
    W_, H_ = scene.W, scene.H
    p = reg["p"]
    if reg["type"] == "portal":
        cx, cy = reg["cx"], reg["cy"]
        R = portal_radius(scene, reg)
        BIG = 6000
        if reg["shape"] == "slab":           # diagonal slab expanding from the focal point
            d = R * 1.05
            ang = math.radians(float(reg.get("angle", -32)))
            ux, uy = math.cos(ang), math.sin(ang)     # slab direction
            nx, ny = -uy, ux                           # half-width direction
            pts = [(cx + ux * BIG + nx * d, cy + uy * BIG + ny * d), (cx - ux * BIG + nx * d, cy - uy * BIG + ny * d),
                   (cx - ux * BIG - nx * d, cy - uy * BIG - ny * d), (cx + ux * BIG - nx * d, cy + uy * BIG - ny * d)]
            if not is_new:
                ctx.rectangle(ox - BIG, oy - BIG, W_ + 2 * BIG, H_ + 2 * BIG)
            ctx.move_to(ox + pts[0][0], oy + pts[0][1])
            for q in pts[1:]:
                ctx.line_to(ox + q[0], oy + q[1])
            ctx.close_path()
        else:
            if not is_new:
                ctx.rectangle(ox - BIG, oy - BIG, W_ + 2 * BIG, H_ + 2 * BIG)
            ctx.new_sub_path()
            ctx.arc(ox + cx, oy + cy, max(0.01, R), 0, 2 * math.pi)
            ctx.close_path()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        return
    d = reg["dir"]
    e = p * (W_ if d in ("left", "right") else H_)
    if d == "left":     # new chapter enters from the RIGHT edge moving left (RTL reading)
        rect = (W_ - e, 0, e, H_) if is_new else (0, 0, W_ - e, H_)
    elif d == "right":
        rect = (0, 0, e, H_) if is_new else (e, 0, W_ - e, H_)
    elif d == "up":
        rect = (0, H_ - e, W_, e) if is_new else (0, 0, W_, H_ - e)
    else:
        rect = (0, 0, W_, e) if is_new else (0, e, W_, H_ - e)
    ctx.rectangle(ox + rect[0] - 4000 * (rect[0] <= 0), oy + rect[1] - 4000 * (rect[1] <= 0),
                  rect[2] + 4000 * (rect[0] <= 0) + 4000 * (rect[0] + rect[2] >= W_ - 1),
                  rect[3] + 4000 * (rect[1] <= 0) + 4000 * (rect[1] + rect[3] >= H_ - 1))


def portal_radius(scene, reg):
    """Iris radius that exactly covers the frame from the focal point at p = 1."""
    cx, cy = reg["cx"], reg["cy"]
    far = max(math.hypot(cx - x, cy - y) for x in (0, scene.W) for y in (0, scene.H))
    return reg["p"] * (far + 40)


def draw_world(ctx, scene, t, collect=None, paint_bg=False):
    """Draw all chapter elements at time t (camera already applied by caller).
    v8: every chapter is drawn inside its own style world (font roles, palette) and — during
    wipe/slide/portal transitions — inside its own region, on its own background colour."""
    screen_space = []
    reg = scene.region_state(t) if hasattr(scene, "region_state") else None
    _text.AUDIT_MUTE = "all" if reg is not None else None     # a transition frame is not a layout to audit
    for ch, ch_alpha in scene.visible_chapters(t):
        if ch_alpha <= 0.003:
            continue
        ox, oy = ch["origin"]
        w = ch.get("_world")
        if w is not None:
            _text.use_roles(w["fonts"])
        clipped = False
        if reg is not None:
            is_new = scene.chapters.index(ch) == reg["ci"]
            ctx.save(); clipped = True
            ctx.new_path()
            _region_path(ctx, scene, reg, is_new, ox, oy)
            ctx.clip()
            ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
            if paint_bg and w is not None:     # each world keeps its own paper inside its region
                ctx.set_source_rgba(*_color(ch["bg"]))
                ctx.paint()
            if reg["type"] in ("slide", "whip_pan"):  # content travels with the edge
                p, d = reg["p"], reg["dir"]
                W_, H_ = scene.W, scene.H
                off = (1 - p) * (W_ if d in ("left", "right") else H_) * (1 if is_new else 0) - \
                      p * (W_ if d in ("left", "right") else H_) * (0 if is_new else 1)
                sx = {"left": off, "right": -off}.get(d, 0); sy = {"up": off, "down": -off}.get(d, 0)
                ctx.translate(sx, sy)
            elif reg["type"] == "portal" and not is_new:     # the old world recedes as the portal opens
                k = 1 - 0.06 * reg["p"]
                cx, cy = ox + reg["cx"], oy + reg["cy"]
                ctx.translate(cx, cy); ctx.scale(k, k); ctx.translate(-cx, -cy)
            elif reg["type"] == "portal" and is_new:          # the new world settles in (slight overscale)
                k = 1.08 - 0.08 * reg["p"]
                cx, cy = ox + reg["cx"], oy + reg["cy"]
                ctx.translate(cx, cy); ctx.scale(k, k); ctx.translate(-cx, -cy)
        _draw_world_layers(ctx, scene, ch, t, ch_alpha, clipped, reg)
        frames = [e for e in ch["elements"] if e.get("_has_children")]
        for e in frames:
            e["_xf"] = None                 # set again by the container's draw if visible this frame
        open_frame = None
        for el in ch["elements"]:
            if open_frame is not None and el.get("parent") != open_frame.get("id"):
                from .showcase import draw_frame_border
                ctx.save(); ctx.translate(ox, oy); draw_frame_border(ctx, open_frame); ctx.restore()
                open_frame = None
            if el.get("_has_children"):
                open_frame = el
            if el.get("_bad"):
                continue
            s = compute_state(el, t, scene)
            if not s.ok():
                continue
            s.alpha *= ch_alpha
            if el.get("behind"):           # v11: drawn by its footage element (between plate and person)
                continue
            if el.get("space") == "screen":
                screen_space.append((el, s, w))
                continue
            ctx.save()
            ctx.translate(ox, oy)
            if el.get("parent"):           # v11: child of a device_frame → its clip + virtual-screen transform
                pel = scene.by_id.get(el["parent"])
                xf = pel.get("_xf") if pel is not None else None
                if not xf:
                    ctx.restore()
                    continue
                from .showcase import apply_parent
                apply_parent(ctx, xf)
                s.alpha *= xf[11]
            fxf = _follow(scene, el, t) if el.get("follow") else None
            if fxf is not None:
                from .fx import _apply_follow
                _apply_follow(ctx, fxf)
            pz = parallax(scene, el, t, ch)
            if pz is not None:
                ctx.translate(*pz[:2])
                if pz[2] != 1.0:
                    ctx.translate(pz[3], pz[4]); ctx.scale(pz[2], pz[2]); ctx.translate(-pz[3], -pz[4])
            _text.AUDIT_TAG = el.get("id", el["type"])
            _mute_prev = _text.AUDIT_MUTE
            if el.get("decor"):               # v11: deliberately faint decoration (ghost numbers, watermarks)
                _text.AUDIT_MUTE = "all"
            fn = REGISTRY[el["type"]]["fn"]
            if el.get("shiver"):
                from .glowfx import shiver_offset
                so = shiver_offset(el, t)
                if so:
                    ctx.translate(so[0], so[1])
            if el.get("fx") and isinstance(el["fx"], list):            # v16 AE-FX stack (mograph/aefx/bridge.py)
                from .aefx import bridge as _AB
                if "_fx" not in el:
                    _t0 = float(el.get("t", 0.0) or 0.0)
                    _t1 = float(el["out"]) if isinstance(el.get("out"), (int, float)) else _t0 + 9999.0
                    el["_fx"] = _AB.resolve_items(el["fx"], lambda v: float(v) if isinstance(v, (int, float))
                                                  else scene.resolve_time(v, 0.0, None), _t0, _t1)
                if _AB.draw_with_fx(ctx, lambda c: fn(c, el, s), el["_fx"], t, float(scene.fps)):
                    _text.AUDIT_MUTE = _mute_prev
                    ctx.restore()
                    continue
            if el.get("glow") or el.get("shadow"):
                from .glowfx import layer_fx
                if el.get("enter") == "whip_blur" or el.get("whip"):
                    bb = layer_fx(ctx, lambda c: _draw_whip(c, scene, el, s, fn), el.get("glow"), el.get("shadow"), t)
                else:
                    bb = layer_fx(ctx, lambda c: fn(c, el, s), el.get("glow"), el.get("shadow"), t)
            elif el.get("enter") == "whip_blur" or el.get("whip"):
                bb = _draw_whip(ctx, scene, el, s, fn)
            else:
                bb = fn(ctx, el, s)
            if bb:
                el["_last_bb"] = bb
            _text.AUDIT_MUTE = _mute_prev
            ctx.restore()

            if collect is not None and bb:
                if pz is not None:
                    bb = parallax_bbox(bb, pz)
                collect.append((el, (bb[0] + ox, bb[1] + oy, bb[2] + ox, bb[3] + oy), "world"))
        if open_frame is not None:          # v11: device_frame border/glass on top of its own children only
            from .showcase import draw_frame_border
            ctx.save(); ctx.translate(ox, oy); draw_frame_border(ctx, open_frame); ctx.restore()
        if clipped:
            ctx.restore()
    if reg is not None:
        _draw_region_edge(ctx, scene, reg)
    _text.AUDIT_MUTE = None
    return screen_space


def _follow(scene, el, t):
    """v11: tracker glue ("follow": "clip.hand") → (dx, dy, scale, rot, ex, ey) or None."""
    from .fx import follow_xform
    try:
        return follow_xform(scene, el, t)
    except Exception as e:  # noqa: BLE001
        el["_follow_err"] = str(e)
        return None


def _draw_world_layers(ctx, scene, ch, t, ch_alpha, clipped, reg):
    """v9: procedural backdrop (bg_mode) + ambient depth layers (bokeh / warp_stars / grid) of ONE chapter,
    painted under its elements.  Fade transitions cross-fade backdrops; pan/thread clip to the chapter's own
    screen; wipe/slide/portal/whip_pan are already clipped to the chapter's region."""
    from .apex import backdrop_spec, paint_backdrop, draw_ambient
    has_bd = backdrop_spec(ch) != "solid"
    has_amb = bool(ch.get("ambient") or (ch.get("_world") or {}).get("ambient"))
    if not (has_bd or has_amb):
        return
    ox, oy = ch["origin"]
    tr = ch["transition"]
    lt = t - ch["t"]
    a = 1.0
    is_cur = scene.chapters[scene.chapter_index(t)] is ch
    if not clipped and is_cur and lt < tr["dur"] and tr["type"] not in ("cut", "pan", "thread") and \
            scene.chapters.index(ch) > 0:
        a = clamp(lt / max(1e-3, tr["dur"]))
    ctx.save()
    if not clipped and tr["type"] in ("pan", "thread") or (not clipped and _neighbours_pan(scene, ch)):
        ctx.rectangle(ox, oy, scene.W, scene.H)
        ctx.clip()
    if has_bd:
        paint_backdrop(ctx, scene, ch, t, a)
    if has_amb:
        draw_ambient(ctx, scene, ch, t, a * (ch_alpha if not clipped else 1.0))
    ctx.restore()


def _neighbours_pan(scene, ch):
    i = scene.chapters.index(ch)
    return i + 1 < len(scene.chapters) and scene.chapters[i + 1]["transition"]["type"] in ("pan", "thread")


def _draw_whip(ctx, scene, el, s, fn):
    """v9 'enter': 'whip_blur' — offset along the whip axis + 1-D motion blur ∝ instantaneous velocity."""
    from .apex import blur_layer, whip_vec, whip_velocity, wants_whip
    if el.get("type") != "kinetic":
        dx, dy = whip_vec(el, s.lt)
        ctx.translate(dx, dy)
    if not wants_whip(el, s):
        return fn(ctx, el, s)
    v, d = whip_velocity(el, s, scene)
    ux, uy = ctx.user_to_device_distance(v * d[0] if d[0] else v, v * d[1] if d[1] else 0)
    L = min(float(el.get("blur_max", 160)), math.hypot(ux, uy) * float(el.get("blur", 0.9)))
    if L < 3:
        return fn(ctx, el, s)
    return blur_layer(ctx, lambda c: fn(c, el, s), L, d)


def _draw_region_edge(ctx, scene, reg):
    """The coloured band that "cuts" the screen (Swiss wipe signature) / the iris ring of a portal."""
    from .core import color as _c
    ox, oy = scene.chapters[reg["ci"]]["origin"]
    p = reg["p"]
    col = _c(reg["color"])
    W_, H_ = scene.W, scene.H
    if reg["type"] == "portal":
        band = reg["edge"] * (1 - abs(2 * p - 1)) ** 0.5
        if band <= 0.6 or p >= 0.995:
            return
        R = portal_radius(scene, reg)
        ctx.set_source_rgba(*col)
        if reg["shape"] == "slab":
            d = R * 1.05
            ang = math.radians(float(reg.get("angle", -32)))
            ux, uy = math.cos(ang), math.sin(ang); nx, ny = -uy, ux
            for sgn in (1, -1):
                cx, cy = ox + reg["cx"] + nx * d * sgn, oy + reg["cy"] + ny * d * sgn
                ctx.set_line_width(band)
                ctx.move_to(cx - ux * 6000, cy - uy * 6000); ctx.line_to(cx + ux * 6000, cy + uy * 6000)
                ctx.stroke()
        else:
            ctx.set_line_width(band)
            ctx.arc(ox + reg["cx"], oy + reg["cy"], R + band / 2, 0, 2 * math.pi); ctx.stroke()
        return
    band = reg["edge"] * (1 - abs(2 * p - 1)) ** 0.6
    if band <= 1:
        return
    ctx.set_source_rgba(*col)
    d = reg["dir"]
    e = p * (W_ if d in ("left", "right") else H_)
    if d == "left":
        ctx.rectangle(ox + W_ - e - band / 2, oy - 4000, band, H_ + 8000)
    elif d == "right":
        ctx.rectangle(ox + e - band / 2, oy - 4000, band, H_ + 8000)
    elif d == "up":
        ctx.rectangle(ox - 4000, oy + H_ - e - band / 2, W_ + 8000, band)
    else:
        ctx.rectangle(ox - 4000, oy + e - band / 2, W_ + 8000, band)
    ctx.fill()


# ---------------------------------------------------------------------------
# v8: 2.5D multi-plane parallax
# ---------------------------------------------------------------------------
def chapter_rest_view(scene, ch):
    """Camera view the chapter was authored against (its arrival view) — parallax is measured from it."""
    v = ch.get("_rest_view")
    if v is None:
        tr = ch["transition"]
        arrive = ch["t"] + (tr["dur"] if tr["type"] in ("pan", "thread") else 0.0) + 1e-3
        cx, cy, z, _ = scene.camera.view(min(arrive, (ch["end"] or arrive + 1) - 1e-3))
        v = ch["_rest_view"] = (cx, cy, z)
    return v


def parallax(scene, el, t, ch):
    """(dx, dy, scale, pivot_x, pivot_y) in chapter-local coords for an element with "z" != 0, else None.
    Camera pans by (dx,dy) from the chapter's rest view → the element moves an EXTRA z·(dx,dy) on screen
    (z>0 = closer, travels faster; z<0 = farther, lags) and scales by 1 + z·(zoom/zoom_rest − 1)."""
    z = el.get("z")
    if not z or el.get("space") == "screen":
        return None
    z = float(z)
    cx, cy, zm, _ = scene.camera.view(t)
    rx, ry, rz = chapter_rest_view(scene, ch)
    ox, oy = ch["origin"]
    dx, dy = cx - rx, cy - ry
    zr = zm / max(1e-6, rz)
    k = 1.0 + z * (zr - 1.0)
    # screen-space extra shift -z*(dx,dy) is a world shift of -z*(dx,dy)/zoom; keep it in world units
    sx, sy = -z * dx, -z * dy
    px, py = cx - ox, cy - oy          # scale about the camera centre (chapter-local)
    return (sx, sy, k, px, py)


def parallax_bbox(bb, pz):
    sx, sy, k, px, py = pz
    f = lambda x, p0: p0 + (x - p0) * k
    return (f(bb[0], px) + sx, f(bb[1], py) + sy, f(bb[2], px) + sx, f(bb[3], py) + sy)


def render_frame(scene, t, scale=1.0, hud=True, bloom=None, plates=True):
    W, H = int(round(scene.W * scale)), int(round(scene.H * scale))
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_GOOD)
    look = scene.look_at(t) if hasattr(scene, "look_at") else {"bg": _color(scene.bg), "region": False}
    ctx.set_source_rgba(*look["bg"])
    ctx.paint()
    ctx.scale(scale, scale)

    plate = None
    if plates and getattr(scene, "plates", None):
        from .media import active_plate, draw_plate, plate_alpha
        plate = active_plate(scene, t)
    if plate is not None and plate["mode"] == "under":
        draw_plate(ctx, scene, plate, t)          # footage BEHIND the vector layer
    if plate is None or plate["mode"] != "replace" or plate_alpha(plate, t) < 0.999:
        ctx.save()
        if scene.spec.get("_shakes") or scene.spec.get("meta", {}).get("beat_pump"):
            from .glowfx import scene_shake, scene_pump
            sx, sy = scene_shake(scene, t)
            pm = scene_pump(scene, t)
            ctx.translate(sx, sy)
            if pm != 1.0:
                ctx.translate(scene.W / 2, scene.H / 2); ctx.scale(pm, pm); ctx.translate(-scene.W / 2, -scene.H / 2)
        scene.camera.apply(ctx, t)
        screen_space = draw_world(ctx, scene, t, paint_bg=look.get("region", False))
        _draw_threads(ctx, scene, t)
        ctx.restore()
        for el, s, w in screen_space:
            if w is not None:
                _text.use_roles(w["fonts"])
            REGISTRY[el["type"]]["fn"](ctx, el, s)
    if plate is not None and plate["mode"] == "replace":
        draw_plate(ctx, scene, plate, t)          # realistic Omni shot replaces the world (cross-fades)
    if getattr(scene, "_has_whip_pan", False):
        from .apex import frame_blur, whip_pan_blur
        bl, horiz = whip_pan_blur(scene, t)
        if bl >= 3:
            frame_blur(surf, bl * scale, horiz)

    if scene.spec.get("_flashes"):
        from .glowfx import scene_flash
        fa, fcol = scene_flash(scene, t)
        if fa > 0.003:
            ctx.save(); ctx.identity_matrix()
            ctx.set_source_rgba(*_color(fcol, fa)); ctx.set_operator(cairo.OPERATOR_ADD); ctx.paint()
            ctx.restore()
    fx = scene.spec.get("meta", {})
    b = look.get("bloom", scene.bloom) if bloom is None else bloom
    pulse = _slam_pulse(scene, t) if look.get("slam_pulse", fx.get("slam_pulse", True)) else 0.0
    vig = float(look.get("vignette", fx.get("vignette", 0.0)) or 0.0)
    grn = float(look.get("grain", fx.get("grain", 0.0)) or 0.0)
    if b > 0 or pulse > 0 or vig or grn:
        surf.flush()
        if b > 0:
            _bloom_inplace(surf, b * (1 + 1.6 * pulse))
        _post_fx(surf, t, vig, grn, pulse)
        surf.mark_dirty()
    if scene.spec.get("_cutfx"):                  # v12: edit-point transitions (zoom blur / whip / glitch …)
        from .transfx import apply as _tfx
        _tfx(scene, surf, t, scale)
    if hud and scene.chapters:
        from .contrast import hud_env
        ci = scene.chapter_index(t)
        ch = scene.chapters[ci]
        hcfg = ch.get("_hud") or scene.hud
        w = ch.get("_world")
        if w is not None:
            _text.use_roles(w["fonts"])
        env = hud_env(surf, scene.W, scene.H, hcfg, scale)      # scene awareness: sample what is REALLY there
        _text.AUDIT_TAG = "hud"
        draw_hud(ctx, scene.W, scene.H, hcfg, scene.chapters, ci, t, scene.subs, env=env,
                 prev_hud=scene.chapters[ci - 1].get("_hud") if ci > 0 else None)
    surf.flush()
    return surf


def _draw_threads(ctx, scene, t):
    """Leader line between chapters: S-curve drawn by a glowing comet head, the tail retracts after
    arrival (no stale line across the frame), and the landing point emits a ring pulse."""
    import math as _m
    from .geometry import draw_path, point_on_path
    from .core import ease_in_out, prog, clamp
    for th in getattr(scene, "threads", []):
        t0, dur = th["t0"], th["dur"]
        p = ease_in_out(prog(t, t0, dur * 0.85))
        if p <= 0:
            continue
        r = ease_in_out(prog(t, t0 + dur + th.get("hold", 1.6) * 0.25, 0.8))   # tail retract
        if r >= 0.999 and t > t0 + dur + 2.0:
            continue
        (ax, ay), (bx, by) = th["a"], th["b"]
        sw = th.get("swing", 140) * (1 if th["ch"] % 2 else -1)
        my = (ay + by) / 2
        d = f"M {ax} {ay} C {ax + sw} {my} {bx - sw} {my} {bx} {by}"
        # soft glow under + crisp core
        draw_path(ctx, d, 0, 0, 1, anchor="none", trim=(r, p), stroke="#ffffff", width=9, alpha=0.10)
        draw_path(ctx, d, 0, 0, 1, anchor="none", trim=(r, p), stroke="#ffffff", width=2.6, alpha=0.95)
        if r < 0.02:
            ctx.set_source_rgba(1, 1, 1, 1 - r * 50)
            ctx.arc(ax, ay, 6, 0, 2 * _m.pi); ctx.fill()
        if p < 0.995:                       # comet head
            hx, hy, _ = point_on_path(d, max(p, 1e-3))
            for rad, al in ((26, 0.08), (15, 0.18), (8, 0.5)):
                ctx.set_source_rgba(1, 1, 1, al); ctx.arc(hx, hy, rad, 0, 2 * _m.pi); ctx.fill()
            ctx.set_source_rgba(1, 1, 1, 1); ctx.arc(hx, hy, 4.5, 0, 2 * _m.pi); ctx.fill()
        else:                               # landing: dot + expanding ring
            land = t0 + dur * 0.85
            k = clamp((t - land) / 0.7)
            ctx.set_source_rgba(1, 1, 1, 1 - clamp((t - land - 1.2) / 0.5))
            ctx.arc(bx, by, 6, 0, 2 * _m.pi); ctx.fill()
            if k < 1:
                ctx.set_source_rgba(1, 1, 1, 0.8 * (1 - k)); ctx.set_line_width(2.4)
                ctx.arc(bx, by, 8 + 46 * k, 0, 2 * _m.pi); ctx.stroke()


_SLAMS_CACHE = {}


def _slam_pulse(scene, t):
    """0..1 brief exposure pulse right after stamp slams (tactile impact)."""
    key = id(scene)
    if key not in _SLAMS_CACHE:
        ts = []
        for ch in scene.chapters:
            for el in ch["elements"]:
                if el.get("type") == "stamp" or el.get("enter") == "slam":
                    ts.append(float(el["t"]) + 0.1)
                st = el.get("stamp")
                if isinstance(st, dict) and "t" in st:
                    ts.append(float(st["t"]) + 0.1)
        _SLAMS_CACHE[key] = ts
    best = 0.0
    for ts in _SLAMS_CACHE[key]:
        d = t - ts
        if 0 <= d < 0.35:
            best = max(best, (1 - d / 0.35) ** 2)
    return best


_VIG = {}


_GRAIN = {}


def _post_fx(surf, t, vignette, grain, pulse):
    """Vignette + exposure pulse + film grain, all in cv2 uint8 ops on the full BGRA buffer
    (~6 ms at 1080x1920; the naive float version was ~130 ms)."""
    W, H, st = surf.get_width(), surf.get_height(), surf.get_stride()
    buf = np.ndarray((H, st // 4, 4), dtype=np.uint8, buffer=surf.get_data())
    if vignette > 0:
        k = (st // 4, H, round(vignette, 3))
        if k not in _VIG:
            yy, xx = np.mgrid[0:H, 0:st // 4].astype(np.float32)
            r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.414
            m = np.clip(1 - vignette * np.clip(r - 0.35, 0, 1) ** 1.6 * 1.6, 0, 1)
            m4 = np.repeat((m * 255).astype(np.uint8)[:, :, None], 4, axis=2)
            m4[:, :, 3] = 255
            _VIG.clear(); _VIG[k] = m4
        cv2.multiply(buf, _VIG[k], dst=buf, scale=1 / 255.0)
    if pulse > 0:
        lift = int(28 * pulse)
        cv2.add(buf, (lift, lift, lift, 0), dst=buf)
    if grain > 0:
        gk = (st // 4, H, round(grain, 4))
        if gk not in _GRAIN:
            _GRAIN.clear()
            rng = np.random.default_rng(7)
            bank = []
            for _ in range(6):    # 6 pre-baked half-res tiles, cycled at 12 fps (film-like, cheap to encode)
                n = rng.normal(0, 255 * grain, (H // 2, st // 8)).astype(np.float32)
                pos = np.clip(n, 0, 255).astype(np.uint8); neg = np.clip(-n, 0, 255).astype(np.uint8)
                bank.append((pos, neg))
            _GRAIN[gk] = bank
        bank = _GRAIN[gk]
        pos, neg = bank[int(t * 12) % len(bank)]
        P = cv2.resize(pos, (st // 4, H), interpolation=cv2.INTER_LINEAR)
        N = cv2.resize(neg, (st // 4, H), interpolation=cv2.INTER_LINEAR)
        P4 = cv2.merge([P, P, P, np.zeros_like(P)]); N4 = cv2.merge([N, N, N, np.zeros_like(N)])
        cv2.add(buf, P4, dst=buf); cv2.subtract(buf, N4, dst=buf)


def _as_array(surf):
    W, H, st = surf.get_width(), surf.get_height(), surf.get_stride()
    a = np.ndarray((H, st // 4, 4), dtype=np.uint8, buffer=surf.get_data())
    return a[:, :W, :]


def _bloom_inplace(surf, strength):
    """Selective optical bloom (mip-pyramid): bright strokes glow, ~8x cheaper than a full-res blur.
    Work is done on 1/8-res grayscale + small blurs, then upsampled once and added."""
    a = _as_array(surf)
    H, W = a.shape[:2]
    g = cv2.cvtColor(a, cv2.COLOR_BGRA2GRAY)          # monochrome style → 1 channel is enough
    q = cv2.resize(g, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    q = cv2.threshold(q, 40, 255, cv2.THRESH_TOZERO)[1]
    g1 = cv2.GaussianBlur(q, (0, 0), 1.6)
    g2 = cv2.GaussianBlur(cv2.resize(q, (W // 32, H // 32), interpolation=cv2.INTER_AREA), (0, 0), 1.6)
    g2 = cv2.resize(g2, (W // 8, H // 8), interpolation=cv2.INTER_LINEAR)
    glow = cv2.addWeighted(g1, 0.9 * strength, g2, 0.9 * strength, 0)
    up = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
    up4 = cv2.cvtColor(up, cv2.COLOR_GRAY2BGRA)
    up4[:, :, 3] = 0
    a[:] = cv2.add(a, up4)


def frame_to_rgb(surf):
    a = _as_array(surf)
    return cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_BGRA2RGB)


def save_frame(scene, t, path, scale=1.0):
    surf = render_frame(scene, t, scale)
    surf.write_to_png(path)
    return path


# ---------------------------------------------------------------------------
# parallel video render
# ---------------------------------------------------------------------------

_G = {}


def _worker(args):
    seg_idx, f0, f1, fps, scale, seg_path, crf, preset = args
    scene = _G["scene"]
    W, H = int(round(scene.W * scale)), int(round(scene.H * scale))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-g", str(int(fps * 2)), "-threads", "2", "-x264-params", "rc-lookahead=15:ref=2",
           "-maxrate", "9M", "-bufsize", "18M", seg_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        surf = render_frame(scene, f / fps, scale)
        a = _as_array(surf)
        proc.stdin.write(np.ascontiguousarray(a).tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg segment {seg_idx} failed ({rc})")
    return seg_idx, f1 - f0


def _avail_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 4096


def auto_workers(scale=1.0):
    """CPU count, capped by RAM (each worker = cairo frame + its own x264 encoder)."""
    per = 420 * scale * scale + 60          # MB per worker at 1080x1920 ≈ 480MB
    by_mem = int(max(1, (_avail_mb() - 150) // per))
    return max(1, min(os.cpu_count() or 2, by_mem))


def render_video(scene, out_path, preview=False, workers=None, start=0.0, end=None, audio_path=None,
                 crf=None, preset=None, log=print):
    fps = 15.0 if preview else scene.fps
    scale = 0.5 if preview else 1.0
    crf = crf or (26 if preview else 18)
    preset = preset or ("ultrafast" if preview else "medium")
    end = scene.duration if end is None else min(end, scene.duration)
    f_start, f_end = int(round(start * fps)), int(round(end * fps))
    n = f_end - f_start
    workers = workers or auto_workers(scale)
    workers = max(1, min(workers, n // 8 or 1))
    tmp = tempfile.mkdtemp(prefix="mograph_")
    chunk = math.ceil(n / workers)
    jobs = []
    for i in range(workers):
        a, b = f_start + i * chunk, min(f_end, f_start + (i + 1) * chunk)
        if a < b:
            jobs.append((i, a, b, fps, scale, os.path.join(tmp, f"seg_{i:03d}.mp4"), crf, preset))
    t0 = time.time()
    _G["scene"] = scene
    log(f"[render] {n} frames @ {fps:g}fps, {int(scene.W * scale)}x{int(scene.H * scale)}, "
        f"{len(jobs)} workers{' (PREVIEW)' if preview else ''}")
    if len(jobs) == 1:
        _worker(jobs[0])
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(len(jobs)) as pool:
            for idx, cnt in pool.imap_unordered(_worker, jobs):
                log(f"  segment {idx} done ({cnt} frames, {time.time() - t0:.1f}s)")
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        for j in jobs:
            f.write(f"file '{j[5]}'\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst]
    if audio_path and os.path.exists(audio_path):
        cmd += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", audio_path, "-map", "0:v:0",
                "-map", "1:a:0", "-c:a", "aac", "-b:a", "256k", "-shortest"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    dt = time.time() - t0
    log(f"[render] done → {out_path}  ({dt:.1f}s, {n / max(dt, 1e-6):.1f} fps)")
    return out_path
