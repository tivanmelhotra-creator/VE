"""
mograph.creative_reels — v13.9 (@novadent 100/100 Master + Creative Viral Reels Engine).

Implements pixel-accurate, optical-light-spill, and real backdrop-blur Glassmorphic widgets
from the User's 7 forensic screenshots of @novadent.ir + the 2 companion viral reels:
  1. novadent_split_dome     (Screenshot 211857: Dashed golden semi-circle with real Glassmorphic
                              backdrop-blur dome + split trapezoidal Spartina/Ozempic panels behind speaker)
  2. novadent_cd_dial        (Screenshots 211908 & 211917: Dark glass vinyl/CD donut ring with 2-line
                              'بیشتر از / این که' white core, true curved Persian arc text, top-only red rim
                              casting red light spill onto speaker's white collar -> 180° Ease-In-Out rotation
                              to bottom red rim casting red light spill onto speaker's hands/cuff!)
  3. novadent_diamond_square (Screenshot 211936: Tilted -16° rounded-square wireframe where 'م' of 'مورد اول'
                              and 'خ/ی' of 3D golden 'خشکی' break across the border + 45° rotated Diamond ◇ badge)
  4. novadent_chair_glow_text (Screenshot 212019: 'ماهایی که / زدیم / چیکار / کنیم' where 'ز' of golden 'زدیم'
                              tucks behind the black chair backrest and casts golden light spill onto the chair edge)
  5. novadent_red_halo       (Screenshot 212032: Concentric 2-layer dark-red & bright-red CD halo behind speaker
                              and chair on a darkened charcoal room backdrop with 'اصلاً / سخت / نیست')
  6. novadent_glass_hex      (Screenshot 212043: True Backdrop-Blur Glassmorphic rounded hexagon over chest with
                              golden border, bright white-gold top-vertex specular bloom, and Ease-In-Out open/close)
  7. orbiting_title_3d       (3D head-orbiting title trajectory & bird-swoop from behind speaker to subtitle bar)
  8. wireframe_badge, circular_dial, glass_hud_frame, zigzag_path_3d, avatar_carousel_3d, silhouette_scribble,
     and compose_before_after_frame / render_before_after_video.
"""
from __future__ import annotations
import math
import os
from functools import lru_cache
import cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import clamp, color as _color, ease_in_out, ease_out, get_ease, lerp, prog, spring
from .elements import element, val, xy
from .reelfx import _ASSETS_3D, _blit_3d, _blit_3d_perspective, _draw_glow_underline, _load_3d_rgba
from .text import draw_text, resolve_font, to_fa_digits


def _cubic_ease_in_out(p: float) -> float:
    """Silky cinematic Ease-In-Out (cubic-bezier(0.65, 0.0, 0.35, 1.0))."""
    p = clamp(p)
    return 4.0 * p * p * p if p < 0.5 else 1.0 - ((-2.0 * p + 2.0) ** 3) / 2.0


def _rounded_poly_path(ctx, pts: list[tuple[float, float]], radius: float = 18.0):
    """Draw a closed polygon with smoothly rounded corners."""
    n = len(pts)
    if n < 3:
        return
    ctx.new_path()
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        v1x, v1y = p_prev[0] - p_curr[0], p_prev[1] - p_curr[1]
        v2x, v2y = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
        l1 = max(1e-4, math.hypot(v1x, v1y))
        l2 = max(1e-4, math.hypot(v2x, v2y))
        r = min(radius, l1 * 0.45, l2 * 0.45)
        ax, ay = p_curr[0] + (v1x / l1) * r, p_curr[1] + (v1y / l1) * r
        bx, by = p_curr[0] + (v2x / l2) * r, p_curr[1] + (v2y / l2) * r
        if i == 0:
            ctx.move_to(ax, ay)
        else:
            ctx.line_to(ax, ay)
        ctx.curve_to(p_curr[0], p_curr[1], p_curr[0], p_curr[1], bx, by)
    ctx.close_path()


def _paint_backdrop_blur(ctx, path_fn, x0: float, y0: float, x1: float, y1: float,
                         blur_k: int = 35, tint_bgr: tuple[float, float, float] = (16.0, 20.0, 26.0),
                         keep_ratio: float = 0.52, alpha: float = 0.94):
    """
    True Glassmorphism (Backdrop-Filter Blur):
    Samples the already-rendered pixels on `ctx.get_target()` inside `(x0, y0, x1, y1)`,
    applies a strong Gaussian blur + subtle glass tint, and paints it clipped to `path_fn(ctx)`.
    """
    surf = ctx.get_target()
    if not isinstance(surf, cairo.ImageSurface):
        return
    surf.flush()
    W, H = surf.get_width(), surf.get_height()
    # Transform user-space bbox corners to device pixels
    dx0, dy0 = ctx.user_to_device(x0, y0)
    dx1, dy1 = ctx.user_to_device(x1, y1)
    ix0 = max(0, int(min(dx0, dx1) - 8))
    iy0 = max(0, int(min(dy0, dy1) - 8))
    ix1 = min(W, int(max(dx0, dx1) + 8))
    iy1 = min(H, int(max(dy0, dy1) + 8))
    if ix1 <= ix0 + 4 or iy1 <= iy0 + 4:
        return
    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape((H, W, 4))
    roi = buf[iy0:iy1, ix0:ix1, :3].copy()
    k = blur_k if blur_k % 2 == 1 else blur_k + 1
    blurred = cv2.GaussianBlur(roi, (k, k), k * 0.32)
    tinted = np.clip(
        blurred.astype(np.float32) * keep_ratio + np.array(tint_bgr, dtype=np.float32) * (1.0 - keep_ratio),
        0, 255
    ).astype(np.uint8)
    bgra = np.dstack([tinted, np.full(tinted.shape[:2], 255, dtype=np.uint8)])
    r_surf = cairo.ImageSurface.create_for_data(bgra, cairo.FORMAT_ARGB32, ix1 - ix0, iy1 - iy0)

    ctx.save()
    path_fn(ctx)
    ctx.clip()
    ctx.identity_matrix()
    ctx.set_source_surface(r_surf, ix0, iy0)
    ctx.paint_with_alpha(clamp(alpha))
    ctx.restore()


@lru_cache(maxsize=64)
def _polar_arc_surface(text: str, radius: int, size: int, font_key: str,
                       color_rgb: tuple[int, int, int], bottom_arc: bool,
                       embossed: bool, span_deg: float):
    """
    Renders a full Persian sentence into a flat high-res RGBA strip via Cairo/HarfBuzz (keeping 100%
    of cursive connections intact), then polar-warps (`cv2.remap`) the strip onto a circular arc of
    radius `radius` so every letter curves smoothly like a true vinyl record label!
    """
    from .text import text_bbox
    from . import text as _text

    _mute = _text.AUDIT_MUTE
    _text.AUDIT_MUTE = "all"
    bb = text_bbox(text, 0, 0, size=float(size), weight=font_key)
    tw = max(40, int(math.ceil(bb[2] - bb[0]) + size * 0.8))
    th = max(24, int(math.ceil(bb[3] - bb[1]) + size * 0.9))

    strip_surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, tw, th)
    sctx = cairo.Context(strip_surf)
    if embossed:
        draw_text(sctx, text, tw * 0.5 + 1.5, th * 0.52 + 1.5, size=float(size), weight=font_key,
                  color="#6E7385", alpha=0.58)
        draw_text(sctx, text, tw * 0.5, th * 0.52, size=float(size), weight=font_key,
                  color="#1B1D24", alpha=0.96)
    else:
        hex_c = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
        draw_text(sctx, text, tw * 0.5, th * 0.52, size=float(size), weight=font_key,
                  color=hex_c, alpha=1.0)
    strip_surf.flush()
    _text.AUDIT_MUTE = _mute

    strip_bgra = np.frombuffer(strip_surf.get_data(), dtype=np.uint8).reshape((th, tw, 4)).copy()

    # Build polar-to-cartesian remap grid on a square canvas of size D x D
    R = float(radius)
    pad_r = th * 0.65
    D = int(math.ceil((R + pad_r) * 2.0)) + 8
    C = D * 0.5
    yy, xx = np.mgrid[0:D, 0:D].astype(np.float32)
    dx = xx - C
    dy = yy - C
    rr = np.hypot(dx, dy)

    # Angular span of the text strip along radius R (cap at span_deg)
    ang_span = min(math.radians(span_deg), (tw * 1.28) / max(40.0, R))
    half_span = ang_span * 0.5

    if not bottom_arc:
        # Top arc: angle 0 is at 12 o'clock (dx=0, dy=-R); positive theta is clockwise (right), negative is left
        theta = np.arctan2(dx, -dy)
        # In the flat strip, x=0 is left end of Persian sentence, x=tw is right end of Persian sentence.
        # On the top arc, right end of sentence is at +half_span, left end is at -half_span!
        u = ((theta + half_span) / max(1e-4, ang_span)) * tw
        # Top of glyph (y=0 in strip) is at larger radius (R + th/2), bottom of glyph (y=th) is at (R - th/2)
        v = (0.5 - (rr - R) / float(th)) * th
    else:
        # Bottom arc: angle 0 is at 6 o'clock (dx=0, dy=+R); positive theta is right (5 o'clock), negative is left (7 o'clock)
        theta = np.arctan2(dx, dy)
        # Right end of Persian sentence is at +half_span (5 o'clock), left end is at -half_span (7 o'clock)
        u = ((theta + half_span) / max(1e-4, ang_span)) * tw
        # Top of glyph (y=0 in strip) is at smaller radius (R - th/2, closer to center), bottom is at (R + th/2)
        v = (0.5 + (rr - R) / float(th)) * th

    warped = cv2.remap(strip_bgra, u.astype(np.float32), v.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped, D


@lru_cache(maxsize=64)
def _polar_arc_surface(text: str, radius: int, size: int, font_key: str,
                       color_rgb: tuple[int, int, int], bottom_arc: bool,
                       embossed: bool, span_deg: float):
    """
    Renders a full Persian sentence into a flat high-res RGBA strip via Cairo/HarfBuzz (keeping 100%
    of cursive connections intact), then polar-warps (`cv2.remap`) the strip onto a circular arc of
    radius `radius` so every letter curves smoothly like a true vinyl record label!
    """
    from .text import text_bbox
    from . import text as _text

    _mute = _text.AUDIT_MUTE
    _text.AUDIT_MUTE = "all"
    bb = text_bbox(text, 0, 0, size=float(size), weight=font_key)
    tw = max(40, int(math.ceil(bb[2] - bb[0]) + size * 0.6))
    th = max(24, int(math.ceil(bb[3] - bb[1]) + size * 0.8))

    strip_surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, tw, th)
    sctx = cairo.Context(strip_surf)
    if embossed:
        draw_text(sctx, text, tw * 0.5 + 1.5, th * 0.52 + 1.5, size=float(size), weight=font_key,
                  color="#707588", alpha=0.58)
        draw_text(sctx, text, tw * 0.5, th * 0.52, size=float(size), weight=font_key,
                  color="#181A20", alpha=0.96)
    else:
        hex_c = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
        draw_text(sctx, text, tw * 0.5, th * 0.52, size=float(size), weight=font_key,
                  color=hex_c, alpha=1.0)
    strip_surf.flush()
    _text.AUDIT_MUTE = _mute

    strip_bgra = np.frombuffer(strip_surf.get_data(), dtype=np.uint8).reshape((th, tw, 4)).copy()

    R = float(radius)
    pad_r = th * 0.65
    D = int(math.ceil((R + pad_r) * 2.0)) + 8
    C = D * 0.5
    yy, xx = np.mgrid[0:D, 0:D].astype(np.float32)
    dx = xx - C
    dy = yy - C
    rr = np.hypot(dx, dy)

    ang_span = min(math.radians(span_deg), (tw * 1.18) / max(40.0, R))
    half_span = ang_span * 0.5

    if not bottom_arc:
        # Top arc: angle 0 at 12 o'clock (dx=0, dy=-R)
        theta = np.arctan2(dx, -dy)
        u = ((theta + half_span) / max(1e-4, ang_span)) * tw
        v = (0.5 - (rr - R) / float(th)) * th
    else:
        # Bottom arc: angle 0 at 6 o'clock (dx=0, dy=+R)
        theta = np.arctan2(dx, dy)
        u = ((theta + half_span) / max(1e-4, ang_span)) * tw
        v = (0.5 + (rr - R) / float(th)) * th

    warped = cv2.remap(strip_bgra, u.astype(np.float32), v.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped, D


def draw_curved_arc_text(ctx, text: str, cx: float, cy: float, radius: float,
                         size: float = 38.0, font_role: str = "subtitle",
                         color_hex: str = "#FFFFFF", alpha: float = 1.0,
                         bottom_arc: bool = False, embossed: bool = False,
                         span_deg: float = 175.0):
    """Draw polar-warped continuous curved Persian text along a circular arc."""
    if not text or alpha <= 0.005:
        return
    f_key = resolve_font(font_role, "bold", "subtitle")
    c_flt = _color(color_hex)
    c_rgb = (int(c_flt[0] * 255), int(c_flt[1] * 255), int(c_flt[2] * 255))
    warped_bgra, D = _polar_arc_surface(text, int(round(radius)), int(round(size)), f_key,
                                        c_rgb, bool(bottom_arc), bool(embossed), float(span_deg))
    surf = cairo.ImageSurface.create_for_data(warped_bgra, cairo.FORMAT_ARGB32, D, D)
    ctx.save()
    ctx.translate(cx - D * 0.5, cy - D * 0.5)
    ctx.set_source_surface(surf, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint_with_alpha(clamp(alpha))
    ctx.restore()



def draw_mixed_font_line(ctx, text: str, cx: float, cy: float, base_size: float = 40.0,
                         thin_font: str = "IRANYekanLight.ttf",
                         bold_font: str = "KalamehFaNum-Black.ttf",
                         callig_font: str = "Delbar-Bold.ttf",
                         base_color: str = "#E2E8F0",
                         emph_color: str = "#FBBF24",
                         emph_scale: float = 1.38,
                         alpha: float = 1.0) -> tuple[float, float]:
    """
    Renders a single horizontal Persian sentence where individual words can have drastically
    different font families, stroke weights (Ultra-Thin vs Extra-Black / Calligraphic), sizes,
    and glowing colors!
      * Plain words -> rendered in `thin_font` (e.g. IRANYekanThin.ttf / IRANYekanLight.ttf) at `1.0x` size in `base_color`
      * `**word**`  -> rendered in `callig_font` (e.g. Delbar-Bold.ttf) at `emph_scale` (1.38x) in `emph_color` with glow!
      * `__word__`  -> rendered in `bold_font` (e.g. IRANYekanExtraBlack2.ttf / KalamehFaNum-Black.ttf) at `emph_scale` in `emph_color`!
    """
    from .text import text_bbox
    from . import text as _text

    raw_tokens = [w for w in text.split() if w]
    if not raw_tokens or alpha <= 0.005:
        return (0.0, 0.0)

    _mute = _text.AUDIT_MUTE
    _text.AUDIT_MUTE = "all"

    items = []
    for tok in raw_tokens:
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            clean = tok[2:-2]
            f_use = callig_font
            sz_use = base_size * emph_scale
            col_use = emph_color
            is_emph = True
        elif tok.startswith("__") and tok.endswith("__") and len(tok) > 4:
            clean = tok[2:-2]
            f_use = bold_font
            sz_use = base_size * (emph_scale * 0.96)
            col_use = emph_color
            is_emph = True
        else:
            clean = tok.replace("**", "").replace("__", "")
            f_use = thin_font
            sz_use = base_size
            col_use = base_color
            is_emph = False
        bb = text_bbox(clean, 0, 0, size=sz_use, weight=f_use)
        w_px = max(10.0, bb[2] - bb[0])
        items.append((clean, f_use, sz_use, col_use, w_px, is_emph))

    space_px = base_size * 0.28
    total_w = sum(it[4] for it in items) + space_px * max(0, len(items) - 1)
    max_h = max(it[2] for it in items)

    # RTL layout: start at right edge (cx + total_w * 0.5) and step left
    cur_right = cx + total_w * 0.5
    for clean, f_use, sz_use, col_use, w_px, is_emph in items:
        wx = cur_right - w_px * 0.5
        wy = cy - (sz_use - base_size) * 0.12
        if is_emph:
            # Soft radial glow behind the emphasized word
            rgb = _color(col_use)
            g = cairo.RadialGradient(wx, wy, 2.0, wx, wy, w_px * 0.85)
            g.add_color_stop_rgba(0.0, rgb[0], rgb[1], rgb[2], 0.36 * alpha)
            g.add_color_stop_rgba(1.0, rgb[0], rgb[1], rgb[2], 0.0)
            ctx.save()
            ctx.set_source(g)
            ctx.arc(wx, wy, w_px * 0.85, 0, 2 * math.pi)
            ctx.fill()
            ctx.restore()
        draw_text(ctx, clean, wx, wy, size=sz_use, weight=f_use, color=col_use, alpha=alpha)
        cur_right -= (w_px + space_px)

    _text.AUDIT_MUTE = _mute
    return (total_w, max_h)


@element("novadent_split_dome", dur=0.75, enter="fade", sfx="whoosh")
def el_novadent_split_dome(ctx, el, s):
    """
    Screenshot 2026-10-19 211857.png ('آمپول‌های لاغری / اسپارتینا / اوزمپیک'):
    Behind the speaker (`behind: "clip"`):
      1. Two angled trapezoidal/diagonal product panels in the upper backdrop (bounded at bottom by a
         crisp white horizontal line at `y_split`):
         - Left trapezoid: emerald-olive product backdrop + tall condensed outline 'SPARTINA' + white 'اسپارتینا'
         - Right trapezoid: warm amber-brown product backdrop + tall condensed outline 'OZEMPIC' + white 'اوزمپیک'
           + tilted 3D product/pen sprite rising up on the right!
      2. A semi-circular Glassmorphic Backdrop-Blur Dome at the top center, bordered by a glowing golden
         dashed/dotted semi-circular arc!
      3. Mixed-font title inside the dome: thin delicate 'آمپول‌های' on top + massive sweeping calligraphic 'لاغری' below!
    """
    W = s.scene.W if s.scene else 1080
    y_split = float(el.get("y_split", 780))
    local_t = max(0.0, s.t - float(el["t"]))
    p_in = _cubic_ease_in_out(clamp(local_t / max(0.25, float(el.get("dur", 0.75)))))
    p_panels = _cubic_ease_in_out(clamp((local_t - float(el.get("panels_at", 0.35))) / 0.55))

    ctx.save()
    # 1. Left & Right Angled Trapezoidal Product Panels (slide in with Ease-In-Out)
    diag_top_x = W * 0.46
    diag_bot_x = W * 0.56
    if p_panels > 0.01:
        # Left Trapezoid (Spartina - dark emerald/olive studio tone)
        ctx.save()
        ctx.move_to(0, 0)
        ctx.line_to(diag_top_x, 0)
        ctx.line_to(diag_bot_x, y_split)
        ctx.line_to(0, y_split)
        ctx.close_path()
        lg_l = cairo.LinearGradient(0, 0, diag_bot_x, y_split)
        lg_l.add_color_stop_rgba(0.0, 0.14, 0.16, 0.10, 0.94 * s.alpha * p_panels)
        lg_l.add_color_stop_rgba(0.6, 0.10, 0.24, 0.18, 0.95 * s.alpha * p_panels)
        lg_l.add_color_stop_rgba(1.0, 0.05, 0.12, 0.09, 0.96 * s.alpha * p_panels)
        ctx.set_source(lg_l)
        ctx.fill_preserve()
        ctx.clip()
        # Subtle product box shapes in background
        ctx.set_source_rgba(0.22, 0.48, 0.36, 0.32 * s.alpha * p_panels)
        ctx.rectangle(W * 0.24, y_split * 0.52, 240, 190)
        ctx.fill()
        # Ghostly tall condensed English outline 'SPARTINA' + crisp white 'اسپارتینا'
        f_en = resolve_font("counter", "bold", "title")
        f_fa = resolve_font("subtitle", "bold", "subtitle")
        draw_text(ctx, el.get("left_en", "SPARTINA"), W * 0.18, y_split - 125, size=86,
                  color="#94A3B8", alpha=0.26 * s.alpha * p_panels, weight=f_en, style="outline")
        draw_text(ctx, el.get("left_fa", "اسپارتینا"), W * 0.19, y_split - 95, size=50,
                  color="#FFFFFF", alpha=0.96 * s.alpha * p_panels, weight=f_fa)
        ctx.restore()

        # Right Trapezoid (Ozempic - warm dark amber/copper studio tone)
        ctx.save()
        ctx.move_to(diag_top_x, 0)
        ctx.line_to(W, 0)
        ctx.line_to(W, y_split)
        ctx.line_to(diag_bot_x, y_split)
        ctx.close_path()
        lg_r = cairo.LinearGradient(diag_top_x, 0, W, y_split)
        lg_r.add_color_stop_rgba(0.0, 0.18, 0.11, 0.06, 0.94 * s.alpha * p_panels)
        lg_r.add_color_stop_rgba(1.0, 0.32, 0.16, 0.08, 0.96 * s.alpha * p_panels)
        ctx.set_source(lg_r)
        ctx.fill_preserve()
        ctx.clip()
        # Tilted 3D/product visual on the right (syringe/pen style cylinder or sprite)
        ctx.save()
        ctx.translate(W * 0.77, y_split * 0.52)
        ctx.rotate(math.radians(-26.0))
        pen_g = cairo.LinearGradient(-42, -260, 42, 260)
        pen_g.add_color_stop_rgba(0.0, 0.85, 0.92, 0.98, 0.92 * s.alpha * p_panels)
        pen_g.add_color_stop_rgba(0.45, 0.25, 0.58, 0.78, 0.95 * s.alpha * p_panels)
        pen_g.add_color_stop_rgba(1.0, 0.45, 0.62, 0.88, 0.95 * s.alpha * p_panels)
        _rounded_poly_path(ctx, [(-42, -260), (42, -260), (42, 240), (-42, 240)], radius=18)
        ctx.set_source(pen_g)
        ctx.fill()
        ctx.restore()
        draw_text(ctx, el.get("right_en", "OZEMPIC"), W * 0.79, y_split - 125, size=86,
                  color="#FDE68A", alpha=0.28 * s.alpha * p_panels, weight=f_en, style="outline")
        draw_text(ctx, el.get("right_fa", "اوزمپیک"), W * 0.80, y_split - 95, size=50,
                  color="#FFFFFF", alpha=0.96 * s.alpha * p_panels, weight=f_fa)
        ctx.restore()

        # Crisp white horizontal divider line at y_split (shoulder level)
        ctx.set_line_width(3.2)
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.92 * s.alpha * p_panels)
        ctx.move_to(0, y_split)
        ctx.line_to(W, y_split)
        ctx.stroke()

    # 2. Semi-Circular Glassmorphic Frosted Dome + Dashed Golden Arc
    dome_cx, dome_cy, dome_r = W * 0.5, float(el.get("dome_y", 145)), float(el.get("dome_r", 430))
    def _dome_path(c):
        c.new_path()
        c.arc(dome_cx, dome_cy, dome_r, math.radians(8), math.radians(172))
        c.close_path()

    _paint_backdrop_blur(ctx, _dome_path, dome_cx - dome_r, 0, dome_cx + dome_r, dome_cy + dome_r,
                         blur_k=41, tint_bgr=(18.0, 22.0, 26.0), keep_ratio=0.48, alpha=0.88 * s.alpha * p_in)

    # Dashed/Dotted Golden Semi-Circular Arc with warm orange glow
    gold_rgb = _color(el.get("accent", "#F59E0B"))
    for lw, al, dsh in [(14.0, 0.18, []), (5.5, 0.35, [22.0, 14.0]), (2.6, 0.96, [22.0, 14.0, 4.0, 14.0])]:
        ctx.save()
        ctx.new_path()
        ctx.arc(dome_cx, dome_cy, dome_r * p_in, math.radians(12), math.radians(168))
        ctx.set_line_width(lw)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        if dsh:
            ctx.set_dash(dsh, -s.t * 28.0)
        ctx.set_source_rgba(gold_rgb[0], gold_rgb[1], gold_rgb[2], al * s.alpha * p_in)
        ctx.stroke()
        ctx.restore()

    # 3. Two-Font Calligraphic Lockup inside the Glassmorphic Semi-Circle ('آمپول‌های' + 'لاغری')
    top_title = el.get("top_title", "آمپول‌های")
    main_title = el.get("title", "لاغری")
    f_thin = "IRANYekanThin.ttf"
    f_callig = "Delbar-Bold.ttf" if "Delbar-Bold.ttf" in resolve_font("handwritten", "bold", "handwritten") else resolve_font("title", "black", "title")

    ty_shift = (1.0 - p_in) * 35.0
    draw_text(ctx, top_title, dome_cx + 15, dome_cy + 135 + ty_shift, size=float(el.get("top_size", 82)),
              color="#E2E8F0", alpha=s.alpha * p_in, weight=f_thin)
    draw_text(ctx, main_title, dome_cx - 25, dome_cy + 275 + ty_shift, size=float(el.get("size", 138)),
              color="#FFFFFF", alpha=s.alpha * p_in, weight=f_callig)

    ctx.restore()
    return None


@element("novadent_cd_dial", dur=0.75, enter="spring", sfx="whoosh")
def el_novadent_cd_dial(ctx, el, s):
    """
    Screenshots 2026-10-19 211908.png & 211917.png ('بیشتر از / این که' Vinyl/CD Ring):
      * Translucent dark-glass CD donut ring (`r = 215`) on speaker's chest with real backdrop blur.
      * White center disc with 2-line lockup: bold black 'بیشتر از' on top + ultra-thin 'این که' below.
      * True curved Persian arc text along the top track ('روی کاهش وزن اثر بذارن!') and bottom track
        ('روی سلامت دهان و دندونتون').
      * State 1 (t < rotate_at): Top outer rim glows crimson-red (bottom outer rim has NO red!),
        top curved text glows white, and a soft `OPERATOR_SCREEN` crimson light spill illuminates
        the speaker's white shirt collar & neck above the disc!
      * State 2 (t >= rotate_at): Rotates 180° with silky Ease-In-Out (`_cubic_ease_in_out`)!
        Top outer rim loses red, top curved text becomes dark embossed charcoal (`embossed=True`),
        bottom outer rim glows red-orange, bottom curved text glows golden-yellow, and the red
        optical light spill moves down onto the speaker's hands and white shirt cuff!
    """
    x, y = xy(el, s.t)
    r = float(val(el, "r", s.t, 218)) * s.scale
    rot_at = float(el.get("rotate_at", float(el["t"]) + 1.45))
    rot_dur = float(el.get("rotate_dur", 0.72))
    p_rot = _cubic_ease_in_out(clamp((s.t - rot_at) / max(0.15, rot_dur)))

    red_rgb = _color(el.get("red_color", "#FF2438"))
    gold_hex = el.get("gold_color", "#FBBF24")

    ctx.save()

    # 1. DYNAMIC OPTICAL RED LIGHT SPILL ON SPEAKER'S CLOTHES/SKIN (Collar in State 1 -> Hand/Cuff in State 2!)
    # In State 1 (p_rot=0), light source is at top rim (x, y - r*0.95) spilling onto white collar & chin.
    # In State 2 (p_rot=1), light source rotates to bottom rim (x - r*0.1, y + r*0.95) spilling onto hands & cuff.
    spill_x = lerp(x, x - r * 0.08, p_rot)
    spill_y = lerp(y - r * 0.92, y + r * 0.92, p_rot)
    spill_r = r * 1.18
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_SCREEN)
    sp_g = cairo.RadialGradient(spill_x, spill_y, spill_r * 0.05, spill_x, spill_y, spill_r)
    sp_g.add_color_stop_rgba(0.0, red_rgb[0], red_rgb[1] * 0.15, red_rgb[2] * 0.18, 0.68 * s.alpha)
    sp_g.add_color_stop_rgba(0.48, red_rgb[0], 0.05, 0.08, 0.34 * s.alpha)
    sp_g.add_color_stop_rgba(1.0, 0.0, 0.0, 0.0, 0.0)
    ctx.set_source(sp_g)
    ctx.arc(spill_x, spill_y, spill_r, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()

    # 2. Dark Translucent Glass CD Donut Body (with subtle backdrop blur)
    def _cd_path(c):
        c.new_path()
        c.arc(x, y + s.dy, r, 0, 2 * math.pi)

    _paint_backdrop_blur(ctx, _cd_path, x - r, y + s.dy - r, x + r, y + s.dy + r,
                         blur_k=25, tint_bgr=(10.0, 10.0, 12.0), keep_ratio=0.22, alpha=0.86 * s.alpha)

    ctx.translate(x, y + s.dy)

    # Subtle vinyl groove rings
    ctx.set_line_width(1.0)
    for gr_f in (0.52, 0.66, 0.82):
        ctx.arc(0, 0, r * gr_f, 0, 2 * math.pi)
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.06 * s.alpha)
        ctx.stroke()

    # Base thin neutral white circumference around the entire circle
    ctx.arc(0, 0, r, 0, 2 * math.pi)
    ctx.set_line_width(1.6)
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.45 * s.alpha)
    ctx.stroke()

    # 3. Active Half-Circumference Red Glowing Rim Arc (rotates smoothly by 180° * p_rot!)
    # At p_rot=0: centered at -90° (top half: -162° .. -18°).
    # At p_rot=1: centered at +90° (bottom half: +18° .. +162°).
    rim_center_rad = math.radians(-90.0 + 180.0 * p_rot)
    rim_half_span = math.radians(72.0)
    for lw, al in [(18.0, 0.22), (7.5, 0.48), (3.2, 0.98)]:
        ctx.new_path()
        ctx.arc(0, 0, r, rim_center_rad - rim_half_span, rim_center_rad + rim_half_span)
        ctx.set_line_width(lw)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_source_rgba(red_rgb[0], red_rgb[1], red_rgb[2], al * s.alpha)
        ctx.stroke()

    # Inner red glow gradient inside the active half of the CD track
    gx_in = math.cos(rim_center_rad) * r * 0.72
    gy_in = math.sin(rim_center_rad) * r * 0.72
    ig = cairo.RadialGradient(gx_in, gy_in, 10, gx_in, gy_in, r * 0.85)
    ig.add_color_stop_rgba(0.0, red_rgb[0], red_rgb[1], red_rgb[2], 0.36 * s.alpha)
    ig.add_color_stop_rgba(1.0, red_rgb[0], red_rgb[1], red_rgb[2], 0.0)
    ctx.arc(0, 0, r * 0.98, 0, 2 * math.pi)
    ctx.set_source(ig)
    ctx.fill()

    # 4. True Curved Circular Arc Persian Text on Top & Bottom Tracks
    top_phrase = el.get("top_text", "روی کاهش وزن اثر بذارن!")
    bot_phrase = el.get("bottom_text", "روی سلامت دهان و دندونتون")
    r_track = r * 0.68
    arc_sz = float(el.get("arc_size", 42))

    # Top arc: bright white when p_rot < 0.5, dark embossed charcoal when p_rot >= 0.5!
    if p_rot < 0.85:
        draw_curved_arc_text(ctx, top_phrase, 0, 0, r_track, size=arc_sz, font_role="subtitle",
                             color_hex="#FFFFFF", alpha=s.alpha * (1.0 - p_rot * 0.95),
                             bottom_arc=False, embossed=False)
    if p_rot > 0.15:
        draw_curved_arc_text(ctx, top_phrase, 0, 0, r_track, size=arc_sz, font_role="subtitle",
                             color_hex="#262830", alpha=s.alpha * p_rot,
                             bottom_arc=False, embossed=True)
        # Bottom arc: lights up in glowing golden-yellow as p_rot -> 1!
        draw_curved_arc_text(ctx, bot_phrase, 0, 0, r_track, size=arc_sz, font_role="subtitle",
                             color_hex=gold_hex, alpha=s.alpha * clamp((p_rot - 0.15) / 0.75),
                             bottom_arc=True, embossed=False)

    # 5. White Center Core with 2-Line Typographic Lockup ('بیشتر از' bold black + 'این که' ultra-thin below)
    cr = r * 0.42
    ctx.arc(0, 0, cr, 0, 2 * math.pi)
    ctx.set_source_rgba(0.98, 0.98, 0.99, 0.98 * s.alpha)
    ctx.fill()

    f_bold = "IRANYekanExtraBlack2.ttf"
    f_thin = "IRANYekanLight.ttf"
    draw_text(ctx, el.get("center_top", "بیشتر از"), 0, -cr * 0.20, size=float(el.get("center_size", 44)),
              color="#0F172A", alpha=s.alpha, weight=f_bold)
    draw_text(ctx, el.get("center_bot", "این کـه"), 0, cr * 0.28, size=float(el.get("center_sub_size", 38)),
              color="#1E293B", alpha=s.alpha * 0.95, weight=f_thin)

    ctx.restore()
    return (x - r * 1.1, y - r * 1.1, x + r * 1.1, y + r * 1.1)


@element("novadent_diamond_square", dur=0.68, enter="spring", sfx="pop")
def el_novadent_diamond_square(ctx, el, s):
    """
    Screenshot 2026-10-19 211936.png ('مورد اول / خشکی / دهان' + 45° Rotated Diamond Badge):
      * A tilted (-16°) rounded-square vector wireframe outline above the speaker's head.
      * Intentional border-breaking typography:
        - 'مورد اول' shifted right so the 'م' breaks across the right wireframe border!
        - 'خشکی' in massive 3D-shaded golden-amber calligraphy whose right and left tips
          break across the wireframe borders!
        - 'دهان' in crisp white below-left.
      * A 45° rotated rounded-square Diamond (◇) badge on the top-left corner with a double
        white/glass border and medical/3D illustration inside!
    """
    x, y = xy(el, s.t)
    r = float(val(el, "r", s.t, 158)) * s.scale
    tilt_deg = float(val(el, "angle", s.t, -16.0))
    gold_hex = el.get("accent", "#F59E0B")
    gold_rgb = _color(gold_hex)

    local_t = max(0.0, s.t - float(el["t"]))
    p_in = _cubic_ease_in_out(clamp(local_t / max(0.25, float(el.get("dur", 0.68)))))

    ctx.save()
    ctx.translate(x, y + s.dy)
    ctx.scale(max(0.15, p_in), max(0.15, p_in))

    # 1. Warm amber-gold radial ambient glow behind the square
    g = cairo.RadialGradient(0, 0, r * 0.1, 0, 0, r * 1.55)
    g.add_color_stop_rgba(0.0, gold_rgb[0], gold_rgb[1], gold_rgb[2], 0.34 * s.alpha)
    g.add_color_stop_rgba(0.6, gold_rgb[0], gold_rgb[1], gold_rgb[2], 0.12 * s.alpha)
    g.add_color_stop_rgba(1.0, gold_rgb[0], gold_rgb[1], gold_rgb[2], 0.0)
    ctx.set_source(g)
    ctx.arc(0, 0, r * 1.55, 0, 2 * math.pi)
    ctx.fill()

    # 2. Tilted Rounded-Square Wireframe Outline (-16°)
    ctx.save()
    ctx.rotate(math.radians(tilt_deg))
    pts = [(-r, -r), (r, -r), (r, r), (-r, r)]
    _rounded_poly_path(ctx, pts, radius=r * 0.28)
    for lw, col_s, al in [(12.0, gold_rgb, 0.22), (4.5, gold_rgb, 0.42), (2.2, (1.0, 0.96, 0.88), 0.94)]:
        ctx.set_line_width(lw)
        ctx.set_source_rgba(col_s[0], col_s[1], col_s[2], al * s.alpha)
        ctx.stroke_preserve()
    ctx.new_path()
    ctx.restore()

    # 3. 45° Rotated Diamond (◇) Rounded-Square Image Badge on Top-Left Corner
    dx_b, dy_b = -r * 0.82, -r * 0.32
    dr = r * 0.52
    ctx.save()
    ctx.translate(dx_b, dy_b)
    ctx.rotate(math.radians(45.0))
    d_pts = [(-dr, -dr), (dr, -dr), (dr, dr), (-dr, dr)]
    # Dark glass fill + clipped visual inside the 45° diamond
    _rounded_poly_path(ctx, d_pts, radius=dr * 0.28)
    ctx.set_source_rgba(0.14, 0.08, 0.09, 0.95 * s.alpha)
    ctx.fill_preserve()
    ctx.save()
    ctx.clip()
    ctx.rotate(math.radians(-45.0))
    sp_name = el.get("corner_sprite")
    if sp_name:
        if not sp_name.endswith(".png"):
            sp_name += ".png"
        _blit_3d(ctx, sp_name, 0, 0, dr * 2.1, alpha=s.alpha)
    else:
        # Procedural high-detail dry-tongue/oral medical close-up inside the diamond (matching Screenshot 211936!)
        tg = cairo.RadialGradient(-12, 8, 8, 0, 0, dr * 1.25)
        tg.add_color_stop_rgba(0.0, 0.92, 0.48, 0.54, 0.98 * s.alpha)
        tg.add_color_stop_rgba(0.65, 0.72, 0.26, 0.34, 0.96 * s.alpha)
        tg.add_color_stop_rgba(1.0, 0.32, 0.12, 0.16, 0.98 * s.alpha)
        ctx.save()
        ctx.rotate(math.radians(28.0))
        ctx.scale(1.0, 1.35)
        ctx.arc(0, 4, dr * 0.68, 0, 2 * math.pi)
        ctx.set_source(tg)
        ctx.fill()
        # Central dry fissure line + texture dots
        ctx.set_line_width(2.0)
        ctx.set_source_rgba(0.45, 0.14, 0.18, 0.65 * s.alpha)
        ctx.move_to(0, -dr * 0.55)
        ctx.line_to(0, dr * 0.55)
        ctx.stroke()
        ctx.restore()
    ctx.restore()
    # Double white/glass border on the 45° diamond
    _rounded_poly_path(ctx, d_pts, radius=dr * 0.28)
    ctx.set_line_width(2.6)
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.92 * s.alpha)
    ctx.stroke()
    d_pts_in = [(-dr * 0.88, -dr * 0.88), (dr * 0.88, -dr * 0.88), (dr * 0.88, dr * 0.88), (-dr * 0.88, dr * 0.88)]
    _rounded_poly_path(ctx, d_pts_in, radius=dr * 0.24)
    ctx.set_line_width(1.0)
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.45 * s.alpha)
    ctx.stroke()
    ctx.restore()

    # 4. Border-Breaking 3-Font Persian Typography ('مورد اول' / 'خشکی' / 'دهان')
    top_txt = el.get("top", "مورد اول")
    hero_txt = el.get("title", "خشکی")
    bot_txt = el.get("bottom", "دهان")

    f_top = "IRANYekanLight.ttf"
    f_hero = "Delbar-Bold.ttf"
    f_bot = "KalamehFaNum-Bold.ttf"

    # 'مورد اول' shifted right (x = +r * 0.52) so 'م' intentionally crosses the right border line!
    draw_text(ctx, top_txt, r * 0.52, -r * 0.34, size=float(el.get("top_size", 46)),
              color="#FFFFFF", alpha=s.alpha, weight=f_top)

    # 'خشکی' in massive 3D-shaded golden calligraphy breaking across left & right borders
    h_sz = float(el.get("size", 112))
    for d_i in range(6, 0, -1):
        draw_text(ctx, hero_txt, r * 0.08 + d_i * 1.6, r * 0.14 + d_i * 2.2, size=h_sz,
                  color="#78350F", alpha=s.alpha * 0.55, weight=f_hero)
    draw_text(ctx, hero_txt, r * 0.08, r * 0.14, size=h_sz,
              color="#FBBF24", alpha=s.alpha, weight=f_hero)

    # 'دهان' near bottom-left inside the wireframe
    draw_text(ctx, bot_txt, -r * 0.24, r * 0.62, size=float(el.get("bot_size", 52)),
              color="#FFFFFF", alpha=s.alpha * 0.96, weight=f_bot)

    ctx.restore()
    return (x - r * 1.4, y - r * 1.2, x + r * 1.2, y + r * 1.2)


@element("novadent_chair_glow_text", dur=0.65, enter="fade", sfx="whoosh")
def el_novadent_chair_glow_text(ctx, el, s):
    """
    Screenshot 2026-10-19 212019.png ('ماهایی که / زدیم / چیکار / کنیم'):
    Placed behind the speaker & chair (`behind: "clip"`):
      * Darkens the left wall behind the speaker into a moody studio vignette.
      * Stacks 'ماهایی که' (white), giant golden-orange 'زدیم' (positioned so the right edge of 'ز'
        tucks behind the black chair backrest / shoulder silhouette!), 'چیکار', and 'کنیم'.
      * Casts a warm golden radial glow onto the background wall and chair edge!
    """
    x, y = xy(el, s.t)
    local_t = max(0.0, s.t - float(el["t"]))
    p_in = _cubic_ease_in_out(clamp(local_t / max(0.25, float(el.get("dur", 0.65)))))
    gold_rgb = _color(el.get("accent", "#F59E0B"))

    ctx.save()
    ctx.translate(x, y + s.dy)

    # 1. Moody dark studio vignette behind the text + warm golden light emission around 'زدیم'
    vg = cairo.RadialGradient(15, 0, 20, 15, 0, 420)
    vg.add_color_stop_rgba(0.0, 0.04, 0.04, 0.06, 0.84 * s.alpha * p_in)
    vg.add_color_stop_rgba(0.7, 0.04, 0.04, 0.06, 0.45 * s.alpha * p_in)
    vg.add_color_stop_rgba(1.0, 0.0, 0.0, 0.0, 0.0)
    ctx.set_source(vg)
    ctx.arc(15, 0, 420, 0, 2 * math.pi)
    ctx.fill()

    g = cairo.RadialGradient(45, -5, 10, 25, -5, 230)
    g.add_color_stop_rgba(0.0, gold_rgb[0], gold_rgb[1], gold_rgb[2], 0.68 * s.alpha * p_in)
    g.add_color_stop_rgba(0.55, gold_rgb[0], gold_rgb[1] * 0.65, 0.04, 0.28 * s.alpha * p_in)
    g.add_color_stop_rgba(1.0, 0.0, 0.0, 0.0, 0.0)
    ctx.set_source(g)
    ctx.arc(25, -5, 230, 0, 2 * math.pi)
    ctx.fill()

    f_reg = "IRANYekanLight.ttf"
    f_hero = "IRANYekanExtraBlack2.ttf"

    p1 = _cubic_ease_in_out(clamp(local_t / 0.35))
    p2 = _cubic_ease_in_out(clamp((local_t - 0.10) / 0.38))
    p3 = _cubic_ease_in_out(clamp((local_t - 0.22) / 0.38))

    draw_text(ctx, el.get("line1", "ماهایی که"), 18, -110 + (1 - p1) * 18, size=float(el.get("size1", 62)),
              color="#FFFFFF", alpha=s.alpha * p1, weight=f_reg)
    # Giant golden 'زدیم' positioned so 'زدیـم' is clearly readable and the right tip of 'ز' tucks behind the chair/shoulder!
    draw_text(ctx, el.get("hero", "زدیم"), 24, -8 + (1 - p2) * 18, size=float(el.get("hero_size", 122)),
              color="#F59E0B", alpha=s.alpha * p2, weight=f_hero)
    draw_text(ctx, el.get("line3", "چیکار"), -12, 82 + (1 - p3) * 18, size=float(el.get("size3", 62)),
              color="#FFFFFF", alpha=s.alpha * p3, weight=f_reg)
    draw_text(ctx, el.get("line4", "کنیم"), -32, 138 + (1 - p3) * 18, size=float(el.get("size4", 46)),
              color="#E2E8F0", alpha=s.alpha * p3, weight=f_reg)

    ctx.restore()
    return (x - 180, y - 160, x + 180, y + 180)


@element("novadent_edge_light_spill", dur=0.65, enter="fade", sfx="none")
def el_novadent_edge_light_spill(ctx, el, s):
    """Foreground additive/screen optical light wrap from a behind-chair glowing word onto the chair/shoulder edge!"""
    x, y = xy(el, s.t)
    r = float(val(el, "r", s.t, 190)) * s.scale
    rgb = _color(el.get("color", "#F59E0B"))
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_SCREEN)
    g = cairo.RadialGradient(x, y + s.dy, r * 0.08, x, y + s.dy, r)
    g.add_color_stop_rgba(0.0, rgb[0], rgb[1], rgb[2], 0.52 * s.alpha)
    g.add_color_stop_rgba(0.55, rgb[0], rgb[1] * 0.65, 0.05, 0.22 * s.alpha)
    g.add_color_stop_rgba(1.0, 0.0, 0.0, 0.0, 0.0)
    ctx.set_source(g)
    ctx.arc(x, y + s.dy, r, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()
    return None


@element("novadent_red_halo", dur=0.70, enter="spring", sfx="hit")
def el_novadent_red_halo(ctx, el, s):
    """
    Screenshot 2026-10-19 212032.png ('اصلاً سخت نیست' Concentric Two-Layer Red CD Halo):
    Behind the speaker & chair (`behind: "clip"`):
      * Darkens the room backdrop into moody charcoal-black (`#0B0C10`).
      * Draws a 2-layer concentric red CD halo directly behind the speaker's head & chair:
        - Outer dark-crimson ring (`r_outer = 320`, `#3B070B` fill + bright neon-red outer stroke `#FF2E3B`
          + wide crimson atmospheric wall glow),
        - Inner vivid-crimson disc (`r_inner = 215`, `#DC1E28` fill),
        - Inside the upper half of the inner disc (above hairline): 'اصلاً' + giant stretched calligraphic
          'سـخـت' + 'نیست'!
    """
    W, H = (s.scene.W, s.scene.H) if s.scene else (1080, 1920)
    x, y = xy(el, s.t)
    r_out = float(val(el, "r_outer", s.t, 325)) * s.scale
    r_in = float(val(el, "r_inner", s.t, 218)) * s.scale
    local_t = max(0.0, s.t - float(el["t"]))
    p_in = _cubic_ease_in_out(clamp(local_t / max(0.25, float(el.get("dur", 0.70)))))
    sc = 0.72 + 0.28 * p_in

    ctx.save()
    # 1. Darken the background room behind the speaker & chair for dramatic contrast
    if el.get("darken_room", True):
        ctx.rectangle(0, 0, W, H)
        ctx.set_source_rgba(0.04, 0.04, 0.055, 0.76 * s.alpha * p_in)
        ctx.fill()

    ctx.translate(x, y + s.dy)
    ctx.scale(sc, sc)

    # 2. Wide crimson atmospheric wall glow radiating behind the halo
    wg = cairo.RadialGradient(0, 0, r_in * 0.4, 0, 0, r_out * 1.48)
    wg.add_color_stop_rgba(0.0, 0.95, 0.12, 0.18, 0.55 * s.alpha)
    wg.add_color_stop_rgba(0.68, 0.65, 0.06, 0.10, 0.28 * s.alpha)
    wg.add_color_stop_rgba(1.0, 0.0, 0.0, 0.0, 0.0)
    ctx.set_source(wg)
    ctx.arc(0, 0, r_out * 1.48, 0, 2 * math.pi)
    ctx.fill()

    # 3. Outer Layer: Dark Crimson Ring + Bright Neon-Red Rim Stroke
    ctx.arc(0, 0, r_out, 0, 2 * math.pi)
    ctx.set_source_rgba(0.26, 0.03, 0.05, 0.88 * s.alpha)
    ctx.fill_preserve()
    for lw, al in [(16.0, 0.24), (6.0, 0.52), (2.8, 0.96)]:
        ctx.set_line_width(lw)
        ctx.set_source_rgba(1.0, 0.18, 0.24, al * s.alpha)
        ctx.stroke_preserve()
    ctx.new_path()

    # 4. Inner Layer: Vivid Glowing Crimson Core Disc
    ig = cairo.RadialGradient(0, -r_in * 0.2, r_in * 0.1, 0, 0, r_in)
    ig.add_color_stop_rgba(0.0, 0.96, 0.14, 0.18, 0.96 * s.alpha)
    ig.add_color_stop_rgba(1.0, 0.78, 0.06, 0.10, 0.96 * s.alpha)
    ctx.arc(0, 0, r_in, 0, 2 * math.pi)
    ctx.set_source(ig)
    ctx.fill()

    # 5. Typography inside the upper half of the inner red core ('اصلاً' / 'سـخـت' / 'نیست')
    f_thin = resolve_font("document", "regular", "subtitle")
    f_callig = "Delbar-Bold.ttf" if "Delbar-Bold.ttf" in resolve_font("handwritten", "bold", "handwritten") else resolve_font("title", "black", "title")
    draw_text(ctx, el.get("top", "اصلاً"), r_in * 0.34, -r_in * 0.50, size=float(el.get("top_size", 48)),
              color="#FECDD3", alpha=s.alpha * 0.92, weight=f_thin)
    draw_text(ctx, el.get("title", "ســـــخت"), 0, -r_in * 0.24, size=float(el.get("size", 112)),
              color="#FFFFFF", alpha=s.alpha, weight=resolve_font("document", "bold", "title"))
    draw_text(ctx, el.get("bottom", "نیست"), -r_in * 0.46, r_in * 0.02, size=float(el.get("bot_size", 46)),
              color="#FFE4E6", alpha=s.alpha * 0.95, weight=f_thin)

    ctx.restore()
    return None


@element("novadent_glass_hex", dur=0.65, enter="fade", sfx="jelly")
def el_novadent_glass_hex(ctx, el, s):
    """
    Screenshot 2026-10-19 212043.png ('از آدامس‌های / بدون قند / برای تحریک بزاق استفاده کنید'):
      * Rounded Hexagon floating over the speaker's chest.
      * True Glassmorphic Backdrop-Blur interior (`_paint_backdrop_blur`) that visibly blurs the
        speaker's black vest and white shirt collar underneath!
      * Thin golden-yellow neon border (`#FBBF24`) + bright white-gold radial specular light bloom
        on the top vertex!
      * Opens and closes with silky Ease-In-Out (`_cubic_ease_in_out`) + staggered 3-line text reveal.
    """
    x, y = xy(el, s.t)
    r = float(val(el, "r", s.t, 182)) * s.scale
    tilt_deg = float(val(el, "angle", s.t, -6.0))
    gold_rgb = _color(el.get("accent", "#FBBF24"))

    local_t = max(0.0, s.t - float(el["t"]))
    p_open = _cubic_ease_in_out(clamp(local_t / max(0.25, float(el.get("dur", 0.55)))))
    if el.get("out") is not None:
        rem = float(el["out"]) - s.t
        if rem < 0.40:
            p_open *= _cubic_ease_in_out(clamp(rem / 0.40))

    r_cur = r * max(0.12, p_open)
    pts_local = [
        (r_cur * math.cos(math.radians(-90 + k * 60 + tilt_deg)),
         r_cur * math.sin(math.radians(-90 + k * 60 + tilt_deg)))
        for k in range(6)
    ]
    pts_world = [(x + px, y + s.dy + py) for (px, py) in pts_local]

    # 1. Real Glassmorphic Backdrop-Blur inside the rounded hexagon
    def _hex_world_path(c):
        _rounded_poly_path(c, pts_world, radius=r_cur * 0.22)

    _paint_backdrop_blur(ctx, _hex_world_path, x - r_cur, y + s.dy - r_cur, x + r_cur, y + s.dy + r_cur,
                         blur_k=39, tint_bgr=(14.0, 22.0, 28.0), keep_ratio=0.46, alpha=0.92 * s.alpha * p_open)

    ctx.save()
    ctx.translate(x, y + s.dy)

    # 2. Golden-yellow neon border around the rounded hexagon
    _rounded_poly_path(ctx, pts_local, radius=r_cur * 0.22)
    for lw, al in [(14.0, 0.18), (5.0, 0.42), (2.4, 0.96)]:
        ctx.set_line_width(lw)
        ctx.set_source_rgba(gold_rgb[0], gold_rgb[1], gold_rgb[2], al * s.alpha * p_open)
        ctx.stroke_preserve()
    ctx.new_path()

    # 3. Signature Bright White-Gold Specular Light Bloom on the Top Vertex!
    top_vx, top_vy = pts_local[0]
    sg = cairo.RadialGradient(top_vx, top_vy, 4, top_vx, top_vy, r_cur * 0.58)
    sg.add_color_stop_rgba(0.0, 1.0, 0.98, 0.88, 0.72 * s.alpha * p_open)
    sg.add_color_stop_rgba(0.45, gold_rgb[0], gold_rgb[1], gold_rgb[2], 0.28 * s.alpha * p_open)
    sg.add_color_stop_rgba(1.0, 1.0, 1.0, 1.0, 0.0)
    ctx.set_source(sg)
    ctx.arc(top_vx, top_vy, r_cur * 0.58, 0, 2 * math.pi)
    ctx.fill()

    # 4. Staggered 3-Line Persian Typography inside the Glass Hexagon
    f_top = "IRANYekanLight.ttf"
    f_mid = "IRANYekanExtraBlack2.ttf"
    f_bot = "IRANYekanThin.ttf"

    p_txt = _cubic_ease_in_out(clamp((local_t - 0.12) / 0.42)) * p_open
    if p_txt > 0.02:
        dy_t = (1.0 - p_txt) * 16.0
        draw_text(ctx, el.get("top", "از آدامس‌های"), r_cur * 0.14, -r_cur * 0.22 + dy_t,
                  size=float(el.get("top_size", 32)), color="#E2E8F0", alpha=s.alpha * p_txt, weight=f_top)
        draw_text(ctx, el.get("title", "بدون قند"), 0, r_cur * 0.06 + dy_t,
                  size=float(el.get("size", 58)), color="#FFFFFF", alpha=s.alpha * p_txt, weight=f_mid)
        draw_text(ctx, el.get("bottom", "برای تحریک بزاق استفاده کنید"), 0, r_cur * 0.34 + dy_t,
                  size=float(el.get("bot_size", 24)), color="#CBD5E1", alpha=s.alpha * p_txt * 0.92, weight=f_bot)

    ctx.restore()
    return (x - r_cur * 1.1, y - r_cur * 1.1, x + r_cur * 1.1, y + r_cur * 1.1)


@element("orbiting_title_3d", dur=1.2, enter="fade", sfx="whoosh")
def el_orbiting_title_3d(ctx, el, s):
    """
    3D Head-Orbiting & Bird-Swoop Title Motion:
      * mode = 'orbit_head_3d': Title orbits in 3D perspective around the speaker's head!
        When paired with a sibling element having `behind: "clip"` and `"half": "back"`,
        the back half of the orbit passes BEHIND the speaker's head (`sin(theta) < 0`) and
        the front half passes IN FRONT of the speaker (`sin(theta) >= 0`)!
      * mode = 'bird_swoop_to_sub': Title emerges from behind the speaker's shoulder, banks
        along a graceful 3D Bezier arc around the speaker with silky Ease-In-Out velocity,
        and settles into the lower-third subtitle slot!
    """
    x, y = xy(el, s.t)
    mode = el.get("mode", "orbit_head_3d")
    txt = el.get("text", "موشن سه‌بعدی چرخان")
    fnt = resolve_font(el.get("font", "handwritten"), "bold", "title")
    size = float(el.get("size", 68))
    color_hex = el.get("color", "#FBBF24")
    local_t = max(0.0, s.t - float(el["t"]))

    if mode == "bird_swoop_to_sub":
        dur_sw = float(el.get("swoop_dur", 1.15))
        u = _cubic_ease_in_out(clamp(local_t / max(0.2, dur_sw)))
        # Cubic Bezier from behind-head (x+220, y-420) -> wide right bank (x+380, y-80) -> lower-third (x, y+320)
        p0 = (x + 180, y - 420)
        p1 = (x + 440, y - 160)
        p2 = (x - 320, y + 180)
        p3 = (x, y + 320)
        om = 1.0 - u
        bx = (om**3)*p0[0] + 3*(om**2)*u*p1[0] + 3*om*(u**2)*p2[0] + (u**3)*p3[0]
        by = (om**3)*p0[1] + 3*(om**2)*u*p1[1] + 3*om*(u**2)*p2[1] + (u**3)*p3[1]
        # Bank angle proportional to horizontal velocity
        bank_deg = math.sin(u * math.pi) * (-18.0)
        sc_u = 0.55 + 0.45 * u
        ctx.save()
        ctx.translate(bx, by)
        ctx.rotate(math.radians(bank_deg))
        ctx.scale(sc_u, sc_u)
        draw_text(ctx, txt, 0, 0, size=size, color=color_hex, alpha=s.alpha, weight=fnt, style="extrude_3d")
        ctx.restore()
        return (bx - 220, by - 60, bx + 220, by + 60)

    # mode == "orbit_head_3d"
    rx = float(el.get("rx", 310))
    ry = float(el.get("ry", 95))
    speed = float(el.get("speed", 2.2))
    theta = local_t * speed + float(el.get("phase", 0.0))
    depth_z = math.sin(theta)  # -1 = behind head, +1 = in front of head
    half_filter = el.get("half", "all")  # "back" | "front" | "all"
    if half_filter == "back" and depth_z > 0.05:
        return None
    if half_filter == "front" and depth_z <= 0.05:
        return None

    ox = x + rx * math.cos(theta)
    oy = y + ry * depth_z
    persp_sc = 0.74 + 0.28 * ((depth_z + 1.0) * 0.5)
    tilt_deg = -math.sin(theta) * 12.0

    ctx.save()
    ctx.translate(ox, oy)
    ctx.rotate(math.radians(tilt_deg))
    ctx.scale(persp_sc, persp_sc)
    draw_text(ctx, txt, 0, 0, size=size, color=color_hex,
              alpha=s.alpha * (0.65 + 0.35 * ((depth_z + 1.0) * 0.5)),
              weight=fnt, style="extrude_3d")
    ctx.restore()
    return (ox - 200, oy - 60, ox + 200, oy + 60)


@element("wireframe_badge", dur=0.65, enter="spring", sfx="pop")
def el_wireframe_badge(ctx, el, s):
    """
    Video 0 (@novadent.ir) Signature Element:
    A delicate, tilted glowing neon wireframe geometric outline (shape = 'squircle' | 'octagon' | 'hexagon' | 'circle' | 'filament')
    with multi-font stacked Persian typography inside (e.g. 'مورد اول' + giant calligraphic 'خشکی' + 'دهان')
    and an optional circular floating 3D/cutout badge pinned to its top-left corner!
    Zero opaque box fill — 100% compliant with the Zero-Box Law!
    """
    x, y = xy(el, s.t)
    r = float(val(el, "r", s.t, 155)) * s.scale
    shape = el.get("shape", "squircle")  # "squircle" | "octagon" | "hexagon" | "circle" | "filament"
    angle_deg = float(val(el, "angle", s.t, 11.0 if shape == "squircle" else 0.0))
    accent_hex = el.get("accent") or "#F59E0B"
    rgb = _color(accent_hex)
    ink_hex = el.get("text_color") or "#FFFFFF"

    local_t = max(0.0, s.t - float(el["t"]))
    sp = spring(min(1.0, local_t / max(0.25, float(el.get("dur", 0.65)))))
    sc = max(0.15, sp)

    ctx.save()
    ctx.translate(x, y + s.dy)

    # 1. Soft radial warm/accent light bloom behind the wireframe
    g = cairo.RadialGradient(0, 0, r * 0.1, 0, 0, r * 1.45)
    g.add_color_stop_rgba(0.0, rgb[0], rgb[1], rgb[2], 0.28 * s.alpha)
    g.add_color_stop_rgba(0.6, rgb[0], rgb[1], rgb[2], 0.08 * s.alpha)
    g.add_color_stop_rgba(1.0, rgb[0], rgb[1], rgb[2], 0.0)
    ctx.set_source(g)
    ctx.arc(0, 0, r * 1.45, 0, 2 * math.pi)
    ctx.fill()

    # 2. Geometric wireframe path (rotated by angle_deg, while text inside stays horizontal!)
    ctx.save()
    ctx.scale(sc, sc)
    ctx.rotate(math.radians(angle_deg))

    if shape == "octagon":
        pts = [(r * math.cos(math.radians(22.5 + k * 45)), r * math.sin(math.radians(22.5 + k * 45))) for k in range(8)]
        _rounded_poly_path(ctx, pts, radius=r * 0.14)
    elif shape == "hexagon":
        pts = [(r * math.cos(math.radians(k * 60)), r * math.sin(math.radians(k * 60))) for k in range(6)]
        _rounded_poly_path(ctx, pts, radius=r * 0.16)
    elif shape == "circle":
        ctx.new_path()
        ctx.arc(0, 0, r, 0, 2 * math.pi)
    elif shape == "filament":
        # Warm glowing tungsten filament curve behind speaker's head (Video 0 00:00-00:03)
        ctx.new_path()
        ctx.move_to(-r * 1.5, -r * 0.5)
        ctx.curve_to(-r * 0.6, r * 0.85, r * 0.6, r * 0.85, r * 1.5, -r * 0.4)
    else:  # "squircle" (tilted rounded square)
        pts = [(-r, -r), (r, -r), (r, r), (-r, r)]
        _rounded_poly_path(ctx, pts, radius=r * 0.34)

    # Multi-pass neon wireframe stroke
    for lw, al in [(14.0, 0.14), (5.5, 0.36), (2.2, 0.96)]:
        ctx.set_line_width(lw)
        ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], al * s.alpha)
        ctx.stroke_preserve()
    ctx.new_path()
    ctx.restore()

    # 3. Optional circular corner cutout badge (pinned at top-left of wireframe, like Video 0 00:10 / 00:20 / 00:32)
    corner_sprite = el.get("corner_sprite") or el.get("sprite")
    if corner_sprite:
        if not corner_sprite.endswith(".png"):
            corner_sprite += ".png"
        bx, by = -r * 0.82 * sc, -r * 0.48 * sc
        br = r * 0.36 * sc
        ctx.save()
        ctx.arc(bx, by, br, 0, 2 * math.pi)
        ctx.set_source_rgba(0.06, 0.07, 0.10, 0.88 * s.alpha)
        ctx.fill_preserve()
        ctx.set_line_width(2.0)
        ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], 0.9 * s.alpha)
        ctx.stroke()
        ctx.restore()
        _blit_3d(ctx, corner_sprite, bx, by, br * 1.75, alpha=s.alpha)

    # 4. Stacked Multi-Font Persian Typography inside the wireframe
    top_txt = el.get("top") or el.get("sub")
    hero_txt = el.get("title") or el.get("text")
    bot_txt = el.get("bottom") or el.get("unit")

    f_top = resolve_font(el.get("top_font", "subtitle"), "bold", "subtitle")
    f_hero = resolve_font(el.get("font", "handwritten"), "bold", "handwritten")
    f_bot = resolve_font(el.get("bot_font", "title"), "black", "title")

    if top_txt:
        draw_text(ctx, top_txt, 0, -r * 0.46, size=float(el.get("top_size", 32)),
                  color=ink_hex, alpha=s.alpha * 0.9, weight=f_top)
    if hero_txt:
        # Soft colored neon bloom behind the calligraphic hero word
        draw_text(ctx, hero_txt, 0, 6 if top_txt else 0, size=float(el.get("size", 74)),
                  color=accent_hex, alpha=s.alpha * 0.38, weight=f_hero)
        draw_text(ctx, hero_txt, 0, 4 if top_txt else -2, size=float(el.get("size", 72)),
                  color=accent_hex, alpha=s.alpha, weight=f_hero)
    if bot_txt:
        draw_text(ctx, bot_txt, 0, r * 0.48, size=float(el.get("bot_size", 36)),
                  color=ink_hex, alpha=s.alpha * 0.95, weight=f_bot)

    ctx.restore()
    return (x - r * 1.3, y - r * 1.3, x + r * 1.3, y + r * 1.3)


@element("circular_dial", dur=0.75, enter="spring", sfx="whoosh")
def el_circular_dial(ctx, el, s):
    """
    Video 0 (@novadent.ir 00:04-00:08) Signature Element:
    Chest-level translucent dark-glass circular radar dial with:
      - Soft crimson/amber radial backlight glow illuminating the speaker's chest,
      - Glowing white center disc with primary keyword (e.g. 'بیشتر'),
      - Sweeping colored radar sector + outer orbital ring,
      - Top & bottom curved arc callout text that rotates into focus!
    """
    x, y = xy(el, s.t)
    r = float(val(el, "r", s.t, 195)) * s.scale
    accent_hex = el.get("accent") or "#EF4444"
    rgb = _color(accent_hex)
    center_txt = el.get("center_text") or el.get("text") or "بیشتر"
    top_arc_txt = el.get("top_text") or "از اینکه روی کاهش وزن"
    bot_arc_txt = el.get("bottom_text") or "روی سلامت دهان و دندونتون"
    flip_at = float(el.get("flip_at", float(el["t"]) + 1.4))

    p_flip = ease_in_out(clamp((s.t - flip_at) / 0.65))

    ctx.save()
    ctx.translate(x, y + s.dy)

    # 1. Ambient radial backlight glow on speaker's chest
    bg_g = cairo.RadialGradient(0, 0, r * 0.2, 0, 0, r * 1.65)
    bg_g.add_color_stop_rgba(0.0, rgb[0], rgb[1], rgb[2], 0.48 * s.alpha)
    bg_g.add_color_stop_rgba(0.65, rgb[0], rgb[1], rgb[2], 0.16 * s.alpha)
    bg_g.add_color_stop_rgba(1.0, rgb[0], rgb[1], rgb[2], 0.0)
    ctx.set_source(bg_g)
    ctx.arc(0, 0, r * 1.65, 0, 2 * math.pi)
    ctx.fill()

    # 2. Dark translucent circular dial body
    ctx.arc(0, 0, r, 0, 2 * math.pi)
    ctx.set_source_rgba(0.04, 0.04, 0.06, 0.78 * s.alpha)
    ctx.fill_preserve()
    ctx.set_line_width(2.0)
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.28 * s.alpha)
    ctx.stroke()

    # 3. Sweeping active radar sector (rotates from top -90° to bottom +90° at flip_at)
    ang_center = math.radians(-90.0 + 180.0 * p_flip)
    span = math.radians(115.0)
    ctx.move_to(0, 0)
    ctx.arc(0, 0, r * 0.96, ang_center - span * 0.5, ang_center + span * 0.5)
    ctx.close_path()
    sec_g = cairo.RadialGradient(0, 0, r * 0.35, 0, 0, r * 0.96)
    sec_g.add_color_stop_rgba(0.0, rgb[0], rgb[1], rgb[2], 0.05 * s.alpha)
    sec_g.add_color_stop_rgba(1.0, rgb[0], rgb[1], rgb[2], 0.52 * s.alpha)
    ctx.set_source(sec_g)
    ctx.fill()

    # 4. Top & bottom arc phrases (curved effect via gentle arc placement + alpha emphasis)
    f_arc = resolve_font("subtitle", "bold", "subtitle")
    top_alpha = s.alpha * (1.0 - 0.55 * p_flip)
    bot_alpha = s.alpha * (0.35 + 0.65 * p_flip)
    draw_text(ctx, top_arc_txt, 0, -r * 0.64, size=float(el.get("arc_size", 30)),
              color="#FFFFFF", alpha=top_alpha, weight=f_arc)
    draw_text(ctx, bot_arc_txt, 0, r * 0.64, size=float(el.get("arc_size", 30)),
              color="#FFD60A" if p_flip > 0.5 else "#FFFFFF", alpha=bot_alpha, weight=f_arc)

    # 5. Glowing white center disc + hero word
    cr = r * 0.42
    ctx.arc(0, 0, cr, 0, 2 * math.pi)
    ctx.set_source_rgba(0.98, 0.98, 0.99, 0.96 * s.alpha)
    ctx.fill()
    f_c = resolve_font("title", "black", "title")
    draw_text(ctx, center_txt, 0, 4, size=float(el.get("center_size", 46)),
              color="#0C0E14", alpha=s.alpha, weight=f_c)

    ctx.restore()
    return (x - r * 1.1, y - r * 1.1, x + r * 1.1, y + r * 1.1)


@element("glass_hud_frame", dur=0.65, enter="spring", sfx="whoosh")
def el_glass_hud_frame(ctx, el, s):
    """
    Video 1 (00:10-00:12) Signature Element:
    A translucent glass square frame held/tilted by the speaker's hands, with kinetic text
    running along its top, bottom, and vertical side borders!
    """
    x, y = xy(el, s.t)
    w = float(val(el, "w", s.t, 520)) * s.scale
    h = float(val(el, "h", s.t, 520)) * s.scale
    tilt_deg = float(val(el, "angle", s.t, -8.5 * math.sin(max(0.0, s.t - float(el["t"])) * 2.6)))
    accent_hex = el.get("accent") or "#38BDF8"
    rgb = _color(accent_hex)

    ctx.save()
    ctx.translate(x, y + s.dy)
    ctx.rotate(math.radians(tilt_deg))

    # Translucent glass border + corner ticks
    hw, hh = w * 0.5, h * 0.5
    pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    _rounded_poly_path(ctx, pts, radius=16.0)
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.05 * s.alpha)
    ctx.fill_preserve()
    ctx.set_line_width(3.0)
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.78 * s.alpha)
    ctx.stroke()

    f_txt = resolve_font("title", "bold", "title")
    sz = float(el.get("size", 38))
    if el.get("top_text"):
        draw_text(ctx, el["top_text"], 0, -hh - 26, size=sz, color="#FFFFFF", alpha=s.alpha, weight=f_txt)
    if el.get("right_text"):
        ctx.save()
        ctx.translate(hw - 28, 0)
        ctx.rotate(math.radians(-90))
        draw_text(ctx, el["right_text"], 0, 0, size=sz * 0.9, color=accent_hex, alpha=s.alpha, weight=f_txt)
        ctx.restore()
    if el.get("bottom_text"):
        draw_text(ctx, el["bottom_text"], 0, hh - 28, size=sz, color="#FFFFFF", alpha=s.alpha, weight=f_txt)

    ctx.restore()
    return (x - w * 0.6, y - h * 0.6, x + w * 0.6, y + h * 0.6)


@element("zigzag_path_3d", dur=1.2, enter="fade", sfx="pen")
def el_zigzag_path_3d(ctx, el, s):
    """
    Video 2 (@mh3ydari 00:16-00:20) Signature Element:
    A glowing 3D emerald zig-zag ribbon climbing/descending on a dark coordinate grid with a
    rolling 3D white sphere that illuminates milestone labels at each vertex!
    """
    x, y = xy(el, s.t)
    w = float(val(el, "w", s.t, 420)) * s.scale
    h = float(val(el, "h", s.t, 1080)) * s.scale
    items = el.get("items") or ["پیدا کردن مشتری", "قیمت‌گذاری", "مدیریت پروژه", "انتخاب مسیر"]
    accent_hex = el.get("accent") or "#10B981"
    rgb = _color(accent_hex)
    dur_walk = float(el.get("walk_dur", 2.8))
    p = clamp((s.t - float(el["t"])) / max(0.3, dur_walk))

    n_verts = max(4, len(items) * 2)
    verts = []
    for i in range(n_verts):
        vy = (y - h * 0.45) + (h * 0.90) * (i / (n_verts - 1))
        vx = x + (w * 0.36 if i % 2 == 0 else -w * 0.12)
        verts.append((vx, vy))

    ctx.save()
    # 1. 3D Extruded depth shadow of the zig-zag ribbon
    for d_idx in range(8, 0, -1):
        ctx.new_path()
        for i, (vx, vy) in enumerate(verts):
            if i == 0:
                ctx.move_to(vx + d_idx * 2.2, vy + d_idx * 3.0)
            else:
                ctx.line_to(vx + d_idx * 2.2, vy + d_idx * 3.0)
        ctx.set_line_width(34.0)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_source_rgba(rgb[0] * 0.25, rgb[1] * 0.25, rgb[2] * 0.25, 0.55 * s.alpha)
        ctx.stroke()

    # 2. Glowing top neon face of the zig-zag ribbon
    for lw, al in [(58.0, 0.16), (34.0, 0.92)]:
        ctx.new_path()
        for i, (vx, vy) in enumerate(verts):
            if i == 0:
                ctx.move_to(vx, vy)
            else:
                ctx.line_to(vx, vy)
        ctx.set_line_width(lw)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], al * s.alpha)
        ctx.stroke()

    # 3. Milestone labels at vertices
    f_lbl = resolve_font("title", "black", "title")
    for idx, lbl in enumerate(items):
        vi = min(len(verts) - 1, idx * 2 + 1)
        vx, vy = verts[vi]
        v_prog = clamp((p * len(items)) - idx * 0.75)
        if v_prog > 0.05:
            lx = x - w * 0.72
            draw_text(ctx, lbl, lx, vy, size=float(el.get("label_size", 50)),
                      color="#FFFFFF" if v_prog > 0.6 else "#94A3B8",
                      alpha=s.alpha * v_prog, weight=f_lbl, style="extrude_3d")

    # 4. Rolling 3D white sphere along the zig-zag path
    seg_f = p * (len(verts) - 1)
    si = min(len(verts) - 2, int(seg_f))
    sf = seg_f - si
    bx = lerp(verts[si][0], verts[si + 1][0], sf)
    by = lerp(verts[si][1], verts[si + 1][1], sf)
    br = 34.0 * s.scale
    # Contact shadow
    ctx.arc(bx + 8, by + 12, br, 0, 2 * math.pi)
    ctx.set_source_rgba(0, 0, 0, 0.45 * s.alpha)
    ctx.fill()
    # 3D shaded sphere
    sg = cairo.RadialGradient(bx - br * 0.35, by - br * 0.35, br * 0.1, bx, by, br)
    sg.add_color_stop_rgba(0.0, 1.0, 1.0, 1.0, s.alpha)
    sg.add_color_stop_rgba(0.7, 0.88, 0.92, 0.95, s.alpha)
    sg.add_color_stop_rgba(1.0, 0.62, 0.68, 0.74, s.alpha)
    ctx.arc(bx, by, br, 0, 2 * math.pi)
    ctx.set_source(sg)
    ctx.fill()

    ctx.restore()
    return (x - w * 0.8, y - h * 0.5, x + w * 0.6, y + h * 0.5)


@element("avatar_carousel_3d", dur=0.9, enter="spring", sfx="whoosh")
def el_avatar_carousel_3d(ctx, el, s):
    """
    Video 1 (00:22-00:24) Signature Element:
    A 3D curved ribbon/train of rounded-square profile cards flying along an S-curve arc
    above and behind the speaker's head!
    """
    x, y = xy(el, s.t)
    n_cards = int(el.get("count", 9))
    sprites = el.get("sprites") or ["logo_apple_3d.png", "logo_surface_3d.png", "logo_hp_3d.png",
                                    "logo_dell_3d.png", "laptop_3d.png", "grade_a_3d.png", "shield_3d.png"]
    local_t = max(0.0, s.t - float(el["t"]))

    ctx.save()
    for i in range(n_cards):
        u = ((i / max(1, n_cards - 1)) - 0.15 + local_t * 0.28) % 1.15
        if not (0.0 <= u <= 1.0):
            continue
        cx_i = x - 440 + u * 880
        cy_i = y + math.sin(u * math.pi * 1.65) * 120.0
        sc_i = (0.62 + 0.52 * math.sin(u * math.pi)) * s.scale
        sz_i = 128.0 * sc_i
        yaw_i = (u - 0.5) * 52.0
        sp_name = sprites[i % len(sprites)]
        _blit_3d_perspective(ctx, sp_name, cx_i, cy_i, sz_i, alpha=s.alpha * math.sin(u * math.pi),
                             yaw_deg=yaw_i, pitch_deg=8.0, shadow=True)
    ctx.restore()
    return (x - 480, y - 180, x + 480, y + 180)


@element("silhouette_scribble", dur=0.6, enter="fade", sfx="pop")
def el_silhouette_scribble(ctx, el, s):
    """
    Video 2 (@mh3ydari 00:08-00:12) Signature Element:
    Draws glowing white sunglasses + an animated chaotic white scribble ('کلاف سردرگم ذهنی')
    over the speaker's head silhouette!
    """
    x, y = xy(el, s.t)
    r = float(val(el, "r", s.t, 95)) * s.scale
    seed_phase = int(s.t * 12.0)  # 12fps boil animation like hand-drawn scribble!
    rng = np.random.default_rng(seed_phase + 42)

    ctx.save()
    ctx.translate(x, y + s.dy)

    # 1. Animated chaotic brain scribble above forehead
    ctx.set_line_width(2.6)
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.95 * s.alpha)
    ctx.new_path()
    n_pts = 38
    for i in range(n_pts):
        ang = i * 0.65
        rx = (r * 0.85) * math.cos(ang) + float(rng.uniform(-18, 18))
        ry = -r * 0.55 + (r * 0.52) * math.sin(ang * 1.3) + float(rng.uniform(-14, 14))
        if i == 0:
            ctx.move_to(rx, ry)
        else:
            ctx.line_to(rx, ry)
    ctx.stroke()

    # 2. Glowing white silhouette sunglasses
    if el.get("glasses", True):
        gy = r * 0.25
        gw, gh = r * 0.62, r * 0.36
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.98 * s.alpha)
        # Left lens
        _rounded_poly_path(ctx, [(-gw * 1.15, gy - gh * 0.5), (-gw * 0.15, gy - gh * 0.5),
                                 (-gw * 0.22, gy + gh * 0.5), (-gw * 1.05, gy + gh * 0.5)], radius=8)
        ctx.fill()
        # Right lens
        _rounded_poly_path(ctx, [(gw * 0.15, gy - gh * 0.5), (gw * 1.15, gy - gh * 0.5),
                                 (gw * 1.05, gy + gh * 0.5), (gw * 0.22, gy + gh * 0.5)], radius=8)
        ctx.fill()
        # Bridge
        ctx.rectangle(-gw * 0.2, gy - gh * 0.35, gw * 0.4, gh * 0.22)
        ctx.fill()

    ctx.restore()
    return (x - r * 1.2, y - r * 1.2, x + r * 1.2, y + r * 0.8)


def compose_before_after_frame(raw_bgr: np.ndarray, after_bgr: np.ndarray,
                               before_label: str = "BEFORE", after_label: str = "AFTER") -> np.ndarray:
    """
    Video 0 (@novadent.ir) Master Layout Composer:
    Composites a raw ungraded frame (`BEFORE` card on left, tilted -6.5°, desaturated) and the
    edited Visual-DNA frame (`AFTER` card on right, upright, larger, with specular border & drop shadow)
    over a warm, heavily blurred atmospheric dark studio backdrop!
    """
    H, W = after_bgr.shape[:2]

    # 1. Warm atmospheric blurred background derived from after_bgr
    small = cv2.resize(after_bgr, (W // 8, H // 8))
    blur_bg = cv2.GaussianBlur(small, (25, 25), 10.0)
    bg = cv2.resize(blur_bg, (W, H)).astype(np.float32) * 0.28
    # Add warm amber top-left & teal bottom-right ambient studio glow
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g_warm = np.exp(-((xx - W * 0.22) ** 2 + (yy - H * 0.20) ** 2) / (2.0 * (W * 0.42) ** 2))
    bg[:, :, 2] += g_warm * 58.0  # R
    bg[:, :, 1] += g_warm * 32.0  # G
    bg[:, :, 0] += g_warm * 8.0   # B
    canvas = np.clip(bg, 0, 255).astype(np.uint8)

    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)

    # Upload base canvas to Cairo (BGR -> BGRA)
    bgra_base = np.dstack([canvas, np.full((H, W), 255, dtype=np.uint8)])
    base_surf = cairo.ImageSurface.create_for_data(bgra_base, cairo.FORMAT_ARGB32, W, H)
    ctx.set_source_surface(base_surf, 0, 0)
    ctx.paint()

    def _draw_video_card(im_bgr, cx, cy, cw, ch, angle_deg, label, is_after=False):
        # Resize image to (cw, ch)
        im_r = cv2.resize(im_bgr, (int(cw), int(ch)), interpolation=cv2.INTER_AREA)
        if not is_after:
            # Slightly desaturate & flatten the BEFORE card so the contrast is unmistakable
            gray = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)
            im_r = cv2.addWeighted(im_r, 0.55, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.45, -8)
        card_bgra = np.dstack([im_r, np.full((int(ch), int(cw)), 255, dtype=np.uint8)])
        c_surf = cairo.ImageSurface.create_for_data(card_bgra, cairo.FORMAT_ARGB32, int(cw), int(ch))

        ctx.save()
        ctx.translate(cx, cy)
        ctx.rotate(math.radians(angle_deg))

        hw, hh = cw * 0.5, ch * 0.5
        pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        rad = 38.0 if is_after else 28.0

        # Deep drop shadow behind card
        for s_off, s_al in [(26.0, 0.18), (14.0, 0.34), (6.0, 0.52)]:
            ctx.save()
            ctx.translate(-8 if is_after else 0, s_off)
            _rounded_poly_path(ctx, pts, radius=rad)
            ctx.set_source_rgba(0, 0, 0, s_al)
            ctx.fill()
            ctx.restore()

        # Clip and paint video frame inside rounded card
        _rounded_poly_path(ctx, pts, radius=rad)
        ctx.save()
        ctx.clip()
        ctx.set_source_surface(c_surf, -hw, -hh)
        ctx.paint()
        ctx.restore()

        # Crisp specular rim border around card
        _rounded_poly_path(ctx, pts, radius=rad)
        ctx.set_line_width(3.2 if is_after else 2.2)
        if is_after:
            rim = cairo.LinearGradient(-hw, -hh, hw, hh)
            rim.add_color_stop_rgba(0.0, 1.0, 0.86, 0.55, 0.92)
            rim.add_color_stop_rgba(0.5, 1.0, 1.0, 1.0, 0.45)
            rim.add_color_stop_rgba(1.0, 0.25, 0.88, 0.78, 0.88)
            ctx.set_source(rim)
        else:
            ctx.set_source_rgba(1.0, 1.0, 1.0, 0.55)
        ctx.stroke()

        # Label ('BEFORE' / 'AFTER') cleanly above top-left corner of the card
        f_lbl = resolve_font("counter", "black", "title")
        draw_text(ctx, label, -hw + (68 if is_after else 58), -hh - 28,
                  size=34 if is_after else 26, color="#FFFFFF", alpha=0.95 if is_after else 0.80, weight=f_lbl)
        ctx.restore()

    # Draw BEFORE card on left (tilted -6.5°) and AFTER card on right (upright, overlapping)
    _draw_video_card(raw_bgr, cx=W * 0.255, cy=H * 0.555, cw=W * 0.36, ch=H * 0.36,
                     angle_deg=-6.5, label=before_label, is_after=False)
    _draw_video_card(after_bgr, cx=W * 0.635, cy=H * 0.505, cw=W * 0.60, ch=H * 0.60,
                     angle_deg=0.0, label=after_label, is_after=True)

    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape((H, W, 4))
    return buf[:, :, :3].copy()


def render_before_after_video(raw_mp4: str, after_mp4: str, out_mp4: str,
                              before_label: str = "BEFORE", after_label: str = "AFTER"):
    """Render a full @novadent.ir style BEFORE / AFTER split-card showcase video with audio from after_mp4."""
    import subprocess
    if not os.path.exists(raw_mp4):
        raise FileNotFoundError(f"Raw video not found: {raw_mp4}")
    if not os.path.exists(after_mp4):
        raise FileNotFoundError(f"After video not found: {after_mp4}")
    cap_raw = cv2.VideoCapture(raw_mp4)
    cap_aft = cv2.VideoCapture(after_mp4)
    fps = cap_aft.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap_aft.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
    H = int(cap_aft.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920
    n_aft = int(cap_aft.get(cv2.CAP_PROP_FRAME_COUNT))
    n_raw = max(1, int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT)))

    tmp_v = out_mp4 + ".novid.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}",
        "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", tmp_v
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last_raw = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(n_aft):
        ok_a, fr_a = cap_aft.read()
        if not ok_a:
            break
        ok_r, fr_r = cap_raw.read()
        if ok_r:
            last_raw = fr_r
        comp = compose_before_after_frame(last_raw, fr_a, before_label=before_label, after_label=after_label)
        proc.stdin.write(comp.tobytes())
    proc.stdin.close()
    proc.wait()
    cap_raw.release()
    cap_aft.release()

    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", tmp_v, "-i", after_mp4,
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", out_mp4
    ], check=True)
    if os.path.exists(tmp_v):
        os.remove(tmp_v)
    return out_mp4
