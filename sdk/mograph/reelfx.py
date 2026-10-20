"""
mograph.reelfx — v13 "Visual-DNA & Creative Director" (Zero-Box Edition).

Implements the Visual-DNA Motion & Procedural Asset Engine aesthetic laws:
  * A1. Engineered Calm & Full-Environment Gridded Backdrops: Procedural 0.6px technical coordinate
    grids with '+' intersection tick marks, diffused radial key-fill lighting, and f/2.8 Depth-of-Field
    speaker defocus (`dof_blur`) — completely replacing ugly rectangular boxes!
  * A2. Tactile Volume & True 3D Perspective Homography Spin (`_blit_3d_perspective`): All hero objects
    (`logo_apple_3d.png`, `logo_surface_3d.png`, `logo_hp_3d.png`, `logo_dell_3d.png`, `laptop_3d.png`,
    `grade_a_3d.png`, `grade_b_3d.png`, `grade_c_3d.png`, `plane_3d.png`, `pin_3d.png`, `pin_green_3d.png`,
    `cash_3d.png`, `warn_3d.png`, `shield_3d.png`) are isolated 3D tactile renders that rotate in 3D space
    with real perspective homography and soft SSAO contact shadows.
  * A3. One Thread of Color: Procedural parametric vector rails, coordinate arcs, and callout tracks.
  * A4. Multi-Font Kinetic Typography Hierarchy: Massive 3D display weights (`KalamehFaNum-Black`,
    `PelakFA-Black`) paired with expressive handwritten/calligraphic fonts (`DigiGhasedak-Bold`,
    `Delbar-Bold`) and mechanical coordinate labels — floating freely in space with ZERO stiff boxes!
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

import cairo
import cv2
import numpy as np

from .core import clamp, color as _color, ease_in_out, ease_out, get_ease, lerp, prog, spring
from .elements import element, xy
from .geometry import compile_path, rounded_rect_d
from .text import draw_text, parse_markup, resolve_font, text_bbox, to_fa_digits

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS_3D = os.path.join(_HERE, "assets_3d")


@lru_cache(maxsize=32)
def _load_3d_rgba(name: str):
    """Load raw unpremultiplied BGRA uint8 array from mograph/assets_3d/."""
    path = os.path.join(_ASSETS_3D, name)
    if not os.path.exists(path):
        return None
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    return im


@lru_cache(maxsize=32)
def _load_3d_sprite(name: str):
    im = _load_3d_rgba(name)
    if im is None:
        return None
    h, w = im.shape[:2]
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    a = im[:, :, 3:4].astype(np.float32) / 255.0
    bgra[:, :, 0] = np.clip(im[:, :, 0:1] * a, 0, 255)[:, :, 0].astype(np.uint8)
    bgra[:, :, 1] = np.clip(im[:, :, 1:2] * a, 0, 255)[:, :, 0].astype(np.uint8)
    bgra[:, :, 2] = np.clip(im[:, :, 2:3] * a, 0, 255)[:, :, 0].astype(np.uint8)
    bgra[:, :, 3] = im[:, :, 3]
    surf = cairo.ImageSurface.create_for_data(bgra, cairo.FORMAT_ARGB32, w, h)
    return (surf, w, h, bgra)


def _blit_3d(ctx, name: str, cx: float, cy: float, target_size: float, alpha: float = 1.0, rot_rad: float = 0.0):
    sp = _load_3d_sprite(name)
    if sp is None or alpha <= 0.005 or target_size <= 1.0:
        return False
    surf, w, h, _ = sp
    sc = target_size / max(w, h)
    ctx.save()
    ctx.translate(cx, cy)
    if rot_rad:
        ctx.rotate(rot_rad)
    ctx.scale(sc, sc)
    ctx.set_source_surface(surf, -w * 0.5, -h * 0.5)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint_with_alpha(clamp(alpha))
    ctx.restore()
    return True


def _blit_3d_perspective(ctx, name: str, cx: float, cy: float, target_size: float, alpha: float = 1.0,
                         yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0,
                         shadow: bool = True, glow_hex: str | None = None):
    """
    Renders a tactile 3D RGBA sprite with TRUE 3D Perspective Homography Rotation (yaw, pitch, roll),
    multi-slice 3D thickness extrusion during spin, optional ambient glow halo, and soft SSAO contact shadow.
    """
    im = _load_3d_rgba(name)
    if im is None or alpha <= 0.005 or target_size <= 2.0:
        return False
    a = clamp(alpha)

    # 1. Optional soft ambient glow halo behind the 3D object
    if glow_hex:
        gc = _color(glow_hex)
        r_gl = target_size * 0.68
        g = cairo.RadialGradient(cx, cy, r_gl * 0.08, cx, cy, r_gl)
        g.add_color_stop_rgba(0.0, gc[0], gc[1], gc[2], 0.42 * a)
        g.add_color_stop_rgba(0.55, gc[0], gc[1], gc[2], 0.14 * a)
        g.add_color_stop_rgba(1.0, gc[0], gc[1], gc[2], 0.0)
        ctx.save()
        ctx.arc(cx, cy, r_gl, 0, math.pi * 2)
        ctx.set_source(g)
        ctx.fill()
        ctx.restore()

    # 2. Soft SSAO elliptical contact shadow underneath the floating 3D object
    if shadow:
        sh_rx = target_size * 0.42 * max(0.35, math.cos(math.radians(yaw_deg)))
        sh_ry = target_size * 0.11
        sh_y = cy + target_size * 0.48
        ctx.save()
        ctx.translate(cx, sh_y)
        ctx.scale(sh_rx, sh_ry)
        sg = cairo.RadialGradient(0, 0, 0.05, 0, 0, 1.0)
        sg.add_color_stop_rgba(0.0, 0, 0, 0, 0.34 * a)
        sg.add_color_stop_rgba(0.6, 0, 0, 0, 0.14 * a)
        sg.add_color_stop_rgba(1.0, 0, 0, 0, 0.0)
        ctx.arc(0, 0, 1.0, 0, math.pi * 2)
        ctx.set_source(sg)
        ctx.fill()
        ctx.restore()

    if abs(yaw_deg) < 1.0 and abs(pitch_deg) < 1.0:
        return _blit_3d(ctx, name, cx, cy, target_size, alpha=a, rot_rad=math.radians(roll_deg))

    # 3. Compute 3D perspective homography matrix on a resized working buffer
    h0, w0 = im.shape[:2]
    work_sz = min(380, max(160, int(target_size * 1.15)))
    sc0 = work_sz / max(w0, h0)
    ws, hs = max(8, int(w0 * sc0)), max(8, int(h0 * sc0))
    small = cv2.resize(im, (ws, hs), interpolation=cv2.INTER_AREA)

    pad = int(max(ws, hs) * 0.28)
    canvas = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    ch, cw = canvas.shape[:2]
    cx_i, cy_i = cw * 0.5, ch * 0.5

    yaw = math.radians(clamp(yaw_deg, -78.0, 78.0))
    pitch = math.radians(clamp(pitch_deg, -65.0, 65.0))
    roll = math.radians(roll_deg)

    cy_w, sy_w = math.cos(yaw), math.sin(yaw)
    cp_w, sp_w = math.cos(pitch), math.sin(pitch)
    cr_w, sr_w = math.cos(roll), math.sin(roll)

    # 3D Rotation matrix R = Rz(roll) * Rx(pitch) * Ry(yaw)
    corners = np.array([
        [-ws * 0.5, -hs * 0.5, 0.0],
        [ ws * 0.5, -hs * 0.5, 0.0],
        [ ws * 0.5,  hs * 0.5, 0.0],
        [-ws * 0.5,  hs * 0.5, 0.0],
    ], dtype=np.float32)

    src_pts = np.array([
        [pad, pad],
        [pad + ws, pad],
        [pad + ws, pad + hs],
        [pad, pad + hs],
    ], dtype=np.float32)

    focal = float(max(ws, hs) * 2.2)
    dst_pts = []
    for x3, y3, z3 in corners:
        # Ry
        x1 = x3 * cy_w + z3 * sy_w
        z1 = -x3 * sy_w + z3 * cy_w
        y1 = y3
        # Rx
        y2 = y1 * cp_w - z1 * sp_w
        z2 = y1 * sp_w + z1 * cp_w
        x2 = x1
        # Rz
        x_f = x2 * cr_w - y2 * sr_w
        y_f = x2 * sr_w + y2 * cr_w
        p_scale = focal / max(focal * 0.35, focal + z2)
        dst_pts.append([cx_i + x_f * p_scale, cy_i + y_f * p_scale])
    dst_pts = np.array(dst_pts, dtype=np.float32)

    H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(canvas, H_mat, (cw, ch), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Premultiply BGRA for Cairo
    wa = warped[:, :, 3:4].astype(np.float32) / 255.0
    bgra = np.empty((ch, cw, 4), dtype=np.uint8)
    bgra[:, :, 0] = np.clip(warped[:, :, 0:1] * wa, 0, 255)[:, :, 0].astype(np.uint8)
    bgra[:, :, 1] = np.clip(warped[:, :, 1:2] * wa, 0, 255)[:, :, 0].astype(np.uint8)
    bgra[:, :, 2] = np.clip(warped[:, :, 2:3] * wa, 0, 255)[:, :, 0].astype(np.uint8)
    bgra[:, :, 3] = warped[:, :, 3]

    # Extruded 3D thickness slices when turning in 3D space
    dx_ext = -math.sin(yaw) * 12.0
    dy_ext = math.sin(pitch) * 8.0
    if abs(dx_ext) > 1.2 or abs(dy_ext) > 1.2:
        dark_bgra = bgra.copy()
        dark_bgra[:, :, :3] = (dark_bgra[:, :, :3].astype(np.uint16) * 35 // 100).astype(np.uint8)
        surf_dark = cairo.ImageSurface.create_for_data(dark_bgra, cairo.FORMAT_ARGB32, cw, ch)
        disp_sc = target_size / max(ws, hs)
        for s_idx in range(3, 0, -1):
            frac = s_idx / 3.0
            ctx.save()
            ctx.translate(cx + dx_ext * frac, cy + dy_ext * frac)
            ctx.scale(disp_sc, disp_sc)
            ctx.set_source_surface(surf_dark, -cw * 0.5, -ch * 0.5)
            ctx.get_source().set_filter(cairo.FILTER_GOOD)
            ctx.paint_with_alpha(a * 0.75)
            ctx.restore()

    surf = cairo.ImageSurface.create_for_data(bgra, cairo.FORMAT_ARGB32, cw, ch)
    disp_sc = target_size / max(ws, hs)
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(disp_sc, disp_sc)
    ctx.set_source_surface(surf, -cw * 0.5, -ch * 0.5)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint_with_alpha(a)
    ctx.restore()
    return True


def _draw_dna_grid_environment(ctx, cx: float, cy: float, accent_hex: str = "#ff2d55", alpha: float = 1.0,
                               lt: float = 0.0, full_bleed: bool = True):
    """
    Visual-DNA Law A1 & A3: Replaces busy/ugly backgrounds with a cohesive, color-graded studio environment
    featuring a procedural 0.6px technical coordinate grid, '+' intersection tick marks, diffused radial
    key-fill illumination, and One Thread of Color orbital tracks — ZERO rectangular boxes!
    """
    W, H = 1080.0, 1920.0
    ctx.save()
    ac = _color(accent_hex)

    if full_bleed:
        # 1. Full-environment studio backdrop pass behind the segmented speaker
        g_bg = cairo.LinearGradient(0, 0, 0, H * 0.85)
        g_bg.add_color_stop_rgba(0.0, 0.04, 0.05, 0.08, 0.94 * alpha)
        g_bg.add_color_stop_rgba(0.55, 0.05, 0.06, 0.10, 0.82 * alpha)
        g_bg.add_color_stop_rgba(1.0, 0.05, 0.06, 0.10, 0.0)
        ctx.rectangle(0, 0, W, H * 0.85)
        ctx.set_source(g_bg)
        ctx.fill()

    # 2. Diffused studio radial key-fill spotlight matching the accent color
    rad = cairo.RadialGradient(cx, cy + 80, 30, cx, cy + 80, 620)
    rad.add_color_stop_rgba(0.0, ac[0], ac[1], ac[2], 0.46 * alpha)
    rad.add_color_stop_rgba(0.45, ac[0], ac[1], ac[2], 0.18 * alpha)
    rad.add_color_stop_rgba(1.0, ac[0], ac[1], ac[2], 0.0)
    ctx.rectangle(0, 0, W, H)
    ctx.set_source(rad)
    ctx.fill()

    # 3. Procedural Visual-DNA Technical Coordinate Grid (0.7px lines + '+' tick marks at intersections)
    cell = 90.0
    drift_y = (lt * 18.0) % cell
    ctx.set_line_width(0.8)
    for gx in np.arange(90.0, W, cell):
        dist_x = abs(gx - cx) / (W * 0.5)
        ga = max(0.0, (1.0 - dist_x * 0.75)) * 0.18 * alpha
        ctx.set_source_rgba(1.0, 1.0, 1.0, ga)
        ctx.move_to(gx, 0)
        ctx.line_to(gx, H * 0.72)
        ctx.stroke()

    for gy in np.arange(40.0 + drift_y, H * 0.70, cell):
        dist_y = abs(gy - (cy + 40)) / (H * 0.45)
        ga = max(0.0, (1.0 - dist_y * 0.85)) * 0.18 * alpha
        ctx.set_source_rgba(1.0, 1.0, 1.0, ga)
        ctx.move_to(40, gy)
        ctx.line_to(W - 40, gy)
        ctx.stroke()
        # Draw crisp '+' coordinate ticks at every other grid intersection
        for gx in np.arange(90.0, W, cell * 2):
            ta = max(0.0, (1.0 - math.hypot(gx - cx, gy - cy) / 620.0)) * 0.45 * alpha
            if ta > 0.03:
                ctx.set_source_rgba(ac[0], ac[1], ac[2], ta)
                ctx.set_line_width(1.5)
                ctx.move_to(gx - 6, gy); ctx.line_to(gx + 6, gy)
                ctx.move_to(gx, gy - 6); ctx.line_to(gx, gy + 6)
                ctx.stroke()
                ctx.set_line_width(0.8)

    # 4. One Thread of Color — Precision Coordinate Rings & Caliper Arcs behind speaker
    for idx_r, r_ring in enumerate((240.0, 370.0)):
        pulse = r_ring + 10.0 * math.sin(lt * 2.8 - idx_r)
        ctx.arc(cx, cy + 160, pulse, math.radians(195), math.radians(345))
        ctx.set_source_rgba(ac[0], ac[1], ac[2], (0.42 - idx_r * 0.14) * alpha)
        ctx.set_line_width(2.2 if idx_r == 0 else 1.4)
        if idx_r == 1:
            ctx.set_dash([12.0, 10.0], -lt * 35.0)
        else:
            ctx.set_dash([])
        ctx.stroke()
    ctx.set_dash([])
    ctx.restore()


def _draw_glow_underline(ctx, cx: float, y: float, w: float, color_hex: str, progress: float, alpha: float = 1.0):
    """Draws a sleek glowing parametric tapered brush/caliper line under a hero title (Zero Box!)."""
    if progress <= 0.01 or alpha <= 0.01:
        return
    cur_w = w * ease_out(clamp(progress))
    c = _color(color_hex)
    ctx.save()
    g = cairo.LinearGradient(cx - cur_w * 0.5, y, cx + cur_w * 0.5, y)
    g.add_color_stop_rgba(0.0, c[0], c[1], c[2], 0.0)
    g.add_color_stop_rgba(0.2, c[0], c[1], c[2], 0.95 * alpha)
    g.add_color_stop_rgba(0.5, 1.0, 1.0, 1.0, 0.98 * alpha)
    g.add_color_stop_rgba(0.8, c[0], c[1], c[2], 0.95 * alpha)
    g.add_color_stop_rgba(1.0, c[0], c[1], c[2], 0.0)
    ctx.set_source(g)
    ctx.set_line_width(5.0)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.move_to(cx - cur_w * 0.5, y)
    ctx.line_to(cx + cur_w * 0.5, y)
    ctx.stroke()
    # Center diamond node on the thread
    ctx.arc(cx, y, 5.5, 0, math.pi * 2)
    ctx.set_source_rgba(1.0, 1.0, 1.0, alpha)
    ctx.fill()
    ctx.restore()


def _S(ctx, kx, ky=None):
    ky = kx if ky is None else ky
    ctx.scale(kx if abs(kx) > 1e-3 else 1e-3, ky if abs(ky) > 1e-3 else 1e-3)


def _fa(s, on=True):
    return to_fa_digits(s) if on else s


def _anchor_t(v, el):
    return float(v) if isinstance(v, (int, float)) else float(el.get("t", 0))


# ---------------------------------------------------------------------------
@element("title_card", dur=0.5, enter="none", sfx="none")
def el_title_card(ctx, el, s):
    """
    v13 Zero-Box Visual-DNA Title Composition:
    Replaces rectangular boxes with a full-bleed Gridded Studio Environment behind the segmented speaker,
    multi-font kinetic typography (`DigiGhasedak-Bold` handwritten accent + `KalamehFaNum-Black` 3D extruded
    display headline), a glowing parametric caliper thread, and 3D perspective-rotating tactile assets!
    """
    x, y = xy(el, s.t)
    lt = s.lt
    a = s.alpha * (1 - s.out_p)
    if a <= 0.003:
        return None
    raw = el.get("text", "")
    txt, emph = parse_markup(raw)
    size = float(el.get("size", 118))
    fnt = resolve_font(el.get("font", "title"), None, "title")
    hand_fnt = resolve_font(el.get("sub_font", "handwritten"), None, "handwritten")
    out_k = 1 - ease_in_out(s.out_p)
    accent_col = el.get("slab", "#ff2d55")

    # 1. Render the Visual-DNA Technical Coordinate Grid & Studio Lighting Environment behind the speaker
    _draw_dna_grid_environment(ctx, x, 240, accent_hex=accent_col, alpha=a * out_k, lt=lt, full_bleed=bool(el.get("behind", True)))

    # 2. Expressive Handwritten Top Lead-In (`DigiGhasedak-Bold`) floating freely without any box
    if el.get("sub"):
        sp_sub = spring(lt, 0.02, stiffness=380, damping=15)
        ctx.save()
        ctx.translate(x, 66)
        _S(ctx, 0.75 + 0.25 * sp_sub, 0.75 + 0.25 * sp_sub)
        draw_text(ctx, el["sub"], 0, 0, size=size * 0.50, color="#ffd60a",
                  alpha=a * out_k * clamp(lt / 0.16), weight=hand_fnt)
        ctx.restore()

    # 3. Massive 3D-Extruded Display Headline (`KalamehFaNum-Black`) floating in gridded space
    col = el.get("color", "#ffffff")
    ec = el.get("emph_color", "#ffd60a")
    per = float(el.get("word_gap", 0.06))

    def ws_b(k):
        wt = 0.04 + k * per
        sp = spring(lt, wt, stiffness=520, damping=16) if lt > wt else 0.0
        return {"color": ec if k in emph else col, "alpha": clamp((lt - wt) / 0.05), "scale": 0.62 + 0.38 * sp}

    draw_text(ctx, txt, x, 146, size=size, weight=fnt, alpha=a * out_k, word_style=ws_b,
              max_width=940, line_height=1.10, style="extrude_3d")

    # 4. One Thread of Color — glowing parametric caliper line under the 3D headline
    _draw_glow_underline(ctx, x, 206, 620, accent_col, lt / 0.38, alpha=a * out_k)

    # 5. Tactile 3D Studio Objects rotating in 3D perspective on left & right behind speaker's shoulders
    left_sp_name = el.get("left_sprite", "laptop_3d.png")
    right_sp_name = el.get("right_sprite", "cash_3d.png" if "سود" in txt else "shield_3d.png")
    flank_sp = spring(lt, 0.06, stiffness=340, damping=14) if lt > 0.06 else 0.0
    if flank_sp > 0.01:
        bob = 9.0 * math.sin(lt * 4.2)
        yaw_spin = (1.0 - flank_sp) * -75.0 + 16.0 * math.sin(lt * 3.0)
        pitch_tilt = 9.0 * math.cos(lt * 2.4)
        _blit_3d_perspective(ctx, left_sp_name, 205, 345 + bob, 245 * flank_sp, alpha=a * out_k,
                             yaw_deg=yaw_spin, pitch_deg=pitch_tilt, roll_deg=-6.0, shadow=True, glow_hex=accent_col)
        _blit_3d_perspective(ctx, right_sp_name, 875, 345 - bob, 235 * flank_sp, alpha=a * out_k,
                             yaw_deg=-yaw_spin, pitch_deg=-pitch_tilt, roll_deg=6.0, shadow=True, glow_hex="#ffd60a")

    return (x - 460, 30, x + 460, 460)


# ---------------------------------------------------------------------------
@element("logo_chip", dur=0.4, enter="none", sfx="none")
def el_logo_chip(ctx, el, s):
    """
    v13 Zero-Box 3D Spinning Brand Emblems:
    Renders the 4 isolated 3D glossy studio brand emblems (`logo_dell_3d.png`, `logo_hp_3d.png`,
    `logo_surface_3d.png`, `logo_apple_3d.png`) floating freely along a Visual-DNA coordinate rail
    and rotating in 3D perspective (`yaw_deg`, `pitch_deg`) with ZERO rectangular boxes!
    """
    x, y = xy(el, s.t)
    items = el.get("items") or []
    a_all = s.alpha * (1 - s.out_p)
    if a_all <= 0.003:
        return None
    lt = s.lt
    fnt = resolve_font(el.get("font", "title"), None, "title")

    # Subtle radial studio spotlight & Visual-DNA coordinate crosshairs behind the 3D logos
    _draw_dna_grid_environment(ctx, x, y - 40, accent_hex="#0a84ff", alpha=a_all * 0.55, lt=lt, full_bleed=False)

    sprite_map = {
        "dell": ("logo_dell_3d.png", "#0076ce"),
        "hp": ("logo_hp_3d.png", "#0096d6"),
        "surface": ("logo_surface_3d.png", "#38bdf8"),
        "macbook": ("logo_apple_3d.png", "#e2e8f0"),
        "apple": ("logo_apple_3d.png", "#e2e8f0"),
    }

    # Symmetrical 2x2 floating 3D stage layout (generous spacing, zero boxes!)
    positions = [
        (x + 215, y - 105),   # Top-Right: DELL
        (x - 215, y - 105),   # Top-Left: HP
        (x + 215, y + 155),   # Bottom-Right: Surface
        (x - 215, y + 155),   # Bottom-Left: MacBook
    ]

    # Procedural Visual-DNA coordinate connector lines between the 4 3D emblems
    rail_p = ease_out(clamp(lt / 0.65))
    if rail_p > 0.02:
        ctx.save()
        ctx.set_source_rgba(0.22, 0.74, 0.98, 0.32 * a_all * rail_p)
        ctx.set_line_width(1.6)
        ctx.set_dash([10.0, 8.0], -lt * 30.0)
        ctx.move_to(x - 215, y - 105); ctx.line_to(x + 215, y - 105)
        ctx.move_to(x - 215, y + 155); ctx.line_to(x + 215, y + 155)
        ctx.move_to(x, y - 180); ctx.line_to(x, y + 230)
        ctx.stroke()
        ctx.restore()

    for i, it in enumerate(items[:4]):
        cx_i, cy_i = positions[i]
        ti = _anchor_t(it.get("t"), el) if it.get("t") is not None else float(el["t"]) + 0.15 * i
        if s.t <= ti:
            continue
        dt = s.t - ti
        sp = spring(s.t, ti, stiffness=380, damping=14)
        a = a_all * clamp(dt / 0.10)
        bname = it["text"].strip().lower()
        sp_file, glow_c = sprite_map.get(bname, ("logo_dell_3d.png", "#0a84ff"))

        # True 3D perspective Y-axis spin on entry (-82 deg -> 0 deg) + continuous 3D orbital tilt!
        spin_decay = math.exp(-dt * 3.2)
        yaw_deg = -82.0 * spin_decay * math.cos(dt * 6.5) + 16.0 * math.sin(dt * 3.1 + i * 1.4)
        pitch_deg = 10.0 * math.cos(dt * 2.4 + i * 0.9)
        bob_y = 7.0 * math.sin(dt * 3.6 + i)

        logo_sz = 195.0 * clamp(sp, 0.0, 1.18)
        _blit_3d_perspective(ctx, sp_file, cx_i, cy_i + bob_y, logo_sz, alpha=a,
                             yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=0.0, shadow=True, glow_hex=glow_c)

        # Clean, box-free 3D extruded brand title floating below the spinning 3D emblem
        draw_text(ctx, it["text"], cx_i, cy_i + 112, size=46, color="#ffffff", alpha=a, weight=fnt, style="extrude_3d")

    return (x - 360, y - 220, x + 360, y + 260)


# ---------------------------------------------------------------------------
def _bez(p0, p1, p2, q):
    return ((1 - q) ** 2 * p0[0] + 2 * (1 - q) * q * p1[0] + q * q * p2[0],
            (1 - q) ** 2 * p0[1] + 2 * (1 - q) * q * p1[1] + q * q * p2[1])


@element("route_arc", dur=1.4, enter="none", sfx="none")
def el_route_arc(ctx, el, s):
    """
    v13 Zero-Box Visual-DNA Flight Path (`امارات` -> `ایران`):
    Parametric coordinate Bezier arc with calibration tick marks, 3D perspective-rotating GPS pins
    (`pin_3d.png`, `pin_green_3d.png`), 3D commercial airliner (`plane_3d.png`), and box-free 3D titles!
    """
    p0 = [float(v) for v in el.get("from", [880, 1180])]
    p2 = [float(v) for v in el.get("to", [200, 1180])]
    bend = float(el.get("bend", 0.28))
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    L = math.hypot(dx, dy) or 1
    p1 = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2 - L * bend)
    lt = s.lt
    a = s.alpha * (1 - s.out_p)
    dur = float(el.get("dur", 1.05))
    q = get_ease(el.get("ease", "in_out_cubic"))(clamp((lt - 0.15) / dur))
    col = el.get("color", "#ffd60a")
    w = float(el.get("width", 9))
    pin_r = float(el.get("pin_r", 46))

    # Subtle Visual-DNA coordinate grid in the flight zone
    _draw_dna_grid_environment(ctx, 540, (p0[1] + p1[1]) * 0.5, accent_hex=col, alpha=a * 0.48, lt=lt, full_bleed=False)

    pts = [_bez(p0, p1, p2, i / 80) for i in range(81)]
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    # Faint full coordinate trajectory + perpendicular calibration tick marks every 10 steps
    ctx.new_path(); ctx.move_to(*pts[0])
    for pt in pts[1:]:
        ctx.line_to(*pt)
    ctx.set_source_rgba(1, 1, 1, a * 0.24)
    ctx.set_line_width(2.0); ctx.set_dash([6.0, 8.0]); ctx.stroke()
    ctx.set_dash([])
    for ti_idx in range(8, 76, 8):
        px_a, py_a = pts[ti_idx - 1]
        px_b, py_b = pts[ti_idx + 1]
        ang_n = math.atan2(py_b - py_a, px_b - px_a) + math.pi * 0.5
        tx_c, ty_c = pts[ti_idx]
        ctx.move_to(tx_c - math.cos(ang_n) * 9, ty_c - math.sin(ang_n) * 9)
        ctx.line_to(tx_c + math.cos(ang_n) * 9, ty_c + math.sin(ang_n) * 9)
        ctx.set_source_rgba(1, 0.84, 0.04, a * (0.65 if ti_idx <= int(q * 80) else 0.20))
        ctx.set_line_width(2.0); ctx.stroke()

    k = int(q * 80)
    if k > 0:
        c = _color(col)
        ctx.new_path(); ctx.move_to(*pts[0])
        for pt in pts[1:k + 1]:
            ctx.line_to(*pt)
        ctx.set_source_rgba(c[0], c[1], c[2], c[3] * a * 0.36)
        ctx.set_line_width(w * 2.5); ctx.stroke()

        ctx.new_path(); ctx.move_to(*pts[0])
        for pt in pts[1:k + 1]:
            ctx.line_to(*pt)
        ctx.set_source_rgba(c[0], c[1], c[2], c[3] * a)
        ctx.set_line_width(w); ctx.set_dash([w * 1.6, w * 1.4], -lt * 45); ctx.stroke()
    ctx.restore()

    # 3D Origin & Destination GPS Pins with 3D perspective oscillation & radar rings
    for pt_pin, is_dest, t_appear, g_hex in ((p0, False, 0.0, "#ff2d55"), (p2, True, 0.20, "#10b981")):
        sp_p = spring(lt, t_appear, stiffness=400, damping=14) if lt > t_appear else 0.0
        if sp_p > 0.01:
            pulse_p = (lt * 1.5 + (0.4 if is_dest else 0.0)) % 1.0
            ctx.save()
            ctx.translate(pt_pin[0], pt_pin[1])
            ctx.scale(1.0, 0.45)
            rc = _color(g_hex)
            ctx.arc(0, 0, pin_r * (0.9 + 2.0 * pulse_p), 0, math.pi * 2)
            ctx.set_source_rgba(rc[0], rc[1], rc[2], a * (1.0 - pulse_p) * 0.85)
            ctx.set_line_width(4.0); ctx.stroke()
            ctx.restore()

            sp_name = "pin_green_3d.png" if is_dest else "pin_3d.png"
            yaw_p = (1.0 - sp_p) * 70.0 + 15.0 * math.sin(lt * 3.4 + (1.5 if is_dest else 0.0))
            _blit_3d_perspective(ctx, sp_name, pt_pin[0], pt_pin[1] - pin_r * 1.15 * sp_p, pin_r * 3.3 * sp_p,
                                 alpha=a, yaw_deg=yaw_p, pitch_deg=6.0, shadow=True, glow_hex=g_hex)

    # 3D Commercial Jet Airliner flying & banking along the curve
    if 0.01 < q <= 1.0:
        hx, hy = _bez(p0, p1, p2, q)
        tx, ty = _bez(p0, p1, p2, min(1.0, q + 0.02))
        ang = math.atan2(ty - hy, tx - hx) if q < 0.98 else math.atan2(p2[1] - pts[-4][1], p2[0] - pts[-4][0])
        rot_plane = math.degrees(ang) - 152.0
        sz = float(el.get("plane_size", 230)) * (0.88 + 0.24 * math.sin(math.pi * q))
        _blit_3d_perspective(ctx, "plane_3d.png", hx, hy - 16, sz, alpha=a,
                             yaw_deg=12.0 * math.sin(q * math.pi), pitch_deg=-8.0 * math.cos(q * math.pi),
                             roll_deg=rot_plane, shadow=True, glow_hex="#ffd60a")

    # Box-Free 3D Extruded Location Titles (`امارات` / `ایران`) + glowing underlines
    fs = float(el.get("size", 62))
    fnt = resolve_font("title", None, "title")
    for lab, pt, tt, acc_c in ((el.get("label_from"), p0, 0.06, "#ff2d55"),
                               (el.get("label_to"), p2, 0.22, "#10b981")):
        if lab and lt > tt:
            lp = spring(lt, tt, stiffness=380, damping=15)
            ly = pt[1] + fs * 1.18
            ctx.save()
            ctx.translate(pt[0], ly)
            _S(ctx, lp, lp)
            draw_text(ctx, lab, 0, 0, size=fs, color="#ffffff", alpha=a, weight=fnt, style="extrude_3d")
            _draw_glow_underline(ctx, 0, fs * 0.58, fs * 2.3, acc_c, (lt - tt) / 0.3, alpha=a)
            ctx.restore()

    return (min(p0[0], p2[0]) - 110, p1[1] - 90, max(p0[0], p2[0]) + 110, max(p0[1], p2[1]) + fs * 2.2)


# ---------------------------------------------------------------------------
@element("price_tag", dur=0.5, enter="none", sfx="none")
def el_price_tag(ctx, el, s):
    """
    v13 Zero-Box 3D Tactile Profit Counter (`+۸ میلیون` + `سود خالص هر لپ‌تاپ`):
    Completely eliminates the stiff rectangular box so text NEVER overflows!
    Pairs a massive free-floating 3D-extruded `PelakFA-Black` counter with an expressive
    `DigiGhasedak-Bold` handwritten subtitle, a glowing emerald caliper arc, and a 3D perspective-spinning
    stack of emerald banknotes & gold coins (`cash_3d.png`).
    """
    x, y = xy(el, s.t)
    lt = s.lt
    a = s.alpha * (1 - s.out_p)
    if a <= 0.003:
        return None
    sp = spring(lt, 0, stiffness=340, damping=14)
    cd = float(el.get("count_dur", 0.78))
    v0, v1 = float(el.get("from", 0)), float(el.get("value", 8))
    cq = get_ease("out_expo")(clamp((lt - 0.08) / cd))
    v = lerp(v0, v1, cq)
    fmt = el.get("fmt", "{:.0f}")
    num = el.get("prefix", "") + fmt.format(v) + el.get("suffix", "")
    num = _fa(num, el.get("fa", True))

    # 1. Soft emerald-gold Visual-DNA radial spotlight & coordinate crosshair behind the counter (Zero Box!)
    _draw_dna_grid_environment(ctx, x, y - 40, accent_hex="#10b981", alpha=a * 0.65, lt=lt, full_bleed=False)

    # 2. Tactile 3D Banknotes & Gold Coins (`cash_3d.png`) spinning in 3D perspective above/beside the counter
    yaw_cash = (1.0 - sp) * -80.0 + 18.0 * math.sin(lt * 3.2)
    pitch_cash = 10.0 * math.cos(lt * 2.5)
    _blit_3d_perspective(ctx, "cash_3d.png", x + 290, y - 25 + 8.0 * math.sin(lt * 4.0), 255.0 * sp,
                         alpha=a, yaw_deg=yaw_cash, pitch_deg=pitch_cash, shadow=True, glow_hex="#10b981")

    # 3. Massive Free-Floating 3D Extruded Number (`+۸ میلیون` in `PelakFA-Black` / `counter`)
    fnt = resolve_font("counter", None, "counter")
    hand_fnt = resolve_font("handwritten", None, "handwritten")
    ctx.save()
    ctx.translate(x - 85, y - 35)
    _S(ctx, 0.7 + 0.3 * sp, 0.7 + 0.3 * sp)
    draw_text(ctx, num, 0, 0, size=134, color="#ffd60a", alpha=a, weight=fnt, style="extrude_3d")
    ctx.restore()

    # 4. Glowing emerald caliper underline + Handwritten `DigiGhasedak-Bold` label underneath (100% unclipped!)
    _draw_glow_underline(ctx, x - 60, y + 48, 560, "#10b981", lt / 0.35, alpha=a)
    if el.get("label"):
        draw_text(ctx, el["label"], x - 60, y + 108, size=72, color="#ffffff", alpha=a * clamp((lt - 0.12) / 0.22),
                  weight=hand_fnt)

    return (x - 440, y - 160, x + 440, y + 165)


# ---------------------------------------------------------------------------
@element("grade_scale", dur=0.5, enter="none", sfx="none")
def el_grade_scale(ctx, el, s):
    """
    v13 Zero-Box Tactile 3D Grade Blocks (`grade_a_3d.png`, `grade_b_3d.png`, `grade_c_3d.png`):
    Replaces flat 2D rectangles with our 3 Octane-rendered tactile 3D frosted-glass & brushed-metal
    grade monuments (`A+`, `B`, `C`) floating along a procedural Visual-DNA coordinate rail!
    """
    x, y = xy(el, s.t)
    lt = s.lt
    a = s.alpha * (1 - s.out_p)
    if a <= 0.003:
        return None
    gs = el.get("grades") or [{"g": "A", "label": "در حد نو"}, {"g": "B", "label": "خط‌وخش جزئی"}, {"g": "C", "label": "کارکرده"}]
    gs = [g if isinstance(g, dict) else {"g": g} for g in gs]

    _draw_dna_grid_environment(ctx, x, y - 30, accent_hex="#10b981", alpha=a * 0.55, lt=lt, full_bleed=False)

    fnt = resolve_font("title", None, "title")
    hand_fnt = resolve_font("handwritten", None, "handwritten")
    if el.get("title"):
        tp = clamp(lt / 0.25)
        draw_text(ctx, el["title"], x, y - 215, size=78, color="#ffd60a",
                  alpha=a * tp, weight=hand_fnt)
        _draw_glow_underline(ctx, x, y - 165, 520, "#10b981", tp, alpha=a)

    # Procedural horizontal Visual-DNA coordinate rail connecting the 3 3D blocks
    w = float(el.get("w", 920))
    ctx.save()
    ctx.set_source_rgba(1, 1, 1, 0.28 * a)
    ctx.set_line_width(2.0)
    ctx.set_dash([10.0, 8.0], -lt * 30.0)
    ctx.move_to(x - w * 0.45, y); ctx.line_to(x + w * 0.45, y)
    ctx.stroke()
    ctx.restore()

    sprites_3d = ["grade_a_3d.png", "grade_b_3d.png", "grade_c_3d.png"]
    glows = ["#10b981", "#f59e0b", "#ef4444"]
    stars_map = ["★ ★ ★", "★ ★ ☆", "★ ☆ ☆"]
    pick = int(el.get("pick", 0))
    at = float(el.get("at", 0.75))
    n = len(gs)
    step_x = 315.0

    for i, g in enumerate(gs[:3]):
        ti = 0.10 + 0.14 * i
        if lt <= ti:
            continue
        dt = lt - ti
        sp = spring(lt, ti, stiffness=390, damping=14)
        cx_i = x + (1 - i) * step_x   # RTL: Grade A+ on the right (x + 315), B in middle (x), C on left (x - 315)
        sel = ease_out(clamp((lt - at) / 0.25)) if i == pick else 0.0
        dim = ease_out(clamp((lt - at) / 0.25)) if i != pick else 0.0
        item_a = a * (1.0 - 0.42 * dim)

        # 3D perspective rotation on entry + continuous 3D hover tilt
        yaw_i = (1.0 - sp) * -75.0 + (14.0 + 6.0 * sel) * math.sin(dt * 3.0 + i)
        pitch_i = 8.0 * math.cos(dt * 2.3 + i)
        sz_i = (240.0 + 48.0 * sel - 22.0 * dim) * clamp(sp, 0.0, 1.15)
        cy_i = y - 24.0 * sel + 6.0 * math.sin(dt * 3.5 + i)

        _blit_3d_perspective(ctx, sprites_3d[i], cx_i, cy_i, sz_i, alpha=item_a,
                             yaw_deg=yaw_i, pitch_deg=pitch_i, shadow=True, glow_hex=glows[i])

        # Floating star rating above the 3D block
        draw_text(ctx, stars_map[i], cx_i, cy_i - sz_i * 0.52, size=34, color="#ffd60a" if i == 0 else "#e2e8f0",
                  alpha=item_a, weight=resolve_font("ui", None, "ui"))

        # Free-floating Persian label below the 3D block (Zero Box!)
        if g.get("label"):
            draw_text(ctx, g["label"], cx_i, cy_i + sz_i * 0.54, size=42 + 6 * sel,
                      color="#ffd60a" if sel > 0.1 else "#ffffff", alpha=item_a, weight=fnt, style="extrude_3d")

        # Crown Grade A+ with the 3D Gold Shield Trophy (`shield_3d.png`)
        if sel > 0.01:
            _blit_3d_perspective(ctx, "shield_3d.png", cx_i + sz_i * 0.38, cy_i - sz_i * 0.38, 125.0 * sel,
                                 alpha=a * sel, yaw_deg=18.0 * math.sin(lt * 4.0), pitch_deg=6.0, shadow=False,
                                 glow_hex="#ffd60a")

    return (x - 470, y - 250, x + 470, y + 210)


# ---------------------------------------------------------------------------
@element("profit_meter", dur=0.6, enter="none", sfx="none")
def el_profit_meter(ctx, el, s):
    x, y = xy(el, s.t)
    lt = s.lt
    a = s.alpha * (1 - s.out_p)
    buy, sell = float(el.get("buy", 40)), float(el.get("sell", 48))
    w, h = float(el.get("w", 900)), float(el.get("h", 110))
    cols = el.get("colors") or ["#3a3a3c", "#19c37d", "#ffd60a"]
    labs = el.get("labels") or ["خرید", "فروش", "سود"]
    fa = el.get("fa", True)
    unit = el.get("unit", "")
    k = w / max(sell, buy)
    gb = get_ease("out_expo")(clamp((lt - 0.1) / 0.6))
    gs = get_ease("out_expo")(clamp((lt - 0.6) / 0.7))
    xr = x + w / 2
    fnt = resolve_font("counter", None, "counter")
    bfn = resolve_font("body", None, "body")
    for row, (val_, c, lab, g) in enumerate(((buy, cols[0], labs[0], gb), (sell, cols[1], labs[1], gs))):
        yy = y + (row - 0.5) * (h + 30)
        bw = val_ * k * g
        if bw > 1:
            c_rgba = _color(c)
            ctx.set_source_rgba(c_rgba[0], c_rgba[1], c_rgba[2], a)
            ctx.rectangle(xr - bw, yy - h / 2, bw, h); ctx.fill()
            draw_text(ctx, _fa(f"{val_ * g:.0f}{unit}", fa), xr - bw + 16, yy, size=h * 0.42, color="#ffffff",
                      alpha=a * clamp(g * 3), weight=fnt, align="left")
        draw_text(ctx, lab, xr - 18, yy, size=h * 0.32, color="#ffffff", alpha=a * clamp(g * 3), weight=bfn, align="right")
    return (x - w / 2, y - h - 30, x + w / 2, y + h + 120)


# ---------------------------------------------------------------------------
@element("sticker", dur=0.4, enter="none", sfx="none")
def el_sticker(ctx, el, s):
    """
    v13 Full-Screen Hero Statement over Blurred/Matte Speaker (`dof_blur` + Zero Box!):
    While `el_footage` softly blurs/mats the entire video frame (including the speaker),
    `el_sticker` turns the FULL 1080x1920 screen into a Visual-DNA kinetic composition:
    full-screen coordinate grid + glowing radial spotlight + massive 3D extruded headline
    (`نکته خیلی مهم!`) + expressive `DigiGhasedak-Bold` handwritten subtitle + 3D perspective-spinning
    warning triangles (`warn_3d.png`) — ZERO rectangular box!
    """
    x, y = xy(el, s.t)
    lt = s.lt
    a = s.alpha * (1 - s.out_p)
    if a <= 0.003:
        return None
    out_k = 1 - ease_in_out(s.out_p)
    sp = spring(lt, 0, stiffness=420, damping=14)
    size = float(el.get("size", 128))
    fnt = resolve_font(el.get("font", "title"), None, "title")
    hand_fnt = resolve_font("handwritten", None, "handwritten")
    txt = el.get("text", "")

    # 1. Full-screen Visual-DNA coordinate grid & golden-amber radial spotlight over the blurred speaker
    _draw_dna_grid_environment(ctx, x, y, accent_hex="#ffb020", alpha=a * out_k * 0.85, lt=lt, full_bleed=False)

    # 2. Large Tactile 3D Warning Triangle (`warn_3d.png`) spinning in 3D perspective above the title
    yaw_w = (1.0 - sp) * -85.0 + 18.0 * math.sin(lt * 3.8)
    pitch_w = 10.0 * math.cos(lt * 2.7)
    _blit_3d_perspective(ctx, "warn_3d.png", x, y - 245 + 8.0 * math.sin(lt * 4.5), 255.0 * sp,
                         alpha=a * out_k, yaw_deg=yaw_w, pitch_deg=pitch_w, shadow=True, glow_hex="#ffb020")

    # 3. Massive Full-Screen 3D Extruded Hero Headline (`نکته خیلی مهم!`) in the center of the screen (Zero Box!)
    ctx.save()
    ctx.translate(x, y - 20)
    _S(ctx, 0.72 + 0.28 * sp, 0.72 + 0.28 * sp)
    draw_text(ctx, txt, 0, 0, size=size, color="#ffd60a", alpha=a * out_k, weight=fnt, style="extrude_3d")
    ctx.restore()

    # 4. Glowing parametric caliper thread + Handwritten `DigiGhasedak-Bold` subtitle below
    _draw_glow_underline(ctx, x, y + 62, 680, "#ffb020", lt / 0.32, alpha=a * out_k)
    if el.get("sub"):
        sp_s = spring(lt, 0.10, stiffness=360, damping=15) if lt > 0.10 else 0.0
        ctx.save()
        ctx.translate(x, y + 135)
        _S(ctx, 0.75 + 0.25 * sp_s, 0.75 + 0.25 * sp_s)
        draw_text(ctx, el["sub"], 0, 0, size=size * 0.54, color="#ffffff",
                  alpha=a * out_k * clamp((lt - 0.08) / 0.18), weight=hand_fnt)
        ctx.restore()

    return (x - 460, y - 380, x + 460, y + 210)


# ---------------------------------------------------------------------------
@element("cta_arrow", dur=0.6, enter="none", sfx="none")
def el_cta_arrow(ctx, el, s):
    """v13 Zero-Box CTA: Free-floating handwritten/3D typography + glowing procedural Bezier arrow."""
    p0 = [float(v) for v in el.get("from", [760, 1180])]
    p2 = [float(v) for v in el.get("to", [700, 1460])]
    lt = s.lt
    a = s.alpha * (1 - s.out_p)
    cv = float(el.get("curve", -0.35))
    mx, my = (p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    p1 = (mx - dy * cv, my + dx * cv)
    d = float(el.get("dur", 0.5))
    q = get_ease("out_cubic")(clamp(lt / d))
    col = _color(el.get("color", "#ffd60a"))
    w = float(el.get("width", 9))
    pts = [_bez(p0, p1, p2, i / 40 * q) for i in range(41)]
    ctx.save()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND); ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    # Outer glow on arrow
    ctx.set_source_rgba(col[0], col[1], col[2], col[3] * a * 0.32); ctx.set_line_width(w * 2.4)
    ctx.move_to(*pts[0])
    for pt in pts[1:]:
        ctx.line_to(*pt)
    ctx.stroke()
    ctx.set_source_rgba(col[0], col[1], col[2], col[3] * a); ctx.set_line_width(w)
    ctx.move_to(*pts[0])
    for pt in pts[1:]:
        ctx.line_to(*pt)
    ctx.stroke()
    if q > 0.85:
        hp = clamp((q - 0.85) / 0.15)
        hx, hy = pts[-1]; px, py = pts[-4]
        ang = math.atan2(hy - py, hx - px)
        hd = float(el.get("head", 36)) * hp
        for sgn in (-1, 1):
            ctx.move_to(hx, hy)
            ctx.line_to(hx - hd * math.cos(ang + sgn * 0.5), hy - hd * math.sin(ang + sgn * 0.5))
        ctx.stroke()
    ctx.restore()
    if el.get("label"):
        lp = spring(lt, 0.0, stiffness=400, damping=15)
        fs = float(el.get("size", 68))
        fnt_c = resolve_font("title", None, "title")
        ly = p0[1] - fs * 0.85
        ctx.save()
        ctx.translate(540, ly)
        _S(ctx, lp, lp)
        draw_text(ctx, el["label"], 0, 0, size=fs, color="#ffd60a", alpha=a, weight=fnt_c, style="extrude_3d")
        _draw_glow_underline(ctx, 0, fs * 0.58, fs * 5.2, "#ff2d55", lt / 0.35, alpha=a)
        ctx.restore()
    return None


# ---------------------------------------------------------------------------
@element("light_leak", dur=0.1, enter="none", sfx="none")
def el_light_leak(ctx, el, s):
    sc = s.scene
    W, H = (sc.W, sc.H) if sc else (1080, 1920)
    d = float(el.get("dur", 0.9))
    q = clamp(s.lt / d)
    if q <= 0 or q >= 1:
        return None
    env = math.sin(math.pi * q)
    a = s.alpha * env * float(el.get("alpha", 0.75))
    dr = float(el.get("dir", 1))
    cx = W * (0.5 + dr * (q - 0.5) * 1.6)
    for c, ox, rr in ((el.get("color", "#ff8a3d"), 0, 0.75), (el.get("color2", "#ff2d55"), -0.25 * dr, 0.5)):
        g = cairo.RadialGradient(cx + ox * W, H * 0.35, 0, cx + ox * W, H * 0.35, W * rr * 1.4)
        cc = _color(c)
        g.add_color_stop_rgba(0, cc[0], cc[1], cc[2], a)
        g.add_color_stop_rgba(1, cc[0], cc[1], cc[2], 0)
        ctx.save(); ctx.set_operator(cairo.OPERATOR_ADD); ctx.set_source(g); ctx.paint(); ctx.restore()
    return None


@element("shine_sweep", dur=0.1, enter="none", sfx="none")
def el_shine_sweep(ctx, el, s):
    return None


@element("progress_bar", dur=0.1, enter="none", sfx="none")
def el_progress_bar(ctx, el, s):
    sc = s.scene
    W, H = (sc.W, sc.H) if sc else (1080, 1920)
    t0 = float(el.get("from", el.get("t", 0))); t1 = float(el.get("to", sc.duration if sc else 10))
    q = clamp((s.t - t0) / max(0.01, t1 - t0))
    y = float(el.get("y", 0)); h = float(el.get("h", 10))
    a = s.alpha * (1 - s.out_p)
    c_bg = _color(el.get("bg", "#ffffff"))
    c_fg = _color(el.get("color", "#ff2d55"))
    ctx.rectangle(0, y, W, h); ctx.set_source_rgba(c_bg[0], c_bg[1], c_bg[2], a * 0.18); ctx.fill()
    ctx.rectangle(W - W * q, y, W * q, h); ctx.set_source_rgba(c_fg[0], c_fg[1], c_fg[2], a); ctx.fill()
    return None


@element("vignette_pulse", dur=0.1, enter="none", sfx="none")
def el_vignette_pulse(ctx, el, s):
    sc = s.scene
    W, H = (sc.W, sc.H) if sc else (1080, 1920)
    amt = float(el.get("amount", 0.45))
    kick = 0.0
    for tt in el.get("at") or []:
        if s.t >= float(tt):
            kick = max(kick, math.exp(-(s.t - float(tt)) * float(el.get("decay", 5))))
    a = s.alpha * (1 - s.out_p) * clamp(amt * (0.55 + 0.45 * kick))
    g = cairo.RadialGradient(W / 2, H * 0.45, W * (0.55 - 0.12 * kick), W / 2, H * 0.45, H * 0.75)
    g.add_color_stop_rgba(0, 0, 0, 0, 0); g.add_color_stop_rgba(1, 0, 0, 0, a)
    ctx.set_source(g); ctx.paint()
    return None
