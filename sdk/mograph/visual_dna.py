"""
mograph.visual_dna — v13.5 Visual-DNA & Zero-Box Procedural Asset Engine.

Implements the 4 signature Visual-DNA elements:
  dna_stage        Engineered Calm coordinate grid environment + key-fill lighting + telemetry
  dna_connector    Single-thread @accent vector structural device (caliper_rail, circuit_bus, etc.)
  tactile_3d       Isolated 3D studio object with 3D perspective homography spin & zero-box callouts
  hud_loading_bar  Game-style cybernetic segmented loading bar (the ONLY allowed rectangular graphic!)
"""
from __future__ import annotations
import math
import os
import cairo
import cv2
import numpy as np

from .core import clamp, color as _color, get_ease, prog, spring
from .elements import element, val, xy
from .reelfx import _ASSETS_3D, _blit_3d_perspective, _draw_glow_underline, _load_3d_rgba
from .text import draw_text, resolve_font, to_fa_digits


@element("dna_stage", dur=0.45, enter="fade", sfx=None)
def el_dna_stage(ctx, el, s):
    """Engineered Calm coordinate grid stage with + intersection ticks, diffused key-fill
    lighting, and live X/Y telemetry readouts."""
    W, H = (s.scene.W, s.scene.H) if s.scene else (1080, 1920)
    mode = el.get("mode", "dark")  # "dark" (#0C0E14) or "light" (#F4F4F2)
    accent_hex = el.get("accent") or "#10B981"
    accent = _color(accent_hex)
    grid_step = int(el.get("grid_step", 64))
    dof_blur = float(val(el, "dof_blur", s.t, 0.0))

    ctx.save()
    if mode == "light":
        g = cairo.RadialGradient(W * 0.45, H * 0.38, 60, W * 0.5, H * 0.5, H * 0.75)
        g.add_color_stop_rgba(0.0, 0.965, 0.965, 0.955, s.alpha)
        g.add_color_stop_rgba(1.0, 0.915, 0.915, 0.898, s.alpha)
        grid_col = (0.12, 0.14, 0.18)
    else:
        g = cairo.RadialGradient(W * 0.5, H * 0.38, 60, W * 0.5, H * 0.5, H * 0.75)
        g.add_color_stop_rgba(0.0, 0.08, 0.10, 0.15, s.alpha)
        g.add_color_stop_rgba(1.0, 0.035, 0.042, 0.065, s.alpha)
        grid_col = accent[:3]
    ctx.set_source(g)
    ctx.rectangle(0, 0, W, H)
    ctx.fill()

    ag = cairo.RadialGradient(W * 0.5, H * 0.42, 20, W * 0.5, H * 0.42, W * 0.62)
    ag.add_color_stop_rgba(0.0, accent[0], accent[1], accent[2], 0.16 * s.alpha)
    ag.add_color_stop_rgba(1.0, accent[0], accent[1], accent[2], 0.0)
    ctx.set_source(ag)
    ctx.rectangle(0, 0, W, H)
    ctx.fill()

    grid_alpha = (0.14 if mode == "dark" else 0.11) * s.alpha * max(0.25, 1.0 - 0.65 * dof_blur)
    ctx.set_line_width(0.75)
    ctx.set_source_rgba(grid_col[0], grid_col[1], grid_col[2], grid_alpha)
    drift_y = (s.t * 14.0) % grid_step
    for gx in range(0, W + grid_step, grid_step):
        ctx.move_to(gx, 0)
        ctx.line_to(gx, H)
    for gy in range(0, H + grid_step, grid_step):
        yy = gy + drift_y
        ctx.move_to(0, yy)
        ctx.line_to(W, yy)
    ctx.stroke()

    ctx.set_line_width(1.3)
    ctx.set_source_rgba(grid_col[0], grid_col[1], grid_col[2], grid_alpha * 1.9)
    for gx in range(grid_step, W, grid_step * 2):
        for gy in range(grid_step, H, grid_step * 2):
            yy = gy + drift_y
            ctx.move_to(gx - 5, yy); ctx.line_to(gx + 5, yy)
            ctx.move_to(gx, yy - 5); ctx.line_to(gx, yy + 5)
    ctx.stroke()

    if el.get("telemetry", True):
        tc_col = "#1A1D24" if mode == "light" else "#94A3B8"
        f_tc = resolve_font("counter", "bold", "title")
        draw_text(ctx, f"DNA.GRID // T:{s.t:05.2f}s", 210, 72, size=22, color=tc_col,
                  alpha=s.alpha * 0.55, weight=f_tc)
        draw_text(ctx, "ZERO-BOX // 3D.STAGE", W - 210, 72, size=22, color=tc_col,
                  alpha=s.alpha * 0.55, weight=f_tc)
    ctx.restore()
    return None


@element("dna_connector", dur=0.85, enter="fade", sfx="pen")
def el_dna_connector(ctx, el, s):
    """Procedural single-thread @accent vector structural connector:
    device = 'caliper_rail' | 'circuit_bus' | 'oscillating_wave' | 'flight_bezier' | 'magnetic_ledger'."""
    p0 = el.get("from", [160, 960])
    p1 = el.get("to", [920, 960])
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dev = el.get("device", "caliper_rail")
    col_hex = el.get("color") or "#10B981"
    rgb = _color(col_hex)
    draw_p = clamp((s.t - float(el["t"])) / max(0.15, float(el.get("dur", 0.85))))
    ep = get_ease("out_cubic")(draw_p)

    ctx.save()
    xm = x0 + (x1 - x0) * ep
    ym = y0 + (y1 - y0) * ep

    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for w_gl, a_gl in [(14.0, 0.14), (6.0, 0.32), (2.6, 0.95)]:
        ctx.set_line_width(w_gl)
        ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], a_gl * s.alpha)
        if dev == "oscillating_wave":
            steps = 48
            for i in range(int(steps * ep) + 1):
                u = i / steps
                px = x0 + (x1 - x0) * u
                py = y0 + (y1 - y0) * u + math.sin(u * math.pi * 4.0 - s.t * 5.0) * float(el.get("amp", 22.0)) * math.sin(u * math.pi)
                if i == 0:
                    ctx.move_to(px, py)
                else:
                    ctx.line_to(px, py)
        elif dev == "circuit_bus":
            mid_x = x0 + (x1 - x0) * 0.5
            ctx.move_to(x0, y0)
            if ep < 0.45:
                ctx.line_to(x0 + (mid_x - x0) * (ep / 0.45), y0)
            elif ep < 0.65:
                ctx.line_to(mid_x, y0)
                ctx.line_to(mid_x, y0 + (y1 - y0) * ((ep - 0.45) / 0.20))
            else:
                ctx.line_to(mid_x, y0)
                ctx.line_to(mid_x, y1)
                ctx.line_to(mid_x + (x1 - mid_x) * ((ep - 0.65) / 0.35), y1)
        else:
            ctx.move_to(x0, y0)
            ctx.line_to(xm, ym)
        ctx.stroke()

    if dev in ("caliper_rail", "magnetic_ledger"):
        ticks = int(el.get("ticks", 9))
        dx, dy = x1 - x0, y1 - y0
        ln = max(1.0, math.hypot(dx, dy))
        nx, ny = -dy / ln, dx / ln
        ctx.set_line_width(1.6)
        for i in range(ticks + 1):
            u = i / max(1, ticks)
            if u > ep:
                break
            tx = x0 + dx * u
            ty = y0 + dy * u
            th = 14.0 if i in (0, ticks, ticks // 2) else 7.0
            ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], 0.85 * s.alpha)
            ctx.move_to(tx - nx * th, ty - ny * th)
            ctx.line_to(tx + nx * th, ty + ny * th)
        ctx.stroke()

    pulse = 1.0 + 0.25 * math.sin(s.t * 8.0)
    ctx.arc(xm, ym, 9.0 * pulse, 0, 2 * math.pi)
    ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], 0.35 * s.alpha)
    ctx.fill()
    ctx.arc(xm, ym, 4.5, 0, 2 * math.pi)
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.95 * s.alpha)
    ctx.fill()
    ctx.restore()
    return None


def _ensure_sprite_name(el, scene) -> str:
    sp_name = el.get("sprite") or el.get("asset")
    if sp_name:
        if not sp_name.endswith(".png"):
            sp_name += ".png"
        if os.path.exists(os.path.join(_ASSETS_3D, sp_name)):
            return sp_name
    src = el.get("src")
    if src:
        abs_p = src if os.path.isabs(src) else os.path.join(scene.base_dir if scene else ".", src)
        if os.path.exists(abs_p):
            stem = os.path.splitext(os.path.basename(abs_p))[0] + "_auto3d.png"
            dst = os.path.join(_ASSETS_3D, stem)
            if not os.path.exists(dst):
                im = cv2.imread(abs_p, cv2.IMREAD_UNCHANGED)
                if im is not None:
                    if im.ndim == 3 and im.shape[2] == 4:
                        cv2.imwrite(dst, im)
                    else:
                        from .extract3d import key_3d_rgba
                        rgba = key_3d_rgba(im, bg_mode=el.get("bg_key", "auto"))
                        cv2.imwrite(dst, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                    _load_3d_rgba.cache_clear()
            return stem
    return "laptop_3d.png"


@element("tactile_3d", dur=0.65, enter="spring", sfx="pop")
def el_tactile_3d(ctx, el, s):
    """Universal Isolated 3D Studio Object with true 3D perspective homography rotation,
    18-25% Gaussian floor contact shadow, and Zero-Box multi-font coordinate callout."""
    x, y = xy(el, s.t)
    size = float(val(el, "size", s.t, 240)) * s.scale
    sp_name = _ensure_sprite_name(el, s.scene)
    local_t = max(0.0, s.t - float(el["t"]))
    sp_in = spring(min(1.0, local_t / max(0.25, float(el.get("dur", 0.65)))))

    yaw0 = float(el.get("yaw_from", -52.0))
    pitch0 = float(el.get("pitch_from", 18.0))
    yaw = yaw0 * (1.0 - sp_in) + float(val(el, "yaw", s.t, 0.0))
    pitch = pitch0 * (1.0 - sp_in) + float(val(el, "pitch", s.t, 0.0))
    roll = float(val(el, "roll", s.t, 0.0))
    if el.get("idle_spin", True):
        yaw += math.sin(s.t * 2.4 + x * 0.01) * float(el.get("spin_amp", 14.0))
        pitch += math.cos(s.t * 1.9 + y * 0.01) * float(el.get("pitch_amp", 7.0))

    cy_float = y + s.dy + math.sin(s.t * 3.0 + x * 0.01) * 7.0
    accent_hex = el.get("accent") or "#10B981"
    _blit_3d_perspective(
        ctx, sp_name, x, cy_float, size * max(0.2, sp_in),
        alpha=s.alpha, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
        shadow=bool(el.get("shadow", True)), glow_hex=accent_hex
    )

    title = el.get("title") or el.get("text")
    sub = el.get("sub") or el.get("label")
    ink_hex = el.get("text_color") or "#FFFFFF"
    if title or sub:
        ty = cy_float + size * 0.56 + float(el.get("label_offset_y", 28))
        if sub:
            f_sub = resolve_font(el.get("sub_font", "handwritten"), "bold", "handwritten")
            draw_text(ctx, sub, x, ty, size=float(el.get("sub_size", 38)), color=accent_hex,
                      alpha=s.alpha, weight=f_sub)
            ty += float(el.get("sub_size", 38)) * 1.45
        if title:
            f_tit = resolve_font(el.get("font", "title"), "black", "title")
            tsize = float(el.get("title_size", 56))
            draw_text(ctx, title, x, ty, size=tsize, color=ink_hex, alpha=s.alpha,
                      weight=f_tit, style="extrude_3d")
            _draw_glow_underline(ctx, x, ty + tsize * 0.55, min(380, len(title) * tsize * 0.42),
                                 accent_hex, sp_in, alpha=s.alpha)

    r = size * 0.6
    return (x - r, cy_float - r, x + r, cy_float + r + (130 if (title or sub) else 0))


@element("hud_loading_bar", dur=0.6, enter="spring", sfx="tick")
def el_hud_loading_bar(ctx, el, s):
    """Game-Style Cybernetic Segmented Loading Bar — the ONLY rectangular UI graphic
    permitted under the User's Strict Zero-Box Law."""
    x, y = xy(el, s.t)
    w = float(val(el, "w", s.t, 560)) * s.scale
    h = float(val(el, "h", s.t, 44)) * s.scale
    v0 = float(el.get("from", 0.0))
    v1 = float(el.get("to", el.get("value", 100.0)))
    load_dur = float(el.get("load_dur", 1.35))
    p = get_ease(el.get("ease", "out_cubic"))(clamp((s.t - float(el["t"])) / max(0.15, load_dur)))
    cur_v = v0 + (v1 - v0) * p
    frac = clamp(cur_v / max(1.0, float(el.get("max", 100.0))))

    accent_hex = el.get("accent") or "#10B981"
    rgb = _color(accent_hex)
    ink_hex = el.get("text_color") or "#FFFFFF"
    segments = int(el.get("segments", 16))

    ctx.save()
    ctx.translate(x, y + s.dy)
    x0, y0 = -w * 0.5, -h * 0.5

    bw = 14.0
    ctx.set_line_width(2.4)
    ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], 0.92 * s.alpha)
    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        cx_c = sx * (w * 0.5 + 8)
        cy_c = sy * (h * 0.5 + 8)
        ctx.move_to(cx_c, cy_c - sy * bw)
        ctx.line_to(cx_c, cy_c)
        ctx.line_to(cx_c - sx * bw, cy_c)
    ctx.stroke()

    gap = 6.0
    seg_w = (w - gap * (segments - 1)) / segments
    active_segs = int(round(frac * segments))
    for i in range(segments):
        idx = (segments - 1 - i) if el.get("rtl", True) else i
        sx0 = x0 + idx * (seg_w + gap)
        is_on = i < active_segs
        if is_on:
            ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], 0.28 * s.alpha)
            ctx.rectangle(sx0 - 3, y0 - 3, seg_w + 6, h + 6)
            ctx.fill()
            ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], 0.95 * s.alpha)
        else:
            ctx.set_source_rgba(1.0, 1.0, 1.0, 0.12 * s.alpha)
        ctx.rectangle(sx0, y0, seg_w, h)
        ctx.fill()

    lbl = el.get("label", "بارگذاری سیستم سه‌بعدی")
    pct_txt = (to_fa_digits(f"{int(round(cur_v))}") + "٪") if el.get("fa", True) else f"{int(round(cur_v))}%"
    f_lbl = resolve_font(el.get("font", "title"), "bold", "title")
    f_cnt = resolve_font("counter", "black", "counter")
    draw_text(ctx, lbl, w * 0.18, y0 - 38, size=float(el.get("label_size", 32)), color=ink_hex,
              alpha=s.alpha, weight=f_lbl)
    draw_text(ctx, pct_txt, -w * 0.36, y0 - 38, size=float(el.get("value_size", 38)), color=accent_hex,
              alpha=s.alpha, weight=f_cnt, style="extrude_3d")
    ctx.restore()
    return (x - w * 0.55, y - h - 65, x + w * 0.55, y + h * 0.6)


def generate_dna_plan_markdown(topic: str) -> str:
    """Generate a 5-part Visual-DNA production plan + ready-to-run 3D asset generation prompts."""
    return f"""# 🧬 Visual-DNA Production Blueprint: {topic}

## 0. Kinetic Concept & Thread Accent
- **Aesthetic Law:** 100% Zero-Box Visual-DNA (No flat 2D boxes/capsules; only isolated 3D tactile objects, coordinate grids, and optional `hud_loading_bar`).
- **Base Environment:** `dna_stage` (`#0C0E14` obsidian dark for reels or `#F4F4F2` warm stone for standalone infographics) with `0.6px` coordinate grid + `+` ticks.
- **Primary Thread Color (`@accent`):** `#10B981` (Emerald Signal) or `#F59E0B` (Warm Amber) or `#0055FF` (International Klein Blue).

## 1. Vector Structural Infrastructure (`dna_connector`)
- `caliper_rail` / `circuit_bus` connecting the stage nodes with live datum ticks and glowing focus dots.

## 2. Isolated 3D Studio Asset Prompts (For In-Workspace `image_gen:generate_image`)
> Generate a `1:1` 2x2 sheet on solid `#000000` black background, then slice with:
> `python3 -m mograph extract3d sheet.jpg --names item1_3d,item2_3d,item3_3d,item4_3d --grid 2x2`

- **Prompt:**
  `A 2x2 grid of 4 isolated high-end 3D studio objects on a pure solid pitch-black (#000000) background with wide empty black margins between the 4 quadrants, zero text, zero boxes, Octane Render / Cinema4D physical materials (frosted translucent glass, brushed anodized metal, matte ceramic, polished acrylic) for topic "{topic}": Top-Left: [Hero 3D Object 1]; Top-Right: [Hero 3D Object 2]; Bottom-Left: [Hero 3D Object 3]; Bottom-Right: [Hero 3D Object 4].`

## 3. Motion Choreography & Multi-Font Pairing
- **Behind-Speaker Chapters (`behind: "clip"`):** Full-environment coordinate grid behind segmented speaker + `KalamehFaNum-Black` (`extrude_3d`) + `DigiGhasedak-Bold` handwritten subtitle.
- **Full-Screen Climax (`dof_blur`):** Smoothly blur the speaker (`f/2.8`) and stage the hero 3D warning/title in full-frame center.
- **3D Perspective Spin:** Every `tactile_3d` and `logo_chip` enters with `yaw_from: -55° -> 0°` and `pitch_from: 18° -> 0°`.
"""
