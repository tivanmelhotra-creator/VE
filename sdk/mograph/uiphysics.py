"""
mograph.uiphysics — v10 Studio Omni: 5th style world ("ui_physics") & Interactive Tactile UI / Physics
Choreography widgets (inspired by Apple / Linear / Rube-Goldberg UI motion films).

Widgets registered here:
  * ui_toggle              Frameless iOS/macOS pill toggle switch + vector cursor click & spring recoil
  * ui_slider              Horizontal progress track + tick marks + overhead rolling odometer tooltip bubble
  * ui_notification_stack  Stack of frosted dark/light notification cards with spring entrance & ball-impact compression
  * ui_keyboard            Mechanical QWERTY + stepped ISO Return keyboard with per-key spring depression & highlight
  * ui_reply_pill          Morphing Reply pill → live typing input field with caret & send button → sent message bubble
  * physics_ball           Continuous squash-and-stretch hero ball hopping across waypoints/UI widgets with bounceArc
  * ui_film_world          Complete 1-line Rube-Goldberg UI chain-reaction scene (Toggle → Slider → Cards → Keyboard → Loop)
"""
from __future__ import annotations

import math

import cairo

from .core import clamp, color as _color, lerp, prog
from .elements import F, element, xy
from .text import _is_rtl_char, draw_text, measure_text, to_fa_digits
from .typo import pal_of


# ---------------------------------------------------------------------------
# Closed-form physics & easing helpers (ported from ui_motion_film.html)
# ---------------------------------------------------------------------------
def smoothstep(u: float) -> float:
    x = clamp(u, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(u: float) -> float:
    x = clamp(u, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def ease_out_cubic(u: float) -> float:
    x = clamp(u, 0.0, 1.0)
    return 1.0 - (1.0 - x) ** 3


def ease_in_quad(u: float) -> float:
    x = clamp(u, 0.0, 1.0)
    return x * x


def ease_out_quad(u: float) -> float:
    x = clamp(u, 0.0, 1.0)
    return 1.0 - (1.0 - x) * (1.0 - x)


def spring_step(dt: float, omega0: float = 26.0, zeta: float = 0.68) -> float:
    if dt <= 0.0:
        return 0.0
    if dt > 0.75:
        return 1.0
    wd = omega0 * math.sqrt(max(1e-6, 1.0 - zeta * zeta))
    env = math.exp(-zeta * omega0 * dt)
    return 1.0 - env * (math.cos(wd * dt) + (zeta * omega0 / wd) * math.sin(wd * dt))


def spring_impulse(dt: float, omega0: float = 34.0, zeta: float = 0.66) -> float:
    if dt <= 0.0 or dt > 0.48:
        return 0.0
    wd = omega0 * math.sqrt(max(1e-6, 1.0 - zeta * zeta))
    return math.exp(-zeta * omega0 * dt) * math.sin(wd * dt)


def bounce_arc(x0: float, y0: float, x1: float, y1: float, u: float, apex_lift: float = 115.0):
    uc = clamp(u, 0.0, 1.0)
    x = lerp(x0, x1, smoothstep(uc))
    y_lin = lerp(y0, y1, uc)
    y_par = -4.0 * apex_lift * uc * (1.0 - uc)
    return x, y_lin + y_par


def rebound_drop_arc(x0: float, y0: float, x1: float, y1: float, u: float, rebound_h: float = 72.0):
    uc = clamp(u, 0.0, 1.0)
    x = lerp(x0, x1, smoothstep(uc))
    split = 0.46
    apex_y = y0 - rebound_h
    if uc < split:
        y = lerp(y0, apex_y, ease_out_quad(uc / split))
    else:
        y = lerp(apex_y, y1, ease_in_quad((uc - split) / (1.0 - split)))
    return x, y


# ---------------------------------------------------------------------------
# Cairo rounded-rect & ISO-Return geometry primitives
# ---------------------------------------------------------------------------
def _round_rect_path(ctx, x: float, y: float, w: float, h: float, r: float):
    rad = max(0.0, min(r, w * 0.5, h * 0.5))
    ctx.new_sub_path()
    ctx.arc(x + w - rad, y + rad, rad, -math.pi * 0.5, 0.0)
    ctx.arc(x + w - rad, y + h - rad, rad, 0.0, math.pi * 0.5)
    ctx.arc(x + rad, y + h - rad, rad, math.pi * 0.5, math.pi)
    ctx.arc(x + rad, y + rad, rad, math.pi, math.pi * 1.5)
    ctx.close_path()


def _fill_round_rect(ctx, x: float, y: float, w: float, h: float, r: float,
                     fill_rgba=None, stroke_rgba=None, line_w: float = 1.5, alpha: float = 1.0):
    _round_rect_path(ctx, x, y, w, h, r)
    if fill_rgba is not None:
        c = _color(fill_rgba) if isinstance(fill_rgba, str) else fill_rgba
        ctx.set_source_rgba(c[0], c[1], c[2], c[3] * alpha)
        if stroke_rgba is not None:
            ctx.fill_preserve()
        else:
            ctx.fill()
    if stroke_rgba is not None:
        sc = _color(stroke_rgba) if isinstance(stroke_rgba, str) else stroke_rgba
        ctx.set_source_rgba(sc[0], sc[1], sc[2], sc[3] * alpha)
        ctx.set_line_width(line_w)
        ctx.stroke()


def _iso_return_path(ctx, x: float, y: float, top_w: float, bot_w: float, h: float, top_h: float, r: float):
    step_x = x + (top_w - bot_w)
    step_y = y + top_h
    right_x = x + top_w
    bot_y = y + h
    ctx.new_path()
    ctx.move_to(x + r, y)
    ctx.line_to(right_x - r, y)
    ctx.curve_to(right_x, y, right_x, y, right_x, y + r)
    ctx.line_to(right_x, bot_y - r)
    ctx.curve_to(right_x, bot_y, right_x, bot_y, right_x - r, bot_y)
    ctx.line_to(step_x + r, bot_y)
    ctx.curve_to(step_x, bot_y, step_x, bot_y, step_x, bot_y - r)
    ctx.line_to(step_x, step_y + r)
    ctx.curve_to(step_x, step_y, step_x, step_y, step_x - r, step_y)
    ctx.line_to(x + r, step_y)
    ctx.curve_to(x, step_y, x, step_y, x, step_y - r)
    ctx.line_to(x, y + r)
    ctx.curve_to(x, y, x, y, x + r, y)
    ctx.close_path()


def _draw_cursor(ctx, cx: float, cy: float, scale: float = 1.0, alpha: float = 1.0):
    if alpha <= 0.01:
        return
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(scale, scale)
    ctx.new_path()
    ctx.move_to(0, 0)
    ctx.line_to(0, 42)
    ctx.line_to(11, 32)
    ctx.line_to(19, 49)
    ctx.line_to(27, 45)
    ctx.line_to(19, 28)
    ctx.line_to(33, 28)
    ctx.close_path()
    ctx.set_source_rgba(0.04, 0.04, 0.05, 0.96 * alpha)
    ctx.fill_preserve()
    ctx.set_source_rgba(1.0, 1.0, 1.0, alpha)
    ctx.set_line_width(3.2)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.stroke()
    ctx.restore()


def _draw_card_icon(ctx, icon_kind: str, ic_cx: float, ic_cy: float, alpha: float = 1.0):
    ctx.save()
    ctx.set_source_rgba(1.0, 1.0, 1.0, alpha)
    ctx.set_line_width(2.8)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    if icon_kind == "slider":
        ctx.move_to(ic_cx - 12, ic_cy)
        ctx.line_to(ic_cx + 12, ic_cy)
        ctx.stroke()
        ctx.arc(ic_cx + 3, ic_cy, 5.5, 0, math.pi * 2)
        ctx.fill()
    elif icon_kind == "gravity":
        ctx.move_to(ic_cx, ic_cy - 10)
        ctx.line_to(ic_cx, ic_cy + 10)
        ctx.move_to(ic_cx - 7, ic_cy + 3)
        ctx.line_to(ic_cx, ic_cy + 10)
        ctx.line_to(ic_cx + 7, ic_cy + 3)
        ctx.stroke()
    else:  # messages / chat bubble
        _round_rect_path(ctx, ic_cx - 11, ic_cy - 9, 22, 16, 6)
        ctx.fill()
        ctx.move_to(ic_cx - 6, ic_cy + 5)
        ctx.line_to(ic_cx - 10, ic_cy + 11)
        ctx.line_to(ic_cx - 1, ic_cy + 6)
        ctx.close_path()
        ctx.fill()
    ctx.restore()


# ---------------------------------------------------------------------------
# 1. @element("ui_toggle")
# ---------------------------------------------------------------------------
@element("ui_toggle", dur=0.35, enter="fade", sfx="click")
def el_ui_toggle(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 154))
    h = float(el.get("h", 78))
    pal = pal_of(el, s)
    acc = el.get("accent") or pal.get("accent", "#0a84ff")
    off_col = el.get("off_color") or pal.get("switch_off", "#3a3a42")
    click_t = float(el.get("click_at", float(el.get("t", 0.0)) + 0.10))
    reset_t = float(el.get("reset_at", 999.0))
    t = s.t

    blue_mix = 0.0
    if click_t <= t < reset_t:
        blue_mix = clamp((t - click_t) / 0.08, 0.0, 1.0)
    elif t >= reset_t:
        blue_mix = 1.0 - clamp((t - reset_t) / 0.12, 0.0, 1.0)

    click_bounce = spring_impulse(t - (click_t - 0.02), 36.0, 0.65) * (-0.06) if (click_t - 0.02 <= t < click_t + 0.25) else 0.0
    reset_bounce = spring_impulse(t - reset_t, 32.0, 0.65) * 0.05 if (reset_t <= t < reset_t + 0.35) else 0.0
    sc = (1.0 + click_bounce + reset_bounce) * s.scale

    ctx.save()
    ctx.translate(x, y)
    ctx.scale(sc, sc)
    _fill_round_rect(ctx, -w * 0.5, -h * 0.5, w, h, h * 0.5, off_col, None, 0, s.alpha)
    if blue_mix > 0.001:
        _fill_round_rect(ctx, -w * 0.5, -h * 0.5, w, h, h * 0.5, acc, None, 0, s.alpha * blue_mix)

    # Optional built-in knob if standalone (when not driven by physics_ball)
    if el.get("knob", True):
        r = float(el.get("knob_r", h * 0.385))
        lx, rx = -w * 0.5 + h * 0.5, w * 0.5 - h * 0.5
        kx = lerp(lx, rx, ease_out_cubic(blue_mix))
        ctx.arc(kx, 0, r, 0, math.pi * 2)
        ctx.set_source_rgba(1, 1, 1, s.alpha)
        ctx.fill()
    ctx.restore()

    if el.get("cursor", True):
        cx0, cy0 = x - w * 0.16, y + h * 0.36
        press = 1.0 - 0.12 * math.sin(clamp((t - (click_t - 0.04)) / 0.14) * math.pi) if (click_t - 0.04 <= t <= click_t + 0.10) else 1.0
        _draw_cursor(ctx, cx0, cy0, scale=press * s.scale, alpha=s.alpha)

    return (x - w * 0.6, y - h * 0.6, x + w * 0.6, y + h * 1.1)


# ---------------------------------------------------------------------------
# 2. @element("ui_slider")
# ---------------------------------------------------------------------------
@element("ui_slider", dur=0.4, enter="fade", sfx="tick")
def el_ui_slider(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 640))
    track_h = float(el.get("track_h", 8))
    num_ticks = int(el.get("ticks", 11))
    pal = pal_of(el, s)
    acc = el.get("accent") or pal.get("accent", "#0a84ff")
    track_bg = el.get("track_bg") or pal.get("track_bg", "#32323a")
    tooltip_bg = el.get("tooltip_bg") or pal.get("panel", "#2c2c34")
    fa = bool(el.get("fa", False))

    t = s.t
    roll_t = float(el.get("roll_at", float(el.get("t", 0.0)) + 0.2))
    roll_dur = float(el.get("roll_dur", 1.6))
    rewind_t = float(el.get("rewind_at", 999.0))
    rewind_dur = float(el.get("rewind_dur", 0.68))

    if t < roll_t:
        p = 0.0
    elif t <= roll_t + roll_dur:
        u = clamp((t - roll_t) / max(1e-4, roll_dur))
        p = 1.0 - (1.0 - u) ** 1.85
    elif t < rewind_t:
        p = 1.0
    elif t <= rewind_t + rewind_dur:
        rw = clamp((t - rewind_t) / max(1e-4, rewind_dur))
        p = (1.0 - rw) ** 2.2
    else:
        p = 0.0

    max_val = float(el.get("max_value", 100))
    exact_val = p * max_val
    x0 = x - w * 0.5
    fill_x = lerp(x0, x0 + w, p)
    tick_y = y + float(el.get("tick_offset", 46))

    _fill_round_rect(ctx, x0, y - track_h * 0.5, w, track_h, track_h * 0.5, track_bg, None, 0, s.alpha)
    if p > 0.002:
        fw = max(track_h, fill_x - x0)
        _fill_round_rect(ctx, x0, y - track_h * 0.5, fw, track_h, track_h * 0.5, acc, None, 0, s.alpha)

    for i in range(max(2, num_ticks)):
        tf = i / (num_ticks - 1)
        tx = lerp(x0, x0 + w, tf)
        is_lit = p > 0.005 and tf <= p + 0.005
        _fill_round_rect(ctx, tx - 2, tick_y - 7, 4, 14, 2, acc if is_lit else track_bg, None, 0, s.alpha)

    if el.get("knob", True):
        kr = float(el.get("knob_r", 28))
        ctx.arc(fill_x, y, kr, 0, math.pi * 2)
        ctx.set_source_rgba(1, 1, 1, s.alpha)
        ctx.fill()

    if el.get("tooltip", True):
        b_scale = 0.0
        if roll_t <= t <= roll_t + roll_dur + 0.15:
            en = spring_step(t - roll_t, 32, 0.68)
            ex = 1.0 - smoothstep((t - (roll_t + roll_dur)) / 0.12) if el.get("hide_tooltip_after", False) and t > roll_t + roll_dur else 1.0
            b_scale = en * ex
        if b_scale > 0.01:
            bx, by = fill_x, y - float(el.get("tooltip_offset", 68))
            bw, bh = 68.0, 44.0
            ctx.save()
            ctx.translate(bx, by)
            ctx.scale(b_scale, b_scale)
            _fill_round_rect(ctx, -bw * 0.5, -bh * 0.5, bw, bh, 11, tooltip_bg, None, 0, s.alpha)
            tc = _color(tooltip_bg)
            ctx.set_source_rgba(tc[0], tc[1], tc[2], s.alpha)
            ctx.move_to(-7, bh * 0.5 - 1)
            ctx.line_to(7, bh * 0.5 - 1)
            ctx.line_to(0, bh * 0.5 + 7)
            ctx.close_path()
            ctx.fill()
            val_str = str(int(round(exact_val)))
            if fa:
                val_str = to_fa_digits(val_str)
            draw_text(ctx, val_str, 0, 0, size=22, color="#ffffff", weight=F(el, "counter"), align="center", alpha=s.alpha)
            ctx.restore()

    return (x0 - 30, y - 95, x0 + w + 30, tick_y + 20)


# ---------------------------------------------------------------------------
# 3. @element("ui_notification_stack")
# ---------------------------------------------------------------------------
@element("ui_notification_stack", dur=0.5, enter="fade", sfx="pop")
def el_ui_notification_stack(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 396))
    card_h = float(el.get("card_h", 92))
    gap = float(el.get("gap", 18))
    pal = pal_of(el, s)
    acc = el.get("accent") or pal.get("accent", "#0a84ff")
    card_bg = el.get("card_bg") or pal.get("panel", "#1c1c21")
    card_border = el.get("card_border") or pal.get("border", "#2c2c34")
    ink = pal.get("ink", "#ffffff")
    mute = "#b8c2be" if pal.get("paper", "#000").startswith("#0") else pal.get("mute", "#6e6e77")

    cards = el.get("cards") or [
        {"title": "Slider", "body": "Reached 100", "icon": "slider"},
        {"title": "Gravity", "body": "Something is falling", "icon": "gravity"},
        {"title": "Messages", "body": "What now?", "icon": "messages"},
    ]
    t0 = float(el.get("t", 0.0))
    stagger = float(el.get("stagger", 0.25))
    t = s.t

    total_h = len(cards) * card_h + max(0, len(cards) - 1) * gap
    top_y = y - total_h * 0.5

    for idx, c in enumerate(cards):
        ap_t = float(c.get("appear_at", t0 + idx * stagger))
        hit_t = float(c.get("hit_at", 999.0))
        fade_t = float(c.get("fade_at", 999.0))
        if t < ap_t:
            continue
        enter_u = clamp((t - ap_t) / 0.18, 0.0, 1.0)
        alpha = smoothstep(enter_u) * s.alpha
        y_off = (1.0 - spring_step(t - ap_t, 26.0, 0.72)) * 40.0
        sc = 1.0
        if t >= hit_t:
            y_off += spring_impulse(t - hit_t, 32.0, 0.65) * 16.0
        if t > fade_t:
            fu = clamp((t - fade_t) / 0.16, 0.0, 1.0)
            alpha *= (1.0 - fu)
            y_off -= fu * 18.0
            sc = 1.0 - fu * 0.04
        if alpha <= 0.01:
            continue

        cy = top_y + idx * (card_h + gap) + card_h * 0.5 + y_off
        cx = x
        rx, ry = cx - w * 0.5, cy - card_h * 0.5

        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(sc, sc)
        ctx.translate(-cx, -cy)

        _fill_round_rect(ctx, rx, ry, w, card_h, 20.0, card_bg, card_border, 1.4, alpha)

        title = str(c.get("title", "Notification"))
        body = str(c.get("body", ""))
        time_lbl = str(c.get("time", "الان" if any(_is_rtl_char(ch) for ch in title + body) else "now"))
        is_rtl = any(_is_rtl_char(ch) for ch in title + body)

        if is_rtl:
            ic_x = rx + w - 20 - 46
            ic_y = ry + (card_h - 46) * 0.5
            _fill_round_rect(ctx, ic_x, ic_y, 46, 46, 11, acc, None, 0, alpha)
            _draw_card_icon(ctx, c.get("icon", "messages"), ic_x + 23, ic_y + 23, alpha)
            draw_text(ctx, title, rx + w - 82, ry + 30, size=21, color=ink, weight=F(el, "ui_title"), align="right", alpha=alpha)
            draw_text(ctx, body, rx + w - 82, ry + 62, size=18, color=mute, weight=F(el, "body"), align="right", alpha=alpha)
            draw_text(ctx, time_lbl, rx + 22, ry + 30, size=16, color=mute, weight=F(el, "ui_label"), align="left", alpha=alpha * 0.8, quiet=True)
        else:
            ic_x = rx + 20
            ic_y = ry + (card_h - 46) * 0.5
            _fill_round_rect(ctx, ic_x, ic_y, 46, 46, 11, acc, None, 0, alpha)
            _draw_card_icon(ctx, c.get("icon", "messages"), ic_x + 23, ic_y + 23, alpha)
            draw_text(ctx, title, rx + 82, ry + 30, size=20, color=ink, weight=F(el, "ui_title"), align="left", alpha=alpha)
            draw_text(ctx, body, rx + 82, ry + 62, size=18, color=mute, weight=F(el, "body"), align="left", alpha=alpha)
            draw_text(ctx, time_lbl, rx + w - 22, ry + 30, size=16, color=mute, weight=F(el, "ui_label"), align="right", alpha=alpha * 0.8, quiet=True)

        ctx.restore()

    return (x - w * 0.5, top_y - 10, x + w * 0.5, top_y + total_h + 20)


# ---------------------------------------------------------------------------
# 4. @element("ui_keyboard")
# ---------------------------------------------------------------------------
@element("ui_keyboard", dur=0.5, enter="fade", sfx="type")
def el_ui_keyboard(ctx, el, s):
    x, y = xy(el, s.t)
    scale = float(el.get("scale", 1.0)) * s.scale
    pal = pal_of(el, s)
    key_bg = el.get("key_bg") or "#2c2c34"
    key_border = el.get("key_border") or "#3d3d47"
    key_active = el.get("key_active") or "#3e3e4a"
    ink = pal.get("ink", "#ffffff")
    hits = el.get("hits") or []
    t = s.t

    key_w, key_h, pitch = 68.0, 68.0, 80.0
    row1 = el.get("row1") or ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"]
    row2 = el.get("row2") or ["A", "S", "D", "F", "G", "H", "J", "K", "L"]

    ctx.save()
    ctx.translate(x, y)
    ctx.scale(scale, scale)

    keys = []
    for idx, lbl in enumerate(row1):
        kx = (idx - 4.8) * pitch
        ky = -40.0
        keys.append({"id": lbl.lower(), "label": lbl, "x": kx, "y": ky, "w": key_w, "h": key_h, "ret": False})
    for idx, lbl in enumerate(row2):
        kx = (idx - 4.45) * pitch
        ky = 40.0
        keys.append({"id": lbl.lower(), "label": lbl, "x": kx, "y": ky, "w": key_w, "h": key_h, "ret": False})

    # Stepped ISO Return key on the right
    p_x = (len(row1) - 1 - 4.8) * pitch
    top_w, bot_w, ret_h = 124.0, 98.0, 148.0
    left_top_x = p_x + key_w * 0.5 + 12.0
    keys.append({
        "id": "return", "label": el.get("return_label", "return"),
        "x": left_top_x + top_w * 0.5, "y": 0.0,
        "w": top_w, "bot_w": bot_w, "h": ret_h, "top_h": key_h, "ret": True,
    })

    for k in keys:
        depress = 0.0
        active_mix = 0.0
        for h_ev in hits:
            ht = float(h_ev.get("t", -99.0))
            hk = str(h_ev.get("key", h_ev.get("id", ""))).lower()
            if hk == k["id"] and t >= ht:
                dt = t - ht
                if k["ret"]:
                    if dt < 0.65:
                        dep = smoothstep(dt / 0.12) if dt < 0.45 else (1.0 - smoothstep((dt - 0.45) / 0.18))
                        depress = max(depress, dep * 10.0)
                        active_mix = max(active_mix, dep)
                elif dt < 0.34:
                    imp = spring_impulse(dt, 36.0, 0.65)
                    depress = max(depress, max(0.0, imp) * 10.0)
                    active_mix = max(active_mix, clamp(1.0 - dt / 0.26, 0.0, 1.0))

        kx, ky = k["x"], k["y"] + depress
        fill_c = key_active if active_mix > 0.25 else key_bg

        if k["ret"]:
            rx, ry = kx - k["w"] * 0.5, ky - k["h"] * 0.5
            # Bottom shadow
            _iso_return_path(ctx, rx, ry + 4, k["w"], k["bot_w"], k["h"], k["top_h"], 13.0)
            ctx.set_source_rgba(0.08, 0.08, 0.10, s.alpha)
            ctx.fill()
            # Key body
            _iso_return_path(ctx, rx, ry, k["w"], k["bot_w"], k["h"], k["top_h"], 13.0)
            c = _color(fill_c)
            ctx.set_source_rgba(c[0], c[1], c[2], s.alpha)
            ctx.fill_preserve()
            bc = _color(key_border)
            ctx.set_source_rgba(bc[0], bc[1], bc[2], s.alpha)
            ctx.set_line_width(1.5)
            ctx.stroke()
            # Return arrow icon
            ac_x = rx + k["w"] * 0.5 + 10
            ac_y = ry + 34
            ctx.set_source_rgba(1, 1, 1, s.alpha)
            ctx.set_line_width(2.2)
            ctx.move_to(ac_x + 11, ac_y - 6)
            ctx.line_to(ac_x + 11, ac_y + 4)
            ctx.line_to(ac_x - 11, ac_y + 4)
            ctx.move_to(ac_x - 5, ac_y - 2)
            ctx.line_to(ac_x - 11, ac_y + 4)
            ctx.line_to(ac_x - 5, ac_y + 10)
            ctx.stroke()
            draw_text(ctx, k["label"], rx + k["w"] - 18, ry + k["h"] - 20, size=14, color="#adadb5",
                      weight=F(el, "ui_label"), align="right", alpha=s.alpha)
        else:
            rx, ry = kx - k["w"] * 0.5, ky - k["h"] * 0.5
            _fill_round_rect(ctx, rx, ry + 4, k["w"], k["h"], 13.0, "#141418", None, 0, s.alpha)
            _fill_round_rect(ctx, rx, ry, k["w"], k["h"], 13.0, fill_c, key_border, 1.5, s.alpha)
            draw_text(ctx, k["label"], kx, ky, size=22, color=ink, weight=F(el, "ui_title"), align="center", alpha=s.alpha)

    ctx.restore()
    return (x - 460 * scale, y - 90 * scale, x + 520 * scale, y + 90 * scale)


# ---------------------------------------------------------------------------
# 5. @element("ui_reply_pill")
# ---------------------------------------------------------------------------
@element("ui_reply_pill", dur=0.4, enter="fade", sfx="none")
def el_ui_reply_pill(ctx, el, s):
    x, y = xy(el, s.t)
    pal = pal_of(el, s)
    acc = el.get("accent") or pal.get("accent", "#0a84ff")
    reply_bg = el.get("reply_bg") or "#24242b"
    reply_border = el.get("reply_border") or "#383842"
    placeholder = str(el.get("placeholder", "Reply"))
    type_at = float(el.get("type_at", float(el.get("t", 0.0)) + 0.6))
    send_at = float(el.get("send_at", 999.0))
    hits = el.get("hits") or []
    t = s.t

    pill_w = float(el.get("pill_w", 196))
    field_w = float(el.get("field_w", 248))
    h = float(el.get("h", 56))

    if t < send_at:
        w = lerp(pill_w, field_w, spring_step(t - type_at, 26.0, 0.70)) if t >= type_at else pill_w
        rx, ry = x - w * 0.5, y - h * 0.5
        _fill_round_rect(ctx, rx, ry, w, h, h * 0.5, reply_bg, reply_border, 1.6, s.alpha)

        if t < type_at:
            draw_text(ctx, placeholder, rx + 26, y, size=19, color="#9a9aa6", weight=F(el, "ui_label"), align="left", alpha=s.alpha, quiet=True)
        else:
            typed = "".join(str(h_ev.get("char", h_ev.get("key", ""))) for h_ev in hits if t >= float(h_ev.get("t", 999.0)) and h_ev.get("char", h_ev.get("key", "")) != "return")
            is_rtl = any(_is_rtl_char(ch) for ch in typed)
            if is_rtl:
                tx = rx + w - 26
                draw_text(ctx, typed, tx, y, size=21, color="#ffffff", weight=F(el, "ui_title"), align="right", alpha=s.alpha)
                tw, _ = measure_text(typed, 21, F(el, "ui_title")) if typed else (0.0, 0.0)
                _fill_round_rect(ctx, tx - tw - 6, y - 13, 2.5, 26, 1.2, acc, None, 0, s.alpha)
                btn_x = rx + 28
            else:
                tx = rx + 26
                draw_text(ctx, typed, tx, y, size=21, color="#ffffff", weight=F(el, "ui_title"), align="left", alpha=s.alpha)
                tw, _ = measure_text(typed, 21, F(el, "ui_title")) if typed else (0.0, 0.0)
                _fill_round_rect(ctx, tx + tw + 4, y - 13, 2.5, 26, 1.2, acc, None, 0, s.alpha)
                btn_x = rx + w - 28

            bs = spring_step(t - type_at, 30.0, 0.65)
            ctx.save()
            ctx.translate(btn_x, y)
            ctx.scale(bs, bs)
            ac = _color(acc)
            ctx.set_source_rgba(ac[0], ac[1], ac[2], s.alpha)
            ctx.arc(0, 0, 17, 0, math.pi * 2)
            ctx.fill()
            ctx.set_source_rgba(1, 1, 1, s.alpha)
            ctx.set_line_width(2.4)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            ctx.move_to(0, 7)
            ctx.line_to(0, -7)
            ctx.move_to(-6, -1)
            ctx.line_to(0, -7)
            ctx.line_to(6, -1)
            ctx.stroke()
            ctx.restore()
    else:
        # Sent blue pill + collapsed reply pill below
        sent_txt = str(el.get("sent_text") or "".join(str(h_ev.get("char", "")) for h_ev in hits) or "retry")
        sp = spring_step(t - send_at, 28.0, 0.68)
        _fill_round_rect(ctx, x - 85, y - 24, 170, 48, 24, reply_bg, reply_border, 1.5, s.alpha)
        draw_text(ctx, placeholder, x - 60, y, size=17, color="#9a9aa6", weight=F(el, "ui_label"), align="left", alpha=s.alpha, quiet=True)
        sy = y - 76 * sp
        sw, sh = max(112.0, len(sent_txt) * 18.0 + 44.0), 50.0
        _fill_round_rect(ctx, x + 70 - sw * 0.5, sy - sh * 0.5, sw, sh, sh * 0.5, acc, None, 0, s.alpha)
        draw_text(ctx, sent_txt, x + 70, sy, size=19, color="#ffffff", weight=F(el, "ui_title"), align="center", alpha=s.alpha)

    return (x - field_w * 0.6, y - 110, x + field_w * 0.6, y + 40)


# ---------------------------------------------------------------------------
# 6. @element("physics_ball")
# ---------------------------------------------------------------------------
@element("physics_ball", dur=0.2, enter="fade", sfx="none")
def el_physics_ball(ctx, el, s):
    r = float(el.get("r", 30))
    col = _color(el.get("color", "#ffffff"))
    wps = el.get("waypoints") or []
    t = s.t

    bx, by = xy(el, t)
    sx, sy, rot = 1.0, 1.0, 0.0

    if len(wps) >= 2:
        if t <= float(wps[0]["t"]):
            bx, by = float(wps[0]["x"]), float(wps[0]["y"])
        elif t >= float(wps[-1]["t"]):
            bx, by = float(wps[-1]["x"]), float(wps[-1]["y"])
            dt = t - float(wps[-1]["t"])
            imp = spring_impulse(dt, 38.0, 0.65) * 0.28
            sx, sy = 1.0 + imp, 1.0 / (1.0 + imp)
        else:
            for i in range(len(wps) - 1):
                w0, w1 = wps[i], wps[i + 1]
                t0, t1 = float(w0["t"]), float(w1["t"])
                if t0 <= t <= t1:
                    dt = t - t0
                    u = dt / max(1e-4, t1 - t0)
                    mode = w1.get("mode", "bounce")
                    lift = float(w1.get("arc", 115.0))
                    if mode == "linear":
                        bx = lerp(float(w0["x"]), float(w1["x"]), ease_out_cubic(u))
                        by = lerp(float(w0["y"]), float(w1["y"]), ease_out_cubic(u))
                    elif mode == "roll":
                        prog_r = 1.0 - (1.0 - u) ** 1.85
                        bx = lerp(float(w0["x"]), float(w1["x"]), prog_r)
                        by = lerp(float(w0["y"]), float(w1["y"]), prog_r)
                    elif mode == "rebound":
                        bx, by = rebound_drop_arc(float(w0["x"]), float(w0["y"]), float(w1["x"]), float(w1["y"]), u, lift)
                    else:
                        bx, by = bounce_arc(float(w0["x"]), float(w0["y"]), float(w1["x"]), float(w1["y"]), u, lift)
                    imp = spring_impulse(dt, 40.0, 0.65) * float(w0.get("squash", 0.32))
                    sx = 1.0 + imp
                    sy = 1.0 / max(0.4, sx)
                    break

    ctx.save()
    ctx.translate(bx, by)
    if rot:
        ctx.rotate(rot)
    ctx.scale(sx * s.scale, sy * s.scale)
    ctx.arc(0, 0, r, 0, math.pi * 2)
    ctx.set_source_rgba(col[0], col[1], col[2], col[3] * s.alpha)
    ctx.fill()
    ctx.restore()
    return (bx - r * 1.5, by - r * 1.5, bx + r * 1.5, by + r * 1.5)


# ---------------------------------------------------------------------------
# 7. @element("ui_film_world") — Complete 10s Rube Goldberg UI Motion Film Macro
# ---------------------------------------------------------------------------
@element("ui_film_world", dur=0.1, enter="none", sfx="none")
def el_ui_film_world(ctx, el, s):
    """Renders the complete, reference-accurate 10s Rube Goldberg UI Motion Film world
    (Toggle -> Slider -> 3 Notification Cards -> Reply Pill -> Keyboard + ISO Return -> Loop Return)
    centered around `(el.x, el.y)` so a single JSON element can drive the entire film!"""
    cx, cy = xy(el, s.t)
    t = (s.t - float(el.get("t", 0.0))) % float(el.get("loop_dur", 10.0))
    pal = pal_of(el, s)
    acc = el.get("accent") or pal.get("accent", "#0a84ff")
    fa = bool(el.get("fa", False))
    cards_cfg = el.get("cards") or [
        {"title": "اسلایدر" if fa else "Slider", "body": "به ۱۰۰ رسید" if fa else "Reached 100", "icon": "slider"},
        {"title": "جاذبه" if fa else "Gravity", "body": "سقوط آزاد توپ" if fa else "Something is falling", "icon": "gravity"},
        {"title": "پیام‌ها" if fa else "Messages", "body": "حالا چی؟" if fa else "What now?", "icon": "messages"},
    ]
    word = str(el.get("word", "retry"))

    ctx.save()
    ctx.translate(cx, cy)

    # World coordinates matching ui_motion_film.html
    tg_cx, tg_cy, tg_w, tg_h = -550.0, -240.0, 154.0, 78.0
    tg_lx, tg_rx = -588.0, -512.0
    sl_x0, sl_x1, sl_y = -250.0, 390.0, -240.0
    c_ys = [-10.0, 100.0, 210.0]
    c_ap = [1.70, 2.10, 2.30]
    c_hit = [2.50, 3.00, 3.50]
    c_fade = [2.76, 3.26, 3.76]
    ball_r = 30.0

    # Key map
    r1_labels = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"]
    r2_labels = ["A", "S", "D", "F", "G", "H", "J", "K", "L"]
    kmap = {}
    klist = []
    for idx, lbl in enumerate(r1_labels):
        ko = {"id": lbl.lower(), "label": lbl, "x": 428.0 + (idx - 3) * 80.0, "y": 365.0, "w": 68.0, "h": 68.0, "ret": False}
        kmap[ko["id"]] = ko
        klist.append(ko)
    for idx, lbl in enumerate(r2_labels):
        ko = {"id": lbl.lower(), "label": lbl, "x": 428.0 + (idx - 2.65) * 80.0, "y": 445.0, "w": 68.0, "h": 68.0, "ret": False}
        kmap[ko["id"]] = ko
        klist.append(ko)
    p_key = kmap["p"]
    l_top_x = p_key["x"] + 34.0 + 12.0
    ret_obj = {"id": "return", "label": "return", "x": l_top_x + 62.0, "y": 405.0, "w": 124.0, "bot_w": 98.0, "h": 148.0, "top_h": 68.0, "ret": True}
    kmap["return"] = ret_obj
    klist.append(ret_obj)

    letters = [ch.lower() for ch in (word[:5] if len(word) >= 5 else "retry")]
    khits = [
        {"t": 4.00, "id": letters[0] if letters[0] in kmap else "r", "char": word[0] if len(word) > 0 else "r"},
        {"t": 4.50, "id": letters[1] if len(letters) > 1 and letters[1] in kmap else "e", "char": word[1] if len(word) > 1 else "e"},
        {"t": 5.00, "id": letters[2] if len(letters) > 2 and letters[2] in kmap else "t", "char": word[2] if len(word) > 2 else "t"},
        {"t": 5.50, "id": letters[3] if len(letters) > 3 and letters[3] in kmap else "r", "char": word[3] if len(word) > 3 else "r"},
        {"t": 6.00, "id": letters[4] if len(letters) > 4 and letters[4] in kmap else "y", "char": word[4] if len(word) > 4 else "y"},
        {"t": 7.00, "id": "return", "char": ""},
    ]

    # Evaluate Ball
    c0_top = c_ys[0] - 46.0 - ball_r + 2.0
    c1_top = c_ys[1] - 46.0 - ball_r + 2.0
    c2_top = c_ys[2] - 46.0 - ball_r + 2.0
    key_top = 365.0 - 34.0 - ball_r + 2.0
    ret_top = 405.0 - 74.0 - ball_r + 2.0

    bx, by, bsx, bsy, brot = tg_lx, tg_cy, 1.0, 1.0, 0.0
    if t < 0.10:
        bx, by = tg_lx, tg_cy
    elif t < 0.18:
        u = (t - 0.10) / 0.08
        bx, by, bsx, bsy = lerp(tg_lx, tg_rx, ease_out_cubic(u)), tg_cy, 1.12, 0.89
    elif t < 0.50:
        bx, by = bounce_arc(tg_rx, tg_cy, sl_x0, sl_y, (t - 0.18) / 0.32, 115.0)
        bsx, bsy = 1.04, 0.96
    elif t < 2.12:
        imp = spring_impulse(t - 0.50, 36.0, 0.62) * 0.36
        ur = clamp((t - 0.54) / (2.10 - 0.54), 0.0, 1.0)
        bx, by = lerp(sl_x0, sl_x1, 1.0 - (1.0 - ur) ** 1.85), sl_y
        bsx = 1.0 + imp; bsy = 1.0 / bsx
    elif t < 2.50:
        u = (t - 2.12) / 0.38
        bx = lerp(sl_x1, 428.0, ease_out_quad(u))
        by = lerp(sl_y, c0_top, u ** 2.15)
        fs = math.sin(math.pi * u) * 0.16
        bsx, bsy = 1.0 - fs * 0.5, 1.0 + fs
    elif t < 3.00:
        dt = t - 2.50
        bx, by = rebound_drop_arc(428.0, c0_top + 8.0, 428.0, c1_top, dt / 0.50, 72.0)
        imp = spring_impulse(dt, 42.0, 0.65) * 0.34
        bsx = 1.0 + imp; bsy = 1.0 / bsx
    elif t < 3.50:
        dt = t - 3.00
        bx, by = rebound_drop_arc(428.0, c1_top + 8.0, 428.0, c2_top, dt / 0.50, 72.0)
        imp = spring_impulse(dt, 42.0, 0.65) * 0.34
        bsx = 1.0 + imp; bsy = 1.0 / bsx
    elif t < 4.00:
        dt = t - 3.50
        bx, by = rebound_drop_arc(428.0, c2_top + 8.0, kmap[khits[0]["id"]]["x"], key_top, dt / 0.50, 68.0)
        imp = spring_impulse(dt, 42.0, 0.65) * 0.34
        bsx = 1.0 + imp; bsy = 1.0 / bsx
    elif t < 6.00:
        idx = min(3, int((t - 4.00) // 0.50))
        dt = t - (4.00 + idx * 0.50)
        k0 = kmap[khits[idx]["id"]]
        k1 = kmap[khits[idx + 1]["id"]]
        bx, by = bounce_arc(k0["x"], key_top + 8.0, k1["x"], key_top, dt / 0.50, 120.0)
        imp = spring_impulse(dt, 42.0, 0.65) * 0.32
        bsx = 1.0 + imp; bsy = 1.0 / bsx
    elif t < 7.00:
        dt = t - 6.00
        k0 = kmap[khits[4]["id"]]
        bx, by = bounce_arc(k0["x"], key_top + 8.0, ret_obj["x"] + 12.0, ret_top, dt / 1.00, 420.0)
        imp = spring_impulse(dt, 42.0, 0.65) * 0.34
        bsx = 1.0 + imp; bsy = 1.0 / bsx
    elif t < 7.62:
        dt = t - 7.00
        imp = spring_impulse(dt, 38.0, 0.64) * 0.34
        hd = smoothstep(dt / 0.12) * 10.0 if dt < 0.45 else 10.0 * (1.0 - smoothstep((dt - 0.45) / 0.17))
        bx, by = ret_obj["x"] + 12.0, ret_top + hd
        bsx = 1.0 + imp; bsy = 1.0 / bsx
    elif t < 8.80:
        u = (t - 7.62) / (8.80 - 7.62)
        bx, by = bounce_arc(ret_obj["x"] + 12.0, ret_top, tg_rx, tg_cy, u, 380.0)
        st = math.sin(math.pi * u) * 0.15
        bsx, bsy, brot = 1.0 + st, 1.0 - st * 0.45, -0.55
    else:
        dt = t - 8.80
        bx = lerp(tg_rx, tg_lx, spring_step(dt, 24.0, 0.72))
        by = tg_cy
        imp = spring_impulse(dt, 34.0, 0.68) * 0.18
        bsx = 1.0 + imp; bsy = 1.0 / bsx

    # 1. Draw Toggle
    bmix = clamp((t - 0.10) / 0.08, 0.0, 1.0) if 0.10 <= t < 8.84 else (1.0 - clamp((t - 8.84) / 0.12, 0.0, 1.0) if 8.84 <= t < 8.96 else 0.0)
    _fill_round_rect(ctx, tg_cx - tg_w * 0.5, tg_cy - tg_h * 0.5, tg_w, tg_h, tg_h * 0.5, "#3a3a42", None, 0, s.alpha)
    if bmix > 0.001:
        _fill_round_rect(ctx, tg_cx - tg_w * 0.5, tg_cy - tg_h * 0.5, tg_w, tg_h, tg_h * 0.5, acc, None, 0, s.alpha * bmix)

    # 2. Draw Slider
    sl_prog = 0.0 if t < 0.50 else (clamp((bx - sl_x0) / (sl_x1 - sl_x0), 0.0, 1.0) if t <= 2.12 else (1.0 if t <= 7.80 else ((1.0 - clamp((t - 7.80) / 0.68, 0.0, 1.0)) ** 2.2 if t <= 8.48 else 0.0)))
    fill_x = lerp(sl_x0, sl_x1, sl_prog)
    _fill_round_rect(ctx, sl_x0, sl_y - 4.0, sl_x1 - sl_x0, 8.0, 4.0, "#32323a", None, 0, s.alpha)
    if sl_prog > 0.002:
        _fill_round_rect(ctx, sl_x0, sl_y - 4.0, max(8.0, fill_x - sl_x0), 8.0, 4.0, acc, None, 0, s.alpha)
    for i in range(11):
        tf = i / 10.0
        tx = lerp(sl_x0, sl_x1, tf)
        _fill_round_rect(ctx, tx - 2.0, -194.0 - 7.0, 4.0, 14.0, 2.0, acc if (sl_prog > 0.005 and tf <= sl_prog + 0.005) else "#32323a", None, 0, s.alpha)
    b_sc = (spring_step(t - 0.52, 32.0, 0.68) * (1.0 - smoothstep((t - 2.12) / 0.10) if t > 2.12 else 1.0)) if 0.52 <= t <= 2.22 else ((smoothstep((t - 7.80) / 0.08) * (1.0 - smoothstep((t - 8.38) / 0.10) if t > 8.38 else 1.0)) if 7.80 <= t <= 8.48 else 0.0)
    if b_sc > 0.01:
        ctx.save()
        ctx.translate(fill_x, sl_y - 68.0)
        ctx.scale(b_sc, b_sc)
        _fill_round_rect(ctx, -32.0, -21.0, 64.0, 42.0, 11.0, "#2c2c34", None, 0, s.alpha)
        vtxt = to_fa_digits(str(int(round(sl_prog * 100)))) if fa else str(int(round(sl_prog * 100)))
        draw_text(ctx, vtxt, 0, 0, size=20, color="#ffffff", weight=F(el, "counter"), align="center", alpha=s.alpha)
        ctx.restore()

    # 3. Draw 3 Notification Cards
    for idx in range(3):
        cc = cards_cfg[idx] if idx < len(cards_cfg) else {"title": "Card", "body": "", "icon": "messages"}
        ca, cy_off = 0.0, 0.0
        if c_ap[idx] <= t <= c_fade[idx] + 0.18:
            ca = smoothstep(clamp((t - c_ap[idx]) / 0.18, 0.0, 1.0))
            cy_off = (1.0 - spring_step(t - c_ap[idx], 26.0, 0.72)) * 40.0
            if t >= c_hit[idx]:
                cy_off += spring_impulse(t - c_hit[idx], 32.0, 0.65) * 16.0
            if t > c_fade[idx]:
                fu = clamp((t - c_fade[idx]) / 0.16, 0.0, 1.0)
                ca *= (1.0 - fu); cy_off -= fu * 18.0
        elif 7.92 <= t <= 8.88:
            re = clamp((t - (7.92 + idx * 0.08)) / 0.20, 0.0, 1.0)
            rx_out = 1.0 - clamp((t - 8.68) / 0.18, 0.0, 1.0) if t > 8.68 else 1.0
            ca = smoothstep(re) * smoothstep(rx_out)
            cy_off = (1.0 - smoothstep(re)) * 24.0
        if ca > 0.01:
            rx, ry = 428.0 - 198.0, c_ys[idx] + cy_off - 46.0
            _fill_round_rect(ctx, rx, ry, 396.0, 92.0, 20.0, "#1c1c21", "#2c2c34", 1.4, s.alpha * ca)
            _fill_round_rect(ctx, rx + 20.0, ry + 23.0, 46.0, 46.0, 11.0, acc, None, 0, s.alpha * ca)
            _draw_card_icon(ctx, cc.get("icon", "messages"), rx + 43.0, ry + 46.0, s.alpha * ca)
            draw_text(ctx, cc["title"], rx + 82.0, ry + 32.0, size=19, color="#ffffff", weight=F(el, "ui_title"), align="left", alpha=s.alpha * ca)
            draw_text(ctx, cc["body"], rx + 82.0, ry + 61.0, size=18, color="#adadb5", weight=F(el, "body"), align="left", alpha=s.alpha * ca)
            draw_text(ctx, "الان" if fa else "now", rx + 396.0 - 22.0, ry + 32.0, size=15, color="#787882", weight=F(el, "ui_label"), align="right", alpha=s.alpha * ca)

    # 4. Draw Reply Pill & Sent Bubble
    if 2.55 <= t <= 6.45:
        ea = clamp((t - 2.55) / 0.25, 0.0, 1.0) * s.alpha
        rw = lerp(196.0, 240.0, spring_step(t - 3.90, 26.0, 0.70)) if t >= 3.90 else 196.0
        rx, ry = 75.0 - rw * 0.5, 218.0 - 28.0
        _fill_round_rect(ctx, rx, ry, rw, 56.0, 28.0, "#24242b", "#383842", 1.6, ea)
        if t < 3.92:
            draw_text(ctx, "پاسخ" if fa else "Reply", rx + 24.0, 218.0, size=19, color="#787882", weight=F(el, "ui_label"), align="left", alpha=ea)
        else:
            typed = "".join(kh["char"] for kh in khits[:5] if t >= kh["t"])
            draw_text(ctx, typed, rx + 26.0, 218.0, size=21, color="#ffffff", weight=F(el, "ui_title"), align="left", alpha=ea)
            tw, _ = measure_text(typed, 21, F(el, "ui_title")) if typed else (0.0, 0.0)
            _fill_round_rect(ctx, rx + 26.0 + tw + 4.0, 218.0 - 13.0, 2.4, 26.0, 1.2, acc, None, 0, ea)
            _fill_round_rect(ctx, rx + rw - 45.0, 218.0 - 17.0, 34.0, 34.0, 17.0, acc, None, 0, ea)
    elif 7.85 <= t <= 8.85:
        wa = smoothstep(clamp((t - 7.85) / 0.18, 0.0, 1.0)) * (1.0 - smoothstep((t - 8.68) / 0.17) if t > 8.68 else 1.0) * s.alpha
        _fill_round_rect(ctx, 75.0 - 85.0, 218.0 - 24.0, 170.0, 48.0, 24.0, "#24242b", "#383842", 1.5, wa)
        draw_text(ctx, "پاسخ" if fa else "Reply", 75.0 - 63.0, 218.0, size=17, color="#787882", weight=F(el, "ui_label"), align="left", alpha=wa)
        _fill_round_rect(ctx, 145.0 - 56.0, 142.0 - 25.0, 112.0, 50.0, 25.0, acc, None, 0, wa)
        draw_text(ctx, word, 145.0, 142.0, size=18, color="#ffffff", weight=F(el, "ui_title"), align="center", alpha=wa)

    # 5. Draw Keyboard (visible t=3.60..8.85)
    if 3.60 <= t <= 8.88:
        ka = smoothstep(clamp((t - 3.60) / 0.25, 0.0, 1.0)) * (1.0 - smoothstep((t - 8.68) / 0.18) if t > 8.68 else 1.0) * s.alpha
        for k in klist:
            dep = 0.0; act = False
            for kh in khits:
                if kh["id"] == k["id"] and t >= kh["t"]:
                    dt = t - kh["t"]
                    if k["ret"] and dt < 0.62:
                        dep = smoothstep(dt / 0.12) * 10.0 if dt < 0.45 else 10.0 * (1.0 - smoothstep((dt - 0.45) / 0.17))
                        act = True
                    elif not k["ret"] and dt < 0.30:
                        dep = max(0.0, spring_impulse(dt, 38.0, 0.65)) * 10.0
                        act = dt < 0.22
            kx, ky = k["x"], k["y"] + dep
            fc = "#3e3e4a" if act else "#2c2c34"
            if k["ret"]:
                rx, ry = kx - k["w"] * 0.5, ky - k["h"] * 0.5
                _iso_return_path(ctx, rx, ry + 4, k["w"], k["bot_w"], k["h"], k["top_h"], 13.0)
                ctx.set_source_rgba(0.08, 0.08, 0.10, ka); ctx.fill()
                _iso_return_path(ctx, rx, ry, k["w"], k["bot_w"], k["h"], k["top_h"], 13.0)
                c = _color(fc); ctx.set_source_rgba(c[0], c[1], c[2], ka); ctx.fill_preserve()
                bc = _color("#3d3d47"); ctx.set_source_rgba(bc[0], bc[1], bc[2], ka); ctx.set_line_width(1.5); ctx.stroke()
                draw_text(ctx, "return", rx + k["w"] - 18.0, ry + k["h"] - 20.0, size=14, color="#adadb5", weight=F(el, "ui_label"), align="right", alpha=ka)
            else:
                rx, ry = kx - k["w"] * 0.5, ky - k["h"] * 0.5
                _fill_round_rect(ctx, rx, ry + 4, k["w"], k["h"], 13.0, "#141418", None, 0, ka)
                _fill_round_rect(ctx, rx, ry, k["w"], k["h"], 13.0, fc, "#3d3d47", 1.5, ka)
                draw_text(ctx, k["label"], kx, ky, size=22, color="#ffffff", weight=F(el, "ui_title"), align="center", alpha=ka)

    # 6. Draw Cursor
    if t <= 0.70 or t >= 7.80:
        cur_y = tg_cy + 28.0 if t <= 0.70 else tg_cy + 140.0
        _draw_cursor(ctx, tg_cx - 24.0, cur_y, scale=1.0, alpha=s.alpha)

    # 7. Draw Hero Ball
    ctx.save()
    ctx.translate(bx, by)
    if brot:
        ctx.rotate(brot)
    ctx.scale(bsx, bsy)
    ctx.arc(0, 0, ball_r, 0, math.pi * 2)
    ctx.set_source_rgba(1.0, 1.0, 1.0, s.alpha)
    ctx.fill()
    ctx.restore()

    ctx.restore()
    return (cx - 700, cy - 350, cx + 950, cy + 520)
