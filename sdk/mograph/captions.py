"""
mograph.captions — v12 reel captions ("Hormozi / CapCut" style), used by hud.draw_subs when
hud.subs.style == "reel".

  "hud": {"subs": {"style": "reel", "y": 0.70, "size": 84, "weight": "title",
                   "color": "#ffffff", "emph": "#ffd60a", "active_bg": "#ff2d55", "active_text": "#ffffff",
                   "reveal": "word" | "group",  "stroke": "#000000", "stroke_w": 10, "shadow": 0.55,
                   "pop": 0.18, "max_width": 0.84, "upper": false, "rotate": 0}}

* group enters with a spring pop (scale 0.82 → 1, slight rise), leaves in 0.08 s
* reveal "word": each word pops in on its own timestamp; "group": whole group visible, active word highlighted
* active word: rounded slab in active_bg behind it (springs in), text in active_text
* emphasised words (**word** or edit.emph) stay in the emph colour and are 8 % larger
* thick dark outline + soft offset shadow → readable on any footage (no contrast box needed)
Alternative position per group: subtitle dict may carry "y" (0..1) or "pos": "top" | "mid" | "low".
"""
from __future__ import annotations

from .core import clamp, color as _color, spring
from .geometry import compile_path, rounded_rect_d
from .text import draw_text, word_boxes


def draw_reel(ctx, W, H, cfg, sub, t, alpha):
    words = sub["words"]
    text = " ".join(w["w"] for w in words)
    size = float(cfg.get("size", 84))
    weight = cfg.get("weight", "title")
    maxw = W * float(cfg.get("max_width", 0.84))
    pos = sub.get("pos")
    y = H * float(sub.get("y", {"top": 0.22, "mid": 0.5, "low": 0.78}.get(pos, cfg.get("y", 0.70))))
    x = W / 2
    t0 = sub["t"]
    sp = spring(t, t0, stiffness=420, damping=20)
    out_a = clamp((sub["end"] - t) / 0.08) if t > sub["end"] - 0.08 else 1.0
    a = alpha * clamp((t - t0) / 0.06) * out_a
    if a <= 0.003:
        return
    pop = float(cfg.get("pop", 0.18))
    sc = (1 - pop) + pop * sp
    col = cfg.get("color", "#ffffff")
    emph = cfg.get("emph", "#ffd60a")
    abg = cfg.get("active_bg", "#ff2d55")
    atx = cfg.get("active_text", "#ffffff")
    reveal = cfg.get("reveal", "word")
    stroke = cfg.get("stroke", "#000000")
    sw = float(cfg.get("stroke_w", size * 0.12))
    shadow = float(cfg.get("shadow", 0.55))
    ctx.save()
    ctx.translate(x, y + (1 - sp) * size * 0.25)
    ctx.scale(max(1e-3, sc), max(1e-3, sc))
    if cfg.get("rotate"):
        import math
        ctx.rotate(math.radians(float(cfg["rotate"])))
    ctx.translate(-x, -y)
    act = max([i for i, w in enumerate(words) if t >= w["t"]], default=-1)

    def ws_factory(mode):
        def ws(k):
            if k >= len(words):
                return None
            w = words[k]
            on = t >= w["t"] - 0.02
            wp = spring(t, w["t"] - 0.02, stiffness=520, damping=18) if on else 0.0
            if reveal == "word" and not on:
                return {"alpha": 0.0}
            st = {}
            big = 1.08 if w.get("emph") else 1.0
            st["scale"] = big * ((0.7 + 0.3 * wp) if reveal == "word" else (1.0 + 0.06 * wp * (k == act)))
            if mode == "shadow":
                st["color"] = (0, 0, 0, shadow)
            elif mode == "stroke":
                st["color"] = stroke
            else:
                if k == act and abg:
                    st["color"] = atx
                elif w.get("emph"):
                    st["color"] = emph
                else:
                    st["color"] = col if (on or reveal == "word") else (*_color(col)[:3], 0.55)
            return st
        return ws

    # active-word slab (behind the glyphs)
    if abg and act >= 0:
        bxs = word_boxes(text, x, y, size, weight, max_width=maxw, line_height=1.25)
        if act in bxs:
            bx0, by0, bx1, by1 = bxs[act]
            ap = spring(t, words[act]["t"] - 0.02, stiffness=600, damping=22)
            pad_x, pad_y = size * 0.18, size * 0.06
            cxm, cym = (bx0 + bx1) / 2, (by0 + by1) / 2 + size * 0.02
            bw = (bx1 - bx0 + 2 * pad_x) * (0.6 + 0.4 * ap) * (1.08 if words[act].get("emph") else 1.0)
            bh = (by1 - by0 + 2 * pad_y) * (0.6 + 0.4 * ap)
            c = _color(abg)
            ctx.new_path()
            ctx.append_path(compile_path(rounded_rect_d(cxm - bw / 2, cym - bh / 2, bw, bh, bh * 0.22)).whole)
            ctx.set_source_rgba(c[0], c[1], c[2], c[3] * a * clamp(ap * 1.5))
            ctx.fill()
    kw = dict(size=size, weight=weight, max_width=maxw, line_height=1.25, quiet=False)
    from . import text as _text
    _mute = _text.AUDIT_MUTE
    _text.AUDIT_MUTE = "all"                      # shadow + outline passes are not the readable layer
    if shadow > 0:
        ctx.save(); ctx.translate(size * 0.04, size * 0.07)
        draw_text(ctx, text, x, y, alpha=a, word_style=ws_factory("shadow"), stroke="#000000",
                  stroke_width=sw * 1.25, color="#000000", **kw)
        ctx.restore()
    if sw > 0:
        draw_text(ctx, text, x, y, alpha=a, word_style=ws_factory("stroke"), stroke=stroke, stroke_width=sw,
                  color=stroke, **kw)
    _text.AUDIT_MUTE = "all"                      # outlined captions are readable on any footage by construction
    bb = draw_text(ctx, text, x, y, alpha=a, word_style=ws_factory("fill"), color=col, **kw)
    _text.AUDIT_MUTE = _mute
    ctx.restore()
    return bb


def draw_novadent_jelly(ctx, W, H, cfg, sub, t, alpha):
    """
    v13.9 @novadent.ir Elastic Jelly Glassmorphic Subtitle Bar (Screenshot 211936):
      * Real backdrop-blur glassmorphism (blurs the video pixels directly underneath the capsule)
      * Volume-preserving Jelly Squash-and-Stretch physics when subtitle width changes (w_t, h_t)
      * Glowing cyan-emerald (#2DD4BF) crescent rim highlights on the left and right curved caps
      * Silky Ease-In-Out word reveal inside the jelly capsule
    """
    import math
    import cairo
    import cv2
    import numpy as np
    from .text import resolve_font, text_bbox

    words = sub["words"]
    text = " ".join(w["w"] for w in words)
    size = float(cfg.get("size", 40))
    fnt = resolve_font(cfg.get("weight", "subtitle"), "bold", "subtitle")
    y = H * float(sub.get("y", cfg.get("y", 0.655)))
    x = W * 0.5
    t0 = float(sub["t"])
    t_end = float(sub["end"])
    dt = max(0.0, t - t0)

    # Smooth Ease-In-Out entry & exit alpha
    in_a = clamp(dt / 0.14)
    out_a = clamp((t_end - t) / 0.12) if t > t_end - 0.12 else 1.0
    a = alpha * in_a * out_a
    if a <= 0.005:
        return

    # Build rich mixed-font text string (auto-emphasize word with "emph": True or **word** or longest key noun!)
    rich_tokens = []
    has_explicit_emph = any(w.get("emph") or ("**" in w["w"]) or ("__" in w["w"]) for w in words)
    # If no explicit emph is marked in words, automatically highlight the most impactful content word in the middle!
    auto_idx = -1
    if not has_explicit_emph and len(words) >= 2:
        candidates = [(i, len(w["w"])) for i, w in enumerate(words) if w["w"] not in ("و", "رو", "را", "از", "به", "در", "که", "هم", "یا", "میکنن", "میشن", "کنید")]
        if candidates:
            auto_idx = max(candidates, key=lambda item: item[1])[0]
    for i, w in enumerate(words):
        wt = w["w"]
        if w.get("emph") or i == auto_idx:
            if not (wt.startswith("**") or wt.startswith("__")):
                wt = f"**{wt}**"
        rich_tokens.append(wt)
    rich_text = " ".join(rich_tokens)

    # Measure target text width accounting for 1.38x emphasized word scale
    clean_text = text.replace("**", "").replace("__", "")
    bb_txt = text_bbox(clean_text, 0, 0, size=size * 1.12, weight=fnt)
    tw = max(150.0, bb_txt[2] - bb_txt[0])
    target_w = tw + size * 1.85
    target_h = size * 1.58

    # Elastic Jelly Squash & Stretch oscillation (damped under-damped spring with volume conservation)
    # Starts slightly narrower/taller (or from prev width), overshoots horizontally while squashing vertically,
    # and wobbles like real jelly!
    omega = 24.0
    zeta = 0.32
    env_j = math.exp(-zeta * omega * dt)
    jelly_osc = 1.0 - 0.28 * env_j * math.cos(omega * math.sqrt(1 - zeta**2) * dt)
    w_cur = target_w * jelly_osc
    # Volume conservation: when width stretches (>1), height slightly squashes (<1), and vice versa!
    h_cur = target_h * ((1.0 / max(0.65, jelly_osc)) ** 0.42)
    rad = h_cur * 0.48

    x0, y0 = x - w_cur * 0.5, y - h_cur * 0.5

    # 1. Real Backdrop-Blur Glassmorphism from underlying Cairo surface
    surf = ctx.get_target()
    if isinstance(surf, cairo.ImageSurface):
        surf.flush()
        sw, sh = surf.get_width(), surf.get_height()
        ix0 = max(0, int(x0 - 8))
        iy0 = max(0, int(y0 - 8))
        ix1 = min(sw, int(x0 + w_cur + 8))
        iy1 = min(sh, int(y0 + h_cur + 8))
        if ix1 > ix0 + 4 and iy1 > iy0 + 4:
            buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape((sh, sw, 4))
            roi = buf[iy0:iy1, ix0:ix1, :3].copy()
            blurred = cv2.GaussianBlur(roi, (29, 29), 10.0)
            # Dark emerald-obsidian glass tint
            blurred = (blurred.astype(np.float32) * 0.36 + np.array([18.0, 24.0, 14.0], dtype=np.float32)).clip(0, 255).astype(np.uint8)
            bgra_roi = np.dstack([blurred, np.full(blurred.shape[:2], 255, dtype=np.uint8)])
            r_surf = cairo.ImageSurface.create_for_data(bgra_roi, cairo.FORMAT_ARGB32, ix1 - ix0, iy1 - iy0)
            ctx.save()
            ctx.new_path()
            ctx.append_path(compile_path(rounded_rect_d(x0, y0, w_cur, h_cur, rad)).whole)
            ctx.clip()
            ctx.set_source_surface(r_surf, ix0, iy0)
            ctx.paint_with_alpha(0.92 * a)
            ctx.restore()

    # 2. Outer soft cyan-emerald ambient glow
    rim_col = _color(cfg.get("rim_color", "#2DD4BF"))
    for gw, ga in [(10.0, 0.14), (4.5, 0.28)]:
        ctx.save()
        ctx.new_path()
        ctx.append_path(compile_path(rounded_rect_d(x0, y0, w_cur, h_cur, rad)).whole)
        ctx.set_line_width(gw)
        ctx.set_source_rgba(rim_col[0], rim_col[1], rim_col[2], ga * a)
        ctx.stroke()
        ctx.restore()

    # 3. Signature @novadent glowing cyan-teal crescent caps on Left & Right ends + subtle top/bottom rim
    ctx.save()
    ctx.new_path()
    ctx.append_path(compile_path(rounded_rect_d(x0, y0, w_cur, h_cur, rad)).whole)
    lg = cairo.LinearGradient(x0, y, x0 + w_cur, y)
    lg.add_color_stop_rgba(0.0, rim_col[0], rim_col[1], rim_col[2], 0.98 * a)
    lg.add_color_stop_rgba(0.16, rim_col[0], rim_col[1], rim_col[2], 0.25 * a)
    lg.add_color_stop_rgba(0.50, 1.0, 1.0, 1.0, 0.18 * a)
    lg.add_color_stop_rgba(0.84, rim_col[0], rim_col[1], rim_col[2], 0.25 * a)
    lg.add_color_stop_rgba(1.0, rim_col[0], rim_col[1], rim_col[2], 0.98 * a)
    ctx.set_line_width(2.6)
    ctx.set_source(lg)
    ctx.stroke()
    ctx.restore()

    # 4. Crisp Persian text inside the jelly capsule with Ease-In-Out reveal
    from . import text as _text
    _mute = _text.AUDIT_MUTE
    _text.AUDIT_MUTE = "all"
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(1.0 + 0.04 * (jelly_osc - 1.0), 1.0)
    from .creative_reels import draw_mixed_font_line
    draw_mixed_font_line(
        ctx, rich_text, 0, 2,
        base_size=size * 0.92,
        thin_font=cfg.get("thin_font", "IRANYekanLight.ttf"),
        bold_font=cfg.get("bold_font", "IRANYekanExtraBlack2.ttf"),
        callig_font=cfg.get("callig_font", "Delbar-Bold.ttf"),
        base_color=cfg.get("color", "#E2E8F0"),
        emph_color=cfg.get("emph", "#FBBF24"),
        emph_scale=float(cfg.get("emph_scale", 1.38)),
        alpha=a
    )
    ctx.restore()
    _text.AUDIT_MUTE = _mute
