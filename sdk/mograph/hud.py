"""
mograph.hud — persistent screen-space overlay: chapter header notch, chapter dots,
word-level karaoke subtitles, handle/watermark.  All configurable via spec["hud"].

spec["hud"] = {
  "header": {"prefix": "پرونده:", "y": 0.055, "show": true},
  "dots":   {"y": 0.82, "style": "dots" | "numbers", "show": true},
  "subs":   {"y": 0.885, "size": 50, "max_width": 0.86, "lines": 2,
             "spoken": "#ffffff", "upcoming": "#4d4d52", "emph": "#ffffff"},
  "handle": "@MY.PAGE"   (optional)
  "fade_out_at": 144.5   (optional, seconds — HUD fades before the end card)
}
Subtitles come from spec["subtitles"]:
  sentence form: {"t": 0, "end": 4, "text": "**شما** انسان‌ها ..."}   (words timed by length)
  word form    : {"t": 0, "end": 4, "words": [{"w": "شما", "t": 0.0, "end": 0.4, "emph": true}, ...]}
"""
from __future__ import annotations

import math

from .core import clamp, color as _color, ease_in_out, ease_out, lerp_tuple, prog
from .geometry import circle_d, draw_path
from .text import draw_text, layout_text, parse_markup, text_bbox, to_fa_digits
from . import contrast as _ct

_BLACK_BG = (0.0, 0.0, 0.0, 1.0)


def _env(env, key):
    return (env or {}).get(key, _BLACK_BG)

DEFAULT_HUD = {
    "header": {"prefix": "پرونده:", "y": 0.055, "show": True, "size": 42},
    "dots": {"y": 0.82, "style": "dots", "show": True},
    "subs": {"y": 0.885, "size": 50, "max_width": 0.86, "spoken": "#ffffff", "upcoming": "#4d4d52",
             "emph": "#ffffff", "show": True, "lead": 0.0},
    "handle": None,
}


def merged_hud(spec):
    h = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_HUD.items()}
    for k, v in (spec.get("hud") or {}).items():
        if isinstance(v, dict) and isinstance(h.get(k), dict):
            h[k].update(v)
        else:
            h[k] = v
    return h


# ---------------------------------------------------------------------------
# subtitle timing
# ---------------------------------------------------------------------------

def _weight(w):
    return max(1.0, len(w.replace("\u200c", ""))) + 1.2  # +pause per word


def prepare_subtitles(subs):
    """Normalise to: [{"t","end","words":[{"w","t","end","emph"}], "text"}]."""
    out = []
    for s in subs or []:
        t0, t1 = float(s["t"]), float(s["end"])
        if "words" in s:
            words = []
            for w in s["words"]:
                words.append({"w": w["w"], "t": float(w["t"]), "end": float(w.get("end", w["t"] + 0.3)),
                              "emph": bool(w.get("emph", False))})
        else:
            clean, emph = parse_markup(s["text"])
            toks = clean.split()
            tot = sum(_weight(w) for w in toks) or 1
            span = (t1 - t0) * float(s.get("speech_ratio", 0.9))
            acc = t0
            words = []
            for i, w in enumerate(toks):
                d = span * _weight(w) / tot
                words.append({"w": w, "t": acc, "end": acc + d, "emph": i in emph})
                acc += d
        o = {"t": t0, "end": t1, "words": words, "text": " ".join(w["w"] for w in words)}
        for k in ("pos", "y", "hide"):            # v12: per-group caption placement
            if k in s:
                o[k] = s[k]
        out.append(o)
    out.sort(key=lambda s: s["t"])
    return out


def active_subtitle(subs, t):
    for s in subs:
        if s["t"] <= t < s["end"]:
            return None if s.get("hide") else s
    return None


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def draw_header(ctx, W, H, cfg, title, alpha, change_p=1.0, env=None):
    if not cfg.get("show", True) or not title:
        return
    y = H * cfg.get("y", 0.055)
    size = cfg.get("size", 42)
    m = W * 0.05
    prefix = cfg.get("prefix", "")
    tb = text_bbox(title, 0, 0, size)
    pb = text_bbox(prefix, 0, 0, size * 0.7) if prefix else (0, 0, 0, 0)
    notch_w = (tb[2] - tb[0]) + (pb[2] - pb[0]) + size * 1.4
    drop = size * 1.05
    x_r = W - m
    x_notch = x_r - notch_w
    d = f"M {m} {y} L {x_notch - drop} {y} L {x_notch} {y + drop} L {x_r} {y + drop}"
    bg = _env(env, "header")
    c_line = _ct.ensure(cfg.get("line", "#8a8a90"), bg, 2.2)
    c_pre = _ct.ensure(cfg.get("prefix_color", "#8a8a90"), bg, 3.0)
    c_title = _ct.ensure(cfg.get("color", "#ffffff"), bg, 4.5, prefer=(cfg.get("color_alt"), "#111111", "#ffffff"))
    draw_path(ctx, d, 0, 0, 1, anchor="none", stroke=c_line, width=2.0, alpha=alpha, fillet=6)
    ty = y + drop - size * 0.72
    # prefix (gray, small) on the far right, title (white) left of it
    xr = x_r - size * 0.2
    if prefix:
        draw_text(ctx, prefix, xr, ty, size=size * 0.7, color=c_pre, align="right", alpha=alpha, weight="ui_label",
                  quiet=True)
        xr -= (pb[2] - pb[0]) + size * 0.35
    draw_text(ctx, title, xr, ty + 2 * (1 - change_p) * 10, size=size, color=c_title, align="right",
              alpha=alpha * change_p, weight="ui_title")


def draw_dots(ctx, W, H, cfg, n, idx, alpha, p_change=1.0, env=None):
    if not cfg.get("show", True) or n <= 1:
        return
    y = H * cfg.get("y", 0.82)
    gap = cfg.get("gap", 34)
    x0 = W / 2 + (n - 1) * gap / 2  # RTL: chapter 1 on the right
    style = cfg.get("style", "dots")
    bg = _env(env, "dots")
    c_on = _ct.ensure(cfg.get("on", "#ffffff"), bg, 4.5, prefer=("#111111", "#ffffff"))
    c_done = _ct.ensure(cfg.get("done", "#8a8a90"), bg, 2.6)
    c_off = _ct.ensure(cfg.get("off", "#3a3a3e"), bg, 1.6)
    c_ring = _ct.ensure("#6a6a6e", bg, 2.0)
    for i in range(n):
        x = x0 - i * gap
        if style == "numbers":
            col = c_on if i == idx else (c_done if i < idx else c_off)
            draw_text(ctx, to_fa_digits(i + 1), x, y, size=28 if i == idx else 22, color=col, alpha=alpha, weight="counter",
                      quiet=i != idx)
            continue
        if i == idx:
            r = 4 + 5 * ease_out(p_change)
            ctx.set_source_rgba(c_on[0], c_on[1], c_on[2], alpha)
            ctx.arc(x, y, r, 0, 2 * math.pi); ctx.fill()
            draw_path(ctx, circle_d(x, y, 15), 0, 0, 1, anchor="none", stroke=c_ring, width=1.4,
                      alpha=alpha * 0.8, trim=ease_out(p_change))
        elif i < idx:
            ctx.set_source_rgba(c_done[0], c_done[1], c_done[2], alpha * 0.9)
            ctx.arc(x, y, 4.5, 0, 2 * math.pi); ctx.fill()
        else:
            ctx.set_source_rgba(c_off[0], c_off[1], c_off[2], alpha)
            ctx.set_line_width(1.6)
            ctx.arc(x, y, 4.5, 0, 2 * math.pi); ctx.stroke()


def draw_subs(ctx, W, H, cfg, sub, t, alpha, env=None):
    if not cfg.get("show", True) or sub is None:
        return
    if cfg.get("style") in ("novadent_jelly", "jelly"):
        from .captions import draw_novadent_jelly
        draw_novadent_jelly(ctx, W, H, cfg, sub, t, alpha)
        return
    if cfg.get("style") == "reel":                 # v12: word-pop reel captions (mograph.captions)
        from .captions import draw_reel
        draw_reel(ctx, W, H, cfg, sub, t, alpha)
        return
    size = cfg.get("size", 50) * (min(W, H) / 1080.0 if W > H * 1.2 else 1.0)    # v9: landscape → scale
    maxw = W * cfg.get("max_width", 0.86 if W <= H * 1.2 else 0.62)
    y = H * cfg.get("y", 0.885)
    text = sub["text"]
    words = sub["words"]
    # scene-aware colours: measured against the REAL pixels under the subtitle zone
    bg = _env(env, "subs")
    light = _ct.is_light(bg)
    spoken = _ct.ensure(cfg.get("emph") or cfg.get("spoken", "#ffffff"), bg, 3.0,   # bold ≥ 24px = WCAG "large"
                        prefer=(cfg.get("accent"), "#111111" if light else "#ffffff"))
    soft = _ct.ensure(cfg.get("spoken", "#ffffff") if not light else cfg.get("spoken", "#111111"), bg, 4.5,
                      prefer=("#1a1a1a",) if light else ((0.9, 0.9, 0.92, 1.0),))
    if not light and _ct.luminance(soft) > 0.85:
        soft = (0.9, 0.9, 0.92, 1.0)   # spoken non-emphasised words stay below the bloom threshold
    upcoming = _ct.dim(_color(cfg.get("upcoming", "#4d4d52")) if not light else soft, bg,
                       float(cfg.get("upcoming_contrast", 1.9)))
    if _ct.ratio(upcoming, bg) < 1.5:
        upcoming = _ct.dim(soft, bg, 1.9)
    lead = cfg.get("lead", 0.0)
    # sentence fade in/out
    a = alpha * min(clamp((t - sub["t"]) / 0.18), clamp((sub["end"] - t) / 0.15) if t > sub["end"] - 0.15 else 1)

    def ws(k):
        if k >= len(words):
            return None
        w = words[k]
        p = clamp((t + lead - w["t"]) / max(0.06, min(0.22, w["end"] - w["t"])))
        if p <= 0:
            return {"color": upcoming, "quiet": True}
        base = spoken if w["emph"] else soft
        col = lerp_tuple(upcoming, base, ease_out(p))
        st = {"color": col, "quiet": p < 0.999}          # mid-fade words are transitional
        if w["emph"]:
            st["scale"] = 1.0 + 0.08 * math.sin(math.pi * clamp(p))
        return st

    weight = cfg.get("weight", "bold")
    L = layout_text(text, size, weight, maxw, 1.45)
    if len(L.lines) > 2:  # shrink to fit 2 lines
        size *= 0.86
    yc = y if len(layout_text(text, size, weight, maxw, 1.45).lines) == 1 else y - size * 0.1
    box = cfg.get("box")
    if box:   # v11: "Hormozi" caption box — rounded slab behind the line (ref02/04/05), pops per sentence
        box = {} if box is True else dict(box)
        bb = text_bbox(text, W / 2, yc, size, weight, max_width=maxw, line_height=1.45)
        pad_x, pad_y = float(box.get("pad_x", size * 0.55)), float(box.get("pad_y", size * 0.32))
        pop = 0.85 + 0.15 * clamp((t - sub["t"]) / 0.16)
        bw, bh = (bb[2] - bb[0] + 2 * pad_x) * pop, (bb[3] - bb[1] + 2 * pad_y) * pop
        from .geometry import compile_path, rounded_rect_d
        bc = _color(box.get("color", "#ffffff"))
        ctx.save()
        ctx.new_path()
        ctx.append_path(compile_path(rounded_rect_d(W / 2 - bw / 2, yc - bh / 2, bw, bh,
                                                     float(box.get("radius", bh * 0.3)))).whole)
        ctx.set_source_rgba(bc[0], bc[1], bc[2], bc[3] * a * float(box.get("alpha", 0.94)))
        ctx.fill()
        ctx.restore()
        bg = bc                                   # words are now judged against the slab, not the frame
        light = _ct.is_light(bg)
        spoken = _ct.ensure(cfg.get("emph") or ("#d63230" if light else "#ffffff"), bg, 3.0, prefer=("#111111",))
        soft = _ct.ensure(box.get("text", "#1a1a1a" if light else "#ffffff"), bg, 4.5)
        upcoming = _ct.dim(soft, bg, 1.9)
    draw_text(ctx, text, W / 2, yc, size=size, weight=weight, max_width=maxw, line_height=1.45,
              word_style=ws, alpha=a)


def draw_handle(ctx, W, H, handle, alpha, env=None):
    if not handle:
        return
    c = _ct.ensure("#5a5a5e", _env(env, "handle"), 2.4)
    draw_text(ctx, handle, W * 0.06, H * 0.79, size=24, color=c, quiet=True, align="left", alpha=alpha * 0.8, weight="ui_label")


def draw_hud(ctx, W, H, hud, chapters, ch_idx, t, subs, env=None, prev_hud=None):
    """env = contrast.hud_env(surface…): background samples per HUD zone (scene awareness).
    v8: `hud` is the CURRENT chapter's HUD (per-chapter style worlds); `prev_hud` lets parts that the new
    world hides (e.g. lineart header → swiss chapter) fade out instead of popping."""
    fade_at = hud.get("fade_out_at")
    a = 1.0 if fade_at is None else 1 - ease_in_out(prog(t, float(fade_at), 1.2))
    if a <= 0.003:
        return
    ch = chapters[ch_idx]
    p_change = ease_out(prog(t, ch["t"], 0.6))
    if prev_hud is not None and p_change < 1:
        for part in ("header", "dots", "subs"):
            pv, cu = prev_hud.get(part) or {}, hud.get(part) or {}
            if pv.get("show", True) and not cu.get("show", True):
                fa = a * (1 - p_change)
                if part == "header":
                    draw_header(ctx, W, H, pv, chapters[ch_idx - 1].get("title"), fa, 1.0, env=env)
                elif part == "dots":
                    draw_dots(ctx, W, H, pv, len(chapters), ch_idx - 1, fa, 1.0, env=env)
                else:
                    draw_subs(ctx, W, H, pv, active_subtitle(subs, t), t, fa, env=env)
    show_in = p_change if (prev_hud is not None and ch_idx > 0) else 1.0
    def _new(part):            # parts that just APPEARED with this world fade in
        return show_in if prev_hud is not None and not (prev_hud.get(part) or {}).get("show", True) else 1.0
    draw_header(ctx, W, H, hud["header"], ch.get("title"), a * _new("header"), p_change if ch_idx > 0 else 1.0, env=env)
    draw_dots(ctx, W, H, hud["dots"], len(chapters), ch_idx, a * _new("dots"), p_change, env=env)
    draw_subs(ctx, W, H, hud["subs"], active_subtitle(subs, t), t, a * _new("subs"), env=env)
    draw_handle(ctx, W, H, hud.get("handle"), a, env=env)
