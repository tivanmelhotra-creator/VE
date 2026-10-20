"""
mograph.elements — the declarative element registry.

Every visual thing in a scene is a dict:
    {"type": "icon", "id": "vial", "name": "vial", "x": 540, "y": 800, "size": 260,
     "t": 35.0, "out": 46.0, "enter": "draw", "exit": "fade"}

Common keys (all optional except type & t):
    id        unique name (callouts/arrows can target it, QA reports it)
    t         enter time (s)           dur      enter duration (type default)
    out       exit start time (s)      out_dur  exit duration (0.45)
    enter     draw | fade | pop | rise | wipe | words | slam | none
    exit      fade | undraw | shrink | drop | none
    ease      easing name for enter (see core.EASINGS)
    x, y      position (px, world space unless "space": "screen")
    tracks    {"x": [[t,v],...], "y":..., "scale":..., "rot":..., "alpha":..., "color": [[t,"#fff"],[t,"#555"]]}
    dim_at    time when the element turns dim gray (background it during focus moments)
    idle      pulse | float | spin | breathe      (subtle life while on screen)
    sfx       override sound: "auto" | none | name   (see audio.py)
    layer     draw order (default 0; higher on top)
    space     world (default, follows camera) | screen (HUD-like, ignores camera)

Add your own element in a pipeline file:

    from mograph.elements import element
    @element("my_widget", dur=0.8, sfx="tick")
    def my_widget(ctx, el, s):        # s = State (s.p = eased enter progress, s.alpha, s.t ...)
        ...draw with cairo...
        return (x0, y0, x1, y1)        # bbox in local/user coords (for QA)
"""
from __future__ import annotations

import math

import cairo

from .core import (Track, clamp, color as _color, ease_out, ease_in_out, get_ease, lerp, prog,
                   spring, spring_preset, wiggle, lerp_tuple)
from .geometry import (arrow_d, circle_d, compile_path, draw_morph, draw_path, point_on_path, polyline_d,
                       rounded_rect_d, stroke_trimmed)
from .icons import draw_icon, icon_bbox
from .text import draw_text, parse_markup, resolve_font, text_bbox, to_fa_digits, word_boxes

REGISTRY: dict = {}

WHITE = "#ffffff"
DIM = "#555559"
MID = "#8a8a90"


def F(el, role, key="font"):
    """Font for an element: el['font'] / el['weight'] override the element's default ROLE."""
    return resolve_font(el.get(key), el.get("weight") if key == "font" else None, role)


def LF(el, role="body"):
    """Font for secondary labels (label_font)."""
    return resolve_font(el.get("label_font"), None, role)


def text_fx(el, s, size=None):
    """v9 — shared typography treatment kwargs for draw_text (text / headline / kinetic / block …):
      "text_style":  "solid" | "outline" | "extrude_3d" | "knockout" | {"kind": ..., "stroke_w", "depth", "angle",
                     "extrude": "@mute", "shade", "pad", "radius", "fill_alpha"}
      "stroke_reveal": true | seconds   (outline only) progressive trim-path drawing from "t" (or "stroke_reveal_at")
      "extrude_grow": seconds           (extrude_3d) depth springs out from 0 after "t"
      "kashida": px | {"px": 260, "word": "کش", "after": 1}    live tatweel stretch of one word
      "kashida_at": "w:کش" (time the stretch starts, default t)  "kashida_dur": 0.7  "kashida_ease": "spring"|ease
      "kashida_back": time (stretch relaxes back to 0)
    Returns {"style": ..., "kashida": ...} (empty dict = plain solid text, zero cost)."""
    out = {}
    st = el.get("text_style")
    t = s.t
    if st and st != "solid":
        st = dict(st) if isinstance(st, dict) else {"kind": st}
        st.setdefault("kind", "solid")
        for k in ("extrude", "fill", "face_stroke"):
            if isinstance(st.get(k), str) and st[k].startswith("@") and s.scene is not None:
                from .typo import pal_of
                st[k] = pal_of(el, s).get(st[k][1:], "#000000")
        sr = el.get("stroke_reveal")
        if st["kind"] == "outline" and sr:
            d = float(sr) if isinstance(sr, (int, float)) and not isinstance(sr, bool) else 1.2
            t0 = float(el.get("stroke_reveal_at", el.get("t", 0.0)))
            st["reveal"] = get_ease(el.get("ease", "in_out"))(prog(t, t0, d))
        if st["kind"] == "extrude_3d" and el.get("extrude_grow"):
            st["depth_p"] = spring(t, float(el.get("t", 0.0)), stiffness=260, damping=16)
        out["style"] = st
    k = el.get("kashida")
    if k:
        k = dict(k) if isinstance(k, dict) else {"px": float(k)}
        t0 = float(el.get("kashida_at", el.get("t", 0.0)))
        dur = float(el.get("kashida_dur", 0.7))
        e = el.get("kashida_ease", "spring")
        if e == "spring":
            p = spring(t, t0, stiffness=140, damping=13) if t >= t0 else 0.0
        else:
            p = get_ease(e)(prog(t, t0, dur))
        if el.get("kashida_back") is not None:
            p *= 1 - ease_in_out(prog(t, float(el["kashida_back"]), dur * 0.8))
        k["px"] = max(0.0, float(k.get("px", 200)) * p)
        k.setdefault("word", el.get("kashida_word"))
        out["kashida"] = k
    return out


def element(name, dur=0.6, enter="fade", sfx="none", exit="fade"):
    def deco(fn):
        REGISTRY[name] = dict(fn=fn, dur=dur, enter=enter, sfx=sfx, exit=exit)
        return fn
    return deco


# ---------------------------------------------------------------------------
# per-frame element state
# ---------------------------------------------------------------------------

class State:
    __slots__ = ("t", "lt", "p", "raw_p", "out_p", "alpha", "trim", "scale", "dy", "visible",
                 "enter", "exit", "color", "scene")

    def ok(self):
        return self.visible and self.alpha > 0.003


def _tracks(el):
    tr = el.get("_tracks")
    if tr is None:
        tr = {}
        for k, keys in (el.get("tracks") or {}).items():
            conv = []
            for kk in keys:
                v = kk[1]
                if isinstance(v, str) and v.startswith("#"):
                    v = _color(v)
                conv.append((kk[0], v, kk[2]) if len(kk) > 2 else (kk[0], v))
            tr[k] = Track(conv)
        el["_tracks"] = tr
    return tr


def val(el, key, t, default=None):
    tr = _tracks(el)
    if key in tr:
        return tr[key](t)
    return el.get(key, default)


def compute_state(el, t, scene=None) -> State:
    spec = REGISTRY[el["type"]]
    s = State()
    s.scene = scene
    s.t = t
    t_in = float(el.get("t", 0.0))
    dur = float(el.get("dur", spec["dur"]))
    t_out = el.get("out")
    out_dur = float(el.get("out_dur", 0.45))
    s.enter = el.get("enter", spec["enter"])
    s.exit = el.get("exit", spec["exit"])
    s.lt = t - t_in
    s.visible = t >= t_in and (t_out is None or t < t_out + out_dur)
    s.raw_p = prog(t, t_in, dur)
    ease = get_ease(el.get("ease", "smooth" if s.enter in ("pop", "rise", "fade") else "in_out"))
    s.p = ease(s.raw_p)
    s.out_p = ease_in_out(prog(t, t_out, out_dur)) if t_out is not None else 0.0
    s.alpha = float(val(el, "alpha", t, 1.0)) * float(el.get("opacity", 1.0))
    s.trim = 1.0
    s.scale = 1.0
    s.dy = 0.0
    if s.enter in ("draw", "words", "wipe"):
        s.trim = s.p
    elif s.enter == "fade":
        s.alpha *= s.p
    elif s.enter == "pop":
        sp = spring_preset(t, t_in, "bouncy") if t >= t_in else 0.0
        s.scale = 0.6 + 0.4 * sp
        s.alpha *= clamp(s.raw_p * 3)
    elif s.enter == "rise":
        s.dy = (1 - s.p) * 40
        s.alpha *= s.p
    elif s.enter == "whip_blur":           # v9: offset + motion blur are applied by render.draw_world
        s.alpha *= clamp(s.lt / 0.05)
    elif s.enter == "slam":
        sp = spring_preset(t, t_in, "stamp") if t >= t_in else 0.0
        s.scale = 1.8 - 0.8 * sp
        s.alpha *= clamp(s.lt / 0.08)
    if s.out_p > 0:
        if s.exit == "fade":
            s.alpha *= 1 - s.out_p
        elif s.exit == "undraw":
            s.trim = (s.out_p, 1.0)
        elif s.exit == "shrink":
            s.scale *= 1 - 0.7 * s.out_p
            s.alpha *= 1 - s.out_p
        elif s.exit == "drop":
            s.dy += 60 * s.out_p ** 2
            s.alpha *= 1 - s.out_p
    # colour (with dim_at / color track)
    dflt = WHITE
    if scene is not None and el.get("color") is None and el.get("_chapter") is not None:
        w = scene.chapters[el["_chapter"]].get("_world") if el["_chapter"] < len(scene.chapters) else None
        if w and w.get("style") != "lineart":        # v8: a lineart element inside a swiss/kinetic world
            dflt = w["palette"].get("ink", WHITE)     # defaults to that world's ink, never white-on-paper
    col = val(el, "color", t, el.get("color", dflt))
    col = _color(col)
    if el.get("dim_at") is not None:
        dp = ease_in_out(prog(t, float(el["dim_at"]), 0.5))
        undim = el.get("undim_at")
        if undim is not None:
            dp *= 1 - ease_in_out(prog(t, float(undim), 0.5))
        col = lerp_tuple(col, _color(el.get("dim_color", "#4a4a4e")), dp)
    s.color = col
    # idle life
    idle = el.get("idle")
    if idle and s.visible:
        seed = hash(el.get("id", el["type"])) % 97
        if idle == "pulse":
            s.scale *= 1 + 0.035 * math.sin(2 * math.pi * 0.8 * t)
        elif idle == "breathe":
            s.scale *= 1 + 0.015 * math.sin(2 * math.pi * 0.35 * t + seed)
        elif idle == "float":
            s.dy += wiggle(t, 0.25, 6, seed)
        elif idle == "vo_pulse" and scene is not None:     # v8: audio-reactive — breathes with the narrator
            s.scale *= 1 + float(el.get("pulse_amount", 0.08)) * scene.vo_env(t)
    return s


def xy(el, t, scene=None):
    """Element position, resolving `target` references."""
    x, y = val(el, "x", t, 540), val(el, "y", t, 960)
    return float(x), float(y)


def resolve_point(p, t, scene):
    """A point can be [x,y] or "id" or "id+dx,dy" or {"id":..., "dx":..,"dy":..}."""
    if p is None:
        return None
    if isinstance(p, (list, tuple)):
        return float(p[0]), float(p[1])
    if isinstance(p, dict):
        tx, ty = resolve_point(p["id"], t, scene)
        return tx + p.get("dx", 0), ty + p.get("dy", 0)
    if isinstance(p, str):
        dx = dy = 0.0
        name = p
        if "+" in p:
            name, off = p.split("+", 1)
            dx, dy = [float(v) for v in off.split(",")]
        tgt = scene.by_id[name] if scene else None
        if tgt is None:
            raise KeyError(f"unknown target id '{name}'")
        x, y = xy(tgt, t, scene)
        return x + dx, y + dy
    raise TypeError(p)


def _apply_xform(ctx, x, y, s, rot=0.0):
    ctx.translate(x, y + s.dy)
    if rot:
        ctx.rotate(math.radians(rot))
    if s.scale != 1.0:
        ctx.scale(s.scale, s.scale)


def _bb_union(*bbs):
    bbs = [b for b in bbs if b]
    if not bbs:
        return None
    return (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))


def value_at(el, s, default_key="to", default_from=None, ease="out_expo"):
    """Numeric value of counter/ring.  Either from/to over the enter duration, or a multi-step
    'value' track:  "tracks": {"value": [["w:یک", 1], ["w:۳۰۰", 1], ["w:۳۰۰+1", 300, "out_expo"]]}
    → ONE element changes in place (no overlapping duplicates)."""
    tr = _tracks(el)
    if "value" in tr:
        return float(tr["value"](s.t))
    v1 = float(el.get(default_key, el.get("value", 0)))
    v0 = float(el.get("from", default_from if default_from is not None else 0))
    return lerp(v0, v1, get_ease(el.get("ease", ease))(s.raw_p))


def value_punch(el, t):
    """0..1 bump right after each value keyframe (scale punch on number changes)."""
    keys = (el.get("tracks") or {}).get("value")
    if not keys or not el.get("punch", True):
        return 0.0
    best = 0.0
    for i in range(1, len(keys)):
        k0, k1 = keys[i - 1], keys[i]
        if float(k1[1]) == float(k0[1]):
            continue
        tk = float(k1[0])
        d = t - tk
        if -0.05 <= d <= 0.45:
            best = max(best, math.sin(math.pi * clamp((d + 0.05) / 0.5)))
    return best


def _morph_draw(ctx, a, b, p, size, s, el):
    from .icons import ICONS
    ia, ib = ICONS.get(a, ICONS["question"]), ICONS.get(b, ICONS["question"])
    da = tuple(ia["primary"] + (ia.get("detail", []) if el.get("detail", True) else []))
    db = tuple(ib["primary"] + (ib.get("detail", []) if el.get("detail", True) else []))
    draw_morph(ctx, da, db, p, 0, 0, scale=size / 100, stroke=s.color, width=el.get("width") or 2.6,
               alpha=s.alpha, twist=el.get("twist", 0.0))


# ===========================================================================
# ELEMENTS
# ===========================================================================

@element("icon", dur=1.1, enter="draw", sfx="draw")
def el_icon(ctx, el, s):
    x, y = xy(el, s.t)
    size = float(val(el, "size", s.t, 160))
    rot = float(val(el, "rot", s.t, 0.0))
    ctx.save()
    _apply_xform(ctx, x, y, s, rot)
    trim = s.trim if isinstance(s.trim, float) else s.trim[1] - s.trim[0]
    mt = el.get("morph_to")
    mp = 0.0
    if mt and el.get("morph_at") is not None:
        mp = get_ease(el.get("morph_ease", "in_out_expo"))(prog(s.t, float(el["morph_at"]), el.get("morph_dur", 0.9)))
    if mp <= 0.0:
        draw_icon(ctx, el["name"], 0, 0, size=size, p=trim, style=el.get("style", "default"),
                  color=s.color, alpha=s.alpha, fill=el.get("fill"), dashed=el.get("dashed"),
                  detail=el.get("detail", True), width=el.get("width"))
    elif mp >= 1.0:
        draw_icon(ctx, mt, 0, 0, size=size, p=1.0, style=el.get("style", "default"), color=s.color,
                  alpha=s.alpha, detail=el.get("detail", True), width=el.get("width"))
    else:
        _morph_draw(ctx, el["name"], mt, mp, size, s, el)
    ctx.restore()
    if s.scene is not None and (el.get("vo_ring") or (el.get("idle") == "vo_pulse" and
                                                      (mt or el.get("name")) in ("money_core", "ai_monogram"))):
        from .horizon import vo_ring
        vo_ring(ctx, x, y + s.dy, size * s.scale, s.scene.vo_env(s.t), s.color, s.alpha)
    if el.get("label"):
        ly = y + size * 0.5 * s.scale + el.get("label_gap", 34)
        draw_text(ctx, el["label"], x, ly + s.dy, size=el.get("label_size", 34), weight=LF(el),
                  color=el.get("label_color", MID), alpha=s.alpha * clamp(s.raw_p * 2 - 0.6))
    h = size / 2 * s.scale
    return (x - h, y - h + s.dy, x + h, y + h + s.dy + (60 if el.get("label") else 0))


@element("text", dur=0.6, enter="rise", sfx="none")
def el_text(ctx, el, s):
    x, y = xy(el, s.t)
    raw = el["text"]
    txt, emph = parse_markup(raw)
    if el.get("fa_digits"):
        txt = to_fa_digits(txt)
    size = float(val(el, "size", s.t, 52))
    emph_col = _color(el.get("emph_color", WHITE))
    base = s.color
    if emph and el.get("emph_dim_others", True):
        base = _color(el.get("base_color", "#c8c8cc"), s.color[3])
    ws = (lambda k: {"color": emph_col} if k in emph else {"color": base}) if emph else None
    reveal = None
    if s.enter in ("words", "wipe"):
        reveal = (s.enter, s.trim if isinstance(s.trim, float) else 1.0)
    ctx.save()
    if s.scale != 1.0:
        ctx.translate(x, y); ctx.scale(s.scale, s.scale); ctx.translate(-x, -y)
    bb = draw_text(ctx, txt, x, y + s.dy, size=size, color=base, weight=F(el, "body"),
                   align=el.get("align", "center"), valign=el.get("valign", "middle"), alpha=s.alpha,
                   max_width=el.get("max_width"), line_height=el.get("line_height", 1.55),
                   word_style=ws, reveal=reveal, **text_fx(el, s, size))
    if el.get("prefix"):                     # v9: ">_" / "✦" / "✓" glyph before the text (reading side)
        from .text import _is_rtl_char
        rtl = any(_is_rtl_char(c) for c in txt)
        psz = size * float(el.get("prefix_scale", 0.8))
        pc = _color(el.get("prefix_color") or el.get("emph_color") or s.color)
        gap = size * 0.35
        px_ = bb[2] + gap if rtl else bb[0] - gap
        pb = draw_text(ctx, el["prefix"], px_, y + s.dy, size=psz, color=pc, weight=el.get("prefix_font", "ui"),
                       align="left" if rtl else "right", alpha=s.alpha * clamp((s.lt - 0.1) / 0.25))
        bb = _bb_union(bb, pb)
    ctx.restore()
    if el.get("underline"):
        up = ease_out(prog(s.t, el["t"] + 0.3, 0.6))
        draw_path(ctx, f"M {bb[2]} {bb[3] + 8} L {bb[0]} {bb[3] + 8}", 0, 0, 1, trim=up, anchor="none",
                  stroke=s.color, width=3, alpha=s.alpha)
    return bb


@element("box", dur=0.9, enter="draw", sfx="draw")
def el_box(ctx, el, s):
    x, y = xy(el, s.t)
    w, h = float(val(el, "w", s.t, 400)), float(val(el, "h", s.t, 240))
    r = float(el.get("r", 22))
    ctx.save()
    _apply_xform(ctx, x, y, s, float(val(el, "rot", s.t, 0)))
    d = rounded_rect_d(-w / 2, -h / 2, w, h, r)
    trim = s.trim
    fill = el.get("fill", "#000000")
    draw_path(ctx, d, 0, 0, 1, trim=trim, stroke=s.color, width=el.get("width", 2.6), anchor="none",
              fill=fill, alpha=s.alpha, dashed=el.get("dashed"))
    if el.get("double"):
        ins = 9
        d2 = rounded_rect_d(-w / 2 + ins, -h / 2 + ins, w - 2 * ins, h - 2 * ins, max(4, r - ins * 0.7))
        tr2 = clamp((s.trim if isinstance(s.trim, float) else 1) * 1.3 - 0.3)
        draw_path(ctx, d2, 0, 0, 1, trim=tr2, stroke=s.color, width=1.3, anchor="none", alpha=s.alpha * 0.55)
    if el.get("tab"):  # folder-like tab label on top-right
        tw = el.get("tab_w", 170)
        tx1 = w / 2 - r - 10
        dtab = f"M {tx1} {-h / 2} L {tx1 - 14} {-h / 2 - 46} L {tx1 - tw + 14} {-h / 2 - 46} L {tx1 - tw} {-h / 2}"
        draw_path(ctx, dtab, 0, 0, 1, trim=s.trim, stroke=s.color, width=2.6, anchor="none", alpha=s.alpha,
                  fillet=8)
        draw_text(ctx, el["tab"], tx1 - tw / 2, -h / 2 - 22, size=30, color=s.color,
                  alpha=s.alpha * clamp(s.raw_p * 2 - 0.8), weight=LF(el, "title"))
    if el.get("label"):
        draw_text(ctx, el["label"], 0, 0, size=el.get("label_size", 40), color=s.color, weight=F(el, "body"),
                  alpha=s.alpha * clamp(s.raw_p * 2 - 0.7), max_width=w - 40)
    if el.get("lines"):  # skeleton text lines inside (document look)
        n = int(el["lines"])
        for i in range(n):
            ly = -h / 2 + 70 + i * ((h - 110) / max(1, n - 1))
            lw = w * (0.7 if i % 3 != 2 else 0.45)
            lp = clamp((s.raw_p - 0.4 - i * 0.05) * 3)
            draw_path(ctx, f"M {lw / 2} {ly} L {-lw / 2} {ly}", 0, 0, 1, trim=lp, anchor="none",
                      stroke="#3c3c40", width=4, alpha=s.alpha)
    ctx.restore()
    return (x - w / 2, y - h / 2 - (46 if el.get("tab") else 0), x + w / 2, y + h / 2)


@element("line", dur=0.7, enter="draw", sfx="draw")
def el_line(ctx, el, s):
    pts = [resolve_point(p, s.t, s.scene) for p in el["points"]]
    d = polyline_d(pts, el.get("closed", False))
    if el.get("arrow"):
        a, b = pts[-2], pts[-1]
        hd = arrow_d(a[0], a[1], b[0], b[1], head=el.get("head", 16))
        d = d + " " + hd[hd.index(" M ") + 1:]
    draw_path(ctx, d, 0, 0, 1, trim=s.trim, stroke=s.color, width=el.get("width", 2.4), anchor="none",
              alpha=s.alpha, dashed=el.get("dashed"), fillet=el.get("fillet", 10))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


@element("arrow", dur=0.6, enter="draw", sfx="whoosh")
def el_arrow(ctx, el, s):
    a = resolve_point(el["from"], s.t, s.scene)
    b = resolve_point(el["to"], s.t, s.scene)
    d = arrow_d(a[0], a[1], b[0], b[1], head=el.get("head", 18), curve=el.get("curve", 0.0))
    draw_path(ctx, d, 0, 0, 1, trim=s.trim, stroke=s.color, width=el.get("width", 2.8), anchor="none",
              alpha=s.alpha, dashed=el.get("dashed"))
    if el.get("label"):
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        off = el.get("label_offset", [40, 0])
        draw_text(ctx, el["label"], mx + off[0], my + off[1], size=el.get("label_size", 32),
                  color=el.get("label_color", MID), alpha=s.alpha * clamp(s.raw_p * 2 - 0.8),
                  align=el.get("label_align", "left"), weight=LF(el))
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


@element("callout", dur=0.8, enter="draw", sfx="tick")
def el_callout(ctx, el, s):
    """Leader line: anchor dot → elbow → shelf, label on the shelf.
    anchor: [x,y] | "id" | "id+dx,dy"; elbow: [dx,dy] relative to anchor; shelf: px (negative = to the left)."""
    ax, ay = resolve_point(el["anchor"], s.t, s.scene)
    ex, ey = el.get("elbow", [90, -90])
    ex, ey = ax + ex, ay + ey
    shelf = float(el.get("shelf", 220))
    sx = ex + shelf
    trim = s.trim if isinstance(s.trim, float) else 1.0
    a = s.alpha
    # anchor double ring
    rp = clamp(trim * 4)
    if rp > 0:
        ctx.set_source_rgba(*_color(s.color, a * rp))
        ctx.arc(ax, ay, 5.5 * rp, 0, 2 * math.pi); ctx.fill()
        ctx.set_line_width(1.6)
        ctx.arc(ax, ay, 11 * rp, 0, 2 * math.pi); ctx.stroke()
    d = f"M {ax} {ay} L {ex} {ey} L {sx} {ey}"
    draw_path(ctx, d, 0, 0, 1, trim=clamp(trim * 1.25 - 0.1), stroke=s.color, width=2.2, anchor="none",
              alpha=a, fillet=14)
    lp = clamp((s.raw_p - 0.5) * 2.5)
    size = el.get("size", 36)
    lx = (ex + sx) / 2
    bb = draw_text(ctx, el["label"], lx, ey - size * 0.62, size=size, color=el.get("label_color", s.color),
                   alpha=a * lp, reveal=("wipe", lp), weight=F(el, "body"))
    return _bb_union((min(ax, ex, sx) - 12, min(ay, ey) - 12, max(ax, ex, sx) + 12, max(ay, ey) + 12), bb)


@element("stamp", dur=0.5, enter="slam", sfx="stamp")
def el_stamp(ctx, el, s):
    x, y = xy(el, s.t)
    txt = el["text"]
    size = el.get("size", 46)
    fnt = F(el, "title")
    tb = text_bbox(txt, 0, 0, size, fnt)
    w = el.get("w", (tb[2] - tb[0]) + 90)
    h = el.get("h", size * 1.9)
    ang = el.get("angle", -7)
    ctx.save()
    _apply_xform(ctx, x, y, s, ang)
    draw_path(ctx, rounded_rect_d(-w / 2, -h / 2, w, h, 16), 0, 0, 1, stroke=s.color, width=3.4,
              anchor="none", fill="#000000", alpha=s.alpha)
    draw_path(ctx, rounded_rect_d(-w / 2 + 9, -h / 2 + 9, w - 18, h - 18, 10), 0, 0, 1, stroke=s.color,
              width=1.5, anchor="none", alpha=s.alpha * 0.7)
    draw_text(ctx, txt, 0, 0, size=size, color=s.color, alpha=s.alpha, weight=fnt)
    ctx.restore()
    rr = math.hypot(w, h) / 2 * s.scale
    return (x - w / 2 * s.scale, y - rr * 0.6, x + w / 2 * s.scale, y + rr * 0.6)


@element("strike", dur=1.4, enter="none", sfx="strike")
def el_strike(ctx, el, s):
    """False word gets struck through live, then the true word rises above it with a check."""
    x, y = xy(el, s.t)
    size = el.get("size", 50)
    sp = ease_in_out(prog(s.lt, 0.0, 0.55))
    tp = ease_out(prog(s.lt, 0.5, 0.6))
    if s.out_p:
        a_out = 1 - s.out_p
    else:
        a_out = 1.0
    a = s.alpha * a_out
    dimv = lerp(1.0, 0.42, sp)
    bb = draw_text(ctx, el["false"], x, y, size=size, color=(dimv, dimv, dimv, 1), alpha=a, weight=F(el, "body"))
    if sp > 0:
        pad = 14
        draw_path(ctx, f"M {bb[2] + pad} {y + 2} L {bb[0] - pad} {y + 2}", 0, 0, 1, trim=sp,
                  anchor="none", stroke=WHITE, width=3.2, alpha=a)
    bb2 = None
    if tp > 0 and el.get("true"):
        ty = y - size * 1.5 - (1 - tp) * 18
        bb2 = draw_text(ctx, el["true"], x, ty, size=size, color=WHITE, alpha=a * tp, weight=F(el, "body"))
        draw_icon(ctx, "check", bb2[0] - 34, ty, size=46, p=tp, width=3.4, alpha=a)
    return _bb_union(bb, bb2)


@element("counter", dur=1.6, enter="none", sfx="counter")
def el_counter(ctx, el, s):
    """Animated number. fmt: '{:,.0f}', prefix/suffix, fa=True for Persian digits."""
    x, y = xy(el, s.t)
    v = value_at(el, s)
    txt = el.get("prefix", "") + el.get("fmt", "{:,.0f}").format(v) + el.get("suffix", "")
    if el.get("fa", True):
        txt = to_fa_digits(txt).replace(",", "٬")
    a = s.alpha * clamp(s.raw_p * 6) * (1 - s.out_p)
    punch = value_punch(el, s.t)
    size = val(el, "size", s.t, 120) * (1 + 0.14 * punch)
    if el.get("roll"):          # v8 odometer digits (prefix/suffix drawn beside the wheels)
        from .horizon import roll_number
        size = val(el, "size", s.t, 120)
        fa = el.get("fa", True)
        bb = roll_number(ctx, v, x, y + s.dy, size, s.color, F(el, "counter"), alpha=a, align="center",
                         fmt=el.get("fmt", "{:,.0f}"), fa=fa)
        pre, suf = el.get("prefix", ""), el.get("suffix", "")
        if pre:     # RTL: prefix sits to the RIGHT of the number
            pb = draw_text(ctx, to_fa_digits(pre) if fa else pre, bb[2] + 12, y + s.dy, size=size * 0.6,
                           color=s.color, alpha=a, weight=F(el, "counter"), align="left")
            bb = (bb[0], min(bb[1], pb[1]), pb[2], max(bb[3], pb[3]))
        if suf:
            sb = draw_text(ctx, to_fa_digits(suf) if fa else suf, bb[0] - 12, y + s.dy, size=size * 0.6,
                           color=s.color, alpha=a, weight=F(el, "counter"), align="right")
            bb = (sb[0], min(bb[1], sb[1]), bb[2], max(bb[3], sb[3]))
    else:
        bb = draw_text(ctx, txt, x, y + s.dy, size=size, color=s.color, alpha=a, weight=F(el, "counter"))
    if el.get("label"):
        draw_text(ctx, el["label"], x, bb[3] + 30, size=el.get("label_size", 34), color=MID, alpha=a, weight=LF(el))
    return bb


@element("dot_grid", dur=1.2, enter="none", sfx="grid")
def el_dot_grid(ctx, el, s):
    """Grid of dots (population/statistics). highlight: count of lit dots (from the top-right, RTL),
    highlight_at: time the non-highlighted dots go dark."""
    x, y = xy(el, s.t)
    rows, cols = int(el.get("rows", 20)), int(el.get("cols", 14))
    gap = float(el.get("gap", 34))
    r = float(el.get("r", 5))
    n = rows * cols
    hi = int(el.get("highlight", 0))
    hat = el.get("highlight_at")
    dark = ease_in_out(prog(s.t, hat, 0.6)) if hat is not None else 0.0
    w, h = (cols - 1) * gap, (rows - 1) * gap
    a_all = s.alpha * (1 - s.out_p)
    for i in range(n):
        rr_, cc = divmod(i, cols)
        cc_rtl = cols - 1 - cc
        px, py = x - w / 2 + cc_rtl * gap, y - h / 2 + rr_ * gap
        # radial-ish stagger from the centre for the build
        dd = math.hypot(cc - cols / 2, rr_ - rows / 2) / math.hypot(cols / 2, rows / 2)
        ap = ease_out(prog(s.lt, dd * 0.7, 0.35))
        if ap <= 0:
            continue
        lit = i < hi
        g = 1.0 if lit else lerp(1.0, 0.22, dark)
        ctx.set_source_rgba(g, g, g, a_all * ap)
        ctx.arc(px, py, r * (1.25 if lit and dark > 0 else 1.0), 0, 2 * math.pi)
        ctx.fill()
    return (x - w / 2 - r, y - h / 2 - r, x + w / 2 + r, y + h / 2 + r)


@element("icon_grid", dur=1.2, enter="none", sfx="grid")
def el_icon_grid(ctx, el, s):
    """Repeated icons in a grid/row (crowd, medicines...). names: str or list (cycled).
    highlight: list of indexes kept white after highlight_at; others dim. cross_at: time to strike all."""
    x, y = xy(el, s.t)
    names = el["names"] if isinstance(el.get("names"), list) else [el.get("names", el.get("name", "human"))]
    rows, cols = int(el.get("rows", 1)), int(el.get("cols", 7))
    gap = float(el.get("gap", 130))
    size = float(el.get("size", 110))
    hl = set(el.get("highlight", []))
    hat = el.get("highlight_at")
    dark = ease_in_out(prog(s.t, hat, 0.6)) if hat is not None else 0.0
    cross_at = el.get("cross_at")
    w, h = (cols - 1) * gap, (rows - 1) * gap
    step = float(el.get("stagger", 0.06))
    a_all = s.alpha * (1 - s.out_p)
    for i in range(rows * cols):
        rr_, cc = divmod(i, cols)
        px = x + w / 2 - cc * gap
        py = y - h / 2 + rr_ * gap
        ip = ease_in_out(prog(s.lt, i * step, 0.8))
        if ip <= 0:
            continue
        lit = (i in hl) or not hl
        g = 1.0 if lit else lerp(1.0, 0.3, dark)
        draw_icon(ctx, names[i % len(names)], px, py, size=size, p=ip, color=(g, g, g, 1), alpha=a_all,
                  width=el.get("width", 2.2))
        if cross_at is not None:
            cp = ease_out(prog(s.t, cross_at + i * 0.012, 0.3))
            if cp > 0:
                hs = size * 0.42
                draw_path(ctx, f"M {px + hs} {py - hs} L {px - hs} {py + hs}", 0, 0, 1, trim=cp, anchor="none",
                          stroke=WHITE, width=2.4, alpha=a_all)
    return (x - w / 2 - size / 2, y - h / 2 - size / 2, x + w / 2 + size / 2, y + h / 2 + size / 2)


@element("bars", dur=1.4, enter="none", sfx="grow")
def el_bars(ctx, el, s):
    """Bar chart. values, labels (under bars), value labels on top. RTL order by default."""
    x, y = xy(el, s.t)
    vals = [float(v) for v in el["values"]]
    labels = el.get("labels", [""] * len(vals))
    W_ = float(el.get("w", 760)); H_ = float(el.get("h", 520))
    bw = W_ / len(vals) * 0.52
    vmax = max(vals) or 1
    rtl = el.get("rtl", True)
    a = s.alpha * (1 - s.out_p)
    base_y = y + H_ / 2
    draw_path(ctx, f"M {x + W_ / 2 + 20} {base_y} L {x - W_ / 2 - 20} {base_y}", 0, 0, 1,
              trim=ease_out(prog(s.lt, 0, 0.5)), anchor="none", stroke=MID, width=2, alpha=a)
    for i, v in enumerate(vals):
        k = (len(vals) - 1 - i) if rtl else i
        cx = x - W_ / 2 + (k + 0.5) * W_ / len(vals)
        gp = get_ease("out_quart")(prog(s.lt, 0.25 + i * 0.12, 0.9))
        bh = H_ * (v / vmax) * gp
        hl = i in set(el.get("highlight", []))
        col = WHITE if (hl or not el.get("highlight")) else "#6a6a6e"
        if bh > 1:
            d = rounded_rect_d(cx - bw / 2, base_y - bh, bw, bh, min(14, bw / 3, bh / 2))
            draw_path(ctx, d, 0, 0, 1, stroke=col, width=2.6, anchor="none",
                      fill=("#ffffff" if hl else None), alpha=a * (0.9 if hl else 1))
        fmt = el.get("fmt", "{:g}")
        vt = to_fa_digits(fmt.format(v)) + el.get("suffix", "")
        draw_text(ctx, vt, cx, base_y - bh - 34, size=el.get("value_size", 34), color=col, alpha=a * gp,
                  weight=F(el, "counter"))
        if labels[i]:
            draw_text(ctx, labels[i], cx, base_y + 42, size=el.get("label_size", 30), color=MID,
                      alpha=a * clamp(gp * 2), weight=LF(el))
    return (x - W_ / 2 - 20, y - H_ / 2 - 60, x + W_ / 2 + 20, base_y + 70)


@element("ring", dur=1.4, enter="none", sfx="grow")
def el_ring(ctx, el, s):
    """Progress ring (percent / gauge). value 0..1, text in the middle, label below."""
    x, y = xy(el, s.t)
    r = float(el.get("r", 150))
    v = value_at(el, s, default_key="value", default_from=0.0, ease="out_quart")
    a = s.alpha * (1 - s.out_p)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_source_rgba(*_color("#2c2c30", a)); ctx.set_line_width(el.get("track_width", 14))
    ctx.new_path(); ctx.arc(x, y, r, 0, 2 * math.pi); ctx.stroke()
    if v > 0.001:
        ctx.set_source_rgba(*_color(s.color, a)); ctx.set_line_width(el.get("width", 14))
        ctx.new_path(); ctx.arc(x, y, r, -math.pi / 2, -math.pi / 2 + 2 * math.pi * v); ctx.stroke()
    txt = el.get("text")
    if txt is None:
        txt = to_fa_digits(f"{round(v * 100)}") + "٪"
    draw_text(ctx, txt, x, y, size=el.get("size", r * 0.55) * (1 + 0.12 * value_punch(el, s.t)), color=WHITE,
              alpha=a, weight=F(el, "counter"))
    if el.get("label"):
        draw_text(ctx, el["label"], x, y + r + 54, size=el.get("label_size", 36), color=MID, alpha=a, weight=LF(el))
    return (x - r - 10, y - r - 10, x + r + 10, y + r + (90 if el.get("label") else 10))


@element("timeline", dur=1.2, enter="draw", sfx="draw")
def el_timeline(ctx, el, s):
    """Horizontal timeline with ticks + labels. RTL: first item on the right."""
    x, y = xy(el, s.t)
    w = float(el.get("w", 860))
    items = el.get("items", [])
    trim = s.trim if isinstance(s.trim, float) else 1.0
    a = s.alpha * (1 - s.out_p)
    draw_path(ctx, arrow_d(x + w / 2, y, x - w / 2, y, head=16), 0, 0, 1, trim=trim, anchor="none",
              stroke=s.color, width=2.4, alpha=a)
    n = len(items)
    for i, lab in enumerate(items):
        px = x + w / 2 - (i + 0.5) * w / max(1, n) if el.get("rtl", True) else x - w / 2 + (i + 0.5) * w / max(1, n)
        tp = clamp(trim * (n + 1) - i - 0.6)
        if tp <= 0:
            continue
        draw_path(ctx, f"M {px} {y - 14} L {px} {y + 14}", 0, 0, 1, trim=tp, anchor="none", stroke=s.color,
                  width=2.4, alpha=a)
        draw_text(ctx, lab, px, y + 56, size=el.get("label_size", 30), color=MID, alpha=a * tp, weight=LF(el))
    return (x - w / 2, y - 20, x + w / 2, y + 80)


@element("pill", dur=0.6, enter="pop", sfx="pop")
def el_pill(ctx, el, s):
    x, y = xy(el, s.t)
    txt = el["text"]
    size = el.get("size", 40)
    fnt = F(el, "title")
    tb = text_bbox(txt, 0, 0, size, fnt)
    w = el.get("w", tb[2] - tb[0] + size * 1.6)
    h = el.get("h", size * 1.8)
    ctx.save()
    _apply_xform(ctx, x, y, s)
    filled = el.get("filled", False)
    draw_path(ctx, rounded_rect_d(-w / 2, -h / 2, w, h, h / 2), 0, 0, 1, stroke=s.color, width=2.6,
              anchor="none", fill=(s.color if filled else "#000000"), alpha=s.alpha)
    draw_text(ctx, txt, 0, 0, size=size, color=("#000000" if filled else s.color), alpha=s.alpha, weight=fnt)
    ctx.restore()
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


@element("divider", dur=0.8, enter="draw", sfx="draw")
def el_divider(ctx, el, s):
    """Split screen divider with optional headings on each side (right first, RTL)."""
    x, y = xy(el, s.t)
    h = float(el.get("h", 700))
    trim = s.trim if isinstance(s.trim, float) else 1.0
    a = s.alpha * (1 - s.out_p)
    draw_path(ctx, f"M {x} {y - h / 2} L {x} {y + h / 2}", 0, 0, 1, trim=trim, anchor="none",
              stroke=el.get("line_color", "#6a6a6e"), width=2, alpha=a)
    off = el.get("heading_offset", 220)
    for side, key in ((1, "right"), (-1, "left")):
        if el.get(key):
            draw_text(ctx, el[key], x + side * off, y - h / 2 + 10, size=el.get("size", 44),
                      color=s.color, alpha=a * clamp(trim * 2 - 0.5), weight=F(el, "title"))
    return (x - off - 140, y - h / 2 - 40, x + off + 140, y + h / 2)


@element("image", dur=0.5, enter="fade", sfx="none")
def el_image(ctx, el, s):
    """Raster PNG (RGBA) — logos, cut-outs, speaker photos. src: path. w: target width."""
    x, y = xy(el, s.t)
    surf = _load_png(el["src"])
    iw, ih = surf.get_width(), surf.get_height()
    w = float(val(el, "w", s.t, iw))
    k = w / iw
    ctx.save()
    _apply_xform(ctx, x, y, s, float(val(el, "rot", s.t, 0)))
    ctx.scale(k, k)
    ctx.translate(-iw / 2, -ih / 2)
    if el.get("radius"):
        draw_path(ctx, rounded_rect_d(0, 0, iw, ih, el["radius"] / k), 0, 0, 1, stroke=None, anchor="none")
        ctx.new_path()
        ctx.append_path(compile_path(rounded_rect_d(0, 0, iw, ih, el["radius"] / k)).whole)
        ctx.clip()
    ctx.set_source_surface(surf, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint_with_alpha(clamp(s.alpha))
    ctx.restore()
    h = ih * k
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


_PNG_CACHE = {}


def _load_png(path):
    if path not in _PNG_CACHE:
        _PNG_CACHE[path] = cairo.ImageSurface.create_from_png(path)
    return _PNG_CACHE[path]


@element("path", dur=1.0, enter="draw", sfx="draw")
def el_path(ctx, el, s):
    """Free SVG path `d` in a 100-box placed at x,y with `size` (like an inline custom icon)."""
    x, y = xy(el, s.t)
    size = float(el.get("size", 200))
    ctx.save()
    _apply_xform(ctx, x, y, s, float(val(el, "rot", s.t, 0)))
    for d in (el["d"] if isinstance(el["d"], list) else [el["d"]]):
        draw_path(ctx, d, 0, 0, size / 100, trim=s.trim, stroke=s.color, width=el.get("width", 2.6),
                  alpha=s.alpha, fill=el.get("fill"), dashed=el.get("dashed"), fillet=el.get("fillet", 0))
    ctx.restore()
    return (x - size / 2, y - size / 2, x + size / 2, y + size / 2)


@element("circle", dur=0.8, enter="draw", sfx="draw")
def el_circle(ctx, el, s):
    """Circle / highlight ring around something. Use target to encircle an element."""
    if el.get("target"):
        x, y = resolve_point(el["target"], s.t, s.scene)
    else:
        x, y = xy(el, s.t)
    r = float(val(el, "r", s.t, 80))
    draw_path(ctx, circle_d(x, y, r, el.get("start", -90)), 0, 0, 1, trim=s.trim, stroke=s.color,
              width=el.get("width", 2.6), anchor="none", alpha=s.alpha, dashed=el.get("dashed"),
              fill=el.get("fill"))
    if el.get("number"):
        draw_text(ctx, to_fa_digits(el["number"]), x + r * 0.75, y - r * 0.75, size=30, color=s.color,
                  alpha=s.alpha * s.raw_p, weight="counter")
    return (x - r, y - r, x + r, y + r)


@element("morph", dur=0.9, enter="none", sfx="whoosh")
def el_morph(ctx, el, s):
    """Icon → icon shape morph.  from/to: icon names (or "chain": ["question","lock","check"] with
    "at": [t1, t2] times). The morph starts at `t` (or each chain time) and lasts `dur`."""
    x, y = xy(el, s.t)
    size = float(val(el, "size", s.t, 220))
    chain = el.get("chain") or [el["from"], el["to"]]
    times = el.get("_chain_t") or [float(el["t"])]
    dur = float(el.get("dur", 0.9))
    a, b, p = chain[0], chain[0], 0.0
    for i, tk in enumerate(times):
        if s.t >= tk and i + 1 < len(chain):
            a, b = chain[i], chain[i + 1]
            p = get_ease(el.get("ease", "in_out_expo"))(prog(s.t, tk, dur))
    pre_in = el.get("draw_in", 0.0)  # optionally trim-draw the first icon before morphing
    ctx.save()
    _apply_xform(ctx, x, y, s, float(val(el, "rot", s.t, 0)))
    if pre_in and s.t < times[0]:
        draw_icon(ctx, chain[0], 0, 0, size=size, p=prog(s.t, times[0] - pre_in, pre_in * 0.9),
                  color=s.color, alpha=s.alpha, width=el.get("width"))
    elif p <= 0:
        draw_icon(ctx, a, 0, 0, size=size, p=1, color=s.color, alpha=s.alpha, width=el.get("width"))
    elif p >= 1:
        draw_icon(ctx, b, 0, 0, size=size, p=1, color=s.color, alpha=s.alpha, width=el.get("width"))
    else:
        _morph_draw(ctx, a, b, p, size, s, el)
    ctx.restore()
    cur = b if p >= 0.5 else a
    if s.scene is not None and el.get("idle") == "vo_pulse" and (el.get("vo_ring") or cur in ("money_core", "ai_monogram")):
        from .horizon import vo_ring
        vo_ring(ctx, x, y + s.dy, size * s.scale, s.scene.vo_env(s.t), s.color, s.alpha * (1 - abs(2 * p - 1) if 0 < p < 1 else 1))
    h = size / 2 * s.scale
    return (x - h, y - h, x + h, y + h)
