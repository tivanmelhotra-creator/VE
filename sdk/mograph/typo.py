"""
mograph.typo — Swiss (International Typographic Style) + Kinetic Typography elements (v7).

GRID (any element):  "col": [start, span]  "row": r        (meta.grid = {"cols":6,"rows":12,"margin":72,"gutter":24})
   → x/w come from the column grid (RTL: column 1 is on the RIGHT), y from the baseline rows.

SWISS
  block      flat colour field that wipes in from a side (mask), optional big text flush inside
             {"type":"block","col":[1,6],"row":2,"rows":3,"color":"@accent","side":"right","text":"…","text_color":"@paper"}
  headline   huge grotesque type, each LINE rises from behind its own mask (the Swiss signature reveal)
             {"type":"headline","text":"قیمتِ\\nواقعیِ\\nنفت","col":[1,6],"row":2,"size":180,"align":"right","stagger":0.12}
  rule       thick rule that extends from one side   {"type":"rule","col":[1,6],"row":5,"weight":10,"color":"@ink"}
  big_number giant numeral (value track supported), unit + caption on the grid
             {"type":"big_number","value":3000,"from":1500,"unit":"تومان","caption":"قیمت هر لیتر","col":[1,6],"row":6}
  index      numbered list on rules ۰۱ ۰۲ ۰۳ with staggered reveal
             {"type":"index","items":["پالایش","حمل","سهم شما"],"col":[1,6],"row":7,"gap":120}
  disc       solid geometric circle that scales in  {"type":"disc","x":800,"y":600,"r":220,"color":"@accent"}

KINETIC  (words come from the NARRATION timing → perfectly in sync with the voice)
  kinetic    {"type":"kinetic","text":"شما **انجام** دادید","mode":"slam|stack|scale|scroll|type",
              "t":"w:شما","size":150,"accent":"@accent"}
     mode slam   one word at a time, centre, scale-punch, previous word knocked out
          stack  words pile up line by line (newest at the bottom), the block re-centres & shrinks to fit
          scale  each word grows from 0 → full while the previous ones step back (dim, smaller)
          scroll words ride up a vertical track (teleprompter), current word white, others muted
          type   typewriter letters with a cursor
     **word** = emphasised → accent colour, bigger, harder hit, small camera punch.
     Word times: matched against subtitles (after "t"); missing → evenly staggered ("rate" words/s).
"""
from __future__ import annotations

import math

import cairo

from .core import clamp, color as _color, ease_in_out, ease_out, get_ease, lerp, prog, spring, spring_preset
from .elements import F, element, text_fx, value_at, value_punch, xy
from .geometry import draw_path, rounded_rect_d
from .text import draw_text, ink_box, layout_text, mask_rect, parse_markup, text_bbox, to_fa_digits

# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
GRID_DEFAULT = {"cols": 6, "rows": 12, "margin": 72, "gutter": 24, "top": 200, "bottom": 1500}


def grid_cfg(scene):
    """v9: the Swiss grid adapts to the aspect — rows span the same fraction of the height, landscape gets
    12 columns and a wider margin, square 8 columns."""
    g = dict(GRID_DEFAULT)
    if scene is not None:
        W, H = scene.W, scene.H
        g["top"], g["bottom"] = round(H * 200 / 1920), round(H * 1500 / 1920)
        if W > H * 1.2:
            g.update(cols=12, rows=8, margin=round(W * 0.06), top=round(H * 0.14), bottom=round(H * 0.8))
        elif abs(W - H) < 40:
            g.update(cols=8, rows=10, top=round(H * 0.12), bottom=round(H * 0.82))
    g.update((scene.spec.get("meta", {}) if scene else {}).get("grid") or {})
    return g


def grid_place(el, scene):
    """Fill x / w / y from col/row (called once at scene compile)."""
    g = grid_cfg(scene)
    W = scene.W
    cw = (W - 2 * g["margin"] - (g["cols"] - 1) * g["gutter"]) / g["cols"]
    if "col" in el:
        c = el["col"] if isinstance(el["col"], list) else [el["col"], 1]
        start, span = int(c[0]), int(c[1]) if len(c) > 1 else 1
        right = W - g["margin"] - (start - 1) * (cw + g["gutter"])      # RTL: col 1 at the right edge
        w = span * cw + (span - 1) * g["gutter"]
        el.setdefault("x", right - w / 2)
        el.setdefault("w", w)
        el["_right"], el["_left"] = right, right - w
    if "row" in el:
        rh = (g["bottom"] - g["top"]) / g["rows"]
        el.setdefault("y", g["top"] + (float(el["row"]) - 0.5) * rh)
        if isinstance(el.get("rows"), (int, float)):      # (versus_table uses "rows" for its data list)
            el.setdefault("h", float(el["rows"]) * rh)
        el["_row_h"] = rh


def pal_of(el, s):
    """v8: the palette of the element's OWN chapter world (falls back to the scene palette)."""
    sc = s.scene if s is not None else None
    if sc is None:
        return {}
    ci = el.get("_chapter") if el is not None else None
    if ci is not None and ci < len(getattr(sc, "chapters", [])):
        w = sc.chapters[ci].get("_world")
        if w:
            return w["palette"]
    return getattr(sc, "palette", None) or {}


def bg_of(el, s):
    sc = s.scene if s is not None else None
    if sc is None:
        return "#000000"
    ci = el.get("_chapter")
    if ci is not None and ci < len(sc.chapters):
        return sc.chapters[ci].get("bg", sc.bg)
    return sc.bg


def _pal(s, key, default, el=None):
    return pal_of(el, s).get(key) or default


def _col(el, key, s, default_key, default):
    v = el.get(key)
    if v is None:
        v = _pal(s, default_key, default, el)
    return _color(v)


# ---------------------------------------------------------------------------
# SWISS
# ---------------------------------------------------------------------------
def _edge_rect(x0, y0, w, h, p, frm):
    if frm == "right":
        return (x0 + w * (1 - p), y0, w * p, h)
    if frm == "left":
        return (x0, y0, w * p, h)
    if frm == "top":
        return (x0, y0, w, h * p)
    return (x0, y0 + h * (1 - p), w, h * p)            # bottom


@element("block", dur=0.6, enter="none", sfx="none")
def el_block(ctx, el, s):
    x, y = xy(el, s.t)
    w, h = float(el.get("w", 600)), float(el.get("h", 300))
    p = get_ease(el.get("ease", "in_out_expo"))(s.raw_p)
    q = get_ease("in_out_expo")(s.out_p) if s.out_p else 0.0
    col = _col(el, "color", s, "accent", "#e3262e")
    frm = el.get("side", "right")
    x0, y0 = x - w / 2, y - h / 2
    rx, ry, rw, rh = _edge_rect(x0, y0, w, h, p, frm)
    if q > 0:      # exit: collapses toward the opposite edge
        opp = {"right": "left", "left": "right", "top": "bottom", "bottom": "top"}[frm]
        rx, ry, rw, rh = _edge_rect(x0, y0, w, h, 1 - q, opp)
    if rw > 0.5 and rh > 0.5:
        ctx.set_source_rgba(col[0], col[1], col[2], col[3] * s.alpha)
        ctx.rectangle(rx, ry, rw, rh); ctx.fill()
    if el.get("text") and p > 0.05:
        ctx.save(); ctx.rectangle(rx, ry, rw, rh); ctx.clip()
        tc = _col(el, "text_color", s, "paper", "#ffffff")
        size = float(el.get("size", 72))
        pad = float(el.get("pad", 36))
        _ib = ink_box(el["text"], 0, 0, size, F(el, "title"), max_width=w - 2 * pad, line_height=el.get("line_height", 1.2))
        _fit = min(1.0, (h - 2 * pad * 0.5) / max(1, _ib[3] - _ib[1]), (w - 2 * pad) / max(1, _ib[2] - _ib[0]))
        size *= _fit                                     # text never larger than its slab
        al = el.get("align", "right")
        tx = x0 + w - pad if al == "right" else (x0 + pad if al == "left" else x)
        tp = ease_out(prog(s.t, float(el["t"]) + 0.25, 0.45))
        ty = {"top": y0 + pad, "bottom": y0 + h - pad}.get(el.get("valign", "middle"), y0 + h / 2)
        draw_text(ctx, el["text"], tx, ty + (1 - tp) * size * 0.6, size=size, color=tc,
                  align=al, valign=el.get("valign", "middle"), alpha=s.alpha * tp, weight=F(el, "title"), max_width=w - 2 * pad,
                  line_height=el.get("line_height", 1.2))
        ctx.restore()
    return (x0, y0, x0 + w, y0 + h)


@element("headline", dur=0.9, enter="none", sfx="none")
def el_headline(ctx, el, s):
    """Each line rises from behind its own mask, staggered."""
    x, y = xy(el, s.t)
    raw_lines = str(el["text"]).split("\n")
    parsed = [parse_markup(rl) for rl in raw_lines]          # markup per line (keeps the line breaks)
    lines = [p_[0] for p_ in parsed]
    size = float(el.get("size", 160))
    lh = float(el.get("line_height", 1.08)) * size
    al = el.get("align", "right")
    ax = el.get("_right", x + float(el.get("w", 900)) / 2) if al == "right" else (
        el.get("_left", x - float(el.get("w", 900)) / 2) if al == "left" else x)
    col = _col(el, "color", s, "ink", "#ffffff")
    acc = _col(el, "accent", s, "accent", "#ffffff")
    stg = float(el.get("stagger", 0.12))
    e = get_ease(el.get("ease", "out_expo"))
    fnt = F(el, "title")
    maxw = float(el.get("w", 936))
    widest = max(text_bbox(ln, 0, 0, size, fnt)[2] - text_bbox(ln, 0, 0, size, fnt)[0] for ln in lines if ln.strip())
    if widest > maxw:                       # Swiss: type fills the column, never overflows it
        size *= maxw / widest
        lh = float(el.get("line_height", 1.08)) * size
    # ink-aware leading: the tail/kasra of line i must never touch the dots of line i+1
    gap = float(el.get("ink_gap", 0.05)) * size
    for i in range(len(lines) - 1):
        if lines[i].strip() and lines[i + 1].strip():
            b0 = ink_box(lines[i], 0, 0, size, fnt); b1 = ink_box(lines[i + 1], 0, 0, size, fnt)
            lh = max(lh, (b0[3] - 0) - (b1[1] - 0) + gap)
    bbs = []
    for i, ln in enumerate(lines):
        ly = y + i * lh - (len(lines) - 1) * lh / 2
        p = e(prog(s.t, float(el["t"]) + i * stg, float(el.get("dur", 0.9))))
        q = get_ease("in_expo")(prog(s.t, float(el["out"]) + i * stg * 0.5, 0.45)) if el.get("out") is not None else 0
        if p <= 0:
            continue
        ctx.save()
        mr = mask_rect(ln, ax, ly, size, fnt, align=al)
        if el.get("text_style") in ("extrude_3d",) or (isinstance(el.get("text_style"), dict)
                                                       and el["text_style"].get("kind") == "extrude_3d"):
            mr = (mr[0], mr[1], mr[2], mr[3] + size * 0.3)            # room for the 3D relief below the face
        ctx.rectangle(*mr)   # mask from REAL ink: dots/kasra/tails never cut
        ctx.clip()
        dy = (1 - p) * lh + q * -lh
        local = parsed[i][1]
        if local and local == set(range(len(ln.split()))):     # whole line emphasised → plain colour (fast path)
            c_line, ws = acc, None
        else:
            c_line = col
            ws = (lambda k, local=local: {"color": acc} if k in local else {}) if local else None
        fx = text_fx(el, s, size)
        if fx.get("kashida") and fx["kashida"].get("line", i) != i:
            fx.pop("kashida")
        bb = draw_text(ctx, ln, ax, ly + dy, size=size, color=c_line, align=al, alpha=s.alpha, weight=fnt,
                       word_style=ws, **fx)
        ctx.restore()
        bbs.append(bb)
    if not bbs:
        return None
    return (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))


@element("rule", dur=0.5, enter="none", sfx="none")
def el_rule(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 900))
    wt = float(el.get("weight", 8))
    p = get_ease(el.get("ease", "in_out_expo"))(s.raw_p) * (1 - get_ease("in_out_expo")(s.out_p))
    col = _col(el, "color", s, "ink", "#ffffff")
    frm = el.get("side", "right")
    x0 = x - w / 2
    ctx.set_source_rgba(col[0], col[1], col[2], col[3] * s.alpha)
    if frm == "right":
        ctx.rectangle(x0 + w * (1 - p), y - wt / 2, w * p, wt)
    elif frm == "center":
        ctx.rectangle(x - w * p / 2, y - wt / 2, w * p, wt)
    else:
        ctx.rectangle(x0, y - wt / 2, w * p, wt)
    ctx.fill()
    return (x0, y - wt / 2 - 2, x0 + w, y + wt / 2 + 2)


@element("big_number", dur=1.2, enter="none", sfx="none")
def el_big_number(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 900))
    size = float(el.get("size", 260))
    v = value_at(el, s, default_key="value", default_from=float(el.get("from", 0)), ease="out_expo")
    txt = el.get("fmt", "{:,.0f}").format(v)
    if el.get("fa", True):
        txt = to_fa_digits(txt).replace(",", "٬")
    if el.get("roll"):
        from .horizon import roll_width
        # the widest value this number will ever show + its unit must fit the column (Swiss: never overflow)
        vmax = max([abs(float(k[1])) for k in (el.get("tracks") or {}).get("value", [])] + [abs(v)])
        need = roll_width(vmax, size, F(el, "counter"), el.get("fmt", "{:,.0f}"), el.get("fa", True))
        if el.get("unit"):
            ub = text_bbox(el["unit"], 0, 0, size * 0.26, F(el, "body", key="unit_font"))
            need += (ub[2] - ub[0]) + 40
        if need > w:
            size *= w / need
    al = el.get("align", "right")
    ax = el.get("_right", x + w / 2) if al == "right" else (el.get("_left", x - w / 2) if al == "left" else x)
    rise = get_ease("out_expo")(clamp(s.raw_p * 2.2))
    col = _col(el, "color", s, "ink", "#ffffff")
    punch = value_punch(el, s.t)
    ctx.save()
    ctx.rectangle(*mask_rect(txt, ax, y, size * 1.06, F(el, "counter"), align=al)); ctx.clip()  # rises from a mask
    if el.get("roll"):          # v8 odometer: every digit slot scrolls through its own ink-exact mask
        from .horizon import roll_number
        bb = roll_number(ctx, v, ax, y + (1 - rise) * size, size, col, F(el, "counter"),
                         alpha=s.alpha * (1 - s.out_p), align=al, fmt=el.get("fmt", "{:,.0f}"), fa=el.get("fa", True))
    else:
        bb = draw_text(ctx, txt, ax, y + (1 - rise) * size, size=size * (1 + 0.06 * punch), color=col, align=al,
                       alpha=s.alpha * (1 - s.out_p), weight=F(el, "counter"))
    ctx.restore()
    a2 = s.alpha * ease_out(prog(s.t, float(el["t"]) + 0.35, 0.5)) * (1 - s.out_p)
    mute = _col(el, "caption_color", s, "mute", "#8a8a90")
    if el.get("unit"):
        ux = bb[0] - 24 if al == "right" else bb[2] + 24
        draw_text(ctx, el["unit"], ux, y - size * 0.05, size=size * 0.26, color=col, alpha=a2,
                  align="right" if al == "right" else "left", weight=F(el, "body", key="unit_font"))
    if el.get("caption"):
        draw_text(ctx, el["caption"], ax, max(y + size * 0.42, bb[3] + float(el.get("caption_size", 40)) * 0.9), size=float(el.get("caption_size", 40)), color=mute,
                  alpha=a2, align=al, weight=F(el, "ui_label", key="caption_font"), quiet=True)
    return (min(bb[0], x - w / 2), bb[1], max(bb[2], x + w / 2), y + size * 0.55)


@element("index", dur=1.0, enter="none", sfx="none")
def el_index(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 900))
    gap = float(el.get("gap", 120))
    items = el.get("items", [])
    size = float(el.get("size", 56))
    stg = float(el.get("stagger", 0.18))
    col = _col(el, "color", s, "ink", "#ffffff")
    acc = _col(el, "accent", s, "accent", "#ffffff")
    mute = _col(el, "mute", s, "mute", "#8a8a90")
    hi = el.get("highlight")
    hi_at = el.get("highlight_at")
    hp = ease_in_out(prog(s.t, float(hi_at), 0.4)) if hi_at is not None else 0.0
    right, left = x + w / 2, x - w / 2
    e = get_ease("out_expo")
    for i, it in enumerate(items):
        iy = y + i * gap
        ti = float(el["t"]) + i * stg
        lp = e(prog(s.t, ti, 0.6))
        if lp <= 0:
            continue
        a = s.alpha * (1 - s.out_p)
        ctx.set_source_rgba(mute[0], mute[1], mute[2], a)
        _ry = max(iy + gap * 0.42, ink_box(it, 0, iy, size, F(el, "body"))[3] + size * 0.12)   # rule clears tails
        ctx.rectangle(right - w * lp, _ry, w * lp, 2.5); ctx.fill()
        tp = e(prog(s.t, ti + 0.12, 0.55))
        is_hi = (hi is not None and i == hi)
        c = tuple(lerp(col[k], acc[k], hp) for k in range(4)) if is_hi else (
            tuple(lerp(col[k], mute[k], hp) for k in range(4)) if hi is not None else col)
        m1 = mask_rect(it, right - size * 1.6, iy, size, F(el, "title" if is_hi else "body"), align="right")
        m2 = mask_rect(it, right - size * 1.6, iy, size, F(el, "body" if is_hi else "title"), align="right")
        top = min(m1[1], m2[1]); bot = max(m1[1] + m1[3], m2[1] + m2[3])
        ctx.save(); ctx.rectangle(-4000, top, 9000, bot - top); ctx.clip()
        draw_text(ctx, to_fa_digits(f"{i + 1:02d}"), right, iy + (1 - tp) * size, size=size * 0.6, color=mute,
                  align="right", alpha=a, weight=F(el, "counter"), quiet=True)
        draw_text(ctx, it, right - size * 1.6, iy + (1 - tp) * size, size=size, color=c, align="right", alpha=a,
                  weight=F(el, "title" if is_hi else "body"), quiet=(hi is not None and not is_hi and hp > 0))
        ctx.restore()
    return (left, y - size, right, y + (len(items) - 1) * gap + gap * 0.5)


@element("disc", dur=0.7, enter="none", sfx="none")
def el_disc(ctx, el, s):
    x, y = xy(el, s.t)
    r = float(el.get("r", 200))
    sp = spring_preset(s.t, float(el["t"]), "default") if s.t >= float(el["t"]) else 0.0
    q = get_ease("in_expo")(s.out_p) if s.out_p else 0.0
    rr = r * sp * (1 - q)
    col = _col(el, "color", s, "accent", "#ffffff")
    if rr > 0.5:
        ctx.set_source_rgba(col[0], col[1], col[2], col[3] * s.alpha)
        ctx.arc(x, y, rr, 0, 2 * math.pi); ctx.fill()
    return (x - r, y - r, x + r, y + r)


# ---------------------------------------------------------------------------
# KINETIC
# ---------------------------------------------------------------------------
def kinetic_words(el, scene):
    """[(word, t, end, emph)] — matched to the narration words (subtitles) after el.t, else staggered."""
    if el.get("_kw") is not None:
        return el["_kw"]
    from .scene import _norm_word
    txt, emph = parse_markup(el["text"])
    toks = txt.split()
    t0 = float(el["t"])
    flat = [w for sb in (scene.subs if scene else []) for w in sb["words"] if w["t"] >= t0 - 0.05]
    lead = float(el.get("lead", 0.12))   # land a hair BEFORE the voice: reads in-sync, fills the chapter's first frames
    out, j = [], 0
    rate = float(el.get("rate", 3.2))
    for i, tk in enumerate(toks):
        hit = None
        for k in range(j, min(j + 6, len(flat))):
            if _norm_word(flat[k]["w"]) == _norm_word(tk):
                hit = k; break
        if hit is not None and el.get("sync", True):
            wt = flat[hit]["t"] - lead
            if not out:
                wt = max(t0, wt - 0.25) if wt - t0 < 0.4 else wt     # first word: no empty start
            out.append((tk, wt, flat[hit]["end"], i in emph)); j = hit + 1
        else:
            prev = out[-1][1] if out else t0 - 1 / rate
            out.append((tk, prev + 1 / rate, prev + 2 / rate, i in emph))
    el["_kw"] = out
    return out


def _kfx(el, s, size, em):
    """v9: kinetic words take text_style (all words) or emph_style (only **emphasised** words)."""
    if em and el.get("emph_style"):
        return text_fx(dict(el, text_style=el["emph_style"]), s, size)
    if el.get("text_style") and not (el.get("emph_style") and em):
        return text_fx(el, s, size)
    return {}


@element("kinetic", dur=0.3, enter="none", sfx="none")
def el_kinetic(ctx, el, s):
    words = kinetic_words(el, s.scene)
    if not words:
        return None
    x, y = xy(el, s.t)
    mode = el.get("mode", "slam")
    size = float(el.get("size", 150))
    col = _col(el, "color", s, "ink", "#ffffff")
    acc = _col(el, "accent", s, "accent", "#ffffff")
    mute = _col(el, "mute", s, "mute", "#6a6a70")
    fnt = F(el, "title")
    a_all = s.alpha * (1 - s.out_p)
    t = s.t
    cur = max([i for i, w in enumerate(words) if t >= w[1]], default=-1)
    if cur < 0:
        return None
    bb = None
    if mode == "slam":
        w, wt, we, em = words[cur]
        sp = spring(t, wt, stiffness=520, damping=22)
        sc = (1.9 if em else 1.45) - (0.9 if em else 0.45) * sp
        nxt = words[cur + 1][1] if cur + 1 < len(words) else (float(el["out"]) if el.get("out") is not None else 1e9)
        k_out = ease_in_out(prog(t, nxt - 0.06, 0.06)) if cur + 1 < len(words) else 0.0
        ctx.save(); ctx.translate(x, y); ctx.scale(sc, sc)
        if el.get("enter") == "whip_blur" or el.get("whip"):
            from .apex import whip_word
            ctx.translate(whip_word(t, wt, size), 0)
        bb = draw_text(ctx, w, 0, 0, size=size * (1.12 if em else 1.0), color=acc if em else col,
                       alpha=a_all * clamp((t - wt) / 0.05) * (1 - k_out), weight=fnt,
                       **_kfx(el, s, size, em))
        ctx.restore()
        bb = (x + bb[0] * sc, y + bb[1] * sc, x + bb[2] * sc, y + bb[3] * sc)
    elif mode == "stack":
        # words wrap into lines (max_width); each new word rises from its own mask in place;
        # when there are more than max_lines lines the block glides up.
        maxw = float(el.get("max_width", 920))
        sz = size
        sp_w = sz * 0.28
        widths = [text_bbox(w[0], 0, 0, sz * (1.1 if w[3] else 1.0), fnt)[2] - text_bbox(w[0], 0, 0, sz * (1.1 if w[3] else 1.0), fnt)[0]
                  for w in words]
        lines, curl, cw = [], [], 0.0
        for i, wd in enumerate(widths):
            if curl and cw + sp_w + wd > maxw:
                lines.append(curl); curl, cw = [], 0.0
            curl.append(i); cw += (sp_w if len(curl) > 1 else 0) + wd
        if curl:
            lines.append(curl)
        line_of = {i: li for li, ln in enumerate(lines) for i in ln}
        lh = sz * 1.18
        cur_line = line_of[cur]
        max_lines = int(el.get("max_lines", 4))
        shown_lines = cur_line + 1
        # vertical centring of the visible block, glides when a new line opens
        t_line = words[lines[cur_line][0]][1]
        glide = get_ease("out_expo")(clamp((t - t_line) / 0.35))
        n_vis = min(shown_lines, max_lines)
        prev_vis = min(max(1, shown_lines - 1), max_lines)
        block_h = lerp((prev_vis - 1) * lh, (n_vis - 1) * lh, glide)
        first_vis = max(0, shown_lines - max_lines)
        scroll = lerp(max(0, shown_lines - 1 - max_lines + 1 - 1) if shown_lines > max_lines else 0, first_vis, glide) * lh
        bbs = []
        for li in range(0, shown_lines):
            ly = y - block_h / 2 + li * lh - scroll
            idxs = [i for i in lines[li] if i <= cur]
            # RTL line layout: first word at the right
            total_w = sum(widths[i] for i in lines[li]) + sp_w * (len(lines[li]) - 1)
            xr = x + total_w / 2
            fade = clamp(1 - (first_vis - li)) if li < first_vis else 1.0
            for i in lines[li]:
                if i > cur:
                    break
                w, wt, we, em = words[i]
                p = get_ease("out_expo")(clamp((t - wt) / 0.32))
                c = acc if em else (col if i == cur else tuple(lerp(col[k], mute[k], 0.25) for k in range(4)))
                wx = xr - widths[i] / 2
                ctx.save(); ctx.rectangle(*mask_rect(w, wx, ly, sz * (1.1 if em else 1.0), fnt)); ctx.clip()
                bbs.append(draw_text(ctx, w, wx, ly + (1 - p) * sz * 0.9, size=sz * (1.1 if em else 1.0), color=c,
                                     alpha=a_all * fade, weight=fnt))
                ctx.restore()
                xr -= widths[i] + sp_w
        bb = (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))
    elif mode == "scale":
        # current word grows in the centre; the previous word shrinks & lifts above it, older ones vanish
        w, wt, we, em = words[cur]
        p = get_ease("out_back")(clamp((t - wt) / 0.32)) if "out_back" in __import__("mograph.core", fromlist=["EASINGS"]).EASINGS else get_ease("out_expo")(clamp((t - wt) / 0.32))
        sz = size * (1.15 if em else 1.0)
        ctx.save(); ctx.translate(x, y); sc_ = 0.3 + 0.7 * p; ctx.scale(sc_, sc_)
        b = draw_text(ctx, w, 0, 0, size=sz, color=acc if em else col, alpha=a_all * clamp((t - wt) / 0.06), weight=fnt,
                      **_kfx(el, s, sz, em))
        ctx.restore()
        bbs = [(x + b[0] * sc_, y + b[1] * sc_, x + b[2] * sc_, y + b[3] * sc_)]
        if cur > 0:
            pw, pt, pe, pem = words[cur - 1]
            q = get_ease("out_expo")(clamp((t - wt) / 0.32))
            psz = size * (1.15 if pem else 1.0)
            ps = lerp(1.0, 0.38, q)
            py = y - lerp(0, sz * 0.62 + psz * 0.38 * 0.62 + 30, q)
            ctx.save(); ctx.translate(x, py); ctx.scale(ps, ps)
            pb = draw_text(ctx, pw, 0, 0, size=psz, color=acc if pem else mute, alpha=a_all * lerp(1, 0.85, q), weight=fnt)
            ctx.restore()
            bbs.append((x + pb[0] * ps, py + pb[1] * ps, x + pb[2] * ps, py + pb[3] * ps))
            if cur > 1:   # the one before fades upward out
                ow, ot, oe, oem = words[cur - 2]
                oa = 1 - q
                if oa > 0.02:
                    osz = size * (1.15 if oem else 1.0) * 0.38
                    oy = py - osz * 0.9 - 20 - q * 40
                    draw_text(ctx, ow, x, oy, size=osz, color=mute, alpha=a_all * oa * 0.85, weight=fnt)
        bb = (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))
    elif mode == "scroll":
        lh = size * 1.2
        # continuous scroll position following the word times
        wt0, wt1 = words[cur][1], (words[cur + 1][1] if cur + 1 < len(words) else words[cur][1] + 0.4)
        f = cur + ease_in_out(clamp((t - wt0) / max(0.08, min(0.3, wt1 - wt0))))
        bbs = []
        ctx.save(); ctx.rectangle(-4000, y - lh * 2.6, 9000, lh * 5.2); ctx.clip()
        for i, (w, wt, we, em) in enumerate(words):
            d = i - f
            if abs(d) > 3 or t < wt - 0.6:
                continue
            ly = y + d * lh
            fade = clamp(1 - abs(d) / 3)
            c = (acc if em else col) if i == cur else mute
            bbs.append(draw_text(ctx, w, x, ly, size=size * (1 if i == cur else 0.8), color=c, alpha=a_all * fade,
                                 weight=fnt))
        ctx.restore()
        bb = (x - 400, y - lh * 2.5, x + 400, y + lh * 2.5) if not bbs else \
            (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))
    elif mode == "masonry":
        bb = _kinetic_masonry(ctx, el, s, words, cur, x, y, size, col, acc, mute, fnt, a_all, t)
    elif mode == "highlight_sweep":
        bb = _kinetic_sweep(ctx, el, s, words, cur, x, y, size, col, acc, fnt, a_all, t)
    elif mode == "type":
        full = " ".join(w[0] for w in words)
        t_start, t_end = words[0][1], words[-1][2]
        n = len(full)
        k = int(n * clamp((t - t_start) / max(0.1, t_end - t_start)))
        part = full[:k]
        bb = draw_text(ctx, part or " ", x, y, size=size, color=col, alpha=a_all, weight=fnt,
                       max_width=el.get("max_width", 900), line_height=1.25)
        if int(t * 2.2) % 2 == 0 or k < n:
            cx = bb[0] - 10 if any("\u0600" <= c <= "\u06ff" for c in full) else bb[2] + 10
            ctx.set_source_rgba(acc[0], acc[1], acc[2], a_all)
            ctx.rectangle(cx - 4, bb[3] - size * 0.95, 8, size * 0.95); ctx.fill()
    return bb


# ---------------------------------------------------------------------------
# v8 kinetic modes
# ---------------------------------------------------------------------------
def _wwidth(w, size, fnt):
    b = text_bbox(w, 0, 0, size, fnt)
    return b[2] - b[0]


def masonry_layout(el, words, fnt):
    """Bento packing: split the words into 2–4 rows (or el["rows"] = [n1, n2, …] words per row) and scale
    every row to the SAME pixel width W_box.  Rows are chosen to minimise the size spread (emph words
    prefer their own row) → a short emphasised row becomes huge, a long row small, the block is a locked
    rectangle.  Returns [(idxs, row_size, row_width_at_ref)] and W_box.  Cached on the element."""
    if el.get("_masonry") is not None:
        return el["_masonry"]
    ref = 100.0
    sp = 0.26 * ref
    n = len(words)
    wid = [_wwidth(w[0], ref, fnt) for w in words]
    W = float(el.get("max_width", 900))
    rows_spec = el.get("rows")
    if isinstance(rows_spec, list) and sum(rows_spec) == n:
        groups, k = [], 0
        for c in rows_spec:
            groups.append(list(range(k, k + int(c)))); k += int(c)
    else:
        best = None
        import itertools
        kmax = min(4, n)
        mx = float(el.get("max_size", 420))
        H_lim = float(el.get("max_height", 1100))
        for k in range(min(2, n), kmax + 1):
            for cuts in itertools.combinations(range(1, n), k - 1):
                b = (0,) + cuts + (n,)
                gs = [list(range(b[i], b[i + 1])) for i in range(k)]
                ws = [sum(wid[j] for j in g) + sp * (len(g) - 1) for g in gs]
                sizes = [W / w_ * ref for w_ in ws]
                if max(sizes) > mx * 1.02 or sum(sizes) * 0.92 > H_lim:
                    continue           # a row would exceed max_size / the block would be too tall
                cost = 0.0
                for g, sz in zip(gs, sizes):     # an emphasised word wants its OWN (= huge) row
                    if any(words[j][3] for j in g):
                        cost += 0.0 if len(g) == 1 else 1.2 * (len(g) - 1)
                cost -= 0.8 * (max(sizes) / min(sizes) - 1)    # reward the bento contrast
                cost += 0.35 * abs(k - 3)                       # 3 rows reads best on 9:16
                if best is None or cost < best[0]:
                    best = (cost, gs)
        groups = best[1] if best else [list(range(n))]
    out = []
    for g in groups:
        rw = sum(wid[j] for j in g) + sp * (len(g) - 1)
        out.append((g, min(float(el.get("max_size", 420)), max(float(el.get("min_size", 40)), W / rw * ref)), rw))
    el["_masonry"] = (out, W)
    return el["_masonry"]


def _kinetic_masonry(ctx, el, s, words, cur, x, y, size, col, acc, mute, fnt, a_all, t):
    rows, W = masonry_layout(el, words, fnt)
    gap = float(el.get("line_gap", 0.10))
    # vertical metrics from REAL ink per row (Persian dots / descenders never collide)
    boxes = []
    for g, rs, _ in rows:
        line = " ".join(words[j][0] for j in g)
        ib = ink_box(line, 0, 0, rs, fnt, align="left", valign="bottom")    # baseline at 0 (same as drawing)
        boxes.append((ib[1], ib[3]))
    heights = [b[1] - b[0] for b in boxes]
    total = sum(heights) + gap * size * (len(rows) - 1)
    ytop = y - total / 2
    right = x + W / 2
    bbs = []
    yy = ytop
    for (g, rs, rw), (bt, bbm), hh in zip(rows, boxes, heights):
        cy = yy - bt                         # baseline offset: ink top sits at yy
        xr = right
        # justify exactly to W: distribute leftover from rounding into the gaps
        widths = [_wwidth(words[j][0], rs, fnt) for j in g]
        spw = (W - sum(widths)) / max(1, len(g) - 1) if len(g) > 1 else 0
        for j, wd in zip(g, widths):
            if j > cur:
                break
            w, wt, we, em = words[j]
            p = get_ease("out_expo")(clamp((t - wt) / 0.34))
            sp = spring(t, wt, stiffness=520, damping=24)
            c = acc if em else (col if j == cur else tuple(lerp(col[q], mute[q], 0.18) for q in range(4)))
            wx = xr - wd if len(g) > 1 else right - W + (W - wd) / 2     # single word: centred in W_box
            mr = mask_rect(w, wx, cy, rs, fnt, align="left", valign="bottom")
            ctx.save(); ctx.rectangle(*mr); ctx.clip()
            sc = 1.0 + (0.18 if em else 0.06) * (1 - sp)
            ox_, oy_ = wx + wd / 2, cy - rs * 0.3
            ctx.translate(ox_, oy_); ctx.scale(sc, sc); ctx.translate(-ox_, -oy_)
            bbs.append(draw_text(ctx, w, wx, cy + (1 - p) * hh * 1.05, size=rs, color=c, align="left", valign="bottom",
                                 alpha=a_all, weight=fnt))
            ctx.restore()
            xr -= wd + spw
        yy += hh + gap * size
    if not bbs:
        return None
    return (right - W, ytop, right, ytop + total)


def _kinetic_sweep(ctx, el, s, words, cur, x, y, size, col, acc, fnt, a_all, t):
    """Whole phrase in @ink (words still arrive on the voice); on each **emph** word an @accent marker slab
    sweeps RTL across that word's real ink box and the glyphs under the slab invert to @paper."""
    txt = " ".join(w[0] for w in words)
    maxw = float(el.get("max_width", 920))
    lh = float(el.get("line_height", 1.42))
    paper = _col(el, "paper", s, "paper", "#000000")
    from .text import layout_text as _lt, word_boxes as _wb
    al = el.get("align", "center")
    sd = float(el.get("sweep_dur", 0.32))
    def ws(k):
        if k >= len(words):
            return None
        w = words[k]
        a = clamp((t - w[1]) / 0.12)
        return {"alpha": 0.0 if a <= 0 else a, "dy": (1 - get_ease("out_expo")(a)) * size * 0.25}
    bb = draw_text(ctx, txt, x, y, size=size, color=col, alpha=a_all, weight=fnt, max_width=maxw,
                   line_height=lh, align=al, word_style=ws)
    boxes = _wb(txt, x, y, size, fnt, al, "middle", maxw, lh)
    pad = float(el.get("slab_pad", 0.12)) * size
    for k, w in enumerate(words):
        if not w[3] or t < w[1] or k not in boxes:
            continue
        p = get_ease("in_out_expo")(clamp((t - w[1]) / sd))
        ib = ink_box(w[0], (boxes[k][0] + boxes[k][2]) / 2, (boxes[k][1] + boxes[k][3]) / 2 + 0.0, size, fnt)
        # vertical extent from the word box (optical centre), horizontal from real ink
        bx0, bx1 = min(ib[0], boxes[k][0]) - pad, max(ib[2], boxes[k][2]) + pad
        by0, by1 = boxes[k][1] - pad * 0.3, boxes[k][3] + pad * 0.3
        wL = (bx1 - bx0) * p
        rx = bx1 - wL                       # RTL: grows from the right edge leftwards
        ctx.set_source_rgba(acc[0], acc[1], acc[2], acc[3] * a_all)
        ctx.rectangle(rx, by0, wL, by1 - by0); ctx.fill()
        ctx.save(); ctx.rectangle(rx, by0, wL, by1 - by0); ctx.clip()
        inv = lambda kk, k=k: ({"alpha": 1.0} if kk == k else {"alpha": 0.0})
        from . import contrast as _ctr
        pc = _ctr.ensure(paper, acc, 3.0, prefer=(paper, "#111111", "#ffffff"))
        from . import text as _T
        _m0, _T.AUDIT_MUTE = _T.AUDIT_MUTE, "all"      # inverted overdraw is clipped to the slab by design
        draw_text(ctx, txt, x, y, size=size, color=pc, alpha=a_all, weight=fnt, max_width=maxw, line_height=lh,
                  align=al, word_style=inv)
        _T.AUDIT_MUTE = _m0
        ctx.restore()
        bb = (min(bb[0], bx0), min(bb[1], by0), max(bb[2], bx1), max(bb[3], by1))
    return bb
