"""
mograph.widgets — v5 compound narrative widgets (@ENSAN.BESHAM signature rigs).

Each is ONE declarative element that would otherwise need 10-30 primitives + careful timing:

  quote_rig   forensic quote card; words get underlined / circled with numbered badges on cue
  flow_loop   closed-loop flow graph (institution ↔ people) with travelling particles + flip pill
  laser_scan  legal document tablet; a laser beam sweeps and lights up one clause
  invoice     receipt ledger: item ....... value rows appear on cue, "?" values, final stamp
  strata      ground cross-section: surface icons, underground layers, drilling pipe → "?"
  scale_rig   two-pan balance; items drop on cue, stack, and the beam tilts with spring physics

All times inside lists ("highlights", "rows", "drops") accept word anchors ("w:...") —
they are resolved by scene.py like any other time.
"""
from __future__ import annotations

import math

import cairo

from .core import clamp, color as _color, ease_in_out, ease_out, get_ease, lerp, prog, spring, spring_preset
from .elements import DIM, MID, WHITE, F, LF, element, xy, _bb_union
from .geometry import arrow_d, circle_d, draw_path, point_on_path, rounded_rect_d
from .icons import draw_icon
from .text import draw_text, mask_rect, text_bbox, to_fa_digits, word_boxes

_BADGES = "①②③④⑤⑥⑦⑧⑨"


def _line(ctx, x0, y0, x1, y1, p=1.0, color=WHITE, width=2.2, alpha=1.0, dashed=None):
    draw_path(ctx, f"M {x0} {y0} L {x1} {y1}", 0, 0, 1, anchor="none", trim=p, stroke=color, width=width,
              alpha=alpha, dashed=dashed)


# ---------------------------------------------------------------------------
# 1. quote_rig
# ---------------------------------------------------------------------------

@element("quote_rig", dur=1.0, enter="none", sfx="draw")
def quote_rig(ctx, el, s):
    """keys: text, source, w, size, font(document), highlights:[{word|index, t, style: underline|circle|box,
    num}], header (optional small caption above), image (optional png, dimmed)."""
    x, y = xy(el, s.t)
    w = float(el.get("w", 900))
    size = float(el.get("size", 46))
    fnt = F(el, "document")
    a = s.alpha * (1 - s.out_p)
    text = el["text"]
    lh = el.get("line_height", 2.0 if el.get("highlights") else 1.7)  # room for circles & badges
    tb = text_bbox(text, x, y, size, fnt, max_width=w - 110, line_height=lh)
    h = (tb[3] - tb[1]) + 150 + (60 if el.get("source") else 0)
    top = y - h / 2
    card_p = ease_in_out(prog(s.lt, 0, 0.8))
    draw_path(ctx, rounded_rect_d(x - w / 2, top, w, h, 26), 0, 0, 1, anchor="none", trim=card_p, stroke=WHITE,
              width=2.4, fill="#000", alpha=a)
    # quote marks
    qa_ = clamp((s.lt - 0.3) / 0.4)
    draw_text(ctx, "«", x + w / 2 - 50, top + 50, size=80, color=MID, alpha=a * qa_, weight="title")
    if el.get("header"):
        draw_text(ctx, el["header"], x, top + 46, size=28, color=MID, alpha=a * qa_, weight=LF(el, "body"))
    ty = top + 75 + (tb[3] - tb[1]) / 2 + (20 if el.get("header") else 0)
    tp = clamp((s.lt - 0.4) / 0.9)
    draw_text(ctx, text, x, ty, size=size, weight=fnt, max_width=w - 110, line_height=lh, alpha=a,
              reveal=("words", tp), color="#e8e8ea")
    if el.get("source"):
        draw_text(ctx, "— " + el["source"], x - w / 2 + 60, top + h - 50, size=28, color=MID, align="left",
                  alpha=a * clamp((s.lt - 1.0) / 0.5), weight=LF(el, "body"))
    # highlights
    boxes = word_boxes(text, x, ty, size, fnt, max_width=w - 110, line_height=lh)
    words = text.split()
    for hi, hl in enumerate(el.get("highlights", [])):
        idx = hl.get("index")
        if idx is None:
            target = hl["word"]
            occ = hl.get("occurrence", 1)
            k = 0
            for i, ww in enumerate(words):
                if ww.strip("«»،.؛:!؟") == target:
                    k += 1
                    if k == occ:
                        idx = i
                        break
        if idx is None or idx not in boxes:
            continue
        bx0, by0, bx1, by1 = boxes[idx]
        hp = ease_in_out(prog(s.t, float(hl["t"]), 0.55))
        if hp <= 0:
            continue
        st = hl.get("style", "underline")
        if st == "underline":
            yy = by1 + 2
            _line(ctx, bx1 - 2, yy, bx0 + 2, yy, hp, WHITE, 3.4, a)
        elif st == "box":
            draw_path(ctx, rounded_rect_d(bx0 - 10, by0 - 4, bx1 - bx0 + 20, by1 - by0 + 8, 10), 0, 0, 1,
                      anchor="none", trim=hp, stroke=WHITE, width=2.4, alpha=a)
        else:  # circle (ellipse around the word)
            cx_, cy_ = (bx0 + bx1) / 2, (by0 + by1) / 2
            rx, ry = (bx1 - bx0) / 2 + 12, (by1 - by0) / 2 + 14
            ctx.save(); ctx.translate(cx_, cy_); ctx.scale(rx / 50, ry / 50)
            draw_path(ctx, circle_d(0, 0, 50, -100), 0, 0, 1, anchor="none", trim=hp, stroke=WHITE,
                      width=2.6 * 50 / max(rx, ry), alpha=a)
            ctx.restore()
        num = hl.get("num")
        if num:
            bp = ease_out(prog(s.t, float(hl["t"]) + 0.35, 0.3))
            # numbered badge: small filled disc ABOVE the word (never covers neighbours)
            bxc, byc = (bx0 + bx1) / 2, by0 - 22
            ctx.set_source_rgba(1, 1, 1, a * bp)
            ctx.arc(bxc, byc, 17 * (0.6 + 0.4 * bp), 0, 2 * math.pi); ctx.fill()
            draw_text(ctx, to_fa_digits(num), bxc, byc + 1, size=24, color="#000000", alpha=a * bp, weight="counter")
    return (x - w / 2, top, x + w / 2, top + h)


# ---------------------------------------------------------------------------
# 2. flow_loop
# ---------------------------------------------------------------------------

@element("flow_loop", dur=1.2, enter="none", sfx="draw")
def flow_loop(ctx, el, s):
    """keys: top {icon,label}, left {icon,label}, right {icon,label}, w, h, speed, particles,
    flips: [[t, "label"], ...] for the centre pill, arrow_labels: [left_label, right_label]."""
    x, y = xy(el, s.t)
    w, h = float(el.get("w", 760)), float(el.get("h", 720))
    a = s.alpha * (1 - s.out_p)
    top = el.get("top", {"icon": "government", "label": "دولت"})
    lft = el.get("left", {"icon": "drop", "label": "منابع"})
    rgt = el.get("right", {"icon": "human", "label": "مردم"})
    pts = {"top": (x, y - h / 2 + 90), "left": (x - w / 2 + 90, y + h / 2 - 110), "right": (x + w / 2 - 90, y + h / 2 - 110)}
    for i, (k, node) in enumerate((("top", top), ("right", rgt), ("left", lft))):
        nx, ny = pts[k]
        ip = ease_in_out(prog(s.lt, i * 0.25, 0.9))
        draw_icon(ctx, node["icon"], nx, ny, size=150, p=ip, alpha=a)
        draw_text(ctx, node.get("label", ""), nx, ny + 110, size=34, color=MID, alpha=a * ip, weight=LF(el))
    # curved arrows: top → right (down), right → left (bottom), left → top (up)
    T, R, L = pts["top"], pts["right"], pts["left"]
    curves = [
        f"M {T[0] + 95} {T[1] + 30} Q {R[0] + 40} {T[1] + 40} {R[0] + 10} {R[1] - 110}",
        f"M {R[0] - 95} {R[1] + 10} Q {x} {R[1] + 120} {L[0] + 95} {L[1] + 10}",
        f"M {L[0] - 10} {L[1] - 110} Q {L[0] - 40} {T[1] + 40} {T[0] - 95} {T[1] + 30}",
    ]
    ap = ease_in_out(prog(s.lt, 0.7, 0.9))
    for d in curves:
        draw_path(ctx, arrow_d_on(d), 0, 0, 1, anchor="none", trim=ap, stroke="#9a9aa0", width=2.2, alpha=a)
    if ap >= 1 and el.get("particles", 6):
        n = int(el.get("particles", 6))
        spd = float(el.get("speed", 0.22))
        for d in curves:
            for k in range(n):
                ph = (s.t * spd + k / n) % 1.0
                px, py, _ = point_on_path(d, ph)
                fade = math.sin(math.pi * ph)
                ctx.set_source_rgba(1, 1, 1, a * fade)
                ctx.arc(px, py, 4.5, 0, 2 * math.pi); ctx.fill()
    for i, lab in enumerate(el.get("arrow_labels", [])):
        if not lab:
            continue
        mx, my, _ = point_on_path(curves[i], 0.5)
        off = [(85, -10), (0, 58), (-85, -10)][i]
        draw_text(ctx, lab, mx + off[0], my + off[1], size=el.get("arrow_label_size", 40), color="#b0b0b6",
                  alpha=a * ap, weight=LF(el, "handwritten"))
    # centre flip pill
    flips = el.get("flips")
    if flips:
        cur, prev, tf = flips[0][1], None, float(flips[0][0])
        for f in flips:
            if s.t >= float(f[0]):
                prev, cur, tf = cur, f[1], float(f[0])
        fp = ease_in_out(prog(s.t, tf, 0.5))
        pp = ease_out(prog(s.lt, 1.2, 0.5))
        cy_ = y + 20
        fnt = F(el, "title")
        tb = text_bbox(cur, 0, 0, 44, fnt)
        pw = max(220, tb[2] - tb[0] + 80)
        draw_path(ctx, rounded_rect_d(x - pw / 2, cy_ - 42, pw, 84, 42), 0, 0, 1, anchor="none", stroke=WHITE,
                  width=2.6, fill="#000", alpha=a * pp)
        # in-place flip: old label squashes out, new one stretches in (vertical card flip)
        _mt = min(mask_rect(cur, x, cy_, 44, fnt)[1], cy_ - 40)
        _mb = max(sum(mask_rect(cur, x, cy_, 44, fnt)[1::2]), cy_ + 40)
        ctx.save(); ctx.rectangle(x - pw / 2 + 4, _mt, pw - 8, _mb - _mt); ctx.clip()   # flip window, ink-safe
        if prev is not None and fp < 1:
            ctx.save(); ctx.translate(x, cy_); ctx.scale(1, max(0.01, 1 - fp * 2) if fp < 0.5 else 0.01)
            draw_text(ctx, prev, 0, 0, size=44, alpha=a * pp, weight=fnt); ctx.restore()
        if fp >= 0.5 or prev is None:
            k = 1.0 if prev is None else (fp - 0.5) * 2
            ctx.save(); ctx.translate(x, cy_); ctx.scale(1, max(0.01, k))
            draw_text(ctx, cur, 0, 0, size=44, alpha=a * pp, weight=fnt); ctx.restore()
        ctx.restore()
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2 + 40)


def arrow_d_on(d, head=16):
    """Append an arrow head to the end of a Q/L path string."""
    import re
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", d)]
    x1, y1 = nums[-2], nums[-1]
    cx, cy = nums[-4], nums[-3]
    L = math.hypot(x1 - cx, y1 - cy) or 1
    ux, uy = (x1 - cx) / L, (y1 - cy) / L
    a = (x1 - ux * head - uy * head * 0.6, y1 - uy * head + ux * head * 0.6)
    b = (x1 - ux * head + uy * head * 0.6, y1 - uy * head - ux * head * 0.6)
    return d + f" M {a[0]:.2f} {a[1]:.2f} L {x1} {y1} L {b[0]:.2f} {b[1]:.2f}"


# ---------------------------------------------------------------------------
# 3. laser_scan
# ---------------------------------------------------------------------------

@element("laser_scan", dur=0.9, enter="draw", sfx="draw")
def laser_scan(ctx, el, s):
    """keys: title, clause, w, h, lines, scan_at (time), scan_dur, clause_row (0..lines-1)."""
    x, y = xy(el, s.t)
    w, h = float(el.get("w", 760)), float(el.get("h", 820))
    a = s.alpha * (1 - s.out_p)
    trim = s.trim if isinstance(s.trim, float) else 1.0
    draw_path(ctx, rounded_rect_d(x - w / 2, y - h / 2, w, h, 30), 0, 0, 1, anchor="none", trim=trim,
              stroke=WHITE, width=2.6, fill="#000", alpha=a)
    draw_path(ctx, rounded_rect_d(x - w / 2 + 12, y - h / 2 + 12, w - 24, h - 24, 22), 0, 0, 1, anchor="none",
              trim=clamp(trim * 1.3 - 0.3), stroke=WHITE, width=1.2, alpha=a * 0.5)
    tp = clamp(s.raw_p * 2 - 0.6)
    draw_text(ctx, el.get("title", "قانون اساسی"), x, y - h / 2 + 80, size=50, alpha=a * tp, weight=F(el, "title"))
    _line(ctx, x - w * 0.36, y - h / 2 + 135, x + w * 0.36, y - h / 2 + 135, tp, "#6a6a70", 1.6, a)
    n = int(el.get("lines", 9))
    row = int(el.get("clause_row", n // 2))
    top_y, bot_y = y - h / 2 + 190, y + h / 2 - 70
    gap = (bot_y - top_y) / max(1, n - 1)
    scan_at = el.get("scan_at")
    sp = ease_in_out(prog(s.t, scan_at, float(el.get("scan_dur", 1.8)))) if scan_at is not None else 0.0
    beam_y = lerp(top_y - 30, bot_y + 20, sp)
    for i in range(n):
        ly = top_y + i * gap
        lw = w * (0.72 if i % 3 != 2 else 0.5)
        lp = clamp((s.raw_p - 0.3 - i * 0.04) * 3)
        if i == row:
            continue
        lit = 0.0
        if 0 < sp < 1:
            lit = clamp(1 - abs(ly - beam_y) / 60)
        g = lerp(0.24, 0.85, lit)
        _line(ctx, x + lw / 2, ly, x - lw / 2, ly, lp, (g, g, g, 1), 5, a)
    # clause box (revealed after the beam passes it)
    cy_ = top_y + row * gap
    cp = ease_out(clamp((beam_y - cy_) / 80)) if scan_at is not None else 0.0
    if cp > 0:
        cw = w * 0.86
        draw_path(ctx, rounded_rect_d(x - cw / 2, cy_ - 36, cw, 72, 18), 0, 0, 1, anchor="none", trim=cp,
                  stroke=WHITE, width=2.2, fill="#000", alpha=a)
        draw_text(ctx, el.get("clause", ""), x, cy_, size=34, alpha=a * cp, weight=F(el, "document"),
                  max_width=cw - 40)
    if 0 < sp < 1:
        for k, (wd, al) in enumerate(((14, 0.10), (6, 0.35), (2.6, 1.0))):
            _line(ctx, x - w / 2 + 26, beam_y, x + w / 2 - 26, beam_y, 1.0, WHITE, wd, a * al)
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


# ---------------------------------------------------------------------------
# 4. invoice
# ---------------------------------------------------------------------------

@element("invoice", dur=0.8, enter="draw", sfx="draw")
def invoice(ctx, el, s):
    """keys: title, rows: [{item, value, t}], total {label, value, t}, stamp {text, t}, w."""
    x, y = xy(el, s.t)
    rows = el.get("rows", [])
    w = float(el.get("w", 800))
    rh = float(el.get("row_h", 92))
    h = 190 + rh * len(rows) + (120 if el.get("total") else 0)
    top = y - h / 2
    a = s.alpha * (1 - s.out_p)
    trim = s.trim if isinstance(s.trim, float) else 1.0
    draw_path(ctx, rounded_rect_d(x - w / 2, top, w, h, 28), 0, 0, 1, anchor="none", trim=trim, stroke=WHITE,
              width=2.6, fill="#000", alpha=a)
    tp = clamp(s.raw_p * 2 - 0.6)
    draw_text(ctx, el.get("title", "صورت‌حساب"), x, top + 72, size=48, alpha=a * tp, weight=F(el, "title"))
    _line(ctx, x - w / 2 + 40, top + 130, x + w / 2 - 40, top + 130, tp, "#6a6a70", 1.6, a, dashed=(8, 8))
    fdoc = F(el, "document")
    fnum = resolve_num = F(el, "counter", key="value_font")
    for i, r in enumerate(rows):
        ry = top + 190 + i * rh
        rp = ease_out(prog(s.t, float(r.get("t", el["t"] + 0.6 + i * 0.4)), 0.5))
        if rp <= 0:
            continue
        ib = draw_text(ctx, r["item"], x + w / 2 - 50, ry, size=38, align="right", alpha=a * rp, weight=fdoc)
        val = str(r.get("value", ""))
        if val == "?":
            bx = x - w / 2 + 90
            draw_path(ctx, rounded_rect_d(bx - 34, ry - 34, 68, 68, 14), 0, 0, 1, anchor="none", trim=rp,
                      stroke=WHITE, width=2.4, alpha=a)
            draw_text(ctx, "؟", bx, ry, size=44, alpha=a * rp, weight="title")
            vb = (bx - 34, ry - 34, bx + 34, ry + 34)
        else:
            vb = draw_text(ctx, val, x - w / 2 + 50, ry, size=38, align="left", alpha=a * rp, weight=fnum)
        # dotted leader between value and item
        lx0, lx1 = vb[2] + 18, ib[0] - 18
        if lx1 > lx0:
            n = int((lx1 - lx0) / 16)
            for k in range(n):
                if (k + 1) / n > rp * 1.4:
                    break
                ctx.set_source_rgba(0.55, 0.55, 0.58, a)
                ctx.arc(lx1 - k * 16, ry + 10, 2.4, 0, 2 * math.pi); ctx.fill()
    tot = el.get("total")
    if tot:
        ty = top + 190 + len(rows) * rh + 30
        tp2 = ease_out(prog(s.t, float(tot.get("t", el["t"] + 2)), 0.5))
        _line(ctx, x + w / 2 - 40, ty - 40, x - w / 2 + 40, ty - 40, tp2, WHITE, 2.2, a)
        draw_text(ctx, tot.get("label", "جمع"), x + w / 2 - 50, ty + 10, size=42, align="right", alpha=a * tp2,
                  weight=F(el, "title"))
        draw_text(ctx, str(tot.get("value", "")), x - w / 2 + 50, ty + 10, size=46, align="left", alpha=a * tp2,
                  weight=fnum)
    st = el.get("stamp")
    if st:
        t0 = float(st.get("t", el["t"] + 3))
        if s.t >= t0:
            sp = spring_preset(s.t, t0, "stamp")
            sc = 1.8 - 0.8 * sp
            sa = a * clamp((s.t - t0) / 0.08)
            ctx.save(); ctx.translate(x, y + h * 0.12); ctx.rotate(math.radians(st.get("angle", -9))); ctx.scale(sc, sc)
            fs = F(el, "title", key="stamp_font")
            tb = text_bbox(st["text"], 0, 0, 64, fs)
            sw, sh = tb[2] - tb[0] + 110, 130
            draw_path(ctx, rounded_rect_d(-sw / 2, -sh / 2, sw, sh, 18), 0, 0, 1, anchor="none", stroke=WHITE,
                      width=4, fill="#000", alpha=sa)
            draw_path(ctx, rounded_rect_d(-sw / 2 + 11, -sh / 2 + 11, sw - 22, sh - 22, 12), 0, 0, 1, anchor="none",
                      stroke=WHITE, width=1.6, alpha=sa * 0.7)
            draw_text(ctx, st["text"], 0, 0, size=64, alpha=sa, weight=fs)
            ctx.restore()
    return (x - w / 2, top, x + w / 2, top + h)


# ---------------------------------------------------------------------------
# 5. strata
# ---------------------------------------------------------------------------

@element("strata", dur=1.0, enter="draw", sfx="draw")
def strata(ctx, el, s):
    """keys: w, surface: [icon names], layers (int), resources (icon name or None), drill_at, drill_x,
    label_top, label_bottom, end ("?" morph tip)."""
    x, y = xy(el, s.t)
    w = float(el.get("w", 960))
    hz = y - 120  # horizon line
    a = s.alpha * (1 - s.out_p)
    trim = s.trim if isinstance(s.trim, float) else 1.0
    _line(ctx, x + w / 2, hz, x - w / 2, hz, trim, WHITE, 2.6, a)
    for i, nm in enumerate(el.get("surface", ["home", "human", "car"])):
        n = len(el.get("surface", [1, 2, 3]))
        sx = x + w / 2 - (i + 0.5) * w / n
        draw_icon(ctx, nm, sx, hz - 70, size=120, p=ease_in_out(prog(s.lt, 0.3 + i * 0.15, 0.8)), alpha=a)
    nl = int(el.get("layers", 4))
    for k in range(nl):
        ly = hz + 90 + k * 95
        amp = 14 + 6 * (k % 2)
        d = f"M {x + w / 2} {ly}"
        segs = 6
        for j in range(segs):
            x0 = x + w / 2 - (j + 1) * w / segs
            d += f" Q {x + w / 2 - (j + 0.5) * w / segs} {ly + (amp if j % 2 else -amp)} {x0} {ly}"
        lp = ease_in_out(prog(s.lt, 0.5 + k * 0.12, 0.9))
        g = 0.42 - k * 0.06
        draw_path(ctx, d, 0, 0, 1, anchor="none", trim=lp, stroke=(g, g, g + 0.02, 1), width=2, alpha=a)
    # drifting resource particles
    rng_seed = 7
    for k in range(int(el.get("particles", 26))):
        ph = ((k * 0.618 + s.t * 0.04) % 1.0)
        px = x + w / 2 - ph * w
        py = hz + 120 + ((k * 53 + rng_seed) % (nl * 95 - 60))
        py += 6 * math.sin(s.t * 1.3 + k)
        ctx.set_source_rgba(0.75, 0.75, 0.78, a * 0.55 * clamp((s.lt - 1.0) / 0.8))
        ctx.arc(px, py, 3.2, 0, 2 * math.pi); ctx.fill()
    if el.get("label_top"):
        draw_text(ctx, el["label_top"], x - w / 2 + 20, hz - 30, size=40, color="#b0b0b6", align="left", alpha=a * trim,
                  weight=LF(el, "handwritten"))
    if el.get("label_bottom"):
        draw_text(ctx, el["label_bottom"], x - w / 2 + 20, hz + nl * 95 + 60, size=40, color="#b0b0b6", align="left",
                  alpha=a * trim, weight=LF(el, "handwritten"))
    dat = el.get("drill_at")
    if dat is not None:
        dx = x + float(el.get("drill_x", 0))
        depth = hz + nl * 95
        dp = ease_in_out(prog(s.t, dat, 1.4))
        up = ease_in_out(prog(s.t, dat + 1.5, 1.0))
        # double-walled dashed pipe: down, then bends and rises back up past the horizon
        rise = float(el.get("rise", 150))
        bend = float(el.get("bend", 180))
        d = f"M {dx} {hz - 40} L {dx} {depth - 40} Q {dx} {depth + 20} {dx - 60} {depth + 20} " \
            f"L {dx - bend + 60} {depth + 20} Q {dx - bend} {depth + 20} {dx - bend} {depth - 40} L {dx - bend} {hz - rise}"
        tot = dp * 0.62 + up * 0.38
        for off in (-9, 9):
            ctx.save(); ctx.translate(off * 0.7, 0)
            draw_path(ctx, d, 0, 0, 1, anchor="none", trim=tot, stroke=WHITE, width=2, alpha=a, dashed=None)
            ctx.restore()
        if up >= 1 and el.get("end", "?"):
            qp = ease_out(prog(s.t, dat + 2.5, 0.6))
            qs = float(el.get("q_size", 130))
            draw_icon(ctx, "question", dx - bend, hz - rise - qs * 0.55, size=qs * (0.6 + 0.4 * qp), p=qp, alpha=a,
                      style="bold")
    top_extent = hz - float(el.get("rise", 150)) - float(el.get("q_size", 130)) * 1.1 if dat is not None else hz - 160
    return (x - w / 2, top_extent, x + w / 2, hz + nl * 95 + 90)


# ---------------------------------------------------------------------------
# 6. scale_rig
# ---------------------------------------------------------------------------

@element("scale_rig", dur=1.0, enter="draw", sfx="draw")
def scale_rig(ctx, el, s):
    """keys: left {label, item}, right {label}, drops: [{t, label}], max_tilt (deg), w."""
    x, y = xy(el, s.t)
    w = float(el.get("w", 820))
    a = s.alpha * (1 - s.out_p)
    trim = s.trim if isinstance(s.trim, float) else 1.0
    drops = el.get("drops", [])
    max_tilt = float(el.get("max_tilt", 16))
    # tilt: each landed drop adds a step; spring for the swing
    tilt = 0.0
    for i, dr in enumerate(drops):
        land = float(dr["t"]) + 0.55
        if s.t >= land:
            tilt += (max_tilt / max(1, len(drops))) * spring(s.t, land, stiffness=150, damping=7)
    ang = math.radians(tilt)  # positive → right pan goes DOWN
    py = y - 260
    # pillar + base
    _line(ctx, x, py, x, y + 330, trim, WHITE, 3, a)
    _line(ctx, x - 150, y + 330, x + 150, y + 330, trim, WHITE, 3, a)
    draw_path(ctx, circle_d(x, py, 14), 0, 0, 1, anchor="none", trim=trim, stroke=WHITE, width=2.6, fill="#000",
              alpha=a)
    half = w / 2 - 40
    lx, ly = x - half * math.cos(ang), py - half * math.sin(ang)
    rx, ry = x + half * math.cos(ang), py + half * math.sin(ang)
    _line(ctx, lx, ly, rx, ry, trim, WHITE, 3.2, a)
    chain = 170
    pans = []
    for side, (hx, hy) in (("left", (lx, ly)), ("right", (rx, ry))):
        pyb = hy + chain
        _line(ctx, hx, hy, hx - 85, pyb, trim, "#9a9aa0", 1.6, a)
        _line(ctx, hx, hy, hx + 85, pyb, trim, "#9a9aa0", 1.6, a)
        d = f"M {hx - 115} {pyb} Q {hx} {pyb + 70} {hx + 115} {pyb}"
        draw_path(ctx, d + f" L {hx - 115} {pyb}", 0, 0, 1, anchor="none", trim=trim, stroke=WHITE, width=2.6,
                  fill="#000", alpha=a)
        pans.append((hx, pyb))
        cfg = el.get(side, {})
        if cfg.get("label"):
            draw_text(ctx, cfg["label"], hx, pyb + 95, size=36, color=MID, alpha=a * trim, weight=LF(el, "body"))
    # left item (single light item, e.g. «یه تکه کاغذ»)
    li = el.get("left", {}).get("item")
    if li:
        hx, pyb = pans[0]
        draw_text(ctx, li, hx, pyb - 38, size=el.get("item_size", 40), alpha=a * clamp(s.raw_p * 2 - 0.5),
                  weight=LF(el, "handwritten"))
    # right stacked drops: 2-column grid
    hx, pyb = pans[1]
    fnt = F(el, "body")
    for i, dr in enumerate(drops):
        t0 = float(dr["t"])
        if s.t < t0:
            continue
        col, rowi = i % 2, i // 2
        tx = hx + (55 if col else -55)
        ty = pyb - 30 - rowi * 54
        fall = ease_in_out(prog(s.t, t0, 0.55)) ** 1.6
        cy_ = lerp(ty - 520, ty, fall)
        tb = text_bbox(dr["label"], 0, 0, 26, fnt)
        bw = max(100, tb[2] - tb[0] + 30)
        draw_path(ctx, rounded_rect_d(tx - bw / 2, cy_ - 24, bw, 48, 12), 0, 0, 1, anchor="none", stroke=WHITE,
                  width=2, fill="#000", alpha=a * clamp((s.t - t0) / 0.15))
        draw_text(ctx, dr["label"], tx, cy_, size=26, alpha=a * clamp((s.t - t0) / 0.15), weight=fnt)
    return (x - w / 2 - 80, py - 40, x + w / 2 + 80, y + 340)
