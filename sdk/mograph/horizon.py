"""
mograph.horizon — v8 "Studio Horizon" elements.

SWISS DATA-JOURNALISM WIDGETS (grid-aware: "col"/"row" fill x / w / y)
  waffle_grid   10×10 percentage matrix; `value` cells pop to @accent at `fill_at`, the rest recede to @mute,
                flush Swiss callout bars (rule + swatch + label) above / below
                {"type":"waffle_grid","col":[1,6],"row":6,"value":1,"total":100,"fill_at":"w:درصد",
                 "accent_label":"۱٪ ثروتمندترین","base_label":"۹۹٪ بقیه مردم","side":620,"cell_r":0}
  swiss_bars    full-bleed slab bars with cut-out type: label + value INSIDE the slab in @paper, or outside in
                @ink when the slab is too short (decided per bar by real text width + contrast.py)
                {"type":"swiss_bars","col":[1,6],"row":7,"items":[{"label":"نفت","value":72,"t":"w:نفت"},...],
                 "max":100,"suffix":"٪","highlight":0,"bar_h":110,"gap":22,"orient":"h|v"}
  versus_table  2-column editorial comparison: heavy top / middle / bottom rules, centre divider, row-by-row
                masked slide-in, @accent reality column (left)
                {"type":"versus_table","col":[1,6],"row":4,"right_title":"ادعای رسمی","left_title":"واقعیت",
                 "rows":[{"right":"یارانه به همه","left":"تورم ۳ برابری","t":"w:تورم"}, ...]}

AUDIO-REACTIVE
  waveform      symmetric bar / line oscilloscope driven by scene.vo_env(t) (real VO envelope, or the
                subtitle cadence offline)   {"type":"waveform","x":540,"y":1300,"w":760,"h":160,"bars":31,
                 "mode":"bars|line|mirror"}
  idle:"vo_pulse" (elements.compute_state) — scale *= 1 + pulse_amount·vo_env(t); icons money_core /
                ai_monogram also emit an inner ring that breathes with the voice.

ODOMETER
  roll_number() — mechanical split-flap / odometer digits: each digit slot scrolls vertically through its own
                ink-exact mask.  Used by big_number / counter with "roll": true.
"""
from __future__ import annotations

import math

from .core import clamp, color as _color, ease_in_out, ease_out, get_ease, lerp, prog, spring
from .elements import F, element, value_at, value_punch, xy
from .text import draw_text, ink_box, mask_rect, text_bbox, to_fa_digits
from . import contrast as _ct


def _pal_col(el, s, key, token, default):
    from .typo import _col
    return _col(el, key, s, token, default)


def _bg(el, s):
    from .typo import bg_of
    return _color(bg_of(el, s))


def _rgba(c, a):
    return (c[0], c[1], c[2], c[3] * a)


# ---------------------------------------------------------------------------
# ODOMETER — shared by big_number / counter ("roll": true)
# ---------------------------------------------------------------------------
_DIGITS_FA = "۰۱۲۳۴۵۶۷۸۹"


def _digit_pos(v, k):
    """Continuous wheel position of the digit with place value 10^k (mechanical carry)."""
    v = max(0.0, v)
    if k == 0:
        return v
    p = 10.0 ** k
    lower = v % p
    roll = clamp(lower - (p - 1))
    return math.floor(v / p) + roll


def _char_w(c, size, weight):
    from .text import ink_box
    b = ink_box(c, 0, 0, size, weight)
    return max(size * 0.14, (b[2] - b[0]) + size * 0.045)

def roll_width(v, size, weight, fmt="{:,.0f}", fa=True, sep="٬", slot_gap=0.04):
    txt = fmt.format(max(0.0, v))
    if fa:
        txt = to_fa_digits(txt).replace(",", sep)
    return sum(_char_w(c, size, weight) * (1.0 + slot_gap) for c in txt)


def roll_number(ctx, v, x, y, size, color, weight, alpha=1.0, align="right", fmt="{:,.0f}", fa=True,
                slot_gap=0.04, sep="٬"):
    """Draw value v as an odometer at (x, y) with proportional Persian digit widths so '۰' and '۱'
    never have wide tabular gaps. Returns ink bbox."""
    txt = fmt.format(max(0.0, round(v, 6)) if v >= 0 else v)
    if fa:
        txt = to_fa_digits(txt).replace(",", sep)
    digits = _DIGITS_FA if fa else "0123456789"
    d_widths = [_char_w(d, size, weight) * (1.0 + slot_gap) for d in digits]
    n_int = sum(1 for c in txt.split(".")[0] if c in digits)
    k_pre = n_int - 1
    widths = []
    for c in txt:
        if c in digits:
            pos = _digit_pos(v, k_pre) if k_pre >= 0 else 0.0
            d0 = int(math.floor(pos)) % 10
            fr = pos - math.floor(pos)
            widths.append(lerp(d_widths[d0], d_widths[(d0 + 1) % 10], fr))
            k_pre -= 1
        else:
            widths.append(_char_w(c, size, weight) * (1.0 + slot_gap))
    total = sum(widths)
    x0 = x - total if align == "right" else (x - total / 2 if align == "center" else x)
    m = mask_rect(digits, 0, y, size, weight)          # ink-exact slot height (dots of ۰..۹ never cut)
    top, hh = m[1], m[3]
    lh = hh * 1.02
    n_int = sum(1 for c in txt.split(".")[0] if c in digits)
    k = n_int - 1
    cx = x0
    bbs = []
    for c, w in zip(txt, widths):
        mid = cx + w / 2
        if c in digits:
            pos = _digit_pos(v, k) if k >= 0 else 0.0
            d0 = int(math.floor(pos)) % 10
            fr = pos - math.floor(pos)
            ctx.save(); ctx.rectangle(cx - 2, top, w + 4, hh); ctx.clip()
            from . import text as _T
            _mute0, _T.AUDIT_MUTE = _T.AUDIT_MUTE, (_T.AUDIT_MUTE or ("clip" if fr > 1e-3 else None))
            leading = k == n_int - 1 and n_int > 1 and math.floor(pos) == 0 and fr > 0
            if not leading:
                bbs.append(draw_text(ctx, digits[d0], mid, y - fr * lh, size=size, color=color, alpha=alpha,
                                     weight=weight))
            if fr > 1e-3:
                draw_text(ctx, digits[(d0 + 1) % 10], mid, y + (1 - fr) * lh, size=size, color=color,
                          alpha=alpha, weight=weight)
            _T.AUDIT_MUTE = _mute0
            ctx.restore()
            k -= 1
        else:
            bbs.append(draw_text(ctx, c, mid, y, size=size, color=color, alpha=alpha, weight=weight))
        cx += w
    # layout bbox = the STATIC number at rest (the wheels move inside their masks; QA must not see them jump)
    ib = ink_box(txt, x0 + total / 2, y, size, weight)
    return (x0, max(top, ib[1]), x0 + total, min(top + hh, ib[3]))


# ---------------------------------------------------------------------------
# WAFFLE GRID
# ---------------------------------------------------------------------------
@element("waffle_grid", dur=1.0, enter="none", sfx="none")
def waffle_grid(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 900))
    side = float(el.get("side") or min(w, float(el.get("h") or w)))
    n_side = int(el.get("n", 10))
    total = float(el.get("total", 100))
    val = float(el.get("value", 1))
    k_acc = int(round(val / total * n_side * n_side))
    gap = float(el.get("gap", side * 0.014))
    cell = (side - gap * (n_side - 1)) / n_side
    r = float(el.get("cell_r", 0.0)) * cell
    ink = _pal_col(el, s, "color", "ink", "#111111")
    acc = _pal_col(el, s, "accent", "accent", "#e3262e")
    mute = _pal_col(el, s, "mute", "mute", "#8c8a86")
    t0 = float(el["t"])
    fat = el.get("fill_at")
    fat = float(fat) if fat is not None else t0 + 1.2
    a_all = s.alpha * (1 - s.out_p)
    al = el.get("align", "right")
    gx1 = el.get("_right", x + w / 2) if al == "right" else (x + side / 2 if al == "center" else el.get("_left", x - w / 2) + side)
    gx0 = gx1 - side
    gy0 = y - side / 2
    dim = ease_in_out(prog(s.t, fat, 0.45))
    build = float(el.get("build", 0.7))
    for i in range(n_side * n_side):
        rr, cc = divmod(i, n_side)
        # fill order: bottom-right → leftwards → upwards (RTL reading, Swiss bottom-up count)
        row = n_side - 1 - rr
        px = gx1 - (cc + 1) * cell - cc * gap
        py = gy0 + row * (cell + gap)
        dd = (rr + cc) / (2 * n_side - 2)
        ap = ease_out(prog(s.t, t0 + dd * build, 0.25))
        if ap <= 0:
            continue
        is_acc = i < k_acc
        if is_acc:
            sp = spring(s.t, fat + i * 0.03, stiffness=420, damping=16) if s.t >= fat + i * 0.03 else 0.0
            col = tuple(lerp(ink[j], acc[j], clamp(sp)) for j in range(4))
            sc = 1.0 + 0.35 * math.sin(math.pi * clamp(sp)) * (1 - clamp((s.t - fat - 0.5) / 0.3)) if sp > 0 else 1.0
        else:
            col = tuple(lerp(ink[j], mute[j], dim * 0.85) for j in range(4))
            sc = 1.0 - 0.12 * dim
        c2 = cell * sc * (0.55 + 0.45 * ap)
        cx_, cy_ = px + cell / 2, py + cell / 2
        ctx.set_source_rgba(*_rgba(col, a_all * ap))
        if r > 0.5:
            rr_ = min(r, c2 / 2)
            x_, y_ = cx_ - c2 / 2, cy_ - c2 / 2
            ctx.new_sub_path()
            ctx.arc(x_ + c2 - rr_, y_ + rr_, rr_, -math.pi / 2, 0)
            ctx.arc(x_ + c2 - rr_, y_ + c2 - rr_, rr_, 0, math.pi / 2)
            ctx.arc(x_ + rr_, y_ + c2 - rr_, rr_, math.pi / 2, math.pi)
            ctx.arc(x_ + rr_, y_ + rr_, rr_, math.pi, 1.5 * math.pi)
            ctx.close_path(); ctx.fill()
        else:
            ctx.rectangle(cx_ - c2 / 2, cy_ - c2 / 2, c2, c2); ctx.fill()
    # flush Swiss callout bars
    lsz = float(el.get("label_size", 44))
    bb = [gx0, gy0, gx1, gy0 + side]
    fnt = F(el, "body", key="label_font")
    for which, label, colr, sign in (("top", el.get("accent_label"), acc, -1), ("bottom", el.get("base_label"), mute, 1)):
        if not label:
            continue
        lp = get_ease("out_expo")(prog(s.t, fat + (0.05 if which == "top" else 0.25), 0.55))
        if lp <= 0:
            continue
        ry = gy0 - lsz * 1.55 if which == "top" else gy0 + side + lsz * 1.55
        rule_y = gy0 - lsz * 0.55 if which == "top" else gy0 + side + lsz * 0.55
        wt = float(el.get("rule_weight", 6))
        ctx.set_source_rgba(*_rgba(ink, a_all))
        ctx.rectangle(gx1 - side * lp, rule_y - wt / 2, side * lp, wt); ctx.fill()
        sw = lsz * 0.62
        ctx.set_source_rgba(*_rgba(colr, a_all * lp))
        ctx.rectangle(gx1 - sw, ry - sw / 2, sw, sw); ctx.fill()
        tc = _ct.ensure(ink, _bg(el, s), 4.5)
        ctx.save(); ctx.rectangle(*mask_rect(label, gx1 - sw * 1.6, ry, lsz, fnt, align="right")); ctx.clip()
        b = draw_text(ctx, label, gx1 - sw * 1.6, ry + (1 - lp) * lsz, size=lsz, color=tc, align="right",
                      alpha=a_all, weight=fnt)
        ctx.restore()
        bb[1] = min(bb[1], b[1]); bb[3] = max(bb[3], b[3])
    return tuple(bb)


# ---------------------------------------------------------------------------
# SWISS BARS
# ---------------------------------------------------------------------------
@element("swiss_bars", dur=0.8, enter="none", sfx="none")
def swiss_bars(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 900))
    items = el.get("items") or []
    n = max(1, len(items))
    orient = el.get("orient", "h")
    vmax = float(el.get("max") or max([float(it.get("value", 0)) for it in items] + [1e-6]))
    bar_h = float(el.get("bar_h", 110))
    gap = float(el.get("gap", 22))
    stg = float(el.get("stagger", 0.22))
    dur = float(el.get("dur", 0.8))
    ink = _pal_col(el, s, "color", "ink", "#111111")
    acc = _pal_col(el, s, "accent", "accent", "#e3262e")
    paper = _pal_col(el, s, "paper", "paper", "#efece6")
    bg = _bg(el, s)
    hi = el.get("highlight")
    hi = hi if isinstance(hi, list) else ([hi] if hi is not None else [])
    a_all = s.alpha * (1 - s.out_p)
    lsz = float(el.get("label_size", bar_h * 0.36))
    vsz = float(el.get("value_size", bar_h * 0.5))
    lf, vf = F(el, "body", key="label_font"), F(el, "counter", key="value_font")
    fmt, suffix = el.get("fmt", "{:,.0f}"), el.get("suffix", "")
    pad = float(el.get("pad", bar_h * 0.22))
    e = get_ease(el.get("ease", "out_expo"))
    right = el.get("_right", x + w / 2)
    if orient == "v":
        return _swiss_bars_v(ctx, el, s, x, y, w, items, vmax, ink, acc, paper, bg, hi, a_all, lsz, vsz, lf, vf, fmt,
                             suffix, e, stg, dur)
    total_h = n * bar_h + (n - 1) * gap
    y0 = y - total_h / 2
    bb = [right - w, y0, right, y0 + total_h]
    for i, it in enumerate(items):
        ti = float(it["t"]) if it.get("t") is not None else float(el["t"]) + i * stg
        p = e(prog(s.t, ti, dur))
        if p <= 0:
            continue
        v = float(it.get("value", 0))
        L = w * clamp(v / vmax) * p
        by = y0 + i * (bar_h + gap)
        col = acc if (i in hi or it.get("accent")) else ink
        ctx.set_source_rgba(*_rgba(col, a_all))
        ctx.rectangle(right - L, by, L, bar_h); ctx.fill()
        vtxt = to_fa_digits(fmt.format(v * p) + suffix).replace(",", "٬")
        vfin = to_fa_digits(fmt.format(v) + suffix).replace(",", "٬")
        lab = it.get("label", "")
        lw = (text_bbox(lab, 0, 0, lsz, lf)[2] - text_bbox(lab, 0, 0, lsz, lf)[0]) if lab else 0
        vw = text_bbox(vfin, 0, 0, vsz, vf)[2] - text_bbox(vfin, 0, 0, vsz, vf)[0]
        Lfin = w * clamp(v / vmax)
        inside = Lfin >= lw + vw + pad * 3.2                     # decided on the FINAL length → no flip mid-grow
        cy = by + bar_h / 2
        tin = _ct.ensure(paper, col, 4.5, prefer=(paper, "#ffffff", "#111111"))
        tout = _ct.ensure(ink, bg, 4.5, prefer=(ink, "#111111", "#ffffff"))
        ta = a_all * clamp((s.t - ti - dur * 0.25) / 0.25)
        if inside:
            ctx.save(); ctx.rectangle(right - L, by, L, bar_h); ctx.clip()
            draw_text(ctx, lab, right - pad, cy, size=lsz, color=tin, align="right", alpha=ta, weight=lf)
            draw_text(ctx, vtxt, right - L + pad, cy, size=vsz, color=tin, align="left", alpha=ta, weight=vf)
            ctx.restore()
        else:                  # short slab: value just outside the bar end, label inside if it fits else outside
            lab_in = Lfin >= lw + pad * 2
            if lab_in:
                ctx.save(); ctx.rectangle(right - L, by, L, bar_h); ctx.clip()
                draw_text(ctx, lab, right - pad, cy, size=lsz, color=tin, align="right", alpha=ta, weight=lf)
                ctx.restore()
                draw_text(ctx, vtxt, right - L - pad, cy, size=vsz, color=tout, align="right", alpha=ta, weight=vf)
            else:
                vb = draw_text(ctx, vtxt, right - L - pad, cy, size=vsz, color=tout, align="right", alpha=ta, weight=vf)
                draw_text(ctx, lab, vb[0] - pad, cy, size=lsz, color=tout, align="right", alpha=ta, weight=lf)
    return tuple(bb)


def _swiss_bars_v(ctx, el, s, x, y, w, items, vmax, ink, acc, paper, bg, hi, a_all, lsz, vsz, lf, vf, fmt, suffix,
                  e, stg, dur):
    n = max(1, len(items))
    h = float(el.get("h", 700))
    gap = float(el.get("gap", 24))
    bw = (w - gap * (n - 1)) / n
    right = el.get("_right", x + w / 2)
    base = y + h / 2
    for i, it in enumerate(items):
        ti = float(it["t"]) if it.get("t") is not None else float(el["t"]) + i * stg
        p = e(prog(s.t, ti, dur))
        if p <= 0:
            continue
        v = float(it.get("value", 0))
        L = h * clamp(v / vmax) * p
        bx = right - (i + 1) * bw - i * gap            # RTL: first bar on the right
        col = acc if (i in hi or it.get("accent")) else ink
        ctx.set_source_rgba(*_rgba(col, a_all))
        ctx.rectangle(bx, base - L, bw, L); ctx.fill()
        ta = a_all * clamp((s.t - ti - dur * 0.25) / 0.25)
        vtxt = to_fa_digits(fmt.format(v * p) + suffix).replace(",", "٬")
        vs = min(vsz, bw * 0.42)
        tin = _ct.ensure(paper, col, 4.5, prefer=(paper, "#ffffff", "#111111"))
        tout = _ct.ensure(ink, _bg(el, s), 4.5, prefer=(ink,))
        if h * clamp(v / vmax) > vs * 2.2:
            ctx.save(); ctx.rectangle(bx, base - L, bw, L); ctx.clip()
            draw_text(ctx, vtxt, bx + bw / 2, base - L + vs * 0.85, size=vs, color=tin, alpha=ta, weight=vf)
            ctx.restore()
        else:
            draw_text(ctx, vtxt, bx + bw / 2, base - L - vs * 0.7, size=vs, color=tout, alpha=ta, weight=vf)
        if it.get("label"):
            draw_text(ctx, it["label"], bx + bw / 2, base + lsz * 1.1, size=min(lsz, bw * 0.3), color=tout, alpha=ta,
                      weight=lf)
    return (right - w, y - h / 2, right, base + lsz * 1.8)


# ---------------------------------------------------------------------------
# VERSUS TABLE
# ---------------------------------------------------------------------------
@element("versus_table", dur=0.7, enter="none", sfx="none")
def versus_table(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 936))
    rows = el.get("rows") or []
    size = float(el.get("size", 52))
    hsz = float(el.get("title_size", size * 0.82))
    hh = float(el.get("head_h", hsz * 2.1))
    rh = float(el.get("row_h", size * 2.15))
    right = el.get("_right", x + w / 2)
    left = right - w
    mid = right - w / 2
    colw = w / 2
    pad = float(el.get("pad", 26))
    ink = _pal_col(el, s, "color", "ink", "#111111")
    acc = _pal_col(el, s, "accent", "accent", "#e3262e")
    paper = _pal_col(el, s, "paper", "paper", "#efece6")
    mute = _pal_col(el, s, "mute", "mute", "#8c8a86")
    bg = _bg(el, s)
    a_all = s.alpha * (1 - s.out_p)
    t0 = float(el["t"])
    e = get_ease("in_out_expo")
    eo = get_ease("out_expo")
    top = y
    hf, bf = F(el, "title", key="title_font"), F(el, "body")
    # heavy top rule
    p_top = e(prog(s.t, t0, 0.5))
    W_top = float(el.get("rule_weight", 12))
    ctx.set_source_rgba(*_rgba(ink, a_all))
    ctx.rectangle(right - w * p_top, top - W_top / 2, w * p_top, W_top); ctx.fill()
    # header: right title plain, left title on an accent slab (the "reality" column)
    hp = eo(prog(s.t, t0 + 0.2, 0.6))
    hy = top + hh / 2
    if hp > 0:
        slab_p = e(prog(s.t, t0 + 0.25, 0.5))
        ctx.set_source_rgba(*_rgba(acc, a_all))
        ctx.rectangle(mid - colw * slab_p, top + W_top / 2 + 6, colw * slab_p - 6, hh - W_top / 2 - 12); ctx.fill()
        rt = el.get("right_title", "")
        lt = el.get("left_title", "")
        fit_r = min(hsz, hsz * (colw - 2 * pad) / max(1, text_bbox(rt, 0, 0, hsz, hf)[2] - text_bbox(rt, 0, 0, hsz, hf)[0]))
        fit_l = min(hsz, hsz * (colw - 2 * pad) / max(1, text_bbox(lt, 0, 0, hsz, hf)[2] - text_bbox(lt, 0, 0, hsz, hf)[0]))
        ctx.save(); ctx.rectangle(*mask_rect(rt, right - pad, hy, fit_r, hf, align="right")); ctx.clip()
        draw_text(ctx, rt, right - pad, hy + (1 - hp) * hh * 0.6, size=fit_r, color=_ct.ensure(ink, bg, 4.5),
                  align="right", alpha=a_all, weight=hf)
        ctx.restore()
        if slab_p > 0.82:
            ctx.save(); ctx.rectangle(mid - colw * slab_p, top + W_top / 2 + 6, max(1.0, colw * slab_p - 6), hh - W_top / 2 - 12)
            ctx.clip()
            draw_text(ctx, lt, mid - pad, hy + (1 - hp) * hh * 0.6, size=fit_l,
                      color=_ct.ensure(paper, acc, 4.5, prefer=(paper, "#ffffff", "#111111")), align="right",
                      alpha=a_all * clamp((slab_p - 0.82) / 0.18), weight=hf)
            ctx.restore()
    # middle rule under the header
    p_mid = e(prog(s.t, t0 + 0.35, 0.5))
    ym = top + hh
    ctx.set_source_rgba(*_rgba(ink, a_all))
    ctx.rectangle(right - w * p_mid, ym - 3, w * p_mid, 6); ctx.fill()
    # rows
    last_t = t0 + 0.6
    n_shown = 0
    for i, r in enumerate(rows):
        ti = float(r["t"]) if r.get("t") is not None else t0 + 0.6 + i * 0.5
        last_t = max(last_t, ti)
        p = eo(prog(s.t, ti, 0.55))
        if p <= 0:
            continue
        n_shown = i + 1
        ry = ym + i * rh + rh / 2
        for side, txt, xr, delay in (("right", r.get("right", ""), right - pad, 0.0), ("left", r.get("left", ""), mid - pad, 0.14)):
            q = eo(prog(s.t, ti + delay, 0.55))
            if q <= 0 or not txt:
                continue
            fnt = bf if side == "right" else hf
            sz = min(size, size * (colw - 2 * pad) / max(1, text_bbox(txt, 0, 0, size, fnt)[2] - text_bbox(txt, 0, 0, size, fnt)[0]))
            col = _ct.ensure(ink, bg, 4.5) if side == "right" else _ct.ensure(acc, bg, 3.0)
            cx0 = mid if side == "right" else left
            m = mask_rect(txt, xr, ry, sz, fnt, align="right")
            ctx.save(); ctx.rectangle(cx0, m[1], colw, m[3]); ctx.clip()           # masked slide-in from the right
            draw_text(ctx, txt, xr + (1 - q) * colw * 0.55, ry, size=sz, color=col, align="right",
                      alpha=a_all * clamp(q * 1.6), weight=fnt)
            ctx.restore()
        if r.get("strike") and r.get("strike_t") is not None:     # optional: strike the claim
            sp = e(prog(s.t, float(r["strike_t"]), 0.35))
            if sp > 0:
                ctx.set_source_rgba(*_rgba(acc, a_all)); ctx.rectangle(right - pad - (colw - 2 * pad) * sp, ry - 3, (colw - 2 * pad) * sp, 6); ctx.fill()
        # row rule
        rp = e(prog(s.t, ti + 0.1, 0.45))
        is_last = i == len(rows) - 1
        wt = W_top if is_last else 2.5
        ctx.set_source_rgba(*_rgba(ink if is_last else mute, a_all))
        ctx.rectangle(right - w * rp, ym + (i + 1) * rh - wt / 2, w * rp, wt); ctx.fill()
    # centre divider grows with the shown rows
    if p_mid > 0:
        dv = top + hh * p_mid + max(0, n_shown) * rh
        ctx.set_source_rgba(*_rgba(ink, a_all))
        ctx.rectangle(mid - 2, top + W_top / 2, 4, max(0, dv - top - W_top / 2)); ctx.fill()
    return (left, top - W_top, right, ym + len(rows) * rh + W_top)


# ---------------------------------------------------------------------------
# WAVEFORM (audio-reactive)
# ---------------------------------------------------------------------------
@element("waveform", dur=0.6, enter="none", sfx="none")
def waveform(ctx, el, s):
    x, y = xy(el, s.t)
    w = float(el.get("w", 760))
    h = float(el.get("h", 160))
    n = int(el.get("bars", 31))
    mode = el.get("mode", "bars")
    sc = s.scene
    env = (lambda tt: sc.vo_env(tt)) if sc is not None else (lambda tt: 0.0)
    if el.get("color") is not None:
        col = _color(el["color"])
    else:
        col = _pal_col(el, s, "accent", "accent", "#ffffff")
    lag = float(el.get("lag", 0.22))
    grow = ease_out(s.raw_p) * (1 - s.out_p)
    a = s.alpha * (1 - s.out_p)
    floor_ = float(el.get("floor", 0.06))
    if mode == "line":
        pts = 120
        ctx.set_line_width(float(el.get("width", 4)))
        ctx.set_source_rgba(*_rgba(col, a))
        for i in range(pts + 1):
            f = i / pts
            d = abs(f - 0.5) * 2
            amp = (floor_ + (1 - floor_) * env(s.t - d * lag)) * h / 2 * grow * (1 - d ** 2)
            yy = y + amp * math.sin(f * math.pi * float(el.get("cycles", 9)) + s.t * 14)
            xx = x + w / 2 - f * w
            (ctx.move_to if i == 0 else ctx.line_to)(xx, yy)
        ctx.stroke()
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    bw = w / n * float(el.get("fill", 0.55))
    for i in range(n):
        f = i / max(1, n - 1)
        d = abs(f - 0.5) * 2                       # 0 at the centre (= now), 1 at the edges (= older)
        e_ = env(s.t - d * lag)
        jitter = 0.75 + 0.25 * math.sin(i * 2.399 + s.t * 9.0)
        bh = max(bw, (floor_ + (1 - floor_) * e_ * jitter) * h * (1 - 0.55 * d ** 1.5)) * grow
        bx = x + w / 2 - (i + 0.5) * (w / n) - bw / 2
        ctx.set_source_rgba(*_rgba(col, a * (0.55 + 0.45 * (1 - d))))
        r = bw / 2
        if mode == "bottom":                     # bars grow up from a baseline
            y0, y1 = y + h / 2 - bh, y + h / 2
        else:                                    # "bars" / "mirror": symmetric around y
            y0, y1 = y - bh / 2, y + bh / 2
        ctx.new_sub_path()
        ctx.arc(bx + r, y0 + r, r, math.pi, 2 * math.pi)
        ctx.arc(bx + r, y1 - r, r, 0, math.pi)
        ctx.close_path(); ctx.fill()
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def vo_ring(ctx, x, y, size, env, color, alpha):
    """Inner ring that breathes with the voice (money_core / ai_monogram with idle:'vo_pulse')."""
    if env <= 0.02:
        return
    c = _color(color)
    for k, (rk, ak) in enumerate(((0.30, 0.9), (0.42, 0.45))):
        ctx.set_source_rgba(c[0], c[1], c[2], c[3] * alpha * ak * env)
        ctx.set_line_width(2.2 if k == 0 else 1.4)
        ctx.arc(x, y, size * (rk + 0.10 * env), 0, 2 * math.pi)
        ctx.stroke()
