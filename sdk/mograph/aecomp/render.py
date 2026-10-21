"""
mograph.aecomp.render — renders an AE-style COMPOSITION (our own JSON, see AECOMP_GUIDE.md) with cairo.

Comp JSON
    {"w": 1080, "h": 1080, "dur": 1.5, "fps": 30, "bg": null,
     "controls": {"c1": "#ffffff", "stroke": 12, "size": 100},      # AE "FX Control" sliders / colours
     "layers": [LAYER, ...]}                                         # first = top (AE order)
LAYER
    {"name", "type": "shape"|"text"|"null", "in": 0, "out": 1.5, "parent": "<name>",
     "matte": {"layer": "<name>", "mode": "alpha"|"alpha_inv"|"luma"|"luma_inv"}, "blend": "normal"|"add"|"screen"|"multiply",
     "tr": {"anchor", "pos", "scale", "rot", "opacity"},             # AE layer transform (values may be animated)
     "shapes": [ITEM, ...]                                           # shape layer contents (AE order: top first)
     "text": {...}  (text layer, see _draw_text)}
ITEM  ("ty")
    gr  group {"it": [...], "tr": {anchor,pos,scale,rot,skew,skew_axis,opacity}}
    el  ellipse {size,pos,rev}   rc rect {size,pos,round}   sr polystar {star|polygon, points, outer, inner,
        outer_round, inner_round, rot, pos}   sh path {"path": PATH or keyed paths}
    fl  fill {color, opacity, rule: "nz"|"eo"}                  st stroke {color, opacity, width, cap, join, miter, dash}
    gf/gs gradient fill/stroke {type: "linear"|"radial", start, end, stops: [[pos, "#hex"], ...]} (+ stroke keys)
    tm  trim {start, end, offset, mode}    zz zigzag {size, ridges, smooth}   rd round corners {radius}
    pb  pucker/bloat {amount}   op offset {amount}   tw twist {angle, center}   mm merge {mode: add|sub|int|xor}
    rp  repeater {copies, offset, order: "above"|"below", tr: {anchor,pos,scale,rot,start_op,end_op}}
Semantics follow AE: modifiers act on all paths ABOVE them in the same group; fills/strokes paint the paths above
them; a repeater duplicates everything above it.

Layer effects (layer key "fx"): {"ty": "radial_wipe", "completion", "start", "center", "dir": cw|ccw|both, "feather"}
    {"ty": "linear_wipe", "completion", "angle", "feather"}   {"ty": "turbulent_displace", "amount", "size",
    "complexity", "evolution", "seed"} (approximation — AE's noise is proprietary).
"""
from __future__ import annotations

import math

import cairo
import numpy as np

from . import geom as Gm
from .values import col, ev, num, vec

CAP = {"butt": cairo.LINE_CAP_BUTT, "round": cairo.LINE_CAP_ROUND, "square": cairo.LINE_CAP_SQUARE}
JOIN = {"miter": cairo.LINE_JOIN_MITER, "round": cairo.LINE_JOIN_ROUND, "bevel": cairo.LINE_JOIN_BEVEL}
OPS = {"normal": cairo.OPERATOR_OVER, "add": cairo.OPERATOR_ADD, "screen": cairo.OPERATOR_SCREEN,
       "multiply": cairo.OPERATOR_MULTIPLY, "overlay": cairo.OPERATOR_OVERLAY, "lighten": cairo.OPERATOR_LIGHTEN,
       "darken": cairo.OPERATOR_DARKEN}


# ------------------------------------------------------------------------------------- shape tree
def _tr_matrix(tr, t, C):
    tr = tr or {}
    return Gm.affine(vec(tr.get("anchor", 0), t, C), vec(tr.get("pos", 0), t, C),
                     vec(tr.get("scale", 100), t, C, default=100.0), num(tr.get("rot", 0), t, C),
                     num(tr.get("skew", 0), t, C), num(tr.get("skew_axis", 0), t, C))


def _path_value(it, t, C):
    pv = ev(it["path"], t, C)
    return Gm.Path(pv["v"], np.array(pv["v"], float) + np.array(pv.get("i", np.zeros((len(pv["v"]), 2))), float),
                   np.array(pv["v"], float) + np.array(pv.get("o", np.zeros((len(pv["v"]), 2))), float),
                   pv.get("c", True))


def _geom_item(it, t, C):
    ty = it["ty"]
    rev = bool(it.get("rev", False))
    if ty == "el":
        return Gm.ellipse(vec(it.get("size", 100), t, C), vec(it.get("pos", 0), t, C), rev)
    if ty == "rc":
        return Gm.rect(vec(it.get("size", 100), t, C), vec(it.get("pos", 0), t, C), num(it.get("round", 0), t, C), rev)
    if ty == "sr":
        return Gm.polystar(it.get("kind", "star"), num(it.get("points", 5), t, C), num(it.get("outer", 100), t, C),
                           num(it.get("inner", 50), t, C), num(it.get("outer_round", 0), t, C),
                           num(it.get("inner_round", 0), t, C), num(it.get("rot", 0), t, C),
                           vec(it.get("pos", 0), t, C), rev)
    if ty == "sh":
        p = _path_value(it, t, C)
        return p.reversed() if rev else p
    return None


def _emit_path(cr, p):
    if not len(p.V):
        return
    cr.move_to(*p.V[0])
    for s in p.segments():
        cr.curve_to(s[1][0], s[1][1], s[2][0], s[2][1], s[3][0], s[3][1])
    if p.closed:
        cr.close_path()


def _source(cr, it, t, C, alpha):
    ty = it["ty"]
    op = num(it.get("opacity", 100), t, C) / 100.0 * alpha
    if ty in ("gf", "gs"):
        s, e = vec(it.get("start", [0, 0]), t, C), vec(it.get("end", [100, 0]), t, C)
        if it.get("type", "linear") == "radial":
            g = cairo.RadialGradient(s[0], s[1], 0, s[0], s[1], math.hypot(e[0] - s[0], e[1] - s[1]) or 1)
        else:
            g = cairo.LinearGradient(s[0], s[1], e[0], e[1])
        for pos, c in it.get("stops", [[0, "#ffffff"], [1, "#000000"]]):
            r = col(c, t, C)
            g.add_color_stop_rgba(float(pos), r[0], r[1], r[2], r[3] * op)
        cr.set_source(g)
    else:
        r = col(it.get("color", "#ffffff"), t, C)
        cr.set_source_rgba(r[0], r[1], r[2], r[3] * op)
    return op


def _paint(cr, paths, it, t, C, alpha, M):
    """fill / stroke the given (already transformed to layer space) paths."""
    if not paths:
        return
    ty = it["ty"]
    if ty in ("fl", "gf") and any(p.op in ("sub", "int", "xor") for p in paths):
        # AE Merge Paths on fills: composite each path with its boolean operator inside a group
        cr.push_group()
        for p in paths:
            cr.new_path()
            _emit_path(cr, p)
            cr.set_operator({"sub": cairo.OPERATOR_DEST_OUT, "int": cairo.OPERATOR_DEST_IN,
                             "xor": cairo.OPERATOR_XOR}.get(p.op, cairo.OPERATOR_OVER))
            cr.set_source_rgba(1, 1, 1, 1)
            if p.op == "sub":
                # AE subtracts exact geometry; raster DEST_OUT leaves an a*(1-a) anti-alias seam where the
                # edges coincide (rings that close to nothing stay visible) → also erase a ~1 px edge band
                cr.fill_preserve()
                ux, uy = cr.device_to_user_distance(1.0, 0.0)
                cr.set_line_width(max(1e-6, math.hypot(ux, uy)))
                cr.stroke()
            else:
                cr.fill()
        mask = cr.pop_group()
        cr.set_operator(cairo.OPERATOR_OVER)
        _source(cr, it, t, C, alpha)
        cr.mask(mask)
        return
    cr.new_path()
    for p in paths:
        _emit_path(cr, p)
    if _source(cr, it, t, C, alpha) <= 0.002:
        return
    if ty in ("fl", "gf"):
        cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD if it.get("rule") == "eo" else cairo.FILL_RULE_WINDING)
        cr.fill()
        return
    w = num(it.get("width", 2), t, C)
    sc = math.sqrt(abs(np.linalg.det(M[:2, :2]))) or 1.0
    if w * sc <= 0.05:
        cr.new_path()
        return
    cr.set_line_width(w * sc)
    cr.set_line_cap(CAP.get(it.get("cap", "round"), cairo.LINE_CAP_ROUND))
    cr.set_line_join(JOIN.get(it.get("join", "round"), cairo.LINE_JOIN_ROUND))
    cr.set_miter_limit(float(it.get("miter", 4)))
    dash = it.get("dash")
    if dash:
        d = [max(0.01, num(x, t, C) * sc) for x in dash.get("d", [])]
        cr.set_dash(d, num(dash.get("offset", 0), t, C) * sc)
    else:
        cr.set_dash([])
    cr.stroke()


def _cm(M):
    return (M[0, 0], M[1, 0], M[0, 1], M[1, 1], M[0, 2], M[1, 2])


def _apply_mod(it, paths, t, C):
    ty = it["ty"]
    if ty == "tm":
        mode = 2 if it.get("mode", 1) in (2, "individual") else 1
        return Gm.trim(paths, num(it.get("start", 0), t, C), num(it.get("end", 100), t, C),
                       num(it.get("offset", 0), t, C), mode)
    if ty == "zz":
        return [Gm.zigzag(p, num(it.get("size", 10), t, C), num(it.get("ridges", 5), t, C),
                          it.get("smooth", False) in (True, 2, "smooth")) for p in paths]
    if ty == "rd":
        return [Gm.round_corners(p, num(it.get("radius", 10), t, C)) for p in paths]
    if ty == "pb":
        return [Gm.pucker_bloat(p, num(it.get("amount", 0), t, C)) for p in paths]
    if ty == "op":
        return [Gm.offset(p, num(it.get("amount", 0), t, C)) for p in paths]
    if ty == "tw":
        return Gm.twist(paths, num(it.get("angle", 0), t, C), vec(it.get("center", 0), t, C))
    if ty == "mm":                                     # AE Merge Paths: first path = base, the rest get the op
        m = it.get("mode", "add")
        try:                                           # exact boolean outline (strokes/trim below follow it)
            return Gm.boolean(paths, m)
        except Exception:  # noqa: BLE001  (no shapely / degenerate) → raster boolean in _paint
            pass
        out = [p.copy() for p in paths]
        for i, p in enumerate(out):
            p.op = "add" if i == 0 else m
        return out
    return paths


class _Ctx:
    __slots__ = ("cr", "t", "C", "W", "H", "base")


MODS = ("tm", "zz", "rd", "pb", "op", "tw", "mm")
PAINTS = ("fl", "st", "gf", "gs")


def _apply_mods(mods, paths, t, C):
    for m in mods:
        if not paths:
            break
        paths = _apply_mod(m, paths, t, C)
    return paths


def _mods_before_merge(mods_at, i, outer):
    """operators below index i up to (not incl.) the next Merge Paths; `outer` only if no merge intervenes."""
    res = []
    for j, m in mods_at:
        if j <= i:
            continue
        if m.get("ty") == "mm":
            return res
        res.append(m)
    return res + [m for m in outer if m.get("ty") != "mm"]


def _collect(X, items, M, alpha, outer):
    """AE shape-group evaluation.  Returns (final_paths_in_this_space, draw_fn).
    - geometry at index i is modified by every operator BELOW it in the same list (top→bottom), then by `outer`
      (operators of enclosing groups that sit below this group);
    - a paint at index p paints the FINAL geometry of everything above it;
    - stacking: items higher in the list are drawn on top;
    - a repeater copies everything above it."""
    t, C = X.t, X.C
    items = [it for it in items if not it.get("hide")]
    r_idx = next((i for i, it in enumerate(items) if it.get("ty") == "rp"), None)
    entries = []                    # (src_index, paths, premodified, draw_fn | None)
    start = 0
    if r_idx is not None:
        rp = items[r_idx]
        later_mods = [m for m in items[r_idx + 1:] if m.get("ty") in MODS] + outer
        copies = int(round(num(rp.get("copies", 3), t, C)))
        offs = num(rp.get("offset", 0), t, C)
        rtr = rp.get("tr", {})
        so, eo = num(rtr.get("start_op", 100), t, C) / 100.0, num(rtr.get("end_op", 100), t, C) / 100.0
        order = list(range(copies))[::-1] if rp.get("order", "below") == "below" else list(range(copies))
        draws, allp = [], []
        for k in order:
            Mk = _repeat_matrix(rtr, k + offs, t, C)
            a_k = so + (eo - so) * (k / (copies - 1) if copies > 1 else 0.0)
            # operators below the repeater (incl. enclosing ones) act on every copy — also on what the copies'
            # own paints draw (AE: Trim/Zig-Zag after a Repeater modify all copies)
            ps, dr = _collect(X, items[:r_idx], M @ Mk, alpha * a_k, later_mods)
            draws.append(dr)
            allp += [p.copy().transform(Mk) for p in ps]
        entries.append((r_idx, allp, True,
                        lambda _d=draws: [d() for d in _d]))
        start = r_idx + 1
    mods_at = [(i, it) for i, it in enumerate(items) if it.get("ty") in MODS and i >= start]
    for i in range(start, len(items)):
        it = items[i]
        ty = it.get("ty")
        if ty in ("el", "rc", "sr", "sh"):
            g = _geom_item(it, t, C)
            if g is not None:
                entries.append((i, [g], False, None))
        elif ty == "gr":
            Mg = _tr_matrix(it.get("tr"), t, C)
            ga = alpha * num((it.get("tr") or {}).get("opacity", 100), t, C) / 100.0
            below = _mods_before_merge(mods_at, i, outer)
            # parent operators act in the parent space; we apply them inside the child (good for scale-free ops)
            ps, dr = _collect(X, it.get("it", []), M @ Mg, ga, below)
            entries.append((i, [p.copy().transform(Mg) for p in ps], True, dr if ga > 0.002 else None))
    # final geometry per entry
    final = []
    for src, ps, pre, dr in entries:
        if not pre:
            ps = _apply_mods(_mods_before_merge(mods_at, src, outer), ps, t, C)
        final.append((src, ps, dr))
    for j, m in mods_at:                 # Merge Paths acts on the COMBINED geometry above it (incl. sub-groups)
        if m.get("ty") != "mm":
            continue
        above = [(k, e) for k, e in enumerate(final) if e[0] < j]
        flat = [p for _, e in above for p in e[1]]
        merged = _apply_mod(m, flat, t, C)
        # operators between this merge and the next one (then the outer ones) act on the MERGED outline
        merged = _apply_mods(_mods_before_merge(mods_at, j, outer), merged, t, C)
        # merged geometry replaces everything above; keep the stacking slot of the last contributor
        for k, e in above:
            final[k] = (e[0], [], e[2])
        if above:
            k_last = above[-1][0]
            final[k_last] = (final[k_last][0], merged, final[k_last][2])
    paints = [(i, it) for i, it in enumerate(items) if it.get("ty") in PAINTS and i >= start]
    mm_idx = max([i for i, it in enumerate(items) if it.get("ty") == "mm"], default=-1)
    if mm_idx >= 0:                    # AE: a Merge Paths disables every fill/stroke above it (incl. nested groups)
        paints = [(i, it) for i, it in paints if i > mm_idx]
        final = [(src, ps, (dr if src > mm_idx else None)) for src, ps, dr in final]

    def draw():
        # bottom → top: walk indices descending; draw nested groups/repeaters and paints in stacking order
        ev_list = [(src, "g", dr) for src, ps, dr in final if dr is not None] + [(i, "p", it) for i, it in paints]
        ev_list.sort(key=lambda e: -e[0])
        for idx, kind, obj in ev_list:
            if kind == "g":
                obj()
                continue
            geo = [p for src, ps, _ in final if src < idx for p in ps]
            if not geo:
                continue
            X.cr.save()
            if obj["ty"] in ("st", "gs"):
                X.cr.identity_matrix()
                _paint(X.cr, [p.copy().transform(M) for p in geo], obj, t, C, alpha, M)
            else:
                X.cr.identity_matrix()
                _paint(X.cr, [p.copy().transform(M) for p in geo], obj, t, C, alpha, M)
            X.cr.restore()
    out_paths = [p for _, ps, _ in final for p in ps]
    return out_paths, draw


def _render_items(X, items, M, alpha):
    _, dr = _collect(X, items, M, alpha, [])
    dr()


def _repeat_matrix(rtr, k, t, C):
    """repeater copy k: anchor/position/scale/rotation compounded k times (AE)."""
    a = vec(rtr.get("anchor", 0), t, C)
    p = vec(rtr.get("pos", [100, 0]), t, C)
    s = vec(rtr.get("scale", 100), t, C, default=100.0)
    r = num(rtr.get("rot", 0), t, C)
    sx, sy = (s[0] / 100.0) ** k if s[0] > 0 else 0.0, (s[1] / 100.0) ** k if s[1] > 0 else 0.0
    T1 = np.array([[1, 0, -a[0]], [0, 1, -a[1]], [0, 0, 1]], float)
    S = np.diag([sx, sy, 1.0])
    rr = math.radians(r * k)
    R = np.array([[math.cos(rr), -math.sin(rr), 0], [math.sin(rr), math.cos(rr), 0], [0, 0, 1]])
    T2 = np.array([[1, 0, a[0] + p[0] * k], [0, 1, a[1] + p[1] * k], [0, 0, 1]], float)
    return T2 @ R @ S @ T1


# ------------------------------------------------------------------------------------- layers
def _layer_matrix(comp, L, t, C, cache):
    key = L.get("name")
    if key in cache:
        return cache[key]
    M = _tr_matrix(L.get("tr"), t, C)
    par = L.get("parent")
    if par:
        P = next((q for q in comp["layers"] if q.get("name") == par), None)
        if P is not None and P is not L:
            M = _layer_matrix(comp, P, t, C, cache) @ M
    cache[key] = M
    return M


def _reveal(rv, t, size):
    """time-based reveal of a whole text block → (dx, dy, sy, alpha).  rv = {"t_in", "t_out", "in": {...}, "out": {...},
    "dur_in": 0.55, "dur_out": 0.45}; anim keys: dx / dy (fraction of the font size), fade, scale_y.
    Easing = AE Easy Ease with a strong 75 % influence (the measured lower-third curve)."""
    from .values import ev as _ev
    ti, to = float(rv.get("t_in", 0.0)), float(rv.get("t_out", 1e9))
    di, do = float(rv.get("dur_in", 0.55)), float(rv.get("dur_out", 0.45))
    curve = {"k": [[0.0, 1.0, "b", 0, 0.75, 0, 0.75], [1.0, 0.0, "b", 0, 0.75, 0, 0.75]]}
    if t < ti or t >= to + do:
        return 0.0, 0.0, 1.0, 0.0
    if t < ti + di:
        u, an = float(_ev(curve, (t - ti) / di)), rv.get("in", {})
    elif t < to:
        return 0.0, 0.0, 1.0, 1.0
    else:
        u, an = 1.0 - float(_ev(curve, min(1.0, (t - to) / do))), rv.get("out", {})
    dx = float(an.get("dx", 0.0)) * size * 1.3 * u
    dy = float(an.get("dy", 0.0)) * size * 1.3 * u
    sy = (1.0 - u) if an.get("scale_y") else 1.0
    a = (1.0 - u) if (an.get("fade") or an.get("scale_y")) else 1.0
    return dx, dy, sy, a


def _draw_text(cr, L, t, C, M, alpha):
    """point text via the v14 glyph engine: Persian shaping, theme stretch, AE text animators (mograph.aefx),
    plus a whole-block `reveal` (slide from behind a track matte / scale open).  align right = RTL anchor at x."""
    from ..v14 import glyphs as G
    tx = L["text"]
    s = ev(tx.get("text", ""), t, C)
    if not s:
        return
    size = num(tx.get("size", 80), t, C)
    font = tx.get("font", "Doran-Bold.ttf")
    stretch = float(tx.get("stretch", G.STRETCH.get(font.split("-")[0].lower(), 1.0)))
    styles = {"thin": {"font": font, "color": _hex(col(tx.get("color", "#ffffff"), t, C)),
                       "scale": 1.0, "glow": 0.0, "stretch": stretch}}
    b = G.typeset(s, size=size, styles=styles, align="center", tracking=float(tx.get("tracking", 0)) / 1000.0)
    x0, y0, x1, y1 = (float(v) for v in b.bbox)
    al = tx.get("align", "center")
    ox = -x1 if al == "right" else (-x0 if al == "left" else -(x0 + x1) / 2)
    oy = 0.0
    a = alpha
    sy = 1.0
    if tx.get("reveal"):
        dx, dy, sy, ra = _reveal(tx["reveal"], t, size)
        ox += dx
        oy += dy
        a *= ra
        if a <= 0.002 or sy <= 0.002:
            return
    motion = None
    if tx.get("animators") or tx.get("preset"):
        from ..aefx import textanim as TA
        if "_anim" not in L:
            L["_anim"] = TA.Animator(tx.get("animators") or tx["preset"], dur=tx.get("preset_dur"),
                                     px=float(tx.get("preset_px", 100.0)))
        st = L["_anim"].frame(b, t - float(tx.get("anim_t", L.get("in", 0.0))))
        motion = lambda g, _st=st: _st.apply(g, G.GS())  # noqa: E731
    cr.save()
    cr.set_matrix(cairo.Matrix(*_cm(M)))
    if sy != 1.0:
        cr.scale(1.0, sy)
    G.draw_block(cr, b, ox, oy, motion=motion, alpha=a, shadow=None)
    cr.restore()


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(max(0, min(1, v)) * 255) for v in c[:3])


def _render_layer(X, comp, L, t, cache):
    W, H = X.W, X.H
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    cr = cairo.Context(surf)
    cr.set_antialias(cairo.ANTIALIAS_BEST)
    X2 = _Ctx()
    X2.cr, X2.t, X2.C, X2.W, X2.H = cr, t, X.C, W, H
    M = X.base @ _layer_matrix(comp, L, t, X.C, cache)
    a = num((L.get("tr") or {}).get("opacity", 100), t, X.C) / 100.0
    if a <= 0.002:
        return surf, 0.0
    if L.get("type", "shape") == "shape":
        _render_items(X2, L.get("shapes", []), M, 1.0)
    elif L.get("type") == "text":
        _draw_text(cr, L, t, X.C, M, 1.0)
    elif L.get("type") == "solid":
        cr.set_matrix(cairo.Matrix(*_cm(M)))
        r = col(L.get("color", "#000000"), t, X.C)
        sz = L.get("size", [comp["w"], comp["h"]])
        cr.rectangle(0, 0, sz[0], sz[1])
        cr.set_source_rgba(*r)
        cr.fill()
    surf.flush()
    return surf, a


def _trace(comp, v, t, C):
    """AE 'Trace Path' expression: point / tangent on the first path of layer v['trace'], at that layer's progress."""
    L = next((q for q in comp["layers"] if q.get("name") == v.get("trace")), None)
    if L is None:
        return None
    me = next((q for q in comp["layers"] if q.get("trace_progress") is not None and
               any(isinstance(x, dict) and x.get("trace") == v.get("trace") for x in (q.get("tr") or {}).values())),
              None)
    prog = num(me["trace_progress"], t, C) / 100.0 if me else 0.0
    p = _first_path(L.get("shapes", []), t, C)
    if p is None:
        return None
    M = _layer_matrix(comp, L, t, C, {})
    p = p.transform(M)
    x, y, ang = Gm.point_at(p, prog % 1.0 if me and me.get("trace_loop") else prog)
    return ang if v.get("what") == "angle" else [x, y]


def _first_path(items, t, C, M=None):
    M = np.eye(3) if M is None else M
    for it in items:
        if it.get("ty") in ("el", "rc", "sr", "sh"):
            g = _geom_item(it, t, C)
            if g is not None:
                return g.transform(M)
        if it.get("ty") == "gr":
            r = _first_path(it.get("it", []), t, C, M @ _tr_matrix(it.get("tr"), t, C))
            if r is not None:
                return r
    return None


def _wipe_mask(fx, t, C, W, H, base):
    """AE Radial Wipe / Linear Wipe as a 0..1 keep-mask in output pixels (centre given in comp coordinates).
    radial: completion % of the circle is wiped (removed) clockwise from start_angle (0 = 12 o'clock);
    wipe 'ccw' / 'both' change the direction.  linear: wipe_angle 90 = wipes from the left edge to the right."""
    comp_t = num(fx.get("completion", 0), t, C) / 100.0
    feather = num(fx.get("feather", 0), t, C) * base[0, 0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if fx["ty"] == "radial_wipe":
        cx, cy = vec(fx.get("center", [0, 0]), t, C)
        px, py = base[0, 0] * cx + base[0, 2], base[1, 1] * cy + base[1, 2]
        a0 = math.radians(num(fx.get("start", 0), t, C))
        ang = (np.arctan2(xx - px, -(yy - py)) - a0) % (2 * math.pi)     # clockwise from 12 o'clock
        d = fx.get("dir", "cw")
        if d == "ccw":
            ang = (2 * math.pi - ang) % (2 * math.pi)
        if d == "both":
            ang = np.minimum(ang, 2 * math.pi - ang) * 2
        edge = comp_t * 2 * math.pi
        if feather > 0:
            r = np.maximum(np.hypot(xx - px, yy - py), 1.0)
            return np.clip((ang - edge) * r / feather + 0.5, 0, 1)
        return (ang >= edge).astype(np.float32)
    # linear wipe
    th = math.radians(num(fx.get("angle", 90), t, C))
    dx, dy = math.sin(th), -math.cos(th)                          # wipe travels along +dir
    proj = xx * dx + yy * dy
    lo = min(0 * dx + 0 * dy, W * dx, H * dy, W * dx + H * dy)
    hi = max(0 * dx + 0 * dy, W * dx, H * dy, W * dx + H * dy)
    edge = lo - feather + comp_t * (hi - lo + 2 * feather)
    if feather > 0:
        return np.clip((proj - edge) / feather + 0.5, 0, 1)
    return (proj >= edge).astype(np.float32)


def _turbulent_displace(A, fx, t, C, base):
    """approximation of AE Turbulent Displace (AE's noise is proprietary): smooth value-noise displacement field,
    amount = max displacement in comp px, size = feature size (AE Size × 10 px, displacement 0.15 × Amount — fitted on reference elements), evolution in degrees animates it."""
    import cv2
    H, W = A.shape[:2]
    k = float(base[0, 0])
    amt = num(fx.get("amount", 50), t, C) * k
    size = max(2.0, num(fx.get("size", 100), t, C) * 10.0 * k)
    evo = num(fx.get("evolution", 0), t, C) / 360.0
    seed = int(num(fx.get("seed", 0), t, C))
    oct_n = max(1, int(round(num(fx.get("complexity", 1), t, C))))
    rng = np.random.default_rng(1234 + seed)
    gw, gh = int(W / size) + 3, int(H / size) + 3
    def field(o):
        f = np.zeros((H, W), np.float32)
        amp, sc, tot = 1.0, 1.0, 0.0
        for _ in range(oct_n):
            g0 = rng.standard_normal((int(gh * sc) + 2, int(gw * sc) + 2)).astype(np.float32)
            g1 = rng.standard_normal(g0.shape).astype(np.float32)
            ph = (evo + o) * 2 * math.pi
            g = g0 * math.cos(ph) + g1 * math.sin(ph)
            f += amp * cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
            tot += amp
            amp, sc = amp * 0.5, sc * 2.0
        return f / tot
    _ka = 0.15                                   # fitted on the 4 reference elements using it (F3 0.76→0.79)
    dx, dy = field(0.0) * amt * _ka, field(0.37) * amt * _ka
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = cv2.remap(np.ascontiguousarray(A), xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    A[...] = out


def _apply_layer_fx(s, L, t, C, X):
    for f in L.get("fx", []):
        if f.get("ty") == "turbulent_displace" and not f.get("off"):
            A = _arr(s)
            _turbulent_displace(A, f, t, C, X.base)
            s.mark_dirty()
    fxs = [f for f in L.get("fx", []) if f.get("ty") in ("radial_wipe", "linear_wipe") and not f.get("off")]
    if not fxs:
        return
    A = _arr(s)
    keep = np.ones(A.shape[:2], np.float32)
    for f in fxs:
        keep *= _wipe_mask(f, t, C, A.shape[1], A.shape[0], X.base)
    A[...] = (A.astype(np.float32) * keep[..., None]).astype(np.uint8)
    s.mark_dirty()


def _arr(s):
    return np.ndarray((s.get_height(), s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :s.get_width()]


def render(comp, t, W=None, H=None, controls=None, bg=None, fit="contain"):
    """→ cairo.ImageSurface (premultiplied BGRA) of the comp at time t, scaled to fit W×H (default comp size)."""
    cw, chh = float(comp.get("w", 1080)), float(comp.get("h", 1080))
    W = int(W or cw)
    H = int(H or chh)
    k = min(W / cw, H / chh) if fit == "contain" else max(W / cw, H / chh)
    X = _Ctx()
    C = dict(comp.get("controls", {}))
    C.update(controls or {})
    X.C, X.t, X.W, X.H = C, t, W, H
    C["__trace__"] = lambda v, tt: _trace(comp, v, tt, C)
    X.base = np.array([[k, 0, (W - cw * k) / 2], [0, k, (H - chh * k) / 2], [0, 0, 1]], float)
    out = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    oc = cairo.Context(out)
    bgc = bg if bg is not None else comp.get("bg")
    if bgc:
        r = col(bgc, t, C)
        oc.set_source_rgba(*r)
        oc.paint()
    layers = comp.get("layers", [])
    cache = {}
    mattes = {L.get("matte", {}).get("layer") for L in layers if L.get("matte")}
    for L in reversed(layers):                        # AE: bottom layer first
        if L.get("type") == "null" or L.get("hide") or L.get("name") in mattes and not L.get("visible_matte"):
            continue
        if not (float(L.get("in", 0)) - 1e-6 <= t < float(L.get("out", 1e9))):
            continue
        s, a = _render_layer(X, comp, L, t, cache)
        if a <= 0.002:
            continue
        _apply_layer_fx(s, L, t, C, X)
        m = L.get("matte")
        if m:
            ML = next((q for q in layers if q.get("name") == m.get("layer")), None)
            if ML is not None:
                ms, ma = _render_layer(X, comp, ML, t, cache) if (float(ML.get("in", 0)) <= t < float(ML.get("out", 1e9))) else (None, 0)
                A = _arr(s)
                if ms is None:
                    mk = np.zeros(A.shape[:2], np.float32)
                else:
                    MA = _arr(ms).astype(np.float32)
                    if m.get("mode", "alpha").startswith("luma"):
                        lum = (0.0722 * MA[..., 0] + 0.7152 * MA[..., 1] + 0.2126 * MA[..., 2])
                        mk = lum / 255.0 * ma
                    else:
                        mk = MA[..., 3] / 255.0 * ma
                if m.get("mode", "alpha").endswith("inv"):
                    mk = 1.0 - mk
                A[...] = (A.astype(np.float32) * mk[..., None]).astype(np.uint8)
                s.mark_dirty()
        oc.set_operator(OPS.get(L.get("blend", "normal"), cairo.OPERATOR_OVER))
        oc.set_source_surface(s, 0, 0)
        oc.paint_with_alpha(a)
    out.flush()
    return out


def render_bgra(comp, t, W=None, H=None, controls=None, bg=None, fit="contain"):
    s = render(comp, t, W, H, controls, bg, fit)
    return np.array(_arr(s))


def render_rgb(comp, t, W=None, H=None, controls=None, bg="#16171c"):
    a = render_bgra(comp, t, W, H, controls, bg)
    return a[..., 2::-1].copy()
