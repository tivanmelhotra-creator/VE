"""
mograph.aecomp.bridge — use aecomp comps (shape elements, lower thirds, imported/own JSON) inside BOTH renderers.

Spec item (same keys for a v5–v13 scene element and a v14 footage layer):
    {"type": "aecomp", "gen": "burst", "seed": 2,                # generator (python3 -m mograph.aecomp list)
     "t": 3.2, "x": 540, "y": 900, "size": 420,                  # start time, centre, box size (px, longest side)
     "c1": "#fff", "c2": "@accent", "stroke": 10, "speed": 1.0, "loop": false, "rot": 0, "opacity": 1}
    {"type": "aecomp", "lower_third": "line_reveal", "title": "دکتر سارا احمدی", "role": "ارتودنتیست",
     "t": 1.0, "out": 6.0, "accent": "@accent"}                  # full-frame overlay (x/y/size ignored)
    {"type": "aecomp", "title_style": "minimal_rule", "hero": "مقدمه را آخر بنویس", "kicker": "نکته", "t": 2.0}
    {"type": "aecomp", "comp": "path/to/comp.json" | {...}}      # any comp JSON
v14: "t"/"out" accept word anchors ("w:مقاله") like every other layer; "@accent"-style colours use the palette.
"""
from __future__ import annotations

import json
import math
import os

import cairo

from . import generators as GEN
from . import lowerthirds as LT
from . import titles as TT
from .render import render

_CACHE = {}


def _resolve_color(c, pal):
    if isinstance(c, str) and c.startswith("@") and pal:
        return pal.get(c[1:], "#ffffff")
    return c


def build(L, pal=None):
    """spec item → (comp, kind) where kind is 'element' (boxed, centred at x/y) or 'overlay' (full frame)."""
    key = id(L)
    if key in _CACHE and _CACHE[key][0] is L:
        return _CACHE[key][1], _CACHE[key][2]
    ctl = {k: _resolve_color(L[k], pal) for k in ("c1", "c2", "c3") if k in L}
    if "stroke" in L:
        ctl["stroke"] = float(L["stroke"])
    if L.get("title_style"):
        kw = {k: L[k] for k in ("y", "font", "font2") if k in L}
        if "text_size" in L:
            kw["size"] = L["text_size"]
        for k in ("accent", "ink"):
            if k in L:
                kw[k] = _resolve_color(L[k], pal)
        kw["dur"] = float(L.get("dur", 4.0))
        if isinstance(L.get("_t1"), (int, float)) and isinstance(L.get("_t0"), (int, float)):
            kw["dur"] = max(1.5, float(L["_t1"]) - float(L["_t0"]))
        if "w" in L:
            kw["w"], kw["h"] = L["w"], L["h"]
        comp, kind = TT.make(L["title_style"], L.get("hero", L.get("text", "")), L.get("kicker", ""), **kw), "overlay"
        _CACHE[key] = (L, comp, kind)
        return comp, kind
    if L.get("lower_third"):
        kw = {k: L[k] for k in ("x", "y", "size", "font", "font2") if k in L and k != "size"}
        if "text_size" in L:
            kw["size"] = L["text_size"]
        if "accent" in L:
            kw["accent"] = _resolve_color(L["accent"], pal)
        if "ink" in L:
            kw["ink"] = _resolve_color(L["ink"], pal)
        dur = float(L.get("dur", 5.0))
        if isinstance(L.get("_t1"), (int, float)) and isinstance(L.get("_t0"), (int, float)):
            dur = max(2.0, float(L["_t1"]) - float(L["_t0"]))
        kw["dur"] = dur
        if "w" in L:
            kw["w"], kw["h"] = L["w"], L["h"]
        comp, kind = LT.make(L["lower_third"], L.get("title", ""), L.get("role", ""), **kw), "overlay"
    elif L.get("comp") is not None:
        c = L["comp"]
        comp = json.load(open(c)) if isinstance(c, str) else c
        comp = dict(comp)
        comp["controls"] = dict(comp.get("controls", {}), **ctl)
        kind = L.get("mode", "element")
    else:
        extra = {k: v for k, v in (L.get("params") or {}).items()}
        comp, kind = GEN.make(L.get("gen", "burst"), seed=int(L.get("seed", 0)), **ctl, **extra), "element"
    _CACHE[key] = (L, comp, kind)
    return comp, kind


def local_time(L, t, t0, comp):
    lt = (t - t0) * float(L.get("speed", 1.0))
    dur = float(comp.get("dur", 1.0))
    if L.get("loop"):
        lt = lt % dur
    return lt


def draw(ctx, L, t, t0, W, H, pal=None, x=None, y=None, scale=1.0):
    """draw the comp onto a cairo context at time t (t0 = item start).  Returns bbox or None."""
    comp, kind = build(L, pal)
    lt = local_time(L, t, t0, comp)
    if lt < 0 or (not L.get("loop") and lt > float(comp.get("dur", 1.0)) + 1e-3 and kind == "element"):
        return None
    op = float(L.get("opacity", 1.0))
    if kind == "overlay":
        surf = render(comp, lt, W, H)
        ctx.save()
        ctx.identity_matrix()
        ctx.set_source_surface(surf, 0, 0)
        ctx.paint_with_alpha(op)
        ctx.restore()
        return (0, 0, W, H)
    size = float(L.get("size", 400)) * scale
    cw, ch = float(comp["w"]), float(comp["h"])
    k = size / max(cw, ch)
    w, h = max(2, int(math.ceil(cw * k))), max(2, int(math.ceil(ch * k)))
    surf = render(comp, lt, w, h)
    x = float(L.get("x", W / 2)) if x is None else x
    y = float(L.get("y", H / 2)) if y is None else y
    ctx.save()
    ctx.translate(x, y)
    if L.get("rot"):
        ctx.rotate(math.radians(float(L["rot"])))
    ctx.set_source_surface(surf, -w / 2.0, -h / 2.0)
    ctx.paint_with_alpha(op)
    ctx.restore()
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


# ------------------------------------------------------------------------------------- v5–v13 element
def _register_v5():
    from ..elements import element

    @element("aecomp", dur=0.0, enter="none", exit="fade")
    def aecomp_element(ctx, el, s):
        tgt = ctx.get_target()
        W = tgt.get_width() if hasattr(tgt, "get_width") else 1080
        H = tgt.get_height() if hasattr(tgt, "get_height") else 1920
        sc = getattr(s, "scene", None)
        if el.get("lower_third") or el.get("title_style"):
            el.setdefault("w", getattr(sc, "W", W) if sc is not None else W)
            el.setdefault("h", getattr(sc, "H", H) if sc is not None else H)
        ctx.push_group()
        bb = draw(ctx, el, s.t, float(el.get("t", 0.0)), W, H)
        pat = ctx.pop_group()
        ctx.set_source(pat)
        ctx.paint_with_alpha(max(0.0, min(1.0, s.alpha)))
        return bb


# ------------------------------------------------------------------------------------- v14 layer
def draw_v14(ctx, L, F):
    from ..v14 import layers as LY
    if L.get("lower_third") or L.get("title_style"):
        L.setdefault("w", F.W)
        L.setdefault("h", F.H)
        return draw(ctx, L, F.t, L["_t0"], F.W, F.H, L.get("_pal"))
    x, y, k = LY.pos(L, F)
    return draw(ctx, L, F.t, L["_t0"], F.W, F.H, L.get("_pal"), x, y, k)


def register_v14(DRAW):
    DRAW["aecomp"] = draw_v14


try:
    _register_v5()
except Exception:  # noqa: BLE001  (elements module unavailable in a stripped install)
    pass
