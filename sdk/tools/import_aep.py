"""
DEV TOOL (not needed to use the SDK): convert an After Effects .aep composition into mograph.aecomp JSON.

Used to STUDY reference packs (timing, easing, structure) and to VALIDATE our renderer against the vendor's own
preview videos (tools/validate_aecomp.py).  Imported vendor comps are NOT shipped in the repo — the SDK ships our own
generators (mograph/aecomp/generators.py) fitted to the measured numbers.

    pip install py_aep
    python3 tools/import_aep.py file.aep [-c COMP] [-o out.json]

Supported: shape layers (group, rect, ellipse, polystar, path, fill, stroke, gradient fill/stroke, trim, repeater,
zig zag, twist, round corners, pucker & bloat, offset, merge), null/solid/text layers, parenting, track mattes,
blend modes, keyframes (speed/influence, per-dimension, spatial tangents, hold/linear), and the common expressions
(FX-control links, loopOut, time*k, wiggle, Trace-Path pointOnPath/tangentOnPath).  Unknown expressions → value kept.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import py_aep

LIN, BEZ, HOLD = 6612, 6613, 6614
MATTE = {5013: "alpha", 5014: "alpha_inv", 5015: "luma", 5016: "luma_inv"}
SUPPORTED_FX = {"ADBE Trace Path", "ADBE Radial Wipe", "ADBE Linear Wipe", "ADBE Turbulent Displace"}
BLEND = {5212: "normal", 5220: "add", 5222: "screen", 5216: "multiply", 5226: "overlay", 5221: "lighten",
         5215: "darken", 5223: "add"}
CAPS = {1: "butt", 2: "round", 3: "square"}
JOINS = {1: "miter", 2: "round", 3: "bevel"}


def _iv(x):
    return int(getattr(x, "value", x))


def _r(v, n=4):
    if isinstance(v, float):
        return round(v, n)
    if isinstance(v, (list, tuple)):
        return [_r(q, n) for q in v]
    return v


def _pathval(v):
    V = [[_r(a, 3), _r(b, 3)] for a, b in v.vertices]
    return {"v": V, "i": [[_r(a, 3), _r(b, 3)] for a, b in v.in_tangents],
            "o": [[_r(a, 3), _r(b, 3)] for a, b in v.out_tangents], "c": bool(v.closed)}


def _conv_v(v, dims=None, scale=1.0):
    if hasattr(v, "vertices"):
        return _pathval(v)
    if isinstance(v, (list, tuple)):
        v = list(v)
        if dims:
            v = v[:dims]
        return [_r(float(q) * scale) for q in v]
    if isinstance(v, (int, float)):
        return _r(float(v) * scale)
    return v


def _ease(lst, scale=1.0):
    if not lst:
        return 0.0, 1 / 3
    sp = [_r(e.speed * scale) for e in lst]
    inf = [_r(max(0.1, min(100.0, e.influence)) / 100.0) for e in lst]
    if len(lst) == 1:
        return sp[0], inf[0]
    return sp, inf


class Ctx:
    def __init__(self, comp):
        self.comp = comp
        self.controls = {}
        self.ctl_names = {}


def keyed(q, X, dims=None, scale=1.0):
    """property → value (static / keys / control link)."""
    expr = None
    try:
        if q.expression_enabled and q.expression:
            expr = q.expression
    except Exception:  # noqa: BLE001
        pass
    base = None
    ks = q.keyframes if hasattr(q, "keyframes") else []
    if ks:
        rows = []
        for k in ks:
            it = _iv(k.out_interpolation_type)
            mode = "h" if it == HOLD else ("l" if it == LIN else "b")
            si, ii = _ease(k.in_temporal_ease, scale)
            so, io = _ease(k.out_temporal_ease, scale)
            row = [_r(k.time, 5), _conv_v(k.value, dims, scale), mode, si, ii, so, io]
            try:
                to, ti = k.out_spatial_tangent, k.in_spatial_tangent
                if to is not None and any(abs(x) > 1e-3 for x in to + ti):
                    row += [_r(list(to)[:2]), _r(list(ti)[:2])]
            except Exception:  # noqa: BLE001
                pass
            rows.append(row)
        # per-row: spatial tangent of key i belongs to the segment i→i+1 (to) and i-1→i (ti) — keep as stored
        base = {"k": rows}
    else:
        base = _conv_v(q.value, dims, scale)
    if expr:
        return translate_expr(expr, base, X, dims, scale)
    return base


_CTL = re.compile(r'thisComp\.layer\("([^"]+)"\)\.effect\("([^"]+)"\)\("([^"]+)"\)')


_CREF = re.compile(r'^(?:thisLayer\.)?((?:content\("[^"]+"\)\.?)+)\.?([A-Za-z]+)\s*;?\s*$')


def _content_ref(s, X, dims, scale):
    """resolve  content("Group 1").content("Ellipse 1").content("Polystar Path 1").points  on the same layer."""
    m = _CREF.match(s.replace("\n", "").strip())
    L = getattr(X, "layer", None)
    if not m or L is None:
        return None
    node = L.property("ADBE Root Vectors Group")
    for nm in re.findall(r'content\("([^"]+)"\)', m.group(1)):
        if node is None:
            return None
        if node.match_name == "ADBE Vector Group":
            node = P(node, "ADBE Vectors Group")
        node = next((q for q in (node.properties or []) if q.name == nm), None)
    if node is None:
        return None
    want = re.sub(r"(?<!^)([A-Z])", r" \1", m.group(2)).lower()
    q = next((q for q in (node.properties or []) if q.name.lower() == want), None)
    if q is None:
        return None
    depth = getattr(X, "_ref_depth", 0)
    if depth > 6:
        return None
    X._ref_depth = depth + 1
    try:
        return keyed(q, X, dims, scale)
    finally:
        X._ref_depth = depth


def translate_expr(e, base, X, dims, scale):
    s = e.replace("\r", "\n").strip()
    if s.startswith("content(") or s.startswith("thisLayer.content("):
        r = _content_ref(s, X, dims, scale)
        if r is not None:
            return r
    m = _CTL.search(s)
    if m and ("pointOnPath" not in s and "tangentOnPath" not in s) and \
            ("Trace Path" in m.group(2) or m.group(3) in ("Progress", "Pseudo/ADBE Trace Path-0001")):
        return {"ctl": "@trace:" + m.group(1)}                # progress of another layer's Trace Path effect
    if m and ("pointOnPath" not in s and "tangentOnPath" not in s):
        name = m.group(2)
        key = X.ctl_names.get((m.group(1), name), name)
        tail = s[m.end():].strip().rstrip(";")
        out = {"ctl": key}
        mm = re.match(r"^\s*([-+*/])\s*([0-9.]+)", tail)
        if mm:
            k = float(mm.group(2))
            out.update({"+": {"add": k}, "-": {"add": -k}, "*": {"mul": k}, "/": {"mul": 1 / k}}[mm.group(1)])
        if re.search(r"\[\s*temp\s*,\s*temp\s*\]", s):
            out["vec"] = 2
        if s.startswith("time*") or s.startswith("time *"):
            return {"rate_ctl": key, "v": 0}
        return out
    if "loopOut" in s:
        if "Trace Path-0002" in s:                           # loop only when the Trace Path 'Loop' box is ticked
            return base                                      # → decided in convert() from trace_loop
        if isinstance(base, dict) and "k" in base:
            t = "pingpong" if "pingpong" in s else ("offset" if "offset" in s else "cycle")
            base = dict(base)
            base["loop"] = t
        return base
    mm = re.match(r"^time\s*\*\s*([-0-9.]+)", s)
    if mm:
        return {"rate": float(mm.group(1)), "v": 0}
    mm = re.search(r"wiggle\(\s*([0-9.]+)\s*,\s*([0-9.]+)", s)
    if mm:
        return {"wiggle": [float(mm.group(1)), float(mm.group(2))], "v": base}
    if "pointOnPath" in s or "tangentOnPath" in s:
        lm = re.search(r'thisComp\.layer\("([^"]+)"\)', s)
        return {"trace": lm.group(1) if lm else None, "what": "angle" if "tangentOnPath" in s and "pointOnPath" not in s
                else "pos", "v": base}
    mm = re.match(r"^x\s*=\s*thisComp\.width;\s*y\s*=\s*thisComp\.height;\s*\[x,\s*y\]", s.replace("\n", ""))
    if mm:
        return [X.comp.width, X.comp.height]
    return base


def P(group, mn):
    for q in group.properties or []:
        if q.match_name == mn:
            return q
    return None


def tr_group(g, X, layer=False):
    if g is None:
        return {}
    pre = "ADBE " if layer else "ADBE Vector "
    out = {}
    names = ({"anchor": "ADBE Anchor Point", "pos": "ADBE Position", "scale": "ADBE Scale", "rot": "ADBE Rotate Z",
              "opacity": "ADBE Opacity"} if layer else
             {"anchor": "ADBE Vector Anchor", "pos": "ADBE Vector Position", "scale": "ADBE Vector Scale",
              "rot": "ADBE Vector Rotation", "opacity": "ADBE Vector Group Opacity", "skew": "ADBE Vector Skew",
              "skew_axis": "ADBE Vector Skew Axis"})
    for k, mn in names.items():
        q = P(g, mn)
        if q is None:
            continue
        v = keyed(q, X, 2 if k in ("anchor", "pos", "scale") else None)
        if layer and k == "pos":
            try:
                if P(g, "ADBE Position").dimensions_separated:
                    v = {"sep": [keyed(P(g, "ADBE Position_0"), X), keyed(P(g, "ADBE Position_1"), X)]}
            except Exception:  # noqa: BLE001
                pass
        dflt = {"anchor": [0, 0], "pos": [0, 0], "scale": [100, 100], "rot": 0, "opacity": 100, "skew": 0,
                "skew_axis": 0}[k]
        if v != dflt:
            out[k] = v
    return out


def dashes(g, X):
    if g is None:
        return None
    act = [q for q in g.properties if q.match_name != "ADBE Vector Stroke Offset" and (q.is_modified or q.keyframes)]
    if not act:
        return None
    d = []
    for i in (1, 2, 3):
        dq, gq = P(g, f"ADBE Vector Stroke Dash {i}"), P(g, f"ADBE Vector Stroke Gap {i}")
        if i == 1 or (dq is not None and (dq.is_modified or dq.keyframes)) or (gq is not None and (gq.is_modified or gq.keyframes)):
            d += [keyed(dq, X), keyed(gq, X)]
    return {"d": d, "offset": keyed(P(g, "ADBE Vector Stroke Offset"), X)}


def grad_stops(q, X):
    try:
        v = q.value
    except Exception:  # noqa: BLE001
        v = None
    if not v:
        return [[0, "#ffffff"], [1, "#000000"]]
    try:
        stops = []
        for s in v.color_stops if hasattr(v, "color_stops") else v:
            stops.append([_r(s[0]), "#%02x%02x%02x" % tuple(int(c * 255) for c in s[1:4])])
        return stops
    except Exception:  # noqa: BLE001
        return [[0, "#ffffff"], [1, "#000000"]]


def item(q, X):
    mn = q.match_name
    g = lambda n: P(q, n)  # noqa: E731
    k = lambda n, d=None, s=1.0: keyed(g(n), X, d, s) if g(n) is not None else None  # noqa: E731
    rev = False
    try:
        rev = int(g("ADBE Vector Shape Direction").value) == 3
    except Exception:  # noqa: BLE001
        pass
    if mn == "ADBE Vector Group":
        vg = P(q, "ADBE Vectors Group")
        it = [x for x in (item(c, X) for c in (vg.properties or [])) if x] if vg is not None else []
        out = {"ty": "gr", "nm": q.name, "it": it, "tr": tr_group(P(q, "ADBE Vector Transform Group"), X)}
        if not q.enabled:
            out["hide"] = True
        return out
    if mn == "ADBE Vector Shape - Ellipse":
        return {"ty": "el", "size": k("ADBE Vector Ellipse Size", 2), "pos": k("ADBE Vector Ellipse Position", 2), "rev": rev}
    if mn == "ADBE Vector Shape - Rect":
        return {"ty": "rc", "size": k("ADBE Vector Rect Size", 2), "pos": k("ADBE Vector Rect Position", 2),
                "round": k("ADBE Vector Rect Roundness"), "rev": rev}
    if mn == "ADBE Vector Shape - Star":
        return {"ty": "sr", "kind": "star" if int(g("ADBE Vector Star Type").value) == 1 else "polygon",
                "points": k("ADBE Vector Star Points"), "pos": k("ADBE Vector Star Position", 2),
                "rot": k("ADBE Vector Star Rotation"), "inner": k("ADBE Vector Star Inner Radius"),
                "outer": k("ADBE Vector Star Outer Radius"), "inner_round": k("ADBE Vector Star Inner Roundess"),
                "outer_round": k("ADBE Vector Star Outer Roundess"), "rev": rev}
    if mn == "ADBE Vector Shape - Group":
        return {"ty": "sh", "path": k("ADBE Vector Shape"), "rev": rev}
    if mn in ("ADBE Vector Graphic - Fill", "ADBE Vector Graphic - G-Fill"):
        out = {"ty": "fl" if "G-" not in mn else "gf", "opacity": k("ADBE Vector Fill Opacity"),
               "rule": "eo" if (g("ADBE Vector Fill Rule") and int(g("ADBE Vector Fill Rule").value) == 2) else "nz"}
        if "G-" in mn:
            out.update(_grad(q, X))
        else:
            out["color"] = k("ADBE Vector Fill Color")
        return out if q.enabled else None
    if mn in ("ADBE Vector Graphic - Stroke", "ADBE Vector Graphic - G-Stroke"):
        out = {"ty": "st" if "G-" not in mn else "gs", "opacity": k("ADBE Vector Stroke Opacity"),
               "width": k("ADBE Vector Stroke Width"),
               "cap": CAPS.get(int(g("ADBE Vector Stroke Line Cap").value), "round"),
               "join": JOINS.get(int(g("ADBE Vector Stroke Line Join").value), "round"),
               "miter": _r(float(g("ADBE Vector Stroke Miter Limit").value))}
        dd = dashes(g("ADBE Vector Stroke Dashes"), X)
        if dd:
            out["dash"] = dd
        if "G-" in mn:
            out.update(_grad(q, X))
        else:
            out["color"] = k("ADBE Vector Stroke Color")
        return out if q.enabled else None
    if mn == "ADBE Vector Filter - Trim":
        return {"ty": "tm", "start": k("ADBE Vector Trim Start"), "end": k("ADBE Vector Trim End"),
                "offset": k("ADBE Vector Trim Offset"), "mode": int(g("ADBE Vector Trim Type").value)}
    if mn == "ADBE Vector Filter - Repeater":
        rt = P(q, "ADBE Vector Repeater Transform")
        kk = lambda n, d=None: keyed(P(rt, n), X, d)  # noqa: E731
        return {"ty": "rp", "copies": k("ADBE Vector Repeater Copies"), "offset": k("ADBE Vector Repeater Offset"),
                "order": "below" if int(g("ADBE Vector Repeater Order").value) == 1 else "above",
                "tr": {"anchor": kk("ADBE Vector Repeater Anchor", 2), "pos": kk("ADBE Vector Repeater Position", 2),
                       "scale": kk("ADBE Vector Repeater Scale", 2), "rot": kk("ADBE Vector Repeater Rotation"),
                       "start_op": kk("ADBE Vector Repeater Opacity 1"), "end_op": kk("ADBE Vector Repeater Opacity 2")}}
    if mn == "ADBE Vector Filter - Zigzag":
        return {"ty": "zz", "size": k("ADBE Vector Zigzag Size"), "ridges": k("ADBE Vector Zigzag Detail"),
                "smooth": int(g("ADBE Vector Zigzag Points").value) == 2}
    if mn == "ADBE Vector Filter - Twist":
        return {"ty": "tw", "angle": k("ADBE Vector Twist Angle"), "center": k("ADBE Vector Twist Center", 2)}
    if mn == "ADBE Vector Filter - RC":
        return {"ty": "rd", "radius": k("ADBE Vector RoundCorner Radius")}
    if mn == "ADBE Vector Filter - PB":
        return {"ty": "pb", "amount": k("ADBE Vector PuckerBloat Amount")}
    if mn == "ADBE Vector Filter - Offset":
        return {"ty": "op", "amount": k("ADBE Vector Offset Amount")}
    if mn == "ADBE Vector Filter - Merge":
        return {"ty": "mm", "mode": {1: "add", 2: "add", 3: "sub", 4: "int", 5: "xor"}.get(
            int(g("ADBE Vector Merge Type").value), "add")}
    return None


def _grad(q, X):
    g = lambda n: P(q, n)  # noqa: E731
    return {"type": "radial" if int(g("ADBE Vector Grad Type").value) == 2 else "linear",
            "start": keyed(g("ADBE Vector Grad Start Pt"), X, 2), "end": keyed(g("ADBE Vector Grad End Pt"), X, 2),
            "stops": grad_stops(g("ADBE Vector Grad Colors"), X)}


def controls(L, X):
    try:
        fx = L.effects.properties
    except Exception:  # noqa: BLE001
        return
    for e in fx:
        if e.match_name not in ("ADBE Slider Control", "ADBE Color Control", "ADBE Point Control", "ADBE Angle Control",
                                "ADBE Checkbox Control"):
            continue
        q = e.properties[0]
        try:
            v = q.value
        except Exception:  # noqa: BLE001
            continue
        if e.match_name == "ADBE Color Control":
            v = "#%02x%02x%02x" % tuple(int(max(0, min(1, c)) * 255) for c in v[:3])
        else:
            v = _conv_v(v, 2)
        X.controls[e.name] = v
        X.ctl_names[(L.name, e.name)] = e.name


def convert(comp):
    X = Ctx(comp)
    for L in comp.layers:
        controls(L, X)
    layers = []
    for L in comp.layers:
        X.layer = L
        tn = type(L).__name__
        d = {"name": L.name, "in": _r(L.in_point, 4), "out": _r(L.out_point, 4)}
        if L.parent is not None:
            d["parent"] = L.parent.name
        try:
            mt = _iv(L.track_matte_type)
        except Exception:  # noqa: BLE001
            mt = 5012
        if mt in MATTE:
            ml = None
            try:
                ml = L.track_matte_layer.name if L.track_matte_layer is not None else None
            except Exception:  # noqa: BLE001
                ml = None
            if ml is None and L.index > 1:
                ml = comp.layers[L.index - 2].name            # legacy: matte = layer directly above
            d["matte"] = {"layer": ml, "mode": MATTE[mt]}
        try:
            b = BLEND.get(_iv(L.blending_mode), "normal")
            if b != "normal":
                d["blend"] = b
        except Exception:  # noqa: BLE001
            pass
        d["tr"] = tr_group(L.property("ADBE Transform Group"), X, layer=True)
        if not L.enabled:
            d["hide"] = True
        try:                                              # layer effects we do not render (diagnostics only)
            uns = [e.match_name for e in L.effects.properties
                   if e.enabled and e.match_name not in SUPPORTED_FX and not e.match_name.startswith("Pseudo/")
                   and "Slider" not in e.match_name and "Color Control" not in e.match_name
                   and "Point Control" not in e.match_name and "Checkbox" not in e.match_name
                   and "Angle" not in e.match_name and "Layer Control" not in e.match_name]
            if uns:
                d["unsupported"] = uns
            fx = []
            for e in L.effects.properties:
                if not e.enabled or e.match_name not in ("ADBE Radial Wipe", "ADBE Linear Wipe",
                                                         "ADBE Turbulent Displace"):
                    continue
                pv = {q.match_name.split("-")[-1]: q for q in e.properties}
                if e.match_name == "ADBE Turbulent Displace":
                    fx.append({"ty": "turbulent_displace", "amount": keyed(pv["0002"], X), "size": keyed(pv["0003"], X),
                               "complexity": keyed(pv["0005"], X), "evolution": keyed(pv["0006"], X),
                               "seed": keyed(pv["0011"], X) if "0011" in pv else 0})
                    continue
                if e.match_name == "ADBE Radial Wipe":
                    w = int(round(float(pv["0004"].value))) if "0004" in pv else 1
                    fx.append({"ty": "radial_wipe", "completion": keyed(pv["0001"], X), "start": keyed(pv["0002"], X),
                               "center": keyed(pv["0003"], X, 2), "dir": {1: "cw", 2: "ccw", 3: "both"}.get(w, "cw"),
                               "feather": keyed(pv["0005"], X)})
                else:
                    fx.append({"ty": "linear_wipe", "completion": keyed(pv["0001"], X), "angle": keyed(pv["0002"], X),
                               "feather": keyed(pv["0003"], X)})
            if fx:
                d["fx"] = fx
        except Exception:  # noqa: BLE001
            pass
        if tn == "ShapeLayer":
            d["type"] = "shape"
            root = L.property("ADBE Root Vectors Group")
            d["shapes"] = [x for x in (item(q, X) for q in root.properties) if x]
        elif tn == "TextLayer":
            d["type"] = "text"
            try:
                td = L.text.source_text.value
                d["text"] = {"text": td.text, "font": td.font, "size": _r(td.font_size),
                             "color": "#%02x%02x%02x" % tuple(int(c * 255) for c in td.fill_color[:3]),
                             "tracking": td.tracking}
            except Exception:  # noqa: BLE001
                d["text"] = {"text": L.name}
        else:
            nl = getattr(L, "null_layer", False)
            d["type"] = "null" if nl or L.name in ("FX Control", "Color_Settings") or not L.has_video else "av"
        layers.append(d)
    # trace-path layers: progress lives on the Trace Path pseudo effect of the same layer
    for L, d in zip(comp.layers, layers):
        try:
            for e in L.effects.properties:
                if "Trace Path" in e.match_name or e.name == "Trace Path":
                    d["trace_progress"] = keyed(e.properties[0], X)
                    try:
                        d["trace_loop"] = bool(e.properties[1].value)
                    except Exception:  # noqa: BLE001
                        d["trace_loop"] = False
                    if d["trace_loop"] and isinstance(d["trace_progress"], dict) and "k" in d["trace_progress"]:
                        d["trace_progress"]["loop"] = "cycle"
                    X.controls["@trace:" + L.name] = d["trace_progress"]
        except Exception:  # noqa: BLE001
            pass
    return {"name": comp.name, "w": comp.width, "h": comp.height, "dur": _r(comp.duration, 4),
            "fps": _r(comp.frame_rate, 3), "controls": X.controls, "layers": layers}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("aep")
    ap.add_argument("-c", "--comp")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    app = py_aep.parse(a.aep)
    comps = app.project.compositions
    comp = next((c for c in comps if c.name == a.comp), None) if a.comp else comps[0]
    js = convert(comp)
    s = json.dumps(js, ensure_ascii=False)
    if a.out:
        open(a.out, "w").write(s)
    else:
        sys.stdout.write(s)


if __name__ == "__main__":
    main()
