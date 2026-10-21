"""
tools/import_text_presets.py — DEV-ONLY: AtomX "Text Animation Presets" (.atom = JSXBIN) → our JSON animator library
(sdk/mograph/aefx/presets/text_presets.json).  Only the animator NUMBERS are read as reference data.

    /tmp/jsxer/bin/release/jsxer "Text Animation Presets.atom" -o atom.js     # decompile once (github.com/AngeloD2022/jsxer)
    python3 tools/import_text_presets.py atom.js

Output per preset:  {"name", "cat", "sub", "dur", "animators": [{"sel": [...], "props": {...}}]}
  selector  {"type": "range", "start", "end", "offset" (percent or curve), "units": "pct"|"index",
             "based": "chars"|"chars_ns"|"words"|"lines", "mode": "add"|..., "amount" (%), "shape": "square"|
             "ramp_up"|"ramp_down"|"triangle"|"round"|"smooth", "smooth" (%), "ease_hi", "ease_lo", "random", "seed"}
            {"type": "wiggly", "mode", "min", "max", "freq" (wiggles/s), "corr" (%), "tphase", "sphase", "lock", "seed",
             "based"}
  props     position [x,y,z] px · anchor [x,y,z] · scale [x,y,z] % (100 = none, value ADDED as delta from 100) ·
            rotation / rx / ry deg · opacity % (0 = invisible) · fill_opacity · blur [x, y] px · skew deg ·
            skew_axis deg · tracking px (+ track_type) · char_offset · char_replace · line_spacing
  Any value can be a curve {"k": [...]} — AtomX keys are AE Easy-Ease (33 %) unless given.
"""
from __future__ import annotations

import argparse
import json
import os
import re

PROP = {"ADBE Text Position 3D": "position", "ADBE Text Anchor Point 3D": "anchor", "ADBE Text Scale 3D": "scale",
        "ADBE Text Rotation": "rotation", "ADBE Text Rotation X": "rx", "ADBE Text Rotation Y": "ry",
        "ADBE Text Opacity": "opacity", "ADBE Text Fill Opacity": "fill_opacity", "ADBE Text Blur": "blur",
        "ADBE Text Skew": "skew", "ADBE Text Skew Axis": "skew_axis", "ADBE Text Tracking Amount": "tracking",
        "ADBE Text Track Type": "track_type", "ADBE Text Character Offset": "char_offset",
        "ADBE Text Character Replace": "char_replace", "ADBE Text Line Spacing": "line_spacing"}
SEL = {"ADBE Text Percent Start": "start", "ADBE Text Percent End": "end", "ADBE Text Percent Offset": "offset",
       "ADBE Text Range Units": "units", "ADBE Text Range Type2": "based", "ADBE Text Selector Mode": "mode",
       "ADBE Text Selector Max Amount": "amount", "ADBE Text Range Shape": "shape",
       "ADBE Text Selector Smoothness": "smooth", "ADBE Text Levels Max Ease": "ease_hi",
       "ADBE Text Levels Min Ease": "ease_lo", "ADBE Text Randomize Order": "random", "ADBE Text Random Seed": "seed",
       # wiggly
       "ADBE Text Wiggly Max Amount": "max", "ADBE Text Wiggly Min Amount": "min",
       "ADBE Text Temporal Freq": "freq", "ADBE Text Character Correlation": "corr",
       "ADBE Text Temporal Phase": "tphase", "ADBE Text Spatial Phase": "sphase",
       "ADBE Text Wiggly Lock Dim": "lock", "ADBE Text Wiggly Random Seed": "seed"}
ENUM = {"units": {1: "pct", 2: "index"}, "based": {1: "chars", 2: "chars_ns", 3: "words", 4: "lines"},
        "mode": {1: "add", 2: "subtract", 3: "intersect", 4: "min", 5: "max", 6: "difference"},
        "shape": {1: "square", 2: "ramp_up", 3: "ramp_down", 4: "triangle", 5: "round", 6: "smooth"}}


def _num(v):
    v = v.strip()
    if "," in v:
        return [float(c) for c in v.split(",")]
    return float(v)


def _value(raw):
    """'0' | '0,961,0' | keys 'v|t#v|t#' (AtomX keys, Easy-Ease) → static or curve."""
    if "|" not in raw:
        return _num(raw)
    keys = []
    for part in raw.split("#"):
        if not part:
            continue
        v, t = part.split("|")
        keys.append([round(float(t), 5), _num(v), "b", 0.0, 1 / 3, 0.0, 1 / 3])
    keys.sort(key=lambda r: r[0])
    return {"k": keys}


def parse_preset(s):
    dims, _, body = s.partition("~")
    anims = []
    for chunk in body.split("~NEXT_ANIMATOR:")[0:]:
        chunk = chunk.replace("NEXT_ANIMATOR:", "")
        if "@ANIMATOR_PROPERTIES~" not in chunk:
            continue
        sel_part, prop_part = chunk.split("@ANIMATOR_PROPERTIES~")
        sels = []
        for sp in sel_part.split("@ANIMATOR_SELECTOR~")[1:]:
            kind, _, params = sp.partition("@")
            sel = {"type": "wiggly" if "Wiggly" in kind else "range"}
            for item in params.split("*"):
                if ":" not in item:
                    continue
                k, v = item.split(":", 1)
                if k in SEL:
                    key = SEL[k]
                    val = _value(v)
                    if key in ENUM and isinstance(val, float):
                        val = ENUM[key].get(int(val), int(val))
                    sel[key] = val
            sels.append(sel)
        props = {}
        for item in prop_part.split("*"):
            if ":" not in item:
                continue
            k, v = item.split(":", 1)
            if k in PROP:
                props[PROP[k]] = _value(v)
        anims.append({"sel": sels, "props": props})
    dur = 0.0
    for a in anims:
        for d in [*a["props"].values(), *[x for s in a["sel"] for x in s.values()]]:
            if isinstance(d, dict):
                dur = max(dur, d["k"][-1][0])
    return {"is3d": dims == "3D", "dur": round(dur, 4), "animators": anims}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("js", help="decompiled .atom (jsxer output)")
    ap.add_argument("-o", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mograph", "aefx",
                                               "presets", "text_presets.json"))
    a = ap.parse_args(argv)
    s = open(a.js, encoding="utf-8").read()
    P = dict(re.findall(r'pack_object\.text_preset\.(ATP_\d+) = "(.*?)";\n', s))
    i = s.find("pack_object.structure = {") + len("pack_object.structure = ")
    st = s[i:s.find("};\n", i) + 1]
    st = json.loads(re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', st))
    meta = {}
    for g, sub in st.items():
        for sg, v in sub.items():
            for k, m in v["preview"].items():
                meta[k] = (g, sg, m["name"])
    out = {}
    for k in sorted(P, key=lambda x: int(x.split("_")[1])):
        g, sg, nm = meta.get(k, ("Misc", "", k))
        rec = parse_preset(P[k])
        out[k.lower()] = {"name": nm, "cat": g, "sub": sg, **rec}
    with open(a.o, "w") as fh:
        json.dump(out, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"{len(out)} text presets → {a.o} ({os.path.getsize(a.o) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
