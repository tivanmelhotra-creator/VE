"""
tools/import_motion_presets.py — DEV-ONLY: Premiere "Motion Presets" .prproj templates → our compact JSON library
(sdk/mograph/aefx/presets/motion_presets.json).  Reads ONLY keyframe numbers (reference data); no vendor code/media.

    python3 tools/import_motion_presets.py "<pack>/templates/Motion Presets"

Each preset = ordered stack of adjustment-layer effects (Premiere track order, bottom → top), time normalised to 0:
    {"name", "cat", "dir": in|out|fx, "dur", "fx": [{"fx": transform|basic3d|fast_blur|wave_warp|twirl, "p": {...}}]}
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mograph.aefx.keyframes import parse_prproj_keys   # noqa: E402

FX = {
    "AE.ADBE Geometry2": ("transform", ["anchor", "position", "uniform", "scale", "scale_w", "skew", "skew_axis",
                                         "rotation", "opacity", "_comp_shutter", "shutter", "_sampling"]),
    "AE.ADBE Basic 3D": ("basic3d", ["swivel", "tilt", "distance", "_specular", "_preview"]),
    "AE.ADBE Fast Blur": ("fast_blur", ["blur", "dims", "_repeat"]),
    "AE.ADBE Wave Warp": ("wave_warp", ["type", "height", "width", "direction", "speed", "pinning", "phase", "_aa"]),
    "AE.ADBE Twirl": ("twirl", ["angle", "radius", "center"]),
}
DEFAULTS = {"transform": {"anchor": [0.5, 0.5], "position": [0.5, 0.5], "uniform": 1.0, "scale": 100.0,
                          "scale_w": 100.0, "skew": 0.0, "skew_axis": 0.0, "rotation": 0.0, "opacity": 100.0,
                          "shutter": 0.0},
            "basic3d": {"swivel": 0.0, "tilt": 0.0, "distance": 0.0},
            "fast_blur": {"blur": 0.0, "dims": 0.0},
            "wave_warp": {"type": 1.0, "height": 10.0, "width": 40.0, "direction": 90.0, "speed": 1.0,
                          "pinning": 0.0, "phase": 0.0},
            "twirl": {"angle": 0.0, "radius": 30.0, "center": [0.5, 0.5]}}


def index_objects(s):
    """top-level objects (tab-indented once; closing tag '\\n\\t</Tag>') by ObjectID / ObjectUID."""
    idx = {}
    for m in re.finditer(r'\n\t<(\w+) Object(?:ID|UID)="([^"]+)"[^>]*>', s):
        end = s.find("\n\t</%s>" % m.group(1), m.end())
        idx[m.group(2)] = (m.group(1), s[m.end():end])
    return idx


def _val(txt, point):
    v = txt.split(",")[1]
    if point:
        return [round(float(c), 5) for c in v.split(":")]
    if v in ("true", "false"):
        return 1.0 if v == "true" else 0.0
    return round(float(v), 5)


def parse_prproj(path):
    s = gzip.open(path).read().decode("utf-8", "ignore")
    O = index_objects(s)
    stack = []
    for _, (tag, g) in [(k, v) for k, v in O.items() if v[0] == "VideoTrackGroup"]:
        for tu in re.findall(r'<Track Index="\d+" ObjectURef="([^"]+)"/>', g):
            for ti in re.findall(r'<TrackItem Index="\d+" ObjectRef="(\d+)"/>', O.get(tu, ("", ""))[1]):
                n, ib = O.get(ti, (None, ""))
                if n != "VideoClipTrackItem":
                    continue
                st = int(re.search(r"<Start>(-?\d+)</Start>", ib).group(1))
                comp = re.search(r'<Components ObjectRef="(\d+)"/>', ib).group(1)
                sub = re.search(r'<SubClip ObjectRef="(\d+)"/>', ib).group(1)
                cr = re.search(r'<Clip ObjectRef="(\d+)"/>', O.get(sub, ("", ""))[1])
                inpoint = 0
                if cr:
                    ip = re.search(r"<InPoint>(-?\d+)</InPoint>", O.get(cr.group(1), ("", ""))[1])
                    inpoint = int(ip.group(1)) if ip else 0
                for cref in re.findall(r'ObjectRef="(\d+)"', O.get(comp, ("", ""))[1]):
                    fb = O.get(cref, ("", ""))[1]
                    mn = re.search(r"<MatchName>(.*?)</MatchName>", fb)
                    if not mn or mn.group(1) not in FX:
                        continue
                    kind, names = FX[mn.group(1)]
                    params = {}
                    for idx, pref in re.findall(r'<Param Index="(\d+)" ObjectRef="(\d+)"/>', fb):
                        i = int(idx)
                        if i >= len(names) or names[i].startswith("_"):
                            continue
                        pn, pb = O.get(pref, ("", ""))
                        point = pn == "PointComponentParam"
                        kf = re.search(r"<Keyframes>(.*?)</Keyframes>", pb)
                        if kf:
                            params[names[i]] = parse_prproj_keys(kf.group(1), inpoint - st, point)
                        else:
                            sk = re.search(r"<StartKeyframe>(.*?)</StartKeyframe>", pb)
                            if sk:
                                v = _val(sk.group(1), point)
                                if v != DEFAULTS[kind].get(names[i]):
                                    params[names[i]] = v
                    if params:
                        stack.append({"fx": kind, "p": params})
    return stack


def normalise(fx):
    t0 = min((v["k"][0][0] for f in fx for v in f["p"].values() if isinstance(v, dict)), default=0.0)
    m = 0.0
    for f in fx:
        for v in f["p"].values():
            if isinstance(v, dict):
                for r in v["k"]:
                    r[0] = round(r[0] - t0, 5)
                m = max(m, v["k"][-1][0])
    return fx, round(max(m, 0.04), 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mograph", "aefx",
                                               "presets", "motion_presets.json"))
    a = ap.parse_args(argv)
    out = {}
    for f in sorted(glob.glob(os.path.join(a.root, "**", "*.prproj"), recursive=True)):
        if "Auto-Save" in f:
            continue
        base = os.path.splitext(os.path.basename(f))[0]
        m = re.match(r"(.*)_(IN|OUT)$", base)
        name, d = (m.group(1), m.group(2).lower()) if m else (base, "fx")
        fx = parse_prproj(f)
        if not fx:
            continue
        fx, dur = normalise(fx)
        key = re.sub(r"[^a-z0-9]+", "_", (f"{name}_{d}" if d != "fx" else name).lower()).strip("_")
        out[key] = {"name": name.replace("_", " "), "cat": os.path.relpath(f, a.root).split(os.sep)[0], "dir": d,
                    "dur": dur, "fx": fx}
    os.makedirs(os.path.dirname(os.path.abspath(a.o)), exist_ok=True)
    with open(a.o, "w") as fh:
        json.dump(out, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"{len(out)} presets → {a.o} ({os.path.getsize(a.o) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
