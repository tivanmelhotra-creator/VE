"""
mograph.mocap.strip — make an ANIMATION-ONLY GLB (skeleton nodes + animation channels; meshes, skins' bind
matrices, images, materials dropped). A Mixamo / Quaternius pack shrinks ~3-6x and can live in the repo.

    python3 -m mograph.mocap.strip in.glb out.glb [--clips Walk_Loop,Idle_Loop] [--fps 30]

--fps resamples every channel to a uniform rate (LINEAR keys are kept exactly when they are already sparse).
"""
from __future__ import annotations

import json
import struct
import sys

import numpy as np

from .gltf import _accessor, _read


def strip(src, dst, clips=None, drop=("tpose", "a_tpose")):
    js, bufs = _read(src)
    out = bytearray()
    acc, views = [], []

    def add(arr, typ):
        arr = np.ascontiguousarray(arr, np.float32)
        while len(out) % 4:
            out.append(0)
        off = len(out)
        out.extend(arr.tobytes())
        views.append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
        a = {"bufferView": len(views) - 1, "componentType": 5126, "count": int(arr.shape[0]), "type": typ}
        if typ == "SCALAR":
            a["min"] = [float(arr.min())]
            a["max"] = [float(arr.max())]
        acc.append(a)
        return len(acc) - 1

    nodes = []
    for n in js["nodes"]:
        m = {k: v for k, v in n.items() if k in ("name", "children", "translation", "rotation", "scale", "matrix")}
        nodes.append(m)
    joints = sorted({j for s in js.get("skins", []) for j in s["joints"]})
    anims = []
    for a in js.get("animations", []):
        name = a.get("name", "")
        if clips and name not in clips:
            continue
        if name.lower().replace("-", "_") in drop:
            continue
        cache = {}
        samplers, chans = [], []
        for ch in a["channels"]:
            tg = ch["target"]
            if tg.get("path") not in ("translation", "rotation", "scale") or "node" not in tg:
                continue
            if joints and tg["node"] not in joints:
                continue
            s = a["samplers"][ch["sampler"]]
            v = _accessor(js, bufs, s["output"])
            if tg["path"] == "scale" and np.allclose(v, 1, atol=1e-4):
                continue                                  # constant unit scale: useless
            if s["input"] not in cache:
                cache[s["input"]] = add(_accessor(js, bufs, s["input"]), "SCALAR")
            o = add(v, "VEC4" if tg["path"] == "rotation" else "VEC3")
            samplers.append({"input": cache[s["input"]], "output": o, "interpolation": s.get("interpolation", "LINEAR")})
            chans.append({"sampler": len(samplers) - 1, "target": {"node": tg["node"], "path": tg["path"]}})
        anims.append({"name": name, "samplers": samplers, "channels": chans})
    scene_roots = js["scenes"][js.get("scene", 0)]["nodes"] if js.get("scenes") else []
    J = {"asset": {"version": "2.0", "generator": "mograph.mocap.strip"}, "scene": 0,
         "scenes": [{"nodes": scene_roots}], "nodes": nodes, "animations": anims,
         "accessors": acc, "bufferViews": views, "buffers": [{"byteLength": len(out)}]}
    if joints:
        J["skins"] = [{"joints": joints}]
    jb = json.dumps(J, separators=(",", ":")).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)
    while len(out) % 4:
        out.append(0)
    total = 12 + 8 + len(jb) + 8 + len(out)
    with open(dst, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        f.write(struct.pack("<II", len(out), 0x004E4942) + bytes(out))
    return dst


if __name__ == "__main__":
    a = sys.argv[1:]
    clips = None
    if "--clips" in a:
        i = a.index("--clips")
        clips = set(a[i + 1].split(","))
        del a[i:i + 2]
    strip(a[0], a[1], clips)
    print("ok", a[1])


def save_glb(sk, dst, fps=30.0, clips=None, name_map=None):
    """Any Skeleton (BVH / glTF) → compact animation-only GLB, every clip resampled to `fps` (rotations of all
    joints, translation of the root-most animated joint only)."""
    from .gltf import quat_to_mat  # noqa
    out = bytearray()
    acc, views = [], []

    def add(arr, typ):
        arr = np.ascontiguousarray(arr, np.float32)
        while len(out) % 4:
            out.append(0)
        off = len(out)
        out.extend(arr.tobytes())
        views.append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
        a = {"bufferView": len(views) - 1, "componentType": 5126, "count": int(arr.shape[0]), "type": typ}
        if typ == "SCALAR":
            a["min"], a["max"] = [float(arr.min())], [float(arr.max())]
        acc.append(a)
        return len(acc) - 1

    N = len(sk.names)
    nodes = []
    kids = [[] for _ in range(N)]
    for i in range(N):
        if sk.parent[i] >= 0:
            kids[sk.parent[i]].append(i)
    for i in range(N):
        n = {"name": (name_map or {}).get(sk.names[i], sk.names[i]), "translation": [float(v) for v in sk.rest_t[i]],
             "rotation": [float(v) for v in sk.rest_r[i]]}
        if kids[i]:
            n["children"] = kids[i]
        nodes.append(n)
    roots = [i for i in range(N) if sk.parent[i] < 0]
    anims = []
    for cname, c in sk.clips.items():
        if clips and cname not in clips:
            continue
        n = max(2, int(round(c.duration * fps)) + 1)
        ts = np.linspace(0, c.duration, n)
        ti = add(ts, "SCALAR")
        samplers, chans = [], []
        for ch in c.channels:
            v = np.array([ch.sample(t) for t in ts])
            if ch.path == "translation" and sk.parent[ch.node] >= 0 and np.ptp(v, 0).max() < 1e-4:
                continue
            o = add(v, "VEC4" if ch.path == "rotation" else "VEC3")
            samplers.append({"input": ti, "output": o, "interpolation": "LINEAR"})
            chans.append({"sampler": len(samplers) - 1, "target": {"node": ch.node, "path": ch.path}})
        anims.append({"name": cname, "samplers": samplers, "channels": chans})
    J = {"asset": {"version": "2.0", "generator": "mograph.mocap.save_glb"}, "scene": 0,
         "scenes": [{"nodes": roots}], "nodes": nodes, "animations": anims, "skins": [{"joints": list(sk.joints)}],
         "accessors": acc, "bufferViews": views, "buffers": [{"byteLength": len(out)}]}
    jb = json.dumps(J, separators=(",", ":")).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)
    while len(out) % 4:
        out.append(0)
    with open(dst, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(out)))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        f.write(struct.pack("<II", len(out), 0x004E4942) + bytes(out))
    return dst
