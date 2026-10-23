"""
mograph.human.assets — fetch + compile the MakeHuman HM08 assets (CC0) into a compact cache OUTSIDE git.

Source: github.com/makehumancommunity/mpfb2 (pinned commit), folders src/mpfb/data/{3dobjs,rigs,targets,mesh_metadata}.
All of these assets are CC0 (LICENSE.ASSETS.md of mpfb2, weights.default.json "license": "CC0").

    python3 -m mograph.human fetch          # ≈ 40 s: sparse git clone (≈ 75 MB) → compile → ~/.cache/mograph_models/human
    python3 -m mograph.human fetch --force

Cache layout (≈ 33 MB):
    hm08_base.npz     verts (19158,3) f32 in DECIMETRES, Y-up (OBJ space) · faces (N,4) · face_group · group names
                      · weights (bone × vertex) · bone table (name, parent, head/tail vertex refs)
    hm08_targets.npz  one pair per target: "<name>__i" int32 vertex ids, "<name>__d" f32 deltas (OBJ space)
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

CACHE = os.path.join(os.environ.get("MOGRAPH_MODELS") or os.path.expanduser("~/.cache/mograph_models"), "human")
REPO = "https://github.com/makehumancommunity/mpfb2.git"
COMMIT = "b1a2dc98776da9d1eea751b159d42cf61778fe2e"
BASE = os.path.join(CACHE, "hm08_base.npz")
TARGETS = os.path.join(CACHE, "hm08_targets.npz")

LOCAL_DIRS = ["stomach", "torso", "hip", "buttocks", "neck", "arms", "legs", "head", "breast", "feet", "hands",
              "pelvis", "chin", "cheek"]


def ready() -> bool:
    return os.path.exists(BASE) and os.path.exists(TARGETS)


def _read_target(path):
    op = gzip.open if path.endswith(".gz") else open
    ids, d = [], []
    with op(path, "rt") as f:
        for ln in f:
            if not ln or ln[0] == "#":
                continue
            p = ln.split()
            if len(p) < 4:
                continue
            ids.append(int(p[0]))
            d.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(ids, np.int32), np.asarray(d, np.float32).reshape(-1, 3)


def compile_from(data_dir: str, out_dir: str = CACHE):
    os.makedirs(out_dir, exist_ok=True)
    V, F, FG, groups = [], [], [], []
    gi = -1
    with open(os.path.join(data_dir, "3dobjs", "base.obj")) as f:
        for ln in f:
            if ln.startswith("v "):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("g "):
                name = ln.split()[1]
                if name not in groups:
                    groups.append(name)
                gi = groups.index(name)
            elif ln.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
                if len(idx) == 3:
                    idx.append(idx[2])
                F.append(idx[:4])
                FG.append(gi)
    V = np.asarray(V, np.float32)
    F = np.asarray(F, np.int32)
    FG = np.asarray(FG, np.int16)
    vg = json.load(open(os.path.join(data_dir, "mesh_metadata", "basemesh_vertex_groups.json")))
    vgroups = {k: np.concatenate([np.arange(a, b + 1) for a, b in v]).astype(np.int32) for k, v in vg.items()}
    rig = json.load(open(os.path.join(data_dir, "rigs", "standard", "rig.default.json")))
    W = json.load(open(os.path.join(data_dir, "rigs", "standard", "weights.default.json")))["weights"]
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        p = rig[n].get("parent") or ""
        if p:
            visit(p)
        seen.add(n)
        order.append(n)
    for n in rig:
        visit(n)
    names = order
    bidx = {n: i for i, n in enumerate(names)}
    parent = np.asarray([bidx.get(rig[n].get("parent") or "", -1) for n in names], np.int32)

    def ref(spec):
        s = spec["strategy"]
        if s == "CUBE":
            return vgroups[spec["cube_name"]].tolist()
        if s == "VERTEX":
            return [int(spec["vertex_index"])]
        return [int(i) for i in spec["vertex_indices"]]
    rows, cols, vals = [], [], []
    for n, lst in W.items():
        if n not in bidx:
            continue
        for vi, w in lst:
            rows.append(bidx[n]); cols.append(vi); vals.append(w)
    meta = {"bones": names, "heads": [ref(rig[n]["head"]) for n in names],
            "tails": [ref(rig[n]["tail"]) for n in names], "groups": groups,
            "vgroups": {k: v.tolist() for k, v in vgroups.items() if not k.startswith("joint-")},
            "joint_cubes": {k: v.tolist() for k, v in vgroups.items() if k.startswith("joint-")},
            "rolls": [float(rig[n].get("roll", 0.0)) for n in names],
            "source": f"{REPO}@{COMMIT}", "license": "CC0"}
    np.savez_compressed(os.path.join(out_dir, "hm08_base.npz"), verts=V, faces=F, face_group=FG, parent=parent,
                        w_rows=np.asarray(rows, np.int16), w_cols=np.asarray(cols, np.int32),
                        w_vals=np.asarray(vals, np.float32), meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    tdir = os.path.join(data_dir, "targets")
    arrays = {}

    def add(key, path):
        i, d = _read_target(path)
        arrays[key + "__i"] = i
        arrays[key + "__d"] = d
    md = os.path.join(tdir, "macrodetails")
    for fn in sorted(os.listdir(md)):
        if fn.endswith(".target.gz"):
            add("macro/" + fn[:-10], os.path.join(md, fn))
    for sub in ("height", "proportions"):
        for fn in sorted(os.listdir(os.path.join(md, sub))):
            if fn.endswith(".target.gz"):
                add(f"{sub}/" + fn[:-10], os.path.join(md, sub, fn))
    for d in LOCAL_DIRS:
        p = os.path.join(tdir, d)
        if not os.path.isdir(p):
            continue
        for fn in sorted(os.listdir(p)):
            if fn.endswith(".target.gz") and not fn.startswith(("female-", "male-")):
                add(f"{d}/" + fn[:-10], os.path.join(p, fn))
    np.savez_compressed(os.path.join(out_dir, "hm08_targets.npz"), **arrays)
    with open(os.path.join(out_dir, "LICENSE.txt"), "w") as f:
        f.write("MakeHuman HM08 base mesh, rig, weights and targets — CC0 1.0 (public domain).\n"
                f"Source: {REPO} @ {COMMIT}\n")
    return len(arrays) // 2


def fetch(force: bool = False) -> str:
    if ready() and not force:
        return CACHE
    tmp = tempfile.mkdtemp(prefix="mh_")
    try:
        git = ["git", "-c", "advice.detachedHead=false"]
        subprocess.check_call(git + ["clone", "--filter=blob:none", "--sparse", "--no-checkout", REPO, tmp],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(git + ["-C", tmp, "sparse-checkout", "set", "src/mpfb/data/3dobjs",
                                     "src/mpfb/data/rigs/standard", "src/mpfb/data/targets",
                                     "src/mpfb/data/mesh_metadata"], stdout=subprocess.DEVNULL)
        subprocess.check_call(git + ["-C", tmp, "checkout", COMMIT], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        n = compile_from(os.path.join(tmp, "src", "mpfb", "data"))
        print(f"[human] compiled {n} targets → {CACHE}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return CACHE


def setup_env():
    """One-shot environment repair after a sandbox reset: python deps, EGL, assets."""
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "moderngl", "pycairo"])
    try:
        import moderngl
        moderngl.create_standalone_context(backend="egl").release()
    except Exception:
        subprocess.call("sudo apt-get install -y -q libegl1 libgl1-mesa-dri >/dev/null 2>&1", shell=True)
    fetch()


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        print(compile_from(sys.argv[1]))
    elif "--setup" in sys.argv:
        setup_env()
    else:
        fetch(force="--force" in sys.argv)
