"""
mograph.pbr3d — v14.6 PHOTOREAL 3D objects: real scanned/PBR models (glTF/GLB/OBJ/FBX/STL) rendered by the
EffectCraft Advanced-3D software rasteriser (Cook-Torrance GGX, HDRI image-based lighting, PCF shadow maps, 2x2 SSAA).

WHY (the v14.5 problem): real3d/sculpt3d build objects from procedural primitives + procedural noise textures →
they look like "3D icons" (clean, plastic, synthetic).  Photoreal needs (a) real geometry detail, (b) real
PBR texture maps (albedo / normal / roughness-metal / AO from photogrammetry) and (c) image-based lighting from a
real HDRI.  pbr3d gets (a)+(b) from free CC0 scans (Poly Haven, 500+ models, `fetch`) or any glTF the agent
generates/imports, and (c) from EffectCraft's renderer — a single static 75 MB CLI binary, no GPU, ~270 MB RAM,
~0.15–0.4 s per 600 px frame on 2 vCPU.  (github.com/storytold/effectcraft, MIT/Apache-2.0)

Design for a low-resource agent sandbox:
  * one CLI call renders a WHOLE pose sequence (per-frame yaw/pitch/roll baked into an expression lookup table),
    so process start-up and scene loading are paid once per object, not per frame;
  * results are cached on disk by a hash of (model, size, hdri, light, poses) → re-renders are free;
  * frames are RGBA sprites (straight alpha) composited by our own engines (v5–v13 elements, v14 layers);
  * a separate shadow-catcher pass gives a REAL contact/cast shadow as a matte (no fake ellipse needed);
  * no new Python dependencies.

Python API
    from mograph import pbr3d
    p = pbr3d.prepare_model("croissant")                    # Poly Haven id | path.gltf/.glb/.obj/.fbx/.stl
    frames = pbr3d.render_sequence(p, size=480, poses=[(yaw, pitch, roll), ...])   # list of RGBA float32
    rgba = pbr3d.render_pose(p, 480, yaw=30, pitch=15)      # one frame (uses the same cache)

CLI
    python3 -m mograph pbr3d-fetch croissant food_apple_01        # download CC0 scans (1k textures)
    python3 -m mograph pbr3d-search food                          # search the Poly Haven catalogue
    python3 -m mograph pbr3d-sheet croissant                      # 8-angle audit sheet → out/pbr3d_<name>.png
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SDK = os.path.dirname(HERE)
ASSETS = os.path.join(HERE, "pbr_assets")
MODELS_DIR = os.environ.get("MOGRAPH_PBR_MODELS", os.path.join(ASSETS, "models"))
HDRI_DIR = os.path.join(ASSETS, "hdri")
CACHE_DIR = os.environ.get("MOGRAPH_PBR_CACHE", os.path.expanduser("~/.cache/mograph_pbr3d"))   # disk, NOT /tmp (tmpfs = RAM)
UA = {"User-Agent": "mograph-sdk/14.6"}
MODEL_EXT = (".gltf", ".glb", ".obj", ".fbx", ".stl")
EC_RELEASE = "https://github.com/storytold/effectcraft/releases/download/v0.11.1/effectcraft-0.11.1-linux-x86_64.tar.gz"


# =========================================================================================== binary
def ec_bin():
    """Path to effectcraft-cli (bundled in sdk/bin, $EFFECTCRAFT_CLI, PATH, or auto-download once)."""
    cands = [os.environ.get("EFFECTCRAFT_CLI"), os.path.join(SDK, "bin", "effectcraft-cli"),
             shutil.which("effectcraft-cli")]
    for c in cands:
        if c and os.path.isfile(c):
            if not os.access(c, os.X_OK):
                os.chmod(c, 0o755)
            return c
    return _download_ec()


def _download_ec():
    dst = os.path.join(SDK, "bin", "effectcraft-cli")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = tempfile.mkdtemp()
    tgz = os.path.join(tmp, "ec.tgz")
    print("[pbr3d] downloading effectcraft-cli (one time, ~64 MB)…", flush=True)
    subprocess.run(["curl", "-sL", "-o", tgz, EC_RELEASE], check=True)
    subprocess.run(["tar", "-xzf", tgz, "-C", tmp, "--wildcards", "*/bin/effectcraft-cli"], check=True)
    for root, _, files in os.walk(tmp):
        if "effectcraft-cli" in files:
            shutil.move(os.path.join(root, "effectcraft-cli"), dst)
    shutil.rmtree(tmp, ignore_errors=True)
    os.chmod(dst, 0o755)
    return dst


def _ec(args, timeout=900):
    r = subprocess.run([ec_bin()] + args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"effectcraft-cli {' '.join(args[:2])} failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout


# =========================================================================================== Poly Haven (CC0)
def _open(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)


def search(query="", category=None, kind="models"):
    """Search the Poly Haven CC0 catalogue → [(id, name, categories)].  kind = models | hdris."""
    url = f"https://api.polyhaven.com/assets?t={kind}" + (f"&c={category}" if category else "")
    d = json.load(_open(url))
    q = query.lower().split()
    out = []
    for k, v in d.items():
        hay = " ".join([k, v.get("name", ""), " ".join(v.get("tags", [])), " ".join(v.get("categories", []))]).lower()
        if all(w in hay for w in q):
            out.append((k, v.get("name", k), v.get("categories", [])))
    return sorted(out)


def fetch(asset, res="1k", kind=None):
    """Download a Poly Haven model (glTF + 1k PBR maps, typically 0.5–3 MB) or HDRI.  Returns the local path."""
    d = json.load(_open(f"https://api.polyhaven.com/files/{asset}"))
    if "gltf" in d and kind != "hdri":
        out = os.path.join(MODELS_DIR, asset)
        g = d["gltf"].get(res) or d["gltf"][sorted(d["gltf"])[0]]
        g = g["gltf"]
        items = [(os.path.basename(g["url"]), g["url"])] + [(k, v["url"]) for k, v in g.get("include", {}).items()]
        main = os.path.join(out, os.path.basename(g["url"]))
    elif "hdri" in d:
        out = HDRI_DIR
        h = d["hdri"].get(res, d["hdri"][sorted(d["hdri"])[0]])["hdr"]
        main = os.path.join(out, asset + ".hdr")
        items = [(asset + ".hdr", h["url"])]
    else:
        raise ValueError(f"{asset}: no glTF/HDRI files on Poly Haven")
    for rel, url in items:
        p = os.path.join(out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(_open(url).read())
    return main


# =========================================================================================== model prep
def resolve_model(src, base_dir=None, auto_fetch=True):
    """Model reference → local file.  Accepts a path (abs / project-relative / models dir) or a Poly Haven id."""
    s = str(src)
    cands = []
    if os.path.isabs(s):
        cands.append(s)
    if base_dir:
        cands.append(os.path.join(base_dir, s))
    cands += [s, os.path.join(MODELS_DIR, s)]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
        if os.path.isdir(c):
            for f in sorted(os.listdir(c)):
                if f.endswith(MODEL_EXT) and ".norm." not in f:
                    return os.path.join(os.path.abspath(c), f)
    if auto_fetch and not os.path.splitext(s)[1]:
        return fetch(s)
    raise FileNotFoundError(f"pbr3d model not found: {src}")


def _gltf_bounds(g, buf_loader=None):
    lo, hi = [], []
    for m in g.get("meshes", []):
        for p in m["primitives"]:
            a = g["accessors"][p["attributes"]["POSITION"]]
            if "min" in a and "max" in a:
                lo.append(a["min"])
                hi.append(a["max"])
    if not lo:
        return None
    return np.min(lo, 0), np.max(hi, 0)


def _node_mats(g):
    """World matrix of every node of the default scene (for bounds of models with node transforms)."""
    def local(n):
        if "matrix" in n:
            return np.array(n["matrix"], np.float64).reshape(4, 4).T
        T = np.eye(4)
        T[:3, 3] = n.get("translation", [0, 0, 0])
        x, y, z, w = n.get("rotation", [0, 0, 0, 1])
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        S = np.diag(n.get("scale", [1, 1, 1]))
        M = np.eye(4)
        M[:3, :3] = R @ S
        return T @ M
    out = {}

    def walk(i, P):
        M = P @ local(g["nodes"][i])
        out[i] = M
        for c in g["nodes"][i].get("children", []):
            walk(c, M)
    sc = g.get("scenes", [{}])[g.get("scene", 0)] if g.get("scenes") else {"nodes": list(range(len(g["nodes"])))}
    for r in sc.get("nodes", []):
        walk(r, np.eye(4))
    return out


def prepare_model(src, base_dir=None):
    """Normalise a model so its bounding sphere is centred on the origin with radius 1 (predictable size + pivot).
    glTF: writes <name>.norm.gltf next to it (wraps the scene in one scale/translate node, textures untouched).
    Other formats are used as-is (size fitted from a probe render).  Returns a dict {path, norm, radius}."""
    path = resolve_model(src, base_dir)
    info = {"src": path, "path": path, "norm": False}
    if path.endswith(".gltf"):
        out = path[:-5] + ".norm.gltf"
        if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
            g = json.load(open(path))
            mats = _node_mats(g)
            pts = []
            for ni, M in mats.items():
                n = g["nodes"][ni]
                if "mesh" not in n:
                    continue
                for p in g["meshes"][n["mesh"]]["primitives"]:
                    a = g["accessors"][p["attributes"]["POSITION"]]
                    lo, hi = np.array(a["min"]), np.array(a["max"])
                    corners = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                        for z in (lo[2], hi[2])])
                    pts.append((corners @ M.T)[:, :3])
            if pts:
                P = np.concatenate(pts)
                lo, hi = P.min(0), P.max(0)
                c = (lo + hi) / 2
                r = float(np.linalg.norm(hi - lo) / 2) or 1.0
                s = 1.0 / r
                sc_i = g.get("scene", 0)
                if not g.get("scenes"):
                    g["scenes"] = [{"nodes": [i for i in range(len(g["nodes"]))]}]
                roots = g["scenes"][sc_i]["nodes"]
                g["nodes"].append({"name": "__mograph_norm", "children": roots, "scale": [s, s, s],
                                   "translation": [float(v) for v in (-c * s)]})
                g["scenes"][sc_i]["nodes"] = [len(g["nodes"]) - 1]
                json.dump(g, open(out, "w"))
        info.update(path=out, norm=True)
    return info


def hdri_path(name=None):
    name = name or "studio"
    if os.path.isfile(str(name)):
        return os.path.abspath(name)
    p = os.path.join(HDRI_DIR, name if name.endswith(".hdr") else name + ".hdr")
    if os.path.isfile(p):
        return p
    return fetch(name, kind="hdri")


# =========================================================================================== scene build + render
DEFAULT_LIGHT = {"key": 60.0, "env": 100.0, "key_pos": (-0.45, -1.0, -0.55), "shadow": 60.0, "diffusion": 14.0}


def _lut_expr(vals, fps, var="A"):
    arr = ",".join(f"{v:.3f}" for v in vals)
    return f"var {var}=[{arr}];{var}[Math.max(0,Math.min({var}.length-1,Math.round(time*{fps})))]"


def _run(P, cmds, save_as=None):
    args = ["run"] + (["--empty"] if save_as else P) + [x for k, p in cmds for x in (k, json.dumps(p))]
    args += (["--save-as", save_as] if save_as else ["--save"]) + ["--json"]
    return json.loads(_ec(args))


def _build_project(pj, model, size, poses, fps, hdri, light, env_rot, floor=0.0, shadow_only=False, hide=False):
    """One EffectCraft project: comp sized to the sprite, model at the centre, Environment light (HDRI: image based
    lighting + reflections) + a Parallel key light that casts shadows; optional shadow-catcher plane `floor` px below
    the centre.  NOTE (tested): a wide Spot cone loses its shadow; a 3D Solid does not catch shadows → plane primitive."""
    D = int(math.ceil(size * 1.30 / 2) * 2)          # sprite canvas: bounding-sphere diameter = size (+30 % margin)
    n = max(1, len(poses))
    c = D / 2
    R = size / 2.0
    L = dict(DEFAULT_LIGHT)
    L.update(light or {})
    kp = np.array(L["key_pos"], np.float64)
    kp = kp / np.linalg.norm(kp) * R * 7
    cmds = [
        ("comp.new", {"name": "S", "width": D, "height": D, "frameRate": fps, "duration": n / fps,
                      "renderer": "advanced3D", "background": "#000000"}),
        ("file.import", {"paths": [hdri]}),                                  # item 2
        ("layer.newModel", {"path": model["path"], "name": "obj"}),
        ("layer.newLight", {"kind": "Environment", "name": "env", "intensity": L["env"]}),
        ("layer.newLight", {"kind": "Parallel", "name": "key", "intensity": L["key"], "castsShadows": True,
                            "shadowDarkness": L["shadow"], "shadowDiffusion": L["diffusion"],
                            "position": [c + kp[0], c + kp[1], kp[2]], "poi": [c, c, 0]}),
        ("layer.addItem", {"item": 2, "index": 99}),
    ]
    if floor:
        cmds += [("layer.newPrimitive", {"kind": "plane", "name": "catch", "width": int(D * 6), "height": int(D * 6),
                                         "baseColor": "#ffffff", "roughness": 90, "metallic": 0})]
    r = _run(None, cmds, save_as=pj)
    env_layer = r["result"][5]["result"]["layer"]
    P = ["--project", pj]
    yaw = [p[0] for p in poses] or [0.0]
    pitch = [p[1] for p in poses] or [0.0]
    roll = [p[2] if len(p) > 2 else 0.0 for p in poses] or [0.0]
    sets = [
        ("layer.environment", {"layers": [env_layer], "on": True}),
        ("material.set", {"layers": ["obj"], "castsShadows": "only" if shadow_only else ("on" if floor else "off")}),
        ("prop.set", {"layer": "obj", "path": "transform/position", "value": [c, c, 0]}),
        ("prop.setExpression", {"layer": "obj", "path": "transform/rotationY", "expression": _lut_expr(yaw, fps)}),
        ("prop.setExpression", {"layer": "obj", "path": "transform/orientation",
                                "expression": "[" + _lut_expr(pitch, fps, "B").replace(";B[", ";(B[") + "),0,0]"}),
        ("prop.setExpression", {"layer": "obj", "path": "transform/rotation", "expression": _lut_expr(roll, fps, "C")}),
    ]
    if env_rot:
        sets.append(("prop.set", {"layer": "env", "path": "lightOptions/rotation", "value": float(env_rot)}))
    if model["norm"]:
        sets.append(("prop.set", {"layer": "obj", "path": "geometryOptions/unitScale", "value": R}))
    if hide:
        sets += [("prop.set", {"layer": "obj", "path": "transform/opacity", "value": 0}),
                 ("material.set", {"layers": ["obj"], "castsShadows": "off"})]
    if floor:
        sets += [("material.set", {"layers": ["catch"], "acceptsShadows": "only", "castsShadows": "off"}),
                 ("prop.set", {"layer": "catch", "path": "transform/position", "value": [c, c + float(floor), 0]}),
                 ("prop.set", {"layer": "catch", "path": "transform/rotationX", "value": 90})]
    _run(P, sets)
    return D


def _key(model, size, poses, hdri, light, env_rot, floor, fps):
    h = hashlib.sha1()
    st = os.stat(model["path"])
    h.update(json.dumps([model["path"], st.st_mtime, st.st_size, size, hdri, light, env_rot, floor, fps,
                         [[round(float(v), 2) for v in p] for p in poses]], sort_keys=True).encode())
    return h.hexdigest()[:20]


def _tmp_root():
    d = os.path.join(CACHE_DIR, "_tmp")
    os.makedirs(d, exist_ok=True)
    return d


def _render_seq(pj, outdir):
    _ec(["render", "--project", pj, "--comp", "S", "--out", os.path.join(outdir, "f.png"),
         "--format", "png", "--channels", "rgba"])


def render_sequence(model, size, poses, hdri=None, light=None, env_rot=0.0, floor="auto", fps=30, shadow=True,
                    chunk=60, verbose=False):
    """Render poses [(yaw, pitch[, roll]), …] → lazy list of float32 frames H×W×5 = (R, G, B, alpha, shadow):
    straight-alpha RGB of the object + a REAL cast-shadow matte on an invisible floor (`floor`="auto": touching the
    object's lowest point; or a number = bounding radii below the centre; 0 / shadow=False → no shadow).
    Cached on disk (8-bit PNGs, ~100 KB/frame) by content hash; one CLI call per chunk → memory stays flat."""
    import cv2
    model = model if isinstance(model, dict) else prepare_model(model)
    hdri = hdri_path(hdri)
    size = int(round(size))
    poses = [tuple(float(v) for v in p) for p in poses]
    fl = (floor if floor == "auto" else float(floor)) if (shadow and floor) else 0.0
    key = _key(model, size, poses, hdri, light, env_rot, fl, fps)
    cdir = os.path.join(CACHE_DIR, key)
    if not os.path.exists(os.path.join(cdir, "done")):
        os.makedirs(cdir, exist_ok=True)
        base_l = None
        floor_px = None
        for s0 in range(0, len(poses), chunk):
            part = poses[s0:s0 + chunk]
            with tempfile.TemporaryDirectory(dir=_tmp_root()) as td:
                # pass 1: the object alone (clean anti-aliased alpha)
                _build_project(os.path.join(td, "a.ecproj"), model, size, part, fps, hdri, light, env_rot)
                _render_seq(os.path.join(td, "a.ecproj"), os.path.join(td, "a"))
                if fl:
                    if floor_px is None:      # ground contact = lowest object pixel over the first chunk (+1 px)
                        lows = []
                        for i in range(len(part)):
                            al = cv2.imread(os.path.join(td, "a", f"f_{i:05d}.png"), cv2.IMREAD_UNCHANGED)[..., 3]
                            ys = np.nonzero(al.max(1) > 40)[0]
                            if len(ys):
                                lows.append(ys.max())
                        D0 = al.shape[0]
                        floor_px = (max(lows) - D0 / 2 + 1) if (lows and fl == "auto") else \
                            size / 2 * float(fl if fl != "auto" else 0.5)
                    hs = max(64, size // 2)   # the shadow is soft: its passes render at half size (~4x cheaper)
                    # pass 2: catcher + object as "shadow only";  base: catcher with nothing on it (once)
                    _build_project(os.path.join(td, "b.ecproj"), model, hs, part, fps, hdri, light, env_rot,
                                   floor_px * hs / size, True)
                    _render_seq(os.path.join(td, "b.ecproj"), os.path.join(td, "b"))
                    if base_l is None:
                        _build_project(os.path.join(td, "c.ecproj"), model, hs, part[:1], fps, hdri, light, env_rot,
                                       floor_px * hs / size, True, hide=True)
                        _ec(["render-frame", "--project", os.path.join(td, "c.ecproj"), "--comp", "S", "--time", "0",
                             "--transparent", "--out", os.path.join(td, "base.png")])
                        bb = cv2.imread(os.path.join(td, "base.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
                        base_l = (bb[..., :3].mean(2) + 2.0, bb[..., 3] > 0)
                for i in range(len(part)):
                    dst = os.path.join(cdir, f"{s0 + i:05d}.png")
                    shutil.move(os.path.join(td, "a", f"f_{i:05d}.png"), dst)
                    if fl:
                        w = cv2.imread(os.path.join(td, "b", f"f_{i:05d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
                        sh = np.clip(1.0 - (w[..., :3].mean(2) + 2.0) / base_l[0], 0, 1) * base_l[1]
                        Df = cv2.imread(dst, cv2.IMREAD_UNCHANGED).shape[0]
                        sh = cv2.resize(sh.astype(np.float32), (Df, Df), interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(os.path.join(cdir, f"{s0 + i:05d}_s.png"), (sh * 255).astype(np.uint8))
            if verbose:
                print(f"[pbr3d] {min(len(poses), s0 + chunk)}/{len(poses)} frames", flush=True)
        open(os.path.join(cdir, "done"), "w").write(str(len(poses)))
    return _CachedSeq(cdir, len(poses))


class _CachedSeq:
    """Lazy list of cached frames (keeps RAM flat: frames are loaded on access)."""

    def __init__(self, d, n):
        self.d, self.n = d, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        """→ float32 H×W×5 (R, G, B, alpha, shadow)."""
        import cv2
        i = max(0, min(self.n - 1, int(i)))
        im = cv2.imread(os.path.join(self.d, f"{i:05d}.png"), cv2.IMREAD_UNCHANGED)
        out = np.zeros(im.shape[:2] + (5,), np.float32)
        out[..., :3] = im[..., 2::-1] * (1 / 255.0)
        out[..., 3] = im[..., 3] * (1 / 255.0)
        sp = os.path.join(self.d, f"{i:05d}_s.png")
        if os.path.exists(sp):
            out[..., 4] = cv2.imread(sp, cv2.IMREAD_GRAYSCALE) * (1 / 255.0)
        return out


def render_pose(model, size, yaw=0.0, pitch=15.0, roll=0.0, **kw):
    return render_sequence(model, size, [(yaw, pitch, roll)], **kw)[0]


# =========================================================================================== compositing helpers
def to_bgra_premul(fr, shadow_strength=1.0, shadow_color=(0, 0, 0), shadow_gain=2.2):
    """frame (RGB, alpha[, shadow]) → premultiplied BGRA float 0..255 (object over its own cast shadow)."""
    rgb, a = fr[..., :3], fr[..., 3:4]
    out = np.zeros(fr.shape[:2] + (4,), np.float32)
    if fr.shape[2] > 4:
        s = np.clip(fr[..., 4:5] * shadow_gain * shadow_strength, 0, 0.85) * (1 - a)
        sc = np.array(shadow_color, np.float32)[None, None, ::-1] / 255.0      # RGB → BGR
        out[..., :3] = sc * s * 255.0
        out[..., 3:4] = s * 255.0
    out[..., :3] = out[..., :3] * (1 - a) + rgb[..., ::-1] * a * 255.0
    out[..., 3:4] = out[..., 3:4] * (1 - a) + a * 255.0
    return out


def paint(ctx, fr, cx, cy, alpha=1.0, shadow_strength=1.0):
    from .obj3d import paint_premul
    D = fr.shape[0]
    paint_premul(ctx, to_bgra_premul(fr, shadow_strength), cx - D * 0.5, cy - D * 0.5, alpha)


def comp_over(img_bgr_f, fr, cx, cy, shadow_strength=1.0):
    """Composite one frame over a float BGR image (0..1) in place (numpy path, no cairo)."""
    pm = to_bgra_premul(fr, shadow_strength) / 255.0
    H, W = img_bgr_f.shape[:2]
    D = pm.shape[0]
    x0, y0 = int(round(cx - D / 2)), int(round(cy - D / 2))
    xs0, ys0, xs1, ys1 = max(0, -x0), max(0, -y0), min(D, W - x0), min(D, H - y0)
    if xs1 <= xs0 or ys1 <= ys0:
        return img_bgr_f
    roi = img_bgr_f[y0 + ys0:y0 + ys1, x0 + xs0:x0 + xs1]
    p = pm[ys0:ys1, xs0:xs1]
    roi[:] = p[..., :3] + roi * (1 - p[..., 3:4])
    return img_bgr_f


# =========================================================================================== audit sheet
def audit_sheet(src, out=None, size=360, bg=(0.11, 0.12, 0.14), hdri=None):
    import cv2
    m = prepare_model(src)
    poses = [(y, p, 0) for p in (15, 45) for y in (0, 90, 180, 270)] + [(0, 89, 0), (0, -60, 0)]
    seq = render_sequence(m, size, poses, hdri=hdri, verbose=True)
    tiles = []
    for i, (y, p, _) in enumerate(poses):
        fr = seq[i]
        im = np.zeros(fr.shape[:2] + (3,), np.float32)
        im[:] = bg
        comp_over(im, fr, fr.shape[1] / 2, fr.shape[0] / 2)
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        cv2.putText(im, f"yaw {y:.0f} pitch {p:.0f}", (8, 22), 0, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(im)
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, 10, 5)]
    sheet = np.vstack(rows)
    out = out or os.path.join(os.getcwd(), "out", f"pbr3d_{os.path.basename(str(src)).split('.')[0]}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, sheet)
    return out


# =========================================================================================== timeline tracks
def _locked(path):
    import fcntl

    class _L:
        def __enter__(self):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.f = open(path, "w")
            fcntl.flock(self.f, fcntl.LOCK_EX)
            return self

        def __exit__(self, *a):
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
    return _L()


_TRACKS = {}


def track(model, size, angles_fn, t0, t1, fps=30, **kw):
    """Pre-render a whole on-screen interval at once: angles_fn(local_t) → (yaw, pitch, roll).
    Returns fn(t) → frame.  Shared between render workers (disk cache + file lock) and frames (RAM memo)."""
    base_dir = kw.pop("base_dir", None)
    n = max(1, int(math.ceil((t1 - t0) * fps)) + 1)
    poses = [tuple(angles_fn(i / fps)) for i in range(n)]
    m = model if isinstance(model, dict) else prepare_model(model, base_dir)
    fl = kw.get("floor", "auto") if kw.get("shadow", True) else 0
    key = _key(m, int(round(size)), poses, hdri_path(kw.get("hdri")), kw.get("light"), kw.get("env_rot", 0.0),
               fl if fl == "auto" or not fl else float(fl), fps)
    if key not in _TRACKS:
        with _locked(os.path.join(CACHE_DIR, key + ".lock")):
            _TRACKS[key] = render_sequence(m, size, poses, fps=fps, **kw)
    seq = _TRACKS[key]
    return lambda t: seq[int(round((t - t0) * fps))]


def paint_scaled(ctx, fr, cx, cy, scale=1.0, alpha=1.0, shadow_strength=1.0):
    """cairo paint of a frame centred at (cx, cy) with an extra scale (springs / pops without re-rendering)."""
    import cairo
    pm = np.ascontiguousarray(np.clip(to_bgra_premul(fr, shadow_strength) * float(alpha), 0, 255).astype(np.uint8))
    h, w = pm.shape[:2]
    surf = cairo.ImageSurface.create_for_data(pm, cairo.FORMAT_ARGB32, w, h)
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(scale, scale)
    ctx.set_source_surface(surf, -w / 2, -h / 2)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint()
    ctx.restore()
    surf.finish()


# =========================================================================================== element (v5–v13 scenes)
def _register():
    from .elements import element, xy

    @element("pbr3d", dur=0.8, enter="pop", sfx="whoosh")
    def el_pbr3d(ctx, el, s):
        """PHOTOREAL 3D model (scanned PBR glTF, HDRI lighting, real cast shadow):
        {"type": "pbr3d", "model": "croissant" | "path/to/model.glb", "w": 460, "x": 540, "y": 900,
         "motion": "turntable_osc" | "spin_360" | "hover" | "spring_entry" | "none", "yaw": 0, "pitch": 15,
         "hdri": "studio", "shadow": true, "light": {"key": 60, "env": 100}}"""
        from .obj3d import motion_angles
        x, y = xy(el, s.t)
        size = float(el.get("w", el.get("size", 420)))
        t0 = float(el.get("t", 0.0))
        t1 = float(el["out"]) + float(el.get("out_dur", 0.45)) if el.get("out") is not None else \
            float(getattr(s.scene, "duration", t0 + 6))
        fps = float(getattr(s.scene, "fps", 30))
        if "_pbr" not in el:
            yaw0, pitch0 = float(el.get("yaw", 0.0)), float(el.get("pitch", 15.0))
            roll0 = float(el.get("roll", 0.0))
            el["_pbr"] = track(el.get("model", "croissant"), size,
                               lambda lt: (*motion_angles(el, lt, yaw0, pitch0), roll0), t0, t1, fps,
                               hdri=el.get("hdri"), light=el.get("light"), shadow=bool(el.get("shadow", True)),
                               env_rot=float(el.get("env_rot", 0.0)), base_dir=getattr(s.scene, "base_dir", None))
        paint_scaled(ctx, el["_pbr"](s.t), x, y + s.dy, s.scale, s.alpha, float(el.get("shadow_strength", 1.0)))
        return (x - size / 2, y - size / 2, x + size / 2, y + size / 2)


try:
    _register()
except Exception:  # pragma: no cover
    pass
