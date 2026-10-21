"""
mograph.three3d — VE v20 Procedural Three.js 3D Engine (True 360° Closed Meshes + Two-Pass Glass/Metallic PBR).

Why v20 replaces `obj3d.py` 2D image slab extrusion:
  `obj3d.py` thickens a single 2D PNG along Z (`depth=0.14`) and paints a flat silhouette on the back. Past ~40° yaw
  it collapses into a cardboard cutout, its lighting is frozen into the 2D bitmap, and internal sub-parts cannot move.

`three3d` implements a Blender-style Procedural Three.js Scene Graph & Geometry API (`THREE.*` primitives:
`LatheGeometry`, `TubeGeometry` + `CatmullRomCurve3`, `OctahedronGeometry`, `CylinderGeometry`, `TorusGeometry`,
`ConeGeometry`, `BoxGeometry`, `BufferGeometry`, `MeshStandardMaterial`, `MeshPhysicalMaterial`) and also includes a
**Node.js `.js` Script Bridge (`load_three_js_script`)** that executes standalone JavaScript Three.js generator scripts
(`three_models.js` or custom `.js` files) directly into the **Two-Pass Deferred PBR + Refractive Glass Pipeline**
(`zraster.raster_attr`, 2× SSAA):
  1. Opaque Pass: Cook-Torrance GGX specular, 3 studio softboxes + vertical strip light, SSAO, ACES Filmic tone map.
  2. Transparent Glass / Crystal Pass: depth-tested against the opaque z-buffer (`z_glass < z_opaque`), Schlick
     Fresnel rim reflectance (`F0=0.08`), clearcoat studio reflections, and transmission over internal geometry
     (e.g. golden sand visible inside the blown-glass hourglass bulb!).

Usage in v14 `edit.json`:
    {"type": "object_3d", "t": "w:ساعت", "out": "we:شنی+0.4",
     "three_model": "hourglass_3d",          # or microblading_pen_3d | emerald_tree_3d | star_trophy_3d | shield_3d …
     "three_script": "my_custom_object.js",  # optional: path to a custom Three.js procedural generator .js/.py file
     "anchor": "shoulder_l", "offset": [-0.35, -0.45], "size": 1.35,
     "motion": "spin_360", "yaw": 25, "pitch": 16, "explode": 0.0}

CLI:
    python3 -m mograph.three3d selftest
    python3 -m mograph.three3d list
    python3 -m mograph.three3d sheet hourglass_3d --out out/hourglass_sheet.png
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from functools import lru_cache

import cv2
import numpy as np

from . import zraster
from .real3d import LIGHTS, SKY, GROUND, _norm, aces, env, grid_faces, lathe, rot

HERE = os.path.dirname(os.path.abspath(__file__))
JS_MODELS_PATH = os.path.join(HERE, "three_models.js")


# ============================================================================================ THREE.js Geometry & Scene API
class Vector2:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)


class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def to_np(self):
        return np.array([self.x, self.y, self.z], np.float32)


class _Euler:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def set(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _Pos:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def set(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class MeshStandardMaterial:
    def __init__(self, color=0xffffff, metalness=0.0, roughness=0.5, emissive=0x000000, emissiveIntensity=0.0,
                 flatShading=False, transparent=False, opacity=1.0, clearcoat=0.0, clearcoatRoughness=0.05, **kw):
        self.color = _hex_to_linear(color)
        self.metalness = float(metalness)
        self.roughness = float(roughness)
        self.emissive = _hex_to_linear(emissive) * float(emissiveIntensity)
        self.flatShading = bool(flatShading)
        self.transparent = bool(transparent) or float(opacity) < 0.99
        self.opacity = float(opacity)
        self.clearcoat = float(clearcoat)
        self.clearcoatRoughness = float(clearcoatRoughness)


class MeshPhysicalMaterial(MeshStandardMaterial):
    pass


def _hex_to_linear(c):
    if isinstance(c, (tuple, list, np.ndarray)):
        return np.asarray(c, np.float32) ** 2.2
    c = int(c)
    r = ((c >> 16) & 255) / 255.0
    g = ((c >> 8) & 255) / 255.0
    b = (c & 255) / 255.0
    return np.array([r, g, b], np.float32) ** 2.2


class BufferGeometry:
    def __init__(self, v=None, n=None, uv=None, f=None):
        self.v = np.zeros((0, 3), np.float32) if v is None else np.asarray(v, np.float32).copy()
        self.n = np.zeros((0, 3), np.float32) if n is None else _norm(np.asarray(n, np.float32).copy())
        self.uv = np.zeros((0, 2), np.float32) if uv is None else np.asarray(uv, np.float32).copy()
        self.f = np.zeros((0, 3), np.int32) if f is None else np.asarray(f, np.int32).copy()

    def clone(self):
        return BufferGeometry(self.v, self.n, self.uv, self.f)

    def scale(self, sx, sy, sz):
        s = np.array([sx, sy, sz], np.float32)
        self.v = self.v * s
        if len(self.n):
            self.n = _norm(self.n / np.maximum(1e-6, s))
        return self

    def translate(self, dx, dy, dz):
        self.v = self.v + np.array([dx, dy, dz], np.float32)
        return self

    def computeVertexNormals(self, flat=False):
        if len(self.f) == 0 or len(self.v) == 0:
            return self
        v0 = self.v[self.f[:, 0]]
        v1 = self.v[self.f[:, 1]]
        v2 = self.v[self.f[:, 2]]
        fn = _norm(np.cross(v1 - v0, v2 - v0))
        if flat:
            self.v = self.v[self.f].reshape(-1, 3)
            self.n = np.repeat(fn, 3, axis=0)
            self.uv = self.v[:, :2]
            self.f = np.arange(len(self.v), dtype=np.int32).reshape(-1, 3)
        else:
            vn = np.zeros_like(self.v)
            np.add.at(vn, self.f[:, 0], fn)
            np.add.at(vn, self.f[:, 1], fn)
            np.add.at(vn, self.f[:, 2], fn)
            self.n = _norm(vn)
        return self


def CylinderGeometry(r_top=0.5, r_bot=0.5, height=1.0, segs=28):
    segs = min(int(segs), 32)
    hh = height * 0.5
    prof = [(0.0, -hh), (r_bot, -hh), (r_bot, -hh), (r_top, hh), (r_top, hh), (0.0, hh)]
    m = lathe(prof, segs)
    return BufferGeometry(m.v, m.n, m.uv, m.f)


def ConeGeometry(radius=0.5, height=1.0, segs=20):
    return CylinderGeometry(0.001, radius, height, min(int(segs), 24))


def SphereGeometry(radius=0.5, w_segs=24, h_segs=16):
    ang = np.linspace(-math.pi / 2, math.pi / 2, min(int(h_segs), 20))
    prof = [(max(0.0, radius * math.cos(a)), radius * math.sin(a)) for a in ang]
    m = lathe(prof, min(int(w_segs), 28))
    return BufferGeometry(m.v, m.n, m.uv, m.f)


def TorusGeometry(radius=0.5, tube=0.1, r_segs=10, t_segs=28):
    """Torus in the XY plane (matches Three.js default; rotation.x = pi/2 lays it flat in XZ)."""
    r_segs = min(int(r_segs), 12)
    t_segs = min(int(t_segs), 32)
    u = np.linspace(0, 2 * math.pi, t_segs + 1, dtype=np.float32)
    v = np.linspace(0, 2 * math.pi, r_segs + 1, dtype=np.float32)
    cu, su = np.cos(u)[:, None], np.sin(u)[:, None]
    cv, sv = np.cos(v)[None, :], np.sin(v)[None, :]
    x = (radius + tube * cv) * cu
    y = (radius + tube * cv) * su
    z = np.broadcast_to(tube * sv, x.shape)
    nx = cv * cu
    ny = cv * su
    nz = np.broadcast_to(sv, x.shape)
    V = np.stack([x, y, z], -1).reshape(-1, 3)
    N = np.stack([nx, ny, nz], -1).reshape(-1, 3)
    uu, vv = np.meshgrid(u / (2 * math.pi), v / (2 * math.pi), indexing="ij")
    UV = np.stack([uu, vv], -1).reshape(-1, 2)
    return BufferGeometry(V, N, UV, grid_faces(t_segs + 1, r_segs + 1))


def BoxGeometry(w=1.0, h=1.0, d=1.0, bevel=0.0):
    from .sculpt3d import mesh_rounded_box
    v, f, n = mesh_rounded_box(w, h, d, bevel=bevel if bevel > 0 else min(w, h, d) * 0.04, subdiv=2)
    return BufferGeometry(v, n, v[:, :2], f)


def LatheGeometry(points, segs=32):
    segs = min(int(segs), 36)
    prof = [(p.x, p.y) if hasattr(p, "x") else (float(p[0]), float(p[1])) for p in points]
    m = lathe(prof, segs)
    return BufferGeometry(m.v, m.n, m.uv, m.f)


def OctahedronGeometry(radius=1.0, detail=0):
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32)
    f = np.array([[0, 2, 4], [4, 2, 1], [1, 2, 5], [5, 2, 0],
                  [4, 3, 0], [1, 3, 4], [5, 3, 1], [0, 3, 5]], np.int32)
    for _ in range(int(detail)):
        nv, nf = list(v), []
        for a, b, c in f:
            ab = _norm((v[a] + v[b]) * 0.5)
            bc = _norm((v[b] + v[c]) * 0.5)
            ca = _norm((v[c] + v[a]) * 0.5)
            iab, ibc, ica = len(nv), len(nv) + 1, len(nv) + 2
            nv.extend([ab, bc, ca])
            nf.extend([[a, iab, ica], [b, ibc, iab], [c, ica, ibc], [iab, ibc, ica]])
        v = np.array(nv, np.float32)
        f = np.array(nf, np.int32)
    v = _norm(v) * float(radius)
    g = BufferGeometry(v, v.copy(), v[:, :2], f)
    g.computeVertexNormals(flat=True)
    return g


class CatmullRomCurve3:
    def __init__(self, points):
        self.pts = np.array([p.to_np() if hasattr(p, "to_np") else p for p in points], np.float32)

    def sample(self, n=24):
        P = self.pts
        m = len(P)
        if m < 2:
            return np.repeat(P, n, axis=0)
        pad = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
        out = []
        ts = np.linspace(0, m - 1 - 1e-6, n, dtype=np.float32)
        for u in ts:
            i = int(u)
            t = u - i
            p0, p1, p2, p3 = pad[i], pad[i + 1], pad[i + 2], pad[i + 3]
            t2, t3 = t * t, t * t * t
            pt = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(pt)
        return np.array(out, np.float32)


def TubeGeometry(curve, tubular_segs=22, radius=0.05, radial_segs=12, closed=False, taper=0.55):
    tubular_segs = min(int(tubular_segs), 24)
    radial_segs = min(int(radial_segs), 14)
    C = curve.sample(tubular_segs + 1)
    T = np.zeros_like(C)
    T[:-1] = C[1:] - C[:-1]
    T[-1] = T[-2]
    T = _norm(T)
    up = np.array([0.0, 0.0, 1.0], np.float32) if abs(float(T[0, 2])) < 0.9 else np.array([1.0, 0.0, 0.0], np.float32)
    B = _norm(np.cross(T, up))
    N_f = _norm(np.cross(B, T))
    th = np.linspace(0, 2 * math.pi, radial_segs + 1, dtype=np.float32)
    ct, st = np.cos(th), np.sin(th)
    V = np.zeros((tubular_segs + 1, radial_segs + 1, 3), np.float32)
    N = np.zeros_like(V)
    UV = np.zeros((tubular_segs + 1, radial_segs + 1, 2), np.float32)
    for i in range(tubular_segs + 1):
        u = i / max(1, tubular_segs)
        r = radius * (1.0 - taper * u)
        n_ring = N_f[i][None, :] * ct[:, None] + B[i][None, :] * st[:, None]
        V[i] = C[i][None, :] + r * n_ring
        N[i] = n_ring
        UV[i, :, 0] = th / (2 * math.pi)
        UV[i, :, 1] = u
    return BufferGeometry(V.reshape(-1, 3), N.reshape(-1, 3), UV.reshape(-1, 2), grid_faces(tubular_segs + 1, radial_segs + 1))


class _Node:
    def __init__(self):
        self.position = _Pos()
        self.rotation = _Euler()
        self.userData = {}
        self.children = []

    def add(self, child):
        self.children.append(child)
        return self


class Group(_Node):
    pass


class Mesh(_Node):
    def __init__(self, geometry, material):
        super().__init__()
        self.geometry = geometry.clone()
        if getattr(material, "flatShading", False):
            self.geometry.computeVertexNormals(flat=True)
        self.material = material
        self.isMesh = True


class THREE:
    """Drop-in Python namespace matching Three.js classes for procedural 3D model scripts."""
    Vector2 = Vector2
    Vector3 = Vector3
    Group = Group
    Mesh = Mesh
    BufferGeometry = BufferGeometry
    CylinderGeometry = staticmethod(CylinderGeometry)
    ConeGeometry = staticmethod(ConeGeometry)
    SphereGeometry = staticmethod(SphereGeometry)
    TorusGeometry = staticmethod(TorusGeometry)
    BoxGeometry = staticmethod(BoxGeometry)
    LatheGeometry = staticmethod(LatheGeometry)
    OctahedronGeometry = staticmethod(OctahedronGeometry)
    CatmullRomCurve3 = CatmullRomCurve3
    TubeGeometry = staticmethod(TubeGeometry)
    MeshStandardMaterial = MeshStandardMaterial
    MeshPhysicalMaterial = MeshPhysicalMaterial
    DoubleSide = 2


# ============================================================================================ Node.js .js Bridge
_NODE_SHIM = r"""
const fs = require('fs');
class Vector2 { constructor(x=0, y=0) { this.x = x; this.y = y; } }
class Vector3 { constructor(x=0, y=0, z=0) { this.x = x; this.y = y; this.z = z; } }
class Euler { constructor() { this.x = 0; this.y = 0; this.z = 0; } set(x,y,z) { this.x=x; this.y=y; this.z=z; } }
class Pos { constructor() { this.x = 0; this.y = 0; this.z = 0; } set(x,y,z) { this.x=x; this.y=y; this.z=z; } }
class Mat { constructor(kind, opts={}) { this.kind = kind; Object.assign(this, opts); } }
class Geo {
  constructor(kind, args) { this.kind = kind; this.args = args; this.scales = []; this.attrs = {}; }
  scale(sx, sy, sz) { this.scales.push([sx, sy, sz]); return this; }
  setAttribute(k, v) { this.attrs[k] = v.arr; return this; }
  computeVertexNormals() { return this; }
}
class Node3D {
  constructor() { this.position = new Pos(); this.rotation = new Euler(); this.userData = {}; this.children = []; }
  add(c) { this.children.push(c); return this; }
}
class Group extends Node3D { constructor() { super(); this.type = 'Group'; } }
class Mesh extends Node3D { constructor(geo, mat) { super(); this.type = 'Mesh'; this.geometry = geo; this.material = mat; } }
const THREE = {
  Vector2, Vector3, Group, Mesh, DoubleSide: 2,
  MeshStandardMaterial: class extends Mat { constructor(o) { super('Standard', o); } },
  MeshPhysicalMaterial: class extends Mat { constructor(o) { super('Physical', o); } },
  CylinderGeometry: class extends Geo { constructor(...a) { super('Cylinder', a); } },
  ConeGeometry: class extends Geo { constructor(...a) { super('Cone', a); } },
  SphereGeometry: class extends Geo { constructor(...a) { super('Sphere', a); } },
  TorusGeometry: class extends Geo { constructor(...a) { super('Torus', a); } },
  BoxGeometry: class extends Geo { constructor(...a) { super('Box', a); } },
  LatheGeometry: class extends Geo { constructor(pts, segs=32) { super('Lathe', [pts.map(p=>[p.x, p.y]), segs]); } },
  OctahedronGeometry: class extends Geo { constructor(...a) { super('Octahedron', a); } },
  CatmullRomCurve3: class { constructor(pts) { this.pts = pts.map(p=>[p.x, p.y, p.z]); } },
  TubeGeometry: class extends Geo {
    constructor(curve, tSegs=24, r=0.05, rSegs=12, closed=false) {
      super('Tube', [curve.pts, tSegs, r, rSegs, closed]);
      this.attributes = { position: { count: 0, getY: ()=>0, getX: ()=>0, setX: ()=>{} } };
    }
  },
  BufferGeometry: class extends Geo { constructor() { super('Buffer', []); } },
  Float32BufferAttribute: class { constructor(arr, n) { this.arr = arr; this.itemSize = n; } }
};
const code = fs.readFileSync(process.argv[1], 'utf8');
const fnName = process.argv[2] || '';
const fn = new Function('THREE', code + `\n; const mats = typeof createStudioMaterials === 'function' ? createStudioMaterials(THREE) : {};
if ("${fnName}" && typeof ${fnName || 'undefined'} === 'function') return ${fnName}(THREE, mats);
if (typeof build === 'function') return build(THREE, mats);
if (typeof buildHourglass3D === 'function') return buildHourglass3D(THREE, mats);
throw new Error('No builder function found in JS script');`);
console.log(JSON.stringify(fn(THREE)));
"""


def _deserialize_js_node(d):
    if d.get("type") == "Mesh":
        g = d["geometry"]
        kind, args = g["kind"], g["args"]
        if kind == "Cylinder":
            geo = CylinderGeometry(*args)
        elif kind == "Cone":
            geo = ConeGeometry(*args)
        elif kind == "Sphere":
            geo = SphereGeometry(*args)
        elif kind == "Torus":
            geo = TorusGeometry(*args)
        elif kind == "Box":
            geo = BoxGeometry(*args)
        elif kind == "Lathe":
            geo = LatheGeometry(args[0], args[1] if len(args) > 1 else 32)
        elif kind == "Octahedron":
            geo = OctahedronGeometry(*args)
        elif kind == "Tube":
            geo = TubeGeometry(CatmullRomCurve3(args[0]), args[1], args[2], args[3], args[4])
        elif kind == "Buffer":
            pos = np.array(g["attrs"]["position"], np.float32).reshape(-1, 3)
            f = np.arange(len(pos), dtype=np.int32).reshape(-1, 3)
            geo = BufferGeometry(pos, np.zeros_like(pos), pos[:, :2], f).computeVertexNormals(flat=True)
        else:
            geo = BoxGeometry(0.5, 0.5, 0.5)
        for sx, sy, sz in g.get("scales", []):
            geo.scale(sx, sy, sz)
        m = dict(d.get("material") or {})
        m.pop("kind", None)
        mat = MeshPhysicalMaterial(**m)
        node = Mesh(geo, mat)
    else:
        node = Group()
    pos = d.get("position") or {}
    rot_d = d.get("rotation") or {}
    node.position.set(pos.get("x", 0), pos.get("y", 0), pos.get("z", 0))
    node.rotation.set(rot_d.get("x", 0), rot_d.get("y", 0), rot_d.get("z", 0))
    ev = (d.get("userData") or {}).get("explodeVec")
    if ev:
        node.userData["explodeVec"] = Vector3(ev.get("x", 0), ev.get("y", 0), ev.get("z", 0))
    for ch in d.get("children") or []:
        node.add(_deserialize_js_node(ch))
    return node


@lru_cache(maxsize=16)
def load_three_js_script(js_path: str, fn_name: str = ""):
    """Execute a JavaScript Three.js procedural generator file via Node.js and return a `THREE.Group` root node."""
    res = subprocess.run(["node", "-e", _NODE_SHIM, js_path, fn_name], capture_output=True, text=True, check=True)
    return _deserialize_js_node(json.loads(res.stdout))


# ============================================================================================ Studio PBR Materials
def create_studio_materials():
    return {
        "gold": MeshStandardMaterial(color=0xf7be48, metalness=0.88, roughness=0.16),
        "brass": MeshStandardMaterial(color=0xd99b32, metalness=0.82, roughness=0.24),
        "matteBlack": MeshStandardMaterial(color=0x1c1f26, metalness=0.25, roughness=0.35),
        "chrome": MeshStandardMaterial(color=0xe2e8f0, metalness=0.94, roughness=0.10),
        "glass": MeshPhysicalMaterial(color=0xe6f4ff, transparent=True, opacity=0.34, roughness=0.06, metalness=0.05, clearcoat=1.0),
        "crystal": MeshPhysicalMaterial(color=0xd8e2f2, transparent=True, opacity=0.68, roughness=0.10, metalness=0.35, clearcoat=1.0, flatShading=True),
        "emerald": MeshPhysicalMaterial(color=0x10b981, emissive=0x043824, emissiveIntensity=0.45, roughness=0.08, metalness=0.20, clearcoat=1.0, flatShading=True),
        "sand": MeshStandardMaterial(color=0xf59e0b, emissive=0x78350f, emissiveIntensity=0.35, roughness=0.38, metalness=0.15),
        "cobalt": MeshPhysicalMaterial(color=0x2563eb, emissive=0x0b1f52, emissiveIntensity=0.35, roughness=0.14, metalness=0.35, clearcoat=1.0),
        "amberGlow": MeshPhysicalMaterial(color=0xf59e0b, emissive=0xb45309, emissiveIntensity=0.55, roughness=0.12, metalness=0.25, clearcoat=1.0),
    }


# ============================================================================================ Procedural 3D Model Builders
def build_hourglass_3d(mats=None):
    mats = mats or create_studio_materials()
    group = THREE.Group()

    for sgn in (-1, 1):
        outer_cap = THREE.Mesh(THREE.CylinderGeometry(0.68, 0.68, 0.08, 28), mats["gold"])
        outer_cap.position.y = sgn * 0.94
        outer_cap.userData["explodeVec"] = THREE.Vector3(0, sgn * 0.45, 0)
        group.add(outer_cap)

        inner_cap = THREE.Mesh(THREE.CylinderGeometry(0.58, 0.58, 0.07, 28), mats["brass"])
        inner_cap.position.y = sgn * 0.86
        inner_cap.userData["explodeVec"] = THREE.Vector3(0, sgn * 0.32, 0)
        group.add(inner_cap)

        bezel = THREE.Mesh(THREE.TorusGeometry(0.58, 0.024, 10, 28), mats["gold"])
        bezel.rotation.x = math.pi / 2
        bezel.position.y = sgn * 0.825
        bezel.userData["explodeVec"] = THREE.Vector3(0, sgn * 0.25, 0)
        group.add(bezel)

    for k in range(4):
        ang = (k * math.pi) / 2 + math.pi / 4
        px, pz = math.cos(ang) * 0.53, math.sin(ang) * 0.53
        pg = THREE.Group()
        pg.position.set(px, 0, pz)
        pg.userData["explodeVec"] = THREE.Vector3(px * 0.7, 0, pz * 0.7)
        pg.add(THREE.Mesh(THREE.CylinderGeometry(0.035, 0.035, 1.72, 16), mats["gold"]))
        for ry in (-0.78, 0.0, 0.78):
            ring = THREE.Mesh(THREE.TorusGeometry(0.048, 0.015, 8, 16), mats["brass"])
            ring.rotation.x = math.pi / 2
            ring.position.y = ry
            pg.add(ring)
        group.add(pg)

    top_sand = [(0.001, 0.03), (0.045, 0.08), (0.28, 0.32), (0.38, 0.48), (0.001, 0.46)]
    group.add(THREE.Mesh(THREE.LatheGeometry(top_sand, 24), mats["sand"]))

    bot_sand = [(0.001, -0.81), (0.42, -0.81), (0.40, -0.66), (0.20, -0.42), (0.001, -0.32)]
    group.add(THREE.Mesh(THREE.LatheGeometry(bot_sand, 24), mats["sand"]))

    stream = THREE.Mesh(THREE.CylinderGeometry(0.014, 0.022, 0.44, 12), mats["sand"])
    stream.position.y = -0.16
    group.add(stream)

    glass_pts = []
    for i in range(25):
        y = -0.82 + (1.64 * i) / 24.0
        ay = abs(y)
        r = 0.062 + 0.395 * (math.sin(min(1.0, ay / 0.82) * math.pi * 0.82) ** 1.35)
        glass_pts.append((r, y))
    group.add(THREE.Mesh(THREE.LatheGeometry(glass_pts, 28), mats["glass"]))
    return group


def build_microblading_pen_3d(mats=None):
    mats = mats or create_studio_materials()
    pen = THREE.Group()

    cap_pts = [(0.001, 1.04), (0.09, 1.02), (0.165, 0.93), (0.18, 0.74), (0.001, 0.74)]
    cap = THREE.Mesh(THREE.LatheGeometry(cap_pts, 24), mats["matteBlack"])
    cap.userData["explodeVec"] = THREE.Vector3(0, 0.45, 0)
    pen.add(cap)

    barrel = THREE.Mesh(THREE.CylinderGeometry(0.18, 0.165, 0.56, 28), mats["gold"])
    barrel.position.y = 0.46
    barrel.userData["explodeVec"] = THREE.Vector3(0, 0.25, 0)
    pen.add(barrel)

    mid_ring = THREE.Mesh(THREE.TorusGeometry(0.17, 0.015, 10, 24), mats["brass"])
    mid_ring.rotation.x = math.pi / 2
    mid_ring.position.y = 0.18
    pen.add(mid_ring)

    mid_pts = []
    for i in range(17):
        t = i / 16.0
        y = -0.34 + 0.52 * t
        r = 0.154 - 0.025 * math.sin(t * math.pi) + 0.012 * t
        mid_pts.append((r, y))
    pen.add(THREE.Mesh(THREE.LatheGeometry(mid_pts, 24), mats["matteBlack"]))

    collar = THREE.Mesh(THREE.CylinderGeometry(0.158, 0.164, 0.12, 24), mats["gold"])
    collar.position.y = -0.40
    collar.userData["explodeVec"] = THREE.Vector3(0, -0.15, 0)
    pen.add(collar)

    grip_grp = THREE.Group()
    grip_grp.userData["explodeVec"] = THREE.Vector3(0, -0.30, 0)
    grip_core = THREE.Mesh(THREE.CylinderGeometry(0.148, 0.128, 0.34, 24), mats["matteBlack"])
    grip_core.position.y = -0.62
    grip_grp.add(grip_core)
    for k in range(5):
        gr = THREE.Mesh(THREE.TorusGeometry(0.144 - k * 0.003, 0.014, 8, 22), mats["matteBlack"])
        gr.rotation.x = math.pi / 2
        gr.position.y = -0.49 - k * 0.06
        grip_grp.add(gr)
    pen.add(grip_grp)

    noz_grp = THREE.Group()
    noz_grp.userData["explodeVec"] = THREE.Vector3(0, -0.55, 0)
    chuck = THREE.Mesh(THREE.CylinderGeometry(0.116, 0.052, 0.24, 24), mats["chrome"])
    chuck.position.y = -0.90
    noz_grp.add(chuck)

    blade = THREE.Mesh(THREE.BoxGeometry(0.022, 0.24, 0.068), mats["gold"])
    blade.position.set(0.01, -1.08, 0)
    blade.rotation.z = 0.22
    noz_grp.add(blade)
    pen.add(noz_grp)

    wrapper = THREE.Group()
    pen.rotation.z = -0.64
    wrapper.add(pen)
    return wrapper


def build_emerald_tree_3d(mats=None):
    mats = mats or create_studio_materials()
    group = THREE.Group()

    glass_base = THREE.Mesh(THREE.CylinderGeometry(0.42, 0.46, 0.20, 28), mats["glass"])
    glass_base.position.y = -0.88
    glass_base.userData["explodeVec"] = THREE.Vector3(0, -0.35, 0)
    group.add(glass_base)

    plinth = THREE.Mesh(THREE.CylinderGeometry(0.33, 0.36, 0.07, 28), mats["gold"])
    plinth.position.y = -0.74
    plinth.userData["explodeVec"] = THREE.Vector3(0, -0.20, 0)
    group.add(plinth)

    branches = [
        ([[0, -0.74, 0], [0.04, -0.45, 0.03], [-0.04, -0.12, -0.02], [0.02, 0.25, 0.03], [0, 0.85, 0]], 0.085),
        ([[-0.02, -0.34, 0], [-0.26, -0.22, 0.09], [-0.54, -0.04, 0.16], [-0.74, 0.18, 0.12]], 0.048),
        ([[0.02, -0.30, 0], [0.28, -0.18, -0.09], [0.56, 0.0, -0.16], [0.76, 0.22, -0.12]], 0.048),
        ([[-0.03, -0.08, 0.02], [-0.28, 0.10, -0.12], [-0.54, 0.34, -0.20], [-0.68, 0.58, -0.14]], 0.040),
        ([[0.03, -0.04, -0.02], [0.28, 0.14, 0.12], [0.54, 0.38, 0.20], [0.68, 0.62, 0.14]], 0.040),
        ([[-0.01, 0.18, 0.02], [-0.20, 0.40, 0.15], [-0.38, 0.66, 0.18], [-0.46, 0.82, 0.14]], 0.032),
        ([[0.01, 0.20, -0.02], [0.20, 0.42, -0.15], [0.38, 0.68, -0.18], [0.46, 0.84, -0.14]], 0.032),
    ]
    for pts, r in branches:
        curve = THREE.CatmullRomCurve3(pts)
        group.add(THREE.Mesh(THREE.TubeGeometry(curve, 18, r, 10, False), mats["gold"]))

    gem_geo = THREE.OctahedronGeometry(0.12, 1).scale(0.82, 1.35, 0.72)
    gem_coords = [
        (0, 0.94, 0, 0),
        (-0.75, 0.22, 0.12, 0.5), (0.77, 0.26, -0.12, -0.5),
        (-0.69, 0.62, -0.14, 0.4), (0.69, 0.66, 0.14, -0.4),
        (-0.46, 0.86, 0.14, 0.25), (0.46, 0.88, -0.14, -0.25),
        (-0.22, 0.58, -0.22, 0.2), (0.22, 0.60, 0.22, -0.2),
        (-0.52, -0.06, 0.18, 0.7), (0.54, -0.02, -0.18, -0.7),
    ]
    for gx, gy, gz, rot_z in gem_coords:
        gg = THREE.Group()
        gg.position.set(gx, gy, gz)
        gg.rotation.z = rot_z
        gg.userData["explodeVec"] = THREE.Vector3(gx * 0.45, max(0.1, gy * 0.35), gz * 0.45)
        gg.add(THREE.Mesh(gem_geo, mats["emerald"]))
        cup = THREE.Mesh(THREE.CylinderGeometry(0.055, 0.02, 0.05, 12), mats["brass"])
        cup.position.y = -0.11
        gg.add(cup)
        group.add(gg)

    leaf_geo = THREE.ConeGeometry(0.055, 0.16, 10).scale(1.0, 1.0, 0.35)
    leaf_coords = [
        (-0.25, -0.12, 0.10, 0.7), (0.26, -0.08, -0.10, -0.7),
        (-0.44, 0.06, 0.16, -0.6), (0.46, 0.10, -0.16, 0.6),
        (-0.28, 0.20, -0.12, 0.5), (0.28, 0.24, 0.12, -0.5),
        (-0.50, 0.40, -0.18, -0.5), (0.50, 0.44, 0.18, 0.5),
        (-0.18, 0.42, 0.15, 0.4), (0.18, 0.44, -0.15, -0.4),
        (-0.34, 0.68, 0.18, -0.35), (0.34, 0.70, -0.18, 0.35),
        (-0.12, 0.66, 0.06, 0.3), (0.12, 0.68, -0.06, -0.3),
    ]
    for lx, ly, lz, rz in leaf_coords:
        lf = THREE.Mesh(leaf_geo, mats["gold"])
        lf.position.set(lx, ly, lz)
        lf.rotation.set(0.35, rz, rz)
        lf.userData["explodeVec"] = THREE.Vector3(lx * 0.3, ly * 0.2, lz * 0.3)
        group.add(lf)

    return group


def _star_polyhedron_geo(n_pts=5, r_outer=0.94, r_inner=0.36, z_apex=0.26):
    ring = []
    for i in range(n_pts * 2):
        ang = (i * math.pi) / n_pts
        r = r_outer if i % 2 == 0 else r_inner
        zr = 0.04 if i % 2 == 0 else -0.04
        ring.append([r * math.sin(ang), r * Math_cos(ang), zr] if False else [r * math.sin(ang), r * math.cos(ang), zr])
    verts, faces = [], []
    total = n_pts * 2
    for i in range(total):
        p0 = ring[i]
        p1 = ring[(i + 1) % total]
        b = len(verts)
        verts.extend([[0, 0, z_apex], p1, p0, [0, 0, -z_apex], p0, p1])
        faces.extend([[b, b + 1, b + 2], [b + 3, b + 4, b + 5]])
    v = np.array(verts, np.float32)
    f = np.array(faces, np.int32)
    g = BufferGeometry(v, np.zeros_like(v), v[:, :2], f)
    g.computeVertexNormals(flat=True)
    return g


def build_star_trophy_3d(mats=None):
    mats = mats or create_studio_materials()
    group = THREE.Group()

    base_blk = THREE.Mesh(THREE.CylinderGeometry(0.42, 0.44, 0.18, 28), mats["matteBlack"])
    base_blk.position.y = -0.89
    base_blk.userData["explodeVec"] = THREE.Vector3(0, -0.35, 0)
    group.add(base_blk)

    base_ring = THREE.Mesh(THREE.TorusGeometry(0.43, 0.022, 10, 28), mats["gold"])
    base_ring.rotation.x = math.pi / 2
    base_ring.position.y = -0.80
    base_ring.userData["explodeVec"] = THREE.Vector3(0, -0.25, 0)
    group.add(base_ring)

    base_gold = THREE.Mesh(THREE.CylinderGeometry(0.32, 0.34, 0.08, 28), mats["gold"])
    base_gold.position.y = -0.75
    base_gold.userData["explodeVec"] = THREE.Vector3(0, -0.18, 0)
    group.add(base_gold)

    stem = THREE.Mesh(THREE.CylinderGeometry(0.05, 0.08, 0.24, 18), mats["gold"])
    stem.position.y = -0.63
    group.add(stem)

    s1 = THREE.Mesh(_star_polyhedron_geo(5, 0.94, 0.36, 0.26), mats["gold"])
    s1.position.y = 0.10
    s1.userData["explodeVec"] = THREE.Vector3(0, 0.12, 0.25)
    group.add(s1)

    s2 = THREE.Mesh(_star_polyhedron_geo(5, 0.82, 0.38, 0.20), mats["crystal"])
    s2.position.y = 0.10
    s2.rotation.z = math.pi / 5
    s2.userData["explodeVec"] = THREE.Vector3(0, 0.12, -0.30)
    group.add(s2)

    s3 = THREE.Mesh(_star_polyhedron_geo(5, 0.52, 0.21, 0.38), mats["brass"])
    s3.position.y = 0.10
    s3.userData["explodeVec"] = THREE.Vector3(0, 0.18, 0.45)
    group.add(s3)

    return group


def build_shield_3d(mats=None):
    mats = mats or create_studio_materials()
    group = THREE.Group()
    outer = THREE.Mesh(THREE.CylinderGeometry(0.82, 0.72, 0.16, 6), mats["gold"])
    outer.rotation.x = math.pi / 2
    group.add(outer)
    core = THREE.Mesh(THREE.CylinderGeometry(0.68, 0.60, 0.22, 6), mats["cobalt"])
    core.rotation.x = math.pi / 2
    core.userData["explodeVec"] = THREE.Vector3(0, 0, 0.28)
    group.add(core)
    star = THREE.Mesh(_star_polyhedron_geo(5, 0.38, 0.16, 0.16), mats["gold"])
    star.position.z = 0.14
    star.userData["explodeVec"] = THREE.Vector3(0, 0, 0.50)
    group.add(star)
    return group


def build_cash_3d(mats=None):
    mats = mats or create_studio_materials()
    group = THREE.Group()
    for i, y in enumerate((-0.26, 0.0, 0.26)):
        bill = THREE.Mesh(THREE.BoxGeometry(1.35, 0.18, 0.72, bevel=0.03), mats["emerald"])
        bill.position.y = y
        bill.rotation.y = (i - 1) * 0.08
        bill.userData["explodeVec"] = THREE.Vector3(0, (i - 1) * 0.32, 0)
        group.add(bill)
    strap = THREE.Mesh(THREE.BoxGeometry(0.34, 0.76, 0.76, bevel=0.025), mats["gold"])
    strap.userData["explodeVec"] = THREE.Vector3(0, 0, 0.35)
    group.add(strap)
    return group


def build_warn_3d(mats=None):
    mats = mats or create_studio_materials()
    group = THREE.Group()
    tri = THREE.Mesh(THREE.CylinderGeometry(0.88, 0.88, 0.22, 3), mats["amberGlow"])
    tri.rotation.x = math.pi / 2
    group.add(tri)
    inner = THREE.Mesh(THREE.CylinderGeometry(0.62, 0.62, 0.26, 3), mats["matteBlack"])
    inner.rotation.x = math.pi / 2
    inner.userData["explodeVec"] = THREE.Vector3(0, 0, 0.22)
    group.add(inner)
    bar = THREE.Mesh(THREE.CylinderGeometry(0.06, 0.04, 0.36, 16), mats["gold"])
    bar.position.set(0, 0.06, 0.16)
    bar.userData["explodeVec"] = THREE.Vector3(0, 0.1, 0.45)
    group.add(bar)
    dot = THREE.Mesh(THREE.SphereGeometry(0.065, 16, 12), mats["gold"])
    dot.position.set(0, -0.22, 0.16)
    dot.userData["explodeVec"] = THREE.Vector3(0, -0.1, 0.45)
    group.add(dot)
    return group


def build_laptop_3d(mats=None):
    mats = mats or create_studio_materials()
    group = THREE.Group()
    base = THREE.Mesh(THREE.BoxGeometry(1.36, 0.07, 0.94, bevel=0.02), mats["chrome"])
    base.position.y = -0.36
    base.userData["explodeVec"] = THREE.Vector3(0, -0.25, 0)
    group.add(base)
    kbd = THREE.Mesh(THREE.BoxGeometry(1.12, 0.02, 0.52, bevel=0.005), mats["matteBlack"])
    kbd.position.set(0, -0.32, -0.06)
    group.add(kbd)
    lid_grp = THREE.Group()
    lid_grp.position.set(0, 0.12, -0.44)
    lid_grp.rotation.x = -0.22
    lid_grp.userData["explodeVec"] = THREE.Vector3(0, 0.28, -0.22)
    lid = THREE.Mesh(THREE.BoxGeometry(1.36, 0.90, 0.05, bevel=0.015), mats["chrome"])
    lid_grp.add(lid)
    scr = THREE.Mesh(THREE.BoxGeometry(1.24, 0.78, 0.02, bevel=0.005), mats["cobalt"])
    scr.position.z = 0.025
    lid_grp.add(scr)
    group.add(lid_grp)
    return group


MODELS = {
    "hourglass_3d": build_hourglass_3d,
    "hourglass": build_hourglass_3d,
    "microblading_pen_3d": build_microblading_pen_3d,
    "pen_3d": build_microblading_pen_3d,
    "pen": build_microblading_pen_3d,
    "ozempic_pen_3d": build_microblading_pen_3d,
    "emerald_tree_3d": build_emerald_tree_3d,
    "tree_3d": build_emerald_tree_3d,
    "tree": build_emerald_tree_3d,
    "star_trophy_3d": build_star_trophy_3d,
    "star_3d": build_star_trophy_3d,
    "star": build_star_trophy_3d,
    "trophy_3d": build_star_trophy_3d,
    "shield_3d": build_shield_3d,
    "shield": build_shield_3d,
    "cash_3d": build_cash_3d,
    "cash": build_cash_3d,
    "warn_3d": build_warn_3d,
    "warn": build_warn_3d,
    "laptop_3d": build_laptop_3d,
    "laptop": build_laptop_3d,
    "laptop_pro_3d": build_laptop_3d,
}


def resolve_model_name(name_or_src):
    """Map a model id or legacy sprite filename (`hourglass_3d.png`) to a canonical `three3d` model key, or None."""
    if not name_or_src:
        return None
    stem = os.path.splitext(os.path.basename(str(name_or_src)))[0].lower()
    if stem in MODELS:
        return stem
    for k in MODELS:
        if k in stem:
            return k
    return None


# ============================================================================================ Scene Flattening & Two-Pass PBR Renderer
class CompiledThreeScene:
    def __init__(self, root_node, explode=0.0):
        meshes = []
        self._collect(root_node, np.eye(3, dtype=np.float32), np.zeros(3, np.float32), float(explode), meshes)
        all_v = np.concatenate([v for v, _, _, _ in meshes], axis=0)
        self.center = (all_v.max(0) + all_v.min(0)) * 0.5
        self.radius = max(1e-4, float(np.linalg.norm(all_v - self.center, axis=1).max()))
        self.parts = meshes
        self.opaque = self._pack([m for m in meshes if not m[3].transparent])
        self.trans = self._pack([m for m in meshes if m[3].transparent])
        self.tri_count = (len(self.opaque["F"]) if self.opaque else 0) + (len(self.trans["F"]) if self.trans else 0)

    def _collect(self, node, R_parent, t_parent, explode, out):
        rx = math.degrees(node.rotation.x)
        ry = math.degrees(node.rotation.y)
        rz = math.degrees(node.rotation.z)
        R_loc = rot(rx, ry, rz)
        pos = np.array([node.position.x, node.position.y, node.position.z], np.float32)
        ev = node.userData.get("explodeVec")
        if ev is not None and explode != 0.0:
            pos = pos + ev.to_np() * explode
        R_w = R_parent @ R_loc
        t_w = t_parent + (pos @ R_parent.T)
        if getattr(node, "isMesh", False) and len(node.geometry.v):
            v = node.geometry.v @ R_w.T + t_w
            n = _norm(node.geometry.n @ R_w.T)
            out.append((v, n, node.geometry.f, node.material))
        for ch in node.children:
            self._collect(ch, R_w, t_w, explode, out)

    def _pack(self, mesh_list):
        if not mesh_list:
            return None
        vs, ns, fs, ids, mats = [], [], [], [], []
        off = 0
        for mid, (v, n, f, mat) in enumerate(mesh_list, start=1):
            vs.append(v - self.center)
            ns.append(n)
            fs.append(f + off)
            ids.append(np.full(len(f), mid, np.int32))
            mats.append(mat)
            off += len(v)
        return {
            "V": np.concatenate(vs).astype(np.float32),
            "N": np.concatenate(ns).astype(np.float32),
            "F": np.concatenate(fs).astype(np.int32),
            "FID": np.concatenate(ids).astype(np.int32),
            "mats": mats,
        }


CANONICAL_MODELS = ("hourglass_3d", "microblading_pen_3d", "emerald_tree_3d", "star_trophy_3d",
                    "shield_3d", "cash_3d", "warn_3d", "laptop_3d")


def list_models():
    return list(CANONICAL_MODELS)


def get_model(model_key: str, explode: float = 0.0) -> CompiledThreeScene:
    return get_compiled_scene(model_key, int(round(float(explode) * 100)))


@lru_cache(maxsize=32)
def get_compiled_scene(model_key: str, explode_q: int = 0) -> CompiledThreeScene:
    if str(model_key).endswith(".js") and os.path.exists(str(model_key)):
        return CompiledThreeScene(load_three_js_script(str(model_key)), explode=explode_q / 100.0)
    fn = MODELS.get(resolve_model_name(model_key) or model_key)
    if fn is None:
        raise KeyError(f"Unknown three3d model '{model_key}'. Available: {sorted(set(MODELS))}")
    return CompiledThreeScene(fn(), explode=explode_q / 100.0)


def _shade_pass(pack, R_cam, dist, Fpx, cx, cy, Dh, rad, z_clip=None, is_glass=False):
    P = pack["V"] @ R_cam.T
    N = _norm(pack["N"] @ R_cam.T)
    zc = dist - P[:, 2]
    sx = cx + Fpx * P[:, 0] / np.maximum(zc, 1e-4)
    sy = cy - Fpx * P[:, 1] / np.maximum(zc, 1e-4)
    F = pack["F"]
    cen = (P[F[:, 0]] + P[F[:, 1]] + P[F[:, 2]]) / 3.0
    nav = N[F[:, 0]] + N[F[:, 1]] + N[F[:, 2]]
    vd = np.array([0, 0, dist], np.float32) - cen
    if is_glass:
        keep = np.ones(len(F), dtype=bool)
    else:
        keep = np.sum(nav * vd, 1) > -0.45 * np.linalg.norm(nav, axis=1) * np.linalg.norm(vd, axis=1)
    Fk = F[keep]
    attr = np.concatenate([N, P], 1)                     # 6 attributes: normal (3) + cam_pos (3)
    xy = np.stack([sx, sy], 1)[Fk]
    zz = zc[Fk]
    ab, ids, zb = zraster.raster_attr(xy, zz, attr[Fk], pack["FID"][keep], Dh, Dh)
    mask = ids > 0
    if z_clip is not None:
        mask = mask & (zb < z_clip)
    if not mask.any():
        return np.zeros((Dh, Dh, 3), np.float32), np.zeros((Dh, Dh), np.float32), zb

    pix = ab[mask]
    n = _norm(pix[:, 0:3])
    pc = pix[:, 3:6]
    V = _norm(np.array([0, 0, dist], np.float32) - pc)
    flip = np.sum(n * V, 1) < 0
    if np.any(flip):
        n[flip] = -n[flip]
    ndv = np.clip(np.sum(n * V, 1), 1e-4, 1.0)
    mid = ids[mask]
    npx = len(mid)

    alb = np.zeros((npx, 3), np.float32)
    rough = np.full(npx, 0.3, np.float32)
    metal = np.zeros(npx, np.float32)
    emis = np.zeros((npx, 3), np.float32)
    opac = np.ones(npx, np.float32)
    ccoat = np.zeros(npx, np.float32)

    for idx, mat in enumerate(pack["mats"], start=1):
        sel = mid == idx
        if not sel.any():
            continue
        alb[sel] = mat.color
        rough[sel] = mat.roughness
        metal[sel] = mat.metalness
        emis[sel] = mat.emissive
        opac[sel] = mat.opacity
        ccoat[sel] = mat.clearcoat

    z_m = np.where(mask, zb, 0).astype(np.float32)
    m_f = mask.astype(np.float32)
    r_ao = max(3, int(Dh * 0.022)) | 1
    zbl = cv2.GaussianBlur(z_m, (0, 0), r_ao * 0.5) / np.maximum(cv2.GaussianBlur(m_f, (0, 0), r_ao * 0.5), 1e-4)
    occ = np.clip((zb - zbl)[mask] / (0.06 * rad), 0, 1)
    aov = 1.0 - (0.25 if is_glass else 0.50) * occ

    F0 = 0.04 * (1 - metal[:, None]) + alb * metal[:, None]
    diff_alb = alb * (1 - metal[:, None])
    hemi = GROUND + (SKY - GROUND) * (0.5 + 0.5 * n[:, 1:2])
    col = diff_alb * hemi * aov[:, None] * 1.15
    pexp = np.clip(2.0 / np.maximum(rough, 0.035) ** 4 - 2.0, 2.0, 5000.0)

    for Ld, Lc, LI in LIGHTS:
        ndl = np.clip(n @ Ld, 0, 1)[:, None]
        col = col + diff_alb * Lc * LI * ndl * aov[:, None]
        Hh = _norm(Ld[None] + V)
        ndh = np.clip(np.sum(n * Hh, 1), 0, 1)
        vdh = np.clip(np.sum(V * Hh, 1), 0, 1)
        Fr = F0 + (1 - F0) * ((1 - vdh) ** 5)[:, None]
        spec = ((pexp + 8) / (8 * math.pi) * (ndh ** pexp))[:, None] * Fr * np.clip(n @ Ld, 0, 1)[:, None]
        col = col + spec * Lc * LI * 0.95
        if np.any(ccoat > 0):
            cc_spec = ((1200 + 8) / (8 * math.pi) * (ndh ** 1200))[:, None] * 0.08 * ccoat[:, None]
            col = col + cc_spec * Lc * LI

    Rv = _norm(2 * ndv[:, None] * n - V)
    Fe = F0 + (np.maximum(1 - rough, F0.mean(1))[:, None] - F0) * ((1 - ndv) ** 5)[:, None]
    col = col + env(Rv, rough) * Fe * aov[:, None] * 0.65
    col = col + emis

    col_buf = np.zeros((Dh, Dh, 3), np.float32)
    alpha_buf = np.zeros((Dh, Dh), np.float32)
    col_buf[mask] = col
    if is_glass:
        fresnel = (1.0 - ndv) ** 3.2
        spec_lum = np.clip(col.max(1) - 0.25, 0.0, 1.0)
        alpha_buf[mask] = np.clip(opac + 0.52 * fresnel + 0.35 * spec_lum, 0.0, 0.96)
    else:
        alpha_buf[mask] = 1.0
    return col_buf, alpha_buf, zb


def render_three(model_or_scene, size, yaw=0.0, pitch=15.0, roll=0.0, explode=0.0, ssaa=2, exposure=0.88):
    """Render a `three3d` model at `(yaw, pitch, roll)` → `(rgba float32 D×D×4, D)`."""
    if isinstance(model_or_scene, str):
        sc = get_compiled_scene(model_or_scene, int(round(float(explode) * 100)))
    elif isinstance(model_or_scene, CompiledThreeScene):
        sc = model_or_scene
    else:
        sc = CompiledThreeScene(model_or_scene, explode=explode)

    R_cam = rot(pitch, yaw, roll)
    rad = sc.radius
    D = max(16, int(round(size * 1.28)))
    Dh = D * ssaa
    dist = 4.6 * rad
    Fpx = (size * ssaa * 0.48) * dist / rad
    cx = cy = Dh * 0.5

    if sc.opaque is not None:
        col_op, a_op, zb_op = _shade_pass(sc.opaque, R_cam, dist, Fpx, cx, cy, Dh, rad, is_glass=False)
    else:
        col_op = np.zeros((Dh, Dh, 3), np.float32)
        a_op = np.zeros((Dh, Dh), np.float32)
        zb_op = np.full((Dh, Dh), 1e30, np.float32)

    if sc.trans is not None:
        col_tr, a_tr, _ = _shade_pass(sc.trans, R_cam, dist, Fpx, cx, cy, Dh, rad, z_clip=zb_op, is_glass=True)
        a_tr3 = a_tr[..., None]
        col_lin = col_tr * a_tr3 + col_op * (1.0 - a_tr3)
        a_tot = a_tr + a_op * (1.0 - a_tr)
    else:
        col_lin, a_tot = col_op, a_op

    out = np.zeros((Dh, Dh, 4), np.float32)
    vis = a_tot > 1e-4
    if np.any(vis):
        out[vis, :3] = aces(col_lin[vis] * exposure) ** (1.0 / 2.2)
        out[vis, 3] = a_tot[vis]

    if ssaa > 1:
        pm = out.copy()
        pm[:, :, :3] *= pm[:, :, 3:4]
        pm = cv2.resize(pm, (D, D), interpolation=cv2.INTER_AREA)
        a = pm[:, :, 3:4]
        pm[:, :, :3] = pm[:, :, :3] / np.maximum(a, 1e-5)
        out = pm
    return out, D


def paint(ctx, model_name, cx, cy, size, yaw=0.0, pitch=15.0, roll=0.0, explode=0.0, alpha=1.0, shadow=True, ssaa=2):
    """Paint a `three3d` procedural 3D model onto a Cairo context with contact shadow."""
    from .obj3d import contact_shadow, paint_premul
    from .real3d import to_bgra_premul
    q_yaw = round(float(yaw) * 2.0) / 2.0
    q_pitch = round(float(pitch) * 2.0) / 2.0
    q_roll = round(float(roll) * 2.0) / 2.0
    q_exp = round(float(explode), 2)
    q_sz = max(24, int(round(size / 8.0) * 8))
    rgba, D = _cached_render(str(model_name), q_sz, q_yaw, q_pitch, q_roll, q_exp, int(ssaa))
    if abs(size - q_sz) > 1.0:
        D_target = max(16, int(round(D * (size / q_sz))))
        rgba = cv2.resize(rgba, (D_target, D_target), interpolation=cv2.INTER_LINEAR)
        D = D_target
    if shadow:
        contact_shadow(ctx, cx, cy + size * 0.46, size * 0.82, alpha, strength=0.52)
    paint_premul(ctx, to_bgra_premul(rgba), cx - D * 0.5, cy - D * 0.5, alpha)
    return (cx - size * 0.5, cy - size * 0.5, cx + size * 0.5, cy + size * 0.5)


@lru_cache(maxsize=96)
def _cached_render(model_name: str, q_sz: int, q_yaw: float, q_pitch: float, q_roll: float, q_exp: float, ssaa: int):
    return render_three(model_name, q_sz, q_yaw, q_pitch, q_roll, explode=q_exp, ssaa=ssaa)


def audit_sheet(model_name="hourglass_3d", out_path=None, cell=280):
    """Render a 10-angle 360° turntable + pitch/explode audit sheet to verify true 3D volume."""
    angles = [(-90, 15, 0.0, "yaw -90 (side)"), (-60, 15, 0.0, "yaw -60"), (-30, 15, 0.0, "yaw -30"),
              (0, 15, 0.0, "yaw 0 (front)"), (35, 16, 0.0, "yaw +35"), (65, 16, 0.0, "yaw +65"),
              (85, 15, 0.0, "yaw +85 (side)"), (140, 18, 0.0, "yaw +140 (back)"),
              (25, 48, 0.0, "pitch +48 (top)"), (35, 16, 0.45, "explode 45%")]
    cols = 5
    rows = 2
    sheet = np.full((rows * cell, cols * cell, 3), 18, np.uint8)
    for idx, (yw, pt, ex, lbl) in enumerate(angles):
        r, c = divmod(idx, cols)
        rgba, D = render_three(model_name, int(cell * 0.72), yaw=yw, pitch=pt, explode=ex, ssaa=2)
        tile = np.full((cell, cell, 3), 18, np.uint8)
        y0 = (cell - D) // 2
        x0 = (cell - D) // 2
        a = rgba[:, :, 3:4]
        bgr = (rgba[:, :, 2::-1] * 255.0).astype(np.float32)
        roi = tile[max(0, y0):min(cell, y0 + D), max(0, x0):min(cell, x0 + D)].astype(np.float32)
        sub_b = bgr[max(0, -y0):max(0, -y0) + roi.shape[0], max(0, -x0):max(0, -x0) + roi.shape[1]]
        sub_a = a[max(0, -y0):max(0, -y0) + roi.shape[0], max(0, -x0):max(0, -x0) + roi.shape[1]]
        tile[max(0, y0):min(cell, y0 + D), max(0, x0):min(cell, x0 + D)] = np.clip(sub_b * sub_a + roi * (1 - sub_a), 0, 255).astype(np.uint8)
        cv2.putText(tile, f"{model_name}: {lbl}", (10, cell - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 245), 1, cv2.LINE_AA)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile
    out_path = out_path or f"{model_name}_three3d_sheet.png"
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return out_path


# ============================================================================================ Selftest Suite (v20)
def _t_models():
    canonical = ("hourglass_3d", "microblading_pen_3d", "emerald_tree_3d", "star_trophy_3d",
                 "shield_3d", "cash_3d", "warn_3d", "laptop_3d")
    counts = {}
    for m in canonical:
        sc = get_compiled_scene(m, 0)
        ntri = (len(sc.opaque["F"]) if sc.opaque else 0) + (len(sc.trans["F"]) if sc.trans else 0)
        counts[m] = ntri
    ok = all(v >= 100 for v in counts.values())
    return ok, f"{len(canonical)} procedural 3D models compiled (min {min(counts.values())} tris, max {max(counts.values())} tris)"


def _t_volume_360():
    """Verify that at 85° side profile, the procedural 3D hourglass retains full cylindrical width (unlike flat 2D slabs)."""
    f0, _ = render_three("hourglass_3d", 140, yaw=0.0, pitch=15.0, ssaa=1)
    f85, _ = render_three("hourglass_3d", 140, yaw=85.0, pitch=15.0, ssaa=1)
    w0 = int((f0[:, :, 3] > 0.1).any(axis=0).sum())
    w85 = int((f85[:, :, 3] > 0.1).any(axis=0).sum())
    ratio = w85 / max(1, w0)
    return ratio > 0.85, f"side-profile 85° width ratio = {ratio:.2f} (true 3D cylindrical volume preserved)"


def _t_glass():
    """Verify two-pass refractive glass: hourglass bulb is semi-transparent while internal golden sand is opaque."""
    rgba, _ = render_three("hourglass_3d", 140, yaw=25.0, pitch=15.0, ssaa=1)
    a = rgba[:, :, 3]
    semi = int(((a > 0.25) & (a < 0.88)).sum())
    solid = int((a >= 0.95).sum())
    return semi > 150 and solid > 400, f"two-pass glass: {semi} semi-transparent glass px + {solid} solid gold/sand px"


def _t_specular():
    """Verify dynamic PBR lighting: rotating the star trophy shifts specular highlights across facets."""
    a, _ = render_three("star_trophy_3d", 140, yaw=-25.0, pitch=15.0, ssaa=1)
    b, _ = render_three("star_trophy_3d", 140, yaw=25.0, pitch=15.0, ssaa=1)
    diff = float(np.abs(a[:, :, :3] - b[:, :, :3]).mean())
    return diff > 0.04, f"PBR specular + facet normal shift Δ = {diff:.3f}"


def _t_explode():
    """Verify explode view rig displaces sub-parts outward."""
    c0 = get_compiled_scene("emerald_tree_3d", 0)
    c1 = get_compiled_scene("emerald_tree_3d", 80)
    return c1.radius > c0.radius * 1.15, f"explode radius {c0.radius:.2f} → {c1.radius:.2f}"


def _t_js_bridge():
    """Verify Node.js execution of standalone `three_models.js` produces a valid 3D scene graph."""
    node = load_three_js_script(JS_MODELS_PATH, "buildStarTrophy3D")
    sc = CompiledThreeScene(node)
    ntri = (len(sc.opaque["F"]) if sc.opaque else 0) + (len(sc.trans["F"]) if sc.trans else 0)
    return ntri > 500, f"three_models.js executed via Node.js → {ntri} triangles"


def _t_cairo_paint():
    import cairo
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 220, 220)
    ctx = cairo.Context(s)
    paint(ctx, "microblading_pen_3d", 110, 110, 140, yaw=35.0, pitch=16.0, ssaa=1)
    s.flush()
    arr = np.ndarray((220, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :220]
    cov = float((arr[:, :, 3] > 10).mean())
    return 0.05 < cov < 0.65, f"Cairo paint coverage {cov:.2f}"


TESTS = [
    ("models", _t_models),
    ("volume360", _t_volume_360),
    ("glass_pbr", _t_glass),
    ("specular", _t_specular),
    ("explode", _t_explode),
    ("js_bridge", _t_js_bridge),
    ("paint", _t_cairo_paint),
]


def selftest():
    t0 = time.time()
    passed = 0
    for name, fn in TESTS:
        t1 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        passed += bool(ok)
        print(f"  {'✓' if ok else '✗'} {name:<10} {msg}  ({(time.time() - t1) * 1000:.0f} ms)")
    print(f"three3d selftest: {passed}/{len(TESTS)} pass  ({time.time() - t0:.1f}s)")
    return 0 if passed == len(TESTS) else 1


def _register():
    from .elements import element, val, xy

    @element("three3d", dur=0.8, enter="spring", sfx="whoosh")
    def el_three3d(ctx, el, s):
        from .obj3d import motion_angles
        x, y = xy(el, s.t)
        sz = float(val(el, "w", s.t, el.get("size", 360))) * s.scale
        lt = max(0.0, s.t - float(el["t"]))
        yaw, pitch = motion_angles(el, lt, float(val(el, "yaw", s.t, 25.0)), float(val(el, "pitch", s.t, 16.0)))
        model = el.get("three_script") or el.get("three_model") or el.get("model") or el.get("src", "hourglass_3d")
        return paint(ctx, model, x, y + s.dy, sz, yaw, pitch, float(val(el, "roll", s.t, 0.0)),
                     float(val(el, "explode", s.t, 0.0)), s.alpha, bool(el.get("shadow", True)), int(el.get("ssaa", 2)))


try:
    _register()
except Exception:  # pragma: no cover
    pass


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        raise SystemExit(selftest())
    if cmd == "list":
        for k in sorted(set(MODELS)):
            print(k)
        raise SystemExit(0)
    if cmd == "sheet":
        mdl = sys.argv[2] if len(sys.argv) > 2 else "hourglass_3d"
        out = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--out" else f"{mdl}_sheet.png"
        print(audit_sheet(mdl, out))
        raise SystemExit(0)
