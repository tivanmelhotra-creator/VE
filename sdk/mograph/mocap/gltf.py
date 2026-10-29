"""
mograph.mocap.gltf — tiny pure-numpy glTF 2.0 / GLB skeleton + animation reader (no three.js, no Blender).

Only what motion capture needs: the node hierarchy (rest TRS), skins (joint list), animation channels
(translation / rotation / scale, LINEAR / STEP / CUBICSPLINE) sampled at any time. Meshes are ignored — the
drawn body is our own MakeHuman character, the GLB only carries the MOTION.

    sk = load("walk.glb")
    sk.clips                         # {"walk": Clip, ...}
    W = sk.world_at("walk", 0.4)     # {joint_name: 4x4 world matrix}   (metres, Y up, as in the file)
"""
from __future__ import annotations

import base64
import json
import os
import struct
from dataclasses import dataclass, field

import numpy as np

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def quat_to_mat(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 1e-12 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]], np.float64)


def slerp(a, b, u):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + (b - a) * u
        return r / np.linalg.norm(r)
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)


def trs(t=None, r=None, s=None):
    M = np.eye(4)
    if r is not None:
        M[:3, :3] = quat_to_mat(r)
    if s is not None:
        M[:3, :3] = M[:3, :3] * np.asarray(s, np.float64)[None, :]
    if t is not None:
        M[:3, 3] = t
    return M


@dataclass
class Channel:
    node: int
    path: str
    times: np.ndarray
    values: np.ndarray
    interp: str = "LINEAR"

    def sample(self, t):
        T = self.times
        V = self.values
        cubic = self.interp == "CUBICSPLINE"
        if cubic:                                     # [in-tangent, value, out-tangent] per key → values only
            V = V.reshape(len(T), 3, -1)[:, 1]
        if t <= T[0]:
            return V[0]
        if t >= T[-1]:
            return V[-1]
        i = int(np.searchsorted(T, t) - 1)
        u = (t - T[i]) / max(1e-9, T[i + 1] - T[i])
        if self.interp == "STEP":
            return V[i]
        if self.path == "rotation":
            return slerp(V[i], V[i + 1], u)
        return V[i] + (V[i + 1] - V[i]) * u


@dataclass
class Clip:
    name: str
    channels: list
    duration: float


@dataclass
class Skeleton:
    names: list
    parent: np.ndarray
    rest_t: np.ndarray            # (N,3)
    rest_r: np.ndarray            # (N,4) xyzw
    rest_s: np.ndarray            # (N,3)
    joints: list                  # node ids that are skin joints (or all bone-like nodes)
    clips: dict = field(default_factory=dict)
    source: str = ""

    def index(self, name):
        return self.names.index(name)

    def local_mats(self, clip=None, t=0.0, T=None, R=None, S=None):
        Tn = self.rest_t.copy() if T is None else T
        Rn = self.rest_r.copy() if R is None else R
        Sn = self.rest_s.copy() if S is None else S
        if clip is not None:
            c = self.clips[clip] if isinstance(clip, str) else clip
            for ch in c.channels:
                v = ch.sample(t)
                if ch.path == "translation":
                    Tn[ch.node] = v
                elif ch.path == "rotation":
                    Rn[ch.node] = v
                elif ch.path == "scale":
                    Sn[ch.node] = v
        return [trs(Tn[i], Rn[i], Sn[i]) for i in range(len(self.names))]

    def world(self, local):
        N = len(self.names)
        Wm = [None] * N
        order = _topo(self.parent)
        for i in order:
            p = self.parent[i]
            Wm[i] = local[i] if p < 0 else Wm[p] @ local[i]
        return Wm

    def world_at(self, clip=None, t=0.0):
        Wm = self.world(self.local_mats(clip, t))
        return {self.names[i]: Wm[i] for i in self.joints}

    def rest_world(self):
        return self.world_at(None, 0.0)


def _topo(parent):
    N = len(parent)
    depth = np.zeros(N, int)
    for i in range(N):
        p, d = parent[i], 0
        while p >= 0:
            d += 1
            p = parent[p]
        depth[i] = d
    return list(np.argsort(depth, kind="stable"))


def _read(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] == b"glTF":
        L = struct.unpack("<I", data[12:16])[0]
        js = json.loads(data[20:20 + L])
        off = 20 + L
        bin_ = b""
        if off < len(data):
            bl = struct.unpack("<I", data[off:off + 4])[0]
            bin_ = data[off + 8:off + 8 + bl]
        return js, [bin_]
    js = json.loads(data)
    bufs = []
    for b in js.get("buffers", []):
        uri = b.get("uri", "")
        if uri.startswith("data:"):
            bufs.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            with open(os.path.join(os.path.dirname(path), uri), "rb") as f:
                bufs.append(f.read())
    return js, bufs


def _accessor(js, bufs, i):
    a = js["accessors"][i]
    bv = js["bufferViews"][a["bufferView"]]
    buf = bufs[bv.get("buffer", 0)]
    dt = np.dtype(_CT[a["componentType"]])
    nc = _NC[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or dt.itemsize * nc
    n = a["count"]
    if stride == dt.itemsize * nc:
        arr = np.frombuffer(buf, dt, n * nc, start).reshape(n, nc)
    else:
        raw = np.frombuffer(buf, np.uint8, stride * (n - 1) + dt.itemsize * nc, start)
        arr = np.stack([np.frombuffer(raw[k * stride:k * stride + dt.itemsize * nc].tobytes(), dt) for k in range(n)])
    arr = arr.astype(np.float64)
    if a.get("normalized") and dt != np.float32:
        arr = arr / float(np.iinfo(dt).max)
    return arr if nc > 1 else arr[:, 0]


def load(path) -> Skeleton:
    js, bufs = _read(path)
    nodes = js["nodes"]
    N = len(nodes)
    parent = np.full(N, -1, int)
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    T = np.zeros((N, 3))
    R = np.tile([0, 0, 0, 1.0], (N, 1))
    S = np.ones((N, 3))
    for i, n in enumerate(nodes):
        if "matrix" in n:
            M = np.asarray(n["matrix"], np.float64).reshape(4, 4).T
            T[i] = M[:3, 3]
            sc = np.linalg.norm(M[:3, :3], axis=0)
            S[i] = sc
            from scipy.spatial.transform import Rotation
            R[i] = Rotation.from_matrix(M[:3, :3] / sc).as_quat()
        else:
            T[i] = n.get("translation", [0, 0, 0])
            R[i] = n.get("rotation", [0, 0, 0, 1])
            S[i] = n.get("scale", [1, 1, 1])
    names = [n.get("name", f"node{i}") for i, n in enumerate(nodes)]
    joints = sorted({j for s in js.get("skins", []) for j in s["joints"]})
    if not joints:                                    # animation-only file: every node that is animated or named
        joints = [i for i, n in enumerate(names) if "mesh" not in nodes[i]]
    sk = Skeleton(names, parent, T, R, S, joints, source=path)
    for k, a in enumerate(js.get("animations", [])):
        chans = []
        dur = 0.0
        for ch in a["channels"]:
            tgt = ch["target"]
            if "node" not in tgt or tgt["path"] not in ("translation", "rotation", "scale"):
                continue
            smp = a["samplers"][ch["sampler"]]
            tt = _accessor(js, bufs, smp["input"])
            vv = _accessor(js, bufs, smp["output"])
            chans.append(Channel(tgt["node"], tgt["path"], tt, vv, smp.get("interpolation", "LINEAR")))
            dur = max(dur, float(tt[-1]))
        sk.clips[a.get("name", f"clip{k}")] = Clip(a.get("name", f"clip{k}"), chans, dur)
    return sk
