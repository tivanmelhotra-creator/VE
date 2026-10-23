"""
mograph.human.body — parametric MakeHuman body (shape) + skeleton + linear-blend skinning.

Units: DECIMETRES, Y-up, +Z = the character's front, +X = the character's LEFT.

    from mograph.human.body import Body
    b = Body(gender=1, age=30, muscle=0.6, weight=0.5, height=0.55, fat=0.0)
    G = b.pose_matrices({"upperarm01.L": R, ...})   # (163,3,4) global skinning matrices
    posed = b.skin(G)                                # (19158,3)

Shape = MakeHuman macro algorithm (gender × age × muscle × weight universal targets, race mix, height and proportion
targets) + local layers (fat / belly / shoulders / hips) so a heavy body really reads heavy.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from . import assets

BODY_N = 13380                         # vertices 0..13379 = skin; the rest = helpers (tights, skirt, hair, eyes…)


@lru_cache(maxsize=1)
def _base():
    if not assets.ready():
        assets.fetch()
    z = np.load(assets.BASE)
    meta = json.loads(bytes(z["meta"]).decode())
    nb = len(meta["bones"])
    V = z["verts"].astype(np.float32)
    W = np.zeros((V.shape[0], nb), np.float32)
    W[z["w_cols"], z["w_rows"].astype(np.int32)] = z["w_vals"]
    W /= np.maximum(W.sum(1, keepdims=True), 1e-6)
    return {"V": V, "F": z["faces"].astype(np.int32), "FG": z["face_group"].astype(np.int32),
            "parent": z["parent"].astype(np.int32), "W": W, "meta": meta}


class _Targets:
    def __init__(self):
        if not assets.ready():
            assets.fetch()
        self.z = np.load(assets.TARGETS)
        self.keys = {k[:-3] for k in self.z.files if k.endswith("__i")}
        self._c = {}

    def has(self, k):
        return k in self.keys

    def get(self, k):
        if k not in self._c:
            self._c[k] = (self.z[k + "__i"], self.z[k + "__d"])
        return self._c[k]


@lru_cache(maxsize=1)
def targets() -> _Targets:
    return _Targets()


def bones() -> list[str]:
    return _base()["meta"]["bones"]


@lru_cache(maxsize=1)
def _bone_lut():
    return {n: i for i, n in enumerate(bones())}


def bone_index(name: str) -> int:
    return _bone_lut()[name]


def group_vertices(name: str) -> np.ndarray:
    m = _base()["meta"]
    return np.asarray(m["vgroups"].get(name) or m["joint_cubes"][name], np.int32)


def group_faces(name: str) -> np.ndarray:
    b = _base()
    gi = b["meta"]["groups"].index(name)
    return b["F"][b["FG"] == gi]


# ------------------------------------------------------------------------------------------------ macro weights
def age_to_value(years: float) -> float:
    y = float(years)
    if y <= 11:
        return max(0.0, (y - 1) / 10 * 0.1875)
    if y <= 25:
        return 0.1875 + (y - 11) / 14 * 0.3125
    return min(1.0, 0.5 + (y - 25) / 65 * 0.5)


def _three(v):
    v = min(1.0, max(0.0, v))
    if v < 0.5:
        return {"min": 1 - v / 0.5, "average": v / 0.5, "max": 0.0}
    return {"min": 0.0, "average": 1 - (v - 0.5) / 0.5, "max": (v - 0.5) / 0.5}


def _age_w(a):
    if a < 0.1875:
        t = a / 0.1875
        return {"baby": 1 - t, "child": t, "young": 0.0, "old": 0.0}
    if a < 0.5:
        t = (a - 0.1875) / 0.3125
        return {"baby": 0.0, "child": 1 - t, "young": t, "old": 0.0}
    t = (a - 0.5) / 0.5
    return {"baby": 0.0, "child": 0.0, "young": 1 - t, "old": t}


def macro_mix(gender=0.5, age=0.5, muscle=0.5, weight=0.5, height=0.5, proportions=0.5,
              race=(1 / 3, 1 / 3, 1 / 3)) -> dict[str, float]:
    T = targets()
    out: dict[str, float] = {}
    G = {"female": 1 - gender, "male": gender}
    A = _age_w(age)
    Mu = _three(muscle)
    We = _three(weight)

    def put(key, k):
        if T.has(key):
            out[key] = out.get(key, 0) + k
    for g, wg in G.items():
        for a, wa in A.items():
            for m, wm in Mu.items():
                for w, ww in We.items():
                    k = wg * wa * wm * ww
                    if k <= 1e-5:
                        continue
                    put(f"macro/universal-{g}-{a}-{m}muscle-{w}weight", k)
                    if abs(height - 0.5) > 1e-3:
                        hk = "maxheight" if height > 0.5 else "minheight"
                        put(f"height/{g}-{a}-{m}muscle-{w}weight-{hk}", k * abs(height - 0.5) * 2)
                    if abs(proportions - 0.5) > 1e-3:
                        pk = "idealproportions" if proportions > 0.5 else "uncommonproportions"
                        put(f"proportions/{g}-{a}-{m}muscle-{w}weight-{pk}", k * abs(proportions - 0.5) * 2)
            for rname, wr in zip(("african", "asian", "caucasian"), race):
                if wg * wa * wr > 1e-5:
                    put(f"macro/{rname}-{g}-{a}", wg * wa * wr)
    return out


FAT_LAYER = {
    "stomach/stomach-pregnant-incr": 0.55, "torso/torso-scale-depth-incr": 0.55, "torso/torso-scale-horiz-incr": 0.35,
    "torso/measure-waist-circ-incr": 0.9, "torso/measure-hips-circ-incr": 0.35, "hip/hip-scale-horiz-incr": 0.35,
    "hip/hip-scale-depth-incr": 0.4, "buttocks/buttocks-volume-incr": 0.4, "neck/neck-double-incr": 0.7,
    "neck/measure-neck-circ-incr": 0.5, "arms/l-upperarm-fat-incr": 0.8, "arms/r-upperarm-fat-incr": 0.8,
    "arms/l-lowerarm-fat-incr": 0.5, "arms/r-lowerarm-fat-incr": 0.5, "legs/l-upperleg-fat-incr": 0.6,
    "legs/r-upperleg-fat-incr": 0.6, "legs/l-lowerleg-fat-incr": 0.45, "legs/r-lowerleg-fat-incr": 0.45,
    "head/head-fat-incr": 0.6,
}
THIN_LAYER = {"torso/measure-waist-circ-decr": 0.6, "stomach/stomach-pregnant-decr": 0.4,
              "hip/hip-scale-horiz-decr": 0.25, "torso/torso-scale-depth-decr": 0.3}
SIGNED = {
    "belly": ({"stomach/stomach-pregnant-incr": 1.0}, {"stomach/stomach-pregnant-decr": 1.0}),
    "shoulders": ({"torso/measure-shoulder-dist-incr": 0.8, "torso/torso-vshape-incr": 0.6},
                  {"torso/measure-shoulder-dist-decr": 0.8}),
    "hips": ({"hip/hip-scale-horiz-incr": 0.7, "torso/measure-hips-circ-incr": 0.5,
              "buttocks/buttocks-volume-incr": 0.4},
             {"hip/hip-scale-horiz-decr": 0.7, "torso/measure-hips-circ-decr": 0.5}),
}


def _layer(spec, amount, out):
    T = targets()
    for k, w in spec.items():
        if T.has(k):
            out[k] = out.get(k, 0) + w * amount


class Body:
    """gender 0 (female) … 1 (male) · age in YEARS (1 … 90) · muscle / weight / height / proportions 0…1
    · fat −1…1 (soft volume layer) · belly / shoulders / hips −1…1 · extra {target_key: weight}"""

    def __init__(self, gender=0.5, age=25, muscle=0.5, weight=0.5, height=0.5, proportions=0.5,
                 fat=0.0, belly=0.0, shoulders=0.0, hips=0.0, race=(1 / 3, 1 / 3, 1 / 3), extra=None):
        b = _base()
        self.params = dict(gender=gender, age=age, muscle=muscle, weight=weight, height=height,
                           proportions=proportions, fat=fat, belly=belly, shoulders=shoulders, hips=hips)
        mix = macro_mix(gender, age_to_value(age), muscle, weight, height, proportions, race)
        if fat > 0:
            _layer(FAT_LAYER, fat, mix)
        elif fat < 0:
            _layer(THIN_LAYER, -fat, mix)
        for key, v in (("belly", belly), ("shoulders", shoulders), ("hips", hips)):
            if v:
                _layer(SIGNED[key][0 if v > 0 else 1], abs(v), mix)
        for k, w in (extra or {}).items():
            if targets().has(k):
                mix[k] = mix.get(k, 0) + w
        V = b["V"].copy()
        T = targets()
        for k, w in mix.items():
            if abs(w) > 1e-5:
                i, d = T.get(k)
                V[i] += d * w
        V[:, 1] -= V[:BODY_N, 1].min()                  # soles on y = 0
        self.rest = V
        self.mix = mix
        m = b["meta"]
        self.heads = np.array([V[h].mean(0) for h in m["heads"]], np.float32)
        self.tails = np.array([V[t].mean(0) for t in m["tails"]], np.float32)
        self.parent = b["parent"]
        self.W = b["W"]
        self.F = b["F"]
        self.FG = b["FG"]

    @property
    def height_cm(self) -> float:
        return float(np.ptp(self.rest[:BODY_N, 1])) * 10

    def head(self, name):
        return self.heads[bone_index(name)]

    def tail(self, name):
        return self.tails[bone_index(name)]

    def bone_dir(self, name):
        i = bone_index(name)
        d = self.tails[i] - self.heads[i]
        return d / (np.linalg.norm(d) + 1e-9)

    def pose_matrices(self, rot: dict | None = None, root_offset=(0, 0, 0), root_rot=None) -> np.ndarray:
        """rot: bone → 3×3 rotation in REST WORLD axes about the bone head (children inherit).
        root_rot rotates the whole body about the pelvis; root_offset translates it (dm)."""
        rot = rot or {}
        lut = _bone_lut()
        B = len(self.heads)
        Rl = np.repeat(np.eye(3, dtype=np.float32)[None], B, 0)
        for n, R in rot.items():
            if n in lut:
                Rl[lut[n]] = R
        G = np.zeros((B, 3, 4), np.float32)
        RR = np.eye(3, dtype=np.float32) if root_rot is None else np.asarray(root_rot, np.float32)
        off = np.asarray(root_offset, np.float32)
        pivot = self.heads[lut["spine05"]]
        for i in range(B):
            h = self.heads[i]
            p = self.parent[i]
            if p < 0:
                PR, Pt = RR, pivot - RR @ pivot + off
            else:
                PR, Pt = G[p, :, :3], G[p, :, 3]
            G[i, :, :3] = PR @ Rl[i]
            G[i, :, 3] = PR @ (h - Rl[i] @ h) + Pt
        return G

    def skin(self, G: np.ndarray, verts=None, W=None) -> np.ndarray:
        V = self.rest if verts is None else verts
        W = self.W if W is None else W
        M = (W @ G.reshape(len(G), 12)).reshape(-1, 3, 4)
        return np.einsum("nij,nj->ni", M[:, :, :3], V) + M[:, :, 3]

    def joint(self, G, name):
        i = bone_index(name)
        return G[i, :, :3] @ self.heads[i] + G[i, :, 3]


def rot_axis(axis, deg) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y, z = a
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], np.float32)


def rot_xyz(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    """World-axis rotation, degrees: X first (pitch, +x = forward-bend for limbs pointing down… see poses.py),
    then Y (yaw), then Z (roll in the frontal plane)."""
    return rot_axis((0, 0, 1), z) @ rot_axis((0, 1, 0), y) @ rot_axis((1, 0, 0), x)
