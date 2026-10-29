"""
mograph.mocap.retarget — REAL motion capture (Mixamo GLB / glTF / BVH) → our MakeHuman line-art body.

Why: hand-written pose curves (poses.walk …) always read as a puppet. A mocap clip carries what a human really
does: pelvis bounce + sway + rotation, weight shift, counter-rotation of shoulders, overlapping arms. This module
moves those rotations onto the MakeHuman skeleton, so the same line renderer (mograph.human.lineart) draws a
figure that walks / talks / dances like a person.

Method (world-delta retarget, "bind-pose retargeting"):
    D_i(t) = Rw_src_i(t) · Rw_src_i(rest)^-1          world rotation delta of a source bone vs its rest (T-pose)
    C_i    = rmin(dir_mh_i(rest), dir_src_i(rest))     rest correction: MakeHuman A-pose segment → source T-pose
    Q_i(t) = D_i(t) · C_i                              global rotation of the MakeHuman bone (rest world axes)
  · twist is kept (D is a full rotation), segment directions match exactly.
  · source facing is normalised (any file whose character faces −Z / ±X is rotated to face +Z, left = +X).
  · HIPS are FREE: the pelvis rotation drives `root`, the pelvis translation drives the whole body (scaled by the
    leg-length ratio), so the vertical bounce and lateral sway of a real gait survive.
  · ground: one constant offset for the whole clip (lowest foot over the clip touches y = 0) → jumps survive,
    no per-frame "snap to floor" jitter.
  · root motion: "file" (as recorded), "in_place" (horizontal drift removed), "plant" (the body is advanced so the
    planted foot never slides — zero foot skating for in-place Mixamo cycles, speed derived from the clip itself).

    from mograph.mocap.retarget import MocapMotion
    mo = MocapMotion("walk.glb", clip="walk", body=character.body, root_motion="plant")
    G, X = mo.pose(t)           # skinning matrices + posed vertices for mograph.human.lineart
"""
from __future__ import annotations

import math
import os
import re

import numpy as np

from ..human.body import Body, BODY_N, bone_index, bones
from ..human.poses import rmin, foot_vertices
from .gltf import Skeleton, load as load_gltf

# --------------------------------------------------------------------------------------------- naming
# canonical names (Mixamo style). The side is parsed first (Left/Right/L/R prefix or .L/_L/.l suffix), then the
# base name is matched against aliases that cover Mixamo, Rigify (Quaternius, Blender), CMU / Truebones BVH, UE.
BASE = {
    "Hips": ["hips", "pelvis", "hip"],
    "Spine": ["spine", "spine001", "abdomen", "lowerback", "spine01", "spine0"],
    "Spine1": ["spine1", "spine002", "chest", "spine02"],
    "Spine2": ["spine2", "spine003", "upperchest", "chest2", "spine03"],
    "Neck": ["neck", "neck1", "neck01", "neck001"],
    "Head": ["head"],
}
SIDED = {
    "Shoulder": ["shoulder", "collar", "clavicle", "clav"],
    "Arm": ["arm", "upperarm", "shldr", "upper_arm", "uparm"],
    "ForeArm": ["forearm", "lowerarm", "elbow", "foreArm"],
    "Hand": ["hand", "wrist"],
    "UpLeg": ["upleg", "thigh", "upperleg", "hip", "femur"],
    "Leg": ["leg", "shin", "lowerleg", "knee", "calf", "crus"],
    "Foot": ["foot", "ankle"],
    "ToeBase": ["toebase", "toe", "toes", "ball"],
}
FING = {"Thumb": ["thumb"], "Index": ["index", "findex", "fingerindex", "pointer"], "Middle": ["middle", "fmiddle"],
        "Ring": ["ring", "fring"], "Pinky": ["pinky", "fpinky", "little", "fpinkie", "pinkie"]}


def _side(n):
    m = re.match(r"^(left|right)(.+)$", n, re.I)
    if m:
        return ("Left" if m.group(1).lower() == "left" else "Right"), m.group(2)
    m = re.match(r"^(.+?)[._\-]?(left|right)$", n, re.I)
    if m:
        return ("Left" if m.group(2).lower() == "left" else "Right"), m.group(1)
    m = re.match(r"^([lr]) (.+)$", n, re.I)              # 3ds Max biped: "L Thigh"
    if m:
        return ("Left" if m.group(1).lower() == "l" else "Right"), m.group(2)
    m = re.match(r"^(.+?)[._\-]([lr])$", n, re.I)
    if m:
        return ("Left" if m.group(2).lower() == "l" else "Right"), m.group(1)
    m = re.match(r"^([lr])[._\-]?([A-Z_].+|[a-z]+)$", n)
    if m and len(n) > 3 and m.group(2).lower().rstrip("0123456789") in \
            {a for al in list(SIDED.values()) + [["hand"]] for a in al} | {"shldr", "collar", "thigh", "shin", "foot",
                                                                            "hand", "forearm", "toe"}:
        return ("Left" if m.group(1).lower() == "l" else "Right"), m.group(2)
    return None, n


def canon(name: str) -> str:
    n = re.sub(r"^.*[:|]", "", name)                  # mixamorig:Hips → Hips,  rig|Hips → Hips
    n = re.sub(r"^(mixamorig\d*|bip0?1[ _]?|def[-_]|org[-_]|b[-_]|bone[-_]|j[-_]|jnt[-_])", "", n, flags=re.I)
    side, base = _side(n)
    k = re.sub(r"[\s_.\-]", "", base).lower()
    if side is None:
        for c in BASE:                                 # exact canonical names win (Mixamo "Spine" ≠ an alias)
            if k == c.lower():
                return c
        for c, al in BASE.items():
            if k in al:
                return c
        return n
    mf = re.match(r"^(?:hand)?(thumb|index|middle|ring|pinky|pinkie|little|f(?:index|middle|ring|pinky))(\d+)", k)
    if mf:
        key = mf.group(1)
        for F, al in FING.items():
            if key == F.lower() or key in al:
                d = int(mf.group(2))
                d = max(1, d)
                return f"{side}Hand{F}{min(3, d)}"
    for c, al in SIDED.items():
        if k == c.lower() or k in al:
            return f"{side}{c}"
    return f"{side}:{k}"


# MakeHuman bone(s) ← canonical source bone.  A tuple = one SEGMENT (head of first → tail of last bone).
SEG_MAP = [
    (("root",), "Hips"),
    (("spine05", "spine04"), "Spine"),
    (("spine03", "spine02"), "Spine1"),
    (("spine01",), "Spine2"),
    (("neck01", "neck02", "neck03"), "Neck"),
    (("head",), "Head"),
]
for s, S in (("L", "Left"), ("R", "Right")):
    SEG_MAP += [
        ((f"clavicle.{s}", f"shoulder01.{s}"), f"{S}Shoulder"),
        ((f"upperarm01.{s}", f"upperarm02.{s}"), f"{S}Arm"),
        ((f"lowerarm01.{s}", f"lowerarm02.{s}"), f"{S}ForeArm"),
        ((f"wrist.{s}",), f"{S}Hand"),
        ((f"upperleg01.{s}", f"upperleg02.{s}"), f"{S}UpLeg"),
        ((f"lowerleg01.{s}", f"lowerleg02.{s}"), f"{S}Leg"),
        ((f"foot.{s}",), f"{S}Foot"),
    ]
    for fi, F in enumerate(("Thumb", "Index", "Middle", "Ring", "Pinky")):
        for k in (1, 2, 3):
            SEG_MAP.append(((f"finger{fi + 1}-{k}.{s}",), f"{S}Hand{F}{k}"))

# direction of a source bone = its joint → this child joint (first one present)
SRC_CHILD = {"Hips": ["Spine", "Spine1", "Spine2", "Neck"], "Spine": ["Spine1", "Spine2", "Neck"],
             "Spine1": ["Spine2", "Neck"], "Spine2": ["Neck", "Head"],
             "Neck": ["Head"], "Head": [None]}
for S in ("Left", "Right"):
    SRC_CHILD.update({f"{S}Shoulder": [f"{S}Arm"], f"{S}Arm": [f"{S}ForeArm"], f"{S}ForeArm": [f"{S}Hand"],
                      f"{S}Hand": [f"{S}HandMiddle1", f"{S}HandIndex1"], f"{S}UpLeg": [f"{S}Leg"],
                      f"{S}Leg": [f"{S}Foot"], f"{S}Foot": [f"{S}ToeBase"], f"{S}ToeBase": [None]})
    for F in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        SRC_CHILD[f"{S}Hand{F}1"] = [f"{S}Hand{F}2"]
        SRC_CHILD[f"{S}Hand{F}2"] = [f"{S}Hand{F}3"]
        SRC_CHILD[f"{S}Hand{F}3"] = [f"{S}Hand{F}4", None]


def _ortho(M):
    U, _, Vt = np.linalg.svd(M[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _yaw(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


# --------------------------------------------------------------------------------------------- loader
_CACHE = {}


def load_motion(path) -> Skeleton:
    path = os.path.expanduser(path)
    if path not in _CACHE:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".glb", ".gltf"):
            sk = load_gltf(path)
        elif ext == ".bvh":
            from .bvh import load as load_bvh
            sk = load_bvh(path)
        else:
            raise ValueError(f"mocap: unsupported file {path} (use .glb / .gltf / .bvh; convert FBX with FBX2glTF)")
        _CACHE[path] = sk
    return _CACHE[path]


class SourceRig:
    """A loaded mocap skeleton seen through canonical names, facing-normalised, in metres."""

    def __init__(self, sk: Skeleton, rest_clip=None):
        self.sk = sk
        self.cmap = {}
        for i in sk.joints:
            c = canon(sk.names[i])
            if c not in self.cmap:
                self.cmap[c] = i
        self._structural_spine()
        nfing = sum(1 for k in self.cmap if "Hand" in k and k[-1].isdigit())
        self.has_fingers = nfing >= 20                   # CMU-style stub hands → keep MakeHuman's relaxed hand
        missing = [b for b in ("Hips", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg") if b not in self.cmap]
        if missing:
            raise ValueError(f"mocap: skeleton has no {missing} (names: {[sk.names[i] for i in sk.joints][:12]}…)")
        Wr = self._world(rest_clip, 0.0, raw=True)
        # facing: left vector from the arms (fallback legs), up = +Y
        lv = Wr["LeftArm"][:3, 3] - Wr["RightArm"][:3, 3]
        lv[1] = 0
        if np.linalg.norm(lv) < 1e-6:
            lv = Wr["LeftUpLeg"][:3, 3] - Wr["RightUpLeg"][:3, 3]
            lv[1] = 0
        ang = math.degrees(math.atan2(-lv[2], lv[0]))            # rotate so left → +X
        self.F = _yaw(ang)
        self.rest = {k: self._apply_F(M) for k, M in Wr.items()}
        self.rest_R = {k: _ortho(M) for k, M in self.rest.items()}
        hips = self.rest["Hips"][:3, 3]
        feet = [self.rest[k][:3, 3] for k in ("LeftFoot", "RightFoot") if k in self.rest]
        self.floor = min(f[1] for f in feet) if feet else 0.0
        self.leg_len = hips[1] - self.floor

    def _structural_spine(self):
        """Spine / neck by HIERARCHY, not by name: the joints on the path hips → head are split into the spine
        (up to the clavicles' parent) and the neck. Handles CMU (LowerBack, Spine, Spine1, Neck, Neck1), Mixamo,
        Rigify (spine.001-.003), UE (spine_01-05). The last neck joint carries the full neck rotation."""
        sk = self.sk
        if "Head" not in self.cmap:
            return
        path = []
        i = self.cmap["Head"]
        while i >= 0:
            path.append(i)
            i = sk.parent[i]
        path = path[::-1]
        hips = self.cmap.get("Hips")
        if hips is None or hips not in path:
            return
        path = path[path.index(hips):]               # hips … head
        mid = path[1:-1]
        cl = self.cmap.get("LeftShoulder") or self.cmap.get("LeftArm")
        top = sk.parent[cl] if cl is not None else None
        if top in mid:
            k = mid.index(top) + 1
            spine, neck = mid[:k], mid[k:]
        else:
            spine, neck = mid[:-1], mid[-1:]
        for key in ("Spine", "Spine1", "Spine2", "Neck"):
            self.cmap.pop(key, None)
        if spine:
            if len(spine) == 1:
                pick = {"Spine2": spine[0]}
            elif len(spine) == 2:
                pick = {"Spine": spine[0], "Spine2": spine[1]}
            else:
                pick = {"Spine": spine[0], "Spine1": spine[len(spine) // 2], "Spine2": spine[-1]}
            self.cmap.update(pick)
        if neck:
            self.cmap["Neck"] = neck[-1]
        self.chain_src = {"spine": [sk.names[j] for j in spine], "neck": [sk.names[j] for j in neck]}

    def _apply_F(self, M):
        if not hasattr(self, "F"):
            return M
        o = M.copy()
        o[:3, :3] = self.F @ M[:3, :3]
        o[:3, 3] = self.F @ M[:3, 3]
        return o

    def _world(self, clip, t, raw=False):
        Wm = self.sk.world(self.sk.local_mats(clip, t))
        out = {c: Wm[i] for c, i in self.cmap.items()}
        return out if raw else {k: self._apply_F(M) for k, M in out.items()}

    def frame(self, clip, t):
        return self._world(clip, t)

    def dir_rest(self, c):
        p = self.rest[c][:3, 3]
        for ch in SRC_CHILD.get(c, [None]):
            if ch is None:
                break
            if ch in self.rest:
                d = self.rest[ch][:3, 3] - p
                if np.linalg.norm(d) > 1e-6:
                    return d / np.linalg.norm(d)
        # leaf: bone's own +Y axis (glTF / Mixamo convention)
        d = self.rest_R[c][:, 1]
        return d / (np.linalg.norm(d) + 1e-12)


# --------------------------------------------------------------------------------------------- motion
class MocapMotion:
    """
    file:   .glb / .gltf / .bvh            clip: animation name (default: the first / longest non-TPose)
    body:   mograph.human.body.Body        root_motion: "file" | "in_place" | "plant"
    speed:  time scale (0.5 = slow-mo)     loop: True → t wraps inside the clip
    mirror: swap left/right (a left-handed take of a right-handed gesture)
    trim:   (t0, t1) seconds inside the clip    hips: False → pelvis translation ignored (debug / comparison)
    fingers: keep finger mocap (True) — False gives relaxed hands (cleaner line art at small sizes)
    """

    def __init__(self, file, clip=None, body: Body | None = None, root_motion="file", speed=1.0, loop=True,
                 mirror=False, trim=None, hips=True, fingers=True, ground="clip", fps_sample=30):
        self.sk = load_motion(file)
        self.file = file
        names = list(self.sk.clips)
        if clip is None:
            cand = [n for n in names if "tpose" not in n.lower().replace("-", "").replace("_", "")] or names
            clip = max(cand, key=lambda n: self.sk.clips[n].duration) if cand else None
        if clip not in self.sk.clips:
            low = {n.lower(): n for n in names}
            clip = low.get(str(clip).lower(), clip)
        if clip not in self.sk.clips:
            raise ValueError(f"mocap: clip {clip!r} not in {file} (have {names})")
        self.clip = clip
        self.duration = self.sk.clips[clip].duration
        self.trim = trim or (0.0, self.duration)
        self.src = SourceRig(self.sk, rest_clip=None)
        self.body = body or Body()
        self.speed, self.loop, self.mirror, self.hips, self.fingers = speed, loop, mirror, hips, fingers
        self.root_motion = root_motion
        self._prepare_target()
        self._ground = 0.0
        self._plant = None
        if ground == "clip" or root_motion == "plant":
            self._precompute(fps_sample, ground == "clip")

    # ----------------------------------------------------------------------------------- target side
    def _prepare_target(self):
        b = self.body
        self.bnames = bones()
        self.lut = {n: i for i, n in enumerate(self.bnames)}
        self.segs = []                                 # (mh bone ids, canonical source, C)
        for mh, c in SEG_MAP:
            if c not in self.src.rest or mh[0] not in self.lut:
                continue
            if (not self.fingers or not self.src.has_fingers) and "Hand" in c and c[-1].isdigit():
                continue
            d_mh = b.tail(mh[-1]) - b.head(mh[0])
            d_mh = d_mh / (np.linalg.norm(d_mh) + 1e-12)
            if c == "Hips":
                C = np.eye(3)
            else:
                C = rmin(d_mh, self.src.dir_rest(c)).astype(np.float64)
            self.segs.append(([self.lut[m] for m in mh], c, C))
        # scale: MakeHuman (dm) vs source (m), measured on the leg (hip joint → ankle), the part that carries the
        # pelvis bounce — so the retargeted bounce has the right amplitude for THIS body
        mh_leg = b.head("upperleg01.L")[1] - b.head("foot.L")[1]
        r = self.src.rest
        src_leg = r["LeftUpLeg"][:3, 3][1] - r["LeftFoot"][:3, 3][1] if "LeftFoot" in r else self.src.leg_len
        self.scale = float(mh_leg / max(1e-6, src_leg))
        self.hips_rest = self.src.rest["Hips"][:3, 3].copy()

    def _mirror_frame(self, Wf):
        """swap left/right + reflect x (a mirrored take)."""
        Mx = np.diag([-1.0, 1, 1])
        out = {}
        for k, M in Wf.items():
            k2 = k.replace("Left", "§").replace("Right", "Left").replace("§", "Right")
            o = M.copy()
            o[:3, :3] = Mx @ M[:3, :3] @ Mx
            o[:3, 3] = Mx @ M[:3, 3]
            out[k2] = o
        return out

    def clip_time(self, t):
        t0, t1 = self.trim
        u = t * self.speed
        L = max(1e-6, t1 - t0)
        if self.loop:
            u = u % L
        else:
            u = min(max(u, 0.0), L)
        return t0 + u

    def _rot_and_root(self, t):
        ct = self.clip_time(t)
        Wf = self.src.frame(self.clip, ct)
        if self.mirror:
            Wf = self._mirror_frame(Wf)
        B = len(self.bnames)
        Q = np.full((B, 3, 3), np.nan)
        rest_R = self.src.rest_R
        for ids, c, C in self.segs:
            if c not in Wf:
                continue
            D = _ortho(Wf[c]) @ rest_R[c].T
            Qi = D @ C
            for i in ids:
                Q[i] = Qi
        if not self.hips:
            Q[self.lut["root"]] = np.eye(3)
        # unmapped bones inherit the parent's global rotation (rigid with the parent)
        par = self.body.parent
        R = {}
        Qf = np.zeros((B, 3, 3))
        for i in range(B):
            p = par[i]
            Qp = Qf[p] if p >= 0 else np.eye(3)
            Qf[i] = Q[i] if not np.isnan(Q[i, 0, 0]) else Qp
            R[self.bnames[i]] = (Qp.T @ Qf[i]).astype(np.float32)
        hp = Wf["Hips"][:3, 3] - self.hips_rest
        off = hp * self.scale if self.hips else np.zeros(3)
        return R, off, ct

    def _skeleton(self, R, off):
        """root rotation is already inside R["root"] (global of root = D_hips)."""
        b = self.body
        G = b.pose_matrices(R, (0, 0, 0), None)
        # pose_matrices rotates root about spine05 head (pivot) — re-pivot to the source hips (root head)
        piv_src = b.heads[self.lut["root"]]
        Groot = G[self.lut["root"]]
        cur = Groot[:, :3] @ piv_src + Groot[:, 3]
        G[:, :, 3] += (piv_src - cur) + np.asarray(off, np.float32)
        return G

    # ----------------------------------------------------------------------------------- clip analysis
    def _feet(self, G):
        fl = G[self.lut["foot.L"], :, :3] @ self.body.heads[self.lut["foot.L"]] + G[self.lut["foot.L"], :, 3]
        fr = G[self.lut["foot.R"], :, :3] @ self.body.heads[self.lut["foot.R"]] + G[self.lut["foot.R"], :, 3]
        tl = G[self.lut["toe1-1.L"], :, :3] @ self.body.heads[self.lut["toe1-1.L"]] + G[self.lut["toe1-1.L"], :, 3]
        tr = G[self.lut["toe1-1.R"], :, :3] @ self.body.heads[self.lut["toe1-1.R"]] + G[self.lut["toe1-1.R"], :, 3]
        return fl, fr, tl, tr

    def _precompute(self, fps, ground):
        L = (self.trim[1] - self.trim[0]) / max(1e-6, self.speed)
        n = max(2, int(math.ceil(L * fps)) + 1)
        ts = np.linspace(0, L, n)
        fv = foot_vertices()
        lows, feet = [], []
        for t in ts:
            R, off, _ = self._rot_and_root(t)
            G = self._skeleton(R, off)
            X = self.body.skin(G[:, :, :], self.body.rest[fv], self.body.W[fv])
            lows.append(float(X[:, 1].min()))
            feet.append(self._feet(G))
        if ground:
            self._ground = -float(np.percentile(lows, 3))
        if self.root_motion == "plant":
            # planted foot each frame = the lower one (heel/toe mean); advance the body by the opposite of its
            # horizontal motion → contact points never slide.
            disp = np.zeros((n, 3))
            for k in range(1, n):
                fl0, fr0, tl0, tr0 = feet[k - 1]
                fl1, fr1, tl1, tr1 = feet[k]
                hl = min(fl0[1], tl0[1]) + min(fl1[1], tl1[1])
                hr = min(fr0[1], tr0[1]) + min(fr1[1], tr1[1])
                if hl <= hr:
                    d = ((fl1 + tl1) - (fl0 + tl0)) / 2
                else:
                    d = ((fr1 + tr1) - (fr0 + tr0)) / 2
                d[1] = 0
                disp[k] = disp[k - 1] - d
            self._plant = (ts, disp, L)
            self.stride_speed = float(np.linalg.norm(disp[-1]) / max(1e-6, L))   # dm / s

    def _plant_offset(self, t):
        ts, disp, L = self._plant
        cyc = math.floor(t / L) if self.loop else 0
        u = t - cyc * L if self.loop else min(max(t, 0), L)
        k = np.clip(np.searchsorted(ts, u) - 1, 0, len(ts) - 2)
        a = (u - ts[k]) / max(1e-9, ts[k + 1] - ts[k])
        d = disp[k] + (disp[k + 1] - disp[k]) * a
        return d + disp[-1] * cyc

    # ----------------------------------------------------------------------------------- public
    def state(self, t):
        """→ (R {bone: local 3x3}, root offset (3,) dm) with root motion + ground applied (blendable)."""
        R, off, ct = self._rot_and_root(t)
        off = off.copy()
        if self.root_motion in ("in_place", "plant"):
            off[0] = 0.0
            off[2] = 0.0
        if self.root_motion == "plant" and self._plant is not None:
            off += self._plant_offset(t)
        off[1] += self._ground
        return R, off

    def build(self, R, off):
        return self._skeleton(R, off)

    def matrices(self, t):
        return self._skeleton(*self.state(t))

    def pose(self, t):
        G = self.matrices(t)
        return G, self.body.skin(G)

    def joint(self, G, name):
        i = self.lut[name]
        return G[i, :, :3] @ self.body.heads[i] + G[i, :, 3]

    def info(self):
        return {"file": os.path.basename(self.file), "clip": self.clip, "duration": round(self.duration, 3),
                "clips": {k: round(v.duration, 3) for k, v in self.sk.clips.items()},
                "mapped_segments": len(self.segs), "scale_dm_per_m": round(self.scale, 3),
                "ground_offset_dm": round(self._ground, 3),
                "stride_speed_dm_s": round(getattr(self, "stride_speed", 0.0), 3)}
