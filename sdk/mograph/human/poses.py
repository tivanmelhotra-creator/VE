"""
mograph.human.poses — anatomical pose language → bone rotations (no hand-tuned bone eulers).

A pose is a dict in BODY SPACE (x = character's left, y = up, z = front):

    {"arm.L": {"upper": (abd, flex), "fore": (abd, flex), "hand": "relaxed"|"fist"|"open"|"point"},
     "leg.R": {"thigh": (abd, flex), "shin": (abd, flex), "foot": pitch_deg},
     "spine": (pitch, side, twist), "neck": (pitch, side, twist),
     "root": {"yaw": deg, "pitch": deg, "roll": deg, "y": dm, "x": dm, "z": dm}, "ground": True}

Segment directions are spherical angles from straight DOWN: `abd` = angle out to the side (frontal plane, + = away
from the body), `flex` = angle to the front (+) / back (−). So an arm hanging = (8, 0), arm forward horizontal =
(0, 90), arm straight up = (0, 180), T-pose = (90, 0). A thigh sitting = (6, 88); its shin = (2, 2).

Aimed segments use a minimal rotation from the parent-rotated rest direction (hinge-like elbows / knees, no candy-
wrap twist). spine / neck rotations are spread over the vertebrae.  `solve(body, pose)` → {bone: R} for
Body.pose_matrices, `pose_body(body, pose)` → (G, verts) with feet ground-locked.
"""
from __future__ import annotations

import math

import numpy as np

from .body import Body, BODY_N, bone_index, bones, rot_axis, group_vertices

SPINE = ["spine05", "spine04", "spine03", "spine02", "spine01"]
NECK = ["neck01", "neck02", "neck03", "head"]
SEG = {
    "arm": {"upper": ["upperarm01", "upperarm02"], "fore": ["lowerarm01", "lowerarm02"]},
    "leg": {"thigh": ["upperleg01", "upperleg02"], "shin": ["lowerleg01", "lowerleg02"]},
}
FINGERS = [[f"finger{f}-{s}" for s in (1, 2, 3)] for f in range(1, 6)]


def sph(abd, flex, side=+1.0):
    """down-based spherical angles → unit vector. side=+1 for the character's LEFT limb, −1 for RIGHT.
    The frontal-plane vector (abd) is swung forward about the x axis by `flex`."""
    a, f = math.radians(abd), math.radians(flex)
    x, y = side * math.sin(a), -math.cos(a)
    return np.array([x, y * math.cos(f), -y * math.sin(f)], np.float64)


def rmin(a, b):
    """minimal rotation taking unit a to unit b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        return rot_axis(np.cross(a, p), 180)
    return rot_axis(v, math.degrees(math.atan2(s, c)))


def euler_body(pitch=0.0, side=0.0, twist=0.0):
    """pitch + = bend forward (about +x), side + = lean to the character's LEFT (about −z), twist + = turn left
    (about +y)."""
    return rot_axis((0, 1, 0), twist) @ rot_axis((0, 0, 1), -side) @ rot_axis((1, 0, 0), pitch)


# --------------------------------------------------------------------------------------------- solver
def solve(body: Body, pose: dict) -> tuple[dict, np.ndarray]:
    lut = {n: i for i, n in enumerate(bones())}
    names = bones()
    B = len(names)
    Q = np.repeat(np.eye(3, dtype=np.float32)[None], B, 0)      # global rotations
    R = {}
    par = body.parent

    # per-bone local rotation requests (in rest world axes, composed after the parent)
    local = {}
    sp = pose.get("spine")
    if sp:
        n = len(SPINE)
        for b in SPINE:
            local[b] = euler_body(sp[0] / n, sp[1] / n, (sp[2] if len(sp) > 2 else 0) / n)
    nk = pose.get("neck")
    if nk:
        n = len(NECK)
        for b in NECK:
            local[b] = euler_body(nk[0] / n, nk[1] / n, (nk[2] if len(nk) > 2 else 0) / n)
    for side_key, sgn in (("L", 1.0), ("R", -1.0)):
        sh = pose.get(f"shoulder.{side_key}")             # (raise_deg, forward_deg) shrug / protraction
        if sh:
            local[f"clavicle.{side_key}"] = rot_axis((0, 0, 1), sgn * sh[0]) @ rot_axis((0, 1, 0), -sgn * sh[1])

    aims = {}
    for limb in ("arm", "leg"):
        for side_key, sgn in (("L", 1.0), ("R", -1.0)):
            spec = pose.get(f"{limb}.{side_key}")
            if not spec:
                continue
            for seg, chain in SEG[limb].items():
                if seg in spec and spec[seg] is not None:
                    v = spec[seg]
                    d = np.asarray(v, float) if len(v) == 3 else sph(v[0], v[1], sgn)
                    aims[f"{chain[0]}.{side_key}"] = (d, f"{chain[-1]}.{side_key}")
            if limb == "leg":
                aims[f"foot.{side_key}"] = ("foot", spec.get("foot", 0.0), spec.get("toe_out", 8.0) * sgn)

    hands = {}
    for side_key in ("L", "R"):
        spec = pose.get(f"arm.{side_key}") or {}
        hands[side_key] = (spec.get("hand", "relaxed"), spec.get("wrist", (0.0, 0.0)))

    for i, n in enumerate(names):
        p = par[i]
        Qp = Q[p] if p >= 0 else np.eye(3, dtype=np.float32)
        if n in aims:
            a = aims[n]
            if isinstance(a[0], str):
                # keep the foot flat (pitch) + toe-out in world, independent of the shin
                rest = body.bone_dir(n)
                flat = np.array([0.0, rest[1], rest[2]])
                tgt = rot_axis((0, 1, 0), a[2]) @ rot_axis((1, 0, 0), -a[1]) @ flat
                cur = Qp @ rest
                Qi = rmin(cur, tgt) @ Qp
            else:
                d_t, tail_bone = a
                # rest segment direction = head of first bone → tail of last bone
                rest = body.tail(tail_bone) - body.head(n)
                rest = rest / np.linalg.norm(rest)
                cur = Qp @ rest
                Qi = rmin(cur, d_t) @ Qp
            Q[i] = Qi
        elif n in local:
            Q[i] = Qp @ local[n]
        else:
            Q[i] = Qp
        R[n] = (Qp.T @ Q[i]).astype(np.float32)

    # hands: wrist bend + finger curl (relative rotations about each finger's own lateral axis)
    for side_key, sgn in (("L", 1.0), ("R", -1.0)):
        kind, wr = hands[side_key]
        w = f"wrist.{side_key}"
        fwd = body.bone_dir(f"lowerarm02.{side_key}")
        m2, m5 = body.head(f"finger2-1.{side_key}"), body.head(f"finger5-1.{side_key}")
        across = m2 - m5
        across /= np.linalg.norm(across)
        palm = np.cross(fwd, across) * sgn            # palm normal (rest)
        if wr and (wr[0] or wr[1]):
            R[w] = (rot_axis(across, wr[0]) @ rot_axis(palm, wr[1])).astype(np.float32) @ R.get(w, np.eye(3))
        curl = {"relaxed": (18, 22, 14, 14), "fist": (85, 95, 70, 40), "open": (4, 4, 2, 0),
                "point": (85, 95, 70, 30), "hold": (40, 55, 35, 20), "flat": (0, 0, 0, 0)}.get(kind, (18, 22, 14, 14))
        for fi, chain in enumerate(FINGERS):
            for si, b in enumerate(chain):
                bn = f"{b}.{side_key}"
                if fi == 0:                            # thumb
                    ang = curl[3] * (0.6 if si == 0 else 1.0)
                    axis = palm
                elif kind == "point" and fi == 1:
                    ang = 2.0
                    axis = across
                else:
                    ang = curl[si] * (1.0 + 0.08 * (fi - 1))
                    axis = across
                if ang:
                    R[bn] = rot_axis(axis, -ang * sgn).astype(np.float32)
    return R, Q


def root_rot(pose):
    r = pose.get("root") or {}
    return rot_axis((0, 1, 0), r.get("yaw", 0)) @ rot_axis((0, 0, 1), -r.get("roll", 0)) @ \
        rot_axis((1, 0, 0), r.get("pitch", 0))


_FEET = {}


def foot_vertices():
    if "f" not in _FEET:
        from .body import _base
        W = _base()["W"]
        idx = [bone_index(n) for n in bones() if n.startswith(("foot.", "toe"))]
        _FEET["f"] = np.nonzero(W[:BODY_N][:, idx].sum(1) > 0.5)[0]
    return _FEET["f"]


def pose_body(body: Body, pose: dict, ground=None):
    """→ (G (B,3,4), posed verts). ground=True: lowest foot vertex on y=0 (default unless root.y is given)."""
    R, _ = solve(body, pose)
    r = pose.get("root") or {}
    off = np.array([r.get("x", 0.0), r.get("y", 0.0), r.get("z", 0.0)], np.float32)
    RR = root_rot(pose)
    G = body.pose_matrices(R, off, RR)
    X = body.skin(G)
    g = pose.get("ground", "y" not in r) if ground is None else ground
    if g:
        fv = foot_vertices()
        dy = -X[fv, 1].min()
        if pose.get("seat") is not None:            # sitting: pelvis on the seat height (dm), feet may hover
            dy = 0.0
        G[:, 1, 3] += dy
        X[:, 1] += dy
    return G, X


# --------------------------------------------------------------------------------------------- library
def arm(upper, fore, hand="relaxed", wrist=(0, 0)):
    return {"upper": upper, "fore": fore, "hand": hand, "wrist": wrist}


def leg(thigh, shin, foot=0.0, toe_out=8.0):
    return {"thigh": thigh, "shin": shin, "foot": foot, "toe_out": toe_out}


def stand(**kw):
    p = {"arm.L": arm((9, 2), (7, 14)), "arm.R": arm((9, 2), (7, 14)),
         "leg.L": leg((3, 0), (2, -1)), "leg.R": leg((3, 0), (2, -1)),
         "spine": (2, 0, 0), "neck": (-2, 0, 0)}
    p.update(kw)
    return p


def stand_contra(**kw):
    """relaxed contrapposto: weight on the right leg, hip shift, left knee soft."""
    p = {"arm.L": arm((10, 0), (8, 18)), "arm.R": arm((8, 4), (6, 12)),
         "leg.L": leg((7, 6), (3, -6), toe_out=14), "leg.R": leg((1, -2), (2, -2)),
         "spine": (2, -4, 4), "neck": (-2, 3, -6), "root": {"roll": 3, "yaw": 0}}
    p.update(kw)
    return p


def walk(phase: float = 0.15, stride=1.0, arm_swing=1.0, **kw):
    """phase 0..1 (0 = left heel strike). Real gait curves (hip / knee / ankle sagittal angles)."""
    def gait(ph):
        t = 2 * math.pi * ph
        hip = (20 * math.cos(2 * math.pi * (ph - 0.92)) + 7) * stride          # + forward (peak flex late swing)
        # knee: small bend at loading (~0.1), big in swing (~0.73)
        knee = 18 * math.exp(-((ph - 0.12) / 0.07) ** 2) + 62 * math.exp(-((ph - 0.72) / 0.12) ** 2) + 3
        ankle = 10 * math.exp(-((ph - 0.0) / 0.06) ** 2) - 14 * math.exp(-((ph - 0.6) / 0.06) ** 2)
        return hip, knee * (0.6 + 0.4 * stride), ankle
    out = {}
    for side, off in (("L", 0.0), ("R", 0.5)):
        h, k, a = gait((phase + off) % 1.0)
        out[f"leg.{side}"] = leg((3, h), (2, h - k), foot=a, toe_out=7)
        sw = -h * 0.75 * arm_swing                                      # opposite arm
        out[f"arm.{side}"] = arm((8, sw), (7, sw + 12 + max(0, sw) * 0.6))
    t = 2 * math.pi * phase
    out["spine"] = (4, 0, 5 * math.cos(t))
    out["neck"] = (-3, 0, -4 * math.cos(t))
    out["root"] = {"yaw": -4 * math.cos(t) * stride, "roll": 2.0 * math.sin(2 * t + 0.3)}
    out.update(kw)
    return out


def run(phase: float = 0.15, **kw):
    def g(ph):
        t = 2 * math.pi * ph
        hip = 38 * math.cos(t) + 10
        knee = 35 * math.exp(-((ph - 0.15) / 0.08) ** 2) + 105 * math.exp(-((ph - 0.7) / 0.13) ** 2) + 10
        return hip, knee
    out = {}
    for side, off in (("L", 0.0), ("R", 0.5)):
        h, k = g((phase + off) % 1.0)
        out[f"leg.{side}"] = leg((3, h), (2, h - k), foot=-5)
        sw = -h * 1.1
        out[f"arm.{side}"] = arm((10, sw * 0.7), (8, sw * 0.7 + 85), hand="fist")
    t = 2 * math.pi * phase
    out["spine"] = (12, 0, 8 * math.cos(t))
    out["neck"] = (-10, 0, -6 * math.cos(t))
    out["root"] = {"yaw": -6 * math.cos(t)}
    out.update(kw)
    return out


def sit(seat_dm=4.6, **kw):
    """chair sitting: thighs horizontal, shins vertical, hands on thighs. seat_dm is informational (feet on
    ground decide the pelvis height)."""
    p = {"leg.L": leg((7, 86), (3, 4), toe_out=10), "leg.R": leg((7, 86), (3, 4), toe_out=10),
         "arm.L": arm((10, 18), (6, 70), hand="relaxed"), "arm.R": arm((10, 18), (6, 70), hand="relaxed"),
         "spine": (6, 0, 0), "neck": (-6, 0, 0)}
    p.update(kw)
    return p


def sit_cross(**kw):
    p = sit()
    p["leg.L"] = leg((2, 95), (-12, 30), toe_out=0)       # left leg over the right knee
    p.update(kw)
    return p


def wave(t: float = 0.0, side="R", **kw):
    p = stand()
    sg = 1 if side == "L" else -1
    p[f"arm.{side}"] = arm((65, 20), (18 + 14 * math.sin(2 * math.pi * t * 2), 175), hand="open")
    p["neck"] = (-2, 4 * sg, 8 * sg)
    p.update(kw)
    return p


def point(side="R", **kw):
    p = stand_contra()
    p[f"arm.{side}"] = arm((55, 55), (55, 75), hand="point")
    p["neck"] = (-2, 0, 18 if side == "L" else -18)
    p.update(kw)
    return p


def present(side="R", **kw):
    """open-palm presenting gesture (explainer videos)."""
    p = stand_contra()
    p[f"arm.{side}"] = arm((30, 25), (45, 70), hand="open", wrist=(-20, 0))
    p.update(kw)
    return p


def think(side="R", **kw):
    """hand to chin."""
    p = stand()
    o = "L" if side == "R" else "R"
    p[f"arm.{side}"] = arm((10, 55), (-35, 160), hand="hold")
    p[f"arm.{o}"] = arm((16, 30), (-70, 100), hand="relaxed")
    p["neck"] = (6, 0, 0)
    p.update(kw)
    return p


def phone(side="R", **kw):
    p = stand_contra()
    p[f"arm.{side}"] = arm((22, 30), (-35, 172), hand="hold")
    p["neck"] = (0, 6 if side == "R" else -6, 0)
    p.update(kw)
    return p


def hands_on_hips(**kw):
    p = stand()
    p["arm.L"] = arm((42, -12), (-38, -20), hand="fist")
    p["arm.R"] = arm((42, -12), (-38, -20), hand="fist")
    p.update(kw)
    return p


def arms_crossed(**kw):
    p = stand()
    p["arm.L"] = arm((16, 32), (-72, 112), hand="relaxed")
    p["arm.R"] = arm((16, 36), (-74, 104), hand="relaxed")
    p.update(kw)
    return p


def typing(**kw):
    p = sit()
    p["arm.L"] = arm((8, 25), (8, 82), hand="relaxed")
    p["arm.R"] = arm((8, 25), (8, 82), hand="relaxed")
    p["spine"] = (10, 0, 0)
    p["neck"] = (8, 0, 0)
    p.update(kw)
    return p


LIBRARY = {"stand": stand, "stand_contra": stand_contra, "walk": walk, "run": run, "sit": sit,
           "sit_cross": sit_cross, "wave": wave, "point": point, "present": present, "think": think,
           "phone": phone, "hands_on_hips": hands_on_hips, "arms_crossed": arms_crossed, "typing": typing}


def get(name: str, **kw) -> dict:
    return LIBRARY[name](**kw)
