"""
mograph.human.regions — smooth per-vertex body coordinates on the REST mesh, used to cut garments, hair and to
style lines per region.  Everything is computed once per Body (shape) and cached on it.

    R = regions(body)
    R.arm_s["L"]   dm along shoulder→elbow→wrist (weighted by "armness", 0 on the torso)
    R.leg_s["L"]   dm along hip→knee→ankle→toe (0 on the pelvis)
    R.arm_len / R.leg_len / R.upper_len / R.thigh_len  (dm)
    R.head, R.hand, R.foot, R.face  (0..1 masks)
    R.y            rest height (dm)    R.lm  landmarks (dm, rest): waist, hip, chest, neck, brow, eye, chin, top, …
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .body import BODY_N, Body, bone_index, bones, group_faces, group_vertices


def _w(body: Body, prefixes) -> np.ndarray:
    idx = [i for i, n in enumerate(bones()) if n.startswith(prefixes)]
    return body.W[:, idx].sum(1)


def _side_w(body, prefixes, side):
    idx = [i for i, n in enumerate(bones()) if n.startswith(prefixes) and n.endswith("." + side)]
    return body.W[:, idx].sum(1)


def _chain_param(P, pts):
    """arc-length parameter of the closest point on polyline pts for every point in P (N,3)."""
    best_d = np.full(len(P), np.inf)
    best_s = np.zeros(len(P))
    s0 = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        L = np.linalg.norm(ab)
        t = np.clip(((P - a) @ ab) / (L * L), 0, 1)
        q = a + t[:, None] * ab
        d = np.linalg.norm(P - q, axis=1)
        m = d < best_d
        best_d[m] = d[m]
        best_s[m] = s0 + t[m] * L
        s0 += L
    return best_s, s0


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


@dataclass
class Regions:
    y: np.ndarray
    arm_s: dict
    leg_s: dict
    armness: dict
    legness: dict
    arm_len: float
    upper_len: float
    leg_len: float
    thigh_len: float
    head: np.ndarray
    hand: np.ndarray
    foot: np.ndarray
    face: np.ndarray
    neck: np.ndarray
    lm: dict = field(default_factory=dict)
    topo: object = None
    tris: np.ndarray | None = None


def body_tris():
    Q = group_faces("body")
    return np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]).astype(np.int32)


def regions(body: Body) -> Regions:
    if getattr(body, "_regions", None) is not None:
        return body._regions
    from .lineart import MeshTopo
    X = body.rest
    N = len(X)
    arm_s, leg_s, armness, legness = {}, {}, {}, {}
    for s in ("L", "R"):
        sh, el, wr = body.head(f"upperarm01.{s}"), body.head(f"lowerarm01.{s}"), body.head(f"wrist.{s}")
        tip = body.tail(f"finger3-3.{s}")
        sp, alen = _chain_param(X, np.array([sh, el, wr, tip]))
        am = np.clip(_side_w(body, ("upperarm", "lowerarm", "wrist", "finger", "metacarpal"), s), 0, 1)
        armness[s] = am
        arm_s[s] = sp * smoothstep(0.15, 0.7, am)
        hp, kn, an = body.head(f"upperleg01.{s}"), body.head(f"lowerleg01.{s}"), body.head(f"foot.{s}")
        toe = body.tail(f"toe3-3.{s}") if f"toe3-3.{s}" in bones() else body.tail(f"foot.{s}")
        sp2, llen = _chain_param(X, np.array([hp, kn, an, toe]))
        lm_ = np.clip(_side_w(body, ("upperleg", "lowerleg", "foot", "toe"), s), 0, 1)
        legness[s] = lm_
        leg_s[s] = sp2 * smoothstep(0.15, 0.7, lm_)
    s = "L"
    upper_len = float(np.linalg.norm(body.head("lowerarm01.L") - body.head("upperarm01.L")))
    arm_len = upper_len + float(np.linalg.norm(body.head("wrist.L") - body.head("lowerarm01.L")))
    thigh_len = float(np.linalg.norm(body.head("lowerleg01.L") - body.head("upperleg01.L")))
    leg_len = thigh_len + float(np.linalg.norm(body.head("foot.L") - body.head("lowerleg01.L")))
    head = np.clip(_w(body, ("head", "jaw", "eye", "special", "oris", "levator", "oculi", "orbicularis",
                             "risorius", "temporalis", "tongue")), 0, 1)
    neck = np.clip(_w(body, ("neck",)), 0, 1)
    hand = np.clip(_w(body, ("finger", "metacarpal", "wrist")), 0, 1)
    foot = np.clip(_w(body, ("foot", "toe")), 0, 1)
    y = X[:, 1].copy()
    eye = X[group_vertices("joint-l-eye")].mean(0)
    top = X[:BODY_N, 1].max()
    chin_y = X[group_vertices("joint-jaw")].mean(0)[1] - 0.25
    lm = {
        "top": float(top), "eye": float(eye[1]), "eye_z": float(eye[2]), "chin": float(chin_y),
        "brow": float(eye[1] + 0.32), "neck": float(body.head("neck01")[1]),
        "shoulder": float(body.head("upperarm01.L")[1]), "chest": float(body.head("spine01")[1] - 0.2),
        "waist": float(body.head("spine03")[1]), "hip": float(body.head("upperleg01.L")[1]),
        "crotch": float(body.head("upperleg01.L")[1] - 0.8), "knee": float(body.head("lowerleg01.L")[1]),
        "ankle": float(body.head("foot.L")[1]), "head_c": body.head("head").copy(),
    }
    # face = front of the head below the brow line, above the chin
    zf = X[:, 2] - lm["head_c"][2]
    face = head * smoothstep(0.25, 0.6, zf) * smoothstep(lm["chin"] - 0.2, lm["chin"] + 0.2, y) * \
        (1 - smoothstep(lm["brow"] + 0.15, lm["brow"] + 0.5, y))
    tris = body_tris()
    topo = MeshTopo(tris, N)
    R = Regions(y=y, arm_s=arm_s, leg_s=leg_s, armness=armness, legness=legness, arm_len=arm_len,
                upper_len=upper_len, leg_len=leg_len, thigh_len=thigh_len, head=head, hand=hand, foot=foot,
                face=face, neck=neck, lm=lm, topo=topo, tris=tris)
    body._regions = R
    return R
