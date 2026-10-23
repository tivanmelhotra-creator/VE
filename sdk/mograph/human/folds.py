"""
mograph.human.folds — pose-driven FOLD / CREASE strokes: the few inner lines an illustrator actually draws.

Each rule watches a joint angle in the CURRENT pose and, when it bends past a threshold, emits short 3D strokes on
the garment (or skin) surface at the anatomical crease:
    elbow   inner-elbow fold (2 short strokes on the sleeve, fanning from the crook)       angle > 25°
    knee    back-of-knee fold on trousers (sitting / walking swing leg)                    angle > 30°
    hip     lap fold at the hip crease when sitting                                       angle > 45°
    armpit  1 drag line from armpit toward the chest when the arm is raised               abduction > 50°
    waist   side-waist fold on the compressed side when the torso bends / twists           side > 4°
The strokes are built in REST space on the garment shell (pick the nearest garment vertices along a short path)
and skinned with the garment → they move with the cloth; visibility is resolved like any other stroke.
Line weight: ~0.38 of the outer silhouette, tapered both ends (an illustrator's "flick").
"""
from __future__ import annotations

import math

import numpy as np

from .body import bone_index
from .lineart import Stroke3D


def _angle(G, body, a, b):
    """angle (deg) between bone a and bone b directions in the posed skeleton (0 = straight)."""
    def d(n):
        i = bone_index(n)
        h = G[i, :, :3] @ body.heads[i] + G[i, :, 3]
        t = G[i, :, :3] @ body.tails[i] + G[i, :, 3]
        v = t - h
        return v / (np.linalg.norm(v) + 1e-9)
    c = float(np.clip(np.dot(d(a), d(b)), -1, 1))
    return math.degrees(math.acos(c))


def _pj(G, body, n, tail=False):
    i = bone_index(n)
    p = body.tails[i] if tail else body.heads[i]
    return G[i, :, :3] @ p + G[i, :, 3]


def _snap_path(P_rest, V_posed, pts_rest, lift=0.02, Vn=None, k=4):
    """Each rest point → inverse-distance blend of its k nearest garment vertices (rest), evaluated on the POSED
    garment (+ lift along the normal). A path densely resampled first, so it never collapses to one vertex."""
    pts_rest = np.asarray(pts_rest, np.float64)
    d = np.linalg.norm(np.diff(pts_rest, axis=0), axis=1)
    s = np.r_[0, np.cumsum(d)]
    n = max(6, int(s[-1] / 0.05))
    u = np.linspace(0, s[-1], n)
    P = np.c_[[np.interp(u, s, pts_rest[:, c]) for c in range(3)]].T
    out = []
    for p in P:
        dd = np.einsum("ij,ij->i", P_rest - p, P_rest - p)
        idx = np.argpartition(dd, k)[:k]
        w = 1.0 / (np.sqrt(dd[idx]) + 1e-3)
        w /= w.sum()
        q = (V_posed[idx] * w[:, None]).sum(0)
        if Vn is not None:
            nn = (Vn[idx] * w[:, None]).sum(0)
            q = q + nn / (np.linalg.norm(nn) + 1e-9) * lift
        out.append(q)
    out = np.asarray(out)
    # the k-NN blend wobbles from vertex to vertex → smooth the 3D path (keeps ends) so folds are confident flicks
    for _ in range(6):
        out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out


def _surface_dir(P, c, d, cone=0.35):
    """offset from c to the garment surface along direction d (closest garment vertex within a cone)."""
    v = P - c
    r = np.linalg.norm(v, axis=1) + 1e-9
    cos = (v @ d) / r
    m = (cos > 1 - cone) & (r < 2.5)
    if not m.any():
        return d * 0.6
    i = np.nonzero(m)[0][np.argmin(r[m] * (2 - cos[m]))]
    return d * float(v[i] @ d)


def _vn(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, T[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def fold_strokes(ch, G, parts) -> list[Stroke3D]:
    """ch: Character, G: posed matrices, parts: list of lineart.Part (posed) aligned with [body] + ch.garments."""
    body = ch.body
    out: list[Stroke3D] = []
    by_name = {p.name: p for p in parts}
    garments = {g.name: g for g in ch.garments}

    def surface_for(region):
        """→ (garment, part) covering the region, or the body."""
        for g in ch.garments:
            if g.kind != "cloth":
                continue
            a = g.attrs
            if region in ("elbow.L", "elbow.R"):
                s = region[-1]
                if f"arm_s.{s}" in a and (a[f"arm_s.{s}"] > ch.R.upper_len + 0.3).any():
                    return g, by_name.get(g.name)
            if region in ("knee.L", "knee.R", "hip.L", "hip.R"):
                s = region[-1]
                if f"leg_s.{s}" in a and (a[f"leg_s.{s}"] > ch.R.thigh_len + 0.3).any():
                    return g, by_name.get(g.name)
        return None, None

    for s in ("L", "R"):
        sg = 1 if s == "L" else -1
        # ------------------------------------------------------------------ elbow
        ang = _angle(G, body, f"upperarm02.{s}", f"lowerarm01.{s}")
        g, p = surface_for(f"elbow.{s}")
        if ang > 25 and g is not None and p is not None:
            amt = min(1.0, (ang - 25) / 60)
            el = body.head(f"lowerarm01.{s}")
            up = body.head(f"upperarm02.{s}")
            wr = body.head(f"wrist.{s}")
            ax_u = (up - el) / np.linalg.norm(up - el)
            ax_f = (wr - el) / np.linalg.norm(wr - el)
            # crook = inner side of the elbow in the rest pose (palm side / front): +z mostly
            inner = np.array([0.0, 0.0, 1.0]) - ax_u * ax_u[2]
            inner /= np.linalg.norm(inner) + 1e-9
            Vn = _vn(p.verts, p.tris)
            off = _surface_dir(g.rest, el, inner)
            side = np.cross(ax_u, inner)
            side /= np.linalg.norm(side) + 1e-9
            # fan of curved strokes radiating from the crook (the classic "Y" sleeve fold)
            fans = [(0.0, 0.95), (0.55, 0.7), (-0.5, 0.6)][: 1 + int(amt > 0.3) + int(amt > 0.65)]
            for k, (spread, ln) in enumerate(fans):
                L0 = (0.45 + 0.55 * amt) * ln
                pts = []
                for t in np.linspace(0, 1, 9):
                    a = ax_u * (L0 * t) + side * (spread * 0.35 * t) + ax_f * (0.18 * (1 - t) ** 2)
                    pts.append(el + off + a - ax_u * 0.12)
                P = _snap_path(g.rest, p.verts, pts, 0.035, Vn)
                out.append(Stroke3D(P, 0.42 - 0.06 * k, "fold", 0.55, pid=p.pid))
        # ------------------------------------------------------------------ knee (back fold / front pull)
        ang = _angle(G, body, f"upperleg02.{s}", f"lowerleg01.{s}")
        g, p = surface_for(f"knee.{s}")
        if ang > 30 and g is not None and p is not None:
            amt = min(1.0, (ang - 30) / 60)
            kn = body.head(f"lowerleg01.{s}")
            Vn = _vn(p.verts, p.tris)
            back = _surface_dir(g.rest, kn, np.array([0, 0, -1.0]))
            for k in range(1 + int(amt > 0.4)):
                c = kn + np.array([0.0, -0.05 - 0.22 * k, 0.0]) + back
                pts = [c + np.array([sg * 0.7 * (t - 0.5), 0.18 * (t - 0.5) ** 2, 0]) for t in np.linspace(0, 1, 9)]
                out.append(Stroke3D(_snap_path(g.rest, p.verts, pts, 0.035, Vn), 0.4, "fold", 0.5, pid=p.pid))
            # front-of-knee pull lines (diagonal, from the knee up the thigh)
            front = _surface_dir(g.rest, kn + np.array([0, 0.3, 0]), np.array([0, 0, 1.0]))
            c = kn + np.array([0.0, 0.3, 0]) + front
            pts = [c + np.array([sg * 0.12 * (t - 0.5), 0.7 * t, 0.0]) for t in np.linspace(0, 1, 6)]
            out.append(Stroke3D(_snap_path(g.rest, p.verts, pts, 0.03, Vn), 0.32, "fold", 0.6, pid=p.pid))
        # ------------------------------------------------------------------ hip crease (sitting)
        hip_ang = _angle(G, body, "spine04", f"upperleg01.{s}")
        g, p = surface_for(f"hip.{s}")
        if hip_ang > 125 and g is not None and p is not None:    # spine up vs thigh forward
            hp = body.head(f"upperleg01.{s}")
            Vn = _vn(p.verts, p.tris)
            c0 = hp + np.array([sg * 0.05, -0.45, 0.0])
            c = c0 + _surface_dir(g.rest, c0, np.array([0, 0, 1.0]))
            for k in range(2):
                pts = [c + np.array([sg * (1.1 * t - 0.35) , -0.32 * t - 0.25 * k, -0.15 * t])
                       for t in np.linspace(0, 1, 9)]
                out.append(Stroke3D(_snap_path(g.rest, p.verts, pts, 0.035, Vn), 0.4 - 0.08 * k, "fold", 0.55,
                                    pid=p.pid))
        # ------------------------------------------------------------------ armpit drag (raised arm)
        abd = _angle(G, body, "spine01", f"upperarm01.{s}")
        if abd < 110:                                          # arm raised (angle to the spine-up axis shrinks)
            for gg in ch.garments:
                if gg.kind == "cloth" and f"arm_s.{s}" in gg.attrs and (gg.attrs[f"arm_s.{s}"] > 0.3).any() \
                        and (gg.attrs["y"] > ch.R.lm["chest"]).any():
                    pp = by_name.get(gg.name)
                    if pp is None:
                        break
                    sh = body.head(f"upperarm01.{s}")
                    Vn = _vn(pp.verts, pp.tris)
                    c = sh + np.array([-sg * 0.25, -0.9, 0.35])
                    pts = [c + np.array([-sg * 0.55 * t, -0.35 * t, 0.15 * t]) for t in np.linspace(0, 1, 6)]
                    out.append(Stroke3D(_snap_path(gg.rest, pp.verts, pts, 0.03, Vn), 0.34, "fold", 0.6,
                                        pid=pp.pid))
                    break
    return out
