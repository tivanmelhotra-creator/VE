"""
mograph.toon.rig — a dedicated 3D TOON CHARACTER MODEL ("ball-head" turnaround: round head, short neck with a
rolled turtleneck collar, baggy sweater with ribbed hem, puffy sleeves gathered into cuffs, big 4-finger cartoon
hands, wide straight trousers with a few creases, big rounded shoes).

It is NOT a deformed realistic body: every piece is modelled procedurally (lofted tubes / superellipsoid blobs)
around the MakeHuman skeleton and SKINNED to its bones (smooth inverse-distance weights along each bone chain),
so any captured motion (Mixamo / CMU / Quaternius via mograph.mocap) drives it directly.

    rig = ToonRig("guy")                      # proportions dict can be overridden
    A = Actor("me", rig, [...mocap segments...])
    toons = {"me": rig}                       # mograph.toon.render handles it like any toon
"""
from __future__ import annotations

import math

import numpy as np

from ..human.body import Body, bone_index, bones
from ..human.lineart import MeshTopo, Part, Stroke3D
from .face import ToonFace

WHITE = (0.985, 0.98, 0.965)
DARK = (0.18, 0.17, 0.17)


# ----------------------------------------------------------------------------------------------- geometry
def _frames(C):
    """parallel-transport frames along a polyline C (N,3) → T, N1, N2."""
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n = np.cross(T[0], ref)
    n /= np.linalg.norm(n)
    N1, N2 = [n], [np.cross(T[0], n)]
    for i in range(1, len(C)):
        n = N1[-1] - np.dot(N1[-1], T[i]) * T[i]
        n /= np.linalg.norm(n) + 1e-12
        N1.append(n)
        N2.append(np.cross(T[i], n))
    return T, np.asarray(N1), np.asarray(N2)


def tube(C, rad, nseg=28, cap0=True, cap1=True, ref=None, rad2=None, off=None):
    """loft a closed tube along centre-line C (N,3); rad(N) [and rad2(N) for elliptical sections, measured along
    `ref` (a world direction projected on each section)] ; off(N,3) shifts each ring. → verts, tris, (rows, cols)."""
    C = np.asarray(C, float)
    n = len(C)
    T, N1, N2 = _frames(C)
    if ref is not None:
        ref = np.asarray(ref, float)
        a = ref - (T @ ref)[:, None] * T
        a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        N1, N2 = a, np.cross(T, a)
    r1 = np.asarray(rad, float)
    r2 = r1 if rad2 is None else np.asarray(rad2, float)
    th = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
    V = (C[:, None, :] + np.cos(th)[None, :, None] * N1[:, None, :] * r1[:, None, None] +
         np.sin(th)[None, :, None] * N2[:, None, :] * r2[:, None, None])
    if off is not None:
        V = V + np.asarray(off, float)[:, None, :]
    V = V.reshape(-1, 3)
    T_ = []
    for i in range(n - 1):
        for j in range(nseg):
            a, b = i * nseg + j, i * nseg + (j + 1) % nseg
            c, d = a + nseg, b + nseg
            T_ += [(a, b, c), (b, d, c)]
    extra = []
    if cap0:
        k = len(V) + len(extra)
        extra.append(C[0] - T[0] * 1e-3)
        T_ += [(k, (j + 1) % nseg, j) for j in range(nseg)]
    if cap1:
        k = len(V) + len(extra)
        extra.append(C[-1] + T[-1] * 1e-3)
        base = (n - 1) * nseg
        T_ += [(k, base + j, base + (j + 1) % nseg) for j in range(nseg)]
    if extra:
        V = np.vstack([V, extra])
    return V, np.asarray(T_, np.int32), (n, nseg)


def blob(center, half, nu=32, nv=18, p=2.6, top_flat=0.0, bottom_flat=1.0, yaw=0.0):
    """superellipsoid (rounded box) — shoes, palms. half = (hx, hy, hz); flatter bottom with bottom_flat."""
    u = np.linspace(-math.pi, math.pi, nu, endpoint=False)
    v = np.linspace(-math.pi / 2 + 0.03, math.pi / 2 - 0.03, nv)
    U, Vv = np.meshgrid(u, v)
    e = 2.0 / p

    def sg(x, k):
        return np.sign(x) * np.abs(x) ** k
    x = sg(np.cos(Vv), e) * sg(np.cos(U), e)
    z = sg(np.cos(Vv), e) * sg(np.sin(U), e)
    y = sg(np.sin(Vv), e if bottom_flat <= 0 else e * 0.6)
    y = np.where(y < 0, y * (1 - 0.0), y)
    P = np.c_[x.ravel() * half[0], y.ravel() * half[1], z.ravel() * half[2]]
    if yaw:
        c, s = math.cos(yaw), math.sin(yaw)
        P = P @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    P += np.asarray(center, float)
    T = []
    for i in range(nv - 1):
        for j in range(nu):
            a, b = i * nu + j, i * nu + (j + 1) % nu
            T += [(a, c_, b) for c_ in [a + nu]] + [(b, a + nu, b + nu)]
    bot, top = len(P), len(P) + 1
    P = np.vstack([P, [np.asarray(center) + [0, -half[1], 0], np.asarray(center) + [0, half[1], 0]]])
    T += [(bot, (j + 1) % nu, j) for j in range(nu)]
    T += [(top, (nv - 1) * nu + j, (nv - 1) * nu + (j + 1) % nu) for j in range(nu)]
    return P, np.asarray(T, np.int32)


def sphere(center, r, nu=48, nv=30, sq=(1, 1, 1)):
    u = np.linspace(-math.pi, math.pi, nu, endpoint=False)
    v = np.linspace(-math.pi / 2 + 0.02, math.pi / 2 - 0.02, nv)
    U, V = np.meshgrid(u, v)
    P = np.c_[(np.sin(U) * np.cos(V)).ravel() * sq[0], np.sin(V).ravel() * sq[1], (np.cos(U) * np.cos(V)).ravel() * sq[2]]
    P = np.asarray(center) + P * r
    T = []
    for i in range(nv - 1):
        for j in range(nu):
            a, b = i * nu + j, i * nu + (j + 1) % nu
            T += [(a, a + nu, b), (b, a + nu, b + nu)]
    bot, top = len(P), len(P) + 1
    P = np.vstack([P, [np.asarray(center) + [0, -r * sq[1], 0], np.asarray(center) + [0, r * sq[1], 0]]])
    T += [(bot, (j + 1) % nu, j) for j in range(nu)]
    T += [(top, (nv - 1) * nu + j, (nv - 1) * nu + (j + 1) % nu) for j in range(nu)]
    return P, np.asarray(T, np.int32)


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(1e-9, ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def chain_weights(P, body, names, power=5.0, tails=None):
    """smooth skin weights of points P on the bones `names` (inverse distance to each bone segment)."""
    lut = {n: i for i, n in enumerate(bones())}
    D = np.stack([_seg_dist(P, body.heads[lut[n]], body.tails[lut[n]]) for n in names], 1)
    w = 1.0 / (D ** power + 1e-6)
    w /= w.sum(1, keepdims=True)
    W = np.zeros((len(P), len(lut)), np.float32)
    for k, n in enumerate(names):
        W[:, lut[n]] = w[:, k]
    return W


# ----------------------------------------------------------------------------------------------- the rig
class _Piece:
    def __init__(self, name, V, T, W, fill="cloth", min_jump=0.3, weight=1.0, smooth=1, grid=None, group=None):
        self.group = group
        self.name, self.V, self.T = name, V.astype(np.float32), T
        self.W = W.astype(np.float32)
        self.topo = MeshTopo(T, len(V))
        self.fill, self.min_jump, self.weight, self.smooth = fill, min_jump, weight, smooth
        self.grid = grid


class BallHead:
    """head frame for ToonFace (face drawn on the ball surface)."""

    def __init__(self, c, r, sq=(1.0, 1.0, 1.0)):
        self.hi = bone_index("head")
        self.c = np.asarray(c, float)
        self.r = float(r)
        self.sq = np.asarray(sq, float)

    def center(self, G):
        M = G[self.hi]
        return M[:, :3] @ self.c + M[:, 3]

    def face_points(self, G, uv, lift=1.01):
        uv = np.asarray(uv, float)
        u, v = uv[:, 0], uv[:, 1]
        d = np.stack([np.sin(u) * np.cos(v), np.sin(v), np.cos(u) * np.cos(v)], -1)
        P = self.c + d * self.r * self.sq * lift
        M = G[self.hi].astype(np.float64)
        n = d / self.sq
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return P @ M[:, :3].T + M[:, 3], n @ M[:, :3].T

    def point(self, G, u, v, k=1.0):
        return self.face_points(G, [[u, v]], lift=k)[0][0]


DEFAULT = dict(
    head_r=2.0,            # ball head radius (dm) — body ≈ 5.3 heads tall
    neck_len=0.25,          # visible neck between collar and head
    torso_w=1.62, torso_d=1.22, shoulder_w=1.9, hem_y=9.15, hem_h=0.38,
    sleeve_r=0.7, sleeve_puff=1.0, cuff_r=0.5, cuff_len=0.42,
    hand=1.35,              # hand scale
    thigh_r=1.02, ankle_r=0.98, pants_hem=1.25,
    shoe_len=3.7, shoe_w=1.12, shoe_h=1.45,
)


class ToonRig:
    def __init__(self, preset="guy", body=None, female=False, kind="adult", seed=0, fills=None, **prop):
        P = dict(DEFAULT, **prop)
        self.P = P
        b = dict(gender=1, age=25, muscle=0.3, weight=0.38, height=0.55)
        b.update(body or {})
        self.body = Body(**b)
        self.ch = self                 # Actor uses actor.ch.body
        self.folds = False
        self.kind, self.female, self.name = kind, female, preset
        self.lut = {n: i for i, n in enumerate(bones())}
        self.fills = {"skin": WHITE, "cloth": WHITE, "hair": DARK, "shoe": WHITE}
        self.fills.update(fills or {})
        self.pieces: list[_Piece] = []
        self.seams = []                # (piece index, vertex index array, weight)
        self._build()
        self.face = ToonFace(self.head, kind=kind, female=female, seed=seed + 3)

    # helpers ---------------------------------------------------------------------------------
    def j(self, n, tail=False):
        i = self.lut[n]
        return (self.body.tails if tail else self.body.heads)[i].astype(float)

    def _add(self, name, V, T, W, **kw):
        self.pieces.append(_Piece(name, V, T, W, **kw))
        return len(self.pieces) - 1

    def _seam(self, k, idx, w=0.55):
        self.seams.append((k, np.asarray(idx), w))

    # build -------------------------------------------------------------------------------------
    def _build(self):
        P, b = self.P, self.body
        neck = self.j("neck01")
        top_sh = self.j("clavicle.L", True)
        # ---------------------------------------------------------------- head (ball on a short neck)
        r = P["head_r"]
        hc = np.array([0.0, neck[1] + 0.45 + P["neck_len"] + r * 0.92, neck[2] + 0.2])
        self.head = BallHead(hc, r)
        Vh, Th = sphere(hc, r)
        Wh = chain_weights(Vh, b, ["head"])
        self.k_head = self._add("toon_head", Vh, Th, Wh, fill="skin", min_jump=0.25, smooth=0)
        # neck stub (hidden mostly) — from collar to head
        C = np.linspace(neck + [0, 0.1, 0.05], hc + [0, -r * 0.7, -0.1], 6)
        Vn, Tn, _ = tube(C, np.full(6, 0.42), 20)
        self._add("neck", Vn, Tn, chain_weights(Vn, b, ["neck01", "neck02", "neck03", "head"]), fill="skin")
        # rolled turtleneck collar: short fat tube around the neck base
        cy = neck[1] - 0.05
        C = np.array([[0, cy - 0.3, neck[2] - 0.08], [0, cy + 0.1, neck[2] - 0.02], [0, cy + 0.42, neck[2] + 0.02],
                      [0, cy + 0.62, neck[2] + 0.04]])
        Vc, Tc, (nr, ns) = tube(C, np.array([0.9, 0.95, 0.9, 0.7]), 28, ref=[1, 0, 0],
                                rad2=np.array([0.78, 0.82, 0.78, 0.6]))
        k = self._add("collar", Vc, Tc, chain_weights(Vc, b, ["spine01", "neck01"], power=3), min_jump=0.12)
        self._seam(k, np.r_[np.arange(ns, 2 * ns), ns], 0.45)
        # ---------------------------------------------------------------- sweater torso
        hem = P["hem_y"]
        ys = np.array([hem + P["hem_h"], hem + 0.9, 10.6, 11.6, 12.6, 13.4, 14.0, 14.5, 14.85, 15.1, 15.3])
        wx = np.array([1.78, 1.78, 1.74, 1.7, 1.7, 1.76, 1.84, 1.84, 1.68, 1.28, 0.75]) * P["torso_w"] / 1.62
        dz = np.array([1.4, 1.38, 1.32, 1.24, 1.22, 1.22, 1.14, 1.02, 0.9, 0.78, 0.6]) * P["torso_d"] / 1.22
        zc = np.array([0.18, 0.2, 0.22, 0.2, 0.18, 0.12, 0.05, 0.0, 0.0, 0.0, 0.0])
        C = np.c_[np.zeros_like(ys), ys, zc]
        Vt, Tt, (nr, ns) = tube(C, wx, 36, cap0=False, cap1=True, ref=[1, 0, 0], rad2=dz)
        spine = ["root", "spine05", "spine04", "spine03", "spine02", "spine01"]
        Wt = chain_weights(Vt, b, spine, power=3)
        # shoulders: blend toward the clavicles near the top sides
        side = np.clip((np.abs(Vt[:, 0]) - 0.9) / 0.8, 0, 1) * np.clip((Vt[:, 1] - 13.8) / 0.8, 0, 1)
        for sd, sgn in (("L", 1), ("R", -1)):
            m = side * (np.sign(Vt[:, 0]) == sgn)
            Wt *= (1 - 0.6 * m)[:, None]
            Wt[:, self.lut[f"clavicle.{sd}"]] += 0.6 * m
        k = self._add("sweater", Vt, Tt, Wt, min_jump=0.3, group="sweater")
        # ribbed hem band (separate slightly-tighter ring so its top edge draws a line)
        C = np.c_[np.zeros(3), np.array([hem - 0.02, hem + P["hem_h"] * 0.5, hem + P["hem_h"]]), np.full(3, 0.18)]
        Vb, Tb, (nr, ns) = tube(C, np.array([1.76, 1.82, 1.84]) * P["torso_w"] / 1.62, 36, cap0=True, cap1=False,
                                ref=[1, 0, 0], rad2=np.array([1.3, 1.34, 1.36]) * P["torso_d"] / 1.22)
        k = self._add("hem", Vb, Tb, chain_weights(Vb, b, ["root", "spine05"], power=3), min_jump=0.12)
        self._seam(k, np.arange(2 * ns, 3 * ns), 0.5)
        # ---------------------------------------------------------------- sleeves + cuffs + hands
        for sd in ("L", "R"):
            self._arm(sd)
        # ---------------------------------------------------------------- trousers + shoes
        for sd in ("L", "R"):
            self._leg(sd)

    def _arm(self, sd):
        P, b = self.P, self.body
        sh = self.j(f"shoulder01.{sd}") * 0.75 + self.j(f"upperarm01.{sd}") * 0.25 + np.array([0, -0.35, -0.02])
        el = self.j(f"lowerarm01.{sd}")
        wr = self.j(f"wrist.{sd}")
        u = np.linspace(0, 1, 9)
        cu = wr + (el - wr) / np.linalg.norm(el - wr) * P["cuff_len"]
        C1 = sh[None] * (1 - u[:, None]) + el[None] * u[:, None]
        u2 = np.linspace(0, 1, 9)[1:]
        C2 = el[None] * (1 - u2[:, None]) + cu[None] * u2[:, None]
        C = np.vstack([C1, C2])
        s = np.linspace(0, 1, len(C))
        # upper arm full, forearm balloons, then gathers hard into the cuff
        rr = np.interp(s, [0, 0.15, 0.45, 0.72, 0.9, 1.0],
                       [P["sleeve_r"] * 0.85, P["sleeve_r"], P["sleeve_r"] * 1.04, P["sleeve_puff"],
                        P["sleeve_puff"] * 0.92, P["cuff_r"] * 1.15])
        V, T, (nr, ns) = tube(C, rr, 26, cap0=True, cap1=True)
        names = [f"shoulder01.{sd}", f"upperarm01.{sd}", f"upperarm02.{sd}", f"lowerarm01.{sd}", f"lowerarm02.{sd}"]
        W = chain_weights(V, b, names + ["spine01"], power=4)
        self._add(f"sleeve.{sd}", V, T, W, min_jump=0.3, group="sweater")
        # cuff: short band, slightly smaller than the sleeve end → a clear gathered edge
        d = (wr - el) / np.linalg.norm(wr - el)
        C = np.array([cu - d * 0.05, cu + d * P["cuff_len"] * 0.5, wr + d * 0.05])
        Vc, Tc, (nr, ns) = tube(C, np.array([P["cuff_r"] * 1.05, P["cuff_r"], P["cuff_r"] * 0.96]), 22)
        k = self._add(f"cuff.{sd}", Vc, Tc, chain_weights(Vc, b, [f"lowerarm02.{sd}", f"wrist.{sd}"], power=4),
                      min_jump=0.12)
        self._seam(k, np.arange(0, ns), 0.5)
        self._hand(sd)

    def _hand(self, sd):
        P, b = self.P, self.body
        S = P["hand"]
        wr = self.j(f"wrist.{sd}")

        def sc(p):
            return wr + (p - wr) * S
        # palm: rounded box between wrist and the knuckles
        kn = np.mean([self.j(f"finger{i}-1.{sd}") for i in (2, 3, 4, 5)], 0)
        Vs, Ts = [], []
        pc = sc((wr + kn) / 2)
        ax = (kn - wr) / np.linalg.norm(kn - wr)
        acr = self.j(f"finger2-1.{sd}") - self.j(f"finger5-1.{sd}")
        acr /= np.linalg.norm(acr)
        nrm = np.cross(ax, acr)
        L = np.linalg.norm(kn - wr) * S
        Vp, Tp = blob([0, 0, 0], (L * 0.7, 0.34 * S, 0.62 * S), 22, 14, p=2.2, bottom_flat=0)
        # local blob axes: x along the hand, y = palm normal, z across
        R = np.c_[ax, nrm, acr]
        Vp = Vp @ R.T + pc
        Vs.append(Vp)
        Ts.append(Tp)
        Wp = chain_weights(Vp, b, [f"wrist.{sd}"])
        parts_W = [Wp]
        # fingers: thumb + 4 sausages along their bone chains (scaled about the wrist)
        for f, rad in ((1, 0.23), (2, 0.24), (3, 0.25), (4, 0.24), (5, 0.0)):
            if rad <= 0:
                continue
            chain = ([f"finger1-1.{sd}", f"finger1-2.{sd}", f"finger1-3.{sd}"] if f == 1 else
                     [f"finger{f}-1.{sd}", f"finger{f}-2.{sd}", f"finger{f}-3.{sd}"])
            pts = [self.j(chain[0])] + [self.j(n, True) for n in chain]
            if f == 1:
                pts[0] = (pts[0] * 0.5 + self.j(f"wrist.{sd}") * 0.5)
            pts = np.array([sc(p) for p in pts])
            # resample
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            tt = np.linspace(0, seg[-1], 8)
            C = np.c_[[np.interp(tt, seg, pts[:, k]) for k in range(3)]].T
            rr = np.full(len(C), rad * S)
            rr[-1] *= 0.85
            C = C[:5]
            rr = rr[:5]
            rr[-1] = rad * S * 0.9
            Vf, Tf, _ = tube(C, rr, 12)
            Wf = chain_weights(Vf, b, chain + [f"wrist.{sd}"], power=4)
            Vs.append(Vf)
            Ts.append(Tf)
            parts_W.append(Wf)
        off = 0
        TT = []
        for V_, T_ in zip(Vs, Ts):
            TT.append(T_ + off)
            off += len(V_)
        self._add(f"hand.{sd}", np.vstack(Vs), np.vstack(TT), np.vstack(parts_W), fill="skin", min_jump=0.12,
                  weight=0.9, smooth=1)

    def _leg(self, sd):
        P, b = self.P, self.body
        sg = 1 if sd == "L" else -1
        hip = self.j(f"upperleg01.{sd}")
        kn = self.j(f"lowerleg01.{sd}")
        an = self.j(f"lowerleg02.{sd}", True)
        top = np.array([sg * 0.78, P["hem_y"] + 1.0, 0.15])
        hip2 = np.array([hip[0] * 0.85, hip[1] - 1.0, hip[2] + 0.12])
        bot = an + np.array([0, P["pants_hem"] - an[1], 0.05])
        pts = np.array([top, hip2, kn + [0, 0, 0.05], bot])
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        tt = np.linspace(0, seg[-1], 22)
        C = np.c_[[np.interp(tt, seg, pts[:, k]) for k in range(3)]].T
        s = tt / seg[-1]
        rr = P["thigh_r"] * (1 - s) + P["ankle_r"] * s
        rr = rr * (1 + 0.06 * np.sin(s * np.pi * 3.0) * (s > 0.5))      # slight bunching near the ankle
        V, T, (nr, ns) = tube(C, rr, 28, cap0=True, cap1=True, ref=[1, 0, 0], rad2=rr * 0.96)
        names = ["root", f"upperleg01.{sd}", f"upperleg02.{sd}", f"lowerleg01.{sd}", f"lowerleg02.{sd}"]
        W = chain_weights(V, b, names, power=4)
        k = self._add(f"pants.{sd}", V, T, W, min_jump=0.3)
        self._pants_creases(k, nr, ns, sg)
        # shoe: big rounded blob, heel on foot.X, front on the toes
        ft = self.j(f"foot.{sd}")
        toe = self.j(f"toe3-1.{sd}")
        L, w, h = P["shoe_len"], P["shoe_w"], P["shoe_h"]
        fwd = np.array([toe[0] - ft[0], 0, toe[2] - ft[2]])
        fwd /= np.linalg.norm(fwd)
        heel = np.array([ft[0], 0, ft[2]]) - fwd * 0.75
        n = 16
        u = np.linspace(0, 1, n)
        C = heel[None] + fwd[None] * (u * L)[:, None]
        # height profile: ankle collar at the back, lower toe box, round tip
        hh = np.interp(u, [0, 0.06, 0.35, 0.6, 0.85, 0.97, 1.0], [0.55, 0.92, 1.0, 0.78, 0.66, 0.45, 0.12]) * h * 0.5
        ww = np.interp(u, [0, 0.05, 0.3, 0.6, 0.85, 0.97, 1.0], [0.55, 0.86, 0.9, 1.0, 0.98, 0.7, 0.15]) * w
        C[:, 1] = hh                                   # centre height = half of the section height
        Vs, Ts, (nr, ns) = tube(C, hh, 26, ref=[0, 1, 0], rad2=ww)
        Vs[:, 1] = np.maximum(Vs[:, 1], 0.0)           # flat sole on the ground
        Wf = chain_weights(Vs, b, [f"lowerleg02.{sd}", f"foot.{sd}", f"toe3-1.{sd}"], power=3)
        # ankle/heel part follows the shin a bit less: keep the shoe rigid on foot + toes mostly
        k = self._add(f"shoe.{sd}", Vs, Ts, Wf, fill="shoe", min_jump=0.15)
        # sole seam: a ring of points at sole height
        js = np.arange(ns)
        ring = []
        for i in range(1, nr - 1):
            ring.append(i * ns + int(np.argmin(np.abs(Vs[i * ns:(i + 1) * ns, 1] - 0.0 - 0.18) +
                                                  (Vs[i * ns:(i + 1) * ns] - C[i]) @ np.cross(fwd, [0, 1, 0]) * 0)))
        side = np.cross([0, 1.0, 0], fwd)
        for sgn in (1, -1):
            idx = []
            for i in range(1, nr - 1):
                R = Vs[i * ns:(i + 1) * ns]
                score = np.abs(R[:, 1] - 0.22) + 3 * np.maximum(0, -sgn * ((R - C[i]) @ side))
                idx.append(i * ns + int(np.argmin(score)))
            self._seam(k, np.array(idx), 0.42)

    def _pants_creases(self, k, nr, ns, sg):
        # a few short soft fold lines (knee, ankle) drawn ON the trouser surface
        def ring_arc(i, a0, a1):
            js = np.round(np.linspace(a0, a1, 6) * ns).astype(int) % ns
            return i * ns + js
        self._seam(k, ring_arc(int(nr * 0.55), 0.12, 0.3), 0.32)
        self._seam(k, ring_arc(int(nr * 0.62), 0.62, 0.78), 0.3)
        self._seam(k, ring_arc(int(nr * 0.86), 0.15, 0.32), 0.3)
        self._seam(k, ring_arc(int(nr * 0.9), 0.66, 0.85), 0.3)

    # posing ------------------------------------------------------------------------------------
    def rig_parts(self, G, base):
        """posed Parts (+ seam strokes) for the line renderer. pid = base + piece index (+1)."""
        if not hasattr(self, "_gid"):
            self._gid = {}
            for k, pc in enumerate(self.pieces):
                if pc.group and pc.group not in self._gid:
                    self._gid[pc.group] = k
        G12 = G.reshape(len(G), 12)
        parts, strokes = [], []
        posed = []
        for k, pc in enumerate(self.pieces):
            M = (pc.W @ G12).reshape(-1, 3, 4)
            X = np.einsum("nij,nj->ni", M[:, :, :3], pc.V) + M[:, :, 3]
            posed.append(X)
            pid = base + 1 + k
            parts.append(Part(pc.name, X.astype(np.float32), pc.T, pc.topo, pid=pid,
                              smooth_iters=pc.smooth, min_jump=pc.min_jump, weight=pc.weight))
        for k, idx, w in self.seams:
            pid = base + 1 + k
            strokes.append(Stroke3D(posed[k][idx], w, "detail", 0.35, True, pid))
        self.groups = {base + 1 + k: (base + 1 + self._gid[pc.group] if pc.group else base + 1 + k)
                       for k, pc in enumerate(self.pieces)}
        return parts, strokes, base + 1 + self.k_head

    def fill_for(self, name):
        if name in ("toon_head", "neck") or name.startswith("hand"):
            return self.fills["skin"]
        if name.startswith("shoe"):
            return self.fills["shoe"]
        if name.startswith("hair"):
            return self.fills["hair"]
        return self.fills["cloth"]
