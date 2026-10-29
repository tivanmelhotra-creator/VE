"""
mograph.toon.head — the cartoon HEAD for toon characters (webcomic-short style: a big clean ball head).

The realistic MakeHuman head is shrunk inside a big round head (a smooth sphere glued to the head bone), so the
silhouette is a clean circle that turns, nods and tilts exactly with the mocap. Optional hair is a real 3D shell
(cap / long curtain / bun / ponytail) that occludes and is occluded correctly; long hair is skinned between head
and chest so it hangs instead of rotating rigidly with every nod.

The FACE is drawn on the head surface: every feature is a 2D shape in (u, v) = (yaw, pitch) angles on the head,
mapped onto the sphere, posed by the head bone and projected — so eyes and mouth slide round the head in 3/4,
foreshorten on a nod and disappear behind the head edge, while still being crisp 2D cartoon drawings with fills.
"""
from __future__ import annotations

import math

import numpy as np

from ..human.body import BODY_N, bone_index
from ..human.lineart import MeshTopo, Part


def _grid_tris(rows, cols, wrap=False):
    T = []
    cc = cols if wrap else cols - 1
    for i in range(rows - 1):
        for j in range(cc):
            a, b = i * cols + j, i * cols + (j + 1) % cols
            c, d = (i + 1) * cols + j, (i + 1) * cols + (j + 1) % cols
            T += [(a, c, b), (b, c, d)]
    return np.asarray(T, np.int32)


def _sphere(nu=48, nv=30):
    v = np.linspace(-math.pi / 2 + 0.02, math.pi / 2 - 0.02, nv)
    u = np.linspace(-math.pi, math.pi, nu, endpoint=False)
    U, V = np.meshgrid(u, v)
    P = np.c_[(np.sin(U) * np.cos(V)).ravel(), np.sin(V).ravel(), (np.cos(U) * np.cos(V)).ravel()]
    T = _grid_tris(nv, nu, wrap=True)
    bot, top = len(P), len(P) + 1
    P = np.vstack([P, [[0, -1, 0], [0, 1, 0]]])
    F = [(bot, (j + 1) % nu, j) for j in range(nu)] + [(top, (nv - 1) * nu + j, (nv - 1) * nu + (j + 1) % nu)
                                                        for j in range(nu)]
    return P, np.vstack([T, np.asarray(F, np.int32)])


HAIR_STYLES = ("bald", "short", "long", "bun", "ponytail", "bob")


class ToonHead:
    """big round head + optional hair + face frame for one Character (built once, posed every frame)."""

    def __init__(self, ch, scale=1.5, hair="bald", squash=(1.0, 1.02, 0.98), shrink=0.62, seed=0, hair_len=1.0):
        self.ch = ch
        X = ch.body.rest
        self.hi = bone_index("head")
        self.ci = bone_index("spine02")
        hv = np.nonzero(ch.R.head[:BODY_N] > 0.6)[0]
        P = X[hv]
        lo, hi_ = P.min(0), P.max(0)
        hh = float(hi_[1] - lo[1])
        self.r = 0.5 * hh * scale
        # the ball sits on the neck: its bottom stays near the real chin, it grows upward
        c = np.array([0.0, lo[1] + self.r * 0.93, float(np.median(P[:, 2])) + 0.06 * hh])
        self.c = c
        self.sq = np.asarray(squash, float)
        self.shrink = shrink
        self.hair = hair
        self.seed = seed
        self.w_head = np.clip(ch.R.head, 0, 1).astype(np.float32)[:, None]
        sp, st = _sphere()
        self.sph = self.c + sp * self.r * self.sq
        self.sph_tris = st
        self.sph_topo = MeshTopo(st, len(sp))
        self.hair_parts = []
        self._build_hair(hair, hair_len)

    def _surf(self, u, v, k=1.0):
        d = np.stack([np.sin(u) * np.cos(v), np.sin(v), np.cos(u) * np.cos(v)], -1)
        return self.c + d * self.r * self.sq * k

    def _build_hair(self, style, L):
        if style == "bald":
            return
        nu, nv = 56, 16
        u = np.linspace(-math.pi, math.pi, nu, endpoint=False)
        k = 1.05
        au = np.abs(u)
        front = np.clip(1 - au / 1.25, 0, 1)
        if style == "short":
            base = 0.5 * front + (-0.05) * (1 - front) - 0.3 * np.clip((au - 2.0) / 1.1, 0, 1)
        else:
            base = (0.42 + 0.08 * np.sin(u * 2.0 + 0.6)) * front + 0.0 * (1 - front)
            base -= 0.45 * np.clip((au - 1.9) / 1.2, 0, 1)
        V = np.zeros((nv, nu))
        for j in range(nu):
            V[:, j] = np.linspace(math.pi / 2 - 0.03, base[j], nv)
        U = np.repeat(u[None], nv, 0)
        P = self._surf(U, V, k).reshape(-1, 3)
        parts = [("hair_cap", P, _grid_tris(nv, nu, wrap=True), np.ones(len(P), np.float32), True)]
        if style in ("long", "bob"):
            nuu, nvv = 44, 14
            uu = np.linspace(-math.pi * 0.6, math.pi * 0.6, nuu) + math.pi
            uu = (uu + math.pi) % (2 * math.pi) - math.pi
            drop = (2.3 if style == "long" else 1.1) * L * self.r
            Pc = []
            for i in range(nvv):
                s = i / (nvv - 1)
                for j in range(nuu):
                    a = uu[j]
                    rr = self.r * k * (1.0 + 0.08 * s)
                    y = self.c[1] + 0.05 * self.r - s * drop
                    pt = np.array([self.c[0] + math.sin(a) * rr * self.sq[0], y,
                                   self.c[2] + math.cos(a) * rr * self.sq[2]])
                    if i == nvv - 1:
                        pt[1] += 0.1 * self.r * math.sin(j * 2.3)
                    Pc.append(pt)
            Pc = np.asarray(Pc)
            sy = (self.c[1] - Pc[:, 1]) / max(1e-6, drop)
            parts.append(("hair_long", Pc, _grid_tris(nvv, nuu), np.clip(1 - sy * 1.15, 0, 1).astype(np.float32),
                          True))
        if style == "bun":
            cb = self._surf(np.array(math.pi * 0.92), np.array(0.7), 1.1)
            sp, st = _sphere(20, 12)
            parts.append(("hair_bun", cb + sp * self.r * 0.36, st, np.ones(len(sp), np.float32), False))
        if style == "ponytail":
            cb = self._surf(np.array(math.pi), np.array(0.3), 1.04)
            n, m = 10, 12
            Pp = []
            for i in range(n):
                s = i / (n - 1)
                cen = cb + np.array([0, -s * 1.6 * self.r * L, -0.3 * self.r * math.sin(s * 2.2)])
                rad = self.r * (0.2 + 0.12 * math.sin(s * math.pi) - 0.1 * s)
                for j in range(m):
                    a = 2 * math.pi * j / m
                    Pp.append(cen + rad * np.array([math.cos(a), 0, math.sin(a)]))
            Pp = np.asarray(Pp)
            Wp = np.clip(1 - np.linspace(0, 1, n).repeat(m) * 0.9, 0, 1).astype(np.float32)
            parts.append(("hair_tail", Pp, _grid_tris(n, m, wrap=True), Wp, False))
        for name, P_, T_, W_, bnd in parts:
            self.hair_parts.append((name, P_.astype(np.float32), T_, MeshTopo(T_, len(P_)), W_, bnd))

    @staticmethod
    def _xf(M, P):
        return P @ M[:, :3].T + M[:, 3]

    def center(self, G):
        return self._xf(G[self.hi], self.c[None])[0]

    def shrink_mesh(self, G, X, hand_scale=1.0):
        cpos = self.center(G)
        X = X + self.w_head * self.shrink * (cpos - X)
        if hand_scale != 1.0:
            # cartoon hands: grow each hand about its wrist (soft mask → no seam at the cuff)
            if not hasattr(self, "_hand_w"):
                from ..human.body import bone_index as bi
                R = self.ch.R
                Xr = self.ch.body.rest
                self._hand_w = {}
                for sd in ("L", "R"):
                    side = (Xr[:, 0] > 0) if sd == "L" else (Xr[:, 0] < 0)
                    self._hand_w[sd] = (np.clip(R.hand, 0, 1) * side).astype(np.float32)[:, None]
                self._wi = {sd: bi(f"wrist.{sd}") for sd in ("L", "R")}
            b = self.ch.body
            for sd in ("L", "R"):
                i = self._wi[sd]
                w = G[i, :, :3] @ b.heads[i] + G[i, :, 3]
                X = X + self._hand_w[sd] * (hand_scale - 1.0) * (X - w)
        return X

    def parts(self, G, pid_head, pid_hair):
        Mh = G[self.hi]
        out = [Part("toon_head", self._xf(Mh, self.sph).astype(np.float32), self.sph_tris, self.sph_topo,
                    pid=pid_head, smooth_iters=0, min_jump=0.25, weight=1.0)]
        Mc = G[self.ci]
        for k, (name, P, T, topo, Wh, bnd) in enumerate(self.hair_parts):
            A = self._xf(Mh, P)
            if (Wh < 0.999).any():
                B = self._xf(Mc, P)
                A = A * Wh[:, None] + B * (1 - Wh[:, None])
            out.append(Part(name, A.astype(np.float32), T, topo, pid=pid_hair + min(k, 3), smooth_iters=1,
                            min_jump=0.4, weight=1.0, boundary=bnd))
        return out

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
