"""
mograph.real3d_models — procedural TRUE-3D product models (geometry + procedural textures sampled from the
reference photo).  Every model is a closed solid: it can turn 360° / be seen from above, below and behind.

  soda_can      lathe body (dome bottom, stand ring, chime, neck, double-seam rim, recessed lid) + pull tab + rivet,
                brushed aluminium + condensation droplets (bump + gloss)
  pizza_slice   polar height-field slab (melted cheese lumps, sauce at the edges, baked dough underside and cut walls),
                puffy lumpy crust tube with pores, pepperoni (cupped, oily, fat specks), mushroom slices, olive ring,
                green pepper strip
  growth_card   two stacked rounded cards (matte plastic) with embossed UI bars / dots and an extruded, bevelled
                zig-zag growth arrow

Reference palette (sampled from work/input/three_objects/ref.jpg): crust (177,109,62) cheese (223,189,143)
pepperoni (171,94,50) can (142,140,134) card (215,216,207) arrow (212,124,94) olive (102,31,0).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .real3d import Mesh, Model, _norm, fbm, grid_faces, lathe, noise_tex, rot, samp, smooth, srgb, xform


# ============================================================================================ helpers
def vnormals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return _norm(n)


def oriented(v, f, expect):
    """Mesh with geometric normals, flipped (normals AND winding) so they agree with `expect` (n,3) or (3,)."""
    n = vnormals(v, f)
    e = np.broadcast_to(np.asarray(expect, np.float32), n.shape)
    if np.mean(np.sum(n * e, 1)) < 0:
        n = -n
        f = f[:, ::-1]
    return Mesh(v, n, v[:, [0, 2]], f)


def merge(*ms):
    v, n, uv, f, lp = [], [], [], [], []
    off = 0
    for m in ms:
        v.append(m.v)
        n.append(m.n)
        uv.append(m.uv)
        lp.append(getattr(m, "lp", m.v))
        f.append(m.f + off)
        off += len(m.v)
    out = Mesh(np.concatenate(v), np.concatenate(n), np.concatenate(uv), np.concatenate(f))
    out.lp = np.concatenate(lp)
    return out


def rbox(w, h, d, b=0.03, n_in=4, n_b=6):
    """Rounded box with ANALYTIC normals and dense samples inside the bevel band (smooth highlights)."""
    hw, hh, hd = w / 2, h / 2, d / 2
    b = min(b, 0.49 * min(hw, hh, hd))

    def axis(h_):
        band = [h_ - b + b * math.sin(t) for t in np.linspace(0, math.pi / 2, n_b)]
        inner = list(np.linspace(-(h_ - b), h_ - b, n_in + 2)[1:-1])
        return np.array([-x for x in band[::-1]] + inner + band, np.float32)

    ax = [axis(hw), axis(hh), axis(hd)]
    half = np.array([hw, hh, hd], np.float32)
    inner = half - b
    parts = []
    for k in range(3):                              # axis of the face normal
        i, j = [a for a in range(3) if a != k]
        for sgn in (1, -1):
            U, V = np.meshgrid(ax[i], ax[j], indexing="ij")
            raw = np.zeros(U.shape + (3,), np.float32)
            raw[..., i] = U
            raw[..., j] = V
            raw[..., k] = sgn * half[k]
            raw = raw.reshape(-1, 3)
            cin = np.clip(raw, -inner, inner)
            dlt = raw - cin
            L = np.linalg.norm(dlt, axis=1, keepdims=True)
            nrm = np.where(L > 1e-7, dlt / np.maximum(L, 1e-9), 0)
            nrm[L[:, 0] <= 1e-7, k] = sgn
            pos = cin + nrm * b
            f = grid_faces(len(ax[i]), len(ax[j]))
            m = Mesh(pos, nrm, pos[:, [i, j]], f)
            # winding: make geometric normal agree with analytic one
            g = np.cross(pos[f[:, 1]] - pos[f[:, 0]], pos[f[:, 2]] - pos[f[:, 0]])
            if np.mean(np.sum(g * nrm[f[:, 0]], 1)) < 0:
                m.f = f[:, ::-1]
            parts.append(m)
    return merge(*parts)


def _earclip(P):
    n = len(P)
    idx = list(range(n))
    tris = []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    guard = 0
    while len(idx) > 3 and guard < 20000:
        guard += 1
        found = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = P[i0], P[i1], P[i2]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(inside(P[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            found = True
            break
        if not found:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.array(tris, np.int32)


def _area(P):
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def bevel_prism(outer, inner, depth, b):
    """Extrude a 2D polygon along +Z with a 45° chamfer.  `inner` = the same polygon inset by b (same vertex
    count / order).  Front cap at z = +depth/2 (inner), back cap at -depth/2 (inner)."""
    O = np.array(outer, np.float32)
    I = np.array(inner, np.float32)
    if _area(O) < 0:
        O, I = O[::-1].copy(), I[::-1].copy()
    n = len(O)
    hz = depth / 2
    tri = _earclip(I)
    parts = []
    for z, s in ((hz, 1), (-hz, -1)):
        v = np.column_stack([I, np.full(n, z)]).astype(np.float32)
        nn = np.tile(np.array([0, 0, s], np.float32), (n, 1))
        parts.append(Mesh(v, nn, I, tri if s > 0 else tri[:, ::-1]))
    sv, sn, sf = [], [], []
    for i in range(n):
        j = (i + 1) % n
        e = O[j] - O[i]
        en = np.array([e[1], -e[0], 0.0], np.float32)
        en /= max(1e-9, np.linalg.norm(en))
        rings = [  # (point_i, point_j, z, normal)
            (I[i], I[j], hz, _norm(en + np.array([0, 0, 1.0], np.float32))),
            (O[i], O[j], hz - b, _norm(en + np.array([0, 0, 1.0], np.float32))),
            (O[i], O[j], hz - b, en),
            (O[i], O[j], -hz + b, en),
            (O[i], O[j], -hz + b, _norm(en + np.array([0, 0, -1.0], np.float32))),
            (I[i], I[j], -hz, _norm(en + np.array([0, 0, -1.0], np.float32))),
        ]
        for q in range(0, 6, 2):
            a, bb = rings[q], rings[q + 1]
            base = len(sv)
            sv += [[*a[0], a[2]], [*a[1], a[2]], [*bb[1], bb[2]], [*bb[0], bb[2]]]
            sn += [a[3], a[3], bb[3], bb[3]]
            sf += [(base, base + 3, base + 2), (base, base + 2, base + 1)]
    sv = np.array(sv, np.float32)
    m = Mesh(sv, np.array(sn, np.float32), sv[:, :2], np.array(sf, np.int32))
    g = np.cross(sv[m.f[:, 1]] - sv[m.f[:, 0]], sv[m.f[:, 2]] - sv[m.f[:, 0]])
    if np.mean(np.sum(g * m.n[m.f[:, 0]], 1)) < 0:
        m.f = m.f[:, ::-1]
    parts.append(m)
    return merge(*parts)


def stroke_arrow(pts, w, head_w, head_len):
    """Polyline → simple polygon of a thick stroke with an arrow head at the end (miter joins)."""
    P = np.array(pts, np.float32)
    L, Rr = [], []
    for k in range(len(P)):
        if k == 0:
            d = P[1] - P[0]
            d /= np.linalg.norm(d)
            nrm = np.array([-d[1], d[0]])
            L.append(P[0] + nrm * w)
            Rr.append(P[0] - nrm * w)
            continue
        if k == len(P) - 1:
            d = P[k] - P[k - 1]
            d /= np.linalg.norm(d)
            nrm = np.array([-d[1], d[0]])
            L.append(P[k] + nrm * w)
            Rr.append(P[k] - nrm * w)
            continue
        d0 = P[k] - P[k - 1]
        d0 /= np.linalg.norm(d0)
        d1 = P[k + 1] - P[k]
        d1 /= np.linalg.norm(d1)
        n0 = np.array([-d0[1], d0[0]])
        n1 = np.array([-d1[1], d1[0]])
        m = n0 + n1
        m /= np.linalg.norm(m)
        ml = w / max(0.35, float(np.dot(m, n0)))
        L.append(P[k] + m * ml)
        Rr.append(P[k] - m * ml)
    d = P[-1] - P[-2]
    d /= np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    end = P[-1]
    head = [end + nrm * head_w, end + d * head_len, end - nrm * head_w]
    poly = L + head + Rr[::-1]
    return np.array(poly, np.float32)


def disc_y(r, t, bevel=0.01, dome=0.0, cup=0.0, segs=48):
    """Flat disc on the XZ plane (axis +Y): rounded edge, optional dome (top bulge) / cup (rim raised)."""
    b = min(bevel, t * 0.49, r * 0.4)
    prof = [(0.0, -t / 2 + cup * 0.5)]
    for k in range(1, 6):
        rr = (r - b) * k / 5
        prof.append((rr, -t / 2 + cup * (0.5 - 0.5 * (rr / r) ** 2) * 0 + 0))
    prof = [(0.0, -t / 2)]
    for k in range(0, 6):
        a = -math.pi / 2 + k * (math.pi / 2) / 5
        prof.append((r - b + b * math.cos(a), -t / 2 + b + b * math.sin(a)))
    for k in range(0, 6):
        a = k * (math.pi / 2) / 5
        prof.append((r - b + b * math.cos(a), t / 2 - b + b * math.sin(a)))
    for k in range(1, 7):
        rr = (r - b) * (1 - k / 6)
        y = t / 2 + dome * (1 - (rr / r) ** 2) + cup * (rr / r) ** 2
        prof.append((rr, y))
    return lathe(prof, segs)


def torus_y(R, r, su=48, sv=20):
    u = np.linspace(0, 2 * math.pi, su + 1).astype(np.float32)
    v = np.linspace(0, 2 * math.pi, sv + 1).astype(np.float32)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    cx = np.sin(U)
    cz = np.cos(U)
    pos = np.stack([(R + r * np.cos(Vv)) * cx, r * np.sin(Vv), (R + r * np.cos(Vv)) * cz], -1).reshape(-1, 3)
    nrm = np.stack([np.cos(Vv) * cx, np.sin(Vv), np.cos(Vv) * cz], -1).reshape(-1, 3)
    f = grid_faces(su + 1, sv + 1)
    m = Mesh(pos, nrm, np.stack([U.ravel() / (2 * math.pi), Vv.ravel() / (2 * math.pi)], 1), f)
    g = np.cross(pos[f[:, 1]] - pos[f[:, 0]], pos[f[:, 2]] - pos[f[:, 0]])
    if np.mean(np.sum(g * nrm[f[:, 0]], 1)) < 0:
        m.f = f[:, ::-1]
    return m


def _hash2(i, j, seed=0):
    h = (i * 374761393 + j * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return (h & 0xFFFF) / 65535.0, ((h >> 16) & 0xFFFF) / 65535.0


def cells(u, v, nu, nv, seed=0, density=1.0):
    """Cellular dots: for each point → (distance to the cell's dot centre / cell size, dot radius frac, present)."""
    x = u * nu
    y = v * nv
    i = np.floor(x).astype(np.int64)
    j = np.floor(y).astype(np.int64)
    a, b = _hash2(i, j, seed)
    c, d = _hash2(i + 7, j + 13, seed + 1)
    cx = i + 0.2 + 0.6 * a
    cy = j + 0.2 + 0.6 * b
    dist = np.hypot(x - cx, y - cy)
    rad = 0.08 + 0.22 * c
    present = d < density
    return dist, rad, present


# ============================================================================================ SODA CAN
@lru_cache(maxsize=2)
def soda_can():
    R = 0.55
    prof = [(0.0, -0.90), (0.20, -0.915), (0.33, -0.95), (0.39, -0.99), (0.415, -1.0), (0.44, -0.995),
            (0.49, -0.965), (0.53, -0.92), (0.548, -0.87), (R, -0.82),
            (R, 0.62), (0.545, 0.68), (0.52, 0.75), (0.49, 0.82), (0.465, 0.88), (0.455, 0.915),
            (0.462, 0.935), (0.466, 0.955), (0.46, 0.975), (0.445, 0.982), (0.432, 0.97), (0.428, 0.95),
            (0.42, 0.932), (0.40, 0.928), (0.0, 0.93)]
    body = lathe(prof, 96)
    # pull tab: rounded plate with a finger hole (two plates), rivet
    tab = rbox(0.30, 0.018, 0.15, b=0.008, n_in=3, n_b=4)
    tab = xform(tab, rot(-4, 0, 0), (0.0, 0.952, 0.06))
    ring = torus_y(0.055, 0.012, 32, 10)
    ring = xform(ring, None, (0.0, 0.955, -0.02))
    rivet = disc_y(0.03, 0.02, bevel=0.009, dome=0.006, segs=24)
    rivet = xform(rivet, None, (0.0, 0.94, 0.06))
    opening = disc_y(0.09, 0.006, bevel=0.003, segs=32)
    opening = xform(opening, rot(0, 0, 0), (0.0, 0.931, 0.24))

    def alu(uv, lp, n):
        y = lp[:, 1]
        ang = np.arctan2(lp[:, 0], lp[:, 2]) / (2 * math.pi) + 0.5
        brushed = fbm(ang * 1.0, y * 18.0, 3, seed=3)
        base = srgb(214, 214, 210) * (0.93 + 0.10 * brushed[:, None])
        rough = 0.30 + 0.08 * brushed
        # condensation droplets on the body only
        body_m = smooth(-0.84, -0.78, y) * (1 - smooth(0.58, 0.66, y))
        h = np.zeros(len(y), np.float32)
        for nu, nv, sd, dens, amp in ((120, 110, 1, 0.30, 0.0045), (230, 210, 5, 0.30, 0.0022), (55, 50, 9, 0.18, 0.007)):
            dist, rad, pres = cells(ang, (y + 1) * 0.5, nu, nv, sd, dens)
            bump = np.clip(1 - (dist / rad) ** 2, 0, 1) * pres
            h = h + np.sqrt(bump) * amp * body_m
        wet = np.clip(h / 0.003, 0, 1)
        rough = rough * (1 - 0.75 * wet)
        base = base * (1 + 0.06 * wet[:, None])
        lid = smooth(0.90, 0.93, y)
        rough = rough * (1 - 0.3 * lid)
        h = h - 0.0016 * lid * (np.abs(np.hypot(lp[:, 0], lp[:, 2] - 0.06) - 0.25) < 0.006)
        return {"albedo": base, "rough": rough, "metal": np.full(len(y), 0.95, np.float32), "height": h}

    def alu_tab(uv, lp, n):
        g = fbm(lp[:, 0] * 30, lp[:, 2] * 4, 2, seed=11)
        return {"albedo": srgb(205, 205, 202) * (0.95 + 0.08 * g[:, None]), "rough": 0.22 + 0.06 * g,
                "metal": np.full(len(g), 0.95, np.float32)}

    def dark(uv, lp, n):
        return {"albedo": np.tile(srgb(40, 40, 42), (len(lp), 1)), "rough": np.full(len(lp), 0.5, np.float32),
                "metal": np.full(len(lp), 0.6, np.float32)}

    return Model([(body, "alu"), (tab, "tab"), (ring, "tab"), (rivet, "tab"), (opening, "dark")],
                 {"alu": alu, "tab": alu_tab, "dark": dark}, "soda_can")


# ============================================================================================ PIZZA SLICE
PZ_R = 1.45
PZ_A = 0.36


def _pz_xyz(r, a, y):
    return np.stack([r * np.sin(a), y, -r * np.cos(a)], -1)


def _cheese_h(r, a):
    """Top-surface height (y) of the slice."""
    rn = r / PZ_R
    an = a / PZ_A
    lump = fbm(r * 2.2, a * 6.5 + 3, 4, seed=21)
    melt = fbm(r * 6.0, a * 18.0, 3, seed=22)
    h = 0.105 + 0.034 * lump + 0.012 * melt
    h -= 0.022 * smooth(0.70, 1.0, np.abs(an))                  # droops toward the cut edges
    h -= 0.020 * (1 - smooth(0.0, 0.18, rn))                    # thinner at the tip
    h += 0.020 * smooth(0.80, 1.0, rn)                          # rises into the crust
    return h


@lru_cache(maxsize=2)
def pizza_slice():
    nr, na = 72, 44
    r = np.linspace(0.0, PZ_R + 0.03, nr).astype(np.float32)
    a = np.linspace(-PZ_A, PZ_A, na).astype(np.float32)
    Rg, Ag = np.meshgrid(r, a, indexing="ij")
    H = _cheese_h(Rg, Ag)
    top_v = _pz_xyz(Rg, Ag, H).reshape(-1, 3)
    top = oriented(top_v, grid_faces(nr, na), (0, 1, 0))
    bot_y = -0.005 + 0.008 * fbm(Rg * 3, Ag * 9, 2, seed=29)
    bot_v = _pz_xyz(Rg, Ag, bot_y).reshape(-1, 3)
    bot = oriented(bot_v, grid_faces(nr, na), (0, -1, 0))
    walls = []
    for side in (0, na - 1):                                   # cut edges
        rr = r
        aa = np.full_like(rr, a[side])
        ys = np.stack([bot_y[:, side], H[:, side]], 1)            # (nr, 2)
        v = np.stack([_pz_xyz(rr, aa, ys[:, 0]), _pz_xyz(rr, aa, ys[:, 1])], 1).reshape(-1, 3)
        out = np.array([math.cos(a[side]), 0, math.sin(a[side])], np.float32) * (1 if side else -1)
        walls.append(oriented(v, grid_faces(nr, 2), out))
    # crust tube along the outer arc
    nt, nc = 80, 28
    t = np.linspace(-PZ_A - 0.05, PZ_A + 0.05, nt).astype(np.float32)
    ph = np.linspace(0, 2 * math.pi, nc + 1).astype(np.float32)
    T, PH = np.meshgrid(t, ph, indexing="ij")
    end = np.clip((PZ_A + 0.05 - np.abs(T)) / 0.09, 0, 1)
    taper = np.sqrt(np.clip(end * (2 - end), 0, 1))
    lumpy = 1.0 + 0.16 * (fbm(T * 7 + 5, PH * 0.9, 3, seed=31) - 0.5) + 0.05 * (fbm(T * 20, PH * 3, 2, seed=32) - 0.5)
    rr_ = 0.135 * lumpy * taper
    ry_ = 0.120 * lumpy * taper
    rc = PZ_R + 0.035 + rr_ * np.cos(PH)
    yc = 0.125 + ry_ * np.sin(PH) * (1 + 0.15 * np.sin(PH))
    crust_v = _pz_xyz(rc, T, yc).reshape(-1, 3)
    cent = _pz_xyz(np.full_like(T, PZ_R + 0.035), T, np.full_like(T, 0.125)).reshape(-1, 3)
    crust = oriented(crust_v, grid_faces(nt, nc + 1), crust_v - cent)
    crust.n = _norm(vnormals(crust.v, crust.f) * (1 if np.mean(np.sum(vnormals(crust.v, crust.f) *
                                                                        (crust_v - cent), 1)) > 0 else -1))
    dough = merge(bot, *walls)

    # toppings: (r frac, a frac, kind, size)
    tops = [(0.78, -0.30, "pep", 0.165), (0.80, 0.50, "pep", 0.150), (0.42, 0.10, "pep", 0.140),
            (0.62, 0.05, "mush", 0.12), (0.55, -0.55, "mush", 0.10), (0.58, 0.55, "olive", 0.05),
            (0.86, 0.12, "pepper", 0.10)]
    peps, mush, oliv, grn = [], [], [], []
    for fr, fa, kind, s in tops:
        rr0, aa0 = fr * PZ_R, fa * PZ_A
        y0 = float(_cheese_h(np.float32(rr0), np.float32(aa0)))
        pos = _pz_xyz(np.float32(rr0), np.float32(aa0), np.float32(y0))
        tilt = rot(4 * (fa - 0.2), 0, 3 * fa)
        if kind == "pep":
            m = disc_y(s, 0.028, bevel=0.012, dome=0.004, cup=0.010, segs=48)
            peps.append(xform(m, tilt, pos + np.array([0, 0.004, 0])))
        elif kind == "mush":
            th = np.linspace(0, math.pi, 22)
            cap = [(math.cos(q) * s, math.sin(q) * s * 0.75 + s * 0.15) for q in th]
            stem = [(-s * 0.32, s * 0.15), (-s * 0.28, -s * 0.55), (s * 0.28, -s * 0.55), (s * 0.32, s * 0.15)]
            outer = np.array(cap[::-1][::-1] + stem, np.float32)
            outer = np.array([cap[i] for i in range(len(cap))] + stem[::-1][1:3], np.float32)
            outer = outer[::-1]
            c = outer.mean(0)
            inner = c + (outer - c) * 0.80
            m = bevel_prism(outer, inner, 0.022, 0.009)
            m = xform(m, rot(-90, 40 * fa + 20, 0), pos + np.array([0, 0.012, 0]))
            mush.append(m)
        elif kind == "olive":
            m = torus_y(s, s * 0.45, 40, 16)
            oliv.append(xform(m, rot(6, 0, 4), pos + np.array([0, 0.016, 0])))
        else:
            q = np.linspace(-0.6, 0.6, 14)
            arc = [(math.sin(x) * s, (1 - math.cos(x)) * s * 0.8) for x in q]
            outer = np.array([(x, y + 0.016) for x, y in arc] + [(x, y - 0.016) for x, y in arc[::-1]], np.float32)
            c = outer.mean(0)
            inner = c + (outer - c) * np.array([0.9, 0.5])
            m = bevel_prism(outer, inner, 0.02, 0.006)
            grn.append(xform(m, rot(-90, 70, 0), pos + np.array([0, 0.012, 0])))

    def cheese(uv, lp, n):
        rr = np.hypot(lp[:, 0], lp[:, 2])
        aa = np.arctan2(lp[:, 0], -lp[:, 2])
        rn, an = rr / PZ_R, aa / PZ_A
        lump = fbm(rr * 2.2, aa * 6.5 + 3, 4, seed=21)
        blis = fbm(rr * 9, aa * 26, 3, seed=24)
        pale = srgb(214, 168, 96)
        gold = srgb(204, 128, 46)
        brown = srgb(150, 74, 26)
        sauce = srgb(170, 50, 18)
        t = smooth(0.30, 0.75, lump * 0.7 + fbm(rr * 14, aa * 40, 3, seed=23) * 0.5)
        col = pale * (1 - t[:, None]) + gold * t[:, None]
        bk = smooth(0.62, 0.80, blis) * smooth(0.5, 0.9, lump)
        col = col * (1 - bk[:, None]) + brown * bk[:, None]
        sa = smooth(0.70, 0.98, np.abs(an)) * (1 - smooth(0.82, 0.95, rn)) + (1 - smooth(0.02, 0.12, rn))
        sa = np.clip(sa * (0.6 + 0.6 * fbm(rr * 5, aa * 14, 2, seed=25)), 0, 1)
        col = col * (1 - sa[:, None]) + sauce * sa[:, None]
        rough = 0.22 + 0.25 * (1 - lump) + 0.25 * bk
        height = 0.007 * fbm(rr * 30, aa * 90, 3, seed=26) + 0.004 * fbm(rr * 9, aa * 26, 3, seed=24)
        return {"albedo": col, "rough": rough, "height": height, "wrap": np.full(len(rr), 0.45, np.float32)}

    def doughm(uv, lp, n):
        y = lp[:, 1]
        rr = np.hypot(lp[:, 0], lp[:, 2])
        aa = np.arctan2(lp[:, 0], -lp[:, 2])
        g = fbm(rr * 8, aa * 25, 3, seed=27)
        under = smooth(0.02, -0.004, y)                         # underside: baked, charred spots
        base = srgb(206, 150, 92) * (0.9 + 0.2 * g[:, None])
        baked = srgb(160, 92, 42) * (0.8 + 0.3 * g[:, None])
        char = smooth(0.60, 0.80, fbm(rr * 4, aa * 12, 3, seed=28))[:, None] * under[:, None]
        dist_, rad_, pres_ = cells(rr * 0.7, aa * 2 + 3, 40, 30, 17, 0.6)
        spot = (np.clip(1 - dist_ / (rad_ * 1.6), 0, 1) * pres_)[:, None] * under[:, None]
        char = np.clip(char + spot * 0.8, 0, 1)
        col = base * (1 - under[:, None]) + baked * under[:, None]
        col = col * (1 - 0.65 * char)
        col = col * (1 - under[:, None] * 0.25 * (1 - smooth(0.3, 0.7, g))[:, None])
        sauce_band = smooth(0.06, 0.08, y)[:, None]           # sauce + cheese visible on the cut wall
        col = col * (1 - sauce_band) + srgb(214, 98, 40) * sauce_band
        return {"albedo": col, "rough": np.full(len(y), 0.8, np.float32),
                "height": 0.003 * fbm(rr * 40, aa * 120 + y * 30, 2, seed=30), "wrap": np.full(len(y), 0.3, np.float32)}

    def crustm(uv, lp, n):
        rr = np.hypot(lp[:, 0], lp[:, 2])
        aa = np.arctan2(lp[:, 0], -lp[:, 2])
        y = lp[:, 1]
        g = fbm(aa * 30, y * 20 + rr * 5, 4, seed=33)
        topness = np.clip(n[:, 1] * 0.5 + 0.5, 0, 1)
        light = srgb(196, 128, 70)
        mid = srgb(160, 88, 40)
        dark = srgb(104, 50, 18)
        t = smooth(0.35, 0.95, topness * 0.7 + g * 0.5)
        col = light * (1 - t[:, None]) + mid * t[:, None]
        dk = smooth(0.62, 0.85, fbm(aa * 12, y * 9, 3, seed=34)) * topness
        col = col * (1 - dk[:, None]) + dark * dk[:, None]
        dist, rad, pres = cells(aa * 0.9 + 5, y * 4 + rr, 260, 18, 7, 0.7)
        pore = np.clip(1 - dist / rad, 0, 1) * pres
        flour = smooth(0.80, 0.95, fbm(aa * 80, y * 60, 2, seed=35)) * (1 - topness)
        col = col * (1 - 0.35 * pore[:, None]) + srgb(240, 230, 210) * flour[:, None] * 0.5
        h = -0.004 * pore + 0.003 * g
        return {"albedo": col, "rough": 0.72 + 0.1 * g, "height": h, "wrap": np.full(len(y), 0.35, np.float32)}

    def pepm(uv, lp, n):
        # local coordinates of each disc are unknown here → use uv (lathe: u = angle, v = arc) + position noise
        rad = uv[:, 1]
        g = fbm(lp[:, 0] * 40, lp[:, 2] * 40, 3, seed=41)
        base = srgb(160, 40, 28) * (0.85 + 0.25 * g[:, None])
        edge = smooth(0.55, 0.75, np.abs(rad - 0.5) * 0) + smooth(0.30, 0.55, rad) * (1 - smooth(0.55, 0.9, rad))
        col = base * (1 - 0.45 * edge[:, None])
        dist, rr, pres = cells(lp[:, 0] * 3 + 9, lp[:, 2] * 3, 22, 22, 12, 0.8)
        fat = np.clip(1 - dist / rr, 0, 1) * pres
        col = col * (1 - fat[:, None] * 0.6) + srgb(232, 150, 120) * fat[:, None] * 0.6
        return {"albedo": col, "rough": 0.26 + 0.2 * g, "height": 0.0015 * g - 0.001 * fat}

    def mushm(uv, lp, n):
        g = fbm(lp[:, 0] * 30, lp[:, 2] * 30, 3, seed=51)
        side = 1 - np.abs(n[:, 1])
        col = srgb(196, 170, 130) * (0.9 + 0.15 * g[:, None])
        col = col * (1 - side[:, None] * 0.5) + srgb(120, 84, 50) * side[:, None] * 0.5
        return {"albedo": col, "rough": np.full(len(g), 0.55, np.float32), "wrap": np.full(len(g), 0.3, np.float32)}

    def olivem(uv, lp, n):
        return {"albedo": np.tile(srgb(22, 18, 16), (len(lp), 1)), "rough": np.full(len(lp), 0.18, np.float32)}

    def greenm(uv, lp, n):
        g = fbm(lp[:, 0] * 40, lp[:, 2] * 40, 2, seed=61)
        return {"albedo": srgb(70, 140, 50) * (0.85 + 0.2 * g[:, None]), "rough": np.full(len(g), 0.25, np.float32),
                "wrap": np.full(len(g), 0.3, np.float32)}

    parts = [(top, "cheese"), (dough, "dough"), (crust, "crust")]
    parts += [(m, "pep") for m in peps] + [(m, "mush") for m in mush] + [(m, "olive") for m in oliv]
    parts += [(m, "green") for m in grn]
    return Model(parts, {"cheese": cheese, "dough": doughm, "crust": crustm, "pep": pepm, "mush": mushm,
                         "olive": olivem, "green": greenm}, "pizza_slice")


# ============================================================================================ GROWTH CARD
@lru_cache(maxsize=2)
def growth_card():
    W, Hh, T = 1.0, 1.38, 0.07
    front = rbox(W, Hh, T, b=0.032, n_in=6, n_b=6)
    back = xform(rbox(W * 0.97, Hh * 0.97, T * 0.8, b=0.03, n_in=4, n_b=5), rot(0, 0, 0), (0.07, -0.02, -0.085))
    zf = T / 2
    deco = []

    def bar(x, y, w, h, d=0.012):
        return xform(rbox(w, h, d * 2, b=min(h, d) * 0.45, n_in=2, n_b=4), None, (x, y, zf))

    deco += [bar(-0.22, 0.56, 0.36, 0.035), bar(-0.27, 0.49, 0.26, 0.03)]
    for k in range(2):
        d = xform(torus_y(0.018, 0.006, 20, 8), rot(90, 0, 0), (0.30 + 0.05 * k, 0.56, zf + 0.004))
        deco.append(d)
    deco += [xform(rbox(0.86, 0.008, 0.012, b=0.003, n_in=2, n_b=3), None, (0, -0.22, zf))]
    deco += [bar(-0.24, -0.32, 0.36, 0.032), bar(-0.27, -0.40, 0.30, 0.032)]
    deco += [bar(0.26, -0.32, 0.18, 0.032), bar(0.28, -0.40, 0.14, 0.032)]
    for k in range(9):
        deco.append(xform(disc_y(0.013, 0.012, bevel=0.005, dome=0.003, segs=16), rot(90, 0, 0),
                          (-0.40 + k * 0.1, -0.60, zf)))
    pts = [(-0.36, -0.02), (-0.08, 0.20), (0.06, 0.05), (0.27, 0.27)]
    outer = stroke_arrow(pts, 0.062, 0.14, 0.19)
    inner = stroke_arrow(pts, 0.062 - 0.02, 0.14 - 0.035, 0.19 - 0.045)
    # pull the inner tip back so the chamfer is even
    arrow = bevel_prism(outer, inner, 0.10, 0.02)
    arrow = xform(arrow, None, (0.0, 0.0, zf + 0.06))

    def plastic(uv, lp, n):
        g = fbm(lp[:, 0] * 8 + lp[:, 2] * 3, lp[:, 1] * 8, 3, seed=71)
        col = srgb(232, 233, 228) * (0.96 + 0.06 * g[:, None])
        return {"albedo": col, "rough": 0.42 + 0.08 * g, "height": 0.0008 * fbm(lp[:, 0] * 80, lp[:, 1] * 80, 2, seed=72),
                "wrap": np.full(len(g), 0.15, np.float32)}

    def plastic2(uv, lp, n):
        g = fbm(lp[:, 0] * 8, lp[:, 1] * 8, 2, seed=73)
        return {"albedo": srgb(214, 216, 210) * (0.95 + 0.06 * g[:, None]), "rough": 0.48 + 0.05 * g,
                "wrap": np.full(len(g), 0.15, np.float32)}

    def emb(uv, lp, n):
        return {"albedo": np.tile(srgb(206, 208, 203), (len(lp), 1)), "rough": np.full(len(lp), 0.5, np.float32)}

    def orange(uv, lp, n):
        g = fbm(lp[:, 0] * 10, lp[:, 1] * 10, 2, seed=74)
        return {"albedo": srgb(226, 104, 66) * (0.95 + 0.08 * g[:, None]), "rough": 0.34 + 0.06 * g,
                "wrap": np.full(len(g), 0.25, np.float32)}

    parts = [(front, "card"), (back, "card2"), (arrow, "arrow")] + [(d, "emb") for d in deco]
    return Model(parts, {"card": plastic, "card2": plastic2, "emb": emb, "arrow": orange}, "growth_card")


MODELS = {"soda_can": soda_can, "pizza_slice": pizza_slice, "growth_card": growth_card}


def get_model(name):
    return MODELS[name]()
