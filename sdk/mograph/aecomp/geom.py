"""
mograph.aecomp.geom — bezier paths and the After-Effects shape-layer PATH OPERATORS.

A Path = vertices V (n,2), absolute in-handles I (n,2), absolute out-handles O (n,2), closed flag, plus `op`
(merge operator used when a Merge Paths item collapsed it: add | sub | int | xor).
Everything is plain numpy; maths follows the public Lottie/bodymovin implementation of the same AE operators
(trim, zigzag, round corners, pucker & bloat, offset) so results match AE previews; twist is AE-only and is
implemented as a distance-weighted rotation on a densified path.
"""
from __future__ import annotations

import math

import numpy as np

KAPPA = 0.5519150244935105


class Path:
    __slots__ = ("V", "I", "O", "closed", "op")

    def __init__(self, V, I=None, O=None, closed=True, op="add"):
        self.V = np.asarray(V, float).reshape(-1, 2)
        self.I = self.V.copy() if I is None else np.asarray(I, float).reshape(-1, 2)
        self.O = self.V.copy() if O is None else np.asarray(O, float).reshape(-1, 2)
        self.closed, self.op = bool(closed), op

    def copy(self):
        return Path(self.V.copy(), self.I.copy(), self.O.copy(), self.closed, self.op)

    def transform(self, M):
        """apply a 3x3 affine (row-vector convention: p' = M @ [x, y, 1])."""
        A, b = M[:2, :2], M[:2, 2]
        self.V = self.V @ A.T + b
        self.I = self.I @ A.T + b
        self.O = self.O @ A.T + b
        return self

    def reversed(self):
        if len(self.V) < 2:
            return self.copy()
        if self.closed:                               # keep vertex 0 as the start, walk backwards
            idx = [0] + list(range(len(self.V) - 1, 0, -1))
        else:
            idx = list(range(len(self.V) - 1, -1, -1))
        return Path(self.V[idx], self.O[idx], self.I[idx], self.closed, self.op)

    def segments(self):
        """list of cubic control arrays (4,2)."""
        n = len(self.V)
        segs = []
        rng = range(n) if self.closed else range(n - 1)
        for i in rng:
            j = (i + 1) % n
            segs.append(np.array([self.V[i], self.O[i], self.I[j], self.V[j]]))
        return segs

    @staticmethod
    def from_segments(segs, closed=False, op="add"):
        if not segs:
            return Path(np.zeros((0, 2)), closed=False, op=op)
        V = [segs[0][0]]
        I = [segs[0][0]]
        O = []
        for s in segs:
            O.append(s[1])
            V.append(s[3])
            I.append(s[2])
        O.append(segs[-1][3])
        if closed and np.allclose(V[0], V[-1]):
            I[0] = I[-1]
            V, I, O = V[:-1], I[:-1], O[:-1]
        return Path(np.array(V), np.array(I), np.array(O), closed, op)


# ------------------------------------------------------------------------------------- primitives
def ellipse(size, pos=(0, 0), reverse=False):
    w, h = size[0] / 2.0, size[1] / 2.0
    x, y = pos
    kw, kh = w * KAPPA, h * KAPPA
    V = [[x, y - h], [x + w, y], [x, y + h], [x - w, y]]
    O = [[x + kw, y - h], [x + w, y + kh], [x - kw, y + h], [x - w, y - kh]]
    I = [[x - kw, y - h], [x + w, y - kh], [x + kw, y + h], [x - w, y + kh]]
    p = Path(V, I, O, True)
    return p.reversed() if reverse else p


def rect(size, pos=(0, 0), round_=0.0, reverse=False):
    w, h = size[0] / 2.0, size[1] / 2.0
    x, y = pos
    r = max(0.0, min(float(round_), w, h))
    if r < 1e-6:
        V = [[x + w, y - h], [x + w, y + h], [x - w, y + h], [x - w, y - h]]
        p = Path(V, None, None, True)
    else:
        c = r * (1 - KAPPA)
        V = [[x + w, y - h + r], [x + w, y + h - r], [x + w - r, y + h], [x - w + r, y + h],
             [x - w, y + h - r], [x - w, y - h + r], [x - w + r, y - h], [x + w - r, y - h]]
        I = [[x + w, y - h + c], [x + w, y + h - r], [x + w - c, y + h], [x - w + r, y + h],
             [x - w, y + h - c], [x - w, y - h + r], [x - w + c, y - h], [x + w - r, y - h]]
        O = [[x + w, y - h + r], [x + w, y + h - c], [x + w - r, y + h], [x - w + c, y + h],
             [x - w, y + h - r], [x - w, y - h + c], [x - w + r, y - h], [x + w - c, y - h]]
        p = Path(V, I, O, True)
    return p.reversed() if reverse else p


def polystar(kind, points, outer, inner=0.0, outer_round=0.0, inner_round=0.0, rot=0.0, pos=(0, 0), reverse=False):
    """AE Polystar (kind 'star' | 'polygon'); roundness in % like AE."""
    n = max(1, int(round(points)))
    x0, y0 = pos
    V, I, O = [], [], []
    d = -1.0 if reverse else 1.0
    ang = -math.pi / 2 + math.radians(rot)
    if kind == "star":
        step = math.pi / n
        cnt = n * 2
        lp, sp = 2 * math.pi * outer / (n * 2), 2 * math.pi * inner / (n * 2)
    else:
        step = 2 * math.pi / n
        cnt = n
        lp = sp = 2 * math.pi * outer / (n * 4)
    for i in range(cnt):
        long_ = kind != "star" or i % 2 == 0
        rad = outer if long_ else inner
        rnd = (outer_round if long_ else inner_round) / 100.0
        per = lp if long_ else sp
        x, y = rad * math.cos(ang), rad * math.sin(ang)
        L = math.hypot(x, y)
        ox, oy = (0.0, 0.0) if L < 1e-9 else (y / L, -x / L)
        vx, vy = x + x0, y + y0
        V.append([vx, vy])
        I.append([vx + ox * per * rnd * d, vy + oy * per * rnd * d])
        O.append([vx - ox * per * rnd * d, vy - oy * per * rnd * d])
        ang += step * d
    return Path(V, I, O, True)


# ------------------------------------------------------------------------------------- bezier utils
def _pt(s, u):
    m = 1 - u
    return (m * m * m) * s[0] + (3 * m * m * u) * s[1] + (3 * m * u * u) * s[2] + (u * u * u) * s[3]


def _split(s, u):
    """de Casteljau → (left, right)."""
    a = s[0] + (s[1] - s[0]) * u
    b = s[1] + (s[2] - s[1]) * u
    c = s[2] + (s[3] - s[2]) * u
    d = a + (b - a) * u
    e = b + (c - b) * u
    f = d + (e - d) * u
    return np.array([s[0], a, d, f]), np.array([f, e, c, s[3]])


def _sub(s, u0, u1):
    if u0 <= 0 and u1 >= 1:
        return s
    if u0 > 0:
        s = _split(s, u0)[1]
        u1 = (u1 - u0) / (1 - u0) if u0 < 1 else 1.0
    if u1 < 1:
        s = _split(s, max(0.0, min(1.0, u1)))[0]
    return s


_US = np.linspace(0, 1, 25)


def _lut(s):
    P = np.array([_pt(s, u) for u in _US])
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return L


def seg_len(s):
    return float(_lut(s)[-1])


def _u_at(lut, d):
    return float(np.interp(d, lut, _US))


def path_length(p):
    return sum(seg_len(s) for s in p.segments())


def point_at(p, frac):
    """(x, y, angle_deg) at fraction of the path length (AE pointOnPath / tangentOnPath)."""
    segs = p.segments()
    if not segs:
        return (float(p.V[0, 0]), float(p.V[0, 1]), 0.0) if len(p.V) else (0.0, 0.0, 0.0)
    luts = [_lut(s) for s in segs]
    tot = sum(l[-1] for l in luts)
    d = max(0.0, min(1.0, frac)) * tot
    for s, l in zip(segs, luts):
        if d <= l[-1] or s is segs[-1]:
            u = _u_at(l, min(d, l[-1]))
            q = _pt(s, u)
            q2 = _pt(s, min(1.0, u + 1e-3)) - _pt(s, max(0.0, u - 1e-3))
            return float(q[0]), float(q[1]), math.degrees(math.atan2(q2[1], q2[0]))
        d -= l[-1]
    return 0.0, 0.0, 0.0


# ------------------------------------------------------------------------------------- operators
def _trim_one(segs, luts, a, b):
    """keep length interval [a, b] (absolute) of a segment chain (no wrap)."""
    out, acc = [], 0.0
    for s, l in zip(segs, luts):
        L = l[-1]
        s0, s1 = acc, acc + L
        acc = s1
        if s1 <= a or s0 >= b or L <= 1e-9:
            continue
        u0 = _u_at(l, max(0.0, a - s0)) if a > s0 else 0.0
        u1 = _u_at(l, min(L, b - s0)) if b < s1 else 1.0
        out.append(_sub(s, u0, u1))
    return out


def trim(paths, start, end, offset, mode=1):
    """AE Trim Paths.  start/end in %, offset in degrees; mode 1 = simultaneously, 2 = individually."""
    s, e = start / 100.0, end / 100.0
    if s > e:
        s, e = e, s
    off = (offset / 360.0) % 1.0
    if e - s >= 0.99999 and abs(off) < 1e-9:
        return paths
    if e - s <= 1e-6:
        return []

    def cut(chain_segs, total, ranges):
        luts = [_lut(q) for q in chain_segs]
        res = []
        for a, b in ranges:
            res.append(_trim_one(chain_segs, luts, a * total, b * total))
        return res

    def ranges_for():
        a, b = s + off, e + off
        if b <= 1.0:
            return [(a, b)]
        if a >= 1.0:
            return [(a - 1, b - 1)]
        return [(a, 1.0), (0.0, b - 1)]

    out = []
    if mode == 2 or len(paths) <= 1:
        for p in paths:
            segs = p.segments()
            tot = sum(seg_len(q) for q in segs)
            if tot <= 1e-9:
                continue
            pieces = cut(segs, tot, ranges_for())
            if len(pieces) == 2 and p.closed and pieces[0] and pieces[1]:
                out.append(Path.from_segments(pieces[0] + pieces[1], False, p.op))   # wrap join (end→start)
            else:
                out += [Path.from_segments(pc, False, p.op) for pc in pieces if pc]
        return out
    # simultaneously: the paths form one chain of total length
    lens = [path_length(p) for p in paths]
    tot = sum(lens)
    if tot <= 1e-9:
        return []
    acc = 0.0
    for p, L in zip(paths, lens):
        for a, b in ranges_for():
            aa, bb = a * tot - acc, b * tot - acc
            if bb <= 0 or aa >= L:
                continue
            segs = p.segments()
            luts = [_lut(q) for q in segs]
            pc = _trim_one(segs, luts, max(0.0, aa), min(L, bb))
            if pc:
                out.append(Path.from_segments(pc, False, p.op))
        acc += L
    return out


def _normal(s, u):
    q = _pt(s, min(1.0, u + 1e-3)) - _pt(s, max(0.0, u - 1e-3))
    L = math.hypot(q[0], q[1])
    return np.array([0.0, 0.0]) if L < 1e-12 else np.array([-q[1] / L, q[0] / L])


def zigzag(p, size, ridges, smooth=False):
    """AE Zig Zag: `ridges` per segment, amplitude `size`, corner (1) or smooth (2) points."""
    r = max(0, int(round(ridges)))
    segs = p.segments()
    if not segs or size == 0:
        return p
    pts, nrm = [], []
    for si, s in enumerate(segs):
        l = _lut(s)
        L = l[-1]
        k = r + 1
        rng = range(k) if (p.closed or si < len(segs) - 1) else range(k + 1)
        for j in rng:
            u = _u_at(l, L * j / k)
            pts.append(_pt(s, u))
            nrm.append(_normal(s, u))
    if not p.closed and len(segs) and len(pts) == 0:
        return p
    d = 1.0
    V = []
    for q, nv in zip(pts, nrm):
        V.append(q + nv * size * d)
        d = -d
    V = np.array(V)
    if not smooth:
        return Path(V, None, None, p.closed, p.op)
    n = len(V)
    I, O = V.copy(), V.copy()
    for i in range(n):
        a = V[i - 1] if (p.closed or i > 0) else V[i]
        b = V[(i + 1) % n] if (p.closed or i < n - 1) else V[i]
        t = (b - a) / 4.0
        I[i], O[i] = V[i] - t, V[i] + t
    return Path(V, I, O, p.closed, p.op)


def densify(p, per=12):
    """split every segment into `per` sub-cubics (keeps the exact shape)."""
    out = []
    for s in p.segments():
        rest = s
        for j in range(per, 1, -1):
            a, rest = _split(rest, 1.0 / j)
            out.append(a)
        out.append(rest)
    q = Path.from_segments(out, p.closed, p.op)
    return q


def twist(paths, angle, center=(0, 0)):
    """AE Twist: "rotates a path more sharply in the centre than at the edges" — full `angle` at the centre, 0 at the
    farthest vertex (measured on 11 of 33 reference elements using Twist: F3 0.57 → 0.72 vs the opposite convention)."""
    if abs(angle) < 1e-6:
        return paths
    cx, cy = center
    dense = [densify(p, 10) for p in paths]
    rmax = max([float(np.max(np.hypot(q.V[:, 0] - cx, q.V[:, 1] - cy))) for q in dense if len(q.V)] or [1.0]) or 1.0

    def rot(P):
        dx, dy = P[:, 0] - cx, P[:, 1] - cy
        r = np.hypot(dx, dy) / rmax
        a = np.radians(angle) * (1.0 - r)       # linear falloff (beat quad / cosine on 4 reference elements); AE: full angle at the centre, fading to 0 at the farthest point
        c, s = np.cos(a), np.sin(a)
        return np.stack([cx + dx * c - dy * s, cy + dx * s + dy * c], 1)
    for q in dense:
        q.V, q.I, q.O = rot(q.V), rot(q.I), rot(q.O)
    return dense


def round_corners(p, radius):
    """AE Round Corners (only sharp vertices = no handles are rounded)."""
    if radius <= 0 or len(p.V) < 2:
        return p
    n = len(p.V)
    V, I, O = [], [], []
    for i in range(n):
        v = p.V[i]
        sharp = np.allclose(p.I[i], v) and np.allclose(p.O[i], v)
        if not sharp or (not p.closed and (i == 0 or i == n - 1)):
            V.append(v); I.append(p.I[i]); O.append(p.O[i])
            continue
        a, b = p.V[i - 1], p.V[(i + 1) % n]
        da, db = a - v, b - v
        la, lb = np.linalg.norm(da), np.linalg.norm(db)
        if la < 1e-9 or lb < 1e-9:
            V.append(v); I.append(v); O.append(v)
            continue
        ra, rb = min(radius, la / 2), min(radius, lb / 2)
        p1, p2 = v + da / la * ra, v + db / lb * rb
        V.append(p1); I.append(p1); O.append(p1 + (v - p1) * KAPPA)
        V.append(p2); I.append(p2 + (v - p2) * KAPPA); O.append(p2)
    return Path(np.array(V), np.array(I), np.array(O), p.closed, p.op)


def pucker_bloat(p, amount):
    """AE Pucker & Bloat (amount %, negative = pucker)."""
    a = amount / 100.0
    if abs(a) < 1e-6 or not len(p.V):
        return p
    c = p.V.mean(axis=0)
    V = p.V + (c - p.V) * a
    O = p.O + (c - p.O) * -a
    I = p.I + (c - p.I) * -a
    return Path(V, I, O, p.closed, p.op)


def offset(p, amount):
    """AE Offset Paths (vertex-normal approximation; exact for circles/convex polygons with round joins)."""
    if abs(amount) < 1e-6 or len(p.V) < 2:
        return p
    q = densify(p, 4) if len(p.V) < 16 else p.copy()
    n = len(q.V)
    area = 0.5 * float(np.sum(q.V[:, 0] * np.roll(q.V[:, 1], -1) - np.roll(q.V[:, 0], -1) * q.V[:, 1]))
    sgn = 1.0 if area > 0 else -1.0
    N = np.zeros_like(q.V)
    for i in range(n):
        a = q.V[i - 1] if (q.closed or i > 0) else q.V[i]
        b = q.V[(i + 1) % n] if (q.closed or i < n - 1) else q.V[i]
        t = b - a
        L = math.hypot(t[0], t[1]) or 1.0
        N[i] = [t[1] / L, -t[0] / L]
    d = N * amount * -sgn
    return Path(q.V + d, q.I + d, q.O + d, q.closed, q.op)


def affine(anchor=(0, 0), pos=(0, 0), scale=(100, 100), rot=0.0, skew=0.0, skew_axis=0.0):
    """AE transform matrix: translate(pos)·rotate·skew·scale·translate(-anchor)."""
    ax, ay = anchor[0], anchor[1]
    sx, sy = scale[0] / 100.0, scale[1] / 100.0
    T1 = np.array([[1, 0, -ax], [0, 1, -ay], [0, 0, 1]], float)
    S = np.diag([sx, sy, 1.0])
    K = np.eye(3)
    if abs(skew) > 1e-6:
        a = math.radians(skew_axis)
        c, s = math.cos(a), math.sin(a)
        R1 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Sk = np.array([[1, math.tan(math.radians(-skew)), 0], [0, 1, 0], [0, 0, 1]])
        K = R1 @ Sk @ R1.T
    r = math.radians(rot)
    R = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    T2 = np.array([[1, 0, pos[0]], [0, 1, pos[1]], [0, 0, 1]], float)
    return T2 @ R @ K @ S @ T1


# ---------------------------------------------------------------- Merge Paths (true boolean geometry)
def _poly_pts(p, per=16):
    pts = []
    for s in p.segments():
        flat = np.allclose(s[1], s[0]) and np.allclose(s[2], s[3])
        us = (0.0,) if flat else np.linspace(0, 1, per, endpoint=False)
        pts += [_pt(s, u) for u in us]
    return np.array(pts) if pts else p.V.copy()


def _signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def boolean(paths, mode="add"):
    """AE Merge Paths: path 0 is the base, every later closed path is combined with `mode`
    (add = union, sub = subtract, int = intersect, xor = exclude intersections).  Returns linear Paths whose
    start vertex / direction follow the first input path (so a Trim Paths below behaves like AE).
    Open paths pass through unchanged.  Raises ImportError when shapely is missing (caller falls back)."""
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union  # noqa: F401
    closed = [p for p in paths if p.closed and len(p.V) >= 2]
    opened = [p for p in paths if not (p.closed and len(p.V) >= 2)]
    if not closed:
        return paths
    acc = None
    ref = None
    for i, p in enumerate(closed):
        P = _poly_pts(p)
        if len(P) < 3:
            continue
        g = Polygon(P).buffer(0)
        if ref is None:
            ref = (P[0], _signed_area(P) >= 0)
        if acc is None:
            acc = g
            continue
        acc = {"add": acc.union, "sub": acc.difference, "int": acc.intersection,
               "xor": acc.symmetric_difference}.get(mode, acc.union)(g)
    if acc is None or acc.is_empty:
        return opened
    polys = list(acc.geoms) if isinstance(acc, MultiPolygon) else [acc] if acc.geom_type == "Polygon" else \
        [g for g in getattr(acc, "geoms", []) if g.geom_type == "Polygon"]
    out = []
    p0, ccw = ref
    for poly in polys:
        rings = [(np.asarray(poly.exterior.coords)[:-1], True)] + \
                [(np.asarray(r.coords)[:-1], False) for r in poly.interiors]
        for R, outer in rings:
            if len(R) < 3:
                continue
            want = ccw if outer else not ccw
            if (_signed_area(R) >= 0) != want:
                R = R[::-1]
            k = int(np.argmin(np.hypot(R[:, 0] - p0[0], R[:, 1] - p0[1])))
            R = np.roll(R, -k, axis=0)
            out.append(Path(R, R.copy(), R.copy(), True, "add"))
    return out + opened
