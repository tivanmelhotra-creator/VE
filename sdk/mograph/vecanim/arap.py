"""As-rigid-as-possible puppet (Position / Bend / Starch / Overlap pins) for VECTOR scenes.

Python port of EffectCraft's `effects/src/puppet/{arap,mesh,mod}.rs`
(https://github.com/storytold/effectcraft, Apache-2.0), after Igarashi, Moscovich & Hughes,
"As-Rigid-As-Possible Shape Manipulation" (SIGGRAPH 2005):

  1. similarity step: every triangle edge (i,j) writes the third vertex in the edge's frame;
     least squares with pinned vertices fixed -> locally a similarity;
  2. scale-adjustment step: each triangle gets the best rigid copy of its rest shape
     (Procrustes about the centroid); positions match those edge vectors (x and y separately);
  3. optional refinement iterations of step 2.
  Starch = per-triangle stiffness weights; Bend/Advanced pins = soft edge-vector targets
  (rotation + scale around the pin, weight 10, exactly as `mod.rs solve`).

Differences from the Rust code (speed in Python, same maths):
  * the mesh is built once per character (outline + jittered hex lattice -> Delaunay,
    triangles outside the outline dropped);
  * for a fixed pin *set* the free-free blocks of both SPD systems are factorised once
    (scipy splu) and reused every frame -> a frame is a few back-substitutions (~2-4 ms);
  * vector control points are embedded in the REST mesh once (triangle + barycentrics) and
    carried by the deformed triangles -> lines stay vector-sharp.
"""
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import cv2


# ------------------------------------------------------------------------- mesh
def outline_from_mask(mask, simplify=1.5, expansion=3):
    """Outer contours of a mask (mesh.rs: threshold -> expand -> trace -> DP simplify)."""
    m = (mask > 0).astype(np.uint8)
    if expansion > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expansion + 1,) * 2))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 50:
            continue
        out.append(cv2.approxPolyDP(c, simplify, True)[:, 0].astype(float))
    return out


def densify(poly, max_len):
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / max_len)))
        for t in np.arange(k) / k:
            out.append(a + (b - a) * t)
    return np.array(out)


def _inside(polys, pts):
    ok = np.zeros(len(pts), bool)
    for p in polys:
        path = p.astype(np.float32).reshape(-1, 1, 2)
        ok |= np.array([cv2.pointPolygonTest(path, (float(x), float(y)), False) >= 0 for x, y in pts])
    return ok


def _bary(a, b, c, p):
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.c_[1 - v - w, v, w]


class Mesh:
    def __init__(self, verts, tris):
        self.verts = np.asarray(verts, float)
        self.tris = np.asarray(tris, np.int64)

    @staticmethod
    def from_mask(mask, edge=18.0, expansion=4, seed=7):
        """Triangulate a character mask: outline (densified to `edge`) + jittered hex lattice."""
        polys = outline_from_mask(mask, simplify=1.2, expansion=expansion)
        if not polys:
            raise ValueError('empty mask')
        B = np.vstack([densify(p, edge) for p in polys])
        x0, y0 = B.min(0); x1, y1 = B.max(0)
        rng = np.random.default_rng(seed)
        hx, hy = edge, edge * np.sqrt(3) / 2
        pts = []
        for r, y in enumerate(np.arange(y0 + hy / 2, y1, hy)):
            pts += [(x, y) for x in np.arange(x0 + (hx / 2 if r % 2 else 0), x1, hx)]
        P = np.array(pts) + rng.uniform(-0.15, 0.15, (len(pts), 2)) * edge
        P = P[_inside(polys, P)]
        if len(P):
            d, _ = cKDTree(B).query(P)
            P = P[d > edge * 0.45]          # no slivers along the outline
        V = np.vstack([B, P])
        T = Delaunay(V).simplices
        ok = _inside(polys, V[T].mean(1))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ok &= _inside(polys, (V[T[:, a]] + V[T[:, b]]) / 2)
        T = T[ok]
        e1, e2 = V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
        cw = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 0
        T[cw] = T[cw][:, [0, 2, 1]]
        used = np.unique(T)
        remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
        return Mesh(V[used], remap[T])

    def embed(self, pts, k=12):
        """Barycentric embedding of points in the rest mesh: (tri index, (N,3) weights).
        Points outside every triangle take the best nearby triangle (extrapolated weights)."""
        V, T = self.verts, self.tris
        pts = np.asarray(pts, float)
        k = min(k, len(T))
        _, cand = cKDTree(V[T].mean(1)).query(pts, k=k)
        cand = cand.reshape(len(pts), k)
        best, bw, bs = None, None, None
        for j in range(k):
            t = T[cand[:, j]]
            w = _bary(V[t[:, 0]], V[t[:, 1]], V[t[:, 2]], pts)
            sc = w.min(1)
            if j == 0:
                best, bw, bs = cand[:, 0].copy(), w, sc
            else:
                m = sc > bs
                best[m] = cand[m, j]; bw[m] = w[m]; bs = np.where(m, sc, bs)
        return best, bw

    def carry(self, deformed, emb):
        t, w = emb
        return (deformed[self.tris[t]] * w[..., None]).sum(1)


# ------------------------------------------------------------------------- pins
class Pin:
    """kind: position | advanced | bend | starch | overlap.  rest (x,y); rotation (deg) / scale for
    bend & advanced; amount / extent for starch; in_front / extent for overlap."""
    def __init__(self, name, rest, kind='position', rotation=0.0, scale=1.0,
                 amount=1.0, extent=30.0, in_front=0.0):
        self.name, self.kind = name, kind
        self.rest = np.asarray(rest, float)
        self.rotation, self.scale, self.amount, self.extent, self.in_front = rotation, scale, amount, extent, in_front

    def moves(self):
        return self.kind in ('position', 'advanced')


REG = 1e-6
WB = 10.0          # bend-pin edge weight (mod.rs)


def _lap(i, j, w, n):
    return sp.coo_matrix((np.r_[w, w, -w, -w], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()


class Puppet:
    """Mesh + fixed pin set.  Per frame: `solve({name: (x,y)}, bend={name: deg})`."""
    def __init__(self, mesh, pins, refine=2):
        self.mesh, self.refine = mesh, refine
        self.pins = {p.name: p for p in pins}
        V, T = mesh.verts, mesh.tris
        n = len(V)
        tree = cKDTree(V)
        self.pin_vert, used = {}, set()
        for p in pins:
            if p.moves():
                _, vs = tree.query(p.rest, k=min(8, n))
                v = int(next(x for x in np.atleast_1d(vs) if int(x) not in used))
                used.add(v); self.pin_vert[p.name] = v
        self.adj = [set() for _ in range(n)]
        for t in T:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                self.adj[a].add(b); self.adj[b].add(a)
        self.bend_vert = {p.name: int(tree.query(p.rest)[1]) for p in pins if p.kind in ('bend', 'advanced')}
        c = V[T].mean(1)
        w = np.ones(len(T))
        dep = np.zeros(len(T))
        for p in pins:
            d = np.linalg.norm(c - p.rest, axis=1)
            if p.kind == 'starch':
                w += np.clip(1 - d / (max(p.extent, 1e-9) * 2), 0, 1) * max(p.amount, 0) * 2
            if p.kind == 'overlap':
                dep = np.where(d <= p.extent, np.maximum(dep, p.in_front), dep)
        self.w, self.depth = w, dep
        self._build()

    def _build(self):
        V, T, w = self.mesh.verts, self.mesh.tris, self.w
        n = len(V)
        e2 = np.mean(((V[T[:, 0]] - V[T[:, 1]]) ** 2).sum(1))
        self.reg = REG * min(1.0, 1.0 / max(e2, 1e-12))
        rows, cols, vals = [], [], []
        for (i_, j_, l_) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            i, j, l = T[:, i_], T[:, j_], T[:, l_]
            e = V[j] - V[i]; d = V[l] - V[i]
            ee = (e ** 2).sum(1)
            x = (d * e).sum(1) / ee
            y = (d[:, 0] * -e[:, 1] + d[:, 1] * e[:, 0]) / ee
            one, zero = np.ones_like(x), np.zeros_like(x)
            B = {'l': np.stack([np.stack([one, zero], -1), np.stack([zero, one], -1)], -2),
                 'i': np.stack([np.stack([x - 1, -y], -1), np.stack([y, x - 1], -1)], -2),
                 'j': np.stack([np.stack([-x, y], -1), np.stack([-y, -x], -1)], -2)}
            idx = {'l': l, 'i': i, 'j': j}
            for a in 'lij':
                for b in 'lij':
                    M = np.einsum('kra,krb->kab', B[a], B[b]) * w[:, None, None]      # ba^T bb
                    for r in range(2):
                        for cc in range(2):
                            rows.append(2 * idx[a] + r); cols.append(2 * idx[b] + cc); vals.append(M[:, r, cc])
        A1 = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(2 * n, 2 * n)).tocsr()
        L = sum(_lap(T[:, a], T[:, b], w, n) for a, b in ((0, 1), (1, 2), (2, 0)))
        bi, bj = [], []
        for v in self.bend_vert.values():
            for j in self.adj[v]:
                bi.append(v); bj.append(j)
        bi, bj = np.array(bi, np.int64), np.array(bj, np.int64)
        self.bend_edges = (bi, bj)
        if len(bi):
            wb = np.full(len(bi), WB)
            for c in range(2):
                A1 = A1 + _lap(2 * bi + c, 2 * bj + c, wb, 2 * n)
            L = L + _lap(bi, bj, wb, n)
        A1 = A1 + sp.identity(2 * n) * self.reg
        L = L + sp.identity(n) * self.reg
        fixed = np.zeros(n, bool)
        fixed[list(self.pin_vert.values())] = True
        f1 = np.repeat(~fixed, 2)
        self.free1, self.fix1 = np.nonzero(f1)[0], np.nonzero(~f1)[0]
        self.free, self.fix = np.nonzero(~fixed)[0], np.nonzero(fixed)[0]
        self.lu1 = splu(A1[self.free1][:, self.free1].tocsc())
        self.A1fx = A1[self.free1][:, self.fix1]
        self.lu2 = splu(L[self.free][:, self.free].tocsc())
        self.Lfx = L[self.free][:, self.fix]
        R = V[T]
        self.rest_c = R - R.mean(1, keepdims=True)

    def solve(self, targets, bend=None, scale=None):
        V, T, w = self.mesh.verts, self.mesh.tris, self.w
        n = len(V)
        X = V.copy()
        disp = []
        for nm, v in self.pin_vert.items():
            p = self.pins[nm]
            X[v] = V[v] + (np.asarray(targets.get(nm, p.rest), float) - p.rest)
            disp.append(X[v] - V[v])
        x0 = V + (np.mean(disp, 0) if disp else 0)
        if len(self.pin_vert) == 1 and not self.bend_vert:
            return x0                         # exact solution (arap.rs #313)
        bvec = None
        if self.bend_vert:
            bi, bj = self.bend_edges
            vec = np.zeros((len(bi), 2)); k = 0
            for nm, v in self.bend_vert.items():
                p = self.pins[nm]
                a = np.radians((bend or {}).get(nm, p.rotation)); s_ = (scale or {}).get(nm, p.scale)
                c, s = np.cos(a), np.sin(a)
                for j in self.adj[v]:
                    e = V[j] - V[v]
                    vec[k] = s_ * np.array([e[0] * c - e[1] * s, e[0] * s + e[1] * c]); k += 1
            bvec = (bi, bj, vec)
        # ---- step 1: similarity
        b1 = self.reg * x0.ravel().copy()
        if bvec is not None:
            bi, bj, vec = bvec
            for c in range(2):
                np.add.at(b1, 2 * bj + c, WB * vec[:, c]); np.add.at(b1, 2 * bi + c, -WB * vec[:, c])
        x = X.ravel().copy()
        x[self.free1] = self.lu1.solve(b1[self.free1] - self.A1fx @ x[self.fix1])
        cur = x.reshape(n, 2)
        # ---- step 2: rigid fit + edge matching (+ refinement)
        for _ in range(self.refine + 1):
            C = cur[T]
            cc = C - C.mean(1, keepdims=True)
            sd = (self.rest_c * cc).sum((1, 2))
            sc = (self.rest_c[..., 0] * cc[..., 1] - self.rest_c[..., 1] * cc[..., 0]).sum(1)
            ang = np.arctan2(sc, sd)
            ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
            Fx = self.rest_c[..., 0] * ca - self.rest_c[..., 1] * sa
            Fy = self.rest_c[..., 0] * sa + self.rest_c[..., 1] * ca
            bx = self.reg * x0[:, 0].copy(); by = self.reg * x0[:, 1].copy()
            for a, b in ((0, 1), (1, 2), (2, 0)):
                i, j = T[:, a], T[:, b]
                dx, dy = (Fx[:, b] - Fx[:, a]) * w, (Fy[:, b] - Fy[:, a]) * w
                np.add.at(bx, j, dx); np.add.at(bx, i, -dx)
                np.add.at(by, j, dy); np.add.at(by, i, -dy)
            if bvec is not None:
                bi, bj, vec = bvec
                np.add.at(bx, bj, WB * vec[:, 0]); np.add.at(bx, bi, -WB * vec[:, 0])
                np.add.at(by, bj, WB * vec[:, 1]); np.add.at(by, bi, -WB * vec[:, 1])
            xs, ys = cur[:, 0].copy(), cur[:, 1].copy()
            xs[self.fix] = X[self.fix, 0]; ys[self.fix] = X[self.fix, 1]
            xs[self.free] = self.lu2.solve(bx[self.free] - self.Lfx @ xs[self.fix])
            ys[self.free] = self.lu2.solve(by[self.free] - self.Lfx @ ys[self.fix])
            cur = np.c_[xs, ys]
        return cur
