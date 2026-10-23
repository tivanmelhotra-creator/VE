"""Cubic Bézier fitting — Python port of VectorCraft's `trace/src/logo/bezier.rs` +
`outline.rs` (fit_open / styled_fit) + `geom.rs` helpers.

Source: https://github.com/storytold/vectorcraft (Apache-2.0).  Algorithm: P. J. Schneider,
"An algorithm for automatically fitting digitized curves" (Graphics Gems, 1990) — least
squares cubic with fixed end tangents, Newton re-parametrisation, split at the worst point.
On top (as in VectorCraft): resample → light smoothing → corner detection by turning angle
→ every run between corners becomes a straight line (if within `line_tol`) or fitted cubics.

A chain of cubics is stored as an (n, 4, 2) float array.
"""
import numpy as np

MAX_DEPTH = 14


# ------------------------------------------------------------------ geometry helpers
def bez(c, t):
    t = np.asarray(t, float)[..., None]
    s = 1 - t
    return s ** 3 * c[0] + 3 * t * s * s * c[1] + 3 * t * t * s * c[2] + t ** 3 * c[3]


def resample(p, step, closed=False):
    """Uniform arc-length resampling (geom.rs resample_open / resample_closed)."""
    p = np.asarray(p, float)
    if closed:
        p = np.vstack([p, p[:1]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    L = d[-1]
    if L < 1e-9:
        return p[:1].copy(), step
    n = max(2, int(round(L / step)) + 1)
    s = np.linspace(0, L, n)
    out = np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]
    if closed:
        out = out[:-1]
    return out, L / (n - 1)


def gauss_kernel(sigma):
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / max(sigma, 1e-6)) ** 2)
    return k / k.sum(), r


def smooth_open(p, sigma):
    """Gaussian smoothing with fixed ends (geom.rs smooth_open: the kernel shrinks near the ends)."""
    if sigma <= 0 or len(p) < 3:
        return p.copy()
    k, r = gauss_kernel(sigma)
    n = len(p)
    out = p.copy()
    for i in range(1, n - 1):
        rr = min(r, i, n - 1 - i)
        kk = k[r - rr:r + rr + 1]
        out[i] = (p[i - rr:i + rr + 1] * kk[:, None]).sum(0) / kk.sum()
    return out


def smooth_closed(p, sigma):
    if sigma <= 0 or len(p) < 3:
        return p.copy()
    k, r = gauss_kernel(sigma)
    n = len(p)
    idx = (np.arange(n)[:, None] + np.arange(-r, r + 1)[None]) % n
    return (p[idx] * k[None, :, None]).sum(1)


def turning(p, w, closed):
    n = len(p)
    i = np.arange(n)
    if closed:
        a, b = p[(i - w) % n], p[(i + w) % n]
    else:
        a, b = p[np.clip(i - w, 0, n - 1)], p[np.clip(i + w, 0, n - 1)]
    v1, v2 = p - a, b - p
    ang = np.degrees(np.abs(np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0], (v1 * v2).sum(1))))
    if not closed:
        ang[:w] = 0
        ang[n - w:] = 0
    return ang


def find_corners(p, w, thr, closed):
    n = len(p)
    if n < 8:
        return []
    ang = turning(p, w, closed)
    idx = []
    for i in range(n):
        a = ang[i]
        if a >= thr and a >= ang[(i - 1) % n] and a > ang[(i + 1) % n] and (not idx or i - idx[-1] > w):
            idx.append(i)
    if closed and len(idx) > 1 and idx[0] + n - idx[-1] <= w:
        idx.pop()
    return idx


def tangent(p, i, k, sign):
    j = min(i + k, len(p) - 1) if sign > 0 else max(i - k, 0)
    d = p[j] - p[i]
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else np.array([1.0, 0.0])


def line_cubic(a, b):
    return np.array([a, a + (b - a) / 3, a + (b - a) * 2 / 3, b])


# ------------------------------------------------------------------ Schneider
def _chord(p):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return d / d[-1] if d[-1] > 0 else d


def _fit_one(p, u, t1, t2):
    p0, p3 = p[0], p[-1]
    s = 1 - u
    b0, b1, b2, b3 = s ** 3, 3 * u * s * s, 3 * u * u * s, u ** 3
    a1 = t1[None] * b1[:, None]
    a2 = t2[None] * b2[:, None]
    c00 = (a1 * a1).sum(); c01 = (a1 * a2).sum(); c11 = (a2 * a2).sum()
    tmp = p - (p0[None] * (b0 + b1)[:, None] + p3[None] * (b2 + b3)[:, None])
    x0 = (a1 * tmp).sum(); x1 = (a2 * tmp).sum()
    det = c00 * c11 - c01 * c01
    seg = np.linalg.norm(p3 - p0)
    if abs(det) > 1e-12:
        al, ar = (x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det
    else:
        al = ar = 0.0
    if not np.isfinite(al) or not np.isfinite(ar) or al <= 1e-6 * seg or ar <= 1e-6 * seg:
        al = ar = seg / 3
    return np.array([p0, p0 + t1 * al, p3 + t2 * ar, p3])


def _reparam(c, p, u):
    s = 1 - u
    d = bez(c, u) - p
    v0, v1, v2 = c[1] - c[0], c[2] - c[1], c[3] - c[2]
    d1 = 3 * (v0[None] * (s * s)[:, None] + v1[None] * (2 * s * u)[:, None] + v2[None] * (u * u)[:, None])
    d2 = 6 * ((v1 - v0)[None] * s[:, None] + (v2 - v1)[None] * u[:, None])
    den = (d1 * d1).sum(1) + (d * d2).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        nu = np.where(np.abs(den) > 1e-12, u - (d * d1).sum(1) / den, u)
    return np.clip(nu, 0, 1)


def _max_err(c, p, u):
    e = np.linalg.norm(bez(c, u) - p, axis=1)
    i = int(np.argmax(e))
    return e[i], i


def _rec(p, t1, t2, tol, depth, out):
    if len(p) == 2:
        d = np.linalg.norm(p[1] - p[0]) / 3
        out.append(np.array([p[0], p[0] + t1 * d, p[1] + t2 * d, p[1]]))
        return
    u = _chord(p)
    c = _fit_one(p, u, t1, t2)
    for _ in range(4):
        e, _i = _max_err(c, p, u)
        if e < tol:
            out.append(c); return
        if e > tol * 6:
            break
        u = _reparam(c, p, u)
        c = _fit_one(p, u, t1, t2)
    e, split = _max_err(c, p, u)
    if e < tol or depth >= MAX_DEPTH or len(p) < 6:
        out.append(c); return
    s = int(np.clip(split, 2, len(p) - 3))
    d = p[s - 1] - p[s + 1]
    n = np.linalg.norm(d)
    tc = d / n if n > 1e-12 else t1
    _rec(p[:s + 1], t1, tc, tol, depth + 1, out)
    _rec(p[s:], -tc, t2, tol, depth + 1, out)


def fit_cubics(p, t1, t2, tol):
    out = []
    p = np.asarray(p, float)
    if len(p) >= 2 and tol > 0:
        _rec(p, np.asarray(t1, float), np.asarray(t2, float), tol, 0, out)
    return out


# ------------------------------------------------------------------ styled fitting
class Style:
    """VectorCraft `Style` (lengths in working pixels)."""
    def __init__(self, step=0.5, sigma=1.2, corner_deg=60.0, corner_win=2.5, tol=0.6, line_tol=0.45):
        self.step, self.sigma, self.corner_deg, self.corner_win = step, sigma, corner_deg, corner_win
        self.tol, self.line_tol = tol, line_tol


def _fit_run(run, sg, st, segs):
    if len(run) < 4:
        if len(run) >= 2 and np.linalg.norm(run[-1] - run[0]) > 1e-9:
            segs.append(line_cubic(run[0], run[-1]))
        return
    sm = smooth_open(run, sg)
    a, b = sm[0], sm[-1]
    L = np.linalg.norm(b - a)
    if L > 1.5:
        nrm = np.array([-(b - a)[1], (b - a)[0]]) / L
        if np.all(np.abs((sm - a) @ nrm) < st.line_tol):
            segs.append(line_cubic(a, b)); return
    segs.extend(fit_cubics(sm, tangent(sm, 0, 4, 1), tangent(sm, len(sm) - 1, 4, -1), st.tol))


def fit_open(p, st):
    """Open polyline → cubic chain, both ends kept (outline.rs fit_open)."""
    p = np.asarray(p, float)
    if len(p) < 2:
        return np.zeros((0, 4, 2))
    r, step = resample(p, st.step)
    if len(r) < 2:
        return np.array([line_cubic(p[0], p[-1])])
    s0 = smooth_open(r, max(1.0, 0.18 / step))
    w = max(2, int(round(st.corner_win / step)))
    cuts = [0] + find_corners(s0, w, st.corner_deg, False) + [len(r) - 1]
    cuts = sorted(set(cuts))
    segs = []
    for a, b in zip(cuts, cuts[1:]):
        n0 = len(segs)
        _fit_run(r[a:b + 1], st.sigma / step, st, segs)
        if len(segs) == n0 and np.linalg.norm(r[b] - r[a]) > 1e-9:
            segs.append(line_cubic(r[a], r[b]))
    if not segs:
        segs = [line_cubic(p[0], p[-1])]
    return np.array(segs)


def fit_closed(p, st):
    """Closed loop → cubic chain (outline.rs styled_fit without arc snapping)."""
    p = np.asarray(p, float)
    if len(p) < 4:
        return None
    r, step = resample(p, st.step, closed=True)
    if len(r) < 6:
        return None
    s0 = smooth_closed(r, max(1.0, 0.18 / step))
    w = max(2, int(round(st.corner_win / step)))
    corners = find_corners(s0, w, st.corner_deg, True)
    n = len(r)
    segs = []
    sg = st.sigma / step
    if not corners:
        s = smooth_closed(r, sg)
        t = s[1] - s[-1]; t = t / max(np.linalg.norm(t), 1e-12)
        q = np.vstack([s, s[:1]])
        segs = fit_cubics(q, t, -t, st.tol)
        if len(segs) < 2:   # a closed editable sub-path needs two anchors
            h = len(q) // 2
            segs = fit_cubics(q[:h + 1], t, tangent(q, h, 3, -1), st.tol) + \
                fit_cubics(q[h:], tangent(q, h, 3, 1), -t, st.tol)
    else:
        for ci, i0 in enumerate(corners):
            i1 = corners[(ci + 1) % len(corners)]
            cnt = (i1 - i0) % n or n
            run = r[(i0 + np.arange(cnt + 1)) % n]
            _fit_run(run, sg, st, segs)
    return np.array(segs) if segs else None


# ------------------------------------------------------------------ chain <-> flat points
def chain_to_points(segs, closed):
    """(n,4,2) → flat control-point array [a0,h,h,a1,h,h,a2…] (closed: last anchor = first, dropped)."""
    if len(segs) == 0:
        return np.zeros((0, 2))
    pts = [segs[0][0]]
    for c in segs:
        pts += [c[1], c[2], c[3]]
    P = np.array(pts)
    return P[:-1] if closed else P


def points_to_chain(P, closed):
    P = np.asarray(P)
    if closed:
        P = np.vstack([P, P[:1]])
    n = (len(P) - 1) // 3
    idx = np.arange(n)[:, None] * 3 + np.arange(4)[None]
    return P[idx]


def sample_chain(segs, per_seg=None, spacing=1.5):
    """Dense points along a chain (for rendering / checks)."""
    out = []
    for c in segs:
        L = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
        k = per_seg or max(2, int(L / spacing) + 1)
        t = np.linspace(0, 1, k, endpoint=False)
        out.append(bez(c, t))
    if len(segs):
        out.append(segs[-1][3][None])
    return np.vstack(out) if out else np.zeros((0, 2))
