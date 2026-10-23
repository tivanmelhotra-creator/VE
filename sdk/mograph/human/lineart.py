"""
mograph.human.lineart — 3D mesh → professional VECTOR line art.

Pipeline (per frame):
 1. GL pass (moderngl/EGL, llvmpipe ok): linear depth + part id buffer at `ss`× the output scale.
 2. Smooth occluding contours (Hertzmann–Zorin): zero set of g = n_s · (cam − p) per triangle, where n_s are normals
    of a Taubin-SMOOTHED copy of the mesh (per-part smoothing → ribs / knuckles / face bumps never make lines).
    Crossing points are keyed by mesh edge → exact chaining into long polylines.
 3. Visibility per point vs. the depth buffer (min of 3×3, tolerance), and LINE IMPORTANCE = depth jump behind the
    contour (5×5 max depth − own depth):  big jump = outer silhouette (heavy), small jump = inner contour (light),
    below `min_jump` = dropped (nose wings, lips, shallow folds).
 4. Extra stroke sources: boundary/iso lines (hems, hairline, cuffs) via `iso_lines`, explicit 3D polylines
    (creases, face features) via `Stroke3D`.
 5. Chain → split at visibility / weight changes → drop short → Chaikin smooth → tapered variable-width ribbons
    (cairo, AA) → RGBA image, plus SVG export.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import cairo
except Exception:  # pragma: no cover
    cairo = None


# ----------------------------------------------------------------------------------------------------- camera
@dataclass
class Camera:
    target: tuple = (0.0, 9.0, 0.0)        # dm
    yaw: float = 0.0                      # deg, 0 = facing the character's front, +90 = character's left side
    pitch: float = 4.0                    # deg, + = camera above looking down
    dist: float = 60.0                    # dm
    fov: float = 18.0                     # vertical deg (long lens = flat, illustration-like)
    W: int = 900
    H: int = 1200
    ortho: bool = False
    ortho_h: float = 20.0                 # world height visible (ortho)

    def eye(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        d = np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)], np.float64)
        return np.asarray(self.target, np.float64) + d * self.dist

    def view(self):
        e = self.eye()
        t = np.asarray(self.target, np.float64)
        f = t - e
        f /= np.linalg.norm(f)
        up = np.array([0, 1.0, 0])
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        M = np.eye(4)
        M[0, :3], M[1, :3], M[2, :3] = r, u, -f
        M[:3, 3] = -M[:3, :3] @ e
        return M

    def proj(self, near=1.0, far=400.0):
        a = self.W / self.H
        P = np.zeros((4, 4))
        if self.ortho:
            h = self.ortho_h / 2
            P[0, 0], P[1, 1] = 1 / (h * a), 1 / h
            P[2, 2], P[2, 3], P[3, 3] = -2 / (far - near), -(far + near) / (far - near), 1
        else:
            f = 1 / math.tan(math.radians(self.fov) / 2)
            P[0, 0], P[1, 1] = f / a, f
            P[2, 2], P[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
            P[3, 2] = -1
        return P

    def project(self, X):
        """world (N,3) → pixel xy (N,2), linear depth (N,)"""
        V = self.view()
        P = self.proj()
        Xh = np.c_[X, np.ones(len(X))]
        c = Xh @ V.T
        clip = c @ P.T
        w = clip[:, 3:4]
        ndc = clip[:, :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)
        px = np.c_[(ndc[:, 0] + 1) * 0.5 * self.W, (1 - ndc[:, 1]) * 0.5 * self.H]
        return px, -c[:, 2]

    def frame(self, P3, margin=0.08):
        """Fit target/dist (or ortho_h) so the 3D points fill the frame with `margin`."""
        lo, hi = P3.min(0), P3.max(0)
        c = (lo + hi) / 2
        self.target = tuple(c)
        h = (hi[1] - lo[1])
        w = max(hi[0] - lo[0], hi[2] - lo[2])
        need = max(h, w * self.H / self.W) * (1 + 2 * margin)
        if self.ortho:
            self.ortho_h = need
        else:
            self.dist = need / 2 / math.tan(math.radians(self.fov) / 2) + w / 2
        return self


# ----------------------------------------------------------------------------------------------------- GL
_GL = {}


def _ctx():
    if "ctx" not in _GL:
        import moderngl
        ctx = moderngl.create_standalone_context(backend="egl")
        prog = ctx.program(
            vertex_shader="""#version 330
            uniform mat4 V; uniform mat4 P;
            in vec3 pos; in float pid;
            out float d; flat out float id;
            void main(){ vec4 c = V*vec4(pos,1.0); d = -c.z; id = pid; gl_Position = P*c; }""",
            fragment_shader="""#version 330
            in float d; flat in float id; out vec4 o;
            void main(){ o = vec4(d, id, 0.0, 1.0); }""")
        _GL.update(ctx=ctx, prog=prog, fbo={})
    return _GL["ctx"], _GL["prog"]


def gl_buffers(cam: Camera, parts, ss: float = 1.0):
    """Render linear depth (inf = empty) and part id (0 = empty) at ss× resolution."""
    import moderngl
    ctx, prog = _ctx()
    W, H = int(cam.W * ss), int(cam.H * ss)
    key = (W, H)
    if key not in _GL["fbo"]:
        tex = ctx.texture((W, H), 4, dtype="f4")
        dep = ctx.depth_renderbuffer((W, H))
        _GL["fbo"] = {key: (ctx.framebuffer([tex], dep), tex)}
    fbo, tex = _GL["fbo"][key]
    fbo.use()
    ctx.viewport = (0, 0, W, H)
    fbo.clear(1e6, 0.0, 0.0, 0.0, depth=1.0)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    prog["V"].write(cam.view().T.astype("f4").tobytes())
    prog["P"].write(cam.proj().T.astype("f4").tobytes())
    for p in parts:
        if not p.occluder:
            continue
        X = p.verts[p.tris].reshape(-1, 3).astype("f4")
        ids = np.full((len(X), 1), float(p.pid), "f4")
        data = np.hstack([X, ids]).astype("f4")
        vbo = ctx.buffer(data.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "3f 1f", "pos", "pid")])
        vao.render(moderngl.TRIANGLES)
        vao.release()
        vbo.release()
    buf = np.frombuffer(tex.read(), "f4").reshape(H, W, 4)[::-1]
    return buf[..., 0].copy(), buf[..., 1].round().astype(np.int32)


# ----------------------------------------------------------------------------------------------------- mesh utils
class MeshTopo:
    """Static topology helpers for a triangle mesh (built once per part type)."""

    def __init__(self, tris: np.ndarray, nverts: int):
        import scipy.sparse as sp
        self.tris = tris.astype(np.int32)
        self.n = nverts
        e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        e = np.sort(e, 1)
        e = np.unique(e, axis=0)
        self.edges = e
        A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(nverts, nverts)).tocsr()
        deg = np.asarray(A.sum(1)).ravel()
        self.used = deg > 0
        deg[deg == 0] = 1
        self.L = sp.diags(1 / deg) @ A                      # row-normalised adjacency (umbrella operator)
        nt = len(tris)
        self.VF = sp.coo_matrix((np.ones(nt * 3), (tris.ravel(), np.repeat(np.arange(nt), 3))),
                                shape=(nverts, nt)).tocsr()
        # boundary edges (open garment hems)
        ee = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), 1)
        u, c = np.unique(ee, axis=0, return_counts=True)
        self.boundary = u[c == 1]

    def smooth(self, X, iters=0, lam=0.5, mu=-0.53, weight=None):
        """Taubin λ|μ smoothing (no shrink). weight (N,) in 0..1 scales the per-vertex step."""
        if iters <= 0:
            return X
        Y = X.copy()
        w = None if weight is None else weight[:, None]
        for _ in range(iters):
            for k in (lam, mu):
                D = self.L @ Y - Y
                Y = Y + (k * D if w is None else k * w * D)
        return Y

    def vnormals(self, X):
        t = self.tris
        fn = np.cross(X[t[:, 1]] - X[t[:, 0]], X[t[:, 2]] - X[t[:, 0]])
        n = self.VF @ fn
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


# ----------------------------------------------------------------------------------------------------- parts
@dataclass
class Part:
    name: str
    verts: np.ndarray                    # posed (N,3)
    tris: np.ndarray                     # (T,3) into verts
    topo: MeshTopo
    pid: int = 1
    smooth_iters: int = 6
    smooth_w: np.ndarray | None = None   # per-vertex smoothing weight (0 keeps detail, 1 full)
    contour: bool = True                 # extract occluding contours
    occluder: bool = True
    min_jump: float = 0.35               # dm — inner contours with less depth jump are dropped
    weight: float = 1.0                  # stroke weight multiplier
    clip: np.ndarray | None = None       # per-vertex scalar; contour points with clip > 0 are discarded
    boundary: bool = False               # draw open-mesh boundary edges (hems) as lines
    kind: str = "body"                   # body | cloth | hair (affects the default style)


@dataclass
class Stroke3D:
    pts: np.ndarray                      # (M,3) world
    weight: float = 0.6
    kind: str = "detail"
    taper: float = 0.5
    check_vis: bool = True
    pid: int = 0                         # part the stroke lies on (visibility uses its own surface)


# ----------------------------------------------------------------------------------------------------- contours
def contour_segments(part: Part, eye: np.ndarray, ortho_dir=None):
    """Zero crossings of g = n·(eye − p). Returns crossing points (K,3), key edges (K,2), segments (S,2) into K,
    plus per-point g-gradient sign (unused)."""
    X = part.verts
    Xs = part.topo.smooth(X, part.smooth_iters, weight=part.smooth_w) if part.smooth_iters else X
    n = part.topo.vnormals(Xs)
    if ortho_dir is not None:
        g = n @ ortho_dir
    else:
        v = eye[None, :] - Xs
        g = np.einsum("ij,ij->i", n, v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9))
    t = part.tris
    gt = g[t]
    s = gt > 0
    cnt = s.sum(1)
    m = (cnt == 1) | (cnt == 2)
    if part.clip is not None:
        m &= ~(part.clip[t] > 0).all(1)
    t, gt = t[m], gt[m]
    if not len(t):
        return np.zeros((0, 3)), np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64), np.zeros(0)
    # for each triangle, the 2 edges with a sign change
    E = np.stack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], 1)          # (T,3,2)
    GE = np.stack([gt[:, [0, 1]], gt[:, [1, 2]], gt[:, [2, 0]]], 1)
    ch = (GE[..., 0] > 0) != (GE[..., 1] > 0)                            # (T,3)
    tri_idx, e_idx = np.nonzero(ch)
    ek = np.sort(E[tri_idx, e_idx], 1)                                    # edge keys
    keys, inv = np.unique(ek, axis=0, return_inverse=True)
    inv = inv.ravel()
    ga, gb = g[keys[:, 0]], g[keys[:, 1]]
    w = ga / (ga - gb + np.where(ga - gb == 0, 1e-12, 0))
    w = np.clip(w, 0, 1)[:, None]
    P = X[keys[:, 0]] * (1 - w) + X[keys[:, 1]] * w                       # on the REAL (unsmoothed) surface
    clipv = None
    if part.clip is not None:
        clipv = part.clip[keys[:, 0]] * (1 - w[:, 0]) + part.clip[keys[:, 1]] * w[:, 0]
    inv = inv.reshape(-1)
    seg = inv.reshape(-1, 2) if len(inv) % 2 == 0 else None
    if seg is None:   # should not happen (exactly 2 crossings per triangle)
        seg = inv[: len(inv) // 2 * 2].reshape(-1, 2)
    return P, keys, seg, clipv


def iso_segments(part: Part, field_v: np.ndarray, level=0.0):
    """Iso line field = level on the mesh (hems, cuffs, hairline). Same output layout as contour_segments."""
    t = part.tris
    gt = field_v[t] - level
    s = gt > 0
    cnt = s.sum(1)
    m = (cnt == 1) | (cnt == 2)
    t, gt = t[m], gt[m]
    if not len(t):
        return np.zeros((0, 3)), np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64)
    E = np.stack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], 1)
    GE = np.stack([gt[:, [0, 1]], gt[:, [1, 2]], gt[:, [2, 0]]], 1)
    ch = (GE[..., 0] > 0) != (GE[..., 1] > 0)
    tri_idx, e_idx = np.nonzero(ch)
    ek = np.sort(E[tri_idx, e_idx], 1)
    keys, inv = np.unique(ek, axis=0, return_inverse=True)
    g = field_v - level
    ga, gb = g[keys[:, 0]], g[keys[:, 1]]
    w = np.clip(ga / np.where(ga - gb == 0, 1e-12, ga - gb), 0, 1)[:, None]
    P = part.verts[keys[:, 0]] * (1 - w) + part.verts[keys[:, 1]] * w
    return P, keys, inv.reshape(-1)[: (len(inv) // 2) * 2].reshape(-1, 2)


def boundary_polyline_segments(part: Part):
    b = part.topo.boundary
    if not len(b):
        return np.zeros((0, 3)), np.zeros((0, 2), np.int64)
    used, inv = np.unique(b.ravel(), return_inverse=True)
    return part.verts[used], inv.reshape(-1, 2)


def chain(seg: np.ndarray, npts: int) -> list[np.ndarray]:
    """Chain undirected segments (S,2) over point ids into polylines (lists of point ids)."""
    if not len(seg):
        return []
    adj = [[] for _ in range(npts)]
    for k, (a, b) in enumerate(seg):
        adj[a].append((b, k))
        adj[b].append((a, k))
    used = np.zeros(len(seg), bool)
    out = []

    def walk(start, first_edge):
        path = [start]
        cur, (nxt, k) = start, first_edge
        while True:
            used[k] = True
            path.append(nxt)
            cur = nxt
            cand = [(b, kk) for b, kk in adj[cur] if not used[kk]]
            if len(cand) != 1 or len(adj[cur]) > 2:
                if cand and len(adj[cur]) <= 2:
                    nxt, k = cand[0]
                    continue
                break
            nxt, k = cand[0]
        return path
    # start at endpoints / junctions first, then loops
    order = sorted(range(npts), key=lambda i: (len(adj[i]) == 2, ))
    for s in order:
        for (b, k) in adj[s]:
            if used[k]:
                continue
            p = walk(s, (b, k))
            out.append(np.asarray(p))
    return out


# ----------------------------------------------------------------------------------------------------- visibility
def _sample(img, xy):
    H, W = img.shape
    x = np.clip(xy[:, 0].astype(int), 0, W - 1)
    y = np.clip(xy[:, 1].astype(int), 0, H - 1)
    return img[y, x]


def depth_maps(depth, rmin=1, rjump=3):
    import cv2
    D = depth.astype(np.float32)
    big = D >= 1e5
    Dmin = cv2.erode(D, np.ones((2 * rmin + 1, 2 * rmin + 1), np.uint8))
    Dn = D.copy()
    Dn[big] = np.float32(1e4)                         # background = very far (silhouette against empty = max jump)
    Dmax = cv2.dilate(Dn, np.ones((2 * rjump + 1, 2 * rjump + 1), np.uint8))
    return Dmin, Dmax


def visibility(P, cam, Dmin, Dmax, ss, tol):
    xy, z = cam.project(P)
    xs = xy * ss
    H, W = Dmin.shape
    inside = (xs[:, 0] >= 0) & (xs[:, 0] < W) & (xs[:, 1] >= 0) & (xs[:, 1] < H)
    dmin = _sample(Dmin, xs)
    dmax = _sample(Dmax, xs)
    vis = inside & (z <= dmin + tol)
    jump = np.clip(dmax - z, 0, 1e4)
    return xy, z, vis, jump


# ----------------------------------------------------------------------------------------------------- styling
@dataclass
class Style:
    color: tuple = (0.07, 0.07, 0.08)
    width: float = 3.2            # px at 1000 px figure height — the OUTER silhouette weight
    inner: float = 0.55           # inner contours width factor
    detail: float = 0.42          # details / creases width factor
    hem: float = 0.62
    taper: float = 0.18           # fraction of length tapered at each end (max 28 px)
    min_len: float = 9.0          # px — shorter strokes are dropped
    jump_outer: float = 3.0       # dm depth jump that counts as full outer silhouette
    smooth: int = 2               # Chaikin passes
    pressure: float = 0.18        # slow width wobble (0 = mechanical, ~0.2 = ink pen)
    bg: tuple | None = None       # None = transparent
    cap_round: bool = True


def chaikin(p, n=2, closed=False):
    for _ in range(n):
        if len(p) < 3:
            return p
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((len(q) * 2, 2))
        mid[0::2], mid[1::2] = q, r
        p = np.vstack([p[:1], mid, p[-1:]]) if not closed else mid
    return p


def simplify(p, eps=0.35):
    """Ramer–Douglas–Peucker (px)."""
    if len(p) < 3:
        return p
    keep = np.zeros(len(p), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = p[b] - p[a]
        L = np.hypot(*d) + 1e-9
        seg = p[a + 1:b] - p[a]
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / L
        i = int(np.argmax(dist))
        if dist[i] > eps:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return p[keep]


@dataclass
class Stroke2D:
    pts: np.ndarray               # (M,2) px
    w: np.ndarray                 # (M,) px widths (full)
    kind: str = "contour"
    closed: bool = False


def split_runs(mask):
    """indices runs where mask is True → list of (a, b) inclusive."""
    runs = []
    a = None
    for i, m in enumerate(mask):
        if m and a is None:
            a = i
        elif not m and a is not None:
            runs.append((a, i - 1))
            a = None
    if a is not None:
        runs.append((a, len(mask) - 1))
    return runs


def taper_profile(s, L, taper_px):
    if L <= 1e-6:
        return np.ones_like(s)
    t = np.minimum(np.minimum(s, L - s) / max(1e-6, taper_px), 1.0)
    return 0.25 + 0.75 * np.sin(t * math.pi / 2) ** 0.8


def make_stroke(xy, wfac, style: Style, scale: float, kind="contour", taper=None, seed=0):
    if len(xy) < 2:
        return None
    p = simplify(xy, 0.3)
    keep_w = np.interp(np.linspace(0, 1, len(p)), np.linspace(0, 1, len(wfac)), wfac)
    p = chaikin(p, style.smooth)
    wf = np.interp(np.linspace(0, 1, len(p)), np.linspace(0, 1, len(keep_w)), keep_w)
    d = np.hypot(*np.diff(p, axis=0).T)
    s = np.r_[0, np.cumsum(d)]
    L = s[-1]
    if L < style.min_len * scale:
        return None
    tp = min(28 * scale, (style.taper if taper is None else taper) * L)
    prof = taper_profile(s, L, tp)
    if style.pressure > 0:
        rng = np.random.default_rng(seed)
        ph = rng.uniform(0, 6.28, 2)
        prof = prof * (1 + style.pressure * 0.5 * (np.sin(s / (90 * scale) + ph[0]) * 0.6 +
                                                    np.sin(s / (37 * scale) + ph[1]) * 0.4))
    w = style.width * scale * wf * prof
    return Stroke2D(p, w, kind)


# ----------------------------------------------------------------------------------------------------- drawing
def ribbon(p, w):
    """left/right offset polygon of a variable-width polyline."""
    d = np.gradient(p, axis=0)
    n = np.c_[-d[:, 1], d[:, 0]]
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    r = (w / 2)[:, None]
    return p + n * r, p - n * r


def draw_strokes(strokes, W, H, style: Style, ctx=None, surface=None):
    if cairo is None:
        raise RuntimeError("pycairo missing: pip install pycairo")
    own = ctx is None
    if own:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        ctx = cairo.Context(surface)
        if style.bg is not None:
            ctx.set_source_rgb(*style.bg)
            ctx.paint()
    ctx.set_source_rgba(*style.color, 1.0)
    for st in strokes:
        if st is None or len(st.pts) < 2:
            continue
        L, R = ribbon(st.pts, st.w)
        ctx.new_path()
        ctx.move_to(*L[0])
        for q in L[1:]:
            ctx.line_to(*q)
        if style.cap_round:
            a = st.pts[-1]
            ctx.arc(a[0], a[1], st.w[-1] / 2, 0, 2 * math.pi) if False else None
        for q in R[::-1]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.fill()
        if style.cap_round:
            for i in (0, -1):
                c = st.pts[i]
                ctx.arc(c[0], c[1], st.w[i] / 2, 0, 2 * math.pi)
                ctx.fill()
    if own:
        surface.flush()
        buf = np.ndarray((H, W, 4), np.uint8, surface.get_data()).copy()
        rgba = buf[..., [2, 1, 0, 3]]
        return rgba
    return None


def strokes_svg(strokes, W, H, style: Style) -> str:
    col = "#%02x%02x%02x" % tuple(int(c * 255) for c in style.color)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    if style.bg is not None:
        out.append(f'<rect width="100%" height="100%" fill="#%02x%02x%02x"/>' % tuple(int(c * 255) for c in style.bg))
    out.append(f'<g fill="{col}" stroke="none">')
    for st in strokes:
        if st is None or len(st.pts) < 2:
            continue
        L, R = ribbon(st.pts, st.w)
        poly = np.vstack([L, R[::-1]])
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in poly) + "Z"
        out.append(f'<path d="{d}"/>')
        for i in (0, -1):
            c = st.pts[i]
            out.append(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{st.w[i] / 2:.2f}"/>')
    out.append("</g></svg>")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------- frame
@dataclass
class LineFrame:
    strokes: list = field(default_factory=list)
    depth: np.ndarray | None = None
    ids: np.ndarray | None = None


def render_lines(parts: list[Part], cam: Camera, style: Style, extra: list[Stroke3D] = (), iso: list = (),
                 ss: float = 1.0, fig_px: float | None = None, vis_tol: float = 0.06, seed: int = 0) -> LineFrame:
    """parts → visible styled 2D strokes.
    iso: list of (part, field_v, level, weight, kind) iso-line requests (hems, hairline…)."""
    depth, ids = gl_buffers(cam, parts, ss)
    Dmin, Dmax = depth_maps(depth, 1, max(2, int(round(3 * ss))))
    scale = (fig_px or cam.H) / 1000.0
    eye = cam.eye()
    odir = None
    if cam.ortho:
        odir = eye - np.asarray(cam.target)
        odir /= np.linalg.norm(odir)
    strokes = []
    k = 0
    for part in parts:
        if not part.contour:
            continue
        P, keys, seg, clipv = contour_segments(part, eye, odir)
        if not len(P):
            continue
        xy, z, vis, jump = visibility(P, cam, Dmin, Dmax, ss, vis_tol)
        importance = np.clip((jump - part.min_jump) / max(1e-6, style.jump_outer - part.min_jump), 0, 1)
        keep = vis & (jump >= part.min_jump)
        for ch in chain(seg, len(P)):
            for a, b in split_runs(keep[ch]):
                ids_ = ch[a:b + 1]
                if len(ids_) < 3:
                    continue
                imp = importance[ids_]
                # smooth importance along the chain so widths flow
                if len(imp) > 5:
                    kern = np.ones(5) / 5
                    imp = np.convolve(np.pad(imp, 2, mode="edge"), kern, "valid")
                wf = part.weight * (style.inner + (1 - style.inner) * imp)
                s = make_stroke(xy[ids_], wf, style, scale, "contour", seed=seed + k)
                k += 1
                if s is not None:
                    strokes.append(s)
        if part.boundary:
            Pb, segb = boundary_polyline_segments(part)
            if len(Pb):
                xy, z, vis, jump = visibility(Pb, cam, Dmin, Dmax, ss, vis_tol * 1.6)
                for ch in chain(segb, len(Pb)):
                    for a, b in split_runs(vis[ch]):
                        ids_ = ch[a:b + 1]
                        if len(ids_) >= 3:
                            s = make_stroke(xy[ids_], np.full(len(ids_), part.weight * style.hem), style, scale,
                                            "hem", seed=seed + k)
                            k += 1
                            if s is not None:
                                strokes.append(s)
    for (part, fv, level, wgt, kind) in iso:
        P, keys, seg = iso_segments(part, fv, level)
        if not len(P):
            continue
        xy, z, vis, jump = visibility(P, cam, Dmin, Dmax, ss, vis_tol * 1.6)
        for ch in chain(seg, len(P)):
            for a, b in split_runs(vis[ch]):
                ids_ = ch[a:b + 1]
                if len(ids_) >= 3:
                    s = make_stroke(xy[ids_], np.full(len(ids_), wgt), style, scale, kind, seed=seed + k)
                    k += 1
                    if s is not None:
                        strokes.append(s)
    for st in extra:
        if len(st.pts) < 2:
            continue
        xy, z, vis, jump = visibility(st.pts, cam, Dmin, Dmax, ss, vis_tol * 2.5)
        if not st.check_vis:
            vis = np.ones(len(xy), bool)
        for a, b in split_runs(vis):
            if b - a >= 1:
                s = make_stroke(xy[a:b + 1], np.full(b - a + 1, st.weight), style, scale, st.kind,
                                taper=st.taper, seed=seed + k)
                k += 1
                if s is not None:
                    strokes.append(s)
    return LineFrame(strokes, depth, ids)
