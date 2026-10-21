"""
mograph.real3d — v14.5 TRUE 3D product renderer (textured meshes, deferred PBR, shadow map, SSAO, bump).

Why: a single image "thickened" (image_to_3d slab/cushion) is not a 3D object — it breaks as soon as it turns.
real3d rebuilds the object from REAL geometry (img2threejs idea: component tree of procedural primitives) and gives
every part a procedural texture whose palette is sampled from the reference photo.  Every angle (360° yaw, any pitch,
top, bottom, back) is a real view of a closed solid.

Pipeline per frame (CPU, numba z-buffer from zraster.py):
  meshes (pos, normal, uv, local pos) → rotate → perspective G-buffer (2× SSAA) → per-pixel materials
  (albedo / roughness / metalness / height) → screen-space bump → lights (key + fill + rim, hemisphere ambient,
  studio softbox environment for reflections) → key-light SHADOW MAP (toppings shadow the cheese, the arrow shadows
  the card) → SSAO → ACES tone map → sRGB → downsample (anti-aliased alpha).

Models: see real3d_models.py   (soda_can, pizza_slice, growth_card …).  Elements / layers: `real3d` / object_3d
"model": "pizza_slice".
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import zraster

# ============================================================================================ noise
@lru_cache(maxsize=8)
def noise_tex(seed=0, n=256, sigma=3.0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    t = gaussian_filter(rng.standard_normal((n, n)).astype(np.float32), sigma, mode="wrap")
    t = (t - t.min()) / max(1e-6, t.max() - t.min())
    return t.astype(np.float32)


def samp(tex, u, v):
    """Bilinear, wrapping sample of a 2D texture at texture-repeat coords (u, v) (any shape)."""
    h, w = tex.shape[:2]
    x = (np.asarray(u, np.float32) * w) % w
    y = (np.asarray(v, np.float32) * h) % h
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx, fy = x - x0, y - y0
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    x0 %= w
    y0 %= h
    a = tex[y0, x0]
    b = tex[y0, x1]
    c = tex[y1, x0]
    d = tex[y1, x1]
    if tex.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(u, v, oct=4, seed=0, gain=0.5):
    t = noise_tex(seed)
    s, a, tot = 1.0, 1.0, 0.0
    out = 0.0
    for k in range(oct):
        out = out + samp(t, u * s + 0.37 * k, v * s + 0.71 * k) * a
        tot += a
        s *= 2.03
        a *= gain
    return out / tot


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def srgb(*c):
    """sRGB 0..255 → linear 0..1 (np array)."""
    return (np.array(c, np.float32) / 255.0) ** 2.2


# ============================================================================================ meshes
class Mesh:
    """v (n,3) n (n,3) uv (n,2) f (m,3).  Local coordinates (texture space = local position)."""

    def __init__(self, v, n, uv, f):
        self.v = np.asarray(v, np.float32)
        self.n = _norm(np.asarray(n, np.float32))
        self.uv = np.asarray(uv, np.float32)
        self.f = np.asarray(f, np.int32)


def _norm(a):
    return a / np.maximum(1e-8, np.linalg.norm(a, axis=-1, keepdims=True))


def grid_faces(rows, cols, wrap=False):
    """Quads of a rows×cols vertex grid → triangles."""
    f = []
    cc = cols if wrap else cols - 1
    for i in range(rows - 1):
        for j in range(cc):
            j1 = (j + 1) % cols
            a, b, c, d = i * cols + j, i * cols + j1, (i + 1) * cols + j, (i + 1) * cols + j1
            f += [(a, c, d), (a, d, b)]
    return np.array(f, np.int32)


def lathe(profile, segs=64):
    """Surface of revolution around +Y.  profile = [(r, y), ...] from bottom centre → outward → top centre.
    Repeat a point to make a CREASE (hard edge).  Normals are analytic (outward = right of the travel direction)."""
    P = np.array(profile, np.float32)
    m = len(P)
    tang = np.zeros_like(P)
    for i in range(m):
        prv = P[i - 1] if i > 0 else None
        nxt = P[i + 1] if i < m - 1 else None
        if nxt is not None and np.allclose(nxt, P[i]):
            d = P[i] - prv if prv is not None else np.array([1.0, 0.0])
        elif prv is not None and np.allclose(prv, P[i]):
            d = nxt - P[i] if nxt is not None else np.array([-1.0, 0.0])
        elif prv is None:
            d = nxt - P[i]
        elif nxt is None:
            d = P[i] - prv
        else:
            d = nxt - prv
        tang[i] = d / max(1e-8, float(np.hypot(*d)))
    n2 = np.stack([tang[:, 1], -tang[:, 0]], 1)                      # (nr, ny)
    seg = np.hypot(*(np.diff(P, axis=0).T))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s /= max(1e-8, s[-1])
    cols = segs + 1
    th = np.linspace(0, 2 * math.pi, cols).astype(np.float32)
    st, ct = np.sin(th), np.cos(th)
    V = np.zeros((m, cols, 3), np.float32)
    N = np.zeros((m, cols, 3), np.float32)
    UV = np.zeros((m, cols, 2), np.float32)
    V[..., 0] = P[:, None, 0] * st[None]
    V[..., 1] = P[:, None, 1]
    V[..., 2] = P[:, None, 0] * ct[None]
    N[..., 0] = n2[:, None, 0] * st[None]
    N[..., 1] = n2[:, None, 1]
    N[..., 2] = n2[:, None, 0] * ct[None]
    UV[..., 0] = (th / (2 * math.pi))[None]
    UV[..., 1] = s[:, None]
    return Mesh(V.reshape(-1, 3), N.reshape(-1, 3), UV.reshape(-1, 2), grid_faces(m, cols))


def from_sculpt(v, f, n):
    """Wrap a sculpt3d mesh; orient normals outward (flip if most point inward)."""
    c = v.mean(0)
    if np.mean(np.sum((v - c) * n, 1)) < 0:
        n = -n
    return Mesh(v, n, v[:, :2], f)


def rbox(w, h, d, bevel=0.03, subdiv=6):
    from .sculpt3d import mesh_rounded_box
    v, f, n = mesh_rounded_box(w, h, d, bevel=bevel, subdiv=subdiv)
    return from_sculpt(v, f, n)


def prism(shape, depth):
    from .sculpt3d import mesh_extrude
    v, f, n = mesh_extrude(shape, depth)
    return from_sculpt(v, f, n)


def cyl_z(r, depth, segs=32, bevel=0.0):
    """Cylinder along Z (disc / joint / button), optional rounded edge."""
    hd = depth / 2
    b = min(bevel, hd * 0.9, r * 0.5)
    if b > 0:
        prof = [(0, -hd), (r - b, -hd)]
        for k in range(1, 5):
            a = -math.pi / 2 + k * (math.pi / 2) / 4
            prof.append((r - b + b * math.cos(a), -hd + b + b * math.sin(a)))
        for k in range(0, 5):
            a = k * (math.pi / 2) / 4
            prof.append((r - b + b * math.cos(a), hd - b + b * math.sin(a)))
        prof.append((0, hd))
    else:
        prof = [(0, -hd), (r, -hd), (r, -hd), (r, hd), (r, hd), (0, hd)]
    m = lathe(prof, segs)
    R = rot(90, 0, 0)                     # Y axis → Z axis
    return xform(m, R, (0, 0, 0))


def xform(m, R=None, t=(0, 0, 0)):
    """Apply a rigid transform to a mesh's positions/normals but KEEP its texture space (uv + local pos)."""
    R = np.eye(3, dtype=np.float32) if R is None else np.asarray(R, np.float32)
    out = Mesh(m.v @ R.T + np.asarray(t, np.float32), m.n @ R.T, m.uv, m.f)
    out.lp = getattr(m, "lp", m.v)
    return out


def rot(rx=0.0, ry=0.0, rz=0.0):
    """R = Rz · Rx · Ry (degrees)."""
    x, y, z = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]], np.float32)
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]], np.float32)
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]], np.float32)
    return Rz @ Rx @ Ry


# ============================================================================================ lighting rig
def _n3(*a):
    v = np.array(a, np.float32)
    return v / np.linalg.norm(v)


LIGHTS = [  # direction TO the light (camera space: x right, y up, z toward viewer), colour, intensity
    (_n3(-0.55, 0.62, 0.56), np.array([1.00, 0.96, 0.90], np.float32), 2.4),   # key softbox (casts shadows)
    (_n3(0.80, 0.10, 0.55), np.array([0.82, 0.88, 1.00], np.float32), 0.75),   # fill
    (_n3(0.35, 0.45, -0.82), np.array([1.00, 1.00, 1.00], np.float32), 1.9),   # rim / back light
]
SKY = np.array([0.24, 0.26, 0.30], np.float32)
GROUND = np.array([0.05, 0.045, 0.04], np.float32)


def env(R, rough):
    """Studio environment radiance for reflection vector R (n,3), roughness (n,)."""
    y = R[:, 1:2]
    base = GROUND + (SKY - GROUND) * smooth(-0.35, 0.6, y)
    out = base * 1.7
    r = rough[:, None]
    for d, c, I, size in ((LIGHTS[0][0], LIGHTS[0][1], 9.0, 0.30), (LIGHTS[1][0], LIGHTS[1][1], 3.2, 0.22),
                          (LIGHTS[2][0], LIGHTS[2][1], 6.0, 0.20)):
        w = size + r * 0.9
        g = np.exp((R @ d)[:, None] - 1.0) ** (1.0 / (w * w))
        out = out + c * I * g * ((size + 0.05) / (w + 0.05)) ** 2
    # vertical strip softbox on the right (long reflection on cylinders)
    strip = np.exp(-((R[:, 0:1] - 0.72) ** 2) / (2 * (0.07 + r * 0.4) ** 2)) * smooth(-0.6, 0.0, R[:, 1:2]) * \
        (1 - smooth(0.6, 0.95, R[:, 1:2])) * smooth(-0.2, 0.3, R[:, 2:3])
    out = out + strip * 2.2 * (0.12 / (0.12 + r * 0.6))
    return out


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


# ============================================================================================ renderer
class Model:
    """parts = [(Mesh, material_name)];  materials = {name: fn(ctx) -> dict}"""

    def __init__(self, parts, materials, name="model"):
        self.parts = parts
        self.materials = materials
        self.name = name
        V = np.concatenate([p[0].v for p in parts])
        self.center = (V.max(0) + V.min(0)) * 0.5
        self.radius = float(np.linalg.norm(V - self.center, axis=1).max())
        self.mat_names = sorted(set(m for _, m in parts))
        # concatenated static buffers
        vs, ns, uvs, lps, fs, ids = [], [], [], [], [], []
        off = 0
        for m, mat in parts:
            vs.append(m.v - self.center)
            ns.append(m.n)
            uvs.append(m.uv)
            lps.append(getattr(m, "lp", m.v))
            fs.append(m.f + off)
            ids.append(np.full(len(m.f), self.mat_names.index(mat) + 1, np.int32))
            off += len(m.v)
        self.V = np.concatenate(vs).astype(np.float32)
        self.N = np.concatenate(ns).astype(np.float32)
        self.UV = np.concatenate(uvs).astype(np.float32)
        self.LP = np.concatenate(lps).astype(np.float32)
        self.F = np.concatenate(fs).astype(np.int32)
        self.FID = np.concatenate(ids).astype(np.int32)


def render(model, size, yaw=0.0, pitch=15.0, roll=0.0, ssaa=2, shadows=True, ao=True, exposure=0.82,
           persp=4.6):
    """→ (rgba float32 D×D×4 straight alpha 0..1, D).  `size` = on-screen diameter of the bounding sphere."""
    R = rot(pitch, yaw, roll)
    rad = model.radius
    D = max(16, int(round(size * 1.32)))
    Dh = D * ssaa
    dist = persp * rad
    Fpx = (size * ssaa * 0.5 * 0.96) * dist / rad
    cx = cy = Dh * 0.5
    P = model.V @ R.T
    N = model.N @ R.T
    zc = dist - P[:, 2]
    sx = cx + Fpx * P[:, 0] / zc
    sy = cy - Fpx * P[:, 1] / zc
    F = model.F
    # cull back faces (smooth normals vs view direction at the face centre)
    cen = (P[F[:, 0]] + P[F[:, 1]] + P[F[:, 2]]) / 3.0
    nav = N[F[:, 0]] + N[F[:, 1]] + N[F[:, 2]]
    vd = np.array([0, 0, dist], np.float32) - cen
    keep = np.sum(nav * vd, 1) > -0.35 * np.linalg.norm(nav, axis=1) * np.linalg.norm(vd, axis=1)
    Fk = F[keep]
    attr = np.concatenate([N, model.UV, model.LP, P], 1)          # 3 + 2 + 3 + 3 = 11
    xy = np.stack([sx, sy], 1)[Fk]
    zz = zc[Fk]
    ab, ids, zb = zraster.raster_attr(xy, zz, attr[Fk], model.FID[keep], Dh, Dh)
    mask = ids > 0
    out = np.zeros((Dh, Dh, 4), np.float32)
    if not mask.any():
        return _down(out, ssaa), D
    pix = ab[mask]
    n = _norm(pix[:, 0:3])
    uv = pix[:, 3:5]
    lp = pix[:, 5:8]
    pc = pix[:, 8:11]
    mid = ids[mask]
    npx = len(mid)
    alb = np.zeros((npx, 3), np.float32)
    rough = np.full(npx, 0.5, np.float32)
    metal = np.zeros(npx, np.float32)
    height = np.zeros(npx, np.float32)
    wrap = np.zeros(npx, np.float32)
    emis = np.zeros((npx, 3), np.float32)
    for k, name in enumerate(model.mat_names):
        sel = mid == (k + 1)
        if not sel.any():
            continue
        r = model.materials[name](uv[sel], lp[sel], n[sel])
        alb[sel] = r["albedo"]
        if "rough" in r:
            rough[sel] = r["rough"]
        if "metal" in r:
            metal[sel] = r["metal"]
        if "height" in r:
            height[sel] = r["height"]
        if "wrap" in r:
            wrap[sel] = r["wrap"]
        if "emissive" in r:
            emis[sel] = r["emissive"]
    # ---------- screen-space bump (surface gradient of the height field, masked at part boundaries)
    if np.any(height != 0):
        Hm = np.zeros((Dh, Dh), np.float32)
        Hm[mask] = height
        gx = np.zeros_like(Hm)
        gy = np.zeros_like(Hm)
        gx[:, 1:-1] = (Hm[:, 2:] - Hm[:, :-2]) * 0.5
        gy[1:-1, :] = (Hm[2:, :] - Hm[:-2, :]) * 0.5
        same_x = np.zeros_like(mask)
        same_y = np.zeros_like(mask)
        same_x[:, 1:-1] = (ids[:, 2:] == ids[:, 1:-1]) & (ids[:, :-2] == ids[:, 1:-1])
        same_y[1:-1, :] = (ids[2:, :] == ids[1:-1, :]) & (ids[:-2, :] == ids[1:-1, :])
        gx[~same_x] = 0
        gy[~same_y] = 0
        pw = zb[mask] / Fpx                                        # world size of one pixel
        k = np.maximum(np.abs(n[:, 2]), 0.25)
        sxl = np.clip(gx[mask] / pw * k, -3, 3)
        syl = np.clip(gy[mask] / pw * k, -3, 3)
        n = _norm(n + np.stack([-sxl, syl, np.zeros_like(sxl)], 1))
    # ---------- shadow map for the key light
    sh = np.ones(npx, np.float32)
    if shadows:
        sh = _shadow(model, P, pc, n, rad)
    # ---------- SSAO
    aov = np.ones(npx, np.float32)
    if ao:
        import cv2
        z = np.where(mask, zb, 0).astype(np.float32)
        m = mask.astype(np.float32)
        r = max(3, int(Dh * 0.025)) | 1
        zbl = cv2.GaussianBlur(z, (0, 0), r * 0.5) / np.maximum(cv2.GaussianBlur(m, (0, 0), r * 0.5), 1e-4)
        occ = np.clip((zb - zbl)[mask] / (0.05 * rad), 0, 1)
        aov = 1.0 - 0.55 * occ
    # ---------- shading
    V = _norm(np.array([0, 0, dist], np.float32) - pc)
    ndv = np.clip(np.sum(n * V, 1), 1e-4, 1)
    F0 = 0.04 * (1 - metal[:, None]) + alb * metal[:, None]
    diff_alb = alb * (1 - metal[:, None])
    hemi = GROUND + (SKY - GROUND) * (0.5 + 0.5 * n[:, 1:2])
    col = diff_alb * hemi * aov[:, None] * 1.1
    pexp = np.clip(2.0 / np.maximum(rough, 0.03) ** 4 - 2.0, 2.0, 6000.0)
    for li, (Ld, Lc, LI) in enumerate(LIGHTS):
        ndl_raw = n @ Ld
        w = wrap[:, None]
        ndl = np.clip((ndl_raw[:, None] + w) / (1 + w), 0, 1)
        s = sh[:, None] if li == 0 else aov[:, None] ** 0.5
        col = col + diff_alb * Lc * LI * ndl * s
        Hh = _norm(Ld[None] + V)
        ndh = np.clip(np.sum(n * Hh, 1), 0, 1)
        vdh = np.clip(np.sum(V * Hh, 1), 0, 1)
        Fr = F0 + (1 - F0) * ((1 - vdh) ** 5)[:, None]
        spec = ((pexp + 8) / (8 * math.pi) * ndh ** pexp)[:, None] * Fr * np.clip(ndl_raw, 0, 1)[:, None]
        col = col + spec * Lc * LI * s * 0.9
    Rv = _norm(2 * ndv[:, None] * n - V)
    Fe = F0 + (np.maximum(1 - rough, F0.mean(1))[:, None] - F0) * ((1 - ndv) ** 5)[:, None]
    col = col + env(Rv, rough) * Fe * aov[:, None] * (0.35 + 0.65 * sh[:, None]) * 0.55
    col = col + emis
    col = aces(col * exposure) ** (1 / 2.2)
    out[mask, :3] = col
    out[mask, 3] = 1.0
    return _down(out, ssaa), D


def _down(out, ssaa):
    if ssaa <= 1:
        return out
    import cv2
    pm = out.copy()
    pm[:, :, :3] *= pm[:, :, 3:4]
    h, w = pm.shape[:2]
    pm = cv2.resize(pm, (w // ssaa, h // ssaa), interpolation=cv2.INTER_AREA)
    a = pm[:, :, 3:4]
    pm[:, :, :3] = pm[:, :, :3] / np.maximum(a, 1e-5)
    return pm


def _shadow(model, P, pc, n, rad, res=1024):
    L = LIGHTS[0][0]
    up = np.array([0, 1, 0], np.float32) if abs(L[1]) < 0.9 else np.array([1, 0, 0], np.float32)
    lx = _norm(np.cross(up, L))
    ly = np.cross(L, lx)
    ext = rad * 1.08
    C = 200.0 * rad

    def proj(Q):
        return ((Q @ lx) / ext * 0.5 + 0.5) * res, ((Q @ ly) / ext * -0.5 + 0.5) * res, C - (Q @ L)

    u, v, d = proj(P)
    F = model.F
    xy = np.stack([u, v], 1)[F]
    _, ids, zb = zraster.raster_attr(xy, d[F], np.zeros((len(F), 3, 1), np.float32), np.ones(len(F), np.int32),
                                     res, res)
    pu, pv, pd = proj(pc)
    ndl = np.clip(n @ L, 0, 1)
    bias = rad * (0.010 + 0.035 * (1 - ndl))
    lit = np.zeros(len(pc), np.float32)
    cnt = 0
    for oy in (-1.5, 0, 1.5):
        for ox in (-1.5, 0, 1.5):
            iu = np.clip((pu + ox).astype(np.int32), 0, res - 1)
            iv = np.clip((pv + oy).astype(np.int32), 0, res - 1)
            lit += (pd - bias <= zb[iv, iu]).astype(np.float32)
            cnt += 1
    s = lit / cnt
    return np.where(ndl > 0, s, 1.0).astype(np.float32) * smooth(0.0, 0.15, ndl) + (1 - smooth(0.0, 0.15, ndl))


def to_bgra_premul(rgba):
    """straight RGBA float 0..1 → premultiplied BGRA float 0..255 (for obj3d.paint_premul / cairo)."""
    out = np.empty_like(rgba)
    a = rgba[:, :, 3:4]
    out[:, :, :3] = rgba[:, :, 2::-1] * a * 255.0
    out[:, :, 3:4] = a * 255.0
    return out


# ============================================================================================ element / paint
def paint(ctx, model_name, cx, cy, size, yaw=0.0, pitch=15.0, roll=0.0, alpha=1.0, shadow=True, ssaa=2):
    from .obj3d import contact_shadow, paint_premul
    from .real3d_models import get_model
    m = get_model(model_name) if isinstance(model_name, str) else model_name
    rgba, D = render(m, size, yaw, pitch, roll, ssaa=ssaa)
    if shadow:
        contact_shadow(ctx, cx, cy + size * 0.47, size * 0.85, alpha, strength=0.5)
    paint_premul(ctx, to_bgra_premul(rgba), cx - D * 0.5, cy - D * 0.5, alpha)
    return (cx - size * 0.5, cy - size * 0.5, cx + size * 0.5, cy + size * 0.5)


def _register():
    from .elements import element, val, xy

    @element("real3d", dur=0.8, enter="spring", sfx="whoosh")
    def el_real3d(ctx, el, s):
        """TRUE 3D product model:  {"type": "real3d", "model": "soda_can" | "pizza_slice" | "growth_card",
        "w": 420, "motion": "turntable_osc" | "spin_360" | "hover" | "spring_entry", "yaw", "pitch"}"""
        from .obj3d import motion_angles
        x, y = xy(el, s.t)
        sz = float(val(el, "w", s.t, el.get("size", 360))) * s.scale
        lt = max(0.0, s.t - float(el["t"]))
        yaw, pitch = motion_angles(el, lt, float(val(el, "yaw", s.t, 0.0)), float(val(el, "pitch", s.t, 15.0)))
        return paint(ctx, el.get("model", "soda_can"), x, y + s.dy, sz, yaw, pitch, float(val(el, "roll", s.t, 0.0)),
                     s.alpha, bool(el.get("shadow", True)), int(el.get("ssaa", 2)))


try:
    _register()
except Exception:  # pragma: no cover (elements registry unavailable in stand-alone use)
    pass
