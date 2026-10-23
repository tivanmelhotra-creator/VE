"""
mograph.human.garments — procedural clothing + hair as SKINNED SHELLS of the shaped body.

Each garment = scalar field f on the REST body (inside where f < 0) → exact iso-cut of the body mesh (smooth hems,
necklines, cuffs — never jagged) → offset along smoothed normals (thickness + looseness) → Taubin-smoothed (cloth
hides ribs, abs, nipples, kneecaps) → skinned with interpolated body weights, so it follows every pose.

Outfit spec (all keys optional):
    {"top": "tshirt|shirt|longsleeve|tank|sweater|jacket|coat|hoodie|none",
     "open": True (jacket/coat open at the front), "sleeve": 0..1 (override sleeve length as a fraction of the arm),
     "tuck": True (top tucked in → hem at the waist),
     "bottom": "trousers|jeans|shorts|skirt|pencil_skirt|none", "length": 0..1 (bottom length override),
     "dress": "none|short|knee|midi|long" (dress replaces top + bottom; sleeve still applies),
     "shoes": "shoes|sneakers|boots|heels|none", "hair": "short|buzz|side|bob|long|bun|ponytail|bald|none",
     "beard": False}
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .body import BODY_N, Body, bone_index, bones, group_faces, group_vertices
from .regions import regions, smoothstep, body_tris


# ------------------------------------------------------------------------------------------ mesh cutting
def cut_mesh(X, tris, f, W):
    """Keep the part of the triangle mesh where f < 0, splitting triangles exactly at f = 0.
    Returns verts (M,3), tris (T,3), W (M,B), src (M,2) source vertex pair, t (M,) interpolation, is_cut (M,)."""
    f = np.asarray(f, np.float64)
    inside = f < 0
    ti = inside[tris]
    full = tris[ti.all(1)]
    mixed = tris[ti.any(1) & ~ti.all(1)]
    used = np.unique(np.r_[full.ravel(), mixed[inside[mixed]].ravel()]) if len(mixed) else np.unique(full.ravel())
    vmap = -np.ones(len(X), np.int64)
    vmap[used] = np.arange(len(used))
    V = [X[used]]
    WW = [W[used]]
    src = [np.c_[used, used]]
    tt = [np.zeros(len(used))]
    cut = [np.zeros(len(used), bool)]
    n = len(used)
    ekey = {}
    newv, neww, nsrc, nt = [], [], [], []
    out = [vmap[full]]

    def edge_pt(a, b):
        nonlocal n
        k = (a, b) if a < b else (b, a)
        if k in ekey:
            return ekey[k]
        fa, fb = f[k[0]], f[k[1]]
        t = fa / (fa - fb)
        newv.append(X[k[0]] * (1 - t) + X[k[1]] * t)
        neww.append(W[k[0]] * (1 - t) + W[k[1]] * t)
        nsrc.append(k)
        nt.append(t)
        ekey[k] = n
        n += 1
        return ekey[k]
    tri_new = []
    for tri in mixed:
        poly = []
        for j in range(3):
            a, b = tri[j], tri[(j + 1) % 3]
            if inside[a]:
                poly.append(vmap[a])
            if inside[a] != inside[b]:
                poly.append(edge_pt(a, b))
        for j in range(1, len(poly) - 1):
            tri_new.append((poly[0], poly[j], poly[j + 1]))
    if newv:
        V.append(np.asarray(newv))
        WW.append(np.asarray(neww))
        src.append(np.asarray(nsrc))
        tt.append(np.asarray(nt))
        cut.append(np.ones(len(newv), bool))
    if tri_new:
        out.append(np.asarray(tri_new, np.int64))
    return (np.vstack(V).astype(np.float32), np.vstack(out).astype(np.int32), np.vstack(WW).astype(np.float32),
            np.vstack(src), np.concatenate(tt), np.concatenate(cut))


def interp_attr(attr, src, t):
    a = attr[src[:, 0]]
    b = attr[src[:, 1]]
    tt = t.reshape((-1,) + (1,) * (a.ndim - 1))
    return a * (1 - tt) + b * tt


# ------------------------------------------------------------------------------------------ garment
@dataclass
class Garment:
    name: str
    kind: str                       # cloth | hair | shoe
    rest: np.ndarray                # (M,3) rest verts (offset)
    tris: np.ndarray
    W: np.ndarray                   # (M,B)
    topo: object = None
    smooth_iters: int = 3
    weight: float = 1.0
    hem: bool = True                # draw open boundaries
    min_jump: float = 0.25
    folds: list = field(default_factory=list)   # fold specs, see folds.py
    cover: np.ndarray | None = None # (BODY verts) field: >0 where this garment hides the body
    attrs: dict = field(default_factory=dict)   # per-vertex attributes (arm_s, leg_s, y, x, front…)
    details: list = field(default_factory=list) # iso-line details: (attr, level, mask_attr, mask_min, weight, kind)


def _vnormals(X, tris, n):
    fn = np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]])
    vn = np.zeros((n, 3))
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def _smooth_normals(body):
    if getattr(body, "_snorm", None) is None:
        R = regions(body)
        Xs = R.topo.smooth(body.rest, 10)
        body._snorm = _vnormals(Xs, R.tris, len(body.rest))
    return body._snorm


def make_shell(body: Body, f, thick, name, kind="cloth", smooth=6, loose=None, weight=1.0, min_jump=0.25,
               cover_margin=0.02, attrs_extra=None, hull=0) -> Garment:
    """thick: scalar or per-body-vertex array (dm). loose: extra smoothing iterations of the rest shell."""
    from .lineart import MeshTopo
    R = regions(body)
    X = body.rest
    Nrm = _smooth_normals(body)
    th = np.broadcast_to(np.asarray(thick, np.float64), (len(X),)).astype(np.float64)
    V, T, Wg, src, t, iscut = cut_mesh(X, R.tris, f, body.W)
    nrm = interp_attr(Nrm, src, t)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    thv = interp_attr(th, src, t)
    V = V + nrm * thv[:, None]
    topo = MeshTopo(T, len(V))
    if hull:
        # plain Laplacian (shrinking) smoothing → bumps (toes, knuckles) merge into one volume; the re-inflate
        # step below pushes the hull back outside the skin
        Y = V.astype(np.float64)
        for _ in range(int(hull)):
            Y = Y + 0.6 * (topo.L @ Y - Y)
        V = Y.astype(np.float32)
    if loose:
        V = topo.smooth(V.astype(np.float64), int(loose)).astype(np.float32)
        # re-inflate where smoothing pulled the cloth inside the skin
        Vb = interp_attr(X, src, t)
        d = np.einsum("ij,ij->i", V - Vb, nrm)
        V = V + nrm * np.maximum(0, thv * 0.8 - d)[:, None]
    attrs = {"y": interp_attr(R.y, src, t), "x": interp_attr(X[:, 0], src, t), "z": interp_attr(X[:, 2], src, t)}
    for s in ("L", "R"):
        attrs[f"arm_s.{s}"] = interp_attr(R.arm_s[s], src, t)
        attrs[f"leg_s.{s}"] = interp_attr(R.leg_s[s], src, t)
    for k, v in (attrs_extra or {}).items():
        attrs[k] = interp_attr(v, src, t)
    g = Garment(name, kind, V.astype(np.float32), T, Wg, topo, smooth, weight, True, min_jump, attrs=attrs)
    g.cover = -np.asarray(f) - cover_margin
    return g


def mesh_garment(body, verts, tris, W, name, kind, smooth=2, weight=1.0, min_jump=0.25):
    from .lineart import MeshTopo
    topo = MeshTopo(tris, len(verts))
    return Garment(name, kind, verts.astype(np.float32), tris.astype(np.int32), W.astype(np.float32), topo,
                   smooth, weight, True, min_jump, attrs={"y": verts[:, 1], "x": verts[:, 0], "z": verts[:, 2]})


def _bone_w(nb, names_w):
    w = np.zeros(nb, np.float32)
    for n, v in names_w.items():
        w[bone_index(n)] = v
    return w / w.sum()


# ------------------------------------------------------------------------------------------ fields
def _sm_max(a, b, k=0.08):
    """smooth maximum (rounded corners where two cuts meet)."""
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0, 1)
    return a * h + b * (1 - h) + k * h * (1 - h)


def field_top(body, hem_y, sleeve, collar="crew", vdepth=0.0, open_w=0.0):
    """inside < 0. sleeve = length in dm along the arm (0 = sleeveless at the shoulder)."""
    R = regions(body)
    X = body.rest
    y = R.y
    L = R.lm
    front = smoothstep(-0.2, 0.8, X[:, 2] - L["head_c"][2])
    neck_y = L["neck"] - 0.25
    if collar == "high":
        neck_y = L["neck"] + 0.55
    elif collar == "turtle":
        neck_y = L["chin"] - 0.15
    cy = neck_y - (0.25 + vdepth) * front - 0.1 * (1 - front)
    if collar == "scoop":
        cy = neck_y - 0.9 * front - 0.25
    f_torso = _sm_max(hem_y - y, y - cy)
    out = f_torso
    for s in ("L", "R"):
        a = smoothstep(0.35, 0.75, R.armness[s])
        if sleeve <= 0.01:
            # sleeveless: cut the armhole as a loop around the shoulder joint
            sh = body.head(f"upperarm01.{s}")
            d = np.linalg.norm(X - sh, axis=1)
            f_arm = 0.95 - d + 0.0
            out = np.where(R.armness[s] > 0.02, np.maximum(out, np.minimum(f_arm * 2, 1) * smoothstep(0.02, 0.4,
                                                                                                       R.armness[s])
                                                           + out * (1 - smoothstep(0.02, 0.4, R.armness[s]))), out)
        else:
            f_arm = R.arm_s[s] - sleeve
            out = out * (1 - a) + np.maximum(f_arm, np.minimum(out, 0) * 0 - 0.0) * a if False else \
                out * (1 - a) + f_arm * a
    # never cover the head or the legs
    out = np.maximum(out, (R.head - 0.35) * 3)
    leg = np.maximum(R.legness["L"], R.legness["R"])
    out = np.where(y < L["crotch"] + 0.3, np.maximum(out, (leg - 0.6) * 3), out)
    if open_w > 0:
        # open front (jacket/coat): a wedge |x| < w(y) on the front, widening toward the neck (lapels)
        t = np.clip((y - (hem_y)) / max(0.1, (cy.max() - hem_y)), 0, 1)
        w = open_w * (0.35 + 0.65 * t ** 1.6)
        wedge = (w - np.abs(X[:, 0])) * smoothstep(0.1, 0.5, front) * (y > hem_y - 0.2)
        wedge = np.where(R.armness["L"] + R.armness["R"] > 0.3, -1.0, wedge)
        out = np.maximum(out, wedge * 2.5)
    return out


def field_bottom(body, waist_y, length, crotch_drop=0.0):
    R = regions(body)
    y = R.y
    out = y - waist_y
    for s in ("L", "R"):
        a = smoothstep(0.3, 0.7, R.legness[s])
        f_leg = R.leg_s[s] - length
        out = out * (1 - a) + np.maximum(f_leg, out) * a
    arm = np.maximum(R.armness["L"], R.armness["R"])
    out = np.maximum(out, (arm - 0.25) * 4)
    out = np.maximum(out, (R.head - 0.3) * 4)
    return out


def field_shoe(body, collar_s):
    R = regions(body)
    out = np.ones(len(body.rest))
    for s in ("L", "R"):
        a = smoothstep(0.3, 0.7, R.legness[s])
        out = np.minimum(out, np.where(a > 0.05, (collar_s - R.leg_s[s]) * a + (1 - a), 1.0))
    return out


def head_angles(body):
    R = regions(body)
    X = body.rest
    c = np.array([0.0, R.lm["eye"], R.lm["head_c"][2] - 0.1])
    th = np.degrees(np.arctan2(X[:, 0], X[:, 2] - c[2]))
    return np.abs(th), c


def field_hair(body, style="short"):
    R = regions(body)
    L = R.lm
    th, c = head_angles(body)
    eye, brow, nk = L["eye"], L["brow"], L["neck"]
    if style == "buzz":
        knots = [(0, brow + 0.55), (40, brow + 0.5), (70, eye + 0.4), (82, eye + 0.05), (92, eye + 0.5),
                 (125, eye + 0.45), (150, nk + 0.75), (180, nk + 0.7)]
    elif style == "side":
        knots = [(0, brow + 0.35), (40, brow + 0.38), (70, eye + 0.32), (82, eye - 0.25), (92, eye + 0.42),
                 (125, eye + 0.4), (150, nk + 0.5), (180, nk + 0.4)]
    else:
        knots = [(0, brow + 0.42), (40, brow + 0.42), (70, eye + 0.33), (82, eye - 0.28), (92, eye + 0.42),
                 (125, eye + 0.4), (150, nk + 0.55), (180, nk + 0.45)]
    a, h = zip(*knots)
    hy = np.interp(th, a, h)
    region = np.clip(R.head + R.neck, 0, 1)
    f = hy - R.y
    f = np.where(region > 0.3, f, 1.0)
    f = np.maximum(f, -(region - 0.3))
    return f


# ------------------------------------------------------------------------------------------ hair extras
def _sphere(c, r, n=14):
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    P = [np.array([0, 1.0, 0]) * r + c]
    for ui in u[1:-1]:
        for vi in v:
            P.append(c + r * np.array([math.sin(ui) * math.cos(vi), math.cos(ui), math.sin(ui) * math.sin(vi)]))
    P.append(c - np.array([0, 1.0, 0]) * r)
    P = np.asarray(P)
    m = 2 * n
    T = []
    for j in range(m):
        T.append((0, 1 + (j + 1) % m, 1 + j))
    for i in range(n - 3):
        for j in range(m):
            a = 1 + i * m + j
            b = 1 + i * m + (j + 1) % m
            T += [(a, b, b + m), (a, b + m, a + m)]
    last = len(P) - 1
    base = 1 + (n - 3) * m
    for j in range(m):
        T.append((last, base + j, base + (j + 1) % m))
    return P, np.asarray(T)


def _tube(path, radii, n=12):
    path = np.asarray(path)
    P, T = [], []
    for i, (p, r) in enumerate(zip(path, radii)):
        d = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        d /= np.linalg.norm(d) + 1e-9
        a = np.cross(d, [1.0, 0, 0])
        if np.linalg.norm(a) < 1e-3:
            a = np.cross(d, [0, 0, 1.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        for k in range(n):
            t = 2 * math.pi * k / n
            P.append(p + r * (math.cos(t) * a + math.sin(t) * b * 0.8))
    for i in range(len(path) - 1):
        for k in range(n):
            a = i * n + k
            b = i * n + (k + 1) % n
            T += [(a, b, b + n), (a, b + n, a + n)]
    # cap the tip
    tip = len(P)
    P.append(path[-1] + (path[-1] - path[-2]) * 0.3)
    base = (len(path) - 1) * n
    for k in range(n):
        T.append((base + k, base + (k + 1) % n, tip))
    return np.asarray(P), np.asarray(T)


# ------------------------------------------------------------------------------------------ outfit
TOPS = {
    #           hem (fraction waist→crotch, >1 below), sleeve (fraction of arm), thick, loose, collar, open
    "tshirt":     dict(hem=0.75, sleeve=0.36, thick=0.07, loose=6, collar="crew"),
    "shirt":      dict(hem=0.85, sleeve=1.0, thick=0.08, loose=6, collar="high", placket=True),
    "longsleeve": dict(hem=0.8, sleeve=1.0, thick=0.07, loose=6, collar="crew"),
    "tank":       dict(hem=0.75, sleeve=0.0, thick=0.06, loose=5, collar="scoop"),
    "sweater":    dict(hem=0.85, sleeve=1.0, thick=0.14, loose=10, collar="crew"),
    "hoodie":     dict(hem=0.95, sleeve=1.0, thick=0.18, loose=12, collar="high"),
    "jacket":     dict(hem=1.15, sleeve=1.0, thick=0.2, loose=12, collar="high", open=0.55),
    "coat":       dict(hem=2.6, sleeve=1.0, thick=0.24, loose=14, collar="high", open=0.5),
    "blouse":     dict(hem=0.8, sleeve=0.5, thick=0.07, loose=6, collar="crew", vdepth=0.35),
}
BOTTOMS = {
    "trousers": dict(length=1.0, thick=0.1, loose=6),
    "jeans":    dict(length=1.0, thick=0.09, loose=4, fly=True),
    "shorts":   dict(length=0.42, thick=0.1, loose=6),
    "leggings": dict(length=1.0, thick=0.03, loose=1),
}
SKIRTS = {"short": 0.45, "knee": 0.62, "midi": 0.82, "long": 1.02}


def build_outfit(body: Body, spec: dict | None) -> list[Garment]:
    spec = dict(spec or {})
    R = regions(body)
    L = R.lm
    out: list[Garment] = []
    waist, crotch = L["waist"], L["crotch"]
    hip_band = waist - crotch
    dress = spec.get("dress")
    top = spec.get("top", "tshirt" if not dress else None)
    bottom = spec.get("bottom", "trousers" if not dress else None)
    tuck = bool(spec.get("tuck"))
    b_thick = BOTTOMS.get(bottom, {}).get("thick", 0.1) if bottom and bottom != "none" else 0.0
    t_spec = TOPS.get(top, {}) if top else {}
    # ---------------------------------------------------------------- bottom
    if bottom and bottom != "none":
        if bottom in BOTTOMS:
            b = BOTTOMS[bottom]
            length = spec.get("length", b["length"]) * (R.leg_len - 0.25)
            wy = waist - 0.15 if bottom != "jeans" else waist - 0.45
            f = field_bottom(body, wy, length)
            th = b["thick"] + (t_spec.get("thick", 0.07) + 0.03 if tuck else 0.0)
            g = make_shell(body, f, th, bottom, smooth=4, loose=b["loose"])
            g.folds = ["knee_back", "crotch"]
            if b.get("fly"):
                g.details.append(("x", 0.0, "fly", 0.0, 0.45, "seam"))
            out.append(g)
        elif bottom in ("skirt", "pencil_skirt"):
            frac = spec.get("length", 0.6)
            out.append(_skirt(body, waist - 0.2, frac, flare=(bottom == "skirt")))
    if dress and dress != "none":
        frac = SKIRTS.get(dress, 0.62) if isinstance(dress, str) else float(dress)
        out.append(_skirt(body, waist + 0.1, frac, flare=True, name="dress_skirt"))
        top = top or "dress_top"
    # ---------------------------------------------------------------- top
    if top and top != "none":
        if top == "dress_top":
            tuck = True
            t = dict(hem=0.15, sleeve=spec.get("sleeve", 0.3), thick=0.06, loose=5, collar="scoop")
        else:
            t = dict(TOPS.get(top, TOPS["tshirt"]))
        if "sleeve" in spec:
            t["sleeve"] = spec["sleeve"]
        hem_frac = 0.12 if spec.get("tuck") else t["hem"]
        hem_y = waist - hem_frac * hip_band
        sleeve = t["sleeve"] * (R.arm_len - 0.05)
        open_w = t.get("open", 0.0) if spec.get("open", True) else 0.0
        f = field_top(body, hem_y, sleeve, t.get("collar", "crew"), t.get("vdepth", 0.0), open_w)
        # layer above the bottom (or the skirt / dress) unless tucked
        under = 0.0 if tuck else (b_thick + 0.04 if b_thick else 0.0)
        if dress or (bottom in ("skirt", "pencil_skirt")):
            under = 0.0 if tuck else 0.12
        hipw = smoothstep(waist + 0.6, waist - 0.2, regions(body).y)
        thick = t["thick"] + under * hipw
        g = make_shell(body, f, thick, top, smooth=4, loose=t["loose"])
        g.folds = ["elbow_in", "armpit", "waist_side"]
        if t.get("placket"):
            g.details.append(("x", 0.0, "front", 0.4, 0.4, "seam"))
        out.append(g)
        if open_w > 0:
            # shirt underneath shows through the open front
            f2 = field_top(body, waist - 0.6 * hip_band, 0.0, "high", 0.0, 0.0)
            g2 = make_shell(body, np.maximum(f2, np.abs(body.rest[:, 0]) - open_w - 0.6), 0.07, "undershirt",
                            smooth=4, loose=5)
            g2.weight = 0.9
            out.append(g2)
    # ---------------------------------------------------------------- shoes
    shoes = spec.get("shoes", "shoes")
    if shoes and shoes != "none":
        for sd in ("L", "R"):
            out.append(build_shoe(body, sd, shoes))
    # ---------------------------------------------------------------- hair
    hair = spec.get("hair", "short" if body.params.get("gender", 0.5) >= 0.5 else "long")
    if hair and hair not in ("none", "bald"):
        out += build_hair(body, hair)
    return out


def _skirt(body, top_y, frac, flare=True, name="skirt"):
    """A-line skirt: helper-skirt tube from the base mesh, cut to length, flared, weights blended to the pelvis."""
    from .lineart import MeshTopo
    R = regions(body)
    idx = group_vertices("helper-skirt")
    Q = group_faces("helper-skirt")
    T = np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    X = body.rest
    hem_y = R.lm["hip"] - frac * (R.lm["hip"] - R.lm["ankle"]) * 1.0
    f = np.full(len(X), 1.0)
    f[idx] = np.maximum(hem_y - X[idx, 1], X[idx, 1] - top_y)
    # skirt helper may not reach the waist: extend with the body torso band
    V, Tt, Wg, src, t, _ = cut_mesh(X, T, f, body.W)
    y = V[:, 1]
    c = np.array([0.0, 0.0, X[idx, 2].mean()])
    if flare:
        k = np.clip((R.lm["hip"] - y) / max(0.3, R.lm["hip"] - hem_y), 0, 1)
        r = V[:, [0, 2]] - c[[0, 2]]
        V[:, 0] += r[:, 0] * 0.32 * k ** 1.2
        V[:, 2] += r[:, 1] * 0.22 * k ** 1.2
    # weights: blend legs → pelvis so walking does not shred the skirt
    pel = np.zeros_like(Wg[0])
    pel[bone_index("spine05")] = 0.6
    pel[bone_index("root")] = 0.4
    k = np.clip((R.lm["hip"] - y) / 3.0, 0, 1)[:, None]
    leg_part = Wg.copy()
    Wn = Wg * (1 - 0.55 * (1 - k)) * 0 + (Wg * 0.45 + pel[None] * 0.55)
    Wn = Wn / Wn.sum(1, keepdims=True)
    g = mesh_garment(body, V, Tt, Wn, name, "cloth", smooth=3, weight=1.0, min_jump=0.3)
    # body coverage: torso band + legs above the hem
    cover = np.where((R.y < top_y) & (R.y > hem_y) & (np.maximum(R.armness["L"], R.armness["R"]) < 0.3), 1.0, -1.0)
    g.cover = cover
    g.folds = ["skirt_drape"]
    # waistband: also cover the gap between the skirt top and the body with a thin band shell
    return g


def build_hair(body, style) -> list[Garment]:
    R = regions(body)
    L = R.lm
    out = []
    th, c = head_angles(body)
    base = "buzz" if style == "buzz" else ("side" if style == "side" else "short")
    f = field_hair(body, base)
    vol = {"buzz": 0.1, "short": 0.34, "side": 0.38, "bob": 0.3, "long": 0.28, "bun": 0.2, "ponytail": 0.18}[style]
    # more volume on the crown, less at the hairline
    hy = R.y - L["eye"]
    thick = vol * (0.45 + 0.55 * smoothstep(0.0, 0.9, hy))
    if style == "side":
        thick = thick * (1 + 0.35 * smoothstep(-0.2, 0.6, body.rest[:, 0]) * smoothstep(0.4, 1.0, hy))
    g = make_shell(body, f, thick, f"hair_{style}", kind="hair", smooth=8, loose=10, min_jump=0.12)
    g.weight = 1.0
    out.append(g)
    nb = len(bones())
    headW = _bone_w(nb, {"head": 1.0})
    if style in ("long", "bob"):
        from .lineart import MeshTopo
        idx = group_vertices("helper-hair")
        Q = group_faces("helper-hair")
        T = np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
        X = body.rest
        cut_y = L["chin"] - 0.1 if style == "bob" else L["chest"] - 1.2
        ff = np.full(len(X), 1.0)
        ff[idx] = cut_y - X[idx, 1]
        # keep hair off the face: cut a front opening below the brow
        front = X[idx, 2] - (L["head_c"][2] + 0.35)
        ff[idx] = np.maximum(ff[idx], np.where((X[idx, 1] < L["brow"] + 0.2) & (np.abs(X[idx, 0]) < 0.95),
                                               front * 4, -1))
        V, Tt, Wg, src, t, _ = cut_mesh(X, T, ff, body.W)
        # head-follow for the top, blend to neck/spine as it falls
        k = np.clip((L["neck"] - V[:, 1]) / 2.5, 0, 1)[:, None]
        sp = _bone_w(nb, {"neck01": 0.5, "spine01": 0.5})
        Wn = headW[None] * (1 - k) + sp[None] * k
        if style == "bob":
            # puff the bob outward
            cc = np.array([0, L["eye"], L["head_c"][2]])
            r = V - cc
            r[:, 1] = 0
            V = V + r * 0.08
        out.append(mesh_garment(body, V, Tt, Wn, f"hair_{style}_fall", "hair", smooth=4, min_jump=0.15))
    if style == "bun":
        hc = L["head_c"]
        cc = np.array([0.0, L["top"] - 0.35, hc[2] - 0.75])
        P, T = _sphere(cc, 0.55, 12)
        out.append(mesh_garment(body, P, T, np.repeat(headW[None], len(P), 0), "hair_bun", "hair", smooth=0,
                                min_jump=0.1))
    if style == "ponytail":
        hc = L["head_c"]
        root = np.array([0.0, L["eye"] + 0.55, hc[2] - 1.05])
        path = [root + np.array([0, -0.1 * i, -0.06 * i * (1 - i / 9)]) * 1.0 for i in range(10)]
        path = [p + np.array([0, -i * 0.25, -0.12]) for i, p in enumerate(path)]
        radii = [0.32 - 0.025 * i for i in range(10)]
        P, T = _tube(path, radii, 10)
        k = np.clip((root[1] - P[:, 1]) / 2.0, 0, 1)[:, None]
        sp = _bone_w(nb, {"neck03": 0.6, "head": 0.4})
        Wn = headW[None] * (1 - k) + sp[None] * k
        out.append(mesh_garment(body, P, T, Wn, "hair_ponytail", "hair", smooth=0, min_jump=0.1))
    return out


# ------------------------------------------------------------------------------------------ shoes (lofted last)
SHOE_STYLE = {
    #            toe_len  width  toe_h  heel_h  collar_h  sole   toe_round
    "shoes":    (1.10,   1.05,  0.95,  0.05,   0.62,     0.07,  0.55),
    "sneakers": (1.12,   1.12,  1.05,  0.10,   0.70,     0.14,  0.75),
    "boots":    (1.10,   1.08,  1.0,   0.12,   2.10,     0.12,  0.7),
    "heels":    (1.12,   0.92,  0.75,  0.75,   0.55,     0.04,  0.25),
    "flats":    (1.05,   0.98,  0.82,  0.0,    0.45,     0.04,  0.45),
}


def build_shoe(body, side, style="shoes"):
    """A clean closed shoe mesh lofted along the foot axis (heel → toe), sized from the foot's own skin vertices,
    weighted to foot/toe bones. Hides every toe and gives a readable shoe silhouette + sole line."""
    from .lineart import MeshTopo
    R = regions(body)
    X = body.rest
    sg = 1 if side == "L" else -1
    fm = (R.foot > 0.45) & (np.sign(X[:, 0]) == sg)
    fm[BODY_N:] = False
    P = X[fm]
    heel_z, toe_z = P[:, 2].min(), P[:, 2].max()
    ank = body.head(f"foot.{side}")
    st = SHOE_STYLE.get(style, SHOE_STYLE["shoes"])
    toe_len, wmul, toe_h, heel_h, collar_h, sole, round_ = st
    L = (toe_z - heel_z)
    toe_z2 = heel_z + L * toe_len
    nu, nv = 40, 32
    zs0, zs1 = heel_z - 0.08, toe_z2
    prof = []
    for i in range(nu):
        u = i / (nu - 1)
        z = zs0 + (zs1 - zs0) * u
        band = P[np.abs(P[:, 2] - z) < max(0.12, L / nu * 1.5)]
        if len(band) < 4:
            band = P[np.argsort(np.abs(P[:, 2] - z))[:12]]
        cx = band[:, 0].mean()
        hw = (band[:, 0].max() - band[:, 0].min()) / 2 * wmul + 0.07
        h_ankle = collar_h + 0.45
        h = float(np.interp(u, [0, 0.15, 0.4, 0.62, 0.85, 1.0],
                            [h_ankle * 0.9, h_ankle, h_ankle * 0.82, toe_h * 0.7, toe_h * 0.58, toe_h * 0.5]))
        if u > 0.3:
            h = max(h, band[:, 1].max() + 0.09)
        lift = heel_h * float(np.interp(u, [0, 0.45, 0.7, 1], [1, 0.8, 0.0, 0.0]))
        prof.append([z, cx, hw, lift - 0.02, h])
    prof = np.asarray(prof)
    # smooth the profile along the foot (no lumps from individual toes)
    for _ in range(4):
        prof[1:-1, 1:] = 0.25 * prof[:-2, 1:] + 0.5 * prof[1:-1, 1:] + 0.25 * prof[2:, 1:]
    # dome the ends: rings shrink toward a tip point (rounded toe cap / heel counter)
    rings = []
    ncap = 7
    def ring_at(z, cx, hw, base, h, k=1.0):
        hh = (h - base) / 2
        yc = base + hh
        out = []
        for j in range(nv):
            t = 2 * math.pi * j / nv
            c, s_ = math.cos(t), math.sin(t)
            ey = 2 / 2.6 if s_ > 0 else 2 / 5.0
            out.append((cx + k * hw * np.sign(c) * abs(c) ** (2 / 3.0),
                        yc + k * hh * np.sign(s_) * abs(s_) ** ey + (1 - k) * (-0.25 * hh), z))
        return out
    p0 = prof[0]
    for m in range(ncap, 0, -1):                       # heel dome
        ang = (m / (ncap + 1)) * (math.pi / 2)
        k = math.cos(ang)
        rings.append(ring_at(p0[0] - 0.22 * math.sin(ang), p0[1], p0[2], p0[3], p0[4], max(k, 0.05)))
    for row in prof:
        rings.append(ring_at(*row))
    p1 = prof[-1]
    toe_r = 0.35 + 0.45 * round_
    for m in range(1, ncap + 1):                       # toe dome
        ang = (m / (ncap + 1)) * (math.pi / 2)
        k = math.cos(ang)
        rings.append(ring_at(p1[0] + toe_r * math.sin(ang), p1[1], p1[2], p1[3], p1[4], max(k, 0.05)))
    nu2 = len(rings)
    V = np.asarray(rings, np.float64).reshape(-1, 3)
    T = []
    for i in range(nu2 - 1):
        for j in range(nv):
            a_ = i * nv + j
            b_ = i * nv + (j + 1) % nv
            T += [(a_, b_, b_ + nv), (a_, b_ + nv, a_ + nv)]
    c0 = len(V); V = np.vstack([V, V[:nv].mean(0)])
    c1 = len(V); V = np.vstack([V, V[(nu2 - 1) * nv:].mean(0)])
    for j in range(nv):
        T.append((c0, (j + 1) % nv, j))
        T.append((c1, (nu2 - 1) * nv + j, (nu2 - 1) * nv + (j + 1) % nv))
    T = np.asarray(T, np.int32)
    zs = np.array([zs0, zs1])
    # collar opening by an exact iso-cut: inside where y < collar line (the line dips toward the instep)
    uu = (V[:, 2] - zs0) / (zs1 - zs0)
    line = collar_h + 0.02 - 0.25 * smoothstep(0.15, 0.45, uu) * (style != "boots")
    line = np.where(uu > 0.42, 9.0, line)            # never cut the vamp / toe box
    fcut = V[:, 1] - line
    Wd = np.zeros((len(V), 1), np.float32)
    V, T, _w, _src, _t, _cut = cut_mesh(V.astype(np.float32), T, fcut, Wd)
    V = V.astype(np.float64)
    # weights: foot bone, toes for the front third
    nb = len(bones())
    wf = _bone_w(nb, {f"foot.{side}": 1.0})
    wt = _bone_w(nb, {f"toe1-1.{side}": 0.2, f"toe2-1.{side}": 0.2, f"toe3-1.{side}": 0.2,
                      f"toe4-1.{side}": 0.2, f"toe5-1.{side}": 0.2})
    wl = _bone_w(nb, {f"lowerleg02.{side}": 1.0})
    u = np.clip((V[:, 2] - zs0) / (zs1 - zs0), 0, 1)
    kt = smoothstep(0.7, 0.95, u)[:, None] * 0.35
    kl = smoothstep(collar_h * 0.7, collar_h * 1.4, V[:, 1])[:, None] if style == "boots" else 0 * kt
    W = wf[None] * (1 - kt - kl) + wt[None] * kt + wl[None] * kl
    g = mesh_garment(body, V.astype(np.float32), T, W, f"{style}.{side}", "shoe", smooth=3, min_jump=0.2)
    g.hem = True
    g.details.append(("y", sole + 0.0, None, 0, 0.5, "sole"))
    # body coverage (skin below the collar is hidden)
    cover = np.where(fm | ((np.sign(X[:, 0]) == sg) & (X[:, 1] < collar_h * 0.95) & (R.foot + R.legness[side] > 0.5)),
                     1.0, -1.0)
    cover[BODY_N:] = -1
    g.cover = cover
    return g
