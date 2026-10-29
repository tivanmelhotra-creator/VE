"""
mograph.toon.shade — soft CEL-SHADED 3D render of toon characters (the "white clay toon" look):
white body, warm-grey soft-edged shadow band + gentle form gradient, faint rim, clean dark outlines (thicker on the
silhouette, thin on inner overlaps / creases), soft contact + cast shadow on a flat warm paper floor.
Faces are NOT 3D: mograph.toon.face draws hand-made line-art expressions on the head ball afterwards.

    img, info = render_shaded(scene, toons, t, cam, W, H)        # img: float RGB (H, W, 3) in 0..1
    draw faces with info["G"][name] / info["head_mask"][name]   (see compose_shaded)

GPU (moderngl / EGL): one pass renders shaded colour + linear depth + part id + view normal (MRT, ss× res);
a second pass renders the meshes flattened onto the floor along the light direction → shadow mask.
CPU (numpy / OpenCV): outlines from id / depth / normal discontinuities, AA downsample, shadow blur, paper.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..human.lineart import _ctx

PAPER = (0.875, 0.80, 0.62)
INK = (0.13, 0.12, 0.115)
LIGHT = np.array([-0.38, 1.25, 0.50])          # world light (upper-left-front), shadows fall right / back

_S = {}

VS = """#version 330
uniform mat4 V; uniform mat4 P;
in vec3 pos; in vec3 nrm; in float pid; in vec3 col;
out float d; out vec3 vn; out vec3 wn; out vec3 c; flat out float id;
void main(){
    vec4 v = V * vec4(pos, 1.0);
    d = -v.z; id = pid; c = col;
    wn = nrm; vn = mat3(V) * nrm;
    gl_Position = P * v;
}"""

FS = """#version 330
uniform vec3 L;            // world light dir
uniform vec3 shadow_col;   // multiplier for the shadow side
uniform vec3 eye_dir;      // world dir towards camera (for rim)
in float d; in vec3 vn; in vec3 wn; in vec3 c; flat in float id;
layout(location=0) out vec4 o_col;
layout(location=1) out vec4 o_buf;   // depth, id, nz(view), 1
layout(location=2) out vec4 o_nrm;   // view normal
void main(){
    vec3 n = normalize(wn);
    float lam = dot(n, normalize(L));
    // soft cel step + gentle form gradient
    float lit = smoothstep(-0.12, 0.30, lam);
    float form = 0.93 + 0.07 * clamp(lam, 0.0, 1.0);
    vec3 base = c * form;
    vec3 col = mix(base * shadow_col, base, lit);
    // bounce light from the floor into the shadow side (keeps it soft, never muddy)
    col += c * shadow_col * 0.06 * clamp(-n.y, 0.0, 1.0);
    // faint rim on the shadow side
    float rim = pow(1.0 - clamp(dot(n, normalize(eye_dir)), 0.0, 1.0), 3.0);
    col = mix(col, c, rim * 0.18 * (1.0 - lit));
    o_col = vec4(col, 1.0);
    vec3 v = normalize(vn);
    o_buf = vec4(d, id, v.z, 1.0);
    o_nrm = vec4(v, 1.0);
}"""


def _prog():
    if "prog" not in _S:
        ctx, _ = _ctx()
        _S["ctx"] = ctx
        _S["prog"] = ctx.program(vertex_shader=VS, fragment_shader=FS)
        _S["fbo"] = {}
    return _S["ctx"], _S["prog"]


def _fbo(W, H):
    ctx, _ = _prog()
    if (W, H) not in _S["fbo"]:
        _S["fbo"].clear()
        tex = [ctx.texture((W, H), 4, dtype="f4") for _ in range(3)]
        dep = ctx.depth_renderbuffer((W, H))
        _S["fbo"][(W, H)] = (ctx.framebuffer(tex, dep), tex)
    return _S["fbo"][(W, H)]


def vertex_normals(X, T):
    a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    fn = np.cross(b - a, c - a)
    N = np.zeros_like(X)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return N


def _draw(ctx, prog, meshes):
    import moderngl
    for X, T, N, pid, col in meshes:
        n = len(X)
        data = np.hstack([X, N, np.full((n, 1), float(pid)), np.tile(np.asarray(col, np.float32), (n, 1))])
        vbo = ctx.buffer(data.astype("f4").tobytes())
        ibo = ctx.buffer(np.asarray(T, np.int32).tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "3f 3f 1f 3f", "pos", "nrm", "pid", "col")], ibo)
        vao.render(moderngl.TRIANGLES)
        vao.release()
        vbo.release()
        ibo.release()


def gl_shade(cam, meshes, ss, light=LIGHT, shadow_col=(0.80, 0.78, 0.76)):
    import moderngl
    ctx, prog = _prog()
    W, H = int(cam.W * ss), int(cam.H * ss)
    fbo, tex = _fbo(W, H)
    fbo.use()
    ctx.viewport = (0, 0, W, H)
    fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    prog["V"].write(cam.view().T.astype("f4").tobytes())
    prog["P"].write(cam.proj().T.astype("f4").tobytes())
    prog["L"].value = tuple(float(v) for v in light / np.linalg.norm(light))
    prog["shadow_col"].value = tuple(shadow_col)
    e = cam.eye() - np.asarray(cam.target, float)
    prog["eye_dir"].value = tuple(float(v) for v in e / np.linalg.norm(e))
    _draw(ctx, prog, meshes)
    out = [np.frombuffer(t.read(), "f4").reshape(H, W, 4)[::-1].copy() for t in tex]
    return out


def shadow_meshes(meshes, light=LIGHT, y0=0.0):
    """flatten every mesh onto the floor y=y0 along the light direction."""
    L = light / np.linalg.norm(light)
    out = []
    for X, T, N, pid, col in meshes:
        h = X[:, 1] - y0
        P = X - np.outer(h / max(1e-3, L[1]), L)
        P[:, 1] = y0 + 0.002
        out.append((P.astype(np.float32), T, np.tile([0, 1.0, 0], (len(P), 1)).astype(np.float32), 999, (0, 0, 0)))
    return out


_DISK = {}


def _disk(r):
    if r not in _DISK:
        _DISK[r] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return _DISK[r]


def outlines(buf, nrm, ss, w_outer, w_inner, crease_deg=0.0, groups=None):
    """alpha masks (ss res) for silhouette (thick) and inner lines (thin)."""
    d = buf[..., 0]
    ids = np.rint(buf[..., 1]).astype(np.int32)
    occ = buf[..., 3] > 0.5
    g = ids
    if groups:
        lut = np.arange(max(int(ids.max()), max(groups)) + 1)
        for a, b in groups.items():
            lut[a] = b
        g = lut[ids]
    dd = np.where(occ, d, 1e5)
    outer = np.zeros(ids.shape, bool)
    inner = np.zeros(ids.shape, bool)
    for sl0, sl1 in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                     ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        o0, o1 = occ[sl0], occ[sl1]
        bg = o0 != o1                                               # object ↔ background
        jump = np.abs(dd[sl0] - dd[sl1]) > (0.02 * np.minimum(dd[sl0], dd[sl1]) + 0.5)   # self / other overlap
        idc = (g[sl0] != g[sl1]) & o0 & o1
        n0, n1 = nrm[sl0][..., :3], nrm[sl1][..., :3]
        cr = ((np.sum(n0 * n1, -1) < math.cos(math.radians(crease_deg))) & o0 & o1 & ~jump) if crease_deg > 0 else False
        outer[sl0] |= bg
        outer[sl1] |= bg
        big = jump & o0 & o1
        inner[sl0] |= big | idc | cr
        inner[sl1] |= big | idc | cr
    # distance-field ink: crisp, exactly `w` px wide (at output res) and anti-aliased after the ss downsample.
    # The edge set marks pixels on BOTH sides of a boundary, so the stroke is centred on the true contour.
    m = np.zeros(ids.shape, np.float32)
    for mask, w in ((outer, w_outer), (inner & ~outer, w_inner)):
        if not mask.any():
            continue
        dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        r = max(0.5, (w * ss - 1.0) / 2)
        m = np.maximum(m, np.clip(r + 0.5 - dist, 0, 1))
    return m


def collect_meshes(scene, toons, t, hidden=()):
    """posed meshes (+ per-actor matrices / head pid) using each toon's rig_parts / parts."""
    from .render import actor_toon_parts
    meshes, Gs, heads, groups = [], {}, {}, {}
    for k, (name, a) in enumerate(scene.actors.items()):
        if name in hidden or name not in toons:
            continue
        base = 40 * (k + 1)
        parts, _extra, G, hp = actor_toon_parts(a, toons[name], scene, t, base)
        for p in parts:
            X = np.asarray(p.verts, np.float32)
            T = np.asarray(p.tris, np.int32)
            N = getattr(p, "normals", None)
            if N is None:
                N = vertex_normals(X.astype(np.float64), T).astype(np.float32)
            meshes.append((X, T, np.asarray(N, np.float32), p.pid,
                           toons[name].fill_for(p.name)))
        Gs[name], heads[name] = G, hp
        groups.update(getattr(toons[name], "groups", None) or {})
    return meshes, Gs, heads, groups


def render_shaded(scene, toons, t, cam, ss=2.0, line_px=None, paper=PAPER, ink=INK, light=LIGHT,
                  shadow_strength=0.32, props=(), hidden=(), shadow=True):
    """→ (rgb float (H, W, 3), info{G, head_pid, ids(ss), head_mask{name}(H,W)})"""
    W, H = cam.W, cam.H
    meshes, Gs, heads, groups = collect_meshes(scene, toons, t, hidden)
    meshes += list(props)
    col, buf, nrm = gl_shade(cam, meshes, ss, light)
    occ = buf[..., 3]
    # ---- floor + cast shadow
    bg = np.empty((int(H * ss), int(W * ss), 3), np.float32)
    bg[:] = paper
    if shadow:
        _c, sbuf, _n = gl_shade(cam, shadow_meshes(meshes, light), ss, light)
        sm = (sbuf[..., 3] > 0.5).astype(np.float32)
        soft = cv2.GaussianBlur(sm, (0, 0), 9.0 * ss) * 0.7 + cv2.GaussianBlur(sm, (0, 0), 3.0 * ss) * 0.3
        bg *= (1 - shadow_strength * soft)[..., None]
        # contact darkening right under the feet (ambient occlusion feel)
        ao = cv2.GaussianBlur((sm * (sbuf[..., 0] > 0)).astype(np.float32), (0, 0), 1.2 * ss)
        bg *= (1 - 0.08 * ao)[..., None]
    img = bg * (1 - occ[..., None]) + col[..., :3] * occ[..., None]
    # ---- outlines
    # thin pen line like the reference sheets: ~0.28 % of the frame height (≈ 2.1 px at 760 px), inner overlaps
    # at 55 % of that so the body's own shading (chest, knees, folds) carries the form instead of heavy ink
    lw = line_px or max(1.25, H * 0.0028)
    a = outlines(buf, nrm, ss, lw, lw * 0.55, groups=groups)
    img = img * (1 - a[..., None]) + np.asarray(ink, np.float32) * a[..., None]
    out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    ids = np.rint(buf[..., 1]).astype(np.int32)
    hm = {}
    for n, hp in heads.items():
        tk = toons[n]
        pids = getattr(tk, "face_pids", None) or [hp]
        m = np.isin(ids, pids)
        # keep only pixels lying on the ball's front surface (hides hair strands / ponytail in front of the face
        # correctly, and never paints the face onto hair behind the head)
        c = tk.head.center(Gs[n])
        zc = float(cam.project(c[None])[1][0])
        r = tk.head.r
        m &= (buf[..., 0] < zc + 0.15 * r) & (buf[..., 0] > zc - 1.25 * r)
        hm[n] = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    return out, dict(G=Gs, head_pid=heads, ids=ids, head_mask=hm, ss=ss)


def compose_shaded(img, info, cam, toons, faces, t, line_scale=1.0):
    """draw the 2D line-art faces on the shaded frame → uint8 RGB."""
    import cairo
    from .render import head_px
    H, W = img.shape[:2]
    rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    a = np.ndarray((H, W, 4), np.uint8, surf.get_data())
    a[..., 0], a[..., 1], a[..., 2], a[..., 3] = rgb[..., 2], rgb[..., 1], rgb[..., 0], 255
    surf.mark_dirty()
    ctx = cairo.Context(surf)
    for name, f in faces.items():
        if name not in info["G"]:
            continue
        G = info["G"][name]
        m = info["head_mask"][name]
        if m.max() < 0.01:
            continue
        tk = toons[name]
        _, rpx = head_px(cam, tk, G)
        wpx = max(1.2, min(8.0, rpx * 0.036)) * line_scale
        layer = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        c2 = cairo.Context(layer)
        c2.set_line_cap(cairo.LINE_CAP_ROUND)
        c2.set_line_join(cairo.LINE_JOIN_ROUND)
        tk.face.draw(c2, G, cam, width=wpx, t=t, **f)
        layer.flush()
        la = np.ndarray((H, W, 4), np.uint8, layer.get_data())
        la[:] = (la.astype(np.float32) * m[..., None]).astype(np.uint8)
        layer.mark_dirty()
        ctx.set_source_surface(layer, 0, 0)
        ctx.paint()
    surf.flush()
    o = np.ndarray((H, W, 4), np.uint8, surf.get_data())
    return o[..., [2, 1, 0]].copy()
