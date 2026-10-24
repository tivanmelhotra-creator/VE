"""
mograph.toon.render — one toon frame: flat fills from the GPU id buffer + ink outlines (slight boil) + faces.

  1. pose every actor (mocap Actor) → body parts (real head shrunk away), ball head sphere, hair shells, props
  2. render_lines(...) → occluding contours + depth + PART-ID buffer
  3. fills: id buffer → colour per part → area-downsampled (antialiased)
  4. ink: contour ribbons with a small per-drawing wobble ("boil"; constant while a drawing is held)
  5. faces: each ToonFace drawn on its own layer masked by that actor's head pixels
Animation is meant to be sampled ON TWOS (quantize(t, 12)) like hand-drawn shorts.
"""
from __future__ import annotations

import math

import cairo
import cv2
import numpy as np

from ..human.lineart import Camera, Style, render_lines, ribbon

PAPER = (0.86, 0.80, 0.65)          # warm beige paper
INK = (0.1, 0.095, 0.09)


def toon_style(width=4.2):
    return Style(color=INK, width=width, inner=0.6, detail=0.42, hem=0.55, pressure=0.2, taper=0.14, min_len=6.0,
                 smooth=2)


def quantize(t, fps_draw=12):
    return math.floor(t * fps_draw + 1e-6) / fps_draw


def actor_toon_parts(actor, toon, scene, t, base):
    if hasattr(toon, "rig_parts"):                      # dedicated 3D toon model (mograph.toon.rig)
        G = actor.matrices(t, scene).astype(np.float32)
        parts, extra, hp = toon.rig_parts(G, base)
        return parts, extra, G, hp
    G, X = actor.posed(t, scene)
    X = toon.head.shrink_mesh(G, X, getattr(toon, 'hand_scale', 1.0))
    parts = toon.ch.parts(G, X)
    for p in parts:
        p.pid = base + p.pid
    extra = []
    if toon.ch.folds:
        for s in toon.ch.fold_strokes(G, parts):
            if s.pid and s.pid < base:
                s.pid += base
            extra.append(s)
    head_pid = base + 15
    parts += toon.head.parts(G, head_pid, base + 16)
    return parts, extra, G, head_pid


def render_toon(scene, toons, t, cam: Camera, style=None, ss=1.5, fig_px=None, seed=0, props=(), prop_strokes=(),
                prop_fills=None, hidden=()):
    """→ dict(strokes, depth, ids, fills{pid: rgb}, G{name}, head_pid{name}, ss)"""
    style = style or toon_style()
    parts, extra, fills, Gs, heads, groups = [], list(prop_strokes), {}, {}, {}, {}
    for k, (name, a) in enumerate(scene.actors.items()):
        if name in hidden or name not in toons:
            continue
        base = 40 * (k + 1)
        p, e, G, hp = actor_toon_parts(a, toons[name], scene, t, base)
        for q in p:
            fills[q.pid] = toons[name].fill_for(q.name)
        groups.update(getattr(toons[name], "groups", {}))
        parts += p
        extra += e
        Gs[name] = G
        heads[name] = hp
    for q in props:
        parts.append(q)
        fills[q.pid] = (prop_fills or {}).get(q.pid, (0.97, 0.96, 0.94))
    lf = render_lines(parts, cam, style, extra=extra, ss=ss, fig_px=fig_px or cam.H * 0.8, seed=seed)
    return dict(strokes=lf.strokes, depth=lf.depth, ids=lf.ids, fills=fills, G=Gs, head_pid=heads, ss=ss,
                groups=groups)


def fill_image(ids, fills, W, H):
    lut_n = int(ids.max()) + 1
    lut = np.zeros((max(lut_n, 1), 4), np.float32)
    for pid, rgb in fills.items():
        if pid < lut_n:
            lut[pid, :3] = rgb
            lut[pid, 3] = 1.0
    return cv2.resize(lut[ids], (W, H), interpolation=cv2.INTER_AREA)


def mask_of(ids, pids, W, H):
    m = np.isin(ids, list(pids)).astype(np.float32)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)


def boil(strokes, t_draw, amp=0.8, seed=0):
    rng = np.random.default_rng(int(round(t_draw * 1000)) + seed * 7919)
    ph = rng.uniform(0, 2 * math.pi, 4)
    out = []
    for s in strokes:
        p = s.pts
        dx = amp * (np.sin(p[:, 1] / 37.0 + ph[0]) * 0.6 + np.sin(p[:, 0] / 23.0 + ph[1]) * 0.4)
        dy = amp * (np.sin(p[:, 0] / 41.0 + ph[2]) * 0.6 + np.sin(p[:, 1] / 29.0 + ph[3]) * 0.4)
        out.append(type(s)(np.c_[p[:, 0] + dx, p[:, 1] + dy], s.w, s.kind, getattr(s, "closed", False)))
    return out


def draw_ink(ctx, strokes, ink=INK, alpha=1.0, wscale=1.0):
    ctx.set_source_rgba(*ink, alpha)
    for s in strokes:
        pts, w = s.pts, s.w * wscale
        if len(pts) < 2:
            continue
        L, R = ribbon(pts, w)
        ctx.new_path()
        ctx.move_to(*L[0])
        for q in L[1:]:
            ctx.line_to(*q)
        for q in R[::-1]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.fill()
        for j in (0, -1):
            ctx.arc(pts[j][0], pts[j][1], w[j] / 2, 0, 2 * math.pi)
            ctx.fill()


def surface_from_rgba(img):
    H, W = img.shape[:2]
    a = np.clip(img[..., 3:4], 0, 1)
    rgb = np.clip(img[..., :3], 0, 1) * a
    buf = np.empty((H, W, 4), np.uint8)
    buf[..., 0] = (rgb[..., 2] * 255).astype(np.uint8)
    buf[..., 1] = (rgb[..., 1] * 255).astype(np.uint8)
    buf[..., 2] = (rgb[..., 0] * 255).astype(np.uint8)
    buf[..., 3] = (a[..., 0] * 255).astype(np.uint8)
    return cairo.ImageSurface.create_for_data(bytearray(buf.tobytes()), cairo.FORMAT_ARGB32, W, H, W * 4)


def masked_layer(ctx, W, H, mask, draw_fn):
    layer = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c2 = cairo.Context(layer)
    c2.set_line_cap(cairo.LINE_CAP_ROUND)
    c2.set_line_join(cairo.LINE_JOIN_ROUND)
    draw_fn(c2)
    layer.flush()
    a = np.ndarray((H, W, 4), np.uint8, layer.get_data())
    a[:] = (a.astype(np.float32) * mask[..., None]).astype(np.uint8)
    layer.mark_dirty()
    ctx.set_source_surface(layer, 0, 0)
    ctx.paint()


def cam_right(cam):
    y = math.radians(cam.yaw)
    return np.array([math.cos(y), 0.0, -math.sin(y)])


def head_px(cam, toon, G):
    c = toon.head.center(G)
    xy, _ = cam.project(np.array([c, c + cam_right(cam) * toon.head.r]))
    return xy[0], float(np.linalg.norm(xy[1] - xy[0]))


_DISK = {}


def edge_ink(ids, depth, W, H, width_px, t_draw=0.0, boil_amp=0.8, rel=0.06, seed=0, groups=None):
    """clean cartoon outlines from the buffers: id changes (piece / background boundaries) + depth jumps inside a
    piece (self overlap) → dilated to width → smoothed → antialiased alpha (H, W). Constant-weight ink like the
    reference turnarounds; always closed (no broken silhouettes)."""
    h, w = ids.shape
    ss = w / W
    g = ids
    if groups:
        lut = np.arange(max(int(ids.max()), max(groups)) + 1)
        for a, b in groups.items():
            lut[a] = b
        g = lut[ids]
    d = np.where(np.isfinite(depth) & (depth < 1e5), depth, 1e5)
    e = np.zeros((h, w), bool)
    # id boundary → line, except between pieces of the same group (sleeve ↔ sweater) at continuous depth
    tol = 0.08
    e[:, 1:] |= (ids[:, 1:] != ids[:, :-1]) & ((g[:, 1:] != g[:, :-1]) | (np.abs(d[:, 1:] - d[:, :-1]) > tol))
    e[1:, :] |= (ids[1:, :] != ids[:-1, :]) & ((g[1:, :] != g[:-1, :]) | (np.abs(d[1:, :] - d[:-1, :]) > tol))
    jx = np.abs(d[:, 1:] - d[:, :-1]) > rel * np.minimum(d[:, 1:], d[:, :-1]) * 0.05 + 0.6
    jy = np.abs(d[1:, :] - d[:-1, :]) > rel * np.minimum(d[1:, :], d[:-1, :]) * 0.05 + 0.6
    e[:, 1:] |= jx
    e[1:, :] |= jy
    r = max(1, int(round(width_px * ss / 2)))
    if r not in _DISK:
        _DISK[r] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.dilate(e.astype(np.uint8) * 255, _DISK[r])
    m = cv2.GaussianBlur(m, (0, 0), max(0.8, r * 0.45))
    m = np.clip((m.astype(np.float32) - 70) / 110, 0, 1)
    if boil_amp > 0:
        rng = np.random.default_rng(int(round(t_draw * 1000)) + seed)
        ph = rng.uniform(0, 6.28, 4)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = boil_amp * ss
        mx = xx + a * (np.sin(yy / (37 * ss) + ph[0]) * 0.6 + np.sin(xx / (23 * ss) + ph[1]) * 0.4)
        my = yy + a * (np.sin(xx / (41 * ss) + ph[2]) * 0.6 + np.sin(yy / (29 * ss) + ph[3]) * 0.4)
        m = cv2.remap(m, mx, my, cv2.INTER_LINEAR)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)


def compose(ctx, fr, cam, toons, faces, W, H, t, boil_amp=0.8, t_draw=None, ink=INK, line_scale=1.0,
            outline="edges", width_px=None):
    """fills + ink + faces onto ctx (already painted with the background).
    outline="edges": buffer-edge outlines + 3D detail strokes (seams/creases); "strokes": 3D contour ribbons."""
    img = fill_image(fr["ids"], fr["fills"], W, H)
    td = t if t_draw is None else t_draw
    if outline == "edges":
        wpx = width_px or max(2.0, H / 1920 * 7.0) * line_scale
        a = edge_ink(fr["ids"], fr["depth"], W, H, wpx, td, boil_amp, groups=fr.get("groups"))
        img = img.copy()
        al = img[..., 3:4]
        rgb = img[..., :3] * al
        # composite ink over fills (premultiplied), keep alpha union
        ia = a[..., None]
        rgb = rgb * (1 - ia) + np.asarray(ink, np.float32) * ia
        al = al * (1 - ia) + ia
        img = np.concatenate([rgb / np.maximum(al, 1e-6), al], -1)
        ctx.set_source_surface(surface_from_rgba(img), 0, 0)
        ctx.paint()
        det = [s for s in fr["strokes"] if s.kind in ("detail", "fold", "crease")]
        st = boil(det, td, boil_amp) if boil_amp > 0 else det
        draw_ink(ctx, st, ink, 1.0, line_scale)
    else:
        ctx.set_source_surface(surface_from_rgba(img), 0, 0)
        ctx.paint()
        st = boil(fr["strokes"], td, boil_amp) if boil_amp > 0 else fr["strokes"]
        draw_ink(ctx, st, ink, 1.0, line_scale)
    for name, f in faces.items():
        if name not in fr["G"]:
            continue
        G = fr["G"][name]
        m = mask_of(fr["ids"], [fr["head_pid"][name]], W, H)
        if m.max() < 0.01:
            continue
        tk = toons[name]
        _, rpx = head_px(cam, tk, G)
        wpx = max(1.5, min(10.0, rpx * 0.05)) * line_scale
        masked_layer(ctx, W, H, m, lambda c2, f=f, G=G, tk=tk, wpx=wpx:
                     tk.face.draw(c2, G, cam, width=wpx, t=t, **f))
