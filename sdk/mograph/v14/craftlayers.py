"""
mograph.v14.craftlayers — v15.1 spec-level layers for the v15 Craft engines (so the agent uses them from JSON,
not from Python).  Registered into layers.DRAW on import.

  shatter_text   a kinetic text block that holds, then EXPLODES into 3D shards at `break_at` (shatter3d.py)
      {"type": "shatter_text", "t": "w:مقاله", "out": 4.2, "text": "*مقاله*", "size": 90, "anchor": "head_top",
       "offset": [0, -0.6], "break_at": "w:نوشتن", "rows": 4, "cols": 7, "force": 380, "gravity": 900}
  puppet_sticker an RGBA sticker (assets_3d / project path) bent by ARAP pins (puppet_arap.py) — a waving flag,
                 a wobbling cartoon, a bending arrow.  Pins: normalised [u, v] on the image + motion.
      {"type": "puppet_sticker", "src": "warn_3d.png", "anchor": "shoulder_r", "size": 0.9,
       "pins": [{"uv": [0.5, 1.0]}, {"uv": [0.5, 0.0], "wiggle": [26, 0], "freq": 1.6}], "grid": 10}
  morph_shape    a filled shape that MORPHS between outlines (autotrace.shape_context_procrustes_morph):
                 shapes = names (circle | hept | hex | star | heart | diamond | rsquare) or point lists (block px)
      {"type": "morph_shape", "shapes": ["circle", "star", "heart"], "at": ["w:اول", "w:دوم"], "dur": 0.45,
       "w": 1.3, "anchor": "chest", "color": "@accent", "glow": 0.6}
All three respect pass ("back"/"front"), the emission context (bloom) and out-fade like the other v14 layers.
"""
from __future__ import annotations

import math

import cairo
import cv2
import numpy as np

from . import glyphs as G
from . import physics as ph


def _alpha_io(L, F, fin=0.15, fout=0.3):
    lt = F.t - L["_t0"]
    if lt < 0:
        return 0.0, lt, None
    olt = (F.t - L["_t1"]) if F.t > L["_t1"] - 1e-6 else None
    a = min(1.0, lt / fin)
    if olt is not None:
        a *= max(0.0, 1.0 - olt / fout)
    return a, lt, olt


# ============================================================================================ shatter_text
def _block_bgra(L, scale):
    """Typeset the block once → premultiplied BGRA ndarray (cached on the layer)."""
    key = f"_shat_img_{scale:.3f}"
    if key in L:
        return L[key]
    from .layers import blk
    b = blk(L)
    pad = 24
    w, h = int(b.w * scale) + 2 * pad, int(b.h * scale) + 2 * pad
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, max(8, w), max(8, h))
    c = cairo.Context(s)
    G.draw_block(c, b, w / 2, h / 2, pal=L["_pal"], extrude=L.get("extrude", {"depth": 6}), scale=scale)
    s.flush()
    arr = np.ndarray((s.get_height(), s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :s.get_width()].copy()
    L[key] = arr
    return arr


def draw_shatter_text(ctx, L, F):
    from .layers import pos
    from ..shatter3d import render_shatter_3d
    a, lt, olt = _alpha_io(L, F, 0.12, 0.25)
    if a <= 0.003:
        return
    x, y, k = pos(L, F)
    img = _block_bgra(L, round(k, 2))
    tb = L.get("_break_at")
    tb = (L["_t0"] + 0.9) if tb is None else tb
    tt = max(0.0, F.t - tb)
    if tt <= 0:                                              # entrance: jelly pop of the whole card
        p = max(0.0, ph.sp(lt, "jelly"))
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(p, p)
        ctx.translate(-x, -y)
    else:
        ctx.save()
    if a < 0.999:
        ctx.push_group()
    render_shatter_3d(ctx, img, x, y, tt, rows=int(L.get("rows", 4)), cols=int(L.get("cols", 7)),
                      force=float(L.get("force", 380)) * k, gravity=float(L.get("gravity", 900)) * k,
                      extrude_px=float(L.get("thickness", 7)) * k, seed=int(L.get("seed", 42)))
    if a < 0.999:
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(a)
    ctx.restore()


# ============================================================================================ puppet_sticker
def draw_puppet_sticker(ctx, L, F):
    from .layers import pos
    from ..puppet_arap import ArapPuppetMesh
    a, lt, olt = _alpha_io(L, F)
    if a <= 0.003:
        return
    mesh = L.get("_mesh")
    if mesh is None:
        from ..obj3d import resolve_asset
        path = resolve_asset(L.get("src", "warn_3d.png"), F.R.base)
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f"puppet_sticker: cannot read {path}")
        if im.shape[2] == 3:
            im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
        mx = 360                                               # work size (the mesh warp is per-triangle)
        sc = mx / max(im.shape[:2])
        if sc < 1:
            im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        pad = int(0.18 * max(im.shape[:2]))                    # room for the deformation
        im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        g = int(L.get("grid", 10))
        mesh = L["_mesh"] = ArapPuppetMesh(im, g, g)
        mesh.pad = pad
    pins = []
    Wc, Hc = mesh.W - 2 * mesh.pad, mesh.H - 2 * mesh.pad
    for pn in L.get("pins") or [{"uv": [0.5, 1.0]}, {"uv": [0.5, 0.0], "wiggle": [24, 0], "freq": 1.5}]:
        u, v = pn["uv"]
        px, py = mesh.pad + u * (Wc - 1), mesh.pad + v * (Hc - 1)
        vi = mesh.nearest_vertex(px / (mesh.W - 1), py / (mesh.H - 1))
        tx, ty = mesh.V_rest[vi]
        wg = pn.get("wiggle")
        if wg:
            fr = float(pn.get("freq", 1.5))
            ph_ = float(pn.get("phase", 0.0))
            tx += float(wg[0]) * math.sin(2 * math.pi * fr * F.t + ph_)
            ty += float(wg[1]) * math.sin(2 * math.pi * fr * F.t + ph_ + 1.3)
        if pn.get("to") is not None:                         # one-shot bend: [dx, dy] reached with a spring
            p = ph.sp(lt, "bouncy")
            tx += float(pn["to"][0]) * p
            ty += float(pn["to"][1]) * p
        pins.append((vi, (tx, ty), 1e4))
    V = mesh.solve(pins, iterations=int(L.get("iterations", 6)))
    img = mesh.warp_image(V)
    # premultiply for cairo
    al = img[..., 3:4].astype(np.float32) / 255.0
    pm = np.ascontiguousarray(np.dstack([(img[..., :3] * al).astype(np.uint8), img[..., 3]]))
    x, y, k = pos(L, F)
    size = float(L.get("size", 0.9)) * F.head_px()
    sc = size / max(Wc, Hc) * max(0.0, ph.sp(lt, "jelly"))
    surf = cairo.ImageSurface.create_for_data(pm, cairo.FORMAT_ARGB32, pm.shape[1], pm.shape[0])
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(sc, sc)
    ctx.set_source_surface(surf, -mesh.W / 2, -mesh.H / 2)
    ctx.paint_with_alpha(a)
    ctx.restore()


# ============================================================================================ morph_shape
def _shape_pts(name, n=96):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    if name == "circle":
        return np.column_stack([np.cos(t), np.sin(t)])
    if name == "star":
        r = np.where((np.arange(n) // (n // 10)) % 2 == 0, 1.0, 0.45)
        tt = np.linspace(0, 2 * math.pi, 11)[:-1] - math.pi / 2
        pts = [[math.cos(a) * (1.0 if i % 2 == 0 else 0.45), math.sin(a) * (1.0 if i % 2 == 0 else 0.45)]
               for i, a in enumerate(tt)]
        return np.asarray(pts)
    if name == "heart":
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        return np.column_stack([x, y]) / 17.0
    nsides = {"hept": 7, "hex": 6, "pent": 5, "diamond": 4, "tri": 3, "oct": 8, "rsquare": 4}.get(name)
    if nsides:
        off = -math.pi / 2 if name != "rsquare" else math.pi / 4
        a = np.arange(nsides) * 2 * math.pi / nsides + off
        return np.column_stack([np.cos(a), np.sin(a)])
    raise KeyError(f"morph_shape: unknown shape '{name}'")


def draw_morph_shape(ctx, L, F):
    from .layers import pos
    from ..autotrace import shape_context_procrustes_morph
    a, lt, olt = _alpha_io(L, F)
    if a <= 0.003:
        return
    shapes = L.get("_shapes")
    if shapes is None:
        shapes = L["_shapes"] = [np.asarray(_shape_pts(s) if isinstance(s, str) else s, np.float32)
                                 for s in L.get("shapes", ["circle", "star"])]
    ats = L.get("_at") or []
    dur = float(L.get("dur", 0.45))
    idx, tm = 0, 0.0
    for i, ta in enumerate(ats[:len(shapes) - 1]):
        if F.t >= ta:
            idx, tm = i, min(1.0, (F.t - ta) / dur)
    if tm >= 1.0 and idx + 1 < len(shapes):
        idx, tm = idx + 1, 0.0
    if tm > 0 and idx + 1 < len(shapes):
        key = (idx, round(ph.sp(tm * 1.0, "jelly") if L.get("spring") else ph.SILK(tm), 3))
        cache = L.setdefault("_mcache", {})
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = shape_context_procrustes_morph(shapes[idx], shapes[idx + 1], key[1], n_pts=96,
                                                        bending_resistance=float(L.get("rigidity", 0.6)))
        P = cache[key]
    else:
        P = shapes[idx]
    x, y, k = pos(L, F)
    R = float(L.get("w", 1.3)) * F.head_px() / 2 * max(0.0, ph.sp(lt, "jelly"))
    col = G.rgba(L.get("color", "@accent"), L["_pal"])

    def path(c):
        c.new_path()
        c.move_to(x + P[0, 0] * R, y + P[0, 1] * R)
        for q in P[1:]:
            c.line_to(x + q[0] * R, y + q[1] * R)
        c.close_path()

    ctx.save()
    path(ctx)
    ctx.set_source_rgba(col[0], col[1], col[2], a * float(L.get("fill_alpha", 0.92)))
    ctx.fill_preserve()
    if L.get("stroke", True):
        ctx.set_source_rgba(1, 1, 1, a * 0.85)
        ctx.set_line_width(max(1.5, 3.0 * k))
        ctx.stroke()
    ctx.new_path()
    ctx.restore()
    gl = float(L.get("glow", 0.6))
    if F.ectx is not None and gl > 0:
        F.ectx.save()
        path(F.ectx)
        F.ectx.set_source_rgba(col[0], col[1], col[2], a * gl)
        F.ectx.fill()
        F.ectx.restore()


def register(DRAW):
    DRAW["shatter_text"] = draw_shatter_text
    DRAW["puppet_sticker"] = draw_puppet_sticker
    DRAW["morph_shape"] = draw_morph_shape
