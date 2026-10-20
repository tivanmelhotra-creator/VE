"""
mograph.genimage — v11 engines that turn AI-generated images (the agent's native image generator writes .jpg/.png
into the workspace) into ANIMATABLE layers.  Everything is preprocessed once and cached next to the image
(<img>.trace.json / <img>.depth/ / <img>.rig/).

ELEMENT auto_trace   line-art image → vector strokes drawn live (trim-path), à la ref07 / ref09 neon line art.
  {"type":"auto_trace","src":"art/elephant.jpg","x":540,"y":900,"w":900,
   "mode":"centerline|outline", "invert":"auto", "thresh":"auto", "min_len":12, "simplify":1.2,
   "color":"#ffffff", "color_mode":"source" (sample the image colours per stroke), "width":3.0, "glow":{"color":"#2ad1ff"},
   "dur":2.5 (draw-on), "order":"top|left|size|center" (stroke order), "stagger":0.6 (0 = all strokes together,
   1 = strictly one after another), "fill_image":0.0..1 (fade the raster in under the strokes at the end)}
  mode centerline = skeleton (thin lines from strokes of any width) · outline = contours of filled shapes.

ELEMENT depth_plate  one picture → 3 depth layers (far / mid / near) with inpainted holes → 2.5D parallax
  {"type":"depth_plate","src":"art/city.jpg","x":540,"y":960,"w":1080,"h":1920,"layers":3,
   "split":"auto|matte|luma|y", "push":[[0,1.0],[4,1.12]] (camera dolly zoom), "pan":[[0,[0,0]],[4,[-60,0]]],
   "strength":1.0, "dof":true (blur far layer), "bg_blur":2.5}
  near layer = subject (portrait matte if available, else saliency), far = top/sky region, mid = the rest.

ELEMENT sprite_rig   2×3 (or N×M) sheet of character parts on a flat green / black background → puppet
  {"type":"sprite_rig","src":"art/puppet_sheet.jpg","grid":[3,2],
   "parts":["head","torso","arm_l","arm_r","leg_l","leg_r"],          # sheet cells, row-major
   "x":540,"y":1100,"scale":1.0,"bg":"auto|green|black|white",
   "pose":{"arm_r":[[0,0],[1,-60,"in_out"],[2,0]], "head":[[0,0],[1,8]]},   # joint angle keyframes (deg)
   "idle":"breathe|wave|walk|talk" (procedural cycles), "speed":1.0}
  Parts are cut out (chroma/luma key + largest component), their joints are found automatically
  (top-centre of limbs, bottom of head, shoulders/hips on the torso) and assembled into a hierarchy.

CLI:  python3 -m mograph prompts spec.json   → image + music prompts the agent should generate (see prompts.py)
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import cairo
import cv2
import numpy as np

from .core import Track, clamp, color as _color, ease_in_out, get_ease, lerp, prog
from .elements import element, xy, val


def _src(el, s):
    p = el["src"]
    if not os.path.isabs(p) and s.scene is not None:
        p = os.path.join(s.scene.base_dir, p)
    return p


def _tag(path, opts):
    st = os.stat(path)
    return hashlib.md5(json.dumps([st.st_size, int(st.st_mtime), opts], sort_keys=True).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# auto_trace
# ---------------------------------------------------------------------------
def _binarize(gray, invert="auto", thresh="auto"):
    if invert == "auto":
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        invert = float(np.median(border)) > 127            # light paper → dark lines → invert
    g = 255 - gray if invert else gray
    if thresh == "auto":
        _t, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _t, b = cv2.threshold(g, int(thresh), 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _skeleton_paths(b, min_len):
    sk = cv2.ximgproc.thinning(b, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    pts = set(zip(*np.nonzero(sk)))
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def deg(p):
        return sum((p[0] + dy, p[1] + dx) in pts for dy, dx in nb)

    paths = []
    visited = set()
    # start from endpoints first, then loops
    starts = [p for p in pts if deg(p) == 1] + list(pts)
    for st in starts:
        if st in visited:
            continue
        path = [st]; visited.add(st); cur = st
        while True:
            nxt = None
            for dy, dx in nb:
                q = (cur[0] + dy, cur[1] + dx)
                if q in pts and q not in visited:
                    nxt = q
                    break
            if nxt is None:
                break
            visited.add(nxt); path.append(nxt); cur = nxt
        if len(path) >= min_len:
            paths.append(np.array([[p[1], p[0]] for p in path], np.float32))
    return paths


def _outline_paths(b, min_len):
    cs, _h = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2).astype(np.float32) for c in cs if len(c) >= min_len]


def trace_image(path, mode="centerline", invert="auto", thresh="auto", min_len=12, simplify=1.2, work=900):
    """→ {"w","h","paths":[[[x,y],…],…], "colors":[hex…]} normalised to the source size, cached as JSON."""
    opts = [mode, invert, thresh, min_len, simplify, work]
    cp = f"{path}.trace_{_tag(path, opts)}.json"
    if os.path.exists(cp):
        return json.load(open(cp))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h0, w0 = img.shape[:2]
    k = min(1.0, work / max(h0, w0))
    im = cv2.resize(img, (int(w0 * k), int(h0 * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # colour line art on black: use max channel as "ink"
    if invert == "auto" and float(np.median(gray)) < 60:
        gray = im.max(axis=2)
    b = _binarize(gray, invert, thresh)
    raw = _skeleton_paths(b, min_len) if mode == "centerline" else _outline_paths(b, min_len)
    paths, cols = [], []
    for p in raw:
        ap = cv2.approxPolyDP(p.reshape(-1, 1, 2), simplify, mode != "centerline").reshape(-1, 2)
        if len(ap) < 2:
            continue
        idx = p[len(p) // 2].astype(int)
        bgr = im[min(im.shape[0] - 1, idx[1]), min(im.shape[1] - 1, idx[0])]
        if mode == "centerline":                               # sample the brightest nearby pixel (line colour)
            y0, x0 = max(0, idx[1] - 2), max(0, idx[0] - 2)
            patch = im[y0:idx[1] + 3, x0:idx[0] + 3].reshape(-1, 3)
            bgr = patch[np.argmax(patch.sum(axis=1))]
        cols.append("#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0])))
        paths.append((ap / k).round(1).tolist())
    data = {"w": w0, "h": h0, "paths": paths, "colors": cols}
    json.dump(data, open(cp, "w"))
    return data


def _order(paths, how, w, h):
    def first(p):
        return p[0]
    idx = list(range(len(paths)))
    if how == "left":
        idx.sort(key=lambda i: min(q[0] for q in paths[i]))
    elif how == "right":
        idx.sort(key=lambda i: -max(q[0] for q in paths[i]))
    elif how == "size":
        idx.sort(key=lambda i: -len(paths[i]))
    elif how == "center":
        idx.sort(key=lambda i: math.hypot(np.mean([q[0] for q in paths[i]]) - w / 2, np.mean([q[1] for q in paths[i]]) - h / 2))
    else:   # top
        idx.sort(key=lambda i: min(q[1] for q in paths[i]))
    return idx


@element("auto_trace", dur=2.5, enter="draw", sfx="draw")
def el_auto_trace(ctx, el, s):
    p = _src(el, s)
    data = el.get("_trace")
    if data is None:
        data = trace_image(p, el.get("mode", "centerline"), el.get("invert", "auto"), el.get("thresh", "auto"),
                           int(el.get("min_len", 12)), float(el.get("simplify", 1.2)))
        L = [sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(q, q[1:])) for q in data["paths"]]
        data = dict(data, lens=L, order=_order(data["paths"], el.get("order", "top"), data["w"], data["h"]))
        el["_trace"] = data
    x, y = xy(el, s.t)
    W = float(el.get("w", 900)); k = W / data["w"]; H = data["h"] * k
    trim = s.trim if isinstance(s.trim, float) else 1.0
    a = s.alpha
    order = data["order"]
    n = max(1, len(order))
    stg = clamp(float(el.get("stagger", 0.6)))
    total = sum(data["lens"]) or 1
    ctx.save()
    ctx.translate(x - W / 2, y + s.dy - H / 2)
    if el.get("fill_image"):
        fa = float(el["fill_image"]) * clamp((trim - 0.85) / 0.15)
        if fa > 0:
            from .media import to_surface
            img = el.get("_img")
            if img is None:
                img = el["_img"] = cv2.imread(p, cv2.IMREAD_COLOR)
            surf, _keep = to_surface(img)
            ctx.save(); ctx.scale(k, k); ctx.set_source_surface(surf, 0, 0)
            ctx.paint_with_alpha(a * fa); ctx.restore()
    ctx.scale(k, k)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND); ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(float(el.get("width", 3.0)) / k)
    mono = "source" if el.get("color_mode") == "source" else el.get("color", "#ffffff")
    acc = 0.0
    for rank, i in enumerate(order):
        pts, Li = data["paths"][i], data["lens"][i]
        # progress window of this stroke: parallel (stg=0) … sequential by length (stg=1)
        if stg >= 0.999:
            a0, a1 = acc / total, (acc + Li) / total
        else:
            st = rank / n * stg
            a0, a1 = st, st + (1 - stg)
        acc += Li
        q = clamp((trim - a0) / max(1e-6, a1 - a0))
        if q <= 0:
            continue
        col = data["colors"][i] if mono == "source" else mono
        ctx.set_source_rgba(*_color(col, a))
        target = q * Li
        ctx.move_to(*pts[0])
        run = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            d = math.hypot(x1 - x0, y1 - y0)
            if run + d >= target:
                f = (target - run) / max(d, 1e-6)
                ctx.line_to(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
                break
            ctx.line_to(x1, y1)
            run += d
        ctx.stroke()
    ctx.restore()
    return (x - W / 2, y - H / 2, x + W / 2, y + H / 2)


# ---------------------------------------------------------------------------
# depth_plate
# ---------------------------------------------------------------------------
def depth_layers(path, n=3, split="auto", work=960):
    """→ list of (bgra uint8) layers far→near, cached as PNGs in <img>.depth_<tag>/"""
    d = f"{path}.depth_{_tag(path, [n, split, work])}"
    files = [os.path.join(d, f"l{i}.png") for i in range(n)]
    if all(os.path.exists(f) for f in files):
        return [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h0, w0 = img.shape[:2]
    k = min(1.0, work / max(h0, w0))
    im = cv2.resize(img, (int(w0 * k), int(h0 * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
    H, W = im.shape[:2]
    near = None
    if split in ("auto", "matte"):
        try:
            from .matte import matte as _matte
            a, _m = _matte(im, "onnx")
            if a.mean() > 0.03:
                near = a
        except Exception:  # noqa: BLE001
            near = None
    if near is None:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sm = sal.computeSaliency(im)
        sm = cv2.GaussianBlur((sm * 255).astype(np.uint8), (0, 0), 9)
        _t, b = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nlab, lab, st, _ = cv2.connectedComponentsWithStats(b)
        if nlab > 1:
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            b = (lab == j).astype(np.uint8) * 255
        mask = np.zeros((H, W), np.uint8)
        try:
            rect = cv2.boundingRect(b)
            gm = np.where(b > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
            bg_, fg_ = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(im, gm, rect, bg_, fg_, 2, cv2.GC_INIT_WITH_MASK)
            mask = np.where((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:  # noqa: BLE001
            mask = b
        near = cv2.GaussianBlur(mask, (0, 0), 2).astype(np.float32) / 255.0
    # far vs mid: luminance-weighted vertical split (sky / distant stuff on top)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    lum = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    far_score = (1 - yy) * 0.75 + cv2.GaussianBlur(lum, (0, 0), 15) * 0.25
    cut = float(np.percentile(far_score, 55))
    far = np.clip((far_score - cut) * 12 + 0.5, 0, 1) * (1 - near)
    mid = np.clip(1 - near - far, 0, 1)
    masks = [np.ones((H, W), np.float32), mid + near, near] if n >= 3 else [np.ones((H, W), np.float32), near]
    layers = []
    holes = (near > 0.35).astype(np.uint8) * 255
    holes = cv2.dilate(holes, np.ones((15, 15), np.uint8))
    base = cv2.inpaint(im, holes, 6, cv2.INPAINT_TELEA)              # far plate has the subject removed
    for i, m in enumerate(masks):
        src = base if i < len(masks) - 1 else im
        if i == 1 and n >= 3:
            src = im.copy(); src[near > 0.35] = base[near > 0.35]
        a8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
        layers.append(np.dstack([src, a8]))
    os.makedirs(d, exist_ok=True)
    for f, L in zip(files, layers):
        cv2.imwrite(f, L)
    return layers


@element("depth_plate", dur=0.6, enter="fade", sfx="none")
def el_depth_plate(ctx, el, s):
    from .matte import premultiplied_surface
    p = _src(el, s)
    L = el.get("_layers")
    if L is None:
        layers = depth_layers(p, int(el.get("layers", 3)), el.get("split", "auto"))
        surfs = []
        for i, l in enumerate(layers):
            bgr = np.ascontiguousarray(l[..., :3])
            if i == 0 and el.get("dof", True):
                bgr = cv2.GaussianBlur(bgr, (0, 0), float(el.get("bg_blur", 2.5)))
            surfs.append(premultiplied_surface(bgr, l[..., 3].astype(np.float32) / 255.0))
        L = el["_layers"] = surfs
    x, y = xy(el, s.t)
    sw, sh = L[0][0].get_width(), L[0][0].get_height()
    W = float(el.get("w", 1080)); H = float(el.get("h", W * sh / sw))
    k0 = max(W / sw, H / sh)
    lt = s.t - float(el["t"])
    push = Track([tuple(kk) for kk in el["push"]])(lt) if el.get("push") else 1.0 + 0.06 * clamp(lt / 6)
    pan = Track([(kk[0], tuple(kk[1])) for kk in el["pan"]])(lt) if el.get("pan") else (0.0, 0.0)
    strength = float(el.get("strength", 1.0))
    n = len(L)
    a = s.alpha * (1 - s.out_p)
    ctx.save()
    ctx.rectangle(x - W / 2, y - H / 2, W, H); ctx.clip()
    for i, (surf, _keep) in enumerate(L):
        depth = i / max(1, n - 1)                       # 0 far … 1 near
        z = 1 + (push - 1) * (0.35 + 1.3 * depth * strength)
        ox = pan[0] * (0.3 + 1.2 * depth * strength); oy = pan[1] * (0.3 + 1.2 * depth * strength)
        k = k0 * z * (1.04 if i == 0 else 1.0)
        ctx.save()
        ctx.translate(x + ox, y + oy); ctx.scale(k, k); ctx.translate(-sw / 2, -sh / 2)
        ctx.set_source_surface(surf, 0, 0); ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint_with_alpha(a)
        ctx.restore()
    ctx.restore()
    return (x - W / 2, y - H / 2, x + W / 2, y + H / 2)


# ---------------------------------------------------------------------------
# sprite_rig
# ---------------------------------------------------------------------------
DEFAULT_PARTS = ["head", "torso", "arm_l", "arm_r", "leg_l", "leg_r"]
# parent, pivot on the part (u,v of its own box), attach point on the parent (u,v of the parent's box), z order
SKELETON = {
    "torso": (None, (0.5, 0.5), (0.5, 0.5), 2),
    "head": ("torso", (0.5, 0.92), (0.5, 0.02), 4),
    "arm_l": ("torso", (0.5, 0.06), (0.06, 0.12), 1),
    "arm_r": ("torso", (0.5, 0.06), (0.94, 0.12), 5),
    "leg_l": ("torso", (0.5, 0.04), (0.32, 0.97), 0),
    "leg_r": ("torso", (0.5, 0.04), (0.68, 0.97), 3),
}


def _key_mask(cell, bg):
    if bg == "auto":
        border = np.concatenate([cell[0], cell[-1], cell[:, 0], cell[:, -1]]).astype(np.float32)
        b = np.median(border, axis=0)
        bg = "green" if b[1] > b[0] + 40 and b[1] > b[2] + 40 else ("black" if b.sum() < 120 else
                                                                    "white" if b.min() > 200 else "color")
        key = b
    if bg == "green":
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        m = ~((hsv[..., 0] > 35) & (hsv[..., 0] < 90) & (hsv[..., 1] > 70))
        m = m.astype(np.uint8) * 255
    elif bg == "black":
        m = (cell.max(axis=2) > 28).astype(np.uint8) * 255
    elif bg == "white":
        m = (cell.min(axis=2) < 225).astype(np.uint8) * 255
    else:
        d = np.linalg.norm(cell.astype(np.float32) - key, axis=2)
        m = (d > 40).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = ((lab == j) * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 0.8), bg


def cut_sheet(path, grid=(3, 2), parts=None, bg="auto"):
    """→ dict part → (bgra cropped, (w, h)), cached as PNGs."""
    parts = parts or DEFAULT_PARTS
    d = f"{path}.rig_{_tag(path, [grid, parts, bg])}"
    files = {p: os.path.join(d, f"{p}.png") for p in parts}
    if all(os.path.exists(f) for f in files.values()):
        return {p: cv2.imread(f, cv2.IMREAD_UNCHANGED) for p, f in files.items()}
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    cols, rows = grid
    out = {}
    for i, name in enumerate(parts):
        r, c = divmod(i, cols)
        cell = img[int(r * H / rows):int((r + 1) * H / rows), int(c * W / cols):int((c + 1) * W / cols)]
        m, _bg = _key_mask(cell, bg)
        ys, xs = np.nonzero(m > 10)
        if not len(ys):
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out[name] = np.dstack([cell[y0:y1, x0:x1], m[y0:y1, x0:x1]])
    os.makedirs(d, exist_ok=True)
    for p, im in out.items():
        cv2.imwrite(files[p], im)
    return out


def _end_centroid(alpha, end="top", frac=0.12):
    """(u, v) of the centroid of the top/bottom `frac` of a part's mask — the joint of a limb."""
    ys, xs = np.nonzero(alpha > 128)
    if not len(ys):
        return (0.5, 0.05 if end == "top" else 0.95)
    h, w = alpha.shape
    y0, y1 = ys.min(), ys.max()
    band = (y1 - y0) * frac
    sel = ys <= y0 + band if end == "top" else ys >= y1 - band
    return (float(xs[sel].mean()) / w, (float(ys[sel].mean()) + (band * 0.35 if end == "top" else -band * 0.2)) / h)


def auto_joints(cut):
    """Joint pivots found from the masks: limbs pivot at their top end, the head at its chin; attach points on
    the torso = shoulders (top corners of the body mask, inset) and hips (bottom, inset)."""
    piv, att = {}, {}
    for n, im in cut.items():
        a = im[..., 3]
        if n == "head":
            piv[n] = _end_centroid(a, "bottom", 0.08)
        elif n != "torso":
            piv[n] = _end_centroid(a, "top", 0.1)
    t = cut.get("torso")
    if t is not None:
        a = t[..., 3]
        h, w = a.shape
        ys, xs = np.nonzero(a > 128)
        rows = lambda v: xs[(ys >= v * h - 2) & (ys <= v * h + 2)]
        r = rows(0.2)
        if len(r):
            att["arm_l"] = ((r.min() + 0.10 * w) / w, 0.2)
            att["arm_r"] = ((r.max() - 0.10 * w) / w, 0.2)
        att["head"] = (float(xs[ys < h * 0.1].mean()) / w if (ys < h * 0.1).any() else 0.5, 0.06)
        att["leg_l"], att["leg_r"] = (0.36, 0.94), (0.64, 0.94)
    return piv, att


def _cycle(idle, name, t, speed):
    w = 2 * math.pi * speed
    if idle == "walk":
        return {"leg_l": 22 * math.sin(w * t), "leg_r": -22 * math.sin(w * t), "arm_l": -18 * math.sin(w * t),
                "arm_r": 18 * math.sin(w * t), "head": 2 * math.sin(2 * w * t), "torso": 1.5 * math.sin(2 * w * t)}.get(name, 0)
    if idle == "wave":
        return {"arm_r": -110 + 25 * math.sin(w * 2 * t), "head": 4 * math.sin(w * t)}.get(name, 0)
    if idle == "talk":
        return {"head": 4 * math.sin(w * 1.7 * t), "arm_r": -15 + 10 * math.sin(w * t),
                "arm_l": 8 * math.sin(w * 0.8 * t + 1)}.get(name, 0)
    return {"head": 2.5 * math.sin(w * 0.5 * t), "arm_l": 3 * math.sin(w * 0.5 * t + 1),
            "arm_r": -3 * math.sin(w * 0.5 * t + 2), "torso": 0.8 * math.sin(w * 0.5 * t)}.get(name, 0)


@element("sprite_rig", dur=0.4, enter="pop", sfx="pop")
def el_sprite_rig(ctx, el, s):
    from .matte import premultiplied_surface
    p = _src(el, s)
    R = el.get("_rig")
    if R is None:
        cut = cut_sheet(p, tuple(el.get("grid", [3, 2])), el.get("parts"), el.get("bg", "auto"))
        R = {}
        for name, im in cut.items():
            surf = premultiplied_surface(np.ascontiguousarray(im[..., :3]), im[..., 3].astype(np.float32) / 255.0)
            R[name] = (surf, im.shape[1], im.shape[0])
        el["_rig"] = R
        el["_pivots"], el["_attach"] = auto_joints(cut)
    skel = dict(SKELETON)
    for k, v in (el.get("joints") or {}).items():        # override pivots: {"arm_r": {"pivot":[.5,.05],"attach":[.9,.1]}}
        b = list(skel.get(k, (None, (0.5, 0.5), (0.5, 0.5), 3)))
        if "parent" in v: b[0] = v["parent"]
        if "pivot" in v: b[1] = tuple(v["pivot"])
        if "attach" in v: b[2] = tuple(v["attach"])
        if "z" in v: b[3] = v["z"]
        skel[k] = tuple(b)
    x, y = xy(el, s.t)
    lt = s.t - float(el["t"])
    sc = float(el.get("scale", 1.0)) * s.scale
    pose = el.get("pose") or {}
    tr_cache = el.setdefault("_ptr", {})
    idle = el.get("idle", "breathe"); spd = float(el.get("speed", 1.0))

    def angle(name):
        a = _cycle(idle, name, lt, spd)
        if name in pose:
            if name not in tr_cache:
                tr_cache[name] = Track([tuple(k) for k in pose[name]])
            a += float(tr_cache[name](lt))
        return a

    # forward kinematics: world matrix of each part = local(rotate about its pivot) × parent attach transform
    mats = {}

    def solve(name):
        if name in mats:
            return mats[name]
        parent, pivot, attach, _z = skel[name]
        _surf, w, h = R[name]
        piv = el["_pivots"].get(name, pivot) if el.get("auto_joints", True) else pivot
        if parent is None or parent not in R:
            ax, ay, rp = x, y + s.dy, 0.0
        else:
            pm = solve(parent)
            _ps, pw, ph = R[parent]
            att = el["_attach"].get(name, attach) if el.get("auto_joints", True) else attach
            ax, ay = pm.transform_point(att[0] * pw, att[1] * ph)
            rp = math.atan2(pm.yx, pm.xx)
        r = rp + math.radians(angle(name))
        c, si = math.cos(r) * sc, math.sin(r) * sc
        m = cairo.Matrix(c, si, -si, c, ax, ay)                      # rotate+scale, origin at the joint
        m = cairo.Matrix(1, 0, 0, 1, -piv[0] * w, -piv[1] * h).multiply(m)
        mats[name] = m
        return m

    names = [n for n in R if n in skel]
    a = s.alpha * (1 - s.out_p)
    xs, ys = [], []
    for name in sorted(names, key=lambda n: skel[n][3]):
        m = solve(name)
        surf, w, h = R[name]
        ctx.save()
        ctx.transform(m)
        ctx.set_source_surface(surf[0], 0, 0); ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint_with_alpha(a)
        ctx.restore()
        for px, py in ((0, 0), (w, 0), (0, h), (w, h)):
            qx, qy = m.transform_point(px, py); xs.append(qx); ys.append(qy)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))
