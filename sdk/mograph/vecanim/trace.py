"""Line-art raster → vector scene (fills with holes + centre-line strokes with width).

Python port / adaptation of VectorCraft's tracer (https://github.com/storytold/vectorcraft,
Apache-2.0): `trace/src/centerline/{mod,graph,thin}.rs` and `fit.rs`.

  1. **Levels.**  The panel is split into *fills* (wide light areas: skin, collars), *light
     strokes* (thin bright lines on the dark paper) and *dark strokes* (thin dark lines drawn
     on top of a fill: eyes, mouth, folds).  "Thin" = VectorCraft's rule: no wider than the
     stroke width (morphological opening with a disc of radius `max_width/2`).
  2. **Skeleton** (Zhang–Suen / skimage) → **graph**: runs between ends and junctions;
     touching junction pixels are one node; spurs shorter than `spur_k·width` beyond the
     junction's maximal disc are pruned (Shaked & Bruckstein significance test, as graph.rs);
     degree-2 nodes are dissolved and smooth continuations through junctions are joined so a
     stroke is as long as the hand that drew it.
  3. **Ink.**  Every skeleton point is moved to the centroid of the ink *across* the line
     (profile sampled along the normal, sub-pixel, contiguous from the darkest/brightest sample
     to where the ink ends or rises again) and the ink sum is the line's width there — the
     anti-aliasing gives both to a fraction of a pixel (centerline/mod.rs `Ink::across`).
     Open ends are extended to where the ink stops, minus the round-cap half width.
  4. **Curves.**  Each run is fitted with VectorCraft's styled Schneider fit (`bezier.py`).
  5. **Fills.**  Sub-pixel contours of the (smoothed) fill mask, holes kept (even-odd), fitted
     as closed chains; the colour is the median tone of the region.

Output: `scene.Scene` (JSON / SVG / renderable).
"""
import numpy as np, cv2
from skimage.morphology import skeletonize
from skimage import measure
from .bezier import Style, fit_open, fit_closed, chain_to_points
from . import scene as S

N8 = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])


# ---------------------------------------------------------------------------- levels
def disc(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def levels(A, bright_thr=0.2, dark_thr=0.5, max_width=7.0, fill_min_area=180, fill_thr=0.45):
    """A: float 0..1 (1 = white).  Returns masks: fill, light (thin bright), dark (thin dark on fill)."""
    bright = (A > bright_thr).astype(np.uint8)
    solid = (A > fill_thr).astype(np.uint8)
    wide = cv2.morphologyEx(solid, cv2.MORPH_OPEN, disc(max_width / 2))
    fill = cv2.morphologyEx(wide, cv2.MORPH_CLOSE, disc(2))          # bridges 1-2 px dark lines only
    n, lab, st, _ = cv2.connectedComponentsWithStats(fill, 8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= fill_min_area
    fill = keep[lab].astype(np.uint8)
    wide = wide & fill
    # thin bright lines: hysteresis (keeps faint continuations of strong lines) + valley split
    # (a pixel much darker than the brightest pixel near it lies *between* two lines)
    weak = (A > bright_thr * 0.55).astype(np.uint8)
    strong = (A > bright_thr * 1.5).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(weak, 8)
    hit = np.zeros(n2, bool); hit[np.unique(lab2[strong > 0])] = True; hit[0] = False
    hyst = hit[lab2]
    peak = cv2.dilate(A, disc(max(2, max_width / 2.5)))
    ridge = A >= 0.5 * peak
    light = (hyst & ridge).astype(np.uint8) & (1 - cv2.dilate(fill, disc(1)))
    # dark detail = dark pixels enclosed by the fill (holes + thin dark lines)
    enclosed = cv2.morphologyEx(fill, cv2.MORPH_CLOSE, disc(5))
    dark = ((A < dark_thr) & (enclosed > 0) & (cv2.erode(enclosed, disc(1)) > 0)).astype(np.uint8)
    return dict(bright=bright, wide=wide, fill=fill, light=light, dark=dark)


# ---------------------------------------------------------------------------- graph
def _graph(sk):
    """Skeleton → list of pixel runs [(N,2) int array of (y,x)], with junction clusters as nodes."""
    sk = sk.astype(np.uint8)
    H, W = sk.shape
    k = np.ones((3, 3), np.float32); k[1, 1] = 0
    deg = cv2.filter2D(sk.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT).astype(np.int32) * sk
    junc = ((deg >= 3) & (sk > 0)).astype(np.uint8)
    nj, jlab = cv2.connectedComponents(junc, connectivity=8)
    on = sk > 0
    used = set()
    runs = []

    def nbrs(y, x):
        for dy, dx in N8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and on[yy, xx]:
                yield yy, xx

    def node_id(p):
        y, x = p
        if jlab[y, x]:
            return ('j', int(jlab[y, x]))
        if deg[y, x] <= 1:
            return ('e', y, x)
        return None

    def walk(a, b):
        path = [a, b]
        used.add((a, b)); used.add((b, a))
        prev, cur = a, b
        while node_id(cur) is None:
            nxt = [q for q in nbrs(*cur) if q != prev and (cur, q) not in used]
            if not nxt:
                break
            # prefer 4-neighbours (no staircase shortcut), then straight continuation
            d0 = (cur[0] - prev[0], cur[1] - prev[1])
            nxt.sort(key=lambda q: (abs(q[0] - cur[0]) + abs(q[1] - cur[1]),
                                    -((q[0] - cur[0]) * d0[0] + (q[1] - cur[1]) * d0[1])))
            q = nxt[0]
            used.add((cur, q)); used.add((q, cur))
            path.append(q); prev, cur = cur, q
        return path

    ys, xs = np.nonzero(on)
    starts = [(y, x) for y, x in zip(ys.tolist(), xs.tolist()) if node_id((y, x)) is not None]
    for s in starts:
        for q in nbrs(*s):
            if (s, q) in used:
                continue
            if jlab[s] and jlab[q] == jlab[s]:
                continue          # inside one junction cluster
            runs.append((walk(s, q), node_id(s)))
    # loops with no node
    for y, x in zip(ys.tolist(), xs.tolist()):
        p = (y, x)
        for q in nbrs(y, x):
            if (p, q) not in used:
                pth = walk(p, q)
                while pth[-1] != p and len(pth) > 2:
                    nxt = [r for r in nbrs(*pth[-1]) if (pth[-1], r) not in used]
                    if not nxt:
                        break
                    used.add((pth[-1], nxt[0])); used.add((nxt[0], pth[-1])); pth.append(nxt[0])
                runs.append((pth, ('loop',)))
    out = []
    for pth, n0 in runs:
        n1 = node_id(pth[-1]) or ('loop',)
        out.append(dict(px=np.array(pth, np.int32), ends=[n0, n1]))
    return out, jlab


def _len(p):
    return float(np.linalg.norm(np.diff(p.astype(float), axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def prune_and_join(runs, dist, width, spur_k=1.6, spur_min=3.0, join_deg=42):
    """Spur pruning beyond the junction disc + joining smooth continuations through nodes."""
    def jkey(e):
        return e if e[0] == 'j' else None
    for _ in range(3):
        cnt = {}
        for r in runs:
            for e in r['ends']:
                if e[0] == 'j':
                    cnt[e] = cnt.get(e, 0) + 1
        keep = []
        for r in runs:
            e0, e1 = r['ends']
            term = (e0[0] == 'e') != (e1[0] == 'e')
            if term:
                j = e0 if e0[0] == 'j' else e1
                rj = float(dist[tuple(r['px'][0 if e0[0] == 'j' else -1])])
                reach = _len(r['px']) - rj
                if reach < max(spur_min, spur_k * min(2 * rj + 1, width)) and cnt.get(j, 0) >= 3:
                    cnt[j] -= 1
                    continue
            if e0[0] == 'e' and e1[0] == 'e' and _len(r['px']) < spur_min:
                continue
            keep.append(r)
        runs = keep
    # join at nodes: greedily pair the two runs whose directions continue best
    cos_t = np.cos(np.radians(join_deg))
    changed = True
    while changed:
        changed = False
        at = {}
        for i, r in enumerate(runs):
            for k, e in enumerate(r['ends']):
                if e[0] == 'j':
                    at.setdefault(e, []).append((i, k))
        for node, lst in at.items():
            if len(lst) < 2:
                continue
            best = None
            dirs = []
            for i, k in lst:
                p = runs[i]['px'].astype(float)
                m = min(len(p) - 1, 8)
                d = (p[m] - p[0]) if k == 0 else (p[-1 - m] - p[-1])   # pointing away from the node
                n = np.linalg.norm(d)
                dirs.append(d / n if n > 1e-9 else np.zeros(2))
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    if lst[a][0] == lst[b][0]:
                        continue
                    c = -float(dirs[a] @ dirs[b])
                    if c > cos_t and (best is None or c > best[0]):
                        best = (c, a, b)
            if len(lst) == 2 and best is None and lst[0][0] != lst[1][0]:
                best = (0, 0, 1)        # degree-2 node: always dissolve
            if best:
                (i, ki), (j, kj) = lst[best[1]], lst[best[2]]
                pi, pj = runs[i]['px'], runs[j]['px']
                ei, ej = runs[i]['ends'], runs[j]['ends']
                if ki == 0:
                    pi, ei = pi[::-1], ei[::-1]
                if kj == 1:
                    pj, ej = pj[::-1], ej[::-1]
                runs[i] = dict(px=np.vstack([pi, pj[1:]]), ends=[ei[0], ej[1]])
                runs.pop(j)
                changed = True
                break
    return runs


# ---------------------------------------------------------------------------- ink
def ink_refine(pts, G, polarity, reach=6.0, step=0.25):
    """pts (N,2) x,y.  G float image.  polarity +1 = bright ink, -1 = dark ink.
    Returns refined pts and per-point widths."""
    n = len(pts)
    if n < 3:
        return pts.astype(float), np.full(n, 1.0)
    p = pts.astype(float)
    a = p[np.clip(np.arange(n) - 2, 0, n - 1)]; b = p[np.clip(np.arange(n) + 2, 0, n - 1)]
    t = b - a; tn = np.linalg.norm(t, axis=1, keepdims=True); t = t / np.maximum(tn, 1e-9)
    nrm = np.c_[-t[:, 1], t[:, 0]]
    ks = np.arange(-reach, reach + 1e-9, step)
    X = (p[:, 0:1] + nrm[:, 0:1] * ks[None]).astype(np.float32)
    Y = (p[:, 1:2] + nrm[:, 1:2] * ks[None]).astype(np.float32)
    prof = cv2.remap(G.astype(np.float32), X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if polarity < 0:
        prof = 1 - prof
    paper = prof.min(1, keepdims=True); inkv = prof.max(1, keepdims=True)
    tone = np.clip((prof - paper) / np.maximum(inkv - paper, 1e-3), 0, 1)
    absprof = np.clip(prof - paper, 0, None)
    c = len(ks) // 2
    win = int(round(2 / step))
    peak = c - win + np.argmax(tone[:, c - win:c + win + 1], axis=1)
    # contiguous ink from the peak: stop where tone < 0.08 or rises again (another line)
    m = np.zeros_like(tone, bool)
    rows = np.arange(n)
    m[rows, peak] = True
    for side in (-1, 1):
        last = tone[rows, peak].copy()
        alive = np.ones(n, bool)
        for k in range(1, len(ks)):
            idx = peak + side * k
            ok = (idx >= 0) & (idx < len(ks)) & alive
            ii = np.clip(idx, 0, len(ks) - 1)
            tv = tone[rows, ii]
            ok &= (tv >= 0.08) & (tv <= last + 0.12)
            m[rows[ok], ii[ok]] = True
            last = np.where(ok, tv, last); alive = ok
            if not ok.any():
                break
    w = tone * m
    s = w.sum(1)
    mid = (w * ks[None]).sum(1) / np.maximum(s, 1e-9)
    width = s * step                                   # width at the line's own peak tone
    ink_amount = (absprof * m).sum(1) * step           # ink integral (tone x px) -> crisp width
    good = (s > 1e-3) & ((inkv[:, 0] - paper[:, 0]) > 0.15)
    off = np.where(good, mid, 0.0)
    off = np.clip(off, -2.0, 2.0)
    q = p + nrm * off[:, None]
    width = np.where(good, width, np.nan)
    ink_refine.last = dict(amount=np.where(good, ink_amount, np.nan), peak=inkv[:, 0], paper=paper[:, 0])
    return q, width


def extend_end(p, G, polarity, width, k_back=4):
    """Move the open end to where the ink ends minus the round-cap half width."""
    if len(p) < k_back + 2:
        return p
    d = p[-1] - p[-1 - k_back]; n = np.linalg.norm(d)
    if n < 1e-9:
        return p
    d = d / n
    s = np.arange(0.25, width + 4, 0.25)
    X = (p[-1, 0] + d[0] * s).astype(np.float32)[None]; Y = (p[-1, 1] + d[1] * s).astype(np.float32)[None]
    prof = cv2.remap(G.astype(np.float32), X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
    v0 = cv2.remap(G.astype(np.float32), np.float32([[p[-1, 0]]]), np.float32([[p[-1, 1]]]), cv2.INTER_LINEAR)[0, 0]
    if polarity < 0:
        prof, v0 = 1 - prof, 1 - v0
    lo = min(prof.min(), v0)
    tone = np.clip((prof - lo) / max(v0 - lo, 1e-3), 0, 1)
    stop = np.argmax(tone < 0.08) if (tone < 0.08).any() else len(tone)
    reach = 0.25 * tone[:stop].sum()
    more = reach - width / 2
    if more > 0.1:
        p = p.copy(); p[-1] = p[-1] + d * more
    return p


# ---------------------------------------------------------------------------- strokes
def trace_strokes(mask, G, polarity, style, min_len=5.0, max_width=7.0, color=1.0, group='line', crisp=1.15):
    sk = skeletonize(mask > 0)
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    widest = float(np.median(dist[sk])) * 2 if sk.any() else 2.0
    runs, _ = _graph(sk)
    runs = prune_and_join(runs, dist, widest)
    items = []
    for r in runs:
        px = r['px']
        closed = r['ends'][0][0] == 'loop' or (len(px) > 8 and abs(px[0] - px[-1]).max() <= 1
                                                and r['ends'][0] == r['ends'][1] and r['ends'][0][0] == 'j')
        if _len(px) < min_len:
            continue
        p = px[:, ::-1].astype(float) + 0.0          # x,y at pixel centres (pixel index = centre)
        q, wv = ink_refine(p, G, polarity, reach=max(4.0, max_width))
        info = ink_refine.last
        if color is None and polarity > 0:
            # crisp line art: draw brighter than the blurred source, same amount of ink
            pk = np.nanpercentile(info['peak'], 75)
            col = float(np.clip(pk * crisp, 0.2, 1.0))
            wv = 0.85 * info['amount'] / max(col - float(np.nanmedian(info['paper'])), 0.05)
        else:
            col = color
        fallback = 2 * dist[px[:, 0], px[:, 1]]
        wv = np.where(np.isfinite(wv), wv, fallback)
        wv = np.clip(cv2.GaussianBlur(wv.reshape(-1, 1).astype(np.float32), (1, 0), 2).ravel()
                     if len(wv) > 4 else wv, 0.35, max_width)
        if not closed:
            for k in range(2):
                e = r['ends'][k]
                if e[0] == 'e':
                    if k == 0:
                        q = extend_end(q[::-1], G, polarity, float(np.median(wv)))[::-1]
                    else:
                        q = extend_end(q, G, polarity, float(np.median(wv)))
        segs = fit_closed(q, style) if closed else fit_open(q, style)
        if segs is None or len(segs) == 0:
            continue
        P = chain_to_points(segs, closed)
        # per-anchor widths: nearest refined point to every anchor
        anchors = P[0::3]
        d2 = ((anchors[:, None] - q[None]) ** 2).sum(-1)
        aw = wv[np.argmin(d2, axis=1)]
        items.append(S.Stroke(P, closed, float(np.median(wv)), aw, col, group=group))
    return items


# ---------------------------------------------------------------------------- fills
def trace_fills(fill, A, style, iso=0.45, min_area=180, group='fill', hole_min=6.0, tone=None):
    """Each fill region -> one compound path: iso-contours of the panel itself at `iso` inside
    the region (sub-pixel, anti-aliasing aware), so dark details enclosed by the fill (eyes,
    mouth, folds) become exact holes."""
    n, lab = cv2.connectedComponents(fill.astype(np.uint8), connectivity=8)
    items = []
    for k in range(1, n):
        reg = (lab == k).astype(np.uint8)
        if reg.sum() < min_area:
            continue
        ys, xs = np.nonzero(reg)
        y0, y1, x0, x1 = max(ys.min() - 6, 0), ys.max() + 7, max(xs.min() - 6, 0), xs.max() + 7
        rr = cv2.dilate(reg[y0:y1, x0:x1], disc(2)).astype(np.float32)
        rr = cv2.GaussianBlur(rr, (0, 0), 1.0)
        field = np.minimum(A[y0:y1, x0:x1], rr * 1.0 + (1 - rr) * 0) * (rr > 0.02)
        field = np.pad(field, 1)
        cs = measure.find_contours(field, iso)
        paths = []
        for c in cs:
            pts = c[:, ::-1] + [x0 - 1, y0 - 1]
            area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
            if abs(area) < hole_min:
                continue
            segs = fit_closed(pts[:-1] if np.allclose(pts[0], pts[-1]) else pts, style)
            if segs is None:
                continue
            paths.append(chain_to_points(segs, True))
        if not paths:
            continue
        if tone is None:
            core = (cv2.erode(reg, disc(2)) > 0) & (A > iso)
            vals = A[core] if core.sum() > 20 else A[reg > 0]
            tn = float(np.median(vals))
        else:
            tn = tone
        items.append(S.Fill(paths, tn, group=group))
    return items


# ---------------------------------------------------------------------------- pipeline
def trace_panel(A0, max_width=7.0, bg=None, fill_style=None, line_style=None, verbose=True, up=2, dark_strokes=False):
    """A0: float32 HxW 0..1 line-art panel (light lines on dark paper, grey fills).
    `up`: working upsample (VectorCraft traces an upsampled field too); the scene is returned in
    A0's pixel coordinates."""
    import time
    t0 = time.time()
    A = cv2.resize(A0, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1 else A0
    A = np.clip(A, 0, 1)
    max_width = max_width * up
    L = levels(A, max_width=max_width, fill_min_area=180 * up * up)
    ls = line_style or Style(step=0.5 * up, sigma=1.0 * up, corner_deg=62, corner_win=2.5 * up, tol=0.45 * up, line_tol=0.3 * up)
    fs = fill_style or Style(step=0.6 * up, sigma=1.4 * up, corner_deg=70, corner_win=3.0 * up, tol=0.6 * up, line_tol=0.4 * up)
    # two tone layers (VectorCraft quantize: one layer per palette colour, stacked):
    # shade (iso 0.32) under light (iso 0.66) -> mid-grey details (smile lines, folds) survive
    reg = cv2.dilate(L['fill'], disc(2 * up))
    lo, hi = 0.32, 0.66
    sel = (A > lo) & (reg > 0)
    t_lo = float(np.percentile(A[sel & (A < hi)], 50)) if (sel & (A < hi)).sum() > 50 else 0.55
    t_hi = float(np.percentile(A[(A >= hi) & (reg > 0)], 50)) if ((A >= hi) & (reg > 0)).sum() > 50 else 0.88
    fills = trace_fills(L['fill'], A, fs, iso=lo, min_area=180 * up * up, group='shade', tone=t_lo) + \
        trace_fills(L['fill'], A, fs, iso=hi, min_area=40 * up * up, group='fill', tone=t_hi)
    t1 = time.time()
    light = trace_strokes(L['light'], A, +1, ls, min_len=5.0 * up, max_width=max_width, color=None, group='light')
    t2 = time.time()
    dark = []
    if dark_strokes:
        dark_tone = float(np.percentile(A[L['dark'] > 0], 20)) if L['dark'].any() else 0.05
        dark = trace_strokes(L['dark'], A, -1, ls, min_len=3.0 * up, max_width=max_width, color=dark_tone, group='dark')
    t3 = time.time()
    if bg is None:
        bg = float(np.median(A[(L['bright'] == 0) & (cv2.dilate(L['fill'], disc(3)) == 0)]))
    items = fills + dark + light
    if up != 1:
        for it in items:
            it.set_points(it.points() / up)
            if it.kind == 'stroke':
                it.width /= up; it.aw = it.aw / up
    sc = S.Scene(A0.shape[1], A0.shape[0], bg, items)
    if verbose:
        print(f'trace: {len(fills)} fills, {len(light)} light strokes, {len(dark)} dark strokes, '
              f'{sc.n_points()} control points | fills {t1-t0:.1f}s light {t2-t1:.1f}s dark {t3-t2:.1f}s')
    return sc, L


def fidelity(A, R, mask=None):
    """Simple numbers: mean abs error, and line IoU at 0.5."""
    d = np.abs(A - R)
    a, r = A > 0.42, R > 0.42
    iou = (a & r).sum() / max((a | r).sum(), 1)
    # tolerant IoU: 1 px slack both ways (centre-line placement)
    k = np.ones((3, 3), np.uint8)
    ad, rd = cv2.dilate(a.astype(np.uint8), k) > 0, cv2.dilate(r.astype(np.uint8), k) > 0
    prec = (r & ad).sum() / max(r.sum(), 1); rec = (a & rd).sum() / max(a.sum(), 1)
    return dict(mae=float(d.mean()), iou=float(iou), precision_1px=float(prec), recall_1px=float(rec))
