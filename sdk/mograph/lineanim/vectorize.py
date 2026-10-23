"""Raster line-art -> centreline strokes (polylines with per-point width).

Pipeline: upscale (Lanczos) -> light blur -> threshold -> skeletonize ->
pixel graph -> split at junctions -> chain -> merge smooth continuations ->
RDP + Chaikin.  Width from the distance transform.
"""
import numpy as np, cv2
from skimage.morphology import skeletonize

N8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def binarize(gray, scale=4.0, invert=False, thr=None, blur=1.2):
    g = gray.astype(np.float32)
    if invert: g = 255 - g
    h, w = g.shape
    up = cv2.resize(g, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
    up = cv2.GaussianBlur(up, (0,0), blur*scale/3)
    if thr is None:
        thr = cv2.threshold(np.clip(up,0,255).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    return up > thr, up

def trace(mask):
    sk = skeletonize(mask)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    H, W = sk.shape
    ys, xs = np.nonzero(sk)
    on = set(zip(ys.tolist(), xs.tolist()))
    def nb(p):
        y, x = p
        return [(y+dy, x+dx) for dy, dx in N8 if (y+dy, x+dx) in on]
    deg = {p: len(nb(p)) for p in on}
    nodes = {p for p, d in deg.items() if d != 2}
    used = set()
    paths = []
    def walk(a, b):
        path = [a, b]; used.add(frozenset((a, b)))
        prev, cur = a, b
        while cur not in nodes:
            nxt = [q for q in nb(cur) if q != prev and frozenset((cur, q)) not in used]
            if not nxt: break
            # prefer straight continuation
            d0 = np.subtract(cur, prev)
            nxt.sort(key=lambda q: -np.dot(np.subtract(q, cur), d0))
            q = nxt[0]; used.add(frozenset((cur, q)))
            path.append(q); prev, cur = cur, q
        return path
    for n in nodes:
        for q in nb(n):
            if frozenset((n, q)) not in used:
                paths.append(walk(n, q))
    # pure loops
    for p in on:
        if p in nodes: continue
        for q in nb(p):
            if frozenset((p, q)) not in used:
                nodes.add(p); paths.append(walk(p, q)); nodes.discard(p)
    out = []
    for pth in paths:
        a = np.array(pth, float)[:, ::-1]  # x,y
        wv = np.array([dist[y, x] for y, x in pth]) * 2
        out.append((a, wv))
    return out

def _dir(p, end, k=6):
    if end == 0: seg = p[:k]; return seg[0] - seg[-1]
    seg = p[-k:]; return seg[-1] - seg[0]

def merge(strokes, ang=40, gap=3.0):
    """Join strokes meeting at junctions when continuation is smooth."""
    S = [list(s) for s in strokes]
    alive = [True]*len(S)
    cos_t = np.cos(np.radians(ang))
    changed = True
    while changed:
        changed = False
        ends = []
        for i, s in enumerate(S):
            if not alive[i] or len(s[0]) < 3: continue
            for e in (0, 1):
                pt = s[0][0] if e == 0 else s[0][-1]
                d = _dir(s[0], e); n = np.linalg.norm(d)
                if n < 1e-6: continue
                ends.append((i, e, pt, d/n))
        if not ends: break
        P = np.array([x[2] for x in ends])
        best = None
        for a in range(len(ends)):
            dd = np.linalg.norm(P - P[a], axis=1)
            for b in np.nonzero(dd < gap)[0]:
                if b <= a or ends[b][0] == ends[a][0]: continue
                c = -np.dot(ends[a][3], ends[b][3])
                if c > cos_t and (best is None or c > best[0]):
                    best = (c, a, b)
        if best:
            _, a, b = best
            i, ei = ends[a][:2]; j, ej = ends[b][:2]
            pi, wi = S[i]; pj, wj = S[j]
            if ei == 0: pi, wi = pi[::-1], wi[::-1]
            if ej == 1: pj, wj = pj[::-1], wj[::-1]
            S[i] = [np.vstack([pi, pj]), np.concatenate([wi, wj])]
            alive[j] = False; changed = True
    return [tuple(s) for s, a in zip(S, alive) if a]

def rdp(p, eps):
    if len(p) < 3: return np.arange(len(p))
    a, b = p[0], p[-1]; ab = b - a; n = np.linalg.norm(ab)
    d = np.abs(np.cross(ab, p - a))/n if n > 1e-9 else np.linalg.norm(p - a, axis=1)
    i = int(np.argmax(d))
    if d[i] > eps:
        l = rdp(p[:i+1], eps); r = rdp(p[i:], eps)
        return np.concatenate([l[:-1], r + i])
    return np.array([0, len(p)-1])

def smooth(p, w, it=3):
    for _ in range(it):
        if len(p) < 3: break
        q = 0.75*p[:-1] + 0.25*p[1:]; r = 0.25*p[:-1] + 0.75*p[1:]
        np_ = np.empty((2*len(q), 2)); np_[0::2] = q; np_[1::2] = r
        wq = 0.75*w[:-1] + 0.25*w[1:]; wr = 0.25*w[:-1] + 0.75*w[1:]
        nw = np.empty(2*len(wq)); nw[0::2] = wq; nw[1::2] = wr
        p = np.vstack([p[:1], np_, p[-1:]]); w = np.concatenate([w[:1], nw, w[-1:]])
    return p, w

def vectorize(gray, scale=4.0, invert=False, min_len=6, eps=0.9, thr=None, blur=1.2):
    mask, up = binarize(gray, scale, invert, thr, blur)
    st = merge(trace(mask))
    out = []
    for p, w in st:
        L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        if L < min_len*scale/4*4: continue
        # pre-smooth raw pixel chain
        k = 3
        if len(p) > 2*k+1:
            ker = np.ones(2*k+1)/(2*k+1)
            px = np.convolve(np.pad(p[:,0],k,mode='edge'),ker,'valid')
            py = np.convolve(np.pad(p[:,1],k,mode='edge'),ker,'valid')
            wv = np.convolve(np.pad(w,k,mode='edge'),ker,'valid')
            p = np.c_[px, py]; w = wv
        idx = rdp(p, eps*scale/4)
        p2, w2 = smooth(p[idx], w[idx], 3)
        out.append({'pts': p2, 'w': w2, 'len': L})
    return out, mask, up
