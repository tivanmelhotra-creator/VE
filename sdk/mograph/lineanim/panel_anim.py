"""Proof: animate a raster line-art panel with code only.
- soft line extraction (keeps filled shapes, antialiased)
- geodesic draw-on along the connected line network (the 'one line' motif)
- puppet warp: smooth rotations around pivots (head tilt, breathing)
- candle flicker, slow camera push, film grain
"""
import numpy as np, cv2
from collections import deque

def soft_lines(pan, s=5, lo=70, rng=40, blur=2.2):
    up = cv2.resize(pan.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), blur)
    return np.clip((up - lo) / rng, 0, 1)

def geodesic_order(alpha, seed, ds=2, thr=0.3):
    small = cv2.resize(alpha, None, fx=1/ds, fy=1/ds, interpolation=cv2.INTER_AREA)
    m = cv2.dilate((small > thr).astype(np.uint8), np.ones((3, 3), np.uint8))
    H, W = m.shape
    D = np.full((H, W), -1.0, np.float32)
    sy, sx = seed[1] // ds, seed[0] // ds
    ys, xs = np.nonzero(m); k = np.argmin((ys - sy) ** 2 + (xs - sx) ** 2)
    q = deque([(ys[k], xs[k])]); D[ys[k], xs[k]] = 0
    nb = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while True:
        while q:
            y, x = q.popleft(); d = D[y, x]
            for dy, dx in nb:
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and D[yy, xx] < 0:
                    D[yy, xx] = d + (1.414 if dy and dx else 1); q.append((yy, xx))
        # jump to nearest unreached component (keeps drawing continuous)
        un = (m > 0) & (D < 0)
        if not un.any(): break
        reached = (D >= 0).astype(np.uint8)
        dist, lab = cv2.distanceTransformWithLabels(1 - reached, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        uy, ux = np.nonzero(un); j = np.argmin(dist[uy, ux])
        y0, x0 = uy[j], ux[j]
        # base distance = nearest reached pixel's distance + jump
        ry, rx = np.nonzero(reached); i = np.argmin((ry-y0)**2 + (rx-x0)**2)
        D[y0, x0] = D[ry[i], rx[i]] + dist[y0, x0] * 0.5; q.append((y0, x0))
    D[D < 0] = D.max()
    D = cv2.resize(D, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
    return D / D.max()

def ease(t): t = np.clip(t, 0, 1); return t*t*(3-2*t)

def warp(img, rigs):
    """rigs: list of (pivot(x,y), angle_rad, radius, falloff, region_center(x,y), shift(x,y))"""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mx, my = xx.copy(), yy.copy()
    for (px, py), ang, R, fall, (cx, cy), (tx, ty) in rigs:
        w = np.exp(-(((xx-cx)**2 + (yy-cy)**2) / (2*R*R)) ** fall)
        c, s = np.cos(-ang*w).astype(np.float32), np.sin(-ang*w).astype(np.float32)
        dx, dy = mx - px - tx*w, my - py - ty*w
        mx, my = px + c*dx - s*dy, py + s*dx + c*dy
    return cv2.remap(img.astype(np.float32), mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def tonal(pan, s=4, black=10, white=190, gamma=0.9, sharpen=0.6):
    """Keep original tones (grey skin fills + dark inner lines + white outlines)."""
    up = cv2.resize(pan.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    bl = cv2.GaussianBlur(up, (0, 0), 1.2*s/2)
    up = cv2.GaussianBlur(up, (0, 0), 0.6*s/2)
    up = up + sharpen*(up - bl)
    return np.clip((up - black) / (white - black), 0, 1) ** gamma
