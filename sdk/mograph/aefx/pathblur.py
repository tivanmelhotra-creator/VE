"""
mograph.aefx.pathblur — blur an image ALONG A PATH (inspired by the PathBlurFX control set; own implementation).

Every pixel is smeared along the path's local direction (a vector field built from the polyline), with the blur
length varying along the path: start_blur → end_blur (+ taper curve), soft fall-off around the path (start/end
fall-off = distance from the path where the effect fades), bidirectional or one-sided ("trail" behind the motion).
Use it for: speed trails behind a moving title/object along its route, swoosh arcs, "comet" smears, curved zoom
streaks, and directional transitions along a swipe path.

    from mograph.aefx.pathblur import path_blur
    out = path_blur(img, path=[(100, 900), (500, 700), (980, 900)], start_blur=0, end_blur=80,
                    falloff=220, bidirectional=False, samples=16)
  path            list of (x, y) px (≥ 2 points; smoothed with Catmull-Rom)  — or "radial"/"spin" with center
  start_blur      blur length (px) at the path start          end_blur  … at the path end
  taper           exponent of the start→end ramp (1 linear, 2 ease-in, 0.5 ease-out)
  falloff         px distance from the path where the effect reaches 0 (None = whole frame follows the field)
  falloff_power   shape of that fade (default 1.5)
  bidirectional   True = symmetric smear ; False = trail BEHIND (opposite to path direction) → motion trail
  reverse         flip the path direction
  samples         taps per pixel (8 fast … 32 silky; auto-capped by blur length)
  dynamic         True = taps ∝ local blur length (Dynamic Sampling)
  ignore_alpha    BGRA: blur colour only, keep alpha
  mask            optional float HxW where it applies
Resolution independent: px values refer to the real frame size.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _catmull(pts, n=24):
    P = np.asarray(pts, np.float32)
    if len(P) < 3:
        t = np.linspace(0, 1, n * max(1, len(P) - 1))[:, None]
        return P[0] + (P[-1] - P[0]) * t
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.asarray(out, np.float32)


def path_field(shape, path, q=4):
    """For a polyline: per-pixel unit direction (dx, dy), distance to path d, and arc-length fraction u (0..1).
    Computed on a 1/q grid (nearest path sample) and upsampled — smooth and fast."""
    H, W = shape[:2]
    P = _catmull(path)
    seg = np.diff(P, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(L)])
    tot = max(1e-6, cum[-1])
    tang = np.vstack([seg, seg[-1:]])
    tang /= np.maximum(1e-6, np.linalg.norm(tang, axis=1, keepdims=True))
    h, w = max(2, H // q), max(2, W // q)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    X = (xs + 0.5) * W / w
    Y = (ys + 0.5) * H / h
    best_d = np.full((h, w), np.inf, np.float32)
    best_i = np.zeros((h, w), np.int32)
    step = max(1, len(P) // 400)
    for i in range(0, len(P), step):
        d = (X - P[i, 0]) ** 2 + (Y - P[i, 1]) ** 2
        m = d < best_d
        best_d[m] = d[m]
        best_i[m] = i
    dx = tang[best_i, 0]
    dy = tang[best_i, 1]
    u = cum[best_i] / tot
    # smooth the direction field so the smear bends continuously
    dx = cv2.GaussianBlur(dx, (0, 0), 2.0)
    dy = cv2.GaussianBlur(dy, (0, 0), 2.0)
    n = np.maximum(1e-6, np.sqrt(dx * dx + dy * dy))
    up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)  # noqa: E731
    return up(dx / n), up(dy / n), up(np.sqrt(best_d)), up(u.astype(np.float32))


def radial_field(shape, center, spin=False):
    H, W = shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    vx, vy = xs - center[0], ys - center[1]
    r = np.maximum(1e-3, np.sqrt(vx * vx + vy * vy))
    if spin:
        vx, vy = -vy, vx
    return vx / r, vy / r, np.zeros((H, W), np.float32), np.clip(r / (0.5 * math.hypot(W, H)), 0, 1)


def path_blur(img, path, start_blur=0.0, end_blur=60.0, taper=1.0, falloff=None, falloff_power=1.5,
              bidirectional=True, reverse=False, samples=16, dynamic=True, ignore_alpha=False, mask=None,
              center=None, field=None):
    H, W = img.shape[:2]
    if field is None:
        if isinstance(path, str):
            c = center or (W / 2, H / 2)
            field = radial_field(img.shape, c, spin=(path == "spin"))
        else:
            field = path_field(img.shape, path)
    dx, dy, dist, u = field
    if reverse:
        dx, dy, u = -dx, -dy, 1 - u
    blen = start_blur + (end_blur - start_blur) * np.power(np.clip(u, 0, 1), max(0.05, float(taper)))
    if falloff:
        k = np.clip(1 - dist / float(falloff), 0, 1) ** float(falloff_power)
        blen = blen * k
    if mask is not None:
        blen = blen * np.asarray(mask, np.float32)
    mx = float(blen.max())
    if mx < 0.75:
        return img
    if dynamic:                                   # Dynamic Sampling: ≥ 1 tap per px of smear (no comb artefacts),
        n = int(min(64, max(int(samples), mx)))   # capped at 64; then a light pre-blur hides the rest
    else:
        n = int(samples)
    if mx / max(1, n - 1) > 1.2:                  # under-sampled → pre-soften along x/y by the tap spacing
        sig = 0.5 * mx / max(1, n - 1)
        img = cv2.GaussianBlur(img, (0, 0), sig)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    src = img.astype(np.float32)
    acc = np.zeros_like(src)
    wsum = 0.0
    for i in range(n):
        s = (i / max(1, n - 1))
        off = (s - 0.5) if bidirectional else -s                  # trail lies BEHIND the travel direction
        w = 1.0 if bidirectional else (1.0 - 0.6 * s)             # one-sided: comet tail fades
        mxp = (xs + dx * blen * off).astype(np.float32)
        myp = (ys + dy * blen * off).astype(np.float32)
        acc += cv2.remap(src, mxp, myp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * w
        wsum += w
    out = acc / wsum
    if ignore_alpha and img.shape[-1] == 4:
        out[..., 3] = src[..., 3]
    return np.clip(out, 0, 255).astype(img.dtype) if img.dtype == np.uint8 else out.astype(img.dtype)
