"""
mograph.aefx.deepglow — physically-plausible multi-scale GLOW (inspired by the Deep Glow 2 feature list; own code).

Light is processed in LINEAR float; the glow is a geometric pyramid of gaussians (each octave 2× wider, weight
falloff^i) → the long smooth tail of real lens glow.  Soft-knee THRESHOLD, chromatic ABERRATION (R wider, B
narrower), ASPECT + ANGLE (anamorphic streak), TINT, SATURATION, EXPOSURE, filmic shoulder (hot cores roll to white,
untouched pixels keep their exact value).  Pyramid from 1/4 res: ≈ 0.1-0.2 s on a 1080×1920 frame (2 vCPU).

    from mograph.aefx.deepglow import deep_glow
    out = deep_glow(img_bgr_u8, radius=120, exposure=0.6, threshold=0.55, tint="#7fd0ff")
    out = deep_glow(img, source=emission_bgr, mask=not_face)   # glow only from `source`, only where mask > 0

  radius 90 (px of the widest octave @1080 wide) · exposure 0 (stops) · intensity 1 · threshold 0.6 · knee 0.25
  falloff 0.72 (0.3 tight … 0.95 huge haze) · iterations auto · aspect 1 (>1 horizontal streak) · angle 0
  tint None + tint_amount 1 · saturation 1 · aberration 0..1 · tonemap "filmic"|"reinhard"|"none" · blend add|screen
"""
from __future__ import annotations

import math

import cv2
import numpy as np

LUM = np.array([0.0722, 0.7152, 0.2126], np.float32)     # BGR luminance weights


def _hex(c):
    c = c.lstrip("#")
    return np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], np.float32) / 255.0


def to_lin(x):
    return np.power(np.clip(x, 0, 1), 2.2, dtype=np.float32)


def to_srgb(x):
    return np.power(np.clip(x, 0, None), 1 / 2.2, dtype=np.float32)


def _blur(img, sigma, aspect=1.0, angle=0.0):
    if abs(aspect - 1.0) < 1e-3:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    sx, sy = sigma * math.sqrt(aspect), sigma / math.sqrt(aspect)
    if abs(angle) < 0.5:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sx, sigmaY=sy)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    r = cv2.GaussianBlur(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), (0, 0), sigmaX=sx, sigmaY=sy)
    return cv2.warpAffine(r, cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0), (w, h),
                          borderMode=cv2.BORDER_REFLECT)


def glow_buffer(src_lin, radius=90.0, falloff=0.72, iterations=None, aspect=1.0, angle=0.0, aberration=0.0,
                ref_w=1080.0):
    """Multi-octave glow of a linear float image → linear glow (energy-normalised)."""
    H, W = src_lin.shape[:2]
    rad = max(2.0, float(radius) * W / ref_w)
    n = int(iterations) if iterations else max(3, min(7, int(math.log2(rad / 3.0)) + 1))
    q0 = 4 if W >= 720 else 2
    base = cv2.resize(src_lin, (max(1, W // q0), max(1, H // q0)), interpolation=cv2.INTER_AREA)
    bh, bw = base.shape[:2]
    acc = np.zeros_like(base)
    wsum = 0.0
    cur = base
    for i in range(n):
        sig_full = rad / (2 ** (n - 1 - i))                   # octave sigma in full-res px (widest = rad)
        q = 2 ** max(0, min(i, int(math.log2(max(1.0, sig_full / q0 / 3.0)))))
        small = cv2.resize(base, (max(1, bw // q), max(1, bh // q)), interpolation=cv2.INTER_AREA) if q > 1 else base
        sig = max(0.6, sig_full / q0 / q)
        if aberration > 0:
            b = np.stack([_blur(np.ascontiguousarray(small[..., c]), sig * k, aspect, angle)
                          for c, k in zip(range(3), (1 - 0.35 * aberration, 1.0, 1 + 0.45 * aberration))], -1)
        else:
            b = _blur(small, sig, aspect, angle)
        if q > 1:
            b = cv2.resize(b, (bw, bh), interpolation=cv2.INTER_LINEAR)
        w = falloff ** (n - 1 - i)                            # tight octaves strongest
        acc += b * w
        wsum += w
    del cur
    return cv2.resize(acc / max(1e-6, wsum), (W, H), interpolation=cv2.INTER_LINEAR)


def _tm(x, mode):
    if mode == "reinhard":
        return x / (1.0 + x)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14              # ACES fit (Narkowicz)
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def shoulder(x, mode="filmic", knee=0.8):
    """identity below `knee`, smooth roll-off to 1.0 above it."""
    if mode == "none":
        return x
    over = np.maximum(x - knee, 0)
    return np.where(x > knee, knee + (1 - knee) * _tm(over / (1 - knee), mode) / _tm(np.float32(1e4), mode), x)


def deep_glow(img, radius=90.0, exposure=0.0, intensity=1.0, threshold=0.6, knee=0.25, falloff=0.72,
              iterations=None, aspect=1.0, angle=0.0, tint=None, tint_amount=None, saturation=1.0, aberration=0.0,
              tonemap="filmic", blend="add", source=None, mask=None):
    """Glow img (BGR/BGRA, uint8 or float 0..1) → same dtype/shape."""
    u8 = img.dtype == np.uint8
    f = img.astype(np.float32) / 255.0 if u8 else img.astype(np.float32)
    rgb = f[..., :3]
    if source is None:
        src = rgb
    else:
        src = (source.astype(np.float32) / 255.0 if source.dtype == np.uint8 else source.astype(np.float32))[..., :3]
    lin = to_lin(src)
    L = lin @ LUM
    th = float(threshold) ** 2.2 if threshold > 0 else 0.0
    kn = max(1e-4, float(knee) * max(th, 0.05))
    soft = np.clip(L - th + kn, 0, 2 * kn)
    soft = soft * soft / (4 * kn)
    hi = lin * np.clip(np.maximum(soft, L - th) / np.maximum(L, 1e-5), 0, None)[..., None]
    g = glow_buffer(hi, radius, falloff, iterations, aspect, angle, aberration)
    if saturation != 1.0:
        gm = g.mean(-1, keepdims=True)
        g = np.clip(gm + (g - gm) * float(saturation), 0, None)
    if tint:
        ta = 1.0 if tint_amount is None else float(tint_amount)
        g = g * (1 - ta) + g.max(-1, keepdims=True) * to_lin(_hex(tint)) * ta
    g *= float(intensity) * (2.0 ** float(exposure)) * 3.0
    if mask is not None:
        g *= np.asarray(mask, np.float32)[..., None]
    base = to_lin(rgb)
    out = 1 - (1 - base) * (1 - np.clip(g, 0, 1)) if blend == "screen" else base + g
    out = shoulder(out, tonemap)
    res = to_srgb(np.clip(out, 0, 1))
    if f.shape[-1] == 4:
        res = np.concatenate([res, np.maximum(f[..., 3:], res.max(-1, keepdims=True))], -1)
    return (np.clip(res, 0, 1) * 255 + 0.5).astype(np.uint8) if u8 else res
