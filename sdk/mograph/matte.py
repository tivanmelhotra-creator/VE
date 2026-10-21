"""
mograph.matte — v9 offline speaker segmentation (3-tier, automatic) + the depth-layered talking-head rig.

Tier 1  onnx     bundled mograph/models/portrait_seg.onnx (MediaPipe Selfie Segmentation, 462 KB, Apache-2.0)
                 via onnxruntime CPU (~25–40 ms/frame) + guided edge feathering + temporal EMA (0.8·a_t + 0.2·a_{t-1})
Tier 2  chroma   YCbCr distance keyer for green / blue screens with spill suppression (key colour auto-sampled
                 from the frame border, or "key": "#00ff00")
Tier 3  grabcut  cv2.grabCut seeded from haarcascade_frontalface_default + torso geometry (+ MOG2 motion mask)
"matte": "auto" (onnx → chroma if the border is a saturated green/blue → grabcut) | "onnx" | "chroma" | "grabcut"

Mattes are cached per (video, frame index, method) on disk next to the video (<video>.matte/NNNNN.png) so
multi-worker renders and re-renders never pay twice.

ELEMENT talking_head — 3-layer Z-sandwich (HyperFrames "embedded climax behind the subject"):
  (a) plate / backdrop (z < 0)   the original footage (graded) or the chapter backdrop (bg: "keep"|"world")
  (b) behind-speaker climax       "behind": [element dicts] — any element types (kinetic, big_number, headline …),
                                  drawn BETWEEN the backdrop and the speaker, so giant words sit behind head & shoulders
  (c) segmented speaker (z > 0)  with optional "rim": "#ff1e42" glow, and a smooth morph full-frame → circular PIP
                                  avatar ("pip_at": t, "pip": {"x":860,"y":1500,"r":150}) whose ring pulses with the VO
  jump cuts: every splice from `mograph voice` (spec._cuts) gets an editorial punch-in 1.0 ↔ 1.12 ("punch": 0.12)
  {"type":"talking_head","src":"media/host.mp4","x":540,"y":960,"w":1080,"h":1920,"bg":"world","rim":"@accent",
   "behind":[{"type":"kinetic","text":"**کافیه!**","mode":"slam","x":540,"y":720,"size":260}], "pip_at":"w:حالا"}
"""
from __future__ import annotations

import hashlib
import math
import os

import cairo
import cv2
import numpy as np

from .core import clamp, color as _color, ease_in_out, lerp, prog

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(_HERE, "models", "portrait_seg.onnx")
_SESS = None
_SESS_PID = None
_PREV = {}


def onnx_available():
    if not os.path.exists(MODEL):
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _session():
    global _SESS, _SESS_PID
    pid = os.getpid()
    if _SESS is None or _SESS_PID != pid:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        _SESS = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
        _SESS_PID = pid
    return _SESS


def matte_onnx(bgr):
    s = _session()
    inp = s.get_inputs()[0]
    h, w = bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    a = s.run(None, {inp.name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return refine(bgr, np.clip(a, 0, 1))


def refine(bgr, a, radius=None):
    """Edge-aware feathering: a fast guided filter (He et al.) using the luminance as guide."""
    h, w = a.shape
    r = radius or max(2, int(min(h, w) / 160))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    eps = 1e-3
    k = (2 * r + 1, 2 * r + 1)
    mg, ma = cv2.blur(g, k), cv2.blur(a, k)
    cov = cv2.blur(g * a, k) - mg * ma
    var = cv2.blur(g * g, k) - mg * mg
    A = cov / (var + eps)
    B = ma - A * mg
    out = cv2.blur(A, k) * g + cv2.blur(B, k)
    out = np.clip(out, 0, 1)
    # tighten the soft band a little (hair stays soft, body solid)
    return np.clip((out - 0.08) / 0.84, 0, 1).astype(np.float32)


def _border_key(bgr):
    b = np.concatenate([bgr[:8].reshape(-1, 3), bgr[-8:].reshape(-1, 3), bgr[:, :8].reshape(-1, 3),
                        bgr[:, -8:].reshape(-1, 3)])
    med = np.median(b, axis=0)
    hsv = cv2.cvtColor(med.astype(np.uint8)[None, None], cv2.COLOR_BGR2HSV)[0, 0]
    return med, hsv


def is_chroma_plate(bgr):
    """A real green/blue screen: the border is strongly saturated in the key hue AND uniform."""
    b = np.concatenate([bgr[:8].reshape(-1, 3), bgr[-8:].reshape(-1, 3), bgr[:, :8].reshape(-1, 3),
                        bgr[:, -8:].reshape(-1, 3)])
    hsv = cv2.cvtColor(b.astype(np.uint8)[None], cv2.COLOR_BGR2HSV)[0]
    hue, sat, val = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    keyish = (sat > 140) & (val > 70) & (((hue >= 40) & (hue <= 80)) | ((hue >= 100) & (hue <= 125)))
    return keyish.mean() > 0.7 and np.std(hue[keyish]) < 8 if keyish.any() else False


def matte_chroma(bgr, key=None, tol=(28, 70), spill=True):
    if key is None:
        key, _ = _border_key(bgr)
    else:
        c = _color(key)
        key = np.array([c[2] * 255, c[1] * 255, c[0] * 255])
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    kc = cv2.cvtColor(np.uint8([[key]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(np.float32)
    d = np.sqrt((ycc[:, :, 1] - kc[1]) ** 2 + (ycc[:, :, 2] - kc[2]) ** 2)
    a = np.clip((d - tol[0]) / (tol[1] - tol[0]), 0, 1).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 1.0)
    return a


def despill(bgr, key_is_green=True):
    """Clamp the key channel to the average of the other two (classic despill)."""
    f = bgr.astype(np.int16)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    if key_is_green:
        f[:, :, 1] = np.minimum(g, (r + b) // 2 + 10)
    else:
        f[:, :, 0] = np.minimum(b, (r + g) // 2 + 10)
    return np.clip(f, 0, 255).astype(np.uint8)


_CASCADE = None
_MOG = {}


def matte_grabcut(bgr, key=None):
    global _CASCADE
    h, w = bgr.shape[:2]
    sc = 360.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * sc), int(h * sc)))
    sh, sw = small.shape[:2]
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    faces = _CASCADE.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.1, 4, minSize=(20, 20))
    mask = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    if len(faces):
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    else:
        fw = fh = int(sw * 0.28); fx, fy = sw // 2 - fw // 2, int(sh * 0.18)
    cx = fx + fw // 2
    # probable foreground: head + torso trapezoid down to the frame bottom
    pts = np.array([[fx - fw * 0.35, fy - fh * 0.45], [fx + fw * 1.35, fy - fh * 0.45],
                    [fx + fw * 1.4, fy + fh * 1.3], [cx + fw * 1.9, sh], [cx - fw * 1.9, sh],
                    [fx - fw * 0.4, fy + fh * 1.3]], np.int32)
    cv2.fillPoly(mask, [pts], cv2.GC_PR_FGD)
    cv2.ellipse(mask, (cx, fy + fh // 2), (int(fw * 0.32), int(fh * 0.4)), 0, 0, 360, cv2.GC_FGD, -1)
    cv2.rectangle(mask, (int(cx - fw * 0.5), int(fy + fh * 1.5)), (int(cx + fw * 0.5), sh - 1), cv2.GC_FGD, -1)
    if key is not None:                     # motion prior (MOG2) — moving pixels are probably the speaker
        mog = _MOG.setdefault(key, cv2.createBackgroundSubtractorMOG2(120, 24, False))
        mv = mog.apply(small)
        mask[(mv > 0) & (mask == cv2.GC_BGD)] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass
    a = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 1.2)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    return refine(bgr, a)


def matte(bgr, method="auto", key=None, ema=0.8, stream=None):
    """α (float32 HxW 0..1) of the person in a BGR frame.  stream = id for temporal smoothing."""
    m = method
    if m == "auto":
        m = "chroma" if is_chroma_plate(bgr) else ("onnx" if onnx_available() else "grabcut")
    if m == "onnx" and not onnx_available():
        m = "grabcut"
    if m == "chroma":
        a = matte_chroma(bgr, key)
    elif m == "onnx":
        a = matte_onnx(bgr)
    else:
        a = matte_grabcut(bgr, key=stream)
    if stream is not None:
        prev = _PREV.get(stream)
        if prev is not None and prev.shape == a.shape:
            a = (ema * a + (1 - ema) * prev).astype(np.float32)
        _PREV[stream] = a
    return a, m


# ---------------------------------------------------------------------------
# disk cache (frame-exact, worker-safe)
# ---------------------------------------------------------------------------
def cached_matte(path, idx, bgr, method="auto", key=None, prev_alpha=None, ema=0.8):
    d = path + ".matte"
    tag = hashlib.sha1(f"{method}|{key}|{os.path.getsize(path) if os.path.exists(path) else 0}".encode()).hexdigest()[:6]
    f = os.path.join(d, f"{tag}_{idx:05d}.png")
    if os.path.exists(f):
        a = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if a is not None and a.shape == bgr.shape[:2]:
            return a.astype(np.float32) / 255.0
    a, _m = matte(bgr, method, key)
    if prev_alpha is not None and prev_alpha.shape == a.shape:
        a = (ema * a + (1 - ema) * prev_alpha).astype(np.float32)
    try:
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(f, (a * 255).astype(np.uint8))
    except OSError:
        pass
    return a


def premultiplied_surface(bgr, alpha):
    """BGR + α → cairo ARGB32 premultiplied surface (+ keep-alive buffer)."""
    h, w = alpha.shape
    st = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    buf = np.zeros((h, st // 4, 4), np.uint8)
    a8 = (alpha * 255).astype(np.uint16)
    buf[:, :w, :3] = (bgr.astype(np.uint16) * a8[:, :, None] // 255).astype(np.uint8)
    buf[:, :w, 3] = a8.astype(np.uint8)
    return cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, w, h, st), buf


def rim_surface(alpha, color, width=10):
    """Glow ring around the matte edge (outer rim light)."""
    a8 = (alpha * 255).astype(np.uint8)
    dil = cv2.dilate(a8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1)))
    ring = cv2.GaussianBlur(np.clip(dil.astype(np.int16) - a8.astype(np.int16), 0, 255).astype(np.uint8), (0, 0),
                            width * 0.6)
    c = _color(color)
    h, w = ring.shape
    bgr = np.zeros((h, w, 3), np.uint8)
    bgr[:] = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
    return premultiplied_surface(bgr, ring.astype(np.float32) / 255.0)
