"""Puppet Pin (After Effects style) for raster/vector line-art layers.

Pins: list of (src_xy, dst_xy). Deformation = Moving Least Squares rigid
(Schaefer et al. 2006), the same family of maths behind AE's Puppet tool:
pins move exactly, everything else bends as rigidly as possible.

- `Pin` / `PinSet` with keyframes + easing (AE-like Position Pins)
- `starch` = weight multiplier region that stays rigid (AE "Starch" pins)
- dense backward map evaluated on a coarse grid then upsampled (fast)
- `Layer` = image (+alpha) + mask + pins; `compose` stacks layers back→front
"""
import numpy as np, cv2


# ----------------------------------------------------------------- easing
def smooth(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)

def ease_io(t, p=3.0):
    t = np.clip(t, 0, 1)
    return np.where(t < .5, 0.5 * (2 * t) ** p, 1 - 0.5 * (2 - 2 * t) ** p)

def overshoot(t, s=1.4):
    t = np.clip(t, 0, 1) - 1
    return t * t * ((s + 1) * t + s) + 1

EASE = {'linear': lambda t: np.clip(t, 0, 1), 'smooth': smooth, 'io': ease_io,
        'back': overshoot}


class Track:
    """Keyframed value (scalar or vector). keys: [(t, value, ease_name)]"""
    def __init__(self, keys):
        self.keys = sorted([(k[0], np.asarray(k[1], float), k[2] if len(k) > 2 else 'io')
                            for k in keys], key=lambda k: k[0])

    def __call__(self, t):
        K = self.keys
        if t <= K[0][0]: return K[0][1]
        for (t0, v0, _), (t1, v1, e) in zip(K, K[1:]):
            if t <= t1:
                u = EASE[e]((t - t0) / max(t1 - t0, 1e-6))
                return v0 + (v1 - v0) * u
        return K[-1][1]


# ------------------------------------------------------------ MLS rigid
def mls_rigid_map(src, dst, H, W, step=8, alpha=1.6, eps=1e-6):
    """Backward map for warping: for each output pixel v, where to sample.
    We solve MLS from dst-pins -> src-pins (inverse deformation)."""
    p = np.asarray(dst, np.float64)   # control points in output space
    q = np.asarray(src, np.float64)   # where they come from
    gy, gx = np.mgrid[0:H + step:step, 0:W + step:step].astype(np.float64)
    v = np.stack([gx, gy], -1)                         # (h,w,2)
    d2 = ((v[..., None, :] - p) ** 2).sum(-1)          # (h,w,n)
    w = 1.0 / (d2 ** alpha + eps)
    ws = w.sum(-1, keepdims=True)
    ps = (w[..., None] * p).sum(-2) / ws               # (h,w,2)
    qs = (w[..., None] * q).sum(-2) / ws
    ph = p - ps[..., None, :]                          # (h,w,n,2)
    qh = q - qs[..., None, :]
    vp = v - ps
    phc = ph[..., 0] + 1j * ph[..., 1]
    qhc = qh[..., 0] + 1j * qh[..., 1]
    vpc = vp[..., 0] + 1j * vp[..., 1]
    fr = (w * qhc * np.conj(phc)).sum(-1) * vpc
    n = np.abs(fr); vn = np.abs(vpc)
    fr = np.where(n > 1e-9, fr / np.maximum(n, 1e-9) * vn, 0)
    out = np.stack([fr.real, fr.imag], -1) + qs
    mx = _up(out[..., 0], H, W, step)
    my = _up(out[..., 1], H, W, step)
    return mx, my

def _up(g, H, W, step):
    """corner-aligned bilinear upsample of a grid sampled at multiples of step"""
    xs = (np.arange(W, dtype=np.float32) / step)[None].repeat(H, 0)
    ys = (np.arange(H, dtype=np.float32) / step)[:, None].repeat(W, 1)
    return cv2.remap(g.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def border_pins(roi, spacing=60):
    x0, y0, x1, y1 = roi
    pts = []
    for x in np.linspace(x0, x1, max(2, int((x1 - x0) / spacing) + 1)):
        pts += [(x, y0), (x, y1)]
    for y in np.linspace(y0, y1, max(2, int((y1 - y0) / spacing) + 1))[1:-1]:
        pts += [(x0, y), (x1, y)]
    return pts


def warp_pins(img, src, dst, step=6, alpha=1.6, roi=None, border=60):
    """Warp img with puppet pins (AE Puppet Position Pins).
    roi=(x0,y0,x1,y1) limits computation; ROI border gets fixed pins so the
    warp blends seamlessly into the untouched image."""
    if len(src) == 0: return img
    if roi is not None and border:
        bp = border_pins(roi, border)
        src = list(src) + bp; dst = list(dst) + bp
    if roi is None:
        H, W = img.shape[:2]
        mx, my = mls_rigid_map(src, dst, H, W, step, alpha)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x0, y0, x1, y1 = [int(v) for v in roi]
    out = img.copy()
    o = np.array([x0, y0], float)
    s = [np.asarray(p, float) - o for p in src]; d = [np.asarray(p, float) - o for p in dst]
    mx, my = mls_rigid_map(s, d, y1 - y0, x1 - x0, step, alpha)
    sub = cv2.remap(img, mx + x0, my + y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[y0:y1, x0:x1] = sub
    return out


# ------------------------------------------------------------------ layers
class Layer:
    """img: float32 HxW (line luminance 0..1); alpha: float32 HxW (coverage)."""
    def __init__(self, name, img, alpha, pins=None, roi=None, pre=None):
        self.name, self.img, self.alpha = name, img.astype(np.float32), alpha.astype(np.float32)
        self.pins = pins or {}          # name -> (src_xy, Track or None)
        self.roi = roi
        self.pre = pre                  # optional fn(t, img, alpha) -> img, alpha (face fx)

    def render(self, t):
        img, a = self.img, self.alpha
        if self.pre is not None:
            img, a = self.pre(t, img, a)
        src, dst = [], []
        for nm, (p0, tr) in self.pins.items():
            src.append(p0)
            dst.append(np.asarray(p0, float) + (tr(t) if tr is not None else 0))
        moved = any(np.abs(np.asarray(d) - np.asarray(s)).max() > 0.05 for s, d in zip(src, dst))
        if moved:
            stack = np.dstack([img, a])
            stack = warp_pins(stack, src, dst, roi=self.roi)
            img, a = stack[..., 0], stack[..., 1]
        return img, a


def compose(layers_rgba, bg=0.0):
    out = None
    for img, a in layers_rgba:
        if out is None:
            out = np.full(img.shape, bg, np.float32)
        out = out * (1 - a) + img * a
    return out


def poly_mask(shape, pts, feather=1.5):
    m = np.zeros(shape, np.float32)
    cv2.fillPoly(m, [np.round(np.asarray(pts)).astype(np.int32)], 1.0, cv2.LINE_AA)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m
