"""
mograph.pro.broll — B-ROLL / CUTAWAY / PICTURE-IN-PICTURE / BACKGROUND REPLACE / SPLIT SCREEN / GREEN SCREEN.
The thing every professional editor does in every talking-head video, as ONE v14 layer type:

    {"type": "broll", "src": "media/clinic.mp4" | "media/photo.jpg" | "media/logo.png (alpha)",
     "t": "w:مقاله", "out": "we:چکیده",            # word anchors like every v14 layer
     "mode": "full" | "pip" | "behind" | "split_top" | "split_bottom",
         full    = cutaway: B-roll covers the frame, the voice continues (classic A-roll / B-roll edit)
         pip     = picture-in-picture card (rounded corners + soft drop shadow, follows "anchor" or "x","y")
         behind  = replaces the BACKGROUND behind the speaker (v14 MODNet matte → the speaker stays in front)
         split_top / split_bottom = reel split screen (B-roll in one half, speaker face-crop in the other)
     "in":  "cut" | "fade" | "dip_black" | "dip_white" | "zoom" | "whip_left" | "whip_right" | "slide_up" |
            "push_left" | "luma_wipe" | "circle" | "blur",     "in_dur": 0.3,     (default fade)
     "out_fx": same names,                                      "out_dur": 0.3,
     "kenburns": true | {"from": [1.0, 0.5, 0.5], "to": [1.12, 0.55, 0.45]},   # [zoom, cx, cy] over the layer life
     "src_in": 2.0, "speed": 1.0, "loop": true,   # video playback (src_in = start offset inside the clip)
     "fit": "cover" | "contain",
     "grade": {"look": "teal_orange"},            # its own colour (pro.color) so stock footage matches the A-roll
     "key": {"color": "#00ff00", "tol": 0.2, "soft": 0.1, "spill": 0.6},   # chroma key (green screen)
     "pip": {"w": 0.42, "radius": 34, "shadow": 0.55, "tilt": 0, "aspect": "src" | 1.0 | 0.8},
     "opacity": 1.0,
     "mask": {"shape": "linear", "angle": -12, "feather": 0.03, "invert": true},   # v22 SDF shape mask (pro.masks)
     "mode": "split_diag"  (v22) = full-frame B-roll behind an angled soft linear mask (diagonal comparison split);
             "diag": {"angle": -14, "feather": 0.025, "side": "top" | "bottom", "cy": 0.5}}

Depth inside the v14 compositor:
    behind            → BACK pass (drawn before the speaker composite → her matte occludes it)
    full / split      → FRONT pass with z = -50 (over the speaker, UNDER titles and subtitles)
    pip               → FRONT pass (z 0, set "z" to reorder)
Default SFX: whoosh for whip/zoom transitions, none otherwise ("sfx" overrides).
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

TRANSITIONS = ("cut", "fade", "dip_black", "dip_white", "zoom", "whip_left", "whip_right", "slide_up", "push_left",
               "luma_wipe", "circle", "blur")
MODES = ("full", "pip", "behind", "split_top", "split_bottom", "split_diag")
_IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ============================================================================================== source reader
class Source:
    """random-access frame reader with a tiny cache (video or still image, PNG alpha kept)."""

    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"broll: '{path}' not found")
        self.still = self.alpha = None
        if os.path.splitext(path)[1].lower() in _IMG:
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise ValueError(f"broll: cannot read image '{path}'")
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            if im.shape[2] == 4:
                self.alpha = im[..., 3].astype(np.float32) / 255.0
                im = im[..., :3]
            self.still = np.ascontiguousarray(im)
            self.h, self.w = im.shape[:2]
            self.fps, self.n, self.duration = 30.0, 1, 1e9
            return
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.w, self.h = int(self.cap.get(3)), int(self.cap.get(4))
        self.duration = self.n / self.fps
        self._pos, self._last, self._cache = -1, None, {}

    def frame(self, st, loop=True):
        if self.still is not None:
            return self.still
        i = int(round(st * self.fps))
        i = i % self.n if loop else max(0, min(self.n - 1, i))
        if i in self._cache:
            return self._cache[i]
        if i != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = self.cap.read()
        if not ok:
            fr = self._last if self._last is not None else np.zeros((self.h, self.w, 3), np.uint8)
        self._pos, self._last = i, fr
        if len(self._cache) > 4:
            self._cache.pop(next(iter(self._cache)))
        self._cache[i] = fr
        return fr


_SOURCES = {}


def source(path):
    if path not in _SOURCES:
        _SOURCES[path] = Source(path)
    return _SOURCES[path]


# ============================================================================================== geometry
def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def fit_frame(img, W, H, zoom=1.0, cx=0.5, cy=0.5, fit="cover"):
    """scale img into W×H around (cx, cy) with ken-burns zoom.  → (BGR, validity mask or None)."""
    h, w = img.shape[:2]
    k = (max(W / w, H / h) if fit == "cover" else min(W / w, H / h)) * zoom
    sx, sy = cx * w, cy * h
    if fit == "cover":                                    # keep the crop inside the image
        hw, hh = W / (2 * k), H / (2 * k)
        sx = min(max(sx, hw), w - hw)
        sy = min(max(sy, hh), h - hh)
    M = np.array([[k, 0, W / 2 - k * sx], [0, k, H / 2 - k * sy]], np.float32)
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = None
    if fit != "cover":
        valid = cv2.warpAffine(np.ones((h, w), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    return out, valid, M


def kenburns(L, u):
    kb = L.get("kenburns")
    if not kb:
        return 1.0, 0.5, 0.5
    if kb is True:
        kb = {"from": [1.0, 0.5, 0.5], "to": [1.10, 0.5, 0.47]}
    a, b = kb.get("from", [1, .5, .5]), kb.get("to", [1.1, .5, .5])
    e = u * u * (3 - 2 * u) if kb.get("ease", "smooth") == "smooth" else u
    return tuple(float(a[i] + (b[i] - a[i]) * e) for i in range(3))


# ============================================================================================== keying
def chroma_key(bgr, cfg):
    """→ (alpha float32 0..1, despilled BGR).  Distance in the YCrCb chroma plane (robust to lighting)."""
    col = cfg.get("color", "#00ff00").lstrip("#")
    kb = [int(col[4:6], 16), int(col[2:4], 16), int(col[0:2], 16)]               # key colour as BGR
    kc = cv2.cvtColor(np.uint8([[kb]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(np.float32)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    d = np.sqrt((ycc[..., 1] - kc[1]) ** 2 + (ycc[..., 2] - kc[2]) ** 2) / 255.0
    tol, soft = float(cfg.get("tol", 0.20)), max(1e-3, float(cfg.get("soft", 0.10)))
    a = np.clip((d - tol) / soft, 0, 1).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), float(cfg.get("feather", 0.8)))
    out = bgr
    sp = float(cfg.get("spill", 0.6))
    if sp > 0:                                            # despill: clamp the key channel to the other two
        f = bgr.astype(np.float32)
        ch = int(np.argmax(kb))
        o1, o2 = [c for c in range(3) if c != ch]
        lim = np.maximum(f[..., o1], f[..., o2])
        f[..., ch] -= np.clip(f[..., ch] - lim, 0, None) * sp
        out = np.clip(f, 0, 255).astype(np.uint8)
    return a, out


# ============================================================================================== transitions
def transition_alpha(kind, p, W, H, entering=True):
    """p 0→1 progress.  → (opacity, mask HxW or None, dx, dy, zoom, blur_px)."""
    p = _smooth(p)
    if kind in (None, "cut"):
        return 1.0, None, 0, 0, 1.0, 0
    if kind == "fade":
        return p, None, 0, 0, 1.0, 0
    if kind in ("dip_black", "dip_white"):
        return (0.0 if p < 0.5 else (p - 0.5) * 2), None, 0, 0, 1.0, 0
    if kind == "zoom":
        return min(1.0, p * 1.6), None, 0, 0, 1.0 + (1 - p) * 0.25, (1 - p) * 18
    if kind in ("whip_left", "whip_right"):
        s = -1 if kind == "whip_left" else 1
        return 1.0, None, s * (1 - p) * W * (1 if entering else -1), 0, 1.0, (1 - p) * 40
    if kind == "push_left":
        return 1.0, None, (1 - p) * W, 0, 1.0, 0
    if kind == "slide_up":
        return 1.0, None, 0, (1 - p) * H, 1.0, 0
    if kind == "blur":
        return p, None, 0, 0, 1.0, (1 - p) * 30
    if kind == "circle":
        yy, xx = np.ogrid[0:H, 0:W]
        r = p * math.hypot(W, H) / 2 * 1.05
        m = np.clip((r - np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)) / 20 + 0.5, 0, 1).astype(np.float32)
        return 1.0, m, 0, 0, 1.0, 0
    if kind == "luma_wipe":                               # soft diagonal wipe
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        m = np.clip((p * 1.25 - (xx / W * 0.6 + yy / H * 0.4)) / 0.25, 0, 1).astype(np.float32)
        return 1.0, m, 0, 0, 1.0, 0
    raise KeyError(f"broll: unknown transition '{kind}'.  Use: {', '.join(TRANSITIONS)}")


def dip_color(kind, p):
    """dip_black / dip_white: full-frame colour overlay strength (peaks in the middle of the transition)."""
    if kind not in ("dip_black", "dip_white"):
        return None, 0.0
    return (0 if kind == "dip_black" else 255), 1.0 - abs(_smooth(p) * 2 - 1)


def _state(L, t):
    t0, t1 = L["_t0"], L["_t1"]
    di, do = float(L.get("in_dur", 0.3)), float(L.get("out_dur", 0.3))
    ki, ko = L.get("in", "fade"), L.get("out_fx", "fade")
    di = 0.0 if ki == "cut" else min(di, (t1 - t0) / 2)
    do = 0.0 if ko == "cut" else min(do, (t1 - t0) / 2)
    if di > 0 and t < t0 + di:
        return "in", ki, (t - t0) / di
    if do > 0 and t > t1 - do:
        return "out", ko, (t1 - t) / do
    return "hold", "cut", 1.0


# ============================================================================================== the layer
def _paste(dst, src, x, y, mode="add"):
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    if mode == "max":
        dst[y0:y1, x0:x1] = np.maximum(dst[y0:y1, x0:x1], s)
    elif mode == "set":
        dst[y0:y1, x0:x1] = s
    else:
        dst[y0:y1, x0:x1] = np.clip(dst[y0:y1, x0:x1] + s, 0, 1)


def _rounded(pw, ph, r):
    card = np.zeros((ph, pw), np.uint8)
    r = int(max(0, min(r, pw // 2, ph // 2)))
    cv2.rectangle(card, (r, 0), (pw - r - 1, ph - 1), 255, -1)
    cv2.rectangle(card, (0, r), (pw - 1, ph - r - 1), 255, -1)
    for (cx, cy) in ((r, r), (pw - r - 1, r), (r, ph - r - 1), (pw - r - 1, ph - r - 1)):
        cv2.circle(card, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(card.astype(np.float32) / 255.0, (0, 0), 0.8)


def render_layer(L, F):
    """→ (BGR W×H uint8, alpha W×H float32) or None outside [t0, t1].  Pure numpy."""
    t = F.t
    t0, t1 = L["_t0"], L["_t1"]
    if t < t0 or t > t1:
        return None
    W, H = F.W, F.H
    src = source(L["_src_path"])
    u = (t - t0) / max(1e-3, t1 - t0)
    st = float(L.get("src_in", 0.0)) + (t - t0) * float(L.get("speed", 1.0))
    img = src.frame(st, bool(L.get("loop", True)))
    a_src = src.alpha
    if L.get("key"):
        a_src, img = chroma_key(img, L["key"])
    if L.get("_grader") is not None:
        img = L["_grader"].apply_lut(img)
    mode = L.get("mode", "full")
    z, cx, cy = kenburns(L, u)
    phase, kind, p = _state(L, t)
    op, tmask, dx, dy, tz, tblur = transition_alpha(kind, p, W, H, phase == "in")
    if mode in ("split_top", "split_bottom"):
        hh = H // 2
        pic, _, _ = fit_frame(img, W, hh, z * tz, cx, cy, "cover")
        rgb = np.zeros((H, W, 3), np.uint8)
        A = np.zeros((H, W), np.float32)
        y0 = 0 if mode == "split_top" else H - hh
        rgb[y0:y0 + hh] = pic
        A[y0:y0 + hh] = 1.0
    elif mode == "pip":
        P = dict({"w": 0.42, "radius": 34, "shadow": 0.55, "tilt": 0.0, "aspect": "src"}, **(L.get("pip") or {}))
        pw = int(W * float(P["w"]))
        ph = int(pw * src.h / max(1, src.w)) if P["aspect"] == "src" else int(pw / float(P["aspect"]))
        ph = max(16, min(ph, int(H * 0.55)))
        pic, _, _ = fit_frame(img, pw, ph, z * tz, cx, cy, "cover")
        if L.get("anchor", "screen") != "screen" and getattr(F, "M", None) is not None:
            from ..v14 import layers as LY
            x, y, _k = LY.pos(L, F)
        else:
            x, y = float(L.get("x", W * 0.5)), float(L.get("y", H * 0.30))
        x = min(max(x, pw / 2 + 28), W - pw / 2 - 28)
        y = min(max(y, ph / 2 + 60), H - ph / 2 - 60)
        X0, Y0 = int(x - pw / 2 + dx * 0.25), int(y - ph / 2 + dy * 0.25)
        ca = _rounded(pw, ph, int(P["radius"]))
        rgb = np.zeros((H, W, 3), np.uint8)
        A = np.zeros((H, W), np.float32)
        cardmask = np.zeros((H, W), np.float32)
        _paste(cardmask, ca, X0, Y0, "set")
        sh = float(P["shadow"])
        if sh > 0:                                        # soft drop shadow = depth (no flat box)
            sa = cv2.GaussianBlur(np.pad(ca, 40), (0, 0), 18) * sh
            _paste(A, sa, X0 - 40 + 10, Y0 - 40 + 22)
        pr = np.zeros((H, W, 3), np.uint8)
        hh_, ww_ = pic.shape[:2]
        xa, ya, xb, yb = max(0, X0), max(0, Y0), min(W, X0 + ww_), min(H, Y0 + hh_)
        if xb > xa and yb > ya:
            pr[ya:yb, xa:xb] = pic[ya - Y0:yb - Y0, xa - X0:xb - X0]
        rgb = (pr.astype(np.float32) * cardmask[..., None]).astype(np.uint8)
        A = np.maximum(A * (1 - cardmask), cardmask)
        if float(P.get("tilt", 0)):
            Rm = cv2.getRotationMatrix2D((x, y), float(P["tilt"]), 1.0)
            rgb = cv2.warpAffine(rgb, Rm, (W, H))
            A = cv2.warpAffine(A, Rm, (W, H))
        dx = dy = 0
        tz = 1.0
    else:                                                 # full / behind
        fit = L.get("fit", "cover")
        rgb, valid, M = fit_frame(img, W, H, z * tz, cx, cy, fit)
        A = np.ones((H, W), np.float32) if valid is None else valid
        if a_src is not None:                             # PNG alpha / chroma key travels with the picture
            A = A * cv2.warpAffine(a_src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if mode == "split_diag" or L.get("mask"):              # v22 analytic SDF masks (resolution independent)
        from .masks import mask_alpha
        if mode == "split_diag":
            dg = dict({"angle": -14, "feather": 0.025, "side": "top", "cy": 0.5}, **(L.get("diag") or {}))
            A = A * mask_alpha({"shape": "linear", "angle": dg["angle"], "feather": dg["feather"], "cy": dg["cy"],
                                "invert": dg["side"] == "top"}, W, H, u)
        if L.get("mask"):
            A = A * mask_alpha(L["mask"], W, H, u)
    if dx or dy:
        Mt = np.float32([[1, 0, dx], [0, 1, dy]])
        rgb = cv2.warpAffine(rgb, Mt, (W, H))
        A = cv2.warpAffine(A, Mt, (W, H))
    if tblur > 0.5:
        if kind in ("whip_left", "whip_right"):
            n = int(tblur) * 2 + 1
            rgb = cv2.filter2D(rgb, -1, np.ones((1, n), np.float32) / n)
        else:
            rgb = cv2.GaussianBlur(rgb, (0, 0), tblur / 3)
    if tmask is not None:
        A = A * tmask
    A = A * op * float(L.get("opacity", 1.0))
    if phase != "hold":
        dc, da = dip_color(kind, p)
        if dc is not None and da > 0:
            rgb = (rgb.astype(np.float32) * (1 - da) + dc * da).astype(np.uint8)
            A = np.maximum(A, da)
    return rgb, A.astype(np.float32)


# ============================================================================================== v14 bridge
def prepare(L, R):
    """v14 PREPARE hook: absolute path, own grader, depth defaults."""
    p = L["src"]
    L["_src_path"] = p if os.path.isabs(p) else os.path.join(R.base, p)
    from .color import Grader
    L["_grader"] = Grader(L["grade"], R.base) if L.get("grade") else None
    mode = L.get("mode", "full")
    if mode not in MODES:
        raise ValueError(f"broll: mode '{mode}' unknown ({', '.join(MODES)})")
    L.setdefault("pass", "back" if mode == "behind" else "front")
    if mode in ("full", "split_top", "split_bottom", "split_diag"):
        L.setdefault("z", -50)
    L.setdefault("sfx", "whoosh" if L.get("in", "fade") in ("whip_left", "whip_right", "zoom") else "none")
    return L


def draw_v14(ctx, L, F):
    """v14 DRAW entry: composites the rendered B-roll onto the current cairo surface (premultiplied, alpha 255)."""
    if getattr(F, "prepass", False) or F.surf is None:
        return None
    r = render_layer(L, F)
    if r is None:
        return None
    rgb, A = r
    from ..v14.compositor import surface_array
    surf = F.surf
    surf.flush()
    arr = surface_array(surf)
    if L.get("mode") in ("split_top", "split_bottom"):
        _split_speaker(arr, L, F)
    a = A[..., None]
    arr[:, :, :3] = (rgb.astype(np.float32) * a + arr[:, :, :3].astype(np.float32) * (1 - a) + 0.5).astype(np.uint8)
    surf.mark_dirty()
    return None


def _split_speaker(arr, L, F):
    """split screen: the speaker half is re-framed on her face (crop of the current composite)."""
    H, W = arr.shape[:2]
    hh = H // 2
    try:
        _, fy = F.anchor("face")
    except Exception:  # noqa: BLE001
        fy = H * 0.35
    y0 = int(min(max(fy - hh * 0.42, 0), H - hh))
    crop = arr[y0:y0 + hh, :, :3].copy()
    if L.get("mode") == "split_top":
        arr[H - hh:, :, :3] = crop
    else:
        arr[0:hh, :, :3] = crop
