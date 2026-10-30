"""
toonface.kit.turn25 — 2.5D head-turn compositor: blank plates (front / 3/4 / mirrored 3/4) + rig25 face, all locked
to ONE head anchor frame.

    ch = Character("girl")                       # loads kit + layered plates
    rgba = ch.render(yaw, face, head)            # plate-space RGBA (H×W×4), face already composited + forelock on top

Head anchor lock: the vector face is drawn in PLATE coordinates, composited onto the plate, and only THEN the
head transform (roll / nod / bob / breathing) is applied to the whole image → the face can never float or jitter
relative to the skull.  Gaze changes move pupils/lids only.

Turn: the face features morph continuously (fixed topology).  The plate drawing swaps front ↔ 3/4 at the middle of
the turn (Moho "switch layer"); before/after the swap each plate's head region is slid toward the other key's head
position so the skull never pops, and the face morph tracks it exactly.
"""
from __future__ import annotations

import json
import math
import os

import cairo
import cv2
import numpy as np

from .blankplate import load_blank
from .rig25 import PITCH_CY, PITCH_SHIFT, PITCH_SQUASH, YAW_KEY, YAW_MAX, Face, Kit, draw_face

HERE = os.path.dirname(os.path.abspath(__file__))
PLATES = os.path.normpath(os.path.join(HERE, "../../../../tests/fixtures/toonface/plates"))


def _surface_bgra(w, h, fn):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    fn(ctx)
    surf.flush()
    a = np.ndarray((h, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :w]
    return a.copy()


def _over_pm(dst, src):
    a = src[:, :, 3:4].astype(np.float32) / 255.0
    return (dst.astype(np.float32) * (1 - a) + src[:, :, :3].astype(np.float32)).clip(0, 255).astype(np.uint8)


def _over(dst, bgra):
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    return (dst.astype(np.float32) * (1 - a) + bgra[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def head_slide(img, dx, y_head_end=700, y_body=800):
    """shift the head region horizontally by dx with a smooth falloff over the neck (body stays)."""
    if abs(dx) < 0.05:
        return img
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    w = np.clip((y_body - yy) / float(y_body - y_head_end), 0, 1)
    w = w * w * (3 - 2 * w)
    return cv2.remap(img, xx - dx * w, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def head_pitch(img, pivot, pitch_deg, y_top=120, y_head_end=700, y_body=790):
    """nod the head drawing: the head region is shifted ±(pitch) and foreshortened vertically about the pivot."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    w = np.clip((y_body - yy) / float(y_body - y_head_end), 0, 1)
    w = w * w * (3 - 2 * w)
    s = math.sin(math.radians(pitch_deg))
    k = 1.0 - PITCH_SQUASH * abs(s)                       # vertical foreshortening
    py = PITCH_CY                                 # head centre (shared with rig25.pitch_map_y)
    sy = py + (yy - py) / k - PITCH_SHIFT * s     # pitch down (+) → head content moves down by up to ≈20 px
    sy = yy * (1 - w) + sy * w
    return cv2.remap(img, xx, sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def head_warp(img, pivot, rot_deg, dy, breath, y_head_end=660, y_body=790):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    w = np.clip((y_body - yy) / float(y_body - y_head_end), 0, 1)
    w = w * w * (3 - 2 * w)
    a = -math.radians(rot_deg) * w
    px, py = pivot
    ca, sa = np.cos(a), np.sin(a)
    X = xx - px
    Y = yy - py - dy * w
    sx = px + ca * X - sa * Y
    sy = py + sa * X + ca * Y
    s = 1.0 + 0.005 * breath
    tb = 1 - w
    sy = sy * (1 - tb) + (H - (H - sy) / s) * tb
    return cv2.remap(img, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


FLOW_SIGMA = 28.0
PLATE_EXTRA = 0.15          # plate extrapolation share beyond the 3/4 key (|yaw| 25→35)


def _feat(bp):
    g = cv2.cvtColor(bp.base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x = 0.4 * g + 0.4 * bp.face.astype(np.float32) * 255 + 0.2 * bp.alpha.astype(np.float32) * 255
    return cv2.GaussianBlur(x, (0, 0), 2).clip(0, 255).astype(np.uint8)


def _anchors(va, chin):
    pts = []
    for sd in ("L", "R"):
        o = va["eyes"][sd]["outline"]
        pts += [o.mean(0), o[np.argmin(o[:, 0])], o[np.argmax(o[:, 0])], o[np.argmin(o[:, 1])]]
        bl = va["brows"][sd]["line"]
        pts += [bl[0], bl[len(bl) // 2], bl[-1]]
    pts += [va["nose"][0], va["nose"][-1], va["mouth"]["left"], va["mouth"]["right"], va["mouth"]["low"],
            np.array(chin, float)]
    return np.array(pts, float)


def _rbf_correct(flow, src_pts, dst_pts, sigma=70.0):
    """add a smooth Gaussian-RBF field so that flow(src_pts) == dst_pts - src_pts exactly (anchor pinning)."""
    H, W = flow.shape[:2]
    cur = np.array([flow[int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))] for x, y in src_pts])
    res = (dst_pts - src_pts) - cur
    D = np.linalg.norm(src_pts[:, None] - src_pts[None], axis=2)
    K = np.exp(-(D / sigma) ** 2) + 1e-6 * np.eye(len(src_pts))
    wts = np.linalg.solve(K, res)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = flow.copy()
    for (x, y), wv in zip(src_pts, wts):
        g = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / sigma ** 2))
        out[..., 0] += g * wv[0]
        out[..., 1] += g * wv[1]
    return out


def turn_fields(A, B, pa, pb, cache=None):
    """dense front↔3/4 correspondence for the two plate drawings.
    f_ab: B(x) ≈ A(x + f_ab(x))   f_ba: A(x) ≈ B(x + f_ba(x)); both pinned to the kit anchors."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return z["ab"].astype(np.float32), z["ba"].astype(np.float32)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    dis.setPatchSize(16)
    dis.setGradientDescentIterations(25)
    fA, fB = _feat(A), _feat(B)
    # heavy smoothing: the turn must read as a rigid-ish skull rotation, never as local "melting" of hair curls
    f_ab = cv2.GaussianBlur(dis.calc(fB, fA, None), (0, 0), FLOW_SIGMA)
    f_ba = cv2.GaussianBlur(dis.calc(fA, fB, None), (0, 0), FLOW_SIGMA)
    f_ab = _rbf_correct(f_ab, pb, pa)
    f_ba = _rbf_correct(f_ba, pa, pb)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, ab=f_ab.astype(np.float16), ba=f_ba.astype(np.float16))
    return f_ab, f_ba


def _mirror_field(f):
    m = f[:, ::-1].copy()
    m[..., 0] *= -1
    return m


class Character:
    def __init__(self, name: str):
        self.kit = Kit(name)
        d = self.kit.d
        self.layers = {}
        for v in ("front", "34"):
            img = cv2.imread(os.path.join(PLATES, d["plates"][v]))
            bp = load_blank(img, skin_seed=tuple(d["skin_seed"][v]), contour_cut_y=d["contour_cut_y"][v])
            self.layers[v] = bp
        q = self.layers["34"]
        W = q.base.shape[1]
        from .blankplate import BlankPlate
        self.layers["34m"] = BlankPlate(base=q.base[:, ::-1].copy(), face=q.face[:, ::-1].copy(),
                                        hair_fg=q.hair_fg[:, ::-1].copy(), alpha=q.alpha[:, ::-1].copy(),
                                        skin_bgr=q.skin_bgr)
        self.H, self.W = q.base.shape[:2]
        cf = np.array(d["chin"]["front"], float)
        cq = np.array(d["chin"]["34"], float)
        self.chin_dx = float(cq[0] - cf[0])                 # head offset of the 3/4 drawing vs the front drawing
        self.pivot = (float(np.mean([cf[0], cq[0]])), 735.0)
        import hashlib
        hsh = hashlib.md5(str(FLOW_SIGMA).encode() + self.layers["front"].base.tobytes()[::97] + self.layers["34"].base.tobytes()[::97]
                          + json.dumps(d["views"], sort_keys=True).encode()).hexdigest()[:10]
        cache = os.path.join(os.path.expanduser("~/.cache/toonface"), f"turn_{name}_{hsh}.npz")
        pa = _anchors(self.kit.front, cf)
        pb = _anchors(self.kit.q, cq)
        f_ab, f_ba = turn_fields(self.layers["front"], self.layers["34"], pa, pb, cache)
        self.fields = {"34": (f_ab, f_ba), "34m": (_mirror_field(f_ab), _mirror_field(f_ba))}
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        wn = np.clip((800 - yy) / 100.0, 0, 1)
        self._grid = (xx, yy, (wn * wn * (3 - 2 * wn))[:, :, None])   # body below the neck never moves

    SWAP_BAND = 0.0             # hard switch-layer swap (dissolve band optional; >0 ghosts the hair line-art)

    def _slid(self, key, sgn, t):
        """(plate, warp field) — the drawing is morphed by optical flow toward the other key by its turn share."""
        qk = "34" if sgn > 0 else "34m"
        f_ab, f_ba = self.fields[qk]
        if key == "front":
            return self.layers["front"], f_ab * min(t, 1.0)
        tq = 1.0 + min(max(t - 1.0, 0.0), 0.4) / 0.4 * PLATE_EXTRA
        tq = min(t, 1.0) if t <= 1.0 else tq
        return self.layers[qk], f_ba * (1.0 - tq)          # tq > 1 → extrapolates past the 3/4 drawing

    def plate_for(self, yaw):
        """→ list of (BlankPlate, warp field, weight) and t (turn share).  Switch-layer swap at t = 0.5."""
        t = min(YAW_MAX / YAW_KEY, abs(yaw) / YAW_KEY)
        sgn = 1.0 if yaw >= 0 else -1.0
        b = max(self.SWAP_BAND, 1e-6)
        if t <= 0.5 - b:
            return [(*self._slid("front", sgn, t), 1.0)], t
        if t >= 0.5 + b:
            return [(*self._slid("q", sgn, t), 1.0)], min(t, 1.0)
        w = (t - (0.5 - b)) / (2 * b)
        w = w * w * (3 - 2 * w)
        return [(*self._slid("front", sgn, t), 1 - w), (*self._slid("q", sgn, t), w)], min(t, 1.0)

    def _warp(self, img, f):
        xx, yy, wn = self._grid
        if not np.any(f):
            return img
        ff = f * wn
        return cv2.remap(img, xx + ff[..., 0], yy + ff[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _plate_layers(self, plates):
        base = hf = fm = al = None
        for bp, fld, w in plates:
            b_ = self._warp(bp.base, fld).astype(np.float32)
            h_ = self._warp(bp.hair_fg, fld).astype(np.float32)
            f_ = self._warp(bp.face.astype(np.uint8) * 255, fld).astype(np.float32)
            a_ = self._warp((bp.alpha * 255).astype(np.uint8), fld).astype(np.float32)
            if base is None:
                base, hf, fm, al = b_ * w, h_ * w, f_ * w, a_ * w
            else:
                base += b_ * w; hf += h_ * w; fm += f_ * w; al += a_ * w
        return (base.clip(0, 255).astype(np.uint8), hf.clip(0, 255).astype(np.uint8),
                fm.clip(0, 255).astype(np.uint8), al.clip(0, 255).astype(np.uint8))

    def render(self, yaw: float, face: Face, head=None, pitch: float = 0.0):
        """plate-space RGBA.  yaw ∈ [-35, 35] (beyond ±25 the 3/4 key is extrapolated slightly), pitch ∈ [-20, 20]."""
        yaw = float(np.clip(yaw, -35.0, 35.0))
        pitch = float(np.clip(pitch, -20.0, 20.0))
        view, t = self.kit.view(yaw, pitch)
        plates, t = self.plate_for(yaw)
        fl = _surface_bgra(self.W, self.H, lambda ctx: draw_face(ctx, self.kit, view, t, face,
                                                                parts=("brows", "eyes", "nose")))
        ml = _surface_bgra(self.W, self.H, lambda ctx: draw_face(ctx, self.kit, view, t, face, parts=("mouth",)))
        base, hf, fm, al = self._plate_layers(plates)
        if abs(pitch) > 0.01:                       # plate pitch: vertical squash/shift of the head region only
            base, hf, fm, al = (head_pitch(x, self.pivot, pitch) for x in (base, hf, fm, al))
        fm = cv2.erode(fm, np.ones((int(self.kit.sw) + 1, int(self.kit.sw) + 1), np.uint8))
        fm = cv2.GaussianBlur(fm, (3, 3), 0).astype(np.float32)[:, :, None] / 255.0
        ml = (ml.astype(np.float32) * fm).astype(np.uint8)          # premultiplied → scale all 4 channels
        out = _over_pm(base, fl)
        out = _over_pm(out, ml)
        out = _over(out, hf)
        rgba = np.dstack([out, cv2.GaussianBlur(al, (3, 3), 0)])
        if head is not None:
            rgba = head_warp(rgba, self.pivot, head.rot, head.dy, head.breath)
        return rgba

    def anchor(self, yaw: float, pitch: float = 0.0, head=None):
        """rigid facial-plane anchor (cx, cy, scale, yaw, pitch, roll) in plate px — the frame the face is locked to.
        cx, cy = eye-midpoint after slide/pitch/head transform; scale = inter-ocular distance / front inter-ocular."""
        view, t = self.kit.view(float(np.clip(yaw, -35, 35)), float(np.clip(pitch, -20, 20)))
        eL, eR = view["eyes"]["L"]["outline"].mean(0), view["eyes"]["R"]["outline"].mean(0)
        f = self.kit.front["eyes"]
        io0 = np.linalg.norm(f["L"]["outline"].mean(0) - f["R"]["outline"].mean(0))
        c = 0.5 * (eL + eR)
        roll = 0.0
        if head is not None:
            a = math.radians(head.rot)
            px, py = self.pivot
            X, Y = c[0] - px, c[1] - py
            c = np.array([px + math.cos(a) * X - math.sin(a) * Y, py + math.sin(a) * X + math.cos(a) * Y + head.dy])
            roll = head.rot
        return dict(cx=float(c[0]), cy=float(c[1]), scale=float(np.linalg.norm(eL - eR) / io0), yaw=yaw, pitch=pitch,
                    roll=roll)
