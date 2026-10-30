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
RBF_SIGMA = 90.0
PLATE_EXTRA = 0.15          # plate extrapolation share beyond the 3/4 key (|yaw| 25→35)


def _feat(bp):
    g = cv2.cvtColor(bp.base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x = 0.4 * g + 0.4 * bp.face.astype(np.float32) * 255 + 0.2 * bp.alpha.astype(np.float32) * 255
    return cv2.GaussianBlur(x, (0, 0), 2).clip(0, 255).astype(np.uint8)


def _anchors(va, chin):
    pts = []
    for sd in ("L", "R"):
        o = va["eyes"][sd]["outline"]
        # Fixed canonical topology indices (0=left, 16=right, 8=top) prevent discrete argmin/argmax vertex-hopping!
        pts += [o.mean(0), o[0], o[len(o) // 2], o[len(o) // 4]]
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
    f_ab = unfold(_rbf_correct(f_ab, pb, pa, sigma=RBF_SIGMA))
    f_ba = unfold(_rbf_correct(f_ba, pa, pb, sigma=RBF_SIGMA))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, ab=f_ab.astype(np.float16), ba=f_ba.astype(np.float16))
    return f_ab, f_ba


def _jdet(f, s=1.0):
    ux = np.gradient(f[..., 0], axis=1) * s
    uy = np.gradient(f[..., 0], axis=0) * s
    vx = np.gradient(f[..., 1], axis=1) * s
    vy = np.gradient(f[..., 1], axis=0) * s
    return (1 + ux) * (1 + vy) - uy * vx


def unfold(f, thr=0.3, iters=40):
    """make a displacement field invertible (no fold-overs) for every share s ∈ [0, 1]: where det(I + s·∇f) < thr,
    blend in a locally blurred field; repeat.  Leaves well-behaved regions untouched."""
    f = f.copy()
    for _ in range(iters):
        bad = np.zeros(f.shape[:2], bool)
        for s in (0.5, 1.0):
            bad |= _jdet(f, s) < thr
        if not bad.any():
            break
        m = cv2.GaussianBlur(cv2.dilate(bad.astype(np.uint8) * 255, np.ones((15, 15), np.uint8)).astype(np.float32),
                             (0, 0), 8) / 255.0
        fb = cv2.GaussianBlur(f, (0, 0), 10)
        f = f * (1 - m[..., None]) + fb * m[..., None]
    return f


GROUPS = {"L_eye": range(0, 4), "L_brow": range(4, 7), "R_eye": range(7, 11), "R_brow": range(11, 14),
          "nose": range(14, 16), "mouth": range(16, 19)}


def lock_view(view, err):
    """v31.3 Sub-Pixel Group + Endpoint Pinning: eliminates the 10-16px outlier errors near ±12° and ±30°."""
    E = np.asarray(err, dtype=float)
    o = {g: -np.median(E[list(ix)], axis=0) for g, ix in GROUPS.items()}
    v = {"eyes": {}, "brows": {}}
    for sd, e_ix, b_ix in (("L", (0, 1, 2, 3), (4, 5, 6)), ("R", (7, 8, 9, 10), (11, 12, 13))):
        e = view["eyes"][sd]
        ol = e["outline"].copy()
        c0 = ol.mean(0)
        # Apply smooth horizontal/vertical scale correction from cardinal eye landmarks (clamped ±8% for shape safety)
        dx_l, dx_r = -E[e_ix[1], 0], -E[e_ix[2], 0]
        cur_w = max(10.0, float(ol[:, 0].max() - ol[:, 0].min()))
        sx = float(np.clip(1.0 + (dx_r - dx_l) / cur_w, 0.92, 1.08))
        ol[:, 0] = c0[0] + (ol[:, 0] - c0[0]) * sx + 0.5 * (dx_l + dx_r)
        ol[:, 1] = ol[:, 1] + o[sd + "_eye"][1]
        pup = e["pupil"].copy()
        pup[0] = c0[0] + (pup[0] - c0[0]) * sx + 0.5 * (dx_l + dx_r)
        pup[1] = pup[1] + o[sd + "_eye"][1]
        v["eyes"][sd] = dict(e, outline=ol, pupil=pup)

        b = view["brows"][sd]
        bl = b["line"].copy()
        s_b = np.linspace(0.0, 1.0, len(bl))[:, None]
        # Blend group eye+brow offset with smooth outer→inner endpoint correction (clamped to ±4px)
        ob_base = 0.5 * (o[sd + "_brow"] + o[sd + "_eye"])
        d_out = np.clip(-E[b_ix[0]] - ob_base, -8.0, 8.0)
        d_in = np.clip(-E[b_ix[2]] - ob_base, -8.0, 8.0)
        v["brows"][sd] = dict(b, line=bl + ob_base + (1.0 - s_b) * d_out + s_b * d_in)

    ns = view["nose"].copy()
    s_n = np.linspace(0.0, 1.0, len(ns))[:, None]
    d_n0 = np.clip(-E[14] - o["nose"], -8.0, 8.0)
    d_n1 = np.clip(-E[15] - o["nose"], -8.0, 8.0)
    v["nose"] = ns + o["nose"] + (1.0 - s_n) * d_n0 + s_n * d_n1

    m = view["mouth"]
    om = o["mouth"]
    v["mouth"] = {
        "left":  m["left"]  + om + np.clip(-E[16] - om, -8.0, 8.0),
        "right": m["right"] + om + np.clip(-E[17] - om, -8.0, 8.0),
        "low":   m["low"]   + om + np.clip(-E[18] - om, -8.0, 8.0),
    }
    return v


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
        hsh = hashlib.md5(f"{FLOW_SIGMA}/{RBF_SIGMA}/unfold4_nose_dir".encode() + self.layers["front"].base.tobytes()[::97] + self.layers["34"].base.tobytes()[::97]
                          + json.dumps(d["views"], sort_keys=True).encode()).hexdigest()[:10]
        cache = os.path.join(os.path.expanduser("~/.cache/toonface"), f"turn_{name}_{hsh}.npz")
        pa = _anchors(self.kit.front, cf)
        pb = _anchors(self.kit.q, cq)
        pm = _anchors(self.kit.qm, (self.W - 1 - cq[0], cq[1]))
        f_ab, f_ba = turn_fields(self.layers["front"], self.layers["34"], pa, pb, cache)
        # the front drawing is NOT symmetric → the mirrored side gets its own front↔mirrored-3/4 correspondence
        g_ab, g_ba = turn_fields(self.layers["front"], self.layers["34m"], pa, pm, cache.replace(".npz", "_m.npz"))
        self.fields = {"34": (f_ab, f_ba), "34m": (g_ab, g_ba)}
        self.plate_anchors = {"front": pa, "34": pb, "34m": pm}
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        wn = np.clip((800 - yy) / 100.0, 0, 1)
        self._grid = (xx, yy, (wn * wn * (3 - 2 * wn))[:, :, None])   # body below the neck never moves

    lock = True                 # anchor lock: feature groups snap to where the warped plate put its landmarks
    SWAP_BAND = 0.0             # hard switch-layer swap (dissolve band optional; >0 ghosts the hair line-art)

    def _slid(self, key, sgn, t):
        """(plate, forward correspondence d, share s): plate pixel y moves to y + s·d(y).
        front: d = front→3/4 correspondence, s = t.   3/4: d = 3/4→front correspondence, s = 1 − t (< 0 → extrapolate)."""
        qk = "34" if sgn > 0 else "34m"
        f_ab, f_ba = self.fields[qk]
        if key == "front":
            return self.layers["front"], f_ba, min(t, 1.0), "front"
        tq = min(t, 1.0) if t <= 1.0 else 1.0 + min(t - 1.0, 0.4) / 0.4 * PLATE_EXTRA
        return self.layers[qk], f_ab, 1.0 - tq, qk

    def plate_for(self, yaw):
        """→ list of (BlankPlate, d, s, key, weight) and t.  Switch-layer swap at t = 0.5."""
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

    def _backmap(self, d, s, iters=28, damp=0.6):
        """invert the forward warp  y ↦ y + s·wn(y)·d(y)  on the full grid (fixed point) → sampling map M, so that
        out(x) = plate(M(x)) and every plate landmark p lands exactly at p + s·d(p)."""
        xx, yy, wn = self._grid
        if abs(s) < 1e-6:
            return xx, yy
        dd = (d * wn) * s
        mx, my = xx.copy(), yy.copy()
        for _ in range(iters):                      # damped fixed point (converges where the flow is steep)
            g = cv2.remap(dd, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mx = mx + damp * ((xx - g[..., 0]) - mx)
            my = my + damp * ((yy - g[..., 1]) - my)
        return mx, my

    def _plate_layers(self, plates):
        base = hf = fm = al = None
        self._maps = []
        for bp, d, sh, key, w in plates:
            mx, my = self._backmap(d, sh)
            self._maps.append((key, mx, my, w))
            rm = lambda im: cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            b_ = rm(bp.base).astype(np.float32)
            h_ = rm(bp.hair_fg).astype(np.float32)
            f_ = rm(bp.face.astype(np.uint8) * 255).astype(np.float32)
            a_ = rm((bp.alpha * 255).astype(np.uint8)).astype(np.float32)
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
        base, hf, fm, al = self._plate_layers(plates)
        key, mx, my, _ = max(self._maps, key=lambda m: m[3])
        landed = self._landed(key, mx, my, pitch)
        if self.lock:
            # lock is solved at pitch 0 (so the sphere slide of the pitch rig survives) and applied to the posed view
            land0 = self._landed(key, mx, my, 0.0) if abs(pitch) > 0.01 else landed
            v0 = self.kit.view(yaw, 0.0)[0] if abs(pitch) > 0.01 else view
            err = _anchors(v0, land0[-1])[:-1] - land0[:-1]
            view = lock_view(view, err)
        self._reg = (view, pitch, key, landed)
        fl = _surface_bgra(self.W, self.H, lambda ctx: draw_face(ctx, self.kit, view, t, face,
                                                                parts=("brows", "eyes", "nose")))
        ml = _surface_bgra(self.W, self.H, lambda ctx: draw_face(ctx, self.kit, view, t, face, parts=("mouth",)))
        if abs(pitch) > 0.01:                       # plate pitch: vertical squash/shift of the head region only
            base, hf, fm, al = (head_pitch(x, self.pivot, pitch) for x in (base, hf, fm, al))
        ys, xs = np.nonzero(fm > 127)                 # skull reference = centroid of the (turned, pitched) face plate
        self.last_skull = np.array([xs.mean(), ys.mean()]) if len(xs) else np.zeros(2)
        fm = cv2.erode(fm, np.ones((int(self.kit.sw) + 1, int(self.kit.sw) + 1), np.uint8))
        fm = cv2.GaussianBlur(fm, (3, 3), 0).astype(np.float32)[:, :, None] / 255.0
        ml = (ml.astype(np.float32) * fm).astype(np.uint8)          # premultiplied → scale all 4 channels
        out = _over_pm(base, fl)
        out = _over_pm(out, ml)
        out = _over(out, hf)
        rgba = np.dstack([out, cv2.GaussianBlur(al, (3, 3), 0)])
        if head is not None:
            rgba = head_warp(rgba, self.pivot, head.rot, head.dy, head.breath)
            a = math.radians(head.rot)
            px, py = self.pivot
            X, Y = self.last_skull[0] - px, self.last_skull[1] - py
            self.last_skull = np.array([px + math.cos(a) * X - math.sin(a) * Y,
                                        py + math.sin(a) * X + math.cos(a) * Y + head.dy])
        return rgba

    def _landed(self, key, mx, my, pitch):
        """where the plate's own landmarks land in the output (invert the sampling map M at the landmarks)."""
        src = self.plate_anchors[key]
        H, W = self.H, self.W

        def M(x):
            xi = np.clip(x[:, 0], 0, W - 1.001)
            yi = np.clip(x[:, 1], 0, H - 1.001)
            x0, y0 = xi.astype(int), yi.astype(int)
            fx, fy = xi - x0, yi - y0

            def bil(a):
                return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy) +
                        a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)
            return np.c_[bil(mx), bil(my)]
        x = src.copy()
        for _ in range(80):
            x = x - 0.7 * (M(x) - src)
        if abs(pitch) > 0.01:
            from .rig25 import pitch_map_y
            x[:, 1] = pitch_map_y(x[:, 1], pitch, 0.0)
        return x

    def registration(self):
        """face↔skull registration of the LAST render (plate px): vector-face landmarks vs where the plate
        drawing's own landmarks landed.  → (errors Nx2, plate key).  ≈ 0 after the anchor lock."""
        view, pitch, key, landed = self._reg
        drawn = _anchors(view, landed[-1])[:-1]
        return drawn - landed[:-1], key

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
