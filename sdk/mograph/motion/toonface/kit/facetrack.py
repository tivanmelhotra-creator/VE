"""
toonface.kit.facetrack — v32 multi-region planar head tracker for blank-face cartoon video (Mocha-style).

Why the old trackers sprang / drifted / tilted:
  * videotrack v31 used the centroid of the skin blob + EMA 0.65 → lag (spring) and jumps whenever a hand (same skin
    colour) touched the face blob; yaw came from blob asymmetry (any hand → fake head turn).
  * sandwich v31.4 used a lower-jaw scan-line anchor (translation only) and a guessed `eye_span_ratio` for size.

v32 tracks the HEAD AS A RIGID PLANE, like a planar tracker in Mocha / AE:
  1. Feature points (Shi-Tomasi) on everything rigidly attached to the skull that has texture: hair curls, ears,
     the ink outline of the face/jaw.  The flat blank face interior has no texture and is never used.
  2. Points are split in 4 REGIONS around the face (crown · left · right · jaw).  Each point keeps its REFERENCE
     coordinate, so every frame solves  ref → current  directly (no accumulated drift).
  3. Pyramidal Lucas-Kanade + forward/backward check, RANSAC similarity, then IRLS refinement with REGION-BALANCED
     Huber weights (every region gets the same vote, so a hand covering the crown cannot drag the face).
  4. One-Euro filter, v34: run forward+backward (zero-phase, 0-frame lag → no spring) + PoseLimits (dead-band so the
     face NEVER rotates when the skull doesn't, clamps, per-frame velocity limits).
  5. Region confidence = inlier ratio per region → occlusion-aware replenishment (never seeds points on a hand).

    trk = HeadTracker(TrackerConfig(fps=24))
    poses = trk.run(frames)          # list[HeadPose2D]  (similarity ref→frame, filtered + limited)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

REGIONS = ("crown", "left", "right", "jaw")


# ================================================================================================= segmentation
def estimate_bg_color(frame: np.ndarray) -> np.ndarray:
    b = 12
    border = np.vstack([frame[:b].reshape(-1, 3), frame[:, :b].reshape(-1, 3), frame[:, -b:].reshape(-1, 3)])
    return np.median(border, axis=0).astype(np.float32)


def estimate_skin_color(frame: np.ndarray, bg_bgr: np.ndarray | None = None) -> np.ndarray:
    """peach skin = largest bright, warm, mid-saturation component in the upper 60 % that is NOT the background.
    (saturation ≥ 72 rejects cream hair; distance to the bg colour rejects the amber wall.)"""
    H = frame.shape[0]
    bg_bgr = estimate_bg_color(frame) if bg_bgr is None else bg_bgr
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    far_bg = np.linalg.norm(frame.astype(np.float32) - bg_bgr[None, None], axis=2) > 45.0
    m = ((hsv[..., 0] >= 4) & (hsv[..., 0] <= 26) & (hsv[..., 1] >= 72) & (hsv[..., 1] <= 150) &
         (hsv[..., 2] >= 225) & far_bg).astype(np.uint8)
    m[int(H * 0.6):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 1:
        return np.array([140, 190, 245], np.float32)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return np.median(frame[lbl == i], axis=0).astype(np.float32)


def segment(frame: np.ndarray, skin_bgr: np.ndarray, bg_bgr: np.ndarray, tol: float = 30.0,
            ink_thr: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (skin, ink, fg) bool masks.  v33: ink = LOCAL contrast (works for dark skin / coloured line-art),
    skin = Lab distance with lightness down-weighted (shading does not break the face)."""
    from .facefind import ink_mask
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(g)
    ink_d = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(np.asarray(skin_bgr, np.float32).reshape(1, 1, 3).astype(np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    d = np.sqrt(0.35 * (lab[..., 0] - ref[0]) ** 2 + (lab[..., 1] - ref[1]) ** 2 + (lab[..., 2] - ref[2]) ** 2)
    skin = (d < tol * 0.5) & ~ink_d
    f = frame.astype(np.float32)
    bg = np.linalg.norm(f - bg_bgr[None, None], axis=2) < 34.0
    n, lbl = cv2.connectedComponents(bg.astype(np.uint8), connectivity=4)
    border = np.unique(np.r_[lbl[0], lbl[:, 0], lbl[:, -1]])
    bgm = np.isin(lbl, border[border > 0])
    return skin, ink, ~bgm


def face_component(skin: np.ndarray, prior: np.ndarray | None = None, seed: tuple[int, int] | None = None,
                   max_outside: float = 0.22) -> np.ndarray:
    """the face = skin components mostly INSIDE the prior face region (a hand component sticks out of it).
    Without a prior: the largest component in the upper 60 % (or the one under `seed`)."""
    H, W = skin.shape
    n, lbl, st, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8), connectivity=4)
    if n <= 1:
        return np.zeros_like(skin)
    if prior is None:
        if seed is not None and lbl[seed[1], seed[0]] > 0:
            return lbl == lbl[seed[1], seed[0]]
        cand = [(st[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if st[i, cv2.CC_STAT_TOP] < H * 0.6]
        return lbl == max(cand)[1] if cand else np.zeros_like(skin)
    keep = np.zeros(n, bool)
    prior_d = cv2.dilate(prior.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    inside = np.bincount(lbl[prior_d], minlength=n)
    area = st[:, cv2.CC_STAT_AREA]
    pa = max(1, int(prior.sum()))
    for i in range(1, n):
        if area[i] < 40 or inside[i] < 0.02 * pa:
            continue
        if (area[i] - inside[i]) / area[i] <= max_outside:
            keep[i] = True
    return keep[lbl]


# ============================================================================================ similarity helpers
def sim_matrix(s: float, th: float, tx: float, ty: float) -> np.ndarray:
    c, d = s * math.cos(th), s * math.sin(th)
    return np.array([[c, -d, tx], [d, c, ty]], np.float64)


def sim_params(M: np.ndarray) -> tuple[float, float, float, float]:
    s = math.hypot(M[0, 0], M[1, 0])
    return s, math.atan2(M[1, 0], M[0, 0]), float(M[0, 2]), float(M[1, 2])


def apply(M: np.ndarray, p) -> np.ndarray:
    p = np.asarray(p, np.float64).reshape(-1, 2)
    return p @ M[:, :2].T + M[:, 2]


def invert(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(M.astype(np.float64))


def weighted_similarity(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    """weighted Umeyama (no reflection) src → dst."""
    w = w / max(w.sum(), 1e-12)
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    a, b = src - ms, dst - md
    var = (w * (a ** 2).sum(1)).sum()
    sxx = (w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])).sum()
    sxy = (w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum()
    th = math.atan2(sxy, sxx)
    s = math.hypot(sxx, sxy) / max(var, 1e-12)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * s
    t = md - R @ ms
    return np.c_[R, t]


# ================================================================================================= filtering
class OneEuro:
    """Casiez et al. 2012 — adaptive low-pass: heavy smoothing at rest, light smoothing when moving fast."""

    def __init__(self, fps: float, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.fps, self.mc, self.beta, self.dc = fps, min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = 0.0

    @staticmethod
    def _a(cutoff, fps):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)

    def __call__(self, x: float) -> float:
        if self.x is None:
            self.x = x
            return x
        dx = (x - self.x) * self.fps
        self.dx += self._a(self.dc, self.fps) * (dx - self.dx)
        cutoff = self.mc + self.beta * abs(self.dx)
        self.x += self._a(cutoff, self.fps) * (x - self.x)
        return self.x


@dataclass
class PoseLimits:
    """explicit, logical limits.  roll_deadband: rotations smaller than this are treated as NO rotation (soft knee),
    so the face never tilts on its own; everything else is clamped to plausible cartoon-head ranges."""
    roll_deadband_deg: float = 1.2
    roll_max_deg: float = 16.0
    scale_min: float = 0.82
    scale_max: float = 1.20
    max_step_px: float = 0.16        # × face width per frame (bigger jump = glitch, limited)
    max_step_roll_deg: float = 5.0


def soft_deadband(x: float, db: float) -> float:
    """exactly 0 inside ±db (no rotation unless the skull really rotates), exactly x beyond ±2·db (no permanent
    under-rotation), C1 smoothstep blend in between (no visible pop when a tilt starts)."""
    a = abs(x)
    if a <= db:
        return 0.0
    if a >= 2.0 * db:
        return x
    t = (a - db) / db
    return x * t * t * (3 - 2 * t)


# ================================================================================================= geometry
@dataclass
class FaceEllipse:
    cx: float
    cy: float
    a: float                       # half width
    b: float                       # half height
    ang: float                     # deg, 0 = upright

    def mask(self, shape, grow: float = 0.0) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        cv2.ellipse(m, ((self.cx, self.cy), (2 * (self.a + grow), 2 * (self.b + grow)), self.ang), 1, -1)
        return m > 0


def fit_face_ellipse(face_filled: np.ndarray) -> FaceEllipse:
    """moment ellipse of the filled face region (robust; a partial-arc fitEllipse extrapolates wildly).
    For a uniform ellipse, half-axis = 2·sqrt(eigenvalue of the covariance)."""
    ys, xs = np.nonzero(face_filled)
    cx, cy = xs.mean(), ys.mean()
    C = np.cov(np.vstack([xs - cx, ys - cy]))
    ev, evec = np.linalg.eigh(C)
    b_ax, a_ax = 2 * math.sqrt(ev[1]), 2 * math.sqrt(ev[0])
    v = evec[:, 1]
    ang = math.degrees(math.atan2(v[1], v[0])) - 90.0
    while ang > 90:
        ang -= 180
    while ang < -90:
        ang += 180
    if abs(ang) > 45:
        a_ax, b_ax = b_ax, a_ax
        ang = ang - 90 if ang > 0 else ang + 90
    return FaceEllipse(float(cx), float(cy), float(a_ax), float(b_ax), float(ang))


# ================================================================================================= tracker
@dataclass
class TrackerConfig:
    fps: float = 24.0
    max_pts_region: int = 70
    min_pts_region: int = 18
    replenish_every: int = 5
    fb_thresh: float = 1.0
    ransac_px: float = 2.5
    huber_px: float = 1.5
    drop_after: int = 3
    ring_k: float = 0.42             # tracking ring = face ellipse grown by this × face width
    pos_min_cutoff: float = 2.2      # One-Euro (Hz) — position
    pos_beta: float = 0.06
    rot_min_cutoff: float = 1.2      # rotation smoothed harder (wobble at the brows is very visible)
    rot_beta: float = 0.25
    scale_min_cutoff: float = 0.8
    scale_beta: float = 0.5
    zero_phase: bool = True          # v34: offline forward+backward smoothing → 0-frame lag (no "spring" face)
    limits: PoseLimits = field(default_factory=PoseLimits)


@dataclass
class HeadPose2D:
    M: np.ndarray                  # 2×3 similarity  reference frame → this frame (filtered + limited)
    M_raw: np.ndarray              # unfiltered RANSAC/IRLS estimate
    scale: float
    roll_deg: float
    roll_raw_deg: float
    center: tuple                  # face-ellipse centre in this frame
    region_conf: dict              # region → inlier ratio 0..1
    n_inliers: int
    residual_px: float             # median inlier residual (attachment quality)


class HeadTracker:
    def __init__(self, cfg: TrackerConfig | None = None, hint=None):
        self.cfg = cfg or TrackerConfig()
        self.hint = hint
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def _hairlike(self, img, face):
        """pixels that belong to the HEAD around the face: hair-coloured, skin-coloured (ears) or line-art.
        Random background clutter is excluded, so static props never vote for the head pose."""
        from .facefind import ink_mask
        st = self.style
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

        def near(bgr, tol):
            r = cv2.cvtColor(np.asarray(bgr, np.float32).reshape(1, 1, 3).astype(np.uint8),
                             cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
            return np.linalg.norm(lab - r, axis=2) < tol
        m = near(st.hair_bgr, 26) | near(st.skin_bgr, 20) | ink_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        # keep only what is connected to the face outline (the head mass), not isolated look-alike props
        n, lbl = cv2.connectedComponents((m | face).astype(np.uint8), connectivity=8)
        ids = np.unique(lbl[cv2.dilate(face.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0])
        return np.isin(lbl, ids[ids > 0])

    # ----------------------------------------------------------------------------------------- reference
    def pick_reference(self, frames, skin_bgr, bg_bgr, probe: int = 24) -> int:
        """cleanest frame among the first `probe`: the face blob with the highest solidity (no hand on it)."""
        best, bi = -1.0, 0
        for i in range(0, min(probe, len(frames)), 2):
            skin, _, _ = segment(frames[i], skin_bgr, bg_bgr)
            f = face_component(skin)
            if f.sum() < 500:
                continue
            cs, _ = cv2.findContours(f.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea)
            sol = cv2.contourArea(c) / max(1.0, cv2.contourArea(cv2.convexHull(c)))
            if sol > best + 0.004:
                best, bi = sol, i
        return bi

    def _region_masks(self, ell: FaceEllipse, fg, face_f, hands) -> dict:
        """ring around the face (hair, ears, outline) split by angle, excluding the textureless face interior,
        the body below the chin and any non-face skin (hands)."""
        shape = fg.shape
        W = 2 * ell.a
        ring = ell.mask(shape, grow=self.cfg.ring_k * W) & fg
        inner = cv2.erode(face_f.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        ring &= ~inner
        ring &= ~(cv2.dilate(hands.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        ring &= yy < ell.cy + ell.b + 0.03 * W
        ang = np.degrees(np.arctan2(yy - ell.cy, xx - ell.cx))
        return {
            "crown": ring & (ang >= -150) & (ang < -30),
            "right": ring & (ang >= -30) & (ang < 50),
            "jaw": ring & (ang >= 50) & (ang < 130),
            "left": ring & ((ang >= 130) | (ang < -150)),
        }

    @staticmethod
    def _detect(gray, mask, n, exclude_pts=None):
        m = mask.astype(np.uint8) * 255
        if exclude_pts is not None and len(exclude_pts):
            for p in exclude_pts:
                cv2.circle(m, (int(p[0]), int(p[1])), 7, 0, -1)
        if n <= 0 or m.max() == 0:
            return np.zeros((0, 2), np.float32)
        p = cv2.goodFeaturesToTrack(gray, maxCorners=n, qualityLevel=0.01, minDistance=7, mask=m, blockSize=7)
        return np.zeros((0, 2), np.float32) if p is None else p.reshape(-1, 2).astype(np.float32)

    # ----------------------------------------------------------------------------------------- solve
    def _solve(self, ref, cur, reg):
        cfg = self.cfg
        if len(ref) < 6:
            return None, np.zeros(len(ref), bool)
        M, inl = cv2.estimateAffinePartial2D(ref, cur, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_px,
                                             maxIters=2000, confidence=0.995, refineIters=10)
        if M is None:
            return None, np.zeros(len(ref), bool)
        for _ in range(6):                                  # IRLS, region-balanced Huber
            r = np.linalg.norm(apply(M, ref) - cur, axis=1)
            hub = np.where(r <= cfg.huber_px, 1.0, cfg.huber_px / np.maximum(r, 1e-9))
            hub[r > 3 * cfg.ransac_px] = 0.0
            w = np.zeros(len(ref))
            for k in range(len(REGIONS)):
                sel = reg == k
                if sel.any():           # equal vote per region, scaled down when the region is sparse
                    w[sel] = hub[sel] / max(hub[sel].sum(), 1e-9) * min(1.0, sel.sum() / cfg.min_pts_region)
            if w.sum() <= 0:
                break
            M = weighted_similarity(ref.astype(np.float64), cur.astype(np.float64), w)
        r = np.linalg.norm(apply(M, ref) - cur, axis=1)
        return M, r < cfg.ransac_px * 1.2

    def _ell_at(self, M):
        e = self.ref_ell
        s, th, _, _ = sim_params(M)
        c = apply(M, [[e.cx, e.cy]])[0]
        return FaceEllipse(float(c[0]), float(c[1]), e.a * s, e.b * s, e.ang + math.degrees(th))

    # ----------------------------------------------------------------------------------------- main
    def run(self, frames: list, ref_idx: int | None = None, skin_bgr=None, bg_bgr=None):
        cfg = self.cfg
        from .facefind import find_face, measure_style
        bg_bgr = estimate_bg_color(frames[0]) if bg_bgr is None else bg_bgr
        if skin_bgr is None:                      # v33: structure-based (any skin colour, busy backgrounds)
            ff0 = find_face(frames[0 if ref_idx is None else ref_idx], hint=self.hint)
            skin_bgr = ff0.skin_bgr
        self.skin_bgr, self.bg_bgr = skin_bgr, bg_bgr
        if ref_idx is None:
            ref_idx = self.pick_reference(frames, skin_bgr, bg_bgr)
        self.ref_idx = ref_idx
        rf = frames[ref_idx]
        skin, _, fg = segment(rf, skin_bgr, bg_bgr)
        try:
            face_f = find_face(rf, hint=self.hint).mask
        except RuntimeError:
            face_f = ndi.binary_fill_holes(face_component(skin))
        self.ref_face = face_f
        self.style = measure_style(rf, face_f)
        fg = fg & self._hairlike(rf, face_f)
        self.ref_ell = ell = fit_face_ellipse(face_f)
        self.face_w = 2 * ell.a
        rmasks = self._region_masks(ell, fg, face_f, skin & ~face_f)
        self.ref_region_masks = rmasks
        g_ref = cv2.cvtColor(rf, cv2.COLOR_BGR2GRAY)
        H, W = g_ref.shape

        raw = {}
        for order in (list(range(ref_idx, len(frames))), list(range(ref_idx, -1, -1))):
            pts, refp, reg = [], [], []
            for k, rn in enumerate(REGIONS):
                p = self._detect(g_ref, rmasks[rn], cfg.max_pts_region)
                pts.append(p)
                refp.append(p.copy())
                reg.append(np.full(len(p), k))
            pts, refp, reg = np.vstack(pts), np.vstack(refp), np.concatenate(reg)
            bad = np.zeros(len(pts), int)
            prev_g = g_ref
            M = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
            for n_i, i in enumerate(order):
                g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                if i != ref_idx and len(pts):
                    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, pts.reshape(-1, 1, 2), None, **self.lk)
                    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev_g, p1, None, **self.lk)
                    fb = np.linalg.norm(p0b.reshape(-1, 2) - pts, axis=1)
                    ok = (st1.ravel() > 0) & (st2.ravel() > 0) & (fb < cfg.fb_thresh)
                    pts, refp, reg, bad = p1.reshape(-1, 2)[ok], refp[ok], reg[ok], bad[ok]
                    M_new, inl = self._solve(refp, pts, reg)
                    if M_new is not None:
                        M = M_new
                        bad = np.where(inl, 0, bad + 1)
                        keep = bad < cfg.drop_after
                        pts, refp, reg, bad, inl = pts[keep], refp[keep], reg[keep], bad[keep], inl[keep]
                    else:
                        inl = np.zeros(len(pts), bool)
                else:
                    inl = np.ones(len(pts), bool)
                r = np.linalg.norm(apply(M, refp) - pts, axis=1) if len(pts) else np.zeros(0)
                conf = {}
                for k, rn in enumerate(REGIONS):
                    sel = reg == k
                    conf[rn] = float(inl[sel].mean()) * min(1.0, sel.sum() / cfg.min_pts_region) if sel.any() else 0.0
                res = float(np.median(r[inl])) if inl.any() else 99.0
                raw[i] = (M.copy(), conf, int(inl.sum()), res)
                # replenish only on healthy frames, never on a hand
                good = sum(v > 0.6 for v in conf.values()) >= 3 and res < 1.6
                if n_i % cfg.replenish_every == 0 and i != ref_idx and good:
                    sk, _, fgi = segment(frames[i], skin_bgr, bg_bgr)
                    prior = cv2.warpAffine(face_f.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
                    fc = face_component(sk, prior=prior)
                    masks_i = self._region_masks(self._ell_at(M), fgi, prior | fc, sk & ~prior & ~fc)
                    Minv = invert(M)
                    add_p, add_r, add_g = [], [], []
                    for k, rn in enumerate(REGIONS):
                        have = int((reg == k).sum())
                        if have >= cfg.min_pts_region * 2:
                            continue
                        newp = self._detect(g, masks_i[rn], cfg.max_pts_region - have, exclude_pts=pts)
                        if len(newp):
                            add_p.append(newp)
                            add_r.append(apply(Minv, newp).astype(np.float32))
                            add_g.append(np.full(len(newp), k))
                    if add_p:
                        pts = np.vstack([pts] + add_p).astype(np.float32)
                        refp = np.vstack([refp] + add_r).astype(np.float32)
                        reg = np.concatenate([reg] + add_g)
                        bad = np.concatenate([bad, np.zeros(sum(len(a) for a in add_p), int)])
                prev_g = g
        self.raw = raw
        return self._filter([raw[i] for i in range(len(frames))])

    # ----------------------------------------------------------------------------------------- filter + limits
    def _filter(self, raw):
        if self.cfg.zero_phase:
            return self._filter_zero_phase(raw)
        return self._filter_causal(raw)

    def _limited(self, raw):
        """per-frame limits on the RAW estimate (dead-band, clamps, glitch step limit) — no smoothing, no lag."""
        lim = self.cfg.limits
        e = self.ref_ell
        c0 = np.array([e.cx, e.cy])
        out, prev = [], None
        for (M, conf, ninl, res) in raw:
            s, th, _, _ = sim_params(M)
            c = apply(M, [c0])[0]
            roll_raw = math.degrees(th)
            roll = float(np.clip(soft_deadband(roll_raw, lim.roll_deadband_deg), -lim.roll_max_deg, lim.roll_max_deg))
            s = float(np.clip(s, lim.scale_min, lim.scale_max))
            if prev is not None:
                step = lim.max_step_px * self.face_w
                c = prev[0] + np.clip(c - prev[0], -step, step)
                roll = prev[1] + float(np.clip(roll - prev[1], -lim.max_step_roll_deg, lim.max_step_roll_deg))
            prev = (c, roll)
            out.append((float(c[0]), float(c[1]), roll, math.log(s), roll_raw))
        return np.array(out)

    @staticmethod
    def _zero_phase(x: np.ndarray, fps: float, mc: float, beta: float) -> np.ndarray:
        """One-Euro run forward AND backward, averaged: the forward lag and the backward lead cancel exactly
        (zero-phase), while the adaptive cutoff still keeps rest frames calm.  Offline only (needs all frames)."""
        f, b = OneEuro(fps, mc, beta), OneEuro(fps, mc, beta)
        fw = np.array([f(float(v)) for v in x])
        bw = np.array([b(float(v)) for v in x[::-1]])[::-1]
        return 0.5 * (fw + bw)

    def _filter_zero_phase(self, raw):
        cfg = self.cfg
        e = self.ref_ell
        c0 = np.array([e.cx, e.cy])
        L = self._limited(raw)
        cx = self._zero_phase(L[:, 0], cfg.fps, cfg.pos_min_cutoff, cfg.pos_beta)
        cy = self._zero_phase(L[:, 1], cfg.fps, cfg.pos_min_cutoff, cfg.pos_beta)
        rl = self._zero_phase(L[:, 2], cfg.fps, cfg.rot_min_cutoff, cfg.rot_beta)
        ls = self._zero_phase(L[:, 3], cfg.fps, cfg.scale_min_cutoff, cfg.scale_beta)
        rl = np.where(np.abs(L[:, 2]) < 1e-9, np.where(np.abs(rl) < 0.15, 0.0, rl), rl)   # keep dead-band level
        out = []
        for i, (M, conf, ninl, res) in enumerate(raw):
            s_f = math.exp(ls[i])
            Mf = sim_matrix(s_f, math.radians(rl[i]), 0.0, 0.0)
            Mf[:, 2] = np.array([cx[i], cy[i]]) - Mf[:, :2] @ c0
            out.append(HeadPose2D(M=Mf, M_raw=M, scale=s_f, roll_deg=float(rl[i]), roll_raw_deg=float(L[i, 4]),
                                  center=(float(cx[i]), float(cy[i])), region_conf=conf, n_inliers=ninl,
                                  residual_px=res))
        return out

    def _filter_causal(self, raw):
        cfg, lim = self.cfg, self.cfg.limits
        e = self.ref_ell
        c0 = np.array([e.cx, e.cy])
        fx = OneEuro(cfg.fps, cfg.pos_min_cutoff, cfg.pos_beta)
        fy = OneEuro(cfg.fps, cfg.pos_min_cutoff, cfg.pos_beta)
        fr = OneEuro(cfg.fps, cfg.rot_min_cutoff, cfg.rot_beta)
        fs = OneEuro(cfg.fps, cfg.scale_min_cutoff, cfg.scale_beta)
        out, prev = [], None
        for (M, conf, ninl, res) in raw:
            s, th, _, _ = sim_params(M)
            c = apply(M, [c0])[0]
            roll_raw = math.degrees(th)
            roll = float(np.clip(soft_deadband(roll_raw, lim.roll_deadband_deg), -lim.roll_max_deg, lim.roll_max_deg))
            s = float(np.clip(s, lim.scale_min, lim.scale_max))
            if prev is not None:
                step = lim.max_step_px * self.face_w
                c = prev[0] + np.clip(c - prev[0], -step, step)
                roll = prev[1] + float(np.clip(roll - prev[1], -lim.max_step_roll_deg, lim.max_step_roll_deg))
            cx, cy = fx(float(c[0])), fy(float(c[1]))
            roll_f = fr(roll)
            s_f = math.exp(fs(math.log(s)))
            prev = (np.array([cx, cy]), roll_f)
            Mf = sim_matrix(s_f, math.radians(roll_f), 0.0, 0.0)
            Mf[:, 2] = np.array([cx, cy]) - Mf[:, :2] @ c0
            out.append(HeadPose2D(M=Mf, M_raw=M, scale=s_f, roll_deg=roll_f, roll_raw_deg=roll_raw,
                                  center=(cx, cy), region_conf=conf, n_inliers=ninl, residual_px=res))
        return out
