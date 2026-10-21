"""
mograph.v14.flow_cut — BIDIRECTIONAL OPTICAL-FLOW MORPH CUT, GAUSSIAN/L1 STABILIZER & PLANAR TRACKER
(Ported from FilmCraft `crates/render/src/flow.rs` & EffectCraft `crates/track/src/{stabilize.rs, klt.rs}`).

1. `morph_cut_frame(frame_a, frame_b, alpha)`:
   Synthesizes an occlusion-aware optical-flow intermediate frame at `alpha in (0, 1)` between two
   adjacent clip frames across a jump-cut boundary (`R.tl.cuts`).
2. `stabilize_camera_trajectory(transforms, sigma=4.0, max_scale=1.08)`:
   Matsushita (PAMI 2006) Gaussian viewpoint smoothing + Grundmann (CVPR 2011) maximum-scale crop relaxation.
3. `track_planar_homography(prev_bgr, cur_bgr, quad_pts)`:
   Pyramidal Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`) + RANSAC 8-DOF homography (`cv2.findHomography`)
   for Mocha-style 4-corner planar surface tracking.
"""
from __future__ import annotations

import cv2
import numpy as np


def morph_cut_frame(frame_a: np.ndarray, frame_b: np.ndarray, alpha: float, work_w: int = 320) -> np.ndarray:
    """
    Ported from FilmCraft `crates/render/src/flow.rs`:
    Computes bidirectional dense optical flow (`u_01` and `u_10`) at `work_w`, checks forward-backward
    consistency, and warps `frame_a` at `x - alpha * u` and `frame_b` at `x + (1 - alpha) * u`.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.01:
        return frame_a
    if alpha >= 0.99:
        return frame_b
    H, W = frame_a.shape[:2]
    work_h = max(32, int(round(H * work_w / W)))
    ga = cv2.cvtColor(cv2.resize(frame_a, (work_w, work_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(frame_b, (work_w, work_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    # Bidirectional Farneback / Lucas-Kanade dense pyramid flow
    f01 = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    f10 = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Forward-backward consistency weight (fade to cross-dissolve where flow disagrees)
    fb_err = np.hypot(f01[..., 0] + f10[..., 0], f01[..., 1] + f10[..., 1])
    agree = np.exp(-fb_err / 6.0).astype(np.float32)[..., None]

    sx, sy = float(W) / work_w, float(H) / work_h
    u01 = cv2.resize(f01, (W, H), interpolation=cv2.INTER_LINEAR)
    u10 = cv2.resize(f10, (W, H), interpolation=cv2.INTER_LINEAR)
    u01[..., 0] *= sx; u01[..., 1] *= sy
    u10[..., 0] *= sx; u10[..., 1] *= sy
    agree_full = cv2.resize(agree, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wa = cv2.remap(frame_a, xx - alpha * u01[..., 0], yy - alpha * u01[..., 1],
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    wb = cv2.remap(frame_b, xx - (1.0 - alpha) * u10[..., 0], yy - (1.0 - alpha) * u10[..., 1],
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    flow_blend = wa.astype(np.float32) * (1.0 - alpha) + wb.astype(np.float32) * alpha
    cross_fade = frame_a.astype(np.float32) * (1.0 - alpha) + frame_b.astype(np.float32) * alpha
    out = flow_blend * agree_full + cross_fade * (1.0 - agree_full)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def stabilize_camera_trajectory(dx_dy_da: np.ndarray, sigma: float = 4.5, max_shift_px: float = 42.0) -> np.ndarray:
    """
    Ported from EffectCraft `crates/track/src/stabilize.rs`:
    1. Accumulates inter-frame [dx, dy, da_rad] into a cumulative camera trajectory.
    2. Convolves with a centered Gaussian kernel (`Matsushita PAMI 2006`) for zero-lag smoothing.
    3. Relaxes corrections exceeding `max_shift_px` (`Grundmann CVPR 2011` maximum-scale constraint).
    """
    if len(dx_dy_da) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    traj = np.cumsum(dx_dy_da.astype(np.float32), axis=0)
    r = max(1, int(round(sigma * 3)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / max(0.5, sigma)) ** 2)
    k /= k.sum()
    pad = np.pad(traj, ((r, r), (0, 0)), mode="edge")
    smooth_traj = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(3)], axis=1)
    corr = smooth_traj - traj
    shift_mag = np.hypot(corr[:, 0], corr[:, 1])
    scale_relax = np.where(shift_mag > max_shift_px, max_shift_px / np.maximum(shift_mag, 1e-4), 1.0)[:, None]
    return corr * scale_relax


def track_planar_homography(prev_bgr: np.ndarray, cur_bgr: np.ndarray, quad_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ported from EffectCraft `crates/track/src/{klt.rs, fit.rs, solve.rs}`:
    Tracks a 4-corner planar region `quad_pts` (shape 4x2) from `prev_bgr` to `cur_bgr` using
    Shi-Tomasi corners inside the polygon + Pyramidal Lucas-Kanade + RANSAC Homography.
    Returns `(new_quad_pts, H_3x3)`.
    """
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY) if prev_bgr.ndim == 3 else prev_bgr
    g1 = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY) if cur_bgr.ndim == 3 else cur_bgr
    mask = np.zeros_like(g0)
    q_int = np.round(quad_pts).astype(np.int32)
    cv2.fillConvexPoly(mask, q_int, 255)
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=120, qualityLevel=0.01, minDistance=5, mask=mask)
    if p0 is None or len(p0) < 4:
        return quad_pts.copy(), np.eye(3, dtype=np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
    good = (st.reshape(-1) == 1)
    if np.sum(good) < 4:
        return quad_pts.copy(), np.eye(3, dtype=np.float32)
    H_mat, _ = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, 3.0)
    if H_mat is None:
        return quad_pts.copy(), np.eye(3, dtype=np.float32)
    H_f = H_mat.astype(np.float32)
    pts_h = np.hstack([quad_pts.astype(np.float32), np.ones((len(quad_pts), 1), dtype=np.float32)]) @ H_f.T
    new_q = pts_h[:, :2] / np.maximum(1e-6, pts_h[:, 2:3])
    return new_q, H_f


# ====================================================================================== v15.1 renderer integration
def _interp_sim(M, a):
    """Fraction `a` of a 2D similarity transform M (3×3): rotation/scale/translation interpolated."""
    sc = float(np.hypot(M[0, 0], M[1, 0]))
    th = float(np.arctan2(M[1, 0], M[0, 0]))
    s_a = sc ** a
    c, s_ = np.cos(th * a) * s_a, np.sin(th * a) * s_a
    return np.array([[c, -s_, M[0, 2] * a], [s_, c, M[1, 2] * a], [0, 0, 1]], np.float64)


class MorphCuts:
    """Optical-flow MORPH CUT on every jump cut of the EDL (Premiere "Morph Cut" idea) — spec:

        "source": {..., "morph_cuts": true | {"dur": 0.24, "cuts": [1, 3] (indices; default all), "max_motion": 60}}

    Inside a window of `dur` seconds centred on cut c the renderer shows morph_cut_frame(A, B, α) instead of the
    raw source frame, where A = last frame before the cut, B = first frame after it and α = SILK ramp 0 → 1.
    The two frames and BOTH dense flows are computed once per cut (cached) → each frame costs 2 remaps (~25 ms).
    Unreliable flow regions fall back to a GLOBAL similarity warp of both frames (no ghost).
    Cuts whose median flow is larger than `max_motion` px or whose mean flow agreement < `min_agree` are skipped —
    a morph would smear; those keep the hard cut (+ the director's punch-zoom + flash).
    The matte is blended with the same α (A·(1−α) + B·α) so the depth sandwich stays clean."""

    def __init__(self, R, cfg):
        cfg = cfg if isinstance(cfg, dict) else {}
        self.R = R
        self.dur = float(cfg.get("dur", 0.24))
        self.max_motion = float(cfg.get("max_motion", 60.0))
        self.min_agree = float(cfg.get("min_agree", 0.5))       # mean forward/backward flow agreement
        self.switch = float(cfg.get("switch", 0.0))             # colour cross-over width (0 = hard, aligned)
        allow = cfg.get("cuts")
        self.cuts = [(i, c) for i, c in enumerate(R.tl.cuts) if allow is None or i in allow]
        self._cache = {}

    def window(self, t):
        h = self.dur / 2
        for i, c in self.cuts:
            if c - h <= t < c + h:
                return i, (t - (c - h)) / self.dur
        return None

    def _prep(self, i):
        if i in self._cache:
            return self._cache[i]
        R = self.R
        # A = the live frame at the window START, B = the live frame at the window END (both sides of the cut)
        # → the morph joins the live video seamlessly on both edges.  (v15.1 first draft used the last/first
        # frame AT the cut: the video then jumped back 0.12 s when the window ended — measured frame diff 31.)
        c = R.tl.cuts[i]
        h = self.dur / 2
        a_src = min(R.tl.seg[i][1] - 0.5 / R.sfps, R.tl.to_src(c - h))
        b_src = max(R.tl.seg[i + 1][0], R.tl.to_src(min(R.tl.duration - 1e-3, c + h)))
        fa, ia = R.src_frame(a_src)
        fa = fa.copy()
        fb, ib = R.src_frame(b_src)
        fb = fb.copy()
        work_w = 320
        H, W = fa.shape[:2]
        work_h = max(32, int(round(H * work_w / W)))
        ga = cv2.cvtColor(cv2.resize(fa, (work_w, work_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(fb, (work_w, work_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        f01 = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 21, 3, 5, 1.2, 0)
        f10 = cv2.calcOpticalFlowFarneback(gb, ga, None, 0.5, 4, 21, 3, 5, 1.2, 0)
        sx, sy = W / work_w, H / work_h
        med = float(np.median(np.hypot(f01[..., 0] * sx, f01[..., 1] * sy)))
        fb_err = np.hypot(f01[..., 0] + f10[..., 0], f01[..., 1] + f10[..., 1])
        agree_s = np.exp(-fb_err / 2.5).astype(np.float32)
        # GLOBAL similarity transform A→B fitted on reliable flow vectors (the head move between two takes is
        # close to rigid).  v15.1: where the dense flow is unreliable the fallback is this global warp — the
        # v15.0 fallback was a plain cross-dissolve → a visible double image ("ghost") at α = 0.5.
        yy_s, xx_s = np.mgrid[0:work_h:4, 0:work_w:4]
        good = agree_s[::4, ::4] > 0.6
        src = np.column_stack([xx_s[good], yy_s[good]]).astype(np.float32)
        dst = src + f01[::4, ::4][good]
        Mg = None
        if len(src) >= 12:
            Mg, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if Mg is None:
            Mg = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        Mg[:, 2] *= [sx, sy]                                      # to full-res pixels
        agree_mean = float(agree_s.mean())
        ok = med <= self.max_motion and agree_mean >= self.min_agree
        agree = cv2.resize(agree_s, (W, H), interpolation=cv2.INTER_LINEAR)
        u01 = cv2.resize(f01, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([sx, sy], np.float32)
        u10 = cv2.resize(f10, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([sx, sy], np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ent = {"ok": ok, "median_px": round(med, 1), "agree_mean": round(agree_mean, 3), "fa": fa, "fb": fb,
               "ia": ia, "ib": ib, "u01": u01, "u10": u10, "agree": agree[..., None], "xx": xx, "yy": yy,
               "Mg": Mg}
        self._cache = {i: ent}                       # keep only the current cut (RAM: ~4 full-res float planes)
        return ent

    def frame(self, i, alpha):
        """→ (bgr, matte) for cut i at blend α, or None when this cut is skipped."""
        e = self._prep(i)
        if not e["ok"]:
            return None
        from .physics import SILK
        a = float(SILK(min(1.0, max(0.0, alpha))))
        # backward warp: pixel x of the in-between frame came from x − a·u01 in A and x − (1−a)·u10 in B
        wa = cv2.remap(e["fa"], e["xx"] - a * e["u01"][..., 0], e["yy"] - a * e["u01"][..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wb = cv2.remap(e["fb"], e["xx"] - (1 - a) * e["u10"][..., 0], e["yy"] - (1 - a) * e["u10"][..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # v15.1 anti-ghost: the GEOMETRY moves with a (warps above), but the colour switches from A to B over a
        # band of width `switch` around α = .5 (default 0 = a clean switch on one frame) instead of a linear mix — two different mouth / eye poses are
        # never shown half-and-half (that is the classic morph-cut "double face").
        sw = float(self.switch)
        q = float(np.clip((a - 0.5 + sw / 2) / max(1e-6, sw), 0.0, 1.0)) if sw > 0 else float(a >= 0.5)
        ac = q * q * (3.0 - 2.0 * q)                              # colour switch (geometry keeps `a`)
        flow_mix = cv2.addWeighted(wa, 1.0 - ac, wb, ac, 0.0)
        # fallback: both frames moved along the interpolated GLOBAL similarity (A forward by a, B back by 1−a)
        Mg = np.vstack([e["Mg"], [0, 0, 1]])
        Ma = _interp_sim(Mg, a)                                   # maps A → in-between
        Mb = Ma @ np.linalg.inv(Mg)                               # maps B → in-between
        H_, W_ = e["fa"].shape[:2]
        ga = cv2.warpAffine(e["fa"], Ma[:2], (W_, H_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        gb = cv2.warpAffine(e["fb"], Mb[:2], (W_, H_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        cross = cv2.addWeighted(ga, 1.0 - ac, gb, ac, 0.0)
        ag = e["agree"]
        out = (flow_mix.astype(np.float32) * ag + cross.astype(np.float32) * (1 - ag) + 0.5).astype(np.uint8)
        ma = self.R.tracks.matte(e["ia"])
        mb = self.R.tracks.matte(e["ib"])
        return out, (ma * (1.0 - ac) + mb * ac).astype(np.float32)
