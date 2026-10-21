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
