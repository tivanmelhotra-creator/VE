"""
toonface.kit.videotrack — v31.3 Moving-Video Head Tracker & 2.5D Rig Compositor.

Tracks a moving cartoon character's head `(cx, cy, scale, roll, yaw, pitch)` across frames of an AI-generated MP4
video (walking, gesturing, turning neck) with EMA temporal smoothing and the Smart Neck-vs-Gaze Clamp (`<= 6px`
drift when neck is stationary), and locks the `rig25` + `turn25` 2.5D vector face onto the moving head.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .rig25 import Face, Kit, draw_face
from .turn25 import Character, _over_pm, _surface_bgra


@dataclass
class TrackedHeadFrame:
    cx: float
    cy: float
    radius: float
    scale: float
    yaw: float
    pitch: float
    roll: float
    skin_bgr: tuple[int, int, int]


def detect_moving_head_pose(
    frame_bgr: np.ndarray,
    prev: TrackedHeadFrame | None = None,
    smooth: float = 0.65,
    ref_radius: float = 195.0,
) -> TrackedHeadFrame:
    """
    Detect the character's head center `(cx, cy)`, radius, scale, and 3D orientation `(yaw, pitch, roll)`
    from a video frame, applying EMA smoothing so the tracked anchor frame never jitters.
    """
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Warm peach/cream skin segmentation bounded by dark ink outlines
    skin_mask = ((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 28) &
                 (hsv[:, :, 1] >= 32) & (hsv[:, :, 1] <= 145) &
                 (hsv[:, :, 2] >= 165) & (gray >= 140)).astype(np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lbl, st, ce = cv2.connectedComponentsWithStats(skin_mask, connectivity=4)

    best_id = None
    best_score = -1.0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < max(600, int(H * W * 0.004)) or y > H * 0.75:
            continue
        aspect = w / float(max(1, h))
        if not (0.55 <= aspect <= 1.65):
            continue
        cx_i, cy_i = float(ce[i][0]), float(ce[i][1])
        score = float(a) * (1.0 - 0.35 * (cy_i / float(H)))
        if prev is not None:
            dist = math.hypot(cx_i - prev.cx, cy_i - prev.cy) / max(20.0, prev.radius)
            score *= math.exp(-0.5 * (dist / 0.85) ** 2)
        if score > best_score:
            best_score = score
            best_id = i

    if best_id is None:
        if prev is not None:
            return prev
        return TrackedHeadFrame(cx=W * 0.5, cy=H * 0.38, radius=ref_radius, scale=1.0,
                                yaw=0.0, pitch=0.0, roll=0.0, skin_bgr=(175, 215, 243))

    comp = (lbl == best_id).astype(np.uint8)
    skin_bgr = tuple(int(v) for v in np.median(frame_bgr[comp > 0], axis=0))
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    if len(hull) >= 5:
        (hcx, hcy), (A, B), ang = cv2.fitEllipse(hull)
        radius = 0.25 * (A + B)
    else:
        (hcx, hcy), radius = cv2.minEnclosingCircle(hull)
    hcx, hcy, radius = float(hcx), float(hcy), max(20.0, float(radius))

    # Estimate yaw from horizontal asymmetry of skin centroid vs hull bounding center
    bx, by, bw, bh = cv2.boundingRect(hull)
    box_cx = bx + 0.5 * bw
    asym_x = (float(ce[best_id][0]) - box_cx) / max(10.0, 0.5 * bw)
    yaw = float(np.clip(asym_x * 42.0, -35.0, 35.0))
    pitch = 0.0
    roll = 0.0
    scale = float(radius / max(20.0, ref_radius))

    cur = TrackedHeadFrame(cx=hcx, cy=hcy, radius=radius, scale=scale,
                           yaw=yaw, pitch=pitch, roll=roll, skin_bgr=skin_bgr)
    if prev is not None and smooth > 0.0:
        k = float(np.clip(smooth, 0.0, 0.92))
        cur = TrackedHeadFrame(
            cx=k * prev.cx + (1 - k) * cur.cx,
            cy=k * prev.cy + (1 - k) * cur.cy,
            radius=k * prev.radius + (1 - k) * cur.radius,
            scale=k * prev.scale + (1 - k) * cur.scale,
            yaw=k * prev.yaw + (1 - k) * cur.yaw,
            pitch=k * prev.pitch + (1 - k) * cur.pitch,
            roll=k * prev.roll + (1 - k) * cur.roll,
            skin_bgr=cur.skin_bgr,
        )
    return cur


def composite_face_on_tracked_frame(
    frame_bgr: np.ndarray,
    tracked: TrackedHeadFrame,
    kit: Kit,
    face_state: Face,
    acting_yaw_delta: float = 0.0,
    max_neck_fixed_px: float = 6.0,
) -> np.ndarray:
    """
    Render the `rig25` vector face locked to `tracked` on `frame_bgr`, enforcing the strict `<= 6px`
    Neck-Fixed Clamp (routing any excess look angle into `face_state.look` pupil gaze).
    """
    max_deg = math.degrees(math.asin(np.clip(max_neck_fixed_px / max(tracked.radius, 25.0), 0.01, 0.12)))
    clamped_dyaw = float(np.clip(acting_yaw_delta, -max_deg, max_deg))
    excess_deg = acting_yaw_delta - clamped_dyaw
    eff_yaw = float(np.clip(tracked.yaw + clamped_dyaw, -35.0, 35.0))

    gx, gy = face_state.look
    gx_eff = float(np.clip(gx + (excess_deg / 25.0) * 0.45, -0.65, 0.65))
    st_eff = Face(
        lid_open=face_state.lid_open, lid_tilt=face_state.lid_tilt, lower=face_state.lower,
        look=(gx_eff, gy), pupil=face_state.pupil,
        brow_up=face_state.brow_up, brow_anger=face_state.brow_anger,
        brow_worry=face_state.brow_worry, furrow=face_state.furrow, bags=face_state.bags,
        mouth=face_state.mouth,
    )

    view, t_turn = kit.view(eff_yaw, tracked.pitch)
    fl = _surface_bgra(kit.W, 1180, lambda ctx: draw_face(ctx, kit, view, t_turn, st_eff))

    # Align kit eye-nose-mouth centroid to tracked `(cx, cy, scale, roll)`
    eL = view["eyes"]["L"]["outline"].mean(0)
    eR = view["eyes"]["R"]["outline"].mean(0)
    src_c = 0.5 * (eL + eR) + np.array([0.0, 38.0])
    s = tracked.scale
    a = math.radians(tracked.roll)
    ca, sa = math.cos(a) * s, math.sin(a) * s
    tx = tracked.cx - (ca * src_c[0] - sa * src_c[1])
    ty = tracked.cy - (sa * src_c[0] + ca * src_c[1])
    M = np.float32([[ca, -sa, tx], [sa, ca, ty]])
    H, W = frame_bgr.shape[:2]
    warped = cv2.warpAffine(fl, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return _over_pm(frame_bgr, warped)
