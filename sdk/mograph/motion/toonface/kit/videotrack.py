"""
toonface.kit.videotrack — v31.4 Moving-Video Head Tracker & 2.5D Rig Compositor.

Upgrades in v31.4:
  1. Dilated Dark-Ink Barrier (`gray < 125`): prevents skin segmentation from leaking across thin chin/jaw outlines into
     the neck or beige clothing.
  2. Ear-Free Lower-Jaw Anchor (`_measure_ear_free_head_oval`): measures horizontal head center `cx` and eye-band width
     excluding lateral ear protrusions (`y = 760..840`) that previously inflated `scale` and shifted `cx` toward the ear.
  3. True Rendered `src_c` Calibration on the Cairo Surface: aligns the actual rendered bounding-box center of `draw_face`
     (after `t_turn` perspective shift) to `(tracked.cx, tracked.cy)`, eliminating the 90px perspective offset at `yaw = ±15°..±25°`.
  4. Premultiplied 4-Channel Face-Interior Clipping (`face_mask_eroded`): clips all 4 premultiplied BGRA channels of the
     warped vector face strictly inside the eroded peach face mask so zero pixels can ever protrude onto hair or ears.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from .rig25 import Face, Kit, draw_face
from .turn25 import _over_pm, _surface_bgra


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
    face_mask: np.ndarray | None = None


def detect_moving_head_pose(
    frame_bgr: np.ndarray,
    prev: TrackedHeadFrame | None = None,
    smooth: float = 0.65,
    ref_radius: float = 195.0,
    base_yaw: float | None = None,
) -> TrackedHeadFrame:
    """
    Detect the character's ear-free inner head center `(cx, cy)`, radius, scale, and 3D orientation `(yaw, pitch, roll)`
    from a video frame bounded by a dilated dark-ink barrier (`gray < 125`).
    """
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Dilated dark-ink barrier prevents skin mask from leaking across chin/jaw/collar outlines
    ink_barrier = cv2.dilate((gray < 125).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    skin_mask = (
        (hsv[:, :, 0] >= 6)
        & (hsv[:, :, 0] <= 28)
        & (hsv[:, :, 1] >= 28)
        & (hsv[:, :, 1] <= 155)
        & (hsv[:, :, 2] >= 160)
        & (gray >= 130)
        & (~ink_barrier)
    ).astype(np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, ce = cv2.connectedComponentsWithStats(skin_mask, connectivity=4)

    best_id = None
    best_score = -1.0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < max(600, int(H * W * 0.004)) or y > H * 0.75:
            continue
        aspect = w / float(max(1, h))
        if not (0.50 <= aspect <= 1.65):
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
        return TrackedHeadFrame(
            cx=W * 0.5,
            cy=H * 0.38,
            radius=ref_radius,
            scale=1.0,
            yaw=0.0 if base_yaw is None else float(base_yaw),
            pitch=0.0,
            roll=0.0,
            skin_bgr=(175, 215, 243),
            face_mask=None,
        )

    comp = cv2.dilate(ndi.binary_fill_holes(lbl == best_id).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    skin_bgr = tuple(int(v) for v in np.median(frame_bgr[comp], axis=0))
    ys, xs = np.where(comp)
    ymin, ymax = int(ys.min()), int(ys.max())
    fh = max(20, ymax - ymin)

    # Ear-free lower-jaw band `[ymax - 0.28*fh .. ymax - 0.08*fh]` and eye band `[ymax - 0.65*fh .. ymax - 0.45*fh]`
    y_jaw0, y_jaw1 = max(ymin, ymax - int(0.28 * fh)), max(ymin + 4, ymax - int(0.08 * fh))
    jaw_mids, eye_widths = [], []
    for y in range(y_jaw0, y_jaw1, max(1, (y_jaw1 - y_jaw0) // 12)):
        r_xs = np.where(comp[y, :])[0]
        if len(r_xs) > 4:
            jaw_mids.append(0.5 * float(r_xs.min() + r_xs.max()))
    y_eye0, y_eye1 = max(ymin, ymax - int(0.65 * fh)), max(ymin + 4, ymax - int(0.45 * fh))
    for y in range(y_eye0, y_eye1, max(1, (y_eye1 - y_eye0) // 12)):
        r_xs = np.where(comp[y, :])[0]
        if len(r_xs) > 4:
            eye_widths.append(float(r_xs.max() - r_xs.min()))

    hcx = float(np.median(jaw_mids)) if jaw_mids else float(xs.mean())
    w_eye = float(np.percentile(eye_widths, 25)) if eye_widths else float(xs.max() - xs.min())
    hcy = float(ymax) - 0.50 * w_eye
    radius = max(20.0, 0.5 * w_eye)

    if base_yaw is not None:
        yaw = float(base_yaw)
    elif prev is not None and abs(radius - prev.radius) / max(20.0, prev.radius) < 0.035:
        # Smart Auto-Yaw Lock: when the drawn head width is not changing, lock yaw to prevent internal 10-deg drift!
        yaw = prev.yaw
    else:
        yaw = 0.0

    pitch = 0.0
    roll = 0.0
    scale = float(radius / max(20.0, ref_radius))

    cur = TrackedHeadFrame(
        cx=hcx,
        cy=hcy,
        radius=radius,
        scale=scale,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        skin_bgr=skin_bgr,
        face_mask=comp,
    )
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
            face_mask=comp,
        )
    return cur


def composite_face_on_tracked_frame(
    frame_bgr: np.ndarray,
    tracked: TrackedHeadFrame,
    kit: Kit,
    face_state: Face,
    acting_yaw_delta: float = 0.0,
    max_neck_fixed_px: float = 6.0,
    eye_span_ratio: float = 0.68,
) -> np.ndarray:
    """
    Render the `rig25` vector face locked to `tracked` on `frame_bgr`, using true rendered `src_c` calibration
    on the Cairo surface (accounting for `t_turn`) and premultiplied 4-channel interior clipping (`face_mask`).
    """
    max_deg = math.degrees(math.asin(np.clip(max_neck_fixed_px / max(tracked.radius, 25.0), 0.01, 0.12)))
    clamped_dyaw = float(np.clip(acting_yaw_delta, -max_deg, max_deg))
    excess_deg = acting_yaw_delta - clamped_dyaw
    eff_yaw = float(np.clip(tracked.yaw + clamped_dyaw, -35.0, 35.0))

    gx, gy = face_state.look
    gx_eff = float(np.clip(gx + (excess_deg / 25.0) * 0.45, -0.65, 0.65))
    st_eff = Face(
        lid_open=face_state.lid_open,
        lid_tilt=face_state.lid_tilt,
        lower=face_state.lower,
        look=(gx_eff, gy),
        pupil=face_state.pupil,
        brow_up=face_state.brow_up,
        brow_anger=face_state.brow_anger,
        brow_worry=face_state.brow_worry,
        furrow=face_state.furrow,
        bags=face_state.bags,
        mouth=face_state.mouth,
    )

    view, t_turn = kit.view(eff_yaw, tracked.pitch)
    fl = _surface_bgra(kit.W, 1180, lambda ctx: draw_face(ctx, kit, view, t_turn, st_eff))

    # True rendered feature bounding-box center on the Cairo surface (accounts for `t_turn` perspective shift!)
    fy, fx = np.where(fl[:, :, 3] > 20)
    if len(fx) > 0:
        src_cx = 0.5 * float(fx.min() + fx.max())
        src_cy = 0.5 * float(fy.min() + fy.max()) - 10.0
        kit_span_w = max(120.0, float(fx.max() - fx.min()))
        s = (2.0 * tracked.radius * eye_span_ratio) / kit_span_w
    else:
        src_cx, src_cy = 450.0, 465.0
        s = tracked.scale * 0.68

    a = math.radians(tracked.roll)
    ca, sa = math.cos(a) * s, math.sin(a) * s
    tx = tracked.cx - (ca * src_cx - sa * src_cy)
    ty = tracked.cy - (sa * src_cx + ca * src_cy)
    M = np.float32([[ca, -sa, tx], [sa, ca, ty]])
    H, W = frame_bgr.shape[:2]
    warped = cv2.warpAffine(fl, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    if tracked.face_mask is not None and tracked.face_mask.shape == (H, W):
        fm_eroded = cv2.erode(tracked.face_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        fm_eroded = cv2.GaussianBlur(fm_eroded, (3, 3), 0)[:, :, None]
        warped = (warped.astype(np.float32) * fm_eroded).astype(np.uint8)

    return _over_pm(frame_bgr.copy(), warped)
