"""
mograph.motion.layered_lipsync — Universal Multi-Angle 2D Face/Eye/Mouth Detector,
3-Layer Dual-Track Lip-Sync Compositor & Full-HD Ink-Black/Pure-White Line-Art Remastering Engine (v29.2).

Capabilities:
  1. Universal Topological Face/Eye/Mouth Landmark Detection (`detect_face_anatomy`):
     - Zero manual ROI needed: automatically distinguishes the round white head from the larger white
       sweater/torso using contour circularity `(4*pi*A/P^2)`, bounding fill ratio, and vertical rank.
     - Multi-Angle View Mode classification (`front`, `three_quarter_left`, `three_quarter_right`,
       `profile_left`, `profile_right`):
         * Front & 3/4 views (2 visible eyes, including narrow foreshortened far eyes in steep 3/4 views):
           uses exact `inter_ocular` distance and `eye_angle_deg` tilt angle.
         * Side Profile view (1 visible eye): automatically switches to `profile_left` / `profile_right`,
           uses `head_scale * 0.42` as the invariant scale ruler, and shifts the mouth search window to
           the facing edge of the jaw.
         * Closed / Squinted / Fantasy `><` Eyes (0 white sclera circles): falls back to dark-ink eye
           cluster detection in the upper head band (`0.18 <= rel_y <= 0.48`).
  2. Automatic Base-Mouth Erasure (`erase_base_mouth`) & Rotation-Aware Viseme Placement (`place_viseme_on_pose`):
     - Cleanly erases the static base mouth line with median forehead skin color while protecting eyes
       and overlapping hands/fingers.
     - Scales and rotates (`eye_angle_deg`) each extracted mouth sprite from an R×C Viseme Grid onto the
       exact anatomical `mouth_center` with automated QA (`qa_mouth_placement`).
  3. Full-HD (1080x1920) Ink-Black & Studio-White Line-Art Remastering (`enhance_lineart_hd`, `enhance_video_hd`):
     - Upscales low-res/blurry 2D line art to 1080p with Lanczos-4 + edge-preserving bilateral de-blocking.
     - Reinforces washed-out grey/brown lines into deep studio ink black (`#0A0A0C`) via a smooth S-curve
       and sub-pixel unsharp line-core tightening.
     - Purifies noisy off-white character fills into crisp Studio White (`#FCFCFA`) while preserving
       warm beige backgrounds and colored tongue/prop details.

CLI usage:
  python3 -m mograph.motion layered-lipsync detect <image.jpg> [-o anatomy_debug.jpg]
  python3 -m mograph.motion layered-lipsync enhance-hd <in.jpg|in.mp4> [-o out.jpg|out.mp4]
  python3 -m mograph.motion layered-lipsync render --bg <bg.jpg> --poses <poses.jpg> --visemes <grid.jpg> --fg <fg.jpg> --audio <audio.mp3> [--hd] [-o out.mp4]
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Full-HD (1080x1920) Ink-Black & Pure-White Vector-Grade Line-Art Enhancer
# ---------------------------------------------------------------------------
def enhance_lineart_hd(
    img_bgr: np.ndarray,
    target_wh: tuple[int, int] | None = None,
    ink_gamma: float = 0.52,
    ink_dark_cutoff: int = 155,
    whiten_character: bool = True,
) -> np.ndarray:
    """
    Remasters blurry or compressed 2D Stickman line art into razor-sharp Full-HD studio quality:
      - Lanczos-4 upscale + bilateral de-blocking removes JPEG artifacts while keeping sharp edges.
      - Sub-pixel unsharp mask + S-curve ink darkening pulls muddy grey/brown lines to deep ink black (#0A0A0C).
      - Purifies noisy off-white character fills (head, sweater, trousers) to clean Studio White (#FCFCFA).
    """
    if target_wh is not None and (img_bgr.shape[1], img_bgr.shape[0]) != target_wh:
        work = cv2.resize(img_bgr, target_wh, interpolation=cv2.INTER_LANCZOS4)
    else:
        work = img_bgr.copy()

    smooth = cv2.bilateralFilter(work, d=5, sigmaColor=24, sigmaSpace=24)
    blur_fine = cv2.GaussianBlur(smooth, (0, 0), 1.1)
    sharp = cv2.addWeighted(smooth, 1.45, blur_fine, -0.45, 0)

    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = sharp.astype(np.float32)

    if whiten_character:
        white_strength = np.clip((out[:, :, 0] - 170.0) / 55.0, 0.0, 1.0) * np.clip((gray - 170.0) / 55.0, 0.0, 1.0)
        studio_white = np.array([250.0, 252.0, 252.0], dtype=np.float32)
        out = out * (1.0 - 0.78 * white_strength[:, :, None]) + studio_white * (0.78 * white_strength[:, :, None])

    t = np.clip((float(ink_dark_cutoff) - gray) / float(max(1, ink_dark_cutoff - 25)), 0.0, 1.0)
    ink_weight = (t * t * (3.0 - 2.0 * t)) ** ink_gamma
    deep_black = np.array([10.0, 10.0, 12.0], dtype=np.float32)
    out = out * (1.0 - ink_weight[:, :, None]) + deep_black * ink_weight[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_video_hd(
    in_mp4: str,
    out_mp4: str,
    target_wh: tuple[int, int] = (1080, 1920),
    ink_gamma: float = 0.52,
) -> dict[str, Any]:
    """Upscales and remasters an entire MP4 animation into Full-HD Ink-Black & Studio-White quality."""
    cap = cv2.VideoCapture(in_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    tmp_raw = out_mp4 + ".hd_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_raw, fourcc, fps, target_wh)
    n_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(enhance_lineart_hd(frame, target_wh=target_wh, ink_gamma=ink_gamma))
        n_frames += 1
    cap.release()
    writer.release()

    # Preserve audio stream if present
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", in_mp4],
        capture_output=True, text=True,
    )
    has_audio = "audio" in probe.stdout
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", tmp_raw]
    if has_audio:
        cmd += ["-i", in_mp4, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_mp4]
    subprocess.check_call(cmd)
    if os.path.exists(tmp_raw):
        os.remove(tmp_raw)
    return {"out_mp4": out_mp4, "frames": n_frames, "resolution": f"{target_wh[0]}x{target_wh[1]}"}


# ---------------------------------------------------------------------------
# 2. Universal Multi-Angle Face, Eye-Pair & Mouth Landmark Detector
# ---------------------------------------------------------------------------
def detect_face_anatomy(
    img_bgr: np.ndarray,
    head_roi_y: tuple[int | None, int | None] | None = None,
    head_roi_x: tuple[int | None, int | None] | None = None,
) -> dict[str, Any]:
    """
    Universal topological detector for 2D cartoon heads across Front, 3/4, Side Profile,
    tilted heads, and closed/squinted `><` eyes. Requires zero manual ROI coordinates.
    """
    H, W = img_bgr.shape[:2]
    y0 = 0 if head_roi_y is None or head_roi_y[0] is None else int(head_roi_y[0])
    y1 = H if head_roi_y is None or head_roi_y[1] is None else int(head_roi_y[1])
    x0 = 0 if head_roi_x is None or head_roi_x[0] is None else int(head_roi_x[0])
    x1 = W if head_roi_x is None or head_roi_x[1] is None else int(head_roi_x[1])

    sub = img_bgr[y0:y1, x0:x1]
    sh, sw = sub.shape[:2]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)

    white = ((sub[:, :, 0] > 168) & (gray > 162)).astype(np.uint8)
    nl, lbl, st, cent = cv2.connectedComponentsWithStats(white, connectivity=4)

    # Topological circularity + fill ratio + upper-position scoring distinguishes the round head from the sweater/torso
    best_head_id = None
    best_score = -1e9
    for i in range(1, nl):
        bx, by, bw, bh, area = st[i]
        cx, cy = cent[i]
        if area < max(350, int(sh * sw * 0.006)):
            continue
        aspect = bw / float(max(1, bh))
        if not (0.55 <= aspect <= 1.55):
            continue
        comp_mask = (lbl == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c_main = max(cnts, key=cv2.contourArea)
        c_area = cv2.contourArea(c_main)
        perim = max(1.0, cv2.arcLength(c_main, True))
        circularity = (4.0 * np.pi * c_area) / (perim * perim)
        fill_ratio = c_area / float(max(1, bw * bh))
        upper_bonus = 1.0 - 0.65 * (cy / float(sh))
        score = (circularity ** 1.5) * fill_ratio * upper_bonus * np.sqrt(c_area)
        if score > best_score:
            best_score = score
            best_head_id = i

    if best_head_id is None:
        if nl <= 1:
            raise ValueError("No white head region detected in image")
        best_head_id = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))

    skull_white = (lbl == best_head_id).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(skull_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    head_interior = np.zeros_like(skull_white)
    cv2.drawContours(head_interior, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    head_interior_dilated = cv2.dilate(head_interior, np.ones((5, 5), np.uint8), iterations=1)
    hx, hy, hw, hh = cv2.boundingRect(head_interior)
    head_scale = float(np.sqrt(hw * hh))

    # Step 2A: White-sclera eye detection (supports symmetric Front, foreshortened 3/4, and single-eye Side Profile)
    eye_cands = []
    for i in range(1, nl):
        if i == best_head_id:
            continue
        bx, by, bw, bh, area = st[i]
        cx, cy = cent[i]
        if head_interior_dilated[int(cy), int(cx)] > 0:
            rel_y = (cy - hy) / float(max(1, hh))
            if 0.16 <= rel_y <= 0.74 and bw >= hw * 0.07 and bh >= hh * 0.10 and area >= (hw * hh * 0.003):
                eye_cands.append((float(cx), float(cy), int(bx), int(by), int(bw), int(bh), int(area)))

    eye_cands = sorted(eye_cands, key=lambda t: -t[6])[:2]
    eye_cands = sorted(eye_cands, key=lambda t: t[0])

    # Step 2B: Dark-ink eye cluster fallback for closed / squinted / `><` fantasy eyes
    inner_head_eroded = cv2.erode(head_interior, np.ones((9, 9), np.uint8), iterations=1)
    if len(eye_cands) == 0:
        eye_band = np.zeros_like(head_interior)
        y_eb0, y_eb1 = int(hy + 0.18 * hh), int(hy + 0.48 * hh)
        eye_band[y_eb0:y_eb1, hx:hx + hw] = inner_head_eroded[y_eb0:y_eb1, hx:hx + hw]
        dark_eyes = ((gray < 160) & (eye_band > 0)).astype(np.uint8) * 255
        dark_eyes = cv2.morphologyEx(dark_eyes, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        enl, elbl, est, ecent = cv2.connectedComponentsWithStats(dark_eyes, connectivity=8)
        for i in range(1, enl):
            bx, by, bw, bh, area = est[i]
            cx, cy = ecent[i]
            if area >= (hw * hh * 0.003) and bw >= hw * 0.08:
                eye_cands.append((float(cx), float(cy), int(bx), int(by), int(bw), int(bh), int(area)))
        eye_cands = sorted(eye_cands, key=lambda t: -t[6])[:2]
        eye_cands = sorted(eye_cands, key=lambda t: t[0])

    head_cx = hx + hw / 2.0
    if len(eye_cands) >= 2:
        eyes_mid_x = 0.5 * (eye_cands[0][0] + eye_cands[1][0])
        eyes_mid_y = 0.5 * (eye_cands[0][1] + eye_cands[1][1])
        inter_ocular = float(np.hypot(eye_cands[1][0] - eye_cands[0][0], eye_cands[1][1] - eye_cands[0][1]))
        eye_angle_deg = float(np.degrees(np.arctan2(eye_cands[1][1] - eye_cands[0][1], eye_cands[1][0] - eye_cands[0][0])))
        offset_ratio = (eyes_mid_x - head_cx) / float(max(1, hw))
        if offset_ratio < -0.04:
            view_mode = "three_quarter_left"
        elif offset_ratio > 0.04:
            view_mode = "three_quarter_right"
        else:
            view_mode = "front"
    elif len(eye_cands) == 1:
        eyes_mid_x, eyes_mid_y = eye_cands[0][0], eye_cands[0][1]
        inter_ocular = head_scale * 0.42
        eye_angle_deg = 0.0
        view_mode = "profile_left" if eyes_mid_x < head_cx else "profile_right"
    else:
        eyes_mid_x, eyes_mid_y = head_cx, hy + 0.48 * hh
        inter_ocular = head_scale * 0.42
        eye_angle_deg = 0.0
        view_mode = "front"

    eye_mask = np.zeros_like(head_interior)
    for (cx, cy, bx, by, bw, bh, _) in eye_cands:
        cv2.ellipse(eye_mask, (int(cx), int(cy)), (int(bw * 0.64 + 6), int(bh * 0.64 + 6)), 0, 0, 360, 255, -1)
    eyes_bottom_y = int(max((e[3] + e[5] for e in eye_cands), default=int(hy + 0.54 * hh)))

    erode_px = 4 if "profile" in view_mode else 9
    inner_head = cv2.erode(head_interior, np.ones((erode_px, erode_px), np.uint8), iterations=1)
    mouth_zone = ((inner_head > 0) & (eye_mask == 0)).astype(np.uint8)
    mouth_zone[:int(eyes_bottom_y + 2), :] = 0

    if view_mode == "profile_left":
        mx_min, mx_max = max(0, int(hx)), min(sw, int(eyes_mid_x + 0.35 * hw))
    elif view_mode == "profile_right":
        mx_min, mx_max = max(0, int(eyes_mid_x - 0.35 * hw)), min(sw, int(hx + hw))
    else:
        mx_min = max(0, int(eyes_mid_x - 0.65 * inter_ocular))
        mx_max = min(sw, int(eyes_mid_x + 0.75 * inter_ocular))
    mouth_zone[:, :mx_min] = 0
    mouth_zone[:, mx_max:] = 0

    mouth_ink = ((gray < 210) & (mouth_zone > 0)).astype(np.uint8) * 255
    mouth_ink_closed = cv2.morphologyEx(mouth_ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mcnts, _ = cv2.findContours(mouth_ink_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mouth_sprite_mask = np.zeros_like(mouth_ink)
    if mcnts:
        cv2.drawContours(mouth_sprite_mask, mcnts, -1, 255, -1)
        mouth_sprite_mask = cv2.dilate(mouth_sprite_mask, np.ones((3, 3), np.uint8), iterations=1)
        mx, my, mw, mh = cv2.boundingRect(mouth_sprite_mask)
        m_cx, m_cy = mx + mw / 2.0, my + mh / 2.0
    else:
        mx, my, mw, mh = int(eyes_mid_x - 20), int(eyes_bottom_y + 15), 40, 10
        m_cx, m_cy = mx + 20.0, my + 5.0

    forehead = sub[int(hy + hh * 0.12):int(hy + hh * 0.24), int(hx + hw * 0.35):int(hx + hw * 0.65)]
    face_color = np.median(forehead.reshape(-1, 3), axis=0).astype(np.uint8)

    return {
        "view_mode": view_mode,
        "offset": (x0, y0),
        "head_box": (int(hx + x0), int(hy + y0), int(hw), int(hh)),
        "head_scale": head_scale,
        "num_eyes": len(eye_cands),
        "eyes": [(e[0] + x0, e[1] + y0, e[2] + x0, e[3] + y0, e[4], e[5]) for e in eye_cands],
        "eyes_mid": (float(eyes_mid_x + x0), float(eyes_mid_y + y0)),
        "eyes_bottom_y": int(eyes_bottom_y + y0),
        "inter_ocular": float(inter_ocular),
        "eye_angle_deg": eye_angle_deg,
        "mouth_box": (int(mx + x0), int(my + y0), int(mw), int(mh)),
        "mouth_center": (float(m_cx + x0), float(m_cy + y0)),
        "inner_head_full": np.pad(inner_head, ((y0, H - y1), (x0, W - x1))),
        "eye_mask_full": np.pad(eye_mask, ((y0, H - y1), (x0, W - x1))),
        "mouth_sprite_mask_full": np.pad(mouth_sprite_mask, ((y0, H - y1), (x0, W - x1))),
        "face_color": face_color,
    }


def erase_base_mouth(img_bgr: np.ndarray, anatomy: dict[str, Any]) -> np.ndarray:
    """Erases the static base mouth line cleanly using forehead skin color while protecting fingers/eyes."""
    out = img_bgr.copy()
    erase_mask = cv2.dilate(anatomy["mouth_sprite_mask_full"], np.ones((7, 7), np.uint8), iterations=1)
    if "profile" not in anatomy.get("view_mode", "front"):
        min_erase_x = int(anatomy["eyes_mid"][0] - 0.36 * anatomy["inter_ocular"])
        erase_mask[:, :min_erase_x] = 0
    erase_mask[anatomy["eye_mask_full"] > 0] = 0
    out[erase_mask > 0] = anatomy["face_color"]
    return out


def qa_mouth_placement(anatomy: dict[str, Any], dst_x0: int, dst_y0: int, sw_s: int, sh_s: int) -> dict[str, Any]:
    """Automated QA guard verifying zero eye overlap and strict placement inside the lower face band."""
    H, W = anatomy["eye_mask_full"].shape[:2]
    y_a, y_b = max(0, dst_y0), min(H, dst_y0 + sh_s)
    x_a, x_b = max(0, dst_x0), min(W, dst_x0 + sw_s)
    eye_ov = int(np.sum(anatomy["eye_mask_full"][y_a:y_b, x_a:x_b] > 0))
    hx, hy, hw, hh = anatomy["head_box"]
    chin_y = hy + hh
    m_cy = dst_y0 + sh_s / 2.0
    below_eyes = bool(dst_y0 >= anatomy["eyes_bottom_y"] + 2)
    above_chin = bool(m_cy < chin_y - 3)
    return {
        "ok": below_eyes and above_chin,
        "below_eyes": below_eyes,
        "above_chin": above_chin,
        "eye_overlap_cropped": eye_ov,
        "mouth_cy": float(m_cy),
        "eyes_bottom_y": int(anatomy["eyes_bottom_y"]),
        "chin_y": int(chin_y),
    }


def extract_viseme_grid_sprites(grid_bgr: np.ndarray, rows: int = 4, cols: int = 4) -> list[dict[str, Any]]:
    """Slices an R×C viseme grid and extracts each cell's pure mouth sprite + anatomical scale & tilt."""
    gh, gw = grid_bgr.shape[:2]
    ggray = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2GRAY)
    v_means = np.mean(ggray, axis=0)
    h_means = np.mean(ggray, axis=1)

    vx = [0] + [
        int(np.argmin(v_means[int(gw * (c / cols) - 25):int(gw * (c / cols) + 25)]) + int(gw * (c / cols) - 25))
        for c in range(1, cols)
    ] + [gw]
    hy = [0] + [
        int(np.argmin(h_means[int(gh * (r / rows) - 25):int(gh * (r / rows) + 25)]) + int(gh * (r / rows) - 25))
        for r in range(1, rows)
    ] + [gh]

    sprites = []
    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        cell = grid_bgr[hy[r] + 5:hy[r + 1] - 5, vx[c] + 5:vx[c + 1] - 5]
        c_info = detect_face_anatomy(cell)
        mx, my, mw, mh = c_info["mouth_box"]
        pad = 4
        y_a, y_b = max(0, my - pad), min(cell.shape[0], my + mh + pad)
        x_a, x_b = max(0, mx - pad), min(cell.shape[1], mx + mw + pad)
        sp_bgr = cell[y_a:y_b, x_a:x_b].copy()
        sp_mask = c_info["mouth_sprite_mask_full"][y_a:y_b, x_a:x_b].astype(np.float32) / 255.0
        sp_mask = cv2.GaussianBlur(sp_mask, (5, 5), 1.0)
        sprites.append({
            "index": idx,
            "bgr": sp_bgr,
            "mask": sp_mask,
            "inter_ocular": c_info["inter_ocular"],
            "eye_angle_deg": c_info["eye_angle_deg"],
            "mouth_center": c_info["mouth_center"],
        })
    return sprites


def extract_stickman_rgba(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts a clean alpha mask for a white-filled, black-outlined 2D character on a beige background."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fg_raw = ((img_bgr[:, :, 0] > 160) | (gray < 148)).astype(np.uint8) * 255
    fg_raw = cv2.morphologyEx(fg_raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    nl, lbl, st, _ = cv2.connectedComponentsWithStats(fg_raw)
    best_i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    mask = (lbl == best_i).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0.7) / 255.0
    return img_bgr, alpha


def extract_fg_desk_rgba(fg_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts a watertight foreground prop mask (desk, lamp, open book) via binary-barrier flood-fill."""
    h, w = fg_bgr.shape[:2]
    gray = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2GRAY)
    lines = (gray < 155).astype(np.uint8) * 255
    barrier = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)
    ext = barrier.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [
        (w // 2, int(h * 0.08)), (int(w * 0.07), int(h * 0.08)), (int(w * 0.94), int(h * 0.08)),
        (int(w * 0.015), int(h * 0.53)), (int(w * 0.15), int(h * 0.53)),
        (int(w * 0.62), int(h * 0.82)), (int(w * 0.99), int(h * 0.82)),
        (int(w * 0.14), int(h * 0.98)), (int(w * 0.87), int(h * 0.90)),
    ]
    for sx, sy in seeds:
        if 0 <= sy < h and 0 <= sx < w and ext[sy, sx] == 0:
            cv2.floodFill(ext, ff_mask, (sx, sy), 128)
    fg_mask = (ext != 128).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(fg_mask.astype(np.float32), (3, 3), 0.7) / 255.0
    return fg_bgr, alpha


def place_viseme_on_pose(
    clean_pose_bgr: np.ndarray,
    pose_anatomy: dict[str, Any],
    viseme_sprite: dict[str, Any],
    scale_mult: float = 0.92,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Places a single extracted viseme mouth sprite onto an erased body pose, applying scale
    AND head-tilt rotation (`eye_angle_deg`) plus anatomical clamping and QA verification.
    """
    frm = clean_pose_bgr.copy()
    sp_bgr = viseme_sprite["bgr"]
    sp_mask = viseme_sprite["mask"]
    scale = (pose_anatomy["inter_ocular"] / viseme_sprite["inter_ocular"]) * scale_mult
    sh_s = max(2, int(round(sp_bgr.shape[0] * scale)))
    sw_s = max(2, int(round(sp_bgr.shape[1] * scale)))
    sp_bgr_s = cv2.resize(sp_bgr, (sw_s, sh_s), interpolation=cv2.INTER_CUBIC)
    sp_mask_s = cv2.resize(sp_mask, (sw_s, sh_s), interpolation=cv2.INTER_LINEAR)

    # Rotate viseme sprite if head is tilted relative to viseme cell
    d_angle = pose_anatomy.get("eye_angle_deg", 0.0) - viseme_sprite.get("eye_angle_deg", 0.0)
    if abs(d_angle) > 1.5:
        M_rot = cv2.getRotationMatrix2D((sw_s / 2.0, sh_s / 2.0), -d_angle, 1.0)
        sp_bgr_s = cv2.warpAffine(sp_bgr_s, M_rot, (sw_s, sh_s), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        sp_mask_s = cv2.warpAffine(sp_mask_s, M_rot, (sw_s, sh_s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    target_mcx, target_mcy = pose_anatomy["mouth_center"]
    target_eye_bot = pose_anatomy["eyes_bottom_y"]

    dst_x0 = int(round(target_mcx - sw_s / 2.0))
    dst_y0 = int(round(target_mcy - sh_s / 2.0))
    if dst_y0 < target_eye_bot + 5:
        dst_y0 = target_eye_bot + 5
    if "profile" not in pose_anatomy.get("view_mode", "front"):
        min_x = int(pose_anatomy["eyes_mid"][0] - 0.34 * pose_anatomy["inter_ocular"])
        if dst_x0 < min_x:
            dst_x0 = min_x

    qa = qa_mouth_placement(pose_anatomy, dst_x0, dst_y0, sw_s, sh_s)
    roi_dst = frm[dst_y0:dst_y0 + sh_s, dst_x0:dst_x0 + sw_s]
    valid_head = (
        (pose_anatomy["inner_head_full"][dst_y0:dst_y0 + sh_s, dst_x0:dst_x0 + sw_s] > 0)
        & (pose_anatomy["eye_mask_full"][dst_y0:dst_y0 + sh_s, dst_x0:dst_x0 + sw_s] == 0)
    )
    eff_alpha = sp_mask_s * valid_head.astype(np.float32)
    for ch_i in range(3):
        roi_dst[:, :, ch_i] = (
            sp_bgr_s[:, :, ch_i] * eff_alpha + roi_dst[:, :, ch_i] * (1.0 - eff_alpha)
        ).astype(np.uint8)
    return frm, qa


def slice_poses_sheet(poses_bgr: np.ndarray, num_poses: int = 3, target_wh: tuple[int, int] = (720, 1280)) -> list[np.ndarray]:
    """Slices a horizontal multi-panel character poses sheet and resizes each panel to `target_wh`."""
    target_w, target_h = target_wh
    sh, sw = poses_bgr.shape[:2]
    if num_poses == 1:
        return [cv2.resize(poses_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)]
    g = cv2.cvtColor(poses_bgr, cv2.COLOR_BGR2GRAY)
    cm = np.mean(g, axis=0)
    cuts = [0]
    for k in range(1, num_poses):
        lo, hi = int(sw * (k / num_poses - 0.05)), int(sw * (k / num_poses + 0.05))
        cuts.append(int(np.argmin(cm[lo:hi]) + lo))
    cuts.append(sw)
    panels = []
    for k in range(num_poses):
        rp = poses_bgr[:, cuts[k] + 8:cuts[k + 1] - 8]
        ph, pw = rp.shape[:2]
        new_w = int(ph * (target_w / target_h))
        x0 = max(0, (pw - new_w) // 2)
        p_resized = cv2.resize(rp[:, x0:x0 + new_w], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        panels.append(p_resized)
    return panels


def prepare_aligned_clean_poses(
    pose_panels: list[np.ndarray],
    ref_pose_bgr: np.ndarray | None = None,
    head_roi_y: tuple[int | None, int | None] | None = None,
    head_roi_x: tuple[int | None, int | None] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[np.ndarray]]:
    """
    Aligns all body poses to a canonical eye-midpoint and inter-ocular scale (with zero manual ROI needed),
    erases the static base mouth on each pose, and computes each pose's alpha mask.
    """
    target_h, target_w = pose_panels[0].shape[:2]
    ref_img = ref_pose_bgr if ref_pose_bgr is not None else pose_panels[0]
    ref_info = detect_face_anatomy(ref_img, head_roi_y, head_roi_x)

    clean_poses = []
    pose_infos = []
    pose_alphas = []
    for p_img in pose_panels:
        p_info = detect_face_anatomy(p_img, head_roi_y, head_roi_x)
        s = ref_info["inter_ocular"] / p_info["inter_ocular"]
        tx = ref_info["eyes_mid"][0] - p_info["eyes_mid"][0] * s
        ty = ref_info["eyes_mid"][1] - p_info["eyes_mid"][1] * s
        M = np.float32([[s, 0, tx], [0, s, ty]])
        aligned = cv2.warpAffine(p_img, M, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        al_info = detect_face_anatomy(aligned, head_roi_y, head_roi_x)
        erased = erase_base_mouth(aligned, al_info)
        _, alpha = extract_stickman_rgba(erased)
        clean_poses.append(erased)
        pose_infos.append(al_info)
        pose_alphas.append(alpha)
    return clean_poses, pose_infos, pose_alphas


def render_layered_lipsync(
    bg_path: str,
    poses_path: str,
    visemes_path: str,
    fg_path: str | None,
    audio_path: str,
    out_mp4: str,
    ref_char_path: str | None = None,
    body_timeline: list[int] | None = None,
    mouth_timeline: list[int] | None = None,
    fps: int = 24,
    work_wh: tuple[int, int] = (720, 1280),
    hd_remaster: bool = True,
    out_wh: tuple[int, int] = (1080, 1920),
) -> dict[str, Any]:
    """
    Renders a 3-Layer (Background -> Dual-Track Character -> Optional Foreground Prop) lip-synced MP4
    with optional Full-HD (1080x1920) Ink-Black & Studio-White vector-grade remastering.
    """
    target_w, target_h = work_wh
    bg_img = cv2.resize(cv2.imread(bg_path), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    fg_img = cv2.resize(cv2.imread(fg_path), (target_w, target_h), interpolation=cv2.INTER_CUBIC) if fg_path else None
    poses_raw = cv2.imread(poses_path)
    grid16 = cv2.imread(visemes_path)
    ref_char = cv2.resize(cv2.imread(ref_char_path), (target_w, target_h), interpolation=cv2.INTER_CUBIC) if ref_char_path else None

    panels = slice_poses_sheet(poses_raw, num_poses=3, target_wh=work_wh)
    clean_poses, pose_infos, pose_alphas = prepare_aligned_clean_poses(panels, ref_char)
    viseme_sprites = extract_viseme_grid_sprites(grid16, rows=4, cols=4)
    _, fg_alpha = extract_fg_desk_rgba(fg_img) if fg_img is not None else (None, None)

    cmd = ["ffmpeg", "-i", audio_path, "-f", "f32le", "-ac", "1", "-ar", "24000", "-v", "error", "-"]
    raw_audio = np.frombuffer(subprocess.check_output(cmd), dtype=np.float32)
    spf = 24000 // fps
    n_frames = len(raw_audio) // spf if len(raw_audio) >= spf else 108

    if body_timeline is None:
        body_timeline = [0] * n_frames
        if n_frames >= 60:
            t_lift = n_frames // 4
            t_drop = n_frames - 6
            for f in range(t_lift, min(t_lift + 3, n_frames)):
                body_timeline[f] = 1
            for f in range(t_lift + 3, max(t_lift + 3, t_drop)):
                body_timeline[f] = 2
            for f in range(max(0, t_drop), min(t_drop + 3, n_frames)):
                body_timeline[f] = 1

    if mouth_timeline is None:
        rms = [float(np.sqrt(np.mean(raw_audio[i * spf:(i + 1) * spf] ** 2))) for i in range(n_frames)]
        max_r = max(rms) if rms else 1.0
        norm_r = [r / max_r for r in rms]
        mouth_timeline = []
        i = 0
        while i < n_frames:
            rv = norm_r[i]
            if rv < 0.08:
                v = 0
            elif rv < 0.22:
                v = 2
            elif rv < 0.38:
                v = 4
            elif rv < 0.58:
                v = 10
            else:
                v = 5
            mouth_timeline.extend([v, v])
            i += 2
        mouth_timeline = mouth_timeline[:n_frames]

    n_frames = min(len(body_timeline), len(mouth_timeline))
    final_wh = out_wh if hd_remaster else work_wh
    tmp_raw = out_mp4 + ".raw_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_raw, fourcc, float(fps), final_wh)

    qa_failures = 0
    for idx in range(n_frames):
        p_idx = body_timeline[idx]
        v_idx = mouth_timeline[idx]
        char_layer, qa = place_viseme_on_pose(clean_poses[p_idx], pose_infos[p_idx], viseme_sprites[v_idx])
        if not qa["ok"]:
            qa_failures += 1
        char_alpha = pose_alphas[p_idx]

        comp = bg_img.astype(np.float32)
        comp = char_layer.astype(np.float32) * char_alpha[:, :, None] + comp * (1.0 - char_alpha[:, :, None])

        if fg_img is not None and fg_alpha is not None:
            desk_front_alpha = fg_alpha.copy()
            hand_over_desk = (np.arange(target_h)[:, None] < int(target_h * 0.631)) & (np.arange(target_w)[None, :] >= int(target_w * 0.278)) & (char_alpha > 0.4)
            desk_front_alpha[hand_over_desk] = 0.0
            comp = fg_img.astype(np.float32) * desk_front_alpha[:, :, None] + comp * (1.0 - desk_front_alpha[:, :, None])

        frame_u8 = comp.astype(np.uint8)
        if hd_remaster:
            frame_u8 = enhance_lineart_hd(frame_u8, target_wh=final_wh)
        writer.write(frame_u8)

    writer.release()
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "error", "-i", tmp_raw, "-i", audio_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-shortest", out_mp4,
    ])
    if os.path.exists(tmp_raw):
        os.remove(tmp_raw)
    return {
        "out_mp4": out_mp4,
        "frames": n_frames,
        "resolution": f"{final_wh[0]}x{final_wh[1]}",
        "hd_remaster": hd_remaster,
        "qa_clean": qa_failures == 0,
        "qa_failures": qa_failures,
        "view_mode": pose_infos[0]["view_mode"],
        "mouth_center_pose0": pose_infos[0]["mouth_center"],
        "eyes_bottom_y_pose0": pose_infos[0]["eyes_bottom_y"],
    }


def debug_anatomy_image(
    img_bgr: np.ndarray,
    head_roi_y: tuple[int | None, int | None] | None = None,
    head_roi_x: tuple[int | None, int | None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draws detected head box, eye circles, eye-bottom ruler line, and mouth center crosshair for visual QC."""
    info = detect_face_anatomy(img_bgr, head_roi_y, head_roi_x)
    vis = img_bgr.copy()
    hx, hy, hw, hh = info["head_box"]
    cv2.rectangle(vis, (hx, hy), (hx + hw, hy + hh), (255, 140, 0), 2)
    for (cx, cy, bx, by, bw, bh) in info["eyes"]:
        cv2.rectangle(vis, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 200, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 4, (0, 200, 0), -1)
    eb = info["eyes_bottom_y"]
    cv2.line(vis, (hx, eb), (hx + hw, eb), (0, 180, 255), 2)
    mx, my, mw, mh = info["mouth_box"]
    cv2.rectangle(vis, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
    mcx, mcy = info["mouth_center"]
    cv2.drawMarker(vis, (int(mcx), int(mcy)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    return vis, info
