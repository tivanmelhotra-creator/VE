"""
toonface.kit.sandwich — v31.4 3-Layer Depth Sandwich & Blank-Face Gesture Rig (`WeldedSkullSandwichRig`).

Provides the complete pipeline for multi-layer 2.5D character compositing:
  - `extract_amber_matte`: Flood-fill flat amber (#D9A066) background to extract clean foreground alpha mattes.
  - `slice_three_panels`: Slice a 16:9 3-panel decomposition or 3-pose blank-face gesture sheet into 9:16 frames.
  - `align_bg_chair_behind_character`: Shift the chair band in Layer 1 (BG) horizontally behind the character's torso.
  - `load_stepped_video_range`: Surgical frame sub-range extraction (`Zero Cross-Shot Morphing`) + 1/3 Frame Thinning (`step_hold=3`).
  - `build_pingpong_gesture_schedule`: Ping-Pong Loop builder (`Forward + Reverse`) for blank-face body gesture clips.
  - `ThreeLayerSandwichRig`: Static-body / multi-pose 3-Layer Depth Sandwich (`Layer 1 BG` + `Layer 2 Blank-Face Char` + `Layer 3 FG`).
  - `WeldedSkullSandwichRig`: Moving-body blank-face gesture 3-Layer Depth Sandwich with:
      1. Dilated dark-ink barrier (`gray < 125`) preventing chin-to-neck or sweater leaks.
      2. Ear-free lower-jaw skull anchor (`y in [ymax - 120 .. ymax - 35]`) immune to ear width and forehead hand occlusion.
      3. True rendered `src_cx, src_cy` calibration on the Cairo surface after `t_turn`.
      4. Smart Auto-Yaw Lock (`yaw = base_yaw, pitch = 0.0` when `|w - w0| / w0 < 0.035`) with zero internal speech bounce.
      5. Shifted reference hull forehead/finger occlusion (`hair_fg`) and premultiplied 4-channel interior clipping (`fm_eroded`).
  - `build_listener_reaction_face`: Expressive silent listener `Face` generator for Intercut Reaction Cutaways (`L-Cut / Over-Dialogue`).
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
from scipy import ndimage as ndi

from .gate import head_warp
from .rig25 import Face, Kit, draw_face
from .turn25 import _over_pm, _surface_bgra

W_DEFAULT, H_DEFAULT = 1080, 1920


def extract_amber_matte(
    img_bgr: np.ndarray,
    border_seeds: str = "top_left_right",
    tol: float = 25.0,
) -> np.ndarray:
    """Flood-fill the flat amber (#D9A066) background from borders to return a [0..1] foreground alpha matte."""
    H, W = img_bgr.shape[:2]
    sm = cv2.medianBlur(img_bgr, 3)
    amber = np.median(
        np.vstack([
            sm[10:min(60, H // 4), 10:min(60, W // 4)].reshape(-1, 3),
            sm[10:min(60, H // 4), max(0, W - 60):max(1, W - 10)].reshape(-1, 3),
        ]),
        axis=0,
    )
    dist = np.linalg.norm(sm.astype(np.float32) - amber.astype(np.float32), axis=2)
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    is_amber = (dist < tol) & (g > 110)

    n, lbl = cv2.connectedComponents(is_amber.astype(np.uint8), connectivity=4)
    if border_seeds == "top_only":
        border_ids = np.unique(np.r_[lbl[0, :], lbl[:min(250, H // 4), 0], lbl[:min(250, H // 4), -1]])
    else:
        border_ids = np.unique(np.r_[lbl[0, :], lbl[:, 0], lbl[:, -1]])
    bg_mask = np.isin(lbl, border_ids[border_ids > 0])
    fg_mask = ndi.binary_fill_holes(~bg_mask)
    if H >= 1800 and W >= 1000:
        fg_mask[int(H * 0.88):, int(W * 0.83):] = False
    alpha = cv2.GaussianBlur(fg_mask.astype(np.float32), (3, 3), 0)
    return np.clip(alpha, 0.0, 1.0)


def slice_three_panels(
    sheet_path: str,
    names: list[str] | None = None,
    out_dir: str | None = None,
    target_wh: tuple[int, int] = (W_DEFAULT, H_DEFAULT),
) -> list[np.ndarray]:
    """Slice a 16:9 3-panel sheet into 3 clean 9:16 vertical BGR arrays (and optionally save PNGs to `out_dir`)."""
    img = cv2.imread(sheet_path)
    if img is None:
        raise FileNotFoundError(f"Missing sheet: {sheet_path}")
    W, H = target_wh
    sh, sw = img.shape[:2]
    third = sw // 3
    panels = []
    for idx in range(3):
        x0 = idx * third
        x1 = (idx + 1) * third if idx < 2 else sw
        mx = int((x1 - x0) * 0.032)
        my = int(sh * 0.015)
        crop = img[my:sh - my, x0 + mx:x1 - mx]
        panel = cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)
        if out_dir and names and idx < len(names):
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{names[idx]}.png"), panel)
        panels.append(panel)
    return panels


def align_bg_chair_behind_character(
    bg_panel: np.ndarray,
    shift_chair_x: int = -140,
    y_split: int = 1180,
) -> np.ndarray:
    """Shift the lower chair region of Layer 1 (BG) horizontally so the chair backrest sits directly behind the character."""
    H, W = bg_panel.shape[:2]
    out = bg_panel.copy()
    amber = np.median(bg_panel[20:min(80, H // 4), 20:min(80, W // 4)].reshape(-1, 3), axis=0).astype(np.uint8)
    y0 = min(y_split, H - 1)
    chair_band = bg_panel[y0:, :].copy()
    M = np.float32([[1, 0, shift_chair_x], [0, 1, 0]])
    shifted = cv2.warpAffine(
        chair_band,
        M,
        (W, H - y0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(v) for v in amber),
    )
    out[y0:, :] = shifted
    return out


def load_stepped_video_range(
    mp4_path: str,
    f_start: int = 0,
    f_end: int | None = None,
    step_hold: int = 3,
    crop_watermark: bool = True,
    target_wh: tuple[int, int] = (W_DEFAULT, H_DEFAULT),
) -> list[np.ndarray]:
    """
    Load an exact clean frame sub-range `[f_start..f_end]` from an AI-generated MP4 video, resize to `target_wh`,
    and apply 1/3 Frame Thinning (`step_hold=3`) with shared array references for low memory footprint.
    """
    W, H = target_wh
    cap = cv2.VideoCapture(mp4_path)
    raw_frames = []
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        raw_frames.append(frm)
    cap.release()
    if f_end is None or f_end >= len(raw_frames):
        f_end = len(raw_frames) - 1
    sub_raw = raw_frames[f_start:f_end + 1]
    n = len(sub_raw)

    resized_cache: dict[int, np.ndarray] = {}
    stepped: list[np.ndarray] = []
    for i in range(n):
        idx = min(n - 1, (i // step_hold) * step_hold)
        if idx not in resized_cache:
            frm = sub_raw[idx]
            if crop_watermark:
                fh, fw = frm.shape[:2]
                ch, cw = int(fh * 0.94), int(fw * 0.94)
                x0 = (fw - cw) // 2
                y0 = int(fh * 0.02)
                frm = cv2.resize(frm[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_CUBIC)
            else:
                frm = cv2.resize(frm, (W, H), interpolation=cv2.INTER_CUBIC)
            resized_cache[idx] = frm
        stepped.append(resized_cache[idx])
    return stepped


def build_pingpong_gesture_schedule(
    stepped_frames: list[np.ndarray],
    segments: list[tuple[float, float, int, int, bool]],
    total_dur: float,
    fps: int = 24,
    step_hold: int = 3,
) -> list[int]:
    """
    Build a frame-index schedule of length `int(round(total_dur * fps))` that maps each dialogue phrase
    `(t0, t1, f_start, f_end, pingpong)` to a seamless Ping-Pong Loop (`Forward + Reverse`) quantized to `step_hold=3`.
    """
    n_total = int(round(total_dur * fps))
    schedule = [0] * n_total
    max_idx = max(0, len(stepped_frames) - 1)

    for f_i in range(n_total):
        t = f_i / float(fps)
        active_seg = segments[-1]
        for seg in segments:
            if seg[0] <= t <= seg[1]:
                active_seg = seg
                break
        t0, t1, s_idx, e_idx, pingpong = active_seg
        u = float(np.clip((t - t0) / max(0.001, t1 - t0), 0.0, 1.0))
        if pingpong:
            forward_indices = list(range(s_idx, e_idx + 1))
            loop_indices = forward_indices + forward_indices[-2:0:-1] if len(forward_indices) > 2 else forward_indices
            k = min(len(loop_indices) - 1, int(round(u * (len(loop_indices) - 1))))
            raw_idx = loop_indices[k]
        else:
            raw_idx = int(round(s_idx + u * (e_idx - s_idx)))
        quant_idx = min(max_idx, (raw_idx // step_hold) * step_hold)
        schedule[f_i] = quant_idx
    return schedule


def camera_ken_burns(
    panel: np.ndarray,
    u: float,
    zoom0: float = 1.04,
    zoom1: float = 1.12,
    pan_x0: float = 0.0,
    pan_x1: float = 0.0,
    pan_y0: float = 0.0,
    pan_y1: float = 0.0,
) -> np.ndarray:
    """Programmatic camera zoom/pan on a frame."""
    H, W = panel.shape[:2]
    u = float(np.clip(u, 0.0, 1.0))
    ue = u * u * (3.0 - 2.0 * u)
    z = zoom0 + (zoom1 - zoom0) * ue
    px = pan_x0 + (pan_x1 - pan_x0) * ue
    py = pan_y0 + (pan_y1 - pan_y0) * ue

    cw = int(round(W / z))
    ch = int(round(H / z))
    cx = int(round(W * 0.5 + px))
    cy = int(round(H * 0.5 + py))
    x0 = int(np.clip(cx - cw // 2, 0, W - cw))
    y0 = int(np.clip(cy - ch // 2, 0, H - ch))
    sub = panel[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(sub, (W, H), interpolation=cv2.INTER_CUBIC)


class ThreeLayerSandwichRig:
    """
    Static-Body / Multi-Pose 3-Layer Depth Sandwich (`Layer 1 Static BG` + `Layer 2 Blank-Face Character` + `Layer 3 FG Table`).
    """

    def __init__(
        self,
        bg_panel: np.ndarray,
        char_panel: np.ndarray,
        fg_panel: np.ndarray,
        char_kind: str,
        face_scale_mult: float = 0.72,
        face_offset_xy: tuple[float, float] = (0.0, 0.0),
        char_scale: float = 0.85,
        char_shift_xy: tuple[int, int] = (0, -90),
        fg_table_shift_y: int = 260,
    ):
        self.bg = bg_panel.copy()
        self.char_base = char_panel.copy()
        self.fg_base = fg_panel.copy()
        self.H, self.W = self.bg.shape[:2]
        self.kit = Kit(char_kind)
        self.char_scale = char_scale
        self.char_shift_xy = char_shift_xy

        self.char_alpha = extract_amber_matte(self.char_base, border_seeds="top_left_right", tol=25.0)
        fg_alpha_raw = extract_amber_matte(self.fg_base, border_seeds="top_only", tol=25.0)
        fg_rgba_raw = np.dstack([self.fg_base, (fg_alpha_raw * 255.0).astype(np.uint8)])
        M_fg = np.float32([[1, 0, 0], [0, 1, fg_table_shift_y]])
        self.fg_rgba = cv2.warpAffine(
            fg_rgba_raw,
            M_fg,
            (self.W, self.H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        g = cv2.cvtColor(self.char_base, cv2.COLOR_BGR2GRAY)
        ink = cv2.dilate((g < 125).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        c_ys, c_xs = np.where(self.char_alpha > 0.5)
        head_y_top = int(c_ys.min())
        head_y_bot = int(c_ys.min() + 0.48 * (c_ys.max() - c_ys.min()))
        sx = int(np.median(c_xs[(c_ys >= head_y_top + 60) & (c_ys <= head_y_bot - 40)]))
        sy = int(head_y_top + 0.58 * (head_y_bot - head_y_top))

        skin_color = np.median(self.char_base[max(0, sy - 15):min(self.H, sy + 16), max(0, sx - 15):min(self.W, sx + 16)].reshape(-1, 3), axis=0)
        dist_skin = np.linalg.norm(self.char_base.astype(np.float32) - skin_color.astype(np.float32), axis=2)
        skin_raw = ((dist_skin < 22.0) & (~ink) & (self.char_alpha > 0.5)).astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(skin_raw, connectivity=4)
        cid = lbl[sy, sx]
        if cid == 0:
            cid = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
        face_mask = cv2.dilate(ndi.binary_fill_holes(lbl == cid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.face_mask = face_mask

        f_ys, f_xs = np.where(face_mask)
        self.ymin, self.ymax = int(f_ys.min()), int(f_ys.max())
        self.xmin, self.xmax = int(f_xs.min()), int(f_xs.max())
        self.cx = 0.5 * float(self.xmin + self.xmax) + face_offset_xy[0]
        self.cy = float(self.ymax) - 0.46 * float(self.ymax - self.ymin) + face_offset_xy[1]
        face_w = float(self.xmax - self.xmin)
        self.scale = (face_w / 390.0) * face_scale_mult

        hull = cv2.convexHull(np.c_[f_xs, f_ys].astype(np.int32))
        hm = np.zeros((self.H, self.W), np.uint8)
        cv2.drawContours(hm, [hull], -1, 1, -1)
        hm = cv2.dilate(hm, np.ones((15, 15), np.uint8)) > 0
        ymid = int(self.ymin + 0.50 * (self.ymax - self.ymin))
        fore = hm & (~face_mask) & (self.char_alpha > 0.5)
        fore[ymid:] = False
        a_hair = cv2.GaussianBlur(fore.astype(np.float32), (3, 3), 0)
        self.hair_fg = np.dstack([self.char_base, (np.clip(a_hair, 0, 1) * 255).astype(np.uint8)])

        self.pivot = (int(self.cx), int(self.ymax + 35))
        self.y_head_end = int(self.ymax - 8)
        self.y_body_start = int(min(self.H - 10, self.ymax + 120))

    def render_sandwich(
        self,
        yaw: float,
        pitch: float,
        face_state: Face,
        rot_deg: float = 0.0,
        dy: float = 0.0,
        breath: float = 0.0,
    ) -> np.ndarray:
        yaw_clamped = float(np.clip(yaw, -35.0, 35.0))
        view, t_turn = self.kit.view(yaw_clamped, pitch)
        fl = _surface_bgra(self.kit.W, 1180, lambda ctx: draw_face(ctx, self.kit, view, t_turn, face_state))
        fy, fx = np.where(fl[:, :, 3] > 20)
        src_cx = 0.5 * float(fx.min() + fx.max()) if len(fx) else 450.0
        src_cy = 0.5 * float(fy.min() + fy.max()) - 10.0 if len(fy) else 465.0

        s = self.scale
        M = np.float32([[s, 0.0, self.cx - s * src_cx], [0.0, s, self.cy - s * src_cy]])
        warped_face = cv2.warpAffine(fl, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        fm_eroded = cv2.erode(self.face_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        fm_eroded = cv2.GaussianBlur(fm_eroded, (3, 3), 0)[:, :, None]
        warped_face = (warped_face.astype(np.float32) * fm_eroded).astype(np.uint8)

        char_bgr = _over_pm(self.char_base.copy(), warped_face)
        al_h = self.hair_fg[:, :, 3:4].astype(np.float32) / 255.0
        char_bgr = (char_bgr.astype(np.float32) * (1.0 - al_h) + self.hair_fg[:, :, :3].astype(np.float32) * al_h).astype(np.uint8)
        char_rgba = np.dstack([char_bgr, (self.char_alpha * 255.0).astype(np.uint8)])

        char_rgba_warped = head_warp(
            char_rgba,
            pivot=self.pivot,
            rot_deg=rot_deg,
            dy=dy,
            breath=breath,
            y_head_end=self.y_head_end,
            y_body_start=self.y_body_start,
        )

        cs = self.char_scale
        dx_c, dy_c = self.char_shift_xy
        M_char = np.float32([
            [cs, 0.0, (1.0 - cs) * (self.W * 0.5) + dx_c],
            [0.0, cs, (1.0 - cs) * (self.H * 0.5) + dy_c],
        ])
        char_rgba_placed = cv2.warpAffine(
            char_rgba_warped,
            M_char,
            (self.W, self.H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        canvas = self.bg.copy()
        al_c = char_rgba_placed[:, :, 3:4].astype(np.float32) / 255.0
        canvas = (canvas.astype(np.float32) * (1.0 - al_c) + char_rgba_placed[:, :, :3].astype(np.float32) * al_c).astype(np.uint8)

        al_f = self.fg_rgba[:, :, 3:4].astype(np.float32) / 255.0
        canvas = (canvas.astype(np.float32) * (1.0 - al_f) + self.fg_rgba[:, :, :3].astype(np.float32) * al_f).astype(np.uint8)
        return canvas


class WeldedSkullSandwichRig:
    """
    Moving Blank-Face Gesture 3-Layer Depth Sandwich (`WeldedSkullSandwichRig`) with:
      1. Dilated dark-ink barrier (`gray < 125`) preventing chin-to-neck or sweater leaks.
      2. Ear-free lower-jaw skull anchor (`y in [ymax - 120 .. ymax - 35]`) immune to ear width and forehead hand occlusion.
      3. True rendered `src_cx, src_cy` calibration on the Cairo surface after `t_turn`.
      4. Smart Auto-Yaw Lock (`yaw = base_yaw, pitch = 0.0`) with zero internal speech bounce (`dy_speech = 0`).
      5. Shifted reference hull forehead/finger occlusion (`hair_fg`) and premultiplied 4-channel interior clipping (`fm_eroded`).
    """

    def __init__(
        self,
        bg_panel: np.ndarray,
        gesture_frames: list[np.ndarray],
        fg_panel: np.ndarray,
        char_kind: str,
        base_yaw: float = 15.0,
        eye_span_ratio: float = 0.67,
        face_offset_xy: tuple[float, float] = (0.0, 0.0),
        char_scale: float = 0.85,
        char_shift_xy: tuple[int, int] = (0, -90),
        fg_table_shift_y: int = 260,
        verify_margins: bool = True,
    ):
        self.bg = bg_panel.copy()
        self.frames = gesture_frames
        self.fg_base = fg_panel.copy()
        self.H, self.W = self.bg.shape[:2]
        self.char_kind = char_kind
        self.kit = Kit(char_kind)
        self.base_yaw = float(base_yaw)
        self.face_offset_xy = face_offset_xy
        self.char_scale = char_scale
        self.char_shift_xy = char_shift_xy

        fg_alpha_raw = extract_amber_matte(self.fg_base, border_seeds="top_only", tol=25.0)
        fg_rgba_raw = np.dstack([self.fg_base, (fg_alpha_raw * 255.0).astype(np.uint8)])
        M_fg = np.float32([[1, 0, 0], [0, 1, fg_table_shift_y]])
        self.fg_rgba = cv2.warpAffine(
            fg_rgba_raw,
            M_fg,
            (self.W, self.H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        f0 = self.frames[0]
        sy_seed, sx_seed = int(self.H * 0.43), int(self.W * 0.47)
        self.skin_color = np.median(
            f0[max(0, sy_seed - 30):min(self.H, sy_seed + 30), max(0, sx_seed - 30):min(self.W, sx_seed + 30)].reshape(-1, 3),
            axis=0,
        )

        fm0, _ = self._extract_ink_bounded_face(f0, seed_xy=(sx_seed, sy_seed))
        ys0, xs0 = np.where(fm0)
        self.ymin0, self.ymax0 = int(ys0.min()), int(ys0.max())
        self.h0 = float(self.ymax0 - self.ymin0)

        self.cx_jaw0, self.ymax_f0, self.w_jaw0 = self._measure_lower_jaw_anchor(fm0)
        eye_y0 = max(self.ymin0 + int(0.15 * self.h0), self.ymax0 - int(0.68 * self.h0))
        eye_y1 = max(eye_y0 + 10, self.ymax0 - int(0.46 * self.h0))
        eye_widths = []
        eye_mids = []
        for y in range(eye_y0, eye_y1, max(1, (eye_y1 - eye_y0) // 18)):
            r_xs = np.where(fm0[y, :])[0]
            if len(r_xs) > 10:
                eye_widths.append(float(r_xs.max() - r_xs.min()))
                eye_mids.append(0.5 * float(r_xs.min() + r_xs.max()))
        self.w_eye0 = float(np.percentile(eye_widths, 25)) if eye_widths else self.w_jaw0
        cx_eye0 = float(np.median(eye_mids)) if eye_mids else self.cx_jaw0

        self.cx0 = (0.55 * self.cx_jaw0 + 0.45 * cx_eye0) + face_offset_xy[0]
        self.cy0 = float(self.ymax0) - 0.52 * self.w_eye0 + face_offset_xy[1]

        self._locked_view, self._locked_t_turn = self.kit.view(self.base_yaw, 0.0)
        st_cal = Face(lid_open=0.82, brow_up=0.35, mouth={"vis": "A", "open": 0.40})
        fl_cal = _surface_bgra(self.kit.W, 1180, lambda ctx: draw_face(ctx, self.kit, self._locked_view, self._locked_t_turn, st_cal))
        fy, fx = np.where(fl_cal[:, :, 3] > 20)
        self._src_cx = 0.5 * float(fx.min() + fx.max())
        self._src_cy = 0.5 * float(fy.min() + fy.max()) - 10.0
        kit_span_w = float(fx.max() - fx.min())
        self.scale0 = (self.w_eye0 * eye_span_ratio) / max(1.0, kit_span_w)

        hull0 = cv2.convexHull(np.c_[xs0, ys0].astype(np.int32))
        self.hm0 = np.zeros((self.H, self.W), np.uint8)
        cv2.drawContours(self.hm0, [hull0], -1, 1, -1)
        self.hm0 = cv2.dilate(self.hm0, np.ones((21, 21), np.uint8))

        self._frame_cache: dict[int, tuple[np.ndarray, float, float, float, float, np.ndarray, np.ndarray]] = {}
        if verify_margins and self.W >= 1000:
            self._verify_all_40_frames(fl_cal)

    def _extract_ink_bounded_face(
        self, frm: np.ndarray, seed_xy: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if seed_xy is None:
            seed_xy = (int(self.W * 0.47), int(self.H * 0.43))
        al = extract_amber_matte(frm, border_seeds="top_left_right", tol=25.0)
        g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        ink = cv2.dilate((g < 125).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        dist = np.linalg.norm(frm.astype(np.float32) - self.skin_color.astype(np.float32), axis=2)
        skin = ((dist < 22.0) & (~ink) & (al > 0.5)).astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(skin, connectivity=4)
        cid = lbl[min(self.H - 1, seed_xy[1]), min(self.W - 1, seed_xy[0])]
        if cid == 0 or st[cid, cv2.CC_STAT_AREA] < max(500, int(self.H * self.W * 0.008)):
            cid = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
        fm = cv2.dilate(ndi.binary_fill_holes(lbl == cid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        return fm, al

    def _measure_lower_jaw_anchor(self, fm: np.ndarray) -> tuple[float, float, float]:
        ys, xs = np.where(fm)
        ymin, ymax = int(ys.min()), int(ys.max())
        fh = max(20, ymax - ymin)
        y_lo = max(ymin + int(0.65 * fh), ymax - int(0.28 * fh))
        y_hi = max(y_lo + 4, ymax - int(0.08 * fh))
        mids = []
        widths = []
        for y in range(y_lo, y_hi, max(1, (y_hi - y_lo) // 15)):
            r_xs = np.where(fm[y, :])[0]
            if len(r_xs) > 6:
                mids.append(0.5 * float(r_xs.min() + r_xs.max()))
                widths.append(float(r_xs.max() - r_xs.min()))
        cx_jaw = float(np.median(mids)) if mids else 0.5 * float(xs.min() + xs.max())
        w_jaw = float(np.median(widths)) if widths else float(xs.max() - xs.min())
        return cx_jaw, float(ymax), w_jaw

    def _get_tracked_frame_data(self, frame_idx: int) -> tuple[np.ndarray, float, float, float, float, np.ndarray, np.ndarray]:
        q_idx = min(len(self.frames) - 1, (frame_idx // 3) * 3)
        if q_idx in self._frame_cache:
            return self._frame_cache[q_idx]
        frm = self.frames[q_idx]
        fm, al = self._extract_ink_bounded_face(frm)
        cx_jaw, ymax, _ = self._measure_lower_jaw_anchor(fm)

        dx = cx_jaw - self.cx_jaw0
        dy = ymax - self.ymax_f0
        cx = self.cx0 + dx
        cy = self.cy0 + dy

        M_h = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        hm_shifted = cv2.warpAffine(self.hm0, M_h, (self.W, self.H), flags=cv2.INTER_NEAREST) > 0
        ymid = int((self.ymin0 + dy) + 0.55 * self.h0)
        fore = hm_shifted & (~fm) & (al > 0.5)
        fore[ymid:] = False
        a_hair_u8 = (np.clip(cv2.GaussianBlur(fore.astype(np.float32), (3, 3), 0), 0.0, 1.0) * 255.0).astype(np.uint8)

        fm_eroded = cv2.erode(fm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
        fm_eroded_u8 = (np.clip(cv2.GaussianBlur(fm_eroded, (5, 5), 0), 0.0, 1.0) * 255.0).astype(np.uint8)
        al_u8 = (al * 255.0).astype(np.uint8)

        data = (al_u8, cx, cy, dx, dy, a_hair_u8, fm_eroded_u8)
        self._frame_cache[q_idx] = data
        return data

    def _verify_all_40_frames(self, fl_cal: np.ndarray) -> None:
        min_l, min_r, min_b = 999.0, 999.0, 999.0
        s = self.scale0
        for q in range(0, len(self.frames), 3):
            _, cx, cy, _, _, _, fm_eroded_u8 = self._get_tracked_frame_data(q)
            M = np.float32([[s, 0.0, cx - s * self._src_cx], [0.0, s, cy - s * self._src_cy]])
            wf = cv2.warpAffine(fl_cal, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            wy, wx = np.where(wf[:, :, 3] > 25)
            f_xmin, f_xmax = int(wx.min()), int(wx.max())
            f_ymin, f_ymax = int(wy.min()), int(wy.max())
            y_eye_mid = int(f_ymin + 0.35 * (f_ymax - f_ymin))
            skin_xs = np.where(fm_eroded_u8[y_eye_mid, :] > 127)[0]
            s_xmin, s_xmax = int(skin_xs.min()), int(skin_xs.max())
            ys_all = np.where(fm_eroded_u8 > 127)[0]
            s_ymax = int(ys_all.max())
            min_l = min(min_l, f_xmin - s_xmin)
            min_r = min(min_r, s_xmax - f_xmax)
            min_b = min(min_b, s_ymax - f_ymax)
        assert min_l >= 45, f"{self.char_kind} left margin {min_l}px < 45px!"
        assert min_r >= 45, f"{self.char_kind} right margin {min_r}px < 45px!"
        assert min_b >= 40, f"{self.char_kind} bottom margin {min_b}px < 40px!"

    def render_sandwich(
        self,
        frame_idx: int,
        face_state: Face,
        cam_zoom: float = 1.0,
        cam_pan_xy: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        q_idx = min(len(self.frames) - 1, (frame_idx // 3) * 3)
        frm = self.frames[q_idx]
        al_u8, cx, cy, dx, dy, a_hair_u8, fm_eroded_u8 = self._get_tracked_frame_data(q_idx)

        view, t_turn = self._locked_view, self._locked_t_turn
        fl = _surface_bgra(self.kit.W, 1180, lambda ctx: draw_face(ctx, self.kit, view, t_turn, face_state))

        s = self.scale0
        M = np.float32([[s, 0.0, cx - s * self._src_cx], [0.0, s, cy - s * self._src_cy]])
        warped_face = cv2.warpAffine(fl, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Premultiplied 4-channel interior clipping (`fm_eroded`) — zero pixels can ever protrude outside the face!
        fm_scale = fm_eroded_u8[:, :, None].astype(np.float32) / 255.0
        warped_face = (warped_face.astype(np.float32) * fm_scale).astype(np.uint8)

        char_bgr = _over_pm(frm.copy(), warped_face)
        y_cut_h = min(self.H, 960)
        al_h = a_hair_u8[:y_cut_h, :, None].astype(np.float32) / 255.0
        char_bgr[:y_cut_h] = (
            char_bgr[:y_cut_h].astype(np.float32) * (1.0 - al_h) + frm[:y_cut_h].astype(np.float32) * al_h
        ).astype(np.uint8)
        char_rgba = np.dstack([char_bgr, al_u8])

        cs = self.char_scale
        dx_c, dy_c = self.char_shift_xy
        M_char = np.float32([
            [cs, 0.0, (1.0 - cs) * (self.W * 0.5) + dx_c],
            [0.0, cs, (1.0 - cs) * (self.H * 0.5) + dy_c],
        ])
        char_rgba_placed = cv2.warpAffine(
            char_rgba,
            M_char,
            (self.W, self.H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        canvas = self.bg.copy()
        al_c = char_rgba_placed[:, :, 3:4].astype(np.float32) / 255.0
        canvas = (canvas.astype(np.float32) * (1.0 - al_c) + char_rgba_placed[:, :, :3].astype(np.float32) * al_c).astype(np.uint8)

        al_f = self.fg_rgba[:, :, 3:4].astype(np.float32) / 255.0
        canvas = (canvas.astype(np.float32) * (1.0 - al_f) + self.fg_rgba[:, :, :3].astype(np.float32) * al_f).astype(np.uint8)

        if cam_zoom != 1.0 or cam_pan_xy != (0.0, 0.0):
            canvas = camera_ken_burns(
                canvas,
                0.5,
                zoom0=cam_zoom,
                zoom1=cam_zoom,
                pan_x0=cam_pan_xy[0],
                pan_x1=cam_pan_xy[0],
                pan_y0=cam_pan_xy[1],
                pan_y1=cam_pan_xy[1],
            )
        return canvas


def build_listener_reaction_face(
    u: float,
    reaction_kind: str = "girl_disbelief_1",
) -> Face:
    """Build an expressive silent listener `Face` state (mouth closed `open=0.0`) for Reaction Cutaways."""
    blink = 1.0 - 0.85 * math.exp(-((u - 0.52) / 0.065) ** 2)
    if reaction_kind == "girl_disbelief_1":
        look_x = -0.22 + 0.10 * math.sin(math.pi * u)
        look_y = -0.05 + 0.06 * math.cos(math.pi * u)
        return Face(
            lid_open=0.80 * blink,
            lid_tilt=0.24,
            look=(look_x, look_y),
            brow_up=0.72,
            brow_anger=0.25,
            furrow=0.30,
            mouth={"vis": "M", "open": 0.0, "wide": 0.15, "corner": -0.22, "skew": 0.18},
        )
    elif reaction_kind == "girl_skeptical_2":
        look_x = -0.16 + 0.08 * math.cos(math.pi * u)
        return Face(
            lid_open=0.56 * blink,
            lid_tilt=0.28,
            lower=0.35,
            look=(look_x, -0.02),
            brow_up=-0.10,
            brow_anger=0.65,
            furrow=0.65,
            mouth={"vis": "M", "open": 0.0, "wide": 0.10, "corner": -0.28, "skew": 0.22},
        )
    else:  # boy_embarrassed
        look_x = 0.14 - 0.12 * math.sin(math.pi * u)
        look_y = -0.10 + 0.08 * u
        return Face(
            lid_open=0.82 * blink,
            lid_tilt=-0.15,
            look=(look_x, look_y),
            brow_up=0.60,
            brow_worry=0.78,
            mouth={"vis": "M", "open": 0.0, "wide": 0.35, "corner": 0.26, "skew": -0.12},
        )
