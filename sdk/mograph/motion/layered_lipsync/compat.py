"""
v29.2 API compatibility layer — every public function of the old single-file `layered_lipsync.py` keeps its name
and signature, implemented on top of the v29.3 package (so old scripts / agent prompts keep working).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .anatomy import detect_face_anatomy
from .layers import character_alpha, props_alpha, split_pose_sheet
from .mouths import GridMouths, place_mouth


def qa_mouth_placement(anatomy: dict[str, Any], dst_x0: int, dst_y0: int, sw_s: int, sh_s: int) -> dict[str, Any]:
    """v29.2 box-based QA: box below the eyes, centre above the chin, eye-mask overlap count."""
    H, W = anatomy["eye_mask_full"].shape[:2]
    ya, yb = max(0, dst_y0), min(H, dst_y0 + sh_s)
    xa, xb = max(0, dst_x0), min(W, dst_x0 + sw_s)
    eye_ov = int(np.sum(anatomy["eye_mask_full"][ya:yb, xa:xb] > 0))
    hx, hy, hw, hh = anatomy["head_box"]
    m_cy = dst_y0 + sh_s / 2.0
    below = bool(dst_y0 >= anatomy["eyes_bottom_y"] + 2)
    above = bool(m_cy < hy + hh - 3)
    return {"ok": below and above, "below_eyes": below, "above_chin": above, "eye_overlap_cropped": eye_ov,
            "mouth_cy": float(m_cy), "eyes_bottom_y": int(anatomy["eyes_bottom_y"]), "chin_y": int(hy + hh)}


def extract_viseme_grid_sprites(grid_bgr: np.ndarray, rows: int = 4, cols: int = 4) -> list[dict[str, Any]]:
    """v29.2: list of {index, bgr, mask, inter_ocular, eye_angle_deg, mouth_center} (one per usable cell)."""
    gm = GridMouths(grid_bgr, rows=rows, cols=cols)
    out = []
    for s in gm.sprites:
        if s is None:
            continue
        out.append(dict(s, mask=s["alpha"], mouth_center=s["anchor"]))
    return out


def extract_stickman_rgba(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v29.2: (bgr, alpha) of the character on flat paper."""
    return img_bgr, character_alpha(img_bgr)


def extract_fg_desk_rgba(fg_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v29.2: (bgr, alpha) of the foreground props."""
    return fg_bgr, props_alpha(fg_bgr)


def place_viseme_on_pose(clean_pose_bgr: np.ndarray, pose_anatomy: dict[str, Any], viseme_sprite: dict[str, Any],
                         scale_mult: float = 0.92) -> tuple[np.ndarray, dict[str, Any]]:
    """v29.2: place a sprite from extract_viseme_grid_sprites()."""
    sp = dict(viseme_sprite)
    sp.setdefault("alpha", sp.get("mask"))
    sp.setdefault("anchor", (sp["bgr"].shape[1] / 2.0, sp["bgr"].shape[0] / 2.0))
    return place_mouth(clean_pose_bgr, pose_anatomy, sp, scale_mult=scale_mult)


def slice_poses_sheet(poses_bgr: np.ndarray, num_poses: int = 3, target_wh: tuple[int, int] = (720, 1280)):
    """v29.2: split a horizontal pose sheet, centre-crop each panel to the target aspect and resize."""
    tw, th = target_wh
    out = []
    for p in split_pose_sheet(poses_bgr, num_poses):
        ph, pw = p.shape[:2]
        nw = min(pw, int(ph * tw / th))
        x0 = max(0, (pw - nw) // 2)
        out.append(cv2.resize(p[:, x0:x0 + nw], (tw, th), interpolation=cv2.INTER_CUBIC))
    return out


def prepare_aligned_clean_poses(pose_panels, ref_pose_bgr=None, head_roi_y=None, head_roi_x=None):
    """v29.2: (clean_poses, pose_infos, pose_alphas) registered on the reference eye midpoint / ruler."""
    from .mouths import erase_base_mouth
    from .render import _align_to
    h, w = pose_panels[0].shape[:2]
    ref = detect_face_anatomy(ref_pose_bgr if ref_pose_bgr is not None else pose_panels[0], head_roi_y, head_roi_x)
    cl, infos, alphas = [], [], []
    for p in pose_panels:
        inf = detect_face_anatomy(p, head_roi_y, head_roi_x)
        al, _ = _align_to(ref, p, None, inf, (w, h))
        ai = detect_face_anatomy(al, head_roi_y, head_roi_x, hint=ref["eyes_mid"])
        er = erase_base_mouth(al, ai)
        cl.append(er)
        infos.append(ai)
        alphas.append(character_alpha(er))
    return cl, infos, alphas


def render_layered_lipsync(bg_path, poses_path, visemes_path, fg_path, audio_path, out_mp4, ref_char_path=None,
                           body_timeline=None, mouth_timeline=None, fps=24, work_wh=(720, 1280), hd_remaster=True,
                           out_wh=(1080, 1920)) -> dict[str, Any]:
    """v29.2 signature → v29.3 render(): 3-panel pose sheet + 4×4 viseme grid (integer mouth_timeline = cell ids)."""
    from .render import render
    rep = render(bg=bg_path, poses=poses_path, audio=audio_path, out=out_mp4, fg=fg_path, visemes=visemes_path,
                 mouth_timeline=mouth_timeline, body_timeline=body_timeline, pose_sheet=3, ref=ref_char_path,
                 fps=fps, wh=out_wh if hd_remaster else work_wh, remaster=hd_remaster)
    rep["hd_remaster"] = hd_remaster
    rep["view_mode"] = rep["poses"][0]["view_mode"]
    rep["mouth_center_pose0"] = tuple(rep["poses"][0]["mouth_center"])
    rep["eyes_bottom_y_pose0"] = rep["poses"][0]["eyes_bottom_y"]
    return rep
