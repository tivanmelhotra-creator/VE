"""
Full-HD ink-black / studio-white line-art remaster for flat 2D cartoon art (stills and videos).

  Lanczos-4 cover-fit → bilateral de-blocking → fine unsharp → INK S-curve on the HSV *value* channel (so dark
  brown / grey lines go to #0A0A0C but saturated bright colours — pink tongue, orange props — are untouched;
  v29.2 used luma, which partially blackened tongues) → low-saturation off-white fills pushed to studio white.

API: enhance_lineart_hd(img, target_wh=None, ...) → bgr · enhance_video_hd(in_mp4, out_mp4, target_wh) → report
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .media import FFmpegWriter, cover_fit, has_audio, video_info

INK = np.array([10.0, 10.0, 12.0], np.float32)          # #0C0A0A (BGR) deep studio ink
STUDIO_WHITE = np.array([250.0, 252.0, 252.0], np.float32)


def enhance_lineart_hd(
    img_bgr: np.ndarray,
    target_wh: tuple[int, int] | None = None,
    ink_gamma: float = 0.52,
    ink_dark_cutoff: int = 155,
    whiten_character: bool = True,
    sharpen: float = 0.45,
    denoise: bool = True,
) -> np.ndarray:
    """
    Remaster blurry / compressed line art: optional cover-fit to target_wh, de-block, sharpen, ink-black lines,
    studio-white fills. Colours with high saturation AND high value are protected from the ink curve.
    """
    work = cover_fit(img_bgr, target_wh) if target_wh is not None else img_bgr.copy()
    if denoise:
        d = 5 if max(work.shape[:2]) <= 2200 else 7
        work = cv2.bilateralFilter(work, d=d, sigmaColor=24, sigmaSpace=24)
    if sharpen > 0:
        blur = cv2.GaussianBlur(work, (0, 0), 1.1)
        work = cv2.addWeighted(work, 1.0 + sharpen, blur, -sharpen, 0)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV).astype(np.float32)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    out = work.astype(np.float32)
    if whiten_character:
        w = np.clip((v - 175.0) / 50.0, 0, 1) * np.clip((52.0 - s) / 30.0, 0, 1)
        out += (STUDIO_WHITE[None, None, :] - out) * (0.85 * w)[:, :, None]
    t = np.clip((float(ink_dark_cutoff) - v) / float(max(1, ink_dark_cutoff - 25)), 0.0, 1.0)
    ink_w = (t * t * (3.0 - 2.0 * t)) ** ink_gamma
    out += (INK[None, None, :] - out) * ink_w[:, :, None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def enhance_video_hd(
    in_mp4: str,
    out_mp4: str,
    target_wh: tuple[int, int] = (1080, 1920),
    ink_gamma: float = 0.52,
    crf: int = 18,
) -> dict[str, Any]:
    """Remaster every frame of a video to target_wh (audio stream copied/encoded in the same ffmpeg pass)."""
    info = video_info(in_mp4)
    cap = cv2.VideoCapture(in_mp4)
    n = 0
    with FFmpegWriter(out_mp4, target_wh, info["fps"], audio=in_mp4 if has_audio(in_mp4) else None, crf=crf) as w:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            w.write(enhance_lineart_hd(frame, target_wh=target_wh, ink_gamma=ink_gamma))
            n += 1
    cap.release()
    if n == 0:
        raise ValueError(f"no frames decoded from {in_mp4}")
    return {"out_mp4": out_mp4, "frames": n, "fps": info["fps"], "resolution": f"{target_wh[0]}x{target_wh[1]}"}
