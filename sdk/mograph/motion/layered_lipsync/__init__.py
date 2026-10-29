"""
mograph.motion.layered_lipsync — v29.3 universal 2D layered lip-sync: anatomy detector, 3-layer dual-track
renderer, procedural / drawn mouths, words.json or audio-only drivers, blinks, video mode, Full-HD ink remaster.

Modules
  anatomy   head / eye-pair / mouth landmarks (front · 3/4 · profile · tilted · closed eyes), convex-hull scoring
  layers    paper colour, character alpha, props alpha, auto occlusion line, pose-sheet splitter
  mouths    VectorMouths (procedural, blended) · GridMouths (drawn sheet, auto viseme mapping) · place / erase
  render    render() 3-layer MP4 · render_video_lipsync() moving character · viseme_sheet() · contact_sheet()
  remaster  enhance_lineart_hd() / enhance_video_hd() ink-black + studio-white Full-HD remaster
  media     image / audio I/O, cover-fit, single-pass ffmpeg writer
  compat    the v29.2 function names (render_layered_lipsync, place_viseme_on_pose, …) on the new engine

CLI: python3 -m mograph.motion layered-lipsync {detect|sheet|render|video|enhance-hd|contact} …   (see --help)
Guide: sdk/MOTION_GUIDE.md §Layered lip-sync
"""
from __future__ import annotations

from .anatomy import anatomy_summary, debug_anatomy_image, detect_face_anatomy, find_heads
from .compat import (extract_fg_desk_rgba, extract_stickman_rgba, extract_viseme_grid_sprites,
                     place_viseme_on_pose, prepare_aligned_clean_poses, qa_mouth_placement,
                     render_layered_lipsync, slice_poses_sheet)
from .layers import character_alpha, occlusion_line, paper_color, props_alpha, split_pose_sheet
from .mouths import VISEMES, GridMouths, VectorMouths, draw_vector_mouth, erase_base_mouth, place_mouth
from .remaster import enhance_lineart_hd, enhance_video_hd
from .render import (apply_blink, audio_viseme_track, auto_body_timeline, blink_schedule, contact_sheet, render,
                     render_video_lipsync, viseme_sheet)

__version__ = "29.3"

__all__ = [
    "detect_face_anatomy", "debug_anatomy_image", "anatomy_summary", "find_heads",
    "character_alpha", "props_alpha", "occlusion_line", "paper_color", "split_pose_sheet",
    "VISEMES", "VectorMouths", "GridMouths", "draw_vector_mouth", "place_mouth", "erase_base_mouth",
    "enhance_lineart_hd", "enhance_video_hd",
    "render", "render_video_lipsync", "audio_viseme_track", "auto_body_timeline", "blink_schedule", "apply_blink",
    "viseme_sheet", "contact_sheet",
    "render_layered_lipsync", "place_viseme_on_pose", "extract_viseme_grid_sprites", "extract_stickman_rgba",
    "extract_fg_desk_rgba", "slice_poses_sheet", "prepare_aligned_clean_poses", "qa_mouth_placement",
]
