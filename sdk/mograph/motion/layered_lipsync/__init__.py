"""
mograph.motion.layered_lipsync — 3-layer dual-track lip-sync + in-place video lip-sync + v29.9 Sculpted Almond 3D-Tracked Elastic Face.

Layer 0  background (remastered once at output resolution)
Layer 1  character: BODY track (pose per frame) + MOUTH/FACE track (viseme / sculpted almond elastic splines per frame) + auto blinks + alive body wave
Layer 2  foreground props (desk / mic — auto occlusion split: hands in front, legs behind)

Primary entry points:
    render(bg, poses, audio, out, ...)       still-plate 3-layer render (native 1080×1920, single libx264+AAC pass)
    render_video_lipsync(video, out, ...)    in-place lip-sync on a moving character video (EMA + temporal hint)
    viseme_sheet(pose, out, ...)             render all 10 visemes on one pose (visual QC before a long render)
    contact_sheet(mp4, out)                  8 evenly-spaced frames of an MP4 into one image

Legacy v29.2 names (`render_layered_lipsync`, `prepare_aligned_clean_poses`, …) are re-exported from `.compat`.
Guide: sdk/MOTION_GUIDE.md §Layered lip-sync
"""
from __future__ import annotations

from .anatomy import anatomy_summary, debug_anatomy_image, detect_face_anatomy, find_heads
from .face_rig import EYE_EXPRESSIONS, MOUTH_EXPRESSIONS, RigParams, draw_face, face_ops, fit_rig, prepare_face, project
from .elastic_face import (
    AESTHETIC_PRESETS,
    AESTHETIC_STYLE_PROFILES,
    AL_EXPRESSIONS,
    AestheticEyeSpec,
    AestheticMouthSpec,
    ElasticEyeSpec,
    ElasticMouthSpec,
    HeadPose3D,
    apply_smart_acting_turn,
    draw_canonical_eyes_and_brows,
    draw_canonical_mouth,
    dynamic_speech_mouth_spec,
    project_canonical,
    render_canonical_face_on_pose,
    prepare_hybrid_character_rig,
    render_elastic_face_on_pose,
    render_hybrid_character_expression,
    track_head_pose_3d,
)
from .compat import (extract_fg_desk_rgba, extract_stickman_rgba, extract_viseme_grid_sprites,
                     place_viseme_on_pose, prepare_aligned_clean_poses, qa_mouth_placement,
                     render_layered_lipsync, slice_poses_sheet)
from .layers import character_alpha, occlusion_line, paper_color, props_alpha, split_pose_sheet
from .media import cover_fit, has_audio, load_image, read_audio_mono, video_info
from .mouths import VISEMES, GridMouths, VectorMouths, draw_vector_mouth, erase_base_mouth, place_mouth
from .remaster import enhance_lineart_hd
from .render import (apply_alive_motion, apply_blink, audio_viseme_track, auto_body_timeline, blink_schedule,
                     contact_sheet, render, render_video_lipsync, speech_mask_from_keys, viseme_sheet)

__version__ = "29.10"

__all__ = [
    "RigParams", "fit_rig", "prepare_face", "face_ops", "draw_face", "project", "EYE_EXPRESSIONS", "MOUTH_EXPRESSIONS",
    "HeadPose3D", "track_head_pose_3d", "project_canonical", "apply_smart_acting_turn",
    "AestheticEyeSpec", "AestheticMouthSpec", "ElasticEyeSpec", "ElasticMouthSpec",
    "AESTHETIC_PRESETS", "AESTHETIC_STYLE_PROFILES", "AL_EXPRESSIONS",
    "draw_canonical_eyes_and_brows", "draw_canonical_mouth",
    "render_canonical_face_on_pose", "render_elastic_face_on_pose", "prepare_hybrid_character_rig", "render_hybrid_character_expression", "dynamic_speech_mouth_spec",
    "detect_face_anatomy", "debug_anatomy_image", "anatomy_summary", "find_heads",
    "character_alpha", "props_alpha", "occlusion_line", "paper_color", "split_pose_sheet",
    "VISEMES", "VectorMouths", "GridMouths", "draw_vector_mouth", "place_mouth", "erase_base_mouth",
    "render", "render_video_lipsync", "audio_viseme_track", "auto_body_timeline", "blink_schedule", "apply_blink",
    "apply_alive_motion", "speech_mask_from_keys", "viseme_sheet", "contact_sheet",
    "enhance_lineart_hd", "load_image", "cover_fit", "read_audio_mono", "video_info", "has_audio",
    "extract_stickman_rgba", "extract_fg_desk_rgba", "slice_poses_sheet", "prepare_aligned_clean_poses",
    "extract_viseme_grid_sprites", "place_viseme_on_pose", "qa_mouth_placement", "render_layered_lipsync",
]
