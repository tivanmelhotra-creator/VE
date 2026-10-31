"""
toonface.kit — v31.4 hand-authored 2.5D multi-view Bezier kit + DIS/RBF turn compositor + Welded Skull 3-Layer Sandwich.
"""
from .acting import HeadPose, MoodKey, Performance, energy_envelope
from .blankplate import BlankPlate, load_blank
from .g2p_fa import g2p
from .rig25 import Face, Kit, brow_line, draw_face, mouth_geom, tongue_inside_poly, tongue_out_poly
from .sandwich import (
    ThreeLayerSandwichRig,
    WeldedSkullSandwichRig,
    align_bg_chair_behind_character,
    build_listener_reaction_face,
    build_pingpong_gesture_schedule,
    camera_ken_burns,
    extract_amber_matte,
    load_stepped_video_range,
    slice_three_panels,
)
from .sheet25 import EXPRESSIONS_18, render_18_sheet
from .turn25 import Character, lock_view
from .videotrack import TrackedHeadFrame, composite_face_on_tracked_frame, detect_moving_head_pose
from .visemes import VISEMES, mix, mouth_at, phonemes, vis, viseme_track

__version__ = "31.4"
from .expressive import EMOTIONS, Beat, ExpressivePerformer, FaceFrame, au, blend
from .facecomp import FaceRigCompositor, LevelPolicy
from .facetrack import HeadTracker, OneEuro, PoseLimits, TrackerConfig

__version__ = "32.0"
