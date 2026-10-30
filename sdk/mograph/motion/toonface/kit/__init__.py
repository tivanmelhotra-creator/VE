"""
toonface.kit — v31.3 hand-authored 2.5D multi-view Bezier kit + DIS/RBF turn compositor + video tracker.
"""
from .acting import HeadPose, MoodKey, Performance, energy_envelope
from .blankplate import BlankPlate, load_blank
from .g2p_fa import g2p
from .rig25 import Face, Kit, brow_line, draw_face, mouth_geom
from .sheet25 import EXPRESSIONS_18, render_18_sheet
from .turn25 import Character, lock_view
from .videotrack import TrackedHeadFrame, composite_face_on_tracked_frame, detect_moving_head_pose
from .visemes import VISEMES, mix, mouth_at, phonemes, vis, viseme_track

__version__ = "31.3"
