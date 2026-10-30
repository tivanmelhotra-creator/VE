"""
toonface.kit.shapes — HAND-AUTHORED key-shapes for our two characters (coordinates measured on the x4 upscaled
character sheet crop: sheet[105:400, 385:600] × 4; same frame as the blank plate supplied by the art team).

Every part keeps a FIXED point count across keys (CANDIDE-style fixed topology) so the renderer can morph between
yaw keys and expressions without popping.  Only the 3/4-right key (yaw +25°) is authored for the v31 quality gate;
front / profile keys plug into the same tables.

Conventions
  eye "outline"   closed polygon, clockwise from the top, smoothed by Catmull-Rom at draw time
  eye "outer"     -1 → the outer corner is on screen-left (near eye of a right-facing 3/4), +1 → screen-right
  brow "line"     centre line from OUTER end to INNER end
  mouth           left / right corners of the neutral line; "near" = side closer to the camera (longer half)
"""
from __future__ import annotations

INK = (0.204, 0.114, 0.071)           # #341D12  (BGR 18, 29, 53)
SCLERA = (0.985, 0.958, 0.885)        # warm cream, never #FFFFFF
PUPIL = (0.145, 0.082, 0.063)
GLINT = (1.0, 1.0, 1.0)
CAVITY = (0.42, 0.075, 0.115)         # dark crimson
CAVITY_DEEP = (0.29, 0.045, 0.075)
TONGUE = (0.93, 0.47, 0.47)
TEETH = (1.0, 0.995, 0.975)

GIRL_34 = {
    "name": "girl_34R",
    "yaw_key": 25.0,
    "stroke": 5.5,
    "hair": (0.988, 0.914, 0.792),            # BGR 202, 233, 252
    "skin": (1.0, 0.765, 0.561),              # blank plate skin (BGR 143, 195, 255)
    "lashes": True,
    "neck_pivot": (560.0, 735.0),
    "eyes": {
        "near": {
            "outline": [(470, 398), (512, 403), (542, 420), (556, 448), (558, 485), (553, 520), (540, 546),
                        (498, 553), (450, 550), (410, 536), (388, 507), (383, 472), (395, 438), (428, 410)],
            "outer": -1,
            "pupil": (520.0, 477.0), "pupil_r": (24.0, 34.0),
            "lid_th": (5.0, 17.0),
            "depth": 1.0,
        },
        "far": {
            "outline": [(688, 397), (710, 408), (722, 435), (725, 475), (719, 512), (702, 537), (672, 543),
                        (646, 532), (634, 503), (631, 465), (636, 430), (656, 405)],
            "outer": +1,
            "pupil": (656.0, 472.0), "pupil_r": (17.0, 32.0),
            "lid_th": (4.0, 14.0),
            "depth": 0.85,
        },
    },
    "brows": {
        "near": {"line": [(390, 364), (418, 346), (458, 335), (502, 337), (538, 353)], "th": 19.0, "outer": -1},
        "far": {"line": [(724, 352), (708, 340), (684, 336), (660, 342), (643, 355)], "th": 15.0, "outer": +1},
    },
    "nose": {"line": [(607, 537), (627, 530), (646, 539), (651, 554), (637, 569)], "nostril": (628.0, 556.0)},
    "mouth": {"left": (522.0, 607.0), "right": (642.0, 628.0), "near": "left", "asym": 0.70, "scale": 0.86},
    "furrow": [(590, 330), (600, 360)],
}

BOY_34 = None   # authored after the quality gate is approved (same tables, measured on the boy sheet)
