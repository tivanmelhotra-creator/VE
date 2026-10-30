"""Freeze the hand-authored multi-view kit data (x4 plate coordinates, crop = sheet[105:400, x0:x0+215]).
Eyes/pupils/lid weights: measured on the original sheets (measure_kit.py, reviewed overlay).
Brows / nose / mouth / furrow / lower lashes: traced by hand from the measurement grids.
Every part is later resampled to a FIXED point count (rig25) → front ↔ 3/4 morph without popping."""
import json, os
import numpy as np

M = json.load(open("/home/user/work_tf/out/measured.json"))
OUT = os.path.join(os.path.dirname(__file__), "../../mograph/motion/toonface/kit/data")


def eye(m, outer, depth, lid_scale, lid_min):
    a, b = m["lid_run"]
    th_outer = a if outer < 0 else b
    th_inner = b if outer < 0 else a
    return {"outline": m["outline"], "pupil": m["pupil"], "pupil_r": m["pupil_r"], "outer": outer, "depth": depth,
            "lid_th": [max(lid_min, th_inner * lid_scale), max(lid_min + 2, th_outer * lid_scale)]}


girl = {
    "name": "girl", "stroke": 5.5, "lashes": True, "lower_lashes": True, "wing": True, "lower_w": 0.62,
    "hair_bgr": [202, 233, 252],
    "plates": {"front": "gf_blank.png", "34": "g34_blank.png"},
    "chin": {"front": [431.4, 729], "34": [504.6, 725]},
    "skin_seed": {"front": [430, 610], "34": [470, 610]},
    "contour_cut_y": {"front": 380, "34": 392},
    "views": {
        "front": {
            "eyes": {"L": eye(M["gf"]["eyes"]["L"], -1, 1.0, 0.85, 6), "R": eye(M["gf"]["eyes"]["R"], +1, 1.0, 0.85, 6)},
            "brows": {"L": {"line": [[249, 361], [268, 340], [295, 326], [328, 321], [362, 329]], "th": 17.0},
                      "R": {"line": [[613, 361], [594, 340], [567, 326], [534, 321], [500, 329]], "th": 17.0}},
            "nose": [[431, 523], [416, 524], [406, 535], [408, 550], [418, 562]],
            "mouth": {"left": [377, 612], "right": [512, 586], "low": [425, 622], "peak": 0.42, "wl": 1.0, "wr": 1.0},
            "furrow": [[425, 322], [431, 352]],
        },
        "34": {
            "eyes": {"L": eye(M["g34"]["eyes"]["L"], -1, 1.0, 0.85, 6), "R": eye(M["g34"]["eyes"]["R"], +1, 0.85, 0.85, 5)},
            "brows": {"L": {"line": [[390, 364], [418, 346], [458, 335], [502, 337], [538, 353]], "th": 19.0},
                      "R": {"line": [[724, 352], [708, 340], [684, 336], [660, 342], [643, 355]], "th": 15.0}},
            "nose": [[607, 537], [627, 531], [645, 539], [650, 553], [638, 568]],
            "mouth": {"left": [519, 605], "right": [647, 628], "low": [600, 637], "peak": 0.45, "wl": 1.0, "wr": 0.72},
            "furrow": [[588, 325], [596, 355]],
        },
    },
}

boy = {
    "name": "boy", "stroke": 7.0, "lashes": False, "lower_lashes": False, "wing": False, "lower_w": 1.0,
    "hair_bgr": [202, 233, 252],
    "plates": {"front": "bf_blank.png", "34": "b34_blank.png"},
    "chin": {"front": [426.9, 729], "34": [529.3, 725]},
    "skin_seed": {"front": [430, 600], "34": [470, 600]},
    "contour_cut_y": {"front": 375, "34": 380},
    "views": {
        "front": {
            "eyes": {"L": eye(M["bf"]["eyes"]["L"], -1, 1.0, 1.0, 7), "R": eye(M["bf"]["eyes"]["R"], +1, 1.0, 1.0, 7)},
            "brows": {"L": {"line": [[251, 364], [272, 343], [300, 330], [330, 326], [362, 336]], "th": 19.0},
                      "R": {"line": [[609, 364], [588, 343], [560, 330], [530, 326], [498, 336]], "th": 19.0}},
            "nose": [[405, 547], [415, 536], [430, 532], [445, 536], [455, 547]],
            "mouth": {"left": [362, 612], "right": [517, 592], "low": [410, 622], "peak": 0.40, "wl": 1.0, "wr": 1.0},
            "furrow": [[424, 330], [430, 360]],
        },
        "34": {
            "eyes": {"L": eye(M["b34"]["eyes"]["L"], -1, 1.0, 1.0, 7), "R": eye(M["b34"]["eyes"]["R"], +1, 0.85, 1.0, 6)},
            "brows": {"L": {"line": [[400, 362], [425, 340], [458, 322], [492, 312], [521, 316]], "th": 21.0},
                      "R": {"line": [[712, 350], [698, 340], [680, 336], [662, 339], [648, 346]], "th": 15.0}},
            "nose": [[590, 538], [604, 528], [621, 526], [634, 532], [640, 543]],
            "mouth": {"left": [520, 607], "right": [637, 618], "low": [595, 630], "peak": 0.55, "wl": 1.0, "wr": 0.74},
            "furrow": [[585, 318], [592, 348]],
        },
    },
}

for k in (girl, boy):
    json.dump(k, open(os.path.join(OUT, k["name"] + ".json"), "w"), indent=1)
    print("wrote", k["name"])
