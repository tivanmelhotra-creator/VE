"""
toonface.controls — a FACS-style control vector for a toon face + the expression and viseme libraries.

Nothing in here is a picture.  An expression is ~40 numbers (lid heights, brow raise / anger / worry, mouth width,
opening, smile, pucker, teeth, tongue, asymmetry, gaze, FX intensities).  Every character, every view and every style
reads the SAME numbers — that is what lets one library drive a male, a female, a profile, a 3/4 view or a character
drawn tomorrow.

Mixing rules
  * continuous fields blend linearly; speech visemes are laid ON TOP of the acting mouth (apply_speech)
  * discrete glyph modes (eye: happy ∩ / squeeze >< / spiral / star / closed; mouth: grit / wavy) are named by the
    heaviest non-default contributor and faded by the numeric eye_glyph / mouth_glyph weight — the renderer swaps
    eye glyphs THROUGH a blink, never with a dissolve
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Iterable

EYE_MODES = ("normal", "happy", "squeeze", "spiral", "star", "closed")
MOUTH_MODES = ("shape", "grit", "wavy")
FX_KEYS = ("blush", "sweat", "tears", "zzz", "idea", "stress", "dizzy", "notes", "cold", "shiver", "crumbs",
           "puff", "sparkle", "anger_vein")


@dataclass
class FaceControls:
    # eyes (l = screen-left eye, r = screen-right eye)
    lid_l: float = 1.0          # upper-lid openness 0 closed · 1 natural · 1.3 wide (lid lifted off the iris)
    lid_r: float = 1.0
    low_l: float = 0.0          # lower-lid raise (cheek squint) 0..1
    low_r: float = 0.0
    lid_tilt: float = 0.0       # + angry (inner lid corner down) · − sad / pleading (outer corner down)
    pupil: float = 1.0          # 0.45 shocked pin-point … 1.75 pleading puppy
    sparkle: float = 0.0        # glossy catch-lights 0..1
    eye_scale: float = 1.0
    gaze_x: float = 0.0         # −1..1 screen left/right (in eye radii)
    gaze_y: float = 0.0         # −1 up … 1 down
    eye_mode: str = "normal"
    eye_glyph: float = 0.0
    # brows
    brow_l: float = 0.0         # raise (+ up)
    brow_r: float = 0.0
    brow_anger: float = 0.0
    brow_worry: float = 0.0
    brow_squeeze: float = 0.0
    # mouth
    jaw: float = 0.0
    width: float = 1.0
    smile: float = 0.25
    upper_lip: float = 0.0
    pucker: float = 0.0
    teeth_up: float = 0.0
    teeth_low: float = 0.0
    tongue: float = 0.0
    tongue_up: float = 0.0
    lip_bite: float = 0.0
    press: float = 0.0
    asym: float = 0.0           # −1..1 smirk (+ = screen-right corner up)
    shift_x: float = 0.0
    wobble: float = 0.0
    mouth_mode: str = "shape"
    mouth_glyph: float = 0.0
    # head acting (degrees, added to the plate / rig pose)
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    squash: float = 0.0
    fx: dict = field(default_factory=dict)

    def copy(self, **kw) -> "FaceControls":
        c = replace(self, fx=dict(self.fx))
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def get_fx(self, key: str) -> float:
        return float(self.fx.get(key, 0.0))


NUMERIC = tuple(f.name for f in fields(FaceControls) if f.type in ("float", float))
NEUTRAL = FaceControls()

EXPRESSIONS: dict[str, dict] = {
    "neutral":    {},
    "smile":      dict(smile=0.6, low_l=0.12, low_r=0.12, brow_l=0.15, brow_r=0.15),
    "happy":      dict(smile=0.85, jaw=0.4, width=1.15, teeth_up=0.7, tongue=0.55, low_l=0.25, low_r=0.25,
                       brow_l=0.45, brow_r=0.45),
    "laugh":      dict(eye_mode="happy", smile=1.0, jaw=0.75, width=1.25, teeth_up=0.8, tongue=0.7,
                       brow_l=0.6, brow_r=0.6, head_pitch=-6, squash=0.15),
    "sad":        dict(smile=-0.55, lid_l=0.75, lid_r=0.75, lid_tilt=-0.6, brow_worry=0.85, gaze_y=0.35),
    "angry":      dict(smile=-0.45, lid_l=0.78, lid_r=0.78, lid_tilt=0.75, brow_anger=1.0, brow_l=-0.4,
                       brow_r=-0.4, jaw=0.2, width=1.05, teeth_up=0.6, teeth_low=0.5, pupil=0.85),
    "surprised":  dict(lid_l=1.25, lid_r=1.25, pupil=0.75, brow_l=1.2, brow_r=1.2, jaw=0.55, width=0.62,
                       pucker=0.55, smile=0.0, squash=-0.1),
    "wink":       dict(lid_r=0.0, smile=0.75, asym=0.5, brow_r=-0.3, brow_l=0.35, low_r=0.4),
    "talk_calm":  dict(smile=0.35, brow_l=0.12, brow_r=0.12),
    # ---- user reference sheet 1
    "roaring":    dict(eye_mode="squeeze", brow_anger=1.0, brow_l=-0.5, brow_r=-0.5, jaw=1.0, width=1.25,
                       smile=-0.15, teeth_up=0.7, tongue=0.8, fx={"stress": 1.0}, squash=0.25, head_pitch=-5),
    "star_eyed":  dict(eye_mode="star", smile=0.95, jaw=0.6, width=1.3, teeth_up=0.85, tongue=0.6,
                       brow_l=0.9, brow_r=0.9, fx={"sparkle": 0.6}),
    "terrified":  dict(lid_l=1.3, lid_r=1.3, eye_scale=1.1, pupil=0.5, brow_worry=1.0, brow_l=0.8, brow_r=0.8,
                       mouth_mode="grit", jaw=0.45, width=1.2, smile=-0.35, fx={"sweat": 1.0}, squash=-0.08),
    "pleading":   dict(pupil=1.75, sparkle=1.0, lid_l=1.1, lid_r=1.1, brow_worry=0.9, brow_l=0.35, brow_r=0.35,
                       smile=0.45, width=0.55, gaze_y=-0.15),
    "shy":        dict(lid_l=0.8, lid_r=0.8, lid_tilt=-0.35, brow_worry=0.75, gaze_x=0.45, gaze_y=0.3,
                       mouth_mode="wavy", wobble=0.8, width=0.62, smile=-0.1, fx={"blush": 1.0, "sweat": 0.6},
                       head_pitch=6),
    "smug":       dict(lid_l=0.55, lid_r=0.55, low_l=0.15, low_r=0.15, gaze_x=0.35, brow_l=0.55, brow_r=-0.2,
                       smile=0.45, asym=0.85, shift_x=0.04, width=0.9, head_roll=-4),
    # ---- user reference sheet 2
    "shocked":    dict(lid_l=1.3, lid_r=1.3, eye_scale=1.12, pupil=0.42, brow_l=1.4, brow_r=1.4, brow_worry=0.3,
                       mouth_mode="grit", jaw=0.62, width=0.9, smile=-0.3, fx={"stress": 1.0, "shiver": 0.6},
                       squash=-0.15),
    "dizzy":      dict(eye_mode="spiral", brow_worry=0.6, brow_l=0.2, brow_r=0.2, mouth_mode="wavy", wobble=1.0,
                       width=0.8, smile=-0.1, fx={"dizzy": 1.0}, head_roll=6),
    "mischievous": dict(lid_l=0.62, lid_r=0.62, lid_tilt=0.65, brow_anger=0.75, brow_l=0.25, brow_r=0.25,
                        smile=1.0, width=1.35, jaw=0.24, teeth_up=1.0, teeth_low=1.0, mouth_mode="grit",
                        gaze_x=-0.25, low_l=0.3, low_r=0.3),
    "yawning":    dict(eye_mode="closed", brow_worry=0.55, brow_l=0.3, brow_r=0.3, jaw=0.9, width=0.62,
                       pucker=0.4, tongue=0.6, smile=-0.05, fx={"zzz": 1.0}, head_pitch=-8, squash=0.1),
    "thinking":   dict(lid_l=0.62, lid_r=0.62, gaze_x=0.6, gaze_y=-0.45, brow_l=0.65, brow_r=-0.25,
                       smile=0.3, asym=0.6, width=0.75, head_roll=5),
    "weeping":    dict(eye_mode="squeeze", brow_worry=1.0, brow_l=0.5, brow_r=0.5, jaw=0.65, width=1.05,
                       smile=-0.75, teeth_up=0.55, tongue=0.6, fx={"tears": 1.0}),
    # ---- user reference sheet 3
    "eating":     dict(eye_mode="happy", brow_l=0.35, brow_r=0.35, smile=0.4, width=0.55, press=1.0,
                       shift_x=-0.03, fx={"puff": 1.0, "crumbs": 1.0}),
    "disgusted":  dict(eye_mode="squeeze", brow_anger=0.55, brow_squeeze=1.0, brow_l=-0.3, brow_r=-0.3,
                       mouth_mode="wavy", wobble=0.9, width=0.82, smile=-0.6, upper_lip=0.5, low_l=0.6,
                       low_r=0.6, asym=-0.3),
    "eureka":     dict(lid_l=1.25, lid_r=1.25, pupil=0.8, brow_l=1.2, brow_r=1.2, jaw=0.55, width=0.85,
                       smile=0.5, teeth_up=0.4, tongue=0.6, gaze_x=0.35, gaze_y=-0.3, fx={"idea": 1.0}),
    "sulking":    dict(lid_l=0.55, lid_r=0.55, lid_tilt=0.7, brow_anger=0.9, brow_l=-0.3, brow_r=-0.3,
                       gaze_x=0.55, smile=-0.7, width=0.6, press=0.6, head_pitch=4),
    "whistling":  dict(eye_mode="happy", brow_l=0.6, brow_r=0.6, pucker=1.0, width=0.32, jaw=0.18,
                       shift_x=0.05, fx={"notes": 1.0}, head_roll=-5),
    "freezing":   dict(lid_l=1.25, lid_r=1.25, pupil=0.55, brow_worry=1.0, brow_l=0.6, brow_r=0.6,
                       mouth_mode="grit", jaw=0.42, width=1.15, smile=-0.25, wobble=0.6,
                       fx={"cold": 1.0, "shiver": 1.0, "sweat": 0.4}),
}

SHEET_ORDER = (
    ["roaring", "star_eyed", "terrified", "pleading", "shy", "smug"],
    ["shocked", "dizzy", "mischievous", "yawning", "thinking", "weeping"],
    ["eating", "disgusted", "eureka", "sulking", "whistling", "freezing"],
)

VISEMES: dict[str, dict] = {
    "rest": dict(jaw=0.0),
    "MBP":  dict(jaw=0.0, press=1.0, width=0.92),
    "AI":   dict(jaw=0.8, width=1.12, teeth_up=0.55, tongue=0.55),
    "E":    dict(jaw=0.4, width=1.25, teeth_up=0.8, teeth_low=0.45, tongue=0.25),
    "I":    dict(jaw=0.26, width=1.22, teeth_up=0.85, teeth_low=0.7),
    "O":    dict(jaw=0.62, width=0.62, pucker=0.7, tongue=0.35),
    "U":    dict(jaw=0.25, width=0.42, pucker=1.0),
    "WQ":   dict(jaw=0.14, width=0.38, pucker=1.0),
    "FV":   dict(jaw=0.12, width=1.0, teeth_up=1.0, lip_bite=1.0),
    "L":    dict(jaw=0.45, width=1.0, teeth_up=0.7, tongue=0.45, tongue_up=1.0),
    "TH":   dict(jaw=0.2, width=1.02, teeth_up=0.8, tongue=0.6, tongue_up=0.7),
    "CDG":  dict(jaw=0.22, width=1.12, teeth_up=0.9, teeth_low=0.9),
    "SH":   dict(jaw=0.2, width=0.72, pucker=0.45, teeth_up=0.9, teeth_low=0.9),
    "R":    dict(jaw=0.24, width=0.72, pucker=0.5, teeth_up=0.4),
}
SPEECH_ABSOLUTE = ("jaw", "pucker", "teeth_up", "teeth_low", "tongue", "tongue_up", "lip_bite", "press")


def expression(name_or_dict) -> FaceControls:
    d = EXPRESSIONS[name_or_dict] if isinstance(name_or_dict, str) else name_or_dict
    c = NEUTRAL.copy()
    for k, v in d.items():
        if k == "fx":
            c.fx = dict(v)
        else:
            setattr(c, k, v)
    c.eye_glyph = 1.0 if c.eye_mode != "normal" else 0.0
    c.mouth_glyph = 1.0 if c.mouth_mode != "shape" else 0.0
    return c


def blend(items: Iterable[tuple[FaceControls, float]]) -> FaceControls:
    items = [(c, float(w)) for c, w in items if w > 1e-6]
    if not items:
        return NEUTRAL.copy()
    tot = sum(w for _, w in items)
    out = NEUTRAL.copy()
    for k in NUMERIC:
        setattr(out, k, sum(getattr(c, k) * w for c, w in items) / tot)
    eyes = [(c, w) for c, w in items if c.eye_mode != "normal"]
    out.eye_mode = max(eyes, key=lambda t: t[1])[0].eye_mode if eyes else "normal"
    mouths = [(c, w) for c, w in items if c.mouth_mode != "shape"]
    out.mouth_mode = max(mouths, key=lambda t: t[1])[0].mouth_mode if mouths else "shape"
    fx: dict[str, float] = {}
    for c, w in items:
        for k, v in c.fx.items():
            fx[k] = fx.get(k, 0.0) + v * w / tot
    out.fx = fx
    return out


def apply_speech(base: FaceControls, mouth: dict, amount: float = 1.0) -> FaceControls:
    """Lay a viseme mouth over an expression; width / smile / asym of the acting survive (talking while smiling)."""
    if amount <= 1e-4:
        return base
    c = base.copy()
    a = float(amount)
    for k in SPEECH_ABSOLUTE:
        tgt = mouth.get(k, 0.0)
        cur = getattr(c, k)
        if k in ("teeth_up", "teeth_low", "tongue"):
            tgt = max(tgt, cur * 0.5)
        setattr(c, k, cur + (tgt - cur) * a)
    c.width = c.width * (1.0 + (mouth.get("width", 1.0) - 1.0) * a)
    c.smile = c.smile * (1.0 - 0.35 * a * mouth.get("pucker", 0.0))
    opening = min(1.0, mouth.get("jaw", 0.0) * 3)
    c.wobble *= (1.0 - 0.7 * a * opening)
    if c.mouth_mode != "shape":
        c.mouth_glyph *= (1.0 - 0.85 * a * opening)
    return c
