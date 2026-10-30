"""
toonface.style — what makes a face look like *this* show and *this* character, separated from the acting.

Measured from the character's own plate (plate.analyze) or built from presets.  Proportions are canonical head radii,
colours are RGB 0..1, line widths are fractions of the head radius so the same style works at 200 px or 2000 px.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


def bgr2rgb01(bgr) -> tuple:
    b, g, r = [float(v) for v in bgr[:3]]
    return (r / 255.0, g / 255.0, b / 255.0)


def mix(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


@dataclass
class ToonStyle:
    name: str = "curly_toon"
    ink: tuple = (0.205, 0.102, 0.031)
    skin: tuple = (0.957, 0.859, 0.710)
    hair: tuple = (0.969, 0.922, 0.812)
    sclera: tuple = (0.996, 0.996, 0.985)
    pupil: tuple = (0.165, 0.078, 0.047)
    cavity: tuple = (0.36, 0.11, 0.11)
    tongue: tuple = (0.93, 0.49, 0.46)
    teeth: tuple = (1.0, 1.0, 1.0)
    blush: tuple = (0.95, 0.47, 0.47)
    tear: tuple = (0.60, 0.81, 0.94)
    cold: tuple = (0.62, 0.78, 0.92)
    line: float = 0.030
    line_fine: float = 0.017
    eye_u: float = 0.355
    eye_v: float = 0.02
    eye_rx: float = 0.235
    eye_ry: float = 0.255
    eye_round: float = 2.25
    eye_tilt: float = 0.0
    pupil_rx: float = 0.058
    pupil_ry: float = 0.082
    pupil_bias: tuple = (0.0, 0.10)
    brow_v: float = -0.315
    brow_len: float = 1.25
    brow_thick: float = 0.052
    brow_arch: float = 0.045
    brow_style: str = "hair"
    nose_v: float = 0.255
    nose_size: float = 0.06
    nose_dir: int = -1
    mouth_v: float = 0.505
    mouth_w: float = 0.19
    mouth_open: float = 0.33
    lashes: bool = False
    lash_weight: float = 1.0
    face_depth: float = 0.92
    cheat: float = 0.46

    def copy(self, **kw) -> "ToonStyle":
        return replace(self, **kw)

    def lw(self, R: float, fine: bool = False) -> float:
        return max(1.0, (self.line_fine if fine else self.line) * R)


MALE_DEFAULT = ToonStyle(name="curly_boy")
FEMALE_DEFAULT = ToonStyle(name="curly_girl", lashes=True, eye_rx=0.225, eye_ry=0.22, eye_round=2.1,
                           pupil_rx=0.055, pupil_ry=0.078, brow_thick=0.045, eye_tilt=0.05)
