"""
toonface.kit.profile — v33 CharacterProfile: a "character creator" for the face rig (game-style sliders).

One JSON per character (sdk/characters/<name>.json) → the same engine renders any blank-face character:
    prof = CharacterProfile.preset("boy");  prof.eye.roundness = 2.6;  prof.save("characters/boy.json")
    prof = CharacterProfile.load("characters/boy.json")

Design (patterns borrowed from open-source avatar / VTuber systems):
  * option-based parts like DiceBear / Avataaars (eye style, brow style, nose style, mouth style),
  * continuous sliders like VRoid / VRM character makers (size, spacing, height, roundness, tilt),
  * colours default to "auto" = measured from the video (skin, line-art ink colour and line weight, hair for brows),
    so a new character with a different skin tone or line style is harmonised automatically,
  * personality block (expressiveness gain, blink rate, resting mood, brow energy) used by the performer.
Units: geometry in FACE WIDTHS (widest row of the blank face = 1.0), y measured UP from the chin.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass

EYE_STYLES = ("round", "almond", "droopy", "cat", "button", "sleepy")
BROW_STYLES = ("soft", "thick", "thin", "angular", "bushy", "round", "line")
NOSE_STYLES = ("bump", "hook", "button", "dots", "line", "none")
MOUTH_STYLES = ("default", "wide", "small", "cat")


@dataclass
class EyeSpec:
    style: str = "round"
    y: float = 0.56            # height above chin
    spacing: float = 0.26      # centre → eye centre
    width: float = 0.36
    height: float = 0.34
    roundness: float = 2.0     # superellipse exponent (2 = ellipse, 3 = boxy, 1.6 = pointy)
    tilt: float = 0.0          # + outer corner up (cat), − down (droopy)
    lower_flat: float = 0.85   # lower-lid curvature vs upper (1 = symmetric)
    pupil: float = 0.125       # pupil radius / eye width
    pupil_aspect: float = 1.33
    lid_weight: float = 1.0    # upper-lid line weight
    lashes: bool = False
    lower_lashes: bool = False
    wing: bool = False
    rest_open: float = 0.80


@dataclass
class BrowSpec:
    style: str = "soft"
    y: float = 0.80            # height of the brow middle above chin
    gap: float = 0.13          # half distance between inner ends
    length: float = 0.24
    arch: float = 0.06         # camber
    slant: float = 0.06        # inner end higher (+) than outer
    thickness: float = 0.040
    gain: float = 1.35         # expressiveness multiplier (brows were "lazy" in v32)
    layer: str = "over"        # "over" = drawn over the hair · "under" = hidden under the hair (clipped to face)
    color: str = "auto"        # "auto" (hair colour from the video) · "ink" · "#rrggbb"
    outline: bool = True


@dataclass
class NoseSpec:
    style: str = "bump"
    y: float = 0.40
    size: float = 0.11
    x: float = 0.0


@dataclass
class MouthSpec:
    style: str = "default"
    y: float = 0.25
    width: float = 0.32
    x: float = 0.01
    curve: float = 0.035       # smile sag at rest (+ = smiling line)
    tilt: float = 0.0          # degrees, + right corner up (built-in smirk of the design)


@dataclass
class ColorSpec:
    skin: str = "auto"
    ink: str = "auto"          # line colour = measured from the face outline
    line_weight: str = "auto"  # or a float in face widths (0.015 ≈ the kits)
    sclera: str = "#fbf4e2"
    pupil: str = "#251510"
    blush: str = "auto"        # mix of skin and rose


@dataclass
class Personality:
    expressiveness: float = 1.0    # global AU gain
    brow_energy: float = 1.0       # extra brow motion on speech accents
    blink_rate: float = 1.0        # × natural rate
    rest_mood: str = "neutral"     # emotion preset blended at 25 % under everything
    squash_stretch: float = 0.6    # jaw drop warps the chin/cheeks of the drawing (0 = rigid)


@dataclass
class Head3D:
    depth: float = 0.42        # ellipsoid depth semi-axis (face widths) → how much features slide on a turn
    chin_depth: float = 0.21   # used to estimate yaw from the chin offset of the drawing
    nose_out: float = 0.10     # nose protrusion (moves more than the eyes on a turn = 2.5D cue)
    max_yaw: float = 38.0
    max_pitch: float = 18.0


@dataclass
class CharacterProfile:
    name: str = "character"
    rig: str = "procedural"    # "procedural" (all sliders live) | "kit:boy" | "kit:girl" (hand-authored shapes)
    eye: EyeSpec = field(default_factory=EyeSpec)
    brow: BrowSpec = field(default_factory=BrowSpec)
    nose: NoseSpec = field(default_factory=NoseSpec)
    mouth: MouthSpec = field(default_factory=MouthSpec)
    color: ColorSpec = field(default_factory=ColorSpec)
    personality: Personality = field(default_factory=Personality)
    head: Head3D = field(default_factory=Head3D)

    # ------------------------------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> str:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "CharacterProfile":
        p = cls()
        _update(p, d)
        p.validate()
        return p

    @classmethod
    def load(cls, path: str) -> "CharacterProfile":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def with_(self, **overrides) -> "CharacterProfile":
        """copy with dotted overrides: prof.with_(**{"eye.width": 0.4, "brow.style": "bushy"})."""
        d = self.to_dict()
        for k, v in overrides.items():
            cur = d
            parts = k.replace("__", ".").split(".")
            for q in parts[:-1]:
                cur = cur[q]
            cur[parts[-1]] = v
        return CharacterProfile.from_dict(d)

    def validate(self):
        """hard limits so a typo in a JSON can never produce a broken face mid-production."""
        e, b, n, m = self.eye, self.brow, self.nose, self.mouth
        assert e.style in EYE_STYLES, f"eye.style must be one of {EYE_STYLES}"
        assert b.style in BROW_STYLES, f"brow.style must be one of {BROW_STYLES}"
        assert n.style in NOSE_STYLES, f"nose.style must be one of {NOSE_STYLES}"
        assert m.style in MOUTH_STYLES, f"mouth.style must be one of {MOUTH_STYLES}"
        assert b.layer in ("over", "under"), "brow.layer must be over | under"
        assert self.rig == "procedural" or self.rig in ("kit:boy", "kit:girl"), "rig: procedural | kit:boy | kit:girl"
        e.width = _clip(e.width, 0.12, 0.46)
        e.height = _clip(e.height, 0.06, 0.46)
        e.spacing = _clip(e.spacing, e.width * 0.5 + 0.01, 0.36)       # eyes never overlap
        e.y = _clip(e.y, 0.40, 0.70)
        e.roundness = _clip(e.roundness, 1.4, 4.0)
        e.pupil = _clip(e.pupil, 0.05, 0.30)
        b.y = _clip(b.y, e.y + e.height * 0.5 + 0.03, 0.98)               # brows always above the eyes
        b.length = _clip(b.length, 0.08, 0.34)
        b.gap = _clip(b.gap, 0.03, 0.25)
        b.gain = _clip(b.gain, 0.5, 2.2)
        n.y = _clip(n.y, m.y + 0.06, e.y - 0.06)
        m.width = _clip(m.width, 0.10, 0.60)
        m.y = _clip(m.y, 0.10, n.y - 0.06)
        return self

    # ------------------------------------------------------------------------------------------ presets
    @classmethod
    def preset(cls, name: str) -> "CharacterProfile":
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        return cls.from_dict({"name": name, **PRESETS[name]})

    @classmethod
    def resolve(cls, spec: str) -> "CharacterProfile":
        """preset name or path to a JSON file."""
        return cls.load(spec) if spec.endswith(".json") else cls.preset(spec)


def _clip(v, lo, hi):
    return float(min(max(v, lo), hi))


def _update(obj, d: dict):
    for f in fields(obj):
        if f.name not in d:
            continue
        cur = getattr(obj, f.name)
        if is_dataclass(cur):
            _update(cur, d[f.name])
        else:
            setattr(obj, f.name, d[f.name])


# measured on the hand-authored v31 kits (front view) + new archetypes
PRESETS: dict[str, dict] = {
    "boy": dict(rig="kit:boy", eye=dict(style="round", y=0.565, spacing=0.262, width=0.368, height=0.346, pupil=0.12,
                         lid_weight=1.0),
                brow=dict(style="soft", y=0.81, gap=0.143, length=0.24, arch=0.05, slant=0.06, thickness=0.040),
                nose=dict(style="bump", y=0.40, size=0.11),
                mouth=dict(y=0.255, width=0.33, x=0.018, curve=0.035, tilt=7.0),
                personality=dict(rest_mood="friendly")),
    "girl": dict(rig="kit:girl", eye=dict(style="almond", y=0.55, spacing=0.237, width=0.345, height=0.315, pupil=0.125,
                          lid_weight=2.4, lashes=True, lower_lashes=True, wing=True, tilt=0.06),
                 brow=dict(style="soft", y=0.815, gap=0.145, length=0.24, arch=0.06, slant=0.065, thickness=0.036),
                 nose=dict(style="hook", y=0.41, size=0.10),
                 mouth=dict(y=0.26, width=0.285, x=0.026, curve=0.04, tilt=10.0),
                 personality=dict(rest_mood="friendly")),
    "cute_round": dict(eye=dict(style="round", width=0.40, height=0.42, pupil=0.17, spacing=0.27, y=0.53),
                       brow=dict(style="round", length=0.12, y=0.82), nose=dict(style="button", size=0.07),
                       mouth=dict(style="small", width=0.20, y=0.24), personality=dict(expressiveness=1.25)),
    "sharp_villain": dict(eye=dict(style="cat", width=0.32, height=0.18, tilt=0.22, pupil=0.09, y=0.58),
                          brow=dict(style="angular", thickness=0.05, slant=-0.05, y=0.75, gap=0.10),
                          nose=dict(style="hook", size=0.14), mouth=dict(style="wide", width=0.40, curve=-0.01),
                          personality=dict(rest_mood="smug", brow_energy=1.3)),
    "sleepy_chill": dict(eye=dict(style="sleepy", height=0.22, rest_open=0.55, tilt=-0.08),
                         brow=dict(style="thin", slant=-0.03), nose=dict(style="line"),
                         personality=dict(expressiveness=0.8, blink_rate=0.6)),
    "bold_hero": dict(eye=dict(style="almond", width=0.33, height=0.26, pupil=0.13),
                      brow=dict(style="thick", thickness=0.06, y=0.76), nose=dict(style="bump", size=0.12),
                      mouth=dict(style="wide", width=0.38), personality=dict(brow_energy=1.2)),
    "fantasy_big_eyes": dict(eye=dict(style="round", width=0.44, height=0.46, pupil=0.21, spacing=0.25, y=0.52,
                                      lashes=True),
                             brow=dict(style="line", y=0.83), nose=dict(style="dots", size=0.06),
                             mouth=dict(style="cat", width=0.18), personality=dict(squash_stretch=1.0)),
}
