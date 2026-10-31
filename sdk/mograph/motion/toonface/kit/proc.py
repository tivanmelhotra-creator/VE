"""
toonface.kit.proc — v33 PROCEDURAL face renderer driven by CharacterProfile (any character, every slider live).

Design space "D": origin = chin centre, x right, y DOWN, 1000 = face width (the widest row of the blank face).
All geometry is generated in D, then pushed through a 2.5D head projection (ellipsoid, yaw) and drawn by cairo
with the frame transform  T = M_track ∘ A_design→frame.

Parts and styles (option + slider system, DiceBear/VRoid-style):
  eyes   round · almond · droopy · cat · button · sleepy        + width/height/roundness/tilt/pupil/lashes/wing
  brows  soft · thick · thin · angular · bushy · round · line    + length/arch/slant/thickness/gain/colour/layer
  nose   bump · hook · button · dots · line · none
  mouth  default · wide · small · cat (rig25 mouth engine: visemes, teeth, tongue) + wavy / grin / ω extras
  eye FX modes (from the expression sheets): star · spiral · x · squeeze (>  <) · heart
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import ink
from .profile import CharacterProfile
from .rig25 import Face, chaikin, draw_mouth

D = 1000.0


# ================================================================================================ projection
@dataclass
class Proj:
    """2.5D head: a point (X, Y) on the face sheet lies on an ellipsoid; yaw rotates it about the vertical axis."""
    yaw_deg: float = 0.0
    depth: float = 0.42
    nose_out: float = 0.10

    def z(self, X):
        u = np.clip(2.0 * np.asarray(X, float) / D, -1.0, 1.0)
        return self.depth * D * np.sqrt(np.maximum(0.04, 1.0 - u * u))

    def __call__(self, P, extra_z: float = 0.0):
        P = np.asarray(P, float)
        if abs(self.yaw_deg) < 1e-3:
            return P.copy()
        a = math.radians(self.yaw_deg)
        out = P.copy()
        out[..., 0] = P[..., 0] * math.cos(a) + (self.z(P[..., 0]) + extra_z) * math.sin(a)
        return out

    def sx(self, X) -> float:
        """local horizontal foreshortening at X (for circles / widths)."""
        e = 2.0
        return float(abs(self(np.array([[X + e, 0.0]]))[0, 0] - self(np.array([[X - e, 0.0]]))[0, 0]) / (2 * e))


# ================================================================================================ helpers
def hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def mix(a, b, t):
    return tuple(float(x + (y - x) * t) for x, y in zip(a, b))


@dataclass
class Palette:
    ink: tuple
    skin: tuple
    hair: tuple
    sclera: tuple
    pupil: tuple
    brow: tuple
    blush: tuple
    sw: float                     # line weight in D units


def make_palette(prof: CharacterProfile, style) -> Palette:
    """'auto' colours come from the video's own drawing → new characters are harmonised automatically."""
    c = prof.color

    def bgr2rgb(v):
        return (float(v[2]) / 255.0, float(v[1]) / 255.0, float(v[0]) / 255.0)

    inkc = bgr2rgb(style.ink_bgr) if c.ink == "auto" else hex_rgb(c.ink)
    skin = bgr2rgb(style.skin_bgr) if c.skin == "auto" else hex_rgb(c.skin)
    hair = bgr2rgb(style.hair_bgr)
    if prof.brow.color == "auto":
        # brows in hair colour, but never invisible on the skin (bald / blond-on-pale): darken toward ink
        lum = lambda q: 0.3 * q[0] + 0.59 * q[1] + 0.11 * q[2]
        brow = hair if abs(lum(hair) - lum(skin)) > 0.10 else mix(hair, inkc, 0.55)
    elif prof.brow.color == "ink":
        brow = inkc
    else:
        brow = hex_rgb(prof.brow.color)
    blush = mix(skin, (0.96, 0.42, 0.45), 0.55) if c.blush == "auto" else hex_rgb(c.blush)
    sw = (style.line_px / max(1.0, style.width) * D) if c.line_weight == "auto" else float(c.line_weight) * D
    return Palette(ink=inkc, skin=skin, hair=hair, sclera=hex_rgb(c.sclera), pupil=hex_rgb(c.pupil), brow=brow,
                   blush=blush, sw=float(np.clip(sw, 6.0, 40.0)))


class _KitShim:
    """the bits of rig25.Kit that draw_mouth needs."""

    def __init__(self, pal: Palette):
        self.sw = pal.sw
        self.d = {}
        self.skin = pal.skin
        self.hair = pal.hair


# ================================================================================================ layout
EYE_STYLE_MODS = {
    "round":  dict(n=2.0, lower=1.0, tilt=0.0),
    "almond": dict(n=1.75, lower=0.70, tilt=0.04),
    "droopy": dict(n=2.0, lower=0.85, tilt=-0.16),
    "cat":    dict(n=1.6, lower=0.65, tilt=0.24),
    "button": dict(n=2.0, lower=1.0, tilt=0.0),
    "sleepy": dict(n=2.2, lower=0.75, tilt=-0.05),
}


def layout(prof: CharacterProfile) -> dict:
    """neutral geometry of every part in D (before expression / projection)."""
    e, b, n, m = prof.eye, prof.brow, prof.nose, prof.mouth
    out = {"eyes": {}, "brows": {}}
    for side, sg in (("L", -1.0), ("R", 1.0)):
        out["eyes"][side] = dict(c=np.array([sg * e.spacing * D, -e.y * D]), w=e.width * D, h=e.height * D,
                                 outer=sg)
        xi, xo = sg * b.gap * D, sg * (b.gap + b.length) * D
        yb = -b.y * D
        out["brows"][side] = dict(inner=np.array([xi, yb - 0.5 * b.slant * D]),
                                  outer=np.array([xo, yb + 0.5 * b.slant * D]), arch=b.arch * D,
                                  th=b.thickness * D, sg=sg)
    out["nose"] = dict(c=np.array([n.x * D, -n.y * D]), s=n.size * D)
    hw = 0.5 * m.width * D
    a = math.radians(m.tilt)
    u = np.array([math.cos(a), -math.sin(a)])                   # y down: + tilt lifts the right corner
    c = np.array([m.x * D, -m.y * D])
    out["mouth"] = dict(left=c - u * hw, right=c + u * hw, c=c, hw=hw, u=u, curve=m.curve * D)
    return out
