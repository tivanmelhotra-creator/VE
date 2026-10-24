"""
mograph.toon.character — a TOON character = MakeHuman body (drives real mocap) + big round ball head + expressive
2D face + flat white fills (the reference "ball-head" look: sweater + trousers + sneakers, all white, dark ink).

    me = ToonCharacter("guy")                     # bald ball head, sweater, trousers
    sis = ToonCharacter("girl", hair="ponytail")  # optional hair shells (dark fill)
    A = Actor("me", me.ch, [...mocap segments...])
"""
from __future__ import annotations

from ..human.character import Character
from ..human.lineart import Style
from .face import ToonFace
from .head import ToonHead

WHITE = (0.985, 0.98, 0.965)
DARK = (0.18, 0.17, 0.17)
GREY = (0.62, 0.62, 0.64)

TOON_PRESETS = {
    "guy":   dict(body=dict(gender=1, age=25, muscle=0.3, weight=0.38, height=0.55),
                  outfit=dict(top="sweater", bottom="trousers", shoes="sneakers"), head=1.55, kind="adult",
                  hair="bald", female=False),
    "gal":   dict(body=dict(gender=0, age=24, muscle=0.3, weight=0.36, height=0.5),
                  outfit=dict(top="sweater", bottom="trousers", shoes="sneakers"), head=1.55, kind="adult",
                  hair="long", female=True),
    "kid":   dict(body=dict(gender=1, age=9, muscle=0.3, weight=0.4),
                  outfit=dict(top="sweater", bottom="trousers", shoes="sneakers"), head=1.6, kind="child",
                  hair="bald", female=False),
    "girl":  dict(body=dict(gender=0, age=7, muscle=0.3, weight=0.4),
                  outfit=dict(top="sweater", bottom="trousers", shoes="sneakers"), head=1.6, kind="child",
                  hair="ponytail", female=True),
    "mom":   dict(body=dict(gender=0, age=40, muscle=0.3, weight=0.5, height=0.5),
                  outfit=dict(top="sweater", bottom="skirt", length=0.85, shoes="shoes"), head=1.5, kind="adult",
                  hair="bun", female=True),
}


class ToonCharacter:
    def __init__(self, preset_name="guy", head=None, hair=None, fills=None, female=None, kind=None, seed=0,
                 body=None, outfit=None, hair_len=1.0, line_style: Style | None = None, hand_scale=1.3):
        p = dict(TOON_PRESETS[preset_name])
        b = dict(p["body"], **(body or {}))
        o = dict(p["outfit"], **(outfit or {}))
        o["hair"] = "bald"
        self.ch = Character(body=b, outfit=o, face="none", style=line_style or Style(), folds=True)
        self.kind = kind or p["kind"]
        self.female = p["female"] if female is None else female
        hs = hair or p["hair"]
        self.head = ToonHead(self.ch, scale=head or p["head"], hair=hs, seed=seed, hair_len=hair_len)
        self.face = ToonFace(self.head, kind=self.kind, female=self.female, seed=seed + 3)
        self.fills = {"skin": WHITE, "cloth": WHITE, "hair": DARK, "shoe": WHITE}
        self.fills.update(fills or {})
        self.name = preset_name
        self.hand_scale = hand_scale

    def fill_for(self, part_name):
        if part_name.startswith("hair"):
            return self.fills["hair"]
        if part_name in ("body", "toon_head"):
            return self.fills["skin"]
        if any(k in part_name for k in ("sneakers", "shoes", "boots", "heels")):
            return self.fills["shoe"]
        for k, v in self.fills.items():
            if part_name.startswith(k):
                return v
        return self.fills["cloth"]
