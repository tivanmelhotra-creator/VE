"""
toonface.kit.expressive — v32 FACS-style expression layer for the rig25 kit.

Why the v31 faces looked "dry":
  * one scalar per feature group (both brows share brow_up, both eyes share lid_open) → no asymmetry, no smirk,
    no single raised brow;
  * every channel jumps or lerps linearly → no anticipation / overshoot / settle;
  * channels are independent → a smile does not squeeze the eyes, a gasp does not lift the brows, looking down
    does not drop the lids.  Real faces (and good animators) always couple them.

This module adds, WITHOUT changing the rig25 renderer:
  1. AU — Action-Unit controls (FACS names), per side where anatomy allows:
       AU1 inner brow raise · AU2 outer brow raise · AU4 brow lower · AU5 upper-lid raise · AU6 cheek raise
       · AU7 lid tighten · AU9 nose wrinkle · AU10 upper-lip raise · AU12 lip-corner pull · AU15 corner depress
       · AU17 chin raise · AU18 pucker · AU20 lip stretch · AU23/24 lip tighten/press · AU26 jaw drop
       · AU43 eyes closing · AU45 blink  + gaze_x / gaze_y / pupil.
  2. EMOTIONS — 19 presets authored as AU sets, mixable with weights (blend()).
  3. COUPLING — physiological links applied to the targets: AU12→AU6 (Duchenne squint), AU4→AU7, AU26→brows up,
     gaze down→upper lid follows, gaze up→brows lift, AU9→AU10.
  4. DYNAMICS — every channel has its own 2nd-order spring: brows overshoot a little and settle, lids are fast,
     gaze moves in saccades, mouth corners lag slightly (overlapping action).  Anticipation dip before brow pops.
  5. FaceFrame — a rig25.Face per SIDE + a shared mouth + secondary lines (nose wrinkle, laugh lines).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .rig25 import Face

SIDED = ("AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU12", "AU15", "AU43", "AU45")
SHARED = ("AU9", "AU10", "AU17", "AU18", "AU20", "AU23", "AU24", "AU26", "gaze_x", "gaze_y", "pupil", "talk")


def au(**kw) -> dict:
    """AU12=0.6 sets both sides; AU12_L=0.6 one side only (L = screen-left eye/brow of the kit)."""
    out = {}
    for k, v in kw.items():
        if k in SIDED:
            out[k + "_L"] = v
            out[k + "_R"] = v
        else:
            out[k] = v
    return out


EMOTIONS: dict[str, dict] = {
    "neutral":     au(),
    "friendly":    au(AU12=0.45, AU6=0.25, AU1=0.15, AU2=0.15),
    "happy":       au(AU12=0.85, AU6=0.65, AU1=0.25, AU2=0.30, AU26=0.15),
    "laugh":       au(AU12=1.0, AU6=0.95, AU1=0.35, AU2=0.40, AU26=0.65, AU7=0.2),
    "explaining":  au(AU1=0.35, AU2=0.40, AU12=0.25, AU5=0.15),
    "curious":     au(AU1_L=0.55, AU2_L=0.75, AU1_R=0.15, AU2_R=0.10, AU4_R=0.25, AU5=0.15, AU12=0.12, gaze_y=-0.15),
    "puzzled":     au(AU1_L=0.70, AU2_L=0.85, AU4_R=0.55, AU7_R=0.30, AU12_R=0.35, AU15_L=0.25, AU20=0.15,
                      gaze_x=0.25, gaze_y=-0.35),
    "unsure":      au(AU1=0.75, AU2=0.45, AU4=0.20, AU15=0.45, AU17=0.45, AU20=0.25, AU12=0.10),
    "surprised":   au(AU1=0.95, AU2=0.95, AU5=0.85, AU26=0.55, pupil=-0.25),
    "shocked":     au(AU1=1.0, AU2=1.0, AU5=1.0, AU26=0.85, AU20=0.3, pupil=-0.35),
    "sad":         au(AU1=0.85, AU4=0.45, AU15=0.70, AU17=0.35, AU43=0.25, gaze_y=0.25),
    "worried":     au(AU1=0.90, AU4=0.55, AU20=0.35, AU5=0.20, AU15=0.25),
    "angry":       au(AU4=0.95, AU5=0.35, AU7=0.55, AU23=0.65, AU24=0.35, AU9=0.20, AU15=0.15),
    "determined":  au(AU4=0.55, AU7=0.45, AU24=0.55, AU17=0.35, AU12=0.15),
    "disgust":     au(AU9=0.85, AU10=0.65, AU4=0.45, AU6=0.35, AU15=0.35, AU12_L=0.2),
    "smug":        au(AU12_L=0.75, AU12_R=0.15, AU6_L=0.30, AU43=0.40, AU2_R=0.30, gaze_x=0.2),
    "skeptical":   au(AU2_L=0.85, AU1_L=0.40, AU4_R=0.55, AU7_R=0.40, AU12_L=0.20, AU15_R=0.20, AU24=0.30),
    "exasperated": au(AU1=0.65, AU4=0.40, AU43=0.45, AU15=0.40, AU20=0.30, AU26=0.15, gaze_y=-0.55),
    "facepalm":    au(AU43=0.95, AU4=0.65, AU1=0.55, AU7=0.40, AU20=0.50, AU15=0.40, AU26=0.10, AU9=0.25),
}


# expressions from the user's reference sheets (exaggerated cartoon set)
EMOTIONS.update({
    "roaring":     au(AU4=1.0, AU7=0.9, AU9=0.6, AU10=0.8, AU26=1.0, AU20=0.5, AU43=0.9),
    "star_eyed":   au(AU12=0.95, AU6=0.4, AU1=0.5, AU2=0.6, AU26=0.55),
    "terrified":   au(AU1=1.0, AU2=0.6, AU4=0.5, AU5=1.0, AU20=1.0, AU26=0.35, AU10=0.4, pupil=-0.45),
    "pleading":    au(AU1=1.0, AU4=0.35, AU12=0.30, AU5=0.25, pupil=0.30),
    "shy":         au(AU1=0.75, AU4=0.30, AU15=0.30, AU20=0.30, AU43=0.20, gaze_x=0.25, gaze_y=0.25),
    "dizzy":       au(AU1=0.6, AU2=0.3, AU15=0.3, AU20=0.4, AU26=0.15),
    "mischievous": au(AU4=0.80, AU2=0.40, AU12=1.0, AU6=0.6, AU10=0.6, AU20=0.4, AU43=0.25),
    "yawning":     au(AU1=0.55, AU43=1.0, AU26=1.0, AU18=0.5, AU4=0.2),
    "thinking":    au(AU2_L=0.85, AU4_R=0.45, AU43=0.25, AU12_L=0.45, gaze_x=0.45, gaze_y=-0.45),
    "weeping":     au(AU1=1.0, AU4=0.80, AU43=0.9, AU26=0.85, AU15=0.8, AU20=0.6),
    "eating":      au(AU6=0.8, AU12=0.6, AU43=0.9, AU24=0.5, AU17=0.5, AU1=0.3),
    "disgusted":   au(AU4=0.9, AU9=1.0, AU10=0.8, AU7=0.7, AU15=0.5, AU43=0.5, AU20=0.3),
    "eureka":      au(AU1=0.9, AU2=1.0, AU5=0.9, AU26=0.45, AU12=0.55),
    "sulking":     au(AU4=0.85, AU15=0.6, AU17=0.7, AU24=0.4, AU43=0.35, gaze_x=0.4),
    "whistling":   au(AU1=0.4, AU2=0.6, AU43=0.6, AU18=1.0, AU26=0.15, gaze_y=-0.2),
    "freezing":    au(AU1=0.9, AU4=0.5, AU5=0.8, AU20=1.0, AU26=0.3, AU10=0.4, pupil=-0.4),
    "joy":         au(AU12=1.0, AU6=1.0, AU1=0.5, AU2=0.6, AU26=0.85, AU43=1.0),
    "sneaky":      au(AU4=0.6, AU2_L=0.5, AU12=0.9, AU12_L=1.0, AU10=0.5, AU43=0.45, AU20=0.3, gaze_x=0.4),
    "excited":     au(AU12=1.0, AU6=0.7, AU1=0.6, AU2=0.7, AU20=0.6, AU10=0.6, AU26=0.25),
    "scared":      au(AU1=1.0, AU4=0.45, AU5=0.85, AU20=0.8, AU26=0.45, AU15=0.4, pupil=-0.3),
    "crying":      au(AU1=1.0, AU4=0.7, AU43=0.95, AU15=0.9, AU17=0.6, AU20=0.5, AU26=0.2),
})

# graphic extras per expression (not muscles): eye_fx star|spiral|x|squeeze|heart|x_spiral, mouth_fx wavy|grit|grin|
# o|cat, blush 0..1, sweat 0..1, tears 0..1 (streams), lines: shock|anger|joy|cold, tongue_out
EXPR_FX: dict[str, dict] = {
    "roaring": dict(eye_fx="squeeze", lines="anger"), "star_eyed": dict(eye_fx="star"),
    "terrified": dict(mouth_fx="grit", sweat=0.8), "pleading": dict(sparkle=1.0),
    "shy": dict(blush=1.0, sweat=0.5, mouth_fx="wavy"), "shocked": dict(mouth_fx="grit", lines="shock"),
    "dizzy": dict(eye_fx="spiral", mouth_fx="wavy"), "mischievous": dict(mouth_fx="grin"),
    "weeping": dict(eye_fx="squeeze", tears=1.0), "crying": dict(tears=1.0, blush=0.6, mouth_fx="wavy"),
    "disgusted": dict(mouth_fx="wavy"), "eureka": dict(lines="idea"), "whistling": dict(mouth_fx="o"),
    "freezing": dict(mouth_fx="grit", sweat=0.4, cold=1.0, lines="cold"), "joy": dict(lines="joy"),
    "sneaky": dict(mouth_fx="grin", blush=0.3), "excited": dict(eye_fx="star", blush=0.8, mouth_fx="grin"),
    "scared": dict(sweat=0.9), "angry": dict(lines="anger", blush=0.5, mouth_fx="grit"),
    "embarrassed": dict(blush=1.0, sweat=0.6),
}
EMOTIONS["embarrassed"] = au(AU1=0.7, AU12=0.35, AU4=0.2, AU43=0.2, gaze_x=-0.3, gaze_y=0.2)


def blend(*pairs) -> dict:
    """blend(("happy", .7), ("surprised", .3), ({"AU12_L": .2}, 1.0)) → AU dict (weighted sum)."""
    out: dict = {}
    for p, w in pairs:
        d = EMOTIONS[p] if isinstance(p, str) else p
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + w * v
    return out


# ------------------------------------------------------------------------------------------------- dynamics
@dataclass
class SpringSpec:
    f: float = 6.0          # natural frequency (Hz)
    zeta: float = 0.75      # damping (< 1 → overshoot)


DEFAULT_SPRINGS = {
    "AU1": SpringSpec(5.0, 0.55), "AU2": SpringSpec(5.0, 0.55), "AU4": SpringSpec(4.5, 0.65),
    "AU5": SpringSpec(9.0, 0.85), "AU7": SpringSpec(7.0, 0.85), "AU43": SpringSpec(9.0, 0.9),
    "AU45": SpringSpec(30.0, 1.0), "AU6": SpringSpec(4.0, 0.8), "AU12": SpringSpec(5.0, 0.7),
    "AU15": SpringSpec(4.5, 0.8), "AU9": SpringSpec(5.0, 0.8), "AU10": SpringSpec(6.0, 0.8),
    "AU17": SpringSpec(5.0, 0.8), "AU18": SpringSpec(6.0, 0.8), "AU20": SpringSpec(6.0, 0.8),
    "AU23": SpringSpec(6.0, 0.8), "AU24": SpringSpec(6.0, 0.8), "AU26": SpringSpec(10.0, 0.9),
    "gaze_x": SpringSpec(14.0, 1.0), "gaze_y": SpringSpec(14.0, 1.0), "pupil": SpringSpec(3.0, 1.0),
    "talk": SpringSpec(20.0, 1.0),
}


class Spring:
    """2nd-order follower, semi-implicit Euler with sub-steps (stable at any fps)."""

    def __init__(self, spec: SpringSpec, x0: float = 0.0):
        self.w = 2 * math.pi * spec.f
        self.z = spec.zeta
        self.x, self.v = x0, 0.0

    def step(self, target: float, dt: float) -> float:
        n = max(1, int(math.ceil(dt * self.w / 0.25)))
        h = dt / n
        for _ in range(n):
            a = self.w * self.w * (target - self.x) - 2 * self.z * self.w * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x


# ------------------------------------------------------------------------------------------------- coupling
def couple(a: dict) -> dict:
    g = dict(a)
    gy = g.get("gaze_y", 0.0)
    jaw = g.get("AU26", 0.0)
    for s in ("L", "R"):
        smile = max(0.0, g.get(f"AU12_{s}", 0.0))
        g[f"AU6_{s}"] = max(g.get(f"AU6_{s}", 0.0), 0.55 * smile ** 1.3)
        g[f"AU7_{s}"] = g.get(f"AU7_{s}", 0.0) + 0.30 * g.get(f"AU4_{s}", 0.0)
        g[f"AU1_{s}"] = g.get(f"AU1_{s}", 0.0) + 0.12 * jaw
        g[f"AU2_{s}"] = g.get(f"AU2_{s}", 0.0) + 0.10 * jaw + 0.25 * max(0.0, -gy)
        g[f"AU43_{s}"] = g.get(f"AU43_{s}", 0.0) + 0.45 * max(0.0, gy)
        g[f"AU5_{s}"] = g.get(f"AU5_{s}", 0.0) + 0.20 * max(0.0, -gy)
    g["AU10"] = g.get("AU10", 0.0) + 0.45 * g.get("AU9", 0.0)
    return {k: float(np.clip(v, -1.0, 1.3)) for k, v in g.items()}


# ------------------------------------------------------------------------------------------------- mapping
@dataclass
class FaceFrame:
    left: Face
    right: Face
    mouth: dict
    nose_dy: float = 0.0            # in eye heights (negative = up)
    wrinkle: float = 0.0            # nose-bridge wrinkle 0..1
    cheek_line: tuple = (0.0, 0.0)  # laugh-line ticks per side 0..1
    au: dict = field(default_factory=dict)
    fx: dict = field(default_factory=dict)   # eye_fx, mouth_fx, blush, sweat, tears, lines, shake (graphic extras)


def to_faces(a: dict, base_lid: float = 0.80, mouth_extra: dict | None = None) -> FaceFrame:
    def G(k, d=0.0):
        return float(a.get(k, d))

    look = (float(np.clip(G("gaze_x"), -0.7, 0.7)), float(np.clip(G("gaze_y"), -0.7, 0.7)))
    pupil = float(np.clip(1.0 + G("pupil"), 0.55, 1.3))
    sides = []
    for s in ("L", "R"):
        au1, au2, au4 = G(f"AU1_{s}"), G(f"AU2_{s}"), G(f"AU4_{s}")
        au5, au6, au7 = G(f"AU5_{s}"), G(f"AU6_{s}"), G(f"AU7_{s}")
        au43, au45 = G(f"AU43_{s}"), G(f"AU45_{s}")
        brow_up = 0.55 * au2 + 0.45 * au1 - 0.35 * au4 - 0.12 * au45
        brow_worry = max(0.0, au1 - 0.6 * au2) * 1.1 + 0.25 * au1 * au4
        brow_anger = max(0.0, au4 - 0.5 * au1) * 1.05
        lid = (base_lid + 0.38 * au5 - 0.18 * au7 - 0.10 * au6 - 0.72 * au43) * (1.0 - np.clip(au45, 0.0, 1.0))
        lower = 0.55 * au6 + 0.25 * au7
        lid_tilt = 0.40 * brow_anger - 0.28 * brow_worry
        sides.append(Face(lid_open=float(np.clip(lid, 0.0, 1.18)), lid_tilt=float(np.clip(lid_tilt, -0.35, 0.4)),
                          lower=float(np.clip(lower, 0.0, 0.55)), look=look, pupil=pupil,
                          brow_up=float(np.clip(brow_up, -0.7, 1.0)),
                          brow_anger=float(np.clip(brow_anger, 0.0, 1.1)),
                          brow_worry=float(np.clip(brow_worry, 0.0, 1.0)),
                          furrow=float(np.clip(0.9 * max(G("AU4_L"), G("AU4_R")) - 0.2, 0.0, 1.0)), bags=0.0))
    L, R = sides
    sm_l, sm_r = G("AU12_L") - G("AU15_L"), G("AU12_R") - G("AU15_R")
    m = dict(open=float(np.clip(0.95 * G("AU26") + 0.1 * G("AU10"), 0.0, 0.9)),
             corner=float(np.clip(0.55 * (sm_l + sm_r) - 0.15 * G("AU17"), -0.6, 0.6)),
             skew=float(np.clip(0.55 * (sm_l - sm_r), -0.5, 0.5)),
             width=float(np.clip(1.0 + 0.06 * (G("AU12_L") + G("AU12_R")) + 0.14 * G("AU20")
                                 - 0.10 * G("AU23") - 0.25 * G("AU18"), 0.8, 1.2)),
             press=float(np.clip(G("AU24") + 0.6 * G("AU23"), 0.0, 1.0)),
             upper=float(np.clip(0.6 * G("AU10"), 0.0, 0.6)),
             round=float(np.clip(G("AU18"), 0.0, 1.0)),
             teeth_up=float(np.clip(0.55 * G("AU26") + 0.6 * G("AU10") + 0.12 * (sm_l + sm_r), 0.0, 0.85)),
             tongue=float(np.clip(0.6 * G("AU26"), 0.0, 0.6)))
    if mouth_extra:       # speech visemes win the jaw/lips; emotion keeps the corners (talking smile / frown)
        talking = float(mouth_extra.get("open", 0.0)) > 0.02 or float(mouth_extra.get("press", 0.0)) > 0.3
        if talking:
            emo_corner, emo_skew = m["corner"], m["skew"]
            for key, v in mouth_extra.items():
                m[key] = v
            m["corner"] = float(np.clip(mouth_extra.get("corner", 0.0) + 0.7 * emo_corner, -0.6, 0.6))
            m["skew"] = float(np.clip(mouth_extra.get("skew", 0.0) + 0.7 * emo_skew, -0.5, 0.5))
    nose_dy = -0.10 * G("AU9") - 0.025 * (G("AU12_L") + G("AU12_R")) + 0.06 * G("AU26")
    cheek = (float(np.clip((G("AU12_L") - 0.55) * 2.2, 0, 1)), float(np.clip((G("AU12_R") - 0.55) * 2.2, 0, 1)))
    return FaceFrame(left=L, right=R, mouth=m, nose_dy=nose_dy, wrinkle=float(np.clip(G("AU9") * 1.2, 0, 1)),
                     cheek_line=cheek, au=dict(a))


# ------------------------------------------------------------------------------------------------- performer
@dataclass
class Beat:
    t: float                         # seconds
    emotion: dict | str              # preset name or AU dict
    weight: float = 1.0
    fx: dict | None = None           # override / add graphic extras for this beat


class ExpressivePerformer:
    """timeline of beats → per-frame FaceFrame (coupling, springs, auto-blinks, micro-saccades, anticipation).

        perf = ExpressivePerformer(fps=24, beats=[Beat(0, "friendly"), Beat(1.6, "puzzled")],
                                   gaze=[(0.0, (0.1, 0)), (1.7, (0.3, -0.4))], visemes=track, seed=3)
        ff = perf.frame(i)
    """

    def __init__(self, fps: float, beats: list, gaze: list | None = None, blinks: list | None = None,
                 visemes: list | None = None, springs: dict | None = None, seed: int = 0,
                 saccades: bool = True, base_lid: float = 0.80):
        self.fps = fps
        self.beats = sorted(beats, key=lambda b: b.t)
        self.gaze = sorted(gaze or [], key=lambda g: g[0])
        self.visemes = visemes
        self.spr = {**DEFAULT_SPRINGS, **(springs or {})}
        self.base_lid = base_lid
        rng = np.random.default_rng(seed)
        end = (self.beats[-1].t + 6.0) if self.beats else 10.0
        if blinks is None:                 # natural blinks every 2.2–4.5 s + one on big beat changes
            blinks, t = [], 0.4 + rng.uniform(0, 0.8)
            while t < end:
                blinks.append(t)
                t += rng.uniform(2.2, 4.5)
            for b in self.beats[1:]:
                if not any(abs(b.t - x) < 0.5 for x in blinks):
                    blinks.append(b.t + 0.05)
        self.blinks = sorted(blinks)
        self.sacc = []
        if saccades:
            t = 0.3
            while t < end:
                self.sacc.append((t, rng.normal(0, 0.05), rng.normal(0, 0.03)))
                t += rng.uniform(0.6, 1.4)
        self._springs: dict = {}
        self._last_i = -1

    @staticmethod
    def _em(b):
        d = EMOTIONS[b.emotion] if isinstance(b.emotion, str) else b.emotion
        return {k: v * b.weight for k, v in d.items()}

    def _target(self, t: float) -> dict:
        bs = self.beats
        cur = self._em(bs[0]) if bs else {}
        for b in bs:
            if t >= b.t:
                cur = self._em(b)
        for b in bs[1:]:                   # anticipation: brows dip ~120 ms before a brow pop
            x = b.t - t
            if 0.0 < x < 0.12:
                nxt = self._em(b)
                up = 0.5 * (nxt.get("AU2_L", 0) + nxt.get("AU2_R", 0) - cur.get("AU2_L", 0) - cur.get("AU2_R", 0))
                if up > 0.3:
                    for s in ("L", "R"):
                        cur[f"AU4_{s}"] = cur.get(f"AU4_{s}", 0.0) + 0.25 * up * math.sin(math.pi * (0.12 - x) / 0.12)
        gx, gy = cur.get("gaze_x", 0.0), cur.get("gaze_y", 0.0)
        for tg, (x, y) in self.gaze:
            if t >= tg:
                gx, gy = x, y
        for ts, dx, dy in self.sacc:
            if t >= ts:
                sx, sy = dx, dy
        if self.sacc and t >= self.sacc[0][0]:
            gx, gy = gx + sx, gy + sy
        cur["gaze_x"], cur["gaze_y"] = gx, gy
        bl = 0.0
        for b in self.blinks:              # close 2 frames, hold 1, open 3
            x = t - b
            if 0 <= x < 0.25:
                bl = max(bl, float(np.interp(x, [0, 0.06, 0.10, 0.25], [0.0, 1.0, 1.0, 0.0])))
        cur["AU45_L"] = cur["AU45_R"] = bl
        return couple(cur)

    def _advance(self, i: int, dt: float) -> dict:
        tgt = self._target(i * dt)
        out = {}
        for k in set(tgt) | set(self._springs):
            base = k[:-2] if k.endswith(("_L", "_R")) else k
            if k not in self._springs:
                self._springs[k] = Spring(self.spr.get(base, SpringSpec()), tgt.get(k, 0.0))
            if base == "AU45":             # blinks bypass the spring (crisp)
                self._springs[k].x = tgt.get(k, 0.0)
                out[k] = self._springs[k].x
            else:
                out[k] = self._springs[k].step(tgt.get(k, 0.0), dt)
        return out

    def frame(self, i: int) -> FaceFrame:
        dt = 1.0 / self.fps
        if i != self._last_i + 1:          # random access → deterministic re-simulation
            self._springs = {}
            for j in range(i):
                self._advance(j, dt)
        a = self._advance(i, dt)
        self._last_i = i
        mouth = None
        if self.visemes:
            from .visemes import mouth_at
            mouth = dict(mouth_at(self.visemes, i * dt))
        ff = to_faces(a, base_lid=self.base_lid, mouth_extra=mouth)
        ff.fx = self.fx_at(i * dt)
        return ff

    def fx_at(self, t: float) -> dict:
        """graphic extras of the active beat; numeric ones fade in over 0.15 s, symbolic ones switch on the beat."""
        cur, t0 = None, 0.0
        for b in self.beats:
            if t >= b.t:
                cur, t0 = b, b.t
        if cur is None:
            return {}
        src = dict(EXPR_FX.get(cur.emotion, {})) if isinstance(cur.emotion, str) else {}
        src.update(getattr(cur, "fx", None) or {})
        k = min(1.0, (t - t0) / 0.15)
        return {key: (v * k if isinstance(v, (int, float)) else v) for key, v in src.items()}
