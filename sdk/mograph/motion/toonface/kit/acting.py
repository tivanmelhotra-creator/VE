"""
toonface.kit.acting — performance layer: mood keys, blinks, saccades, head micro-motion from audio energy,
stepped body breathing.  Everything is sampled ON TWOS (12 drawings / s at 24 fps) like the reference shows.

    perf = Performance(words, energy, dur, keys)       keys = [(t, MoodKey)] authored per line
    st, head = perf.at(t)                               → FaceState (draw25) + HeadPose(dx, dy, rot_deg, breath)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .draw25 import FaceState
from .visemes import mouth_at, viseme_track


@dataclass
class MoodKey:
    lid_open: float = 0.72
    lid_tilt: float = 0.0
    lower: float = 0.0
    look: tuple = (0.0, 0.0)
    brow_up: float = 0.0
    brow_in: float = 0.0
    furrow: float = 0.0
    bags: float = 0.0
    corner: float = 0.0
    skew: float = 0.0
    tilt: float = 0.0          # head roll, degrees (+ = clockwise on screen)
    nod: float = 0.0           # head pitch proxy, px down


FIELDS = ["lid_open", "lid_tilt", "lower", "brow_up", "brow_in", "furrow", "bags", "corner", "skew", "tilt", "nod"]


def _ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp_key(a: MoodKey, b: MoodKey, t: float) -> MoodKey:
    kw = {f: getattr(a, f) + (getattr(b, f) - getattr(a, f)) * t for f in FIELDS}
    kw["look"] = tuple(a.look[i] + (b.look[i] - a.look[i]) * t for i in range(2))
    return MoodKey(**kw)


@dataclass
class HeadPose:
    dx: float = 0.0
    dy: float = 0.0
    rot: float = 0.0
    breath: float = 0.0


class Performance:
    def __init__(self, words, energy: np.ndarray, energy_fps: float, keys, blinks=(), accents=(), ease_s=0.16,
                 step_fps: float = 12.0):
        self.track = viseme_track(words)
        self.energy = energy
        self.efps = energy_fps
        self.keys = sorted(keys, key=lambda k: k[0])
        self.blinks = list(blinks)           # blink start times (s)
        self.accents = list(accents)         # (t, kind, amount)  kind: nod | tilt | shake | brow
        self.ease_s = ease_s
        self.step = 1.0 / step_fps

    def _mood(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1]
        for (t0, a), (t1, b) in zip(ks, ks[1:]):
            if t0 <= t < t1:
                # hold, then a short eased transition into the next key (pose-to-pose like a key animator)
                d = min(self.ease_s * 1.6, t1 - t0)
                return lerp_key(a, b, _ease((t - (t1 - d)) / d)) if t > t1 - d else a
        return ks[-1][1]

    def _blink(self, t):
        # 0 open … 1 closed;  close 2 drawings, hold 1, open 2 (on twos) ≈ 0.25 s
        for b in self.blinks:
            x = t - b
            if 0 <= x < 0.30:
                return [0.55, 1.0, 1.0, 0.6, 0.25][min(4, int(x / 0.06))]
        return 0.0

    def _energy(self, t):
        i = int(t * self.efps)
        if 0 <= i < len(self.energy):
            return float(self.energy[i])
        return 0.0

    def at(self, t: float):
        ts = math.floor(t / self.step + 1e-6) * self.step          # ON TWOS
        k = self._mood(ts)
        m = dict(mouth_at(self.track, ts))
        m["corner"] = m.get("corner", 0.0) + k.corner * (1.0 - 0.6 * min(1.0, m.get("open", 0.0) * 1.5))
        m["skew"] = m.get("skew", 0.0) + k.skew
        bl = self._blink(ts)
        lid = k.lid_open * (1 - bl) + 0.0 * bl
        # eyelids react to loud syllables a touch (alive), never past the authored open
        e = self._energy(ts)
        brow_up = k.brow_up + 3.0 * e
        tilt, nod, shake = k.tilt, k.nod, 0.0
        for (ta, kind, amt) in self.accents:
            x = ts - ta
            if -0.05 <= x < 0.55:
                env = math.sin(math.pi * min(1.0, max(0.0, (x + 0.05) / 0.45))) if x < 0.4 else \
                    max(0.0, 1 - (x - 0.4) / 0.15) * 0.0
                if kind == "nod":
                    nod += amt * env
                elif kind == "tilt":
                    tilt += amt * env
                elif kind == "shake":
                    shake += amt * math.sin(2 * math.pi * x / 0.24) * env
                elif kind == "brow":
                    brow_up += amt * env
        st = FaceState(lid_open=lid, lid_tilt=k.lid_tilt, lower=k.lower, look=k.look, brow_up=brow_up,
                       brow_in=k.brow_in, furrow=k.furrow, bags=k.bags, yaw=shake, mouth=m)
        # head micro-bob from speech energy + breathing (body), both stepped
        bob = 2.2 * e
        breath = math.sin(2 * math.pi * ts / 3.2)
        head = HeadPose(dx=0.0, dy=nod + bob + 1.2 * breath, rot=tilt + 0.6 * e * math.sin(ts * 7.0), breath=breath)
        return st, head


def energy_envelope(pcm: np.ndarray, sr: int, fps: float = 48.0) -> np.ndarray:
    hop = int(sr / fps)
    n = len(pcm) // hop
    r = np.array([np.sqrt(np.mean(pcm[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    r = r / (np.percentile(r, 95) + 1e-9)
    k = np.ones(3) / 3
    return np.clip(np.convolve(r, k, mode="same"), 0, 1.2)
