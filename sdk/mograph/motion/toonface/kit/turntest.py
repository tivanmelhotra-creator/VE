"""
toonface.kit.turntest — Head Turn & Stability test (girl + boy side by side, ≈5 s, talking + blinking).

    python3 -m mograph.motion.toonface.kit.turntest --audio line.mp3 --out turn_test.mp4 [--dur 5.0]

Turn script (pose-to-pose, eased, on twos): front → 3/4 right (+25°) → front → 3/4 left (−25°, mirrored key) →
front, with a small pitch nod in the middle hold.  The mouth is lip-synced from the Persian G2P viseme track, the
eyes blink and saccade.

Stability metric (printed + burnt into the HUD), measured per drawing:
    offset(t) = face anchor (eye midpoint after the head transform)  −  skull centroid (turned plate face mask)
    jitter    = |offset(t) − 2·offset(t−1) + offset(t−2)|  (second difference → any pop / float shows up here)
    bob       = vertical anchor motion from nod/bob/breath (golden rule ≤ ±6–8 px)
"""
from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np

from .acting import MoodKey, Performance, energy_envelope
from .gate import FIX, ffmpeg_writer, load_audio, stage_bg, subtitle
from .turn25 import Character

W, H, S = 1920, 1080, 1.08
CROP_Y = 70


def turn_keys():
    def K(yaw, pitch=0.0, **kw):
        base = dict(lid_open=0.84, look=(0.0, 0.0))
        base.update(kw)
        return MoodKey(yaw=yaw, pitch=pitch, **base)
    keys = [(0.00, K(0, brow_up=0.2, corner=0.10)),
            (0.95, K(25, look=(-0.12, 0.0), brow_up=0.6, corner=0.05)),     # turn → 3/4 right
            (1.90, K(0, pitch=6, brow_anger=0.6, furrow=0.6, lid_tilt=0.2, corner=-0.15)),
            (3.10, K(-25, look=(0.12, 0.0), brow_worry=0.5, corner=-0.10)),  # turn → 3/4 left (mirrored)
            (4.20, K(0, brow_up=0.3, corner=0.15))]
    blinks = [0.55, 2.30, 3.75]
    accents = [(1.20, "nod", 5), (1.25, "brow", 0.4), (2.60, "tilt", 1.5), (4.40, "nod", 4)]
    return keys, blinks, accents


def _place(frame, rgba, ox, oy):
    big = cv2.resize(rgba[CROP_Y:], (int(rgba.shape[1] * S), int((rgba.shape[0] - CROP_Y) * S)),
                     interpolation=cv2.INTER_CUBIC)
    h2, w2 = big.shape[:2]
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + w2), min(H, oy + h2)
    crop = big[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    al = crop[:, :, 3:4].astype(np.float32) / 255.0
    frame[y0:y1, x0:x1] = (frame[y0:y1, x0:x1] * (1 - al) + crop[:, :, :3] * al).astype(np.uint8)


def _hud(frame, x, lines):
    for j, s in enumerate(lines):
        cv2.putText(frame, s, (x, 40 + 34 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.putText(frame, s, (x, 40 + 34 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def render(audio, out, dur=5.0, fps=24, log=print, strip=None):
    meta = json.load(open(os.path.join(FIX, "gate_line_fa.json")))
    words = [tuple(w) for w in meta["words"] if w[1] < dur]
    pcm, sr = load_audio(audio)
    pcm = pcm[:int(dur * sr)]
    env = energy_envelope(pcm, sr, 48.0)
    keys, blinks, accents = turn_keys()
    perf = Performance(words, env, 48.0, keys, blinks, accents, turn_s=0.62)
    chars = [Character("girl"), Character("boy")]
    xs = [-40, W // 2 - 60]
    bg = stage_bg(W, H)
    tmp = out + ".tmp.mp4"
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio, "-t", f"{dur:.3f}", "-c:a", "aac", tmp + ".m4a"],
                   check=True)
    proc = ffmpeg_writer(out, W, H, fps, tmp + ".m4a")
    hist = [[], []]
    stats = {"girl": {"jitter_max": 0.0, "bob_max": 0.0}, "boy": {"jitter_max": 0.0, "bob_max": 0.0}}
    cache_key, cache = None, None
    stills = []
    n = int(round(dur * fps))
    for i in range(n):
        t = i / fps
        key = math.floor(t * 12 + 1e-6)
        if key != cache_key:
            st, head = perf.at(t)
            frame = bg.copy()
            hud = []
            for ci, (ch, ox) in enumerate(zip(chars, xs)):
                rgba = ch.render(head.yaw, st, head, pitch=head.pitch)
                _place(frame, rgba, ox, 0)
                an = ch.anchor(head.yaw, head.pitch, head)
                off = np.array([an["cx"], an["cy"]]) - ch.last_skull
                hist[ci].append(off)
                jit = 0.0
                if len(hist[ci]) >= 3:
                    a, b, c = hist[ci][-3:]
                    jit = float(np.linalg.norm(c - 2 * b + a))
                nm = ("girl", "boy")[ci]
                stats[nm]["jitter_max"] = max(stats[nm]["jitter_max"], jit)
                stats[nm]["bob_max"] = max(stats[nm]["bob_max"], abs(head.dy))
                hud.append((ox + 70, [f"{nm}  yaw {head.yaw:+5.1f}  pitch {head.pitch:+4.1f}",
                                      f"anchor-skull jitter {jit:4.2f}px  bob {head.dy:+4.1f}px"]))
            for x, ls in hud:
                _hud(frame, x, ls)
            sub = "ببین، من که بهت گفتم..." if t < 3.0 else "دیگه اصلاً حوصله‌ی ..."
            frame = subtitle(frame, sub, y=H - 110)
            cache_key, cache = key, frame
            if strip is not None and key % 6 == 0:
                stills.append(cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA))
        proc.stdin.write(cache.tobytes())
    proc.stdin.close()
    proc.wait()
    os.remove(tmp + ".m4a")
    if strip is not None and stills:
        rows = [np.hstack(stills[k:k + 5] + [np.zeros_like(stills[0])] * (5 - len(stills[k:k + 5])))
                for k in range(0, len(stills), 5)]
        cv2.imwrite(strip, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    for nm in stats:
        stats[nm] = {k: round(v, 3) for k, v in stats[nm].items()}
    log(f"[turntest] {out}  {n} frames  {dur:.2f}s  stability={stats}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--strip")
    a = ap.parse_args()
    render(a.audio, a.out, a.dur, strip=a.strip)


if __name__ == "__main__":
    main()
