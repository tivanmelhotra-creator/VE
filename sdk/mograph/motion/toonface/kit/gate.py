"""
toonface.kit.gate — v31 quality-gate render: the girl (3/4) says one Persian line with full acting.

    python3 -m mograph.motion.toonface.kit.gate --audio line.mp3 --out gate.mp4 [--sheet compare.jpg]

Pipeline per drawing (on twos):
  turn25.Character (blank plates + rig25 multi-view vector face, yaw/pitch) → forelock on top → head/neck warp
  (rotation + nod around the neck pivot with a smooth falloff, body untouched) → stepped breathing on the torso →
  vertical 1080×1920 stage + Persian subtitle → ffmpeg (libx264 + AAC)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess

import cv2
import numpy as np

from .acting import MoodKey, Performance, energy_envelope
from .turn25 import Character

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.normpath(os.path.join(HERE, "../../../../tests/fixtures/toonface"))
SDK = os.path.normpath(os.path.join(HERE, "../../../.."))


def girl_line_keys(yaw=25.0):
    """«ببین، من که بهت گفتم... دیگه اصلاً حوصله‌ی این بازی‌ها رو ندارم!» — bored → telling → sigh/eye-roll → fed-up."""
    Y = yaw
    bored = MoodKey(lid_open=0.66, look=(0.10, 0.05), corner=-0.15, tilt=-1.2, yaw=Y)
    tell = MoodKey(lid_open=0.86, look=(-0.05, 0.0), brow_up=0.9, pupil=0.95, corner=-0.05, tilt=1.0, yaw=Y)
    insist = MoodKey(lid_open=0.80, look=(-0.10, 0.05), brow_up=0.35, brow_anger=0.25, lid_tilt=0.10,
                     corner=-0.15, tilt=0.5, yaw=Y)
    sigh = MoodKey(lid_open=0.50, look=(0.35, -0.65), brow_up=-0.25, brow_worry=0.55, corner=-0.30, skew=0.25,
                   tilt=-2.5, yaw=Y)
    fed = MoodKey(lid_open=0.70, look=(-0.05, 0.08), brow_anger=1.0, furrow=1.0, lid_tilt=0.30, corner=-0.30,
                  tilt=1.5, yaw=Y)
    final = MoodKey(lid_open=0.62, look=(0.15, 0.10), brow_anger=0.85, furrow=0.9, lid_tilt=0.28, corner=-0.35,
                    skew=0.15, tilt=-1.0, yaw=Y)
    keys = [(0.0, bored), (0.12, tell), (0.85, insist), (2.35, sigh), (3.45, fed), (5.75, final)]
    blinks = [0.74, 2.40, 3.30, 6.05]
    accents = [(0.22, "nod", 5), (0.22, "brow", 0.45), (1.66, "nod", 6), (1.70, "tilt", 1.5), (3.92, "shake", 3.0),
               (3.90, "brow", -0.3), (5.52, "nod", 6), (5.55, "tilt", -1.5)]
    return keys, blinks, accents


def head_warp(img, pivot, rot_deg, dy, breath, y_head_end, y_body_start):
    """rotate/shift everything above the neck around the pivot, fade to identity over the neck, and breathe the
    torso (tiny vertical stretch anchored at the bottom).  Inverse mapping for cv2.remap."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    w = np.clip((y_body_start - yy) / max(1.0, (y_body_start - y_head_end)), 0, 1)
    w = w * w * (3 - 2 * w)
    a = -math.radians(rot_deg) * w
    px, py = pivot
    ca, sa = np.cos(a), np.sin(a)
    X = xx - px
    Y = yy - py - dy * w
    sx = px + ca * X - sa * Y
    sy = py + sa * X + ca * Y
    # torso breathing: stretch below the neck by up to 0.5 %, anchored at the bottom edge
    s = 1.0 + 0.005 * breath
    tb = (1 - w)
    sy = sy * (1 - tb) + (H - (H - sy) / s) * tb
    return cv2.remap(img, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def stage_bg(W=1080, H=1920):
    y = np.linspace(0, 1, H)[:, None, None]
    top = np.array([150, 176, 214], np.float32)       # warm peach room (BGR)
    bot = np.array([112, 140, 186], np.float32)
    bg = top * (1 - y) + bot * y
    bg = np.repeat(bg, W, axis=1)
    xx = np.linspace(-1, 1, W)[None, :, None]
    yy = np.linspace(-1, 1, H)[:, None, None]
    vig = 1 - 0.18 * (xx ** 2 + 0.6 * yy ** 2)
    return np.clip(bg * vig, 0, 255).astype(np.uint8)


def subtitle(canvas, text, y=1640):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(os.path.join(SDK, "fonts/Vazirmatn-Bold.ttf"), 54)
    bb = d.textbbox((0, 0), text, font=font, direction="rtl", language="fa")
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (canvas.shape[1] - tw) // 2
    d.rounded_rectangle((x - 26, y - 18, x + tw + 26, y + th + 30), 12, fill=(12, 10, 10))
    d.text((x - bb[0], y - bb[1] + 4), text, font=font, fill=(255, 255, 255), direction="rtl", language="fa")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def load_audio(path, sr=48000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32), sr


class Stage:
    """vertical 1080×1920 stage: character plate scaled & placed, optional subtitle."""

    def __init__(self, W=1080, H=1920, S=1.22, ox=None, oy=300, pw=860, ph=1180):
        self.W, self.H, self.S = W, H, S
        self.bg = stage_bg(W, H)
        self.ox = int((W - pw * S) / 2) + 20 if ox is None else ox
        self.oy = oy

    def compose(self, rgba, sub=None, bg=None):
        big = cv2.resize(rgba, (int(rgba.shape[1] * self.S), int(rgba.shape[0] * self.S)), interpolation=cv2.INTER_CUBIC)
        frame = (self.bg if bg is None else bg).copy()
        h2, w2 = big.shape[:2]
        ox, oy, W, H = self.ox, self.oy, self.W, self.H
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(W, ox + w2), min(H, oy + h2)
        crop = big[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        al = crop[:, :, 3:4].astype(np.float32) / 255.0
        frame[y0:y1, x0:x1] = (frame[y0:y1, x0:x1] * (1 - al) + crop[:, :, :3] * al).astype(np.uint8)
        if sub:
            frame = subtitle(frame, sub)
        return frame


def ffmpeg_writer(out, W, H, fps, audio=None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-"]
    if audio:
        cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render(audio, out, character="girl", yaw=25.0, fps=24, log=print):
    meta = json.load(open(os.path.join(FIX, "gate_line_fa.json")))
    words = [tuple(w) for w in meta["words"]]
    pcm, sr = load_audio(audio)
    dur = len(pcm) / sr
    env = energy_envelope(pcm, sr, 48.0)
    keys, blinks, accents = girl_line_keys(yaw)
    perf = Performance(words, env, 48.0, keys, blinks, accents)
    ch = Character(character)
    stage = Stage()
    proc = ffmpeg_writer(out, stage.W, stage.H, fps, audio)
    n = int(math.ceil(dur * fps))
    cache_key, cache = None, None
    stills = {}
    for i in range(n):
        t = i / fps
        key = round(math.floor(t * 12 + 1e-6) / 12, 4)                  # ON TWOS
        if key != cache_key:
            st, head = perf.at(t)
            rgba = ch.render(head.yaw, st, head, pitch=head.pitch)
            sub = None
            if 0.1 < t < dur - 0.05:
                sub = "ببین، من که بهت گفتم..." if t < 3.0 else "دیگه اصلاً حوصله‌ی این بازی‌ها رو ندارم!"
            cache_key, cache = key, stage.compose(rgba, sub)
            for name, tt in (("tell", 0.42), ("gofta", 1.80), ("sigh", 2.80), ("asla", 4.05), ("nadaram", 5.70)):
                if name not in stills and t >= tt:
                    stills[name] = cache.copy()
        proc.stdin.write(cache.tobytes())
    proc.stdin.close()
    proc.wait()
    log(f"[gate] {out}  {n} frames  {dur:.2f}s")
    return stills


def comparison_sheet(stills, refs, out, ref_labels=None):
    """row 1: our close-ups; row 2: reference close-ups (خواب / روشنفکر) — same crop size."""
    tw, th = 540, 600
    ours = []
    for k in ("tell", "sigh", "asla", "nadaram"):
        f = stills[k]
        c = f[330:330 + 900, 140:140 + 810]
        ours.append(cv2.resize(c, (tw, th), interpolation=cv2.INTER_AREA))
    rr = [cv2.resize(r, (tw, th), interpolation=cv2.INTER_AREA) for r in refs[:4]]
    def lab(img, s):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (20, 20, 20), -1)
        cv2.putText(img, s, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img
    ours = [lab(o, n) for o, n in zip(ours, ["v31.3: 'bebin' (brow jump)", "v31.3: sigh / eye-roll (worry)",
                                             "v31.3: 'asla' (anger V)", "v31.3: 'nadaram'"])]
    rows = [np.hstack(ours)]
    if rr:
        rr = [lab(r, n) for r, n in zip(rr, ref_labels or ["ref"] * len(rr))]
        while len(rr) < 4:
            rr.append(np.full_like(rr[0], 40))
        rows.append(np.hstack(rr))
    sheet = np.vstack(rows)
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--character", default="girl")
    ap.add_argument("--yaw", type=float, default=25.0)
    ap.add_argument("--sheet")
    ap.add_argument("--refs", nargs="*", default=[])
    a = ap.parse_args()
    stills = render(a.audio, a.out, a.character, a.yaw)
    if a.sheet:
        ref_paths = a.refs or [os.path.join(FIX, f"refs/ref_{i}.jpg") for i in (1, 2, 3, 4)
                               if os.path.exists(os.path.join(FIX, f"refs/ref_{i}.jpg"))]
        refs = [cv2.imread(p) for p in ref_paths if os.path.exists(p)]
        ref_labels = ["ref: khab 14.0s", "ref: khab 18.5s", "ref: roshanfekr 15.0s", "ref: roshanfekr 17.0s"]
        comparison_sheet(stills, refs, a.sheet, ref_labels=ref_labels)
        print("[gate] sheet", a.sheet)


if __name__ == "__main__":
    main()
