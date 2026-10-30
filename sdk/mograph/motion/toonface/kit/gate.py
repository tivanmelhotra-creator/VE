"""
toonface.kit.gate — v31 quality-gate render: the girl (3/4) says one Persian line with full acting.

    python3 -m mograph.motion.toonface.kit.gate --audio line.mp3 --out gate.mp4 [--sheet compare.jpg]

Pipeline per drawing (on twos):
  blank plate (art-team image edit) → vector face (draw25) → forelock layer on top → head/neck warp
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
from .blankplate import load_blank
from .sheet25 import compose
from .shapes import GIRL_34

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.normpath(os.path.join(HERE, "../../../../tests/fixtures/toonface"))
SDK = os.path.normpath(os.path.join(HERE, "../../../.."))


def girl_line_keys():
    bored = MoodKey(lid_open=0.60, look=(0.10, 0.05), bags=0.65, corner=-0.15, tilt=-1.2)
    tell = MoodKey(lid_open=0.74, look=(-0.05, 0.0), bags=0.5, brow_up=7, corner=-0.05, tilt=1.0)
    insist = MoodKey(lid_open=0.70, look=(-0.10, 0.05), bags=0.5, brow_up=3, lid_tilt=0.12, corner=-0.15, tilt=0.5)
    sigh = MoodKey(lid_open=0.46, look=(0.35, -0.65), bags=0.7, brow_up=-2, corner=-0.30, skew=0.25, tilt=-2.5)
    fed = MoodKey(lid_open=0.66, look=(-0.05, 0.08), bags=0.55, brow_in=0.95, furrow=1.0, lid_tilt=0.35,
                  brow_up=-5, corner=-0.30, tilt=1.5)
    final = MoodKey(lid_open=0.58, look=(0.15, 0.10), bags=0.6, brow_in=0.8, furrow=0.9, lid_tilt=0.30, brow_up=-6,
                    corner=-0.35, skew=0.15, tilt=-1.0)
    keys = [(0.0, bored), (0.12, tell), (0.85, insist), (2.35, sigh), (3.45, fed), (5.75, final)]
    blinks = [0.74, 2.40, 3.30, 6.05]
    accents = [(0.22, "nod", 7), (0.22, "brow", 6), (1.66, "nod", 9), (1.70, "tilt", 1.5), (3.92, "shake", 4.0),
               (3.90, "brow", -3), (5.52, "nod", 10), (5.55, "tilt", -1.5)]
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


def render(audio, out, plate_path=None, fps=24, sheet=None, log=print):
    meta = json.load(open(os.path.join(FIX, "gate_line_fa.json")))
    words = [tuple(w) for w in meta["words"]]
    pcm, sr = load_audio(audio)
    dur = len(pcm) / sr
    env = energy_envelope(pcm, sr, 48.0)
    keys, blinks, accents = girl_line_keys()
    perf = Performance(words, env, 48.0, keys, blinks, accents)
    plate = cv2.imread(plate_path or os.path.join(FIX, "girl34_blank_x4.png"))
    bp = load_blank(plate, contour_cut_y=392)
    a = bp.alpha
    a = cv2.GaussianBlur(a, (3, 3), 0)
    kit = GIRL_34
    pivot = kit["neck_pivot"]
    # stage placement: character scaled to 1.22 of plate → head in the upper third, like the close-ups
    S = 1.22
    W, H = 1080, 1920
    bg = stage_bg(W, H)
    ph, pw = plate.shape[:2]
    ox, oy = int((W - pw * S) / 2) + 20, 300
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", str(fps), "-i", "-", "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                             "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                             "-shortest", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = int(math.ceil(dur * fps))
    cache_key, cache = None, None
    sub_text = meta["text"]
    stills = {}
    for i in range(n):
        t = i / fps
        st, head = perf.at(t)
        key = round(math.floor(t * 12 + 1e-6) / 12, 4)
        if key != cache_key:
            rgb = compose(bp, kit, st)
            rgba = np.dstack([rgb, (a * 255).astype(np.uint8)])
            rgba = head_warp(rgba, pivot, head.rot, head.dy, head.breath, 660, 790)
            big = cv2.resize(rgba, (int(pw * S), int(ph * S)), interpolation=cv2.INTER_AREA if S < 1 else
                             cv2.INTER_CUBIC)
            frame = bg.copy()
            h2, w2 = big.shape[:2]
            x0, y0 = max(0, ox), max(0, oy)
            x1, y1 = min(W, ox + w2), min(H, oy + h2)
            crop = big[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
            al = crop[:, :, 3:4].astype(np.float32) / 255.0
            frame[y0:y1, x0:x1] = (frame[y0:y1, x0:x1] * (1 - al) + crop[:, :, :3] * al).astype(np.uint8)
            if 0.1 < t < dur - 0.05:
                frame = subtitle(frame, "ببین، من که بهت گفتم..." if t < 3.0 else "دیگه اصلاً حوصله‌ی این بازی‌ها رو ندارم!")
            cache_key, cache = key, frame
            for name, tt in (("tell", 0.42), ("gofta", 1.80), ("sigh", 2.80), ("asla", 4.05), ("nadaram", 5.70)):
                if abs(t - tt) < 0.5 / fps + 1e-6 or (name not in stills and t >= tt):
                    stills.setdefault(name, frame.copy())
        proc.stdin.write(cache.tobytes())
    proc.stdin.close()
    proc.wait()
    log(f"[gate] {out}  {n} frames  {dur:.2f}s")
    return stills


def comparison_sheet(stills, refs, out):
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
    ours = [lab(o, n) for o, n in zip(ours, ["v31 kit: 'bebin'", "v31 kit: sigh / eye-roll", "v31 kit: 'asla' (frown)",
                                             "v31 kit: 'nadaram'"])]
    rr = [lab(r, n) for r, n in zip(rr, ["ref: khab 14.0s", "ref: khab 18.5s", "ref: roshanfekr 13.5s",
                                         "ref: roshanfekr 18.0s"])]
    sheet = np.vstack([np.hstack(ours), np.hstack(rr)])
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plate")
    ap.add_argument("--sheet")
    ap.add_argument("--refs", nargs="*", default=[])
    a = ap.parse_args()
    stills = render(a.audio, a.out, a.plate)
    if a.sheet:
        refs = [cv2.imread(p) for p in a.refs]
        comparison_sheet(stills, refs, a.sheet)
        print("[gate] sheet", a.sheet)


if __name__ == "__main__":
    main()
