"""
Animated 3×3 "mood sheet" — the reference character sheets brought to life: every cell is the SAME ball-head toon
character driven by a REAL motion-capture clip, with a matching cartoon expression + 2D effects, rendered on twos.

    python3 examples/toon/mood_sheet.py OUT_DIR [--sheet A|B] [--dur 6] [--jobs 2] [--still t]
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import cairo  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from mograph.human.lineart import Camera  # noqa: E402
from mograph.mocap.director import Actor, Scene  # noqa: E402
from mograph.toon import PAPER, ToonCharacter, compose, head_px, quantize, render_toon, toon_style  # noqa: E402
from mograph.toon import fx  # noqa: E402

CW, CH = 360, 480                  # cell
W, H = CW * 3, CH * 3              # 1080 × 1440
FPS = 24
FONT = "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"

# label, clip, clip offset (s), speed, heading, expression(s) [(t, expr)], fx list, cam (target y, fov), talk
SHEETS = {
    "A": [
        ("در حال دویدن با عجله", "ual1:Jog_Fwd_Loop", 0.0, 1.0, 55, [(0, "furious")], ["speed"], (8.6, 25), 0),
        ("خسته و بی‌حال", "cmu:142_15", 0.5, 0.8, -35, [(0, "tired")], ["sweat"], (8.6, 25), 0),
        ("بسیار ترسیده و مضطرب", "cmu:22_03", 0.4, 1.0, 10, [(0, "scared")], ["fly_sweat", "shake"], (7.6, 25), 0),
        ("در حال پرش یا مبارزه", "ual1:Punch_Cross", 0.0, 0.9, 35, [(0, "angry")], ["speed_r"], (8.6, 25), 0),
        ("متعجب و شوکه", "xbot:idle", 0.0, 1.0, 10, [(0, "neutral"), (1.4, "shock")], ["mark"], (8.6, 25), 0),
        ("خوشحال و پرانرژی", "cmu:79_69", 0.8, 1.0, 15, [(0, "happy")], ["sparkles"], (9.0, 27), 0),
        ("عصبانی", "ual2:Idle_FoldArms_Loop", 0.0, 1.0, 20, [(0, "angry")], ["steam"], (8.6, 25), 0),
        ("فکر کردن و سردرگمی", "ual1:Idle_Talking_Loop", 0.0, 0.7, 25, [(0, "thinking")], ["q"], (8.6, 25), 0),
        ("افسرده و غمگین", "cmu:82_10", 0.0, 0.7, -15, [(0, "sad")], [], (8.6, 25), 0),
    ],
    "B": [
        ("در حال فکر کردن عمیق", "cmu:13_04", 3.8, 0.6, 30, [(0, "thinking")], [], (6.8, 25), 0),
        ("بسیار هیجان‌زده", "cmu:13_10", 0.4, 1.0, 15, [(0, "excited")], ["sparkles"], (9.2, 28), 0),
        ("بسیار گرسنه", "cmu:142_15", 1.0, 0.7, 15, [(0, "hungry")], ["growl"], (8.6, 25), 0),
        ("ترسیده از دیدن چیزی", "cmu:15_07", 3.8, 1.0, 10, [(0, "scared")], ["fly_sweat"], (8.6, 25), 0),
        ("در حال چرت زدن", "ual1:Sitting_Idle_Loop", 0.0, 0.6, 15, [(0, "sleepy")], ["zzz"], (6.5, 25), 0),
        ("مغرور و با اعتماد به نفس", "ual2:Idle_FoldArms_Loop", 0.0, 0.8, 20, [(0, "smug")], [], (8.6, 25), 0),
        ("در حال دویدن از ترس", "ual1:Sprint_Loop", 0.0, 0.9, 50, [(0, "scared")], ["fly_sweat", "speed"], (8.6, 25),
         0),
        ("بی‌حوصله و کلافه", "cmu:22_03", 1.2, 0.6, 20, [(0, "bored")], [], (7.8, 25), 0),
        ("در حال صحبت با هیجان", "ual1:Idle_Talking_Loop", 0.0, 1.0, 20, [(0, "happy")], [], (8.6, 25), 1),
    ],
}


def build(sheet):
    cells = []
    for k, (lab, clip, off, sp, yaw, exprs, fxs, camc, talk) in enumerate(SHEETS[sheet]):
        tc = ToonCharacter("guy", seed=k)
        A = Actor("a", tc.ch, [{"clip": clip, "t": -off / sp, "at": [0, 0], "heading": yaw, "root": "in_place",
                                "speed": sp}], breath=0.6, seed=k,
                  look=[{"t": -10, "dur": 0.1, "target": [0, 9.0, 62.0], "w": 0.75}])
        cells.append(dict(lab=lab, tc=tc, S=Scene([A]), exprs=exprs, fx=fxs, cam=camc, talk=talk))
    return cells


def expr_at(exprs, t):
    e = exprs[0][1]
    for t0, x in exprs:
        if t >= t0:
            e = x
    return e


_LABELS = {}


def label_img(text, w):
    if text not in _LABELS:
        f = ImageFont.truetype(FONT, 30)
        im = Image.new("RGBA", (w, 56), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((w // 2, 30), text, font=f, fill=(25, 22, 20, 255), anchor="mm", direction="rtl", language="fa")
        _LABELS[text] = np.asarray(im)
    return _LABELS[text]


def render_cell(c, t):
    tq = quantize(t, 12)
    ty, fov = c["cam"]
    cam = Camera(target=(0, ty, 0), yaw=0, pitch=3, fov=fov, W=CW, H=CH)
    cam.dist = 62
    fr = render_toon(c["S"], {"a": c["tc"]}, tq, cam, toon_style(4.6), ss=1.5, fig_px=CH * 0.9, seed=int(tq * 12))
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, CW, CH)
    ctx = cairo.Context(s)
    ctx.set_source_rgb(*PAPER)
    ctx.paint()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ex = expr_at(c["exprs"], t)
    talk = (0.5 + 0.5 * math.sin(tq * 19)) * c["talk"]
    compose(ctx, fr, cam, {"a": c["tc"]}, {"a": dict(expr=ex, look=None, talk=talk)}, CW, CH, tq, boil_amp=0.7,
            t_draw=tq)
    G = fr["G"]["a"]
    (hx, hy), r = head_px(cam, c["tc"], G)
    w = max(1.6, r * 0.045)
    for f in c["fx"]:
        if f == "speed":
            fx.speed_lines(ctx, hx, hy + r * 2.5, r, tq, direction=-1, w=w)
        elif f == "speed_r":
            fx.speed_lines(ctx, hx, hy + r * 1.5, r, tq, direction=-1, w=w, n=3)
        elif f == "sweat":
            fx.sweat(ctx, hx, hy, r, tq, side=1, w=w)
        elif f == "fly_sweat":
            fx.fly_sweat(ctx, hx, hy, r, tq, w=w)
        elif f == "shake":
            fx.shake_lines(ctx, hx, hy + r * 3.0, r * 1.2, tq, w=w)
        elif f == "mark" and t >= 1.4:
            fx.mark(ctx, hx, hy, r, t, "!", pop_t=1.4)
        elif f == "sparkles":
            fx.sparkles(ctx, hx, hy, r, tq, w=w)
        elif f == "steam":
            fx.steam(ctx, hx, hy, r, tq, w=w, side=1)
            fx.steam(ctx, hx, hy, r, tq + 0.4, w=w, side=-1)
        elif f == "q":
            fx.mark(ctx, hx, hy, r, t, "?", pop_t=0.3)
        elif f == "zzz":
            fx.zzz(ctx, hx, hy, r, tq)
        elif f == "growl":
            fx.growl(ctx, hx, hy + r * 3.2, r, tq, w=w, text="قور")
    s.flush()
    a = np.ndarray((CH, CW, 4), np.uint8, s.get_data())[..., :3][..., ::-1].copy()      # RGB
    L = label_img(c["lab"], CW)
    al = L[..., 3:4] / 255.0
    a[6:62] = (a[6:62] * (1 - al) + L[..., :3] * al).astype(np.uint8)
    return a


def frame(cells, t):
    img = np.zeros((H, W, 3), np.uint8)
    for k, c in enumerate(cells):
        y, x = (k // 3) * CH, (k % 3) * CW
        img[y:y + CH, x:x + CW] = render_cell(c, t)
    img[:, CW - 2:CW + 2] = 250
    img[:, 2 * CW - 2:2 * CW + 2] = 250
    img[CH - 2:CH + 2] = 250
    img[2 * CH - 2:2 * CH + 2] = 250
    return img


def worker(args):
    sheet, idx, out = args
    cells = build(sheet)
    for i in idx:
        p = os.path.join(out, f"f{i:05d}.png")
        if not os.path.exists(p):
            cv2.imwrite(p, frame(cells, i / FPS)[..., ::-1])
    return len(idx)


def main():
    out = sys.argv[1]
    sheet = sys.argv[sys.argv.index("--sheet") + 1] if "--sheet" in sys.argv else "A"
    dur = float(sys.argv[sys.argv.index("--dur") + 1]) if "--dur" in sys.argv else 6.0
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 2
    os.makedirs(out, exist_ok=True)
    if "--still" in sys.argv:
        t = float(sys.argv[sys.argv.index("--still") + 1])
        cv2.imwrite(os.path.join(out, f"still_{sheet}.png"), frame(build(sheet), t)[..., ::-1])
        return
    fd = os.path.join(out, f"frames_{sheet}")
    os.makedirs(fd, exist_ok=True)
    n = int(dur * FPS)
    t0 = time.time()
    from multiprocessing import Pool
    chunks = [(sheet, list(range(j, n, jobs)), fd) for j in range(jobs)]
    with Pool(jobs) as p:
        p.map(worker, chunks)
    print(f"frames {time.time() - t0:.0f}s")
    mp4 = os.path.join(out, f"mood_sheet_{sheet}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(fd, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-threads", "2", "-preset", "medium",
                    "-x264-params", "rc-lookahead=10:ref=2", mp4], check=True)
    print(mp4)


if __name__ == "__main__":
    main()
