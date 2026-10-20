"""
mograph.v14.showcase — the @novadent "BEFORE / AFTER" editor-showcase wrapper.

Two tilted rounded cards on a heavily blurred, darkened copy of the AFTER video:
  BEFORE (raw take, small, tilted back, desaturated) behind  ·  AFTER (the edit, big, front, teal rim)
Cards float (slow 3D sway), the AFTER card punches to full screen on "full_at" windows (ease in/out).

python3 -m mograph.v14.showcase after.mp4 raw.mp4 out.mp4 --segments "1.12-7.14,7.44-12.06,12.24-17.62" \
        --full "6.0-6.8,9.9-10.6"
"""
from __future__ import annotations

import argparse
import math
import subprocess

import cairo
import cv2
import numpy as np

from . import physics as ph


def _rr(c, x, y, w, h, r):
    c.new_sub_path()
    c.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    c.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    c.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    c.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    c.close_path()


def _card(ctx, img, cx, cy, w, h, rot, rim, label, a=1.0, ry=0.0):
    ih, iw = img.shape[:2]
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, iw, ih)
    arr = np.ndarray((ih, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :iw]
    arr[:, :, :3] = img
    arr[:, :, 3] = 255
    surf.mark_dirty()
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(math.radians(rot))
    ctx.scale(math.cos(math.radians(ry)), 1.0)
    r = w * 0.07
    # shadow
    for i in range(6):
        _rr(ctx, -w / 2 + 6, -h / 2 + 18 + i * 3, w, h, r)
        ctx.set_source_rgba(0, 0, 0, 0.07 * a)
        ctx.fill()
    _rr(ctx, -w / 2, -h / 2, w, h, r)
    ctx.save()
    ctx.clip()
    ctx.scale(w / iw, h / ih)
    ctx.set_source_surface(surf, -iw / 2, -ih / 2)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint_with_alpha(a)
    ctx.restore()
    _rr(ctx, -w / 2, -h / 2, w, h, r)
    ctx.set_source_rgba(rim[0], rim[1], rim[2], 0.85 * a)
    ctx.set_line_width(2.5)
    ctx.stroke()
    # label
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    fs = w * 0.055
    ctx.set_font_size(fs)
    ex = ctx.text_extents(label)
    ctx.move_to(-ex.width / 2, -h / 2 - fs * 0.6)
    ctx.set_source_rgba(1, 1, 1, 0.92 * a)
    ctx.show_text(label)
    ctx.restore()


def build(after, raw, out, segments, full=(), W=1080, H=1920, fps=30):
    ca, cr = cv2.VideoCapture(after), cv2.VideoCapture(raw)
    n = int(ca.get(cv2.CAP_PROP_FRAME_COUNT))
    rfps = cr.get(cv2.CAP_PROP_FPS) or 30
    cum = [0.0]
    for a, b in segments:
        cum.append(cum[-1] + b - a)

    def raw_t(t):
        for i, (a, b) in enumerate(segments):
            if t <= cum[i + 1]:
                return a + t - cum[i]
        return segments[-1][1]
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-i", after, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264",
                          "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", out],
                         stdin=subprocess.PIPE)
    last_ri, rimg = -1, None
    for i in range(n):
        ok, fa = ca.read()
        if not ok:
            break
        t = i / fps
        ri = int(raw_t(t) * rfps)
        if ri != last_ri:
            if ri != last_ri + 1:
                cr.set(cv2.CAP_PROP_POS_FRAMES, ri)
            ok2, fr = cr.read()
            if ok2:
                rimg = fr
            last_ri = ri
        # full-screen weight
        fw = 0.0
        for a, b in full:
            fw = max(fw, min(ph.SILK((t - a + 0.35) / 0.35), ph.SILK((b + 0.35 - t) / 0.35)))
        # background
        bg = cv2.resize(cv2.GaussianBlur(cv2.resize(fa, (W // 8, H // 8)), (0, 0), 6), (W, H))
        bg = (bg.astype(np.float32) * 0.42).astype(np.uint8)
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        arr = np.ndarray((H, surf.get_stride() // 4, 4), np.uint8, surf.get_data())[:, :W]
        arr[:, :, :3] = bg
        arr[:, :, 3] = 255
        surf.mark_dirty()
        c = cairo.Context(surf)
        sway = ph.osc(t, 0.12, 1.0, 2)
        g = cv2.cvtColor(cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if rimg is not None else fa
        rimg_d = cv2.addWeighted(rimg, 0.55, g, 0.45, 0) if rimg is not None else fa
        bw, bh = 330, 330 * 16 / 9
        _card(c, rimg_d, 285 - 60 * fw, 1000, bw, bh, -7 + sway, (1, 1, 1), "BEFORE", a=1 - fw * 0.9, ry=18)
        aw = 560 + (W - 560) * fw
        ah = aw * 16 / 9
        _card(c, fa, 640 + (540 - 640) * fw, 960 + 0 * fw, aw, min(ah, H + 40 * (1 - fw)), (2.5 + sway * .6) * (1 - fw),
              (0.18, 0.83, 0.75), "AFTER" if fw < 0.6 else "", ry=-6 * (1 - fw))
        surf.flush()
        p.stdin.write(np.ascontiguousarray(arr[:, :, :3]).tobytes())
    p.stdin.close()
    p.wait()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("after"); ap.add_argument("raw"); ap.add_argument("out")
    ap.add_argument("--segments", required=True)
    ap.add_argument("--full", default="")
    a = ap.parse_args()
    seg = [tuple(float(v) for v in s.split("-")) for s in a.segments.split(",")]
    full = [tuple(float(v) for v in s.split("-")) for s in a.full.split(",") if s]
    build(a.after, a.raw, a.out, seg, full)
