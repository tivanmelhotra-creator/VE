"""
mograph.human.clip — animated line-art clips (alpha video / PNG sequence / SVG frames) from a Character.

    from mograph.human.clip import render_clip
    render_clip("office_man", "walk_cycle", "out.mov", seconds=3, fps=24, yaw=70, W=720, H=1280)

Motion specs (`motion`):
    "walk_cycle" / "run_cycle"      in-place cycle (period from `cycle_s`, default 1.1 s / 0.7 s)
    "walk_across"                   walks across the frame (root translates, camera fixed)
    "wave" / "idle"                 gesture loops
    {"keys": [(t, "pose_name", {kw}), …]}   keyframed library poses, eased (smootherstep) per segment
    callable(t) → pose dict

Stability rules (what makes a line-art figure usable in an edit):
  · the camera is framed ONCE from the whole motion (union bounding box) → no breathing/zoom per frame
  · line width scale is computed once (figure height in px) → no width flicker
  · stroke wobble seed is fixed per character → the "hand" is consistent frame to frame
  · output is premultiplied-alpha RGBA; mov = ProRes 4444 (alpha), webm = VP9 alpha, or png sequence
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile

import numpy as np

from .character import Character, preset
from .lineart import Style, draw_strokes, strokes_svg
from . import poses as PZ


def smoother(t):
    t = min(1.0, max(0.0, t))
    return t * t * t * (t * (6 * t - 15) + 10)


def _lerp_pose(a: dict, b: dict, u: float) -> dict:
    """blend two pose dicts (numbers / tuples / nested dicts); strings switch at u = 0.5."""
    out = {}
    for k in set(a) | set(b):
        va, vb = a.get(k, b.get(k)), b.get(k, a.get(k))
        if isinstance(va, dict) and isinstance(vb, dict):
            out[k] = _lerp_pose(va, vb, u)
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            out[k] = va + (vb - va) * u
        elif isinstance(va, (tuple, list)) and isinstance(vb, (tuple, list)) and len(va) == len(vb):
            out[k] = tuple(x + (y - x) * u for x, y in zip(va, vb))
        else:
            out[k] = vb if u >= 0.5 else va
    return out


def motion_fn(motion, cycle_s=None, speed_dm_s=13.0):
    """→ f(t) returning a pose dict."""
    if callable(motion):
        return motion
    if isinstance(motion, dict) and "keys" in motion:
        keys = sorted(motion["keys"], key=lambda k: k[0])
        poses = [(t, PZ.get(n, **(kw or {}))) for t, n, *rest in keys for kw in [rest[0] if rest else {}]]

        def f(t):
            if t <= poses[0][0]:
                return poses[0][1]
            for (t0, p0), (t1, p1) in zip(poses[:-1], poses[1:]):
                if t <= t1:
                    return _lerp_pose(p0, p1, smoother((t - t0) / max(1e-6, t1 - t0)))
            return poses[-1][1]
        return f
    if motion in ("walk_cycle", "walk_across"):
        T = cycle_s or 1.1

        def f(t):
            p = PZ.walk(phase=(t / T) % 1.0)
            if motion == "walk_across":
                r = dict(p.get("root", {}))
                r["z"] = speed_dm_s * t * (T / 1.1) * 0.95
                p["root"] = r
            return p
        return f
    if motion == "run_cycle":
        T = cycle_s or 0.7
        return lambda t: PZ.run(phase=(t / T) % 1.0)
    if motion == "wave":
        return lambda t: PZ.wave(t=t)
    if motion == "idle":
        def f(t):
            p = PZ.stand_contra()
            br = math.sin(2 * math.pi * t / 4.0)
            p["spine"] = (2 + 0.8 * br, -4, 4)
            p["neck"] = (-2 - 0.6 * br, 3 + 1.5 * math.sin(2 * math.pi * t / 6.3), -6)
            return p
        return f
    if isinstance(motion, str):
        return lambda t: PZ.get(motion)
    raise ValueError(f"unknown motion {motion!r}")


def render_frames(ch: Character, motion, seconds=3.0, fps=24, yaw=30.0, W=720, H=1280, pitch=4.0, fov=18.0,
                  margin=0.06, ss=1.5, cycle_s=None, yaw_end=None, svg_dir=None, follow=False):
    f = motion_fn(motion, cycle_s)
    n = max(1, int(round(seconds * fps)))
    ts = [i / fps for i in range(n)]
    # pass 1: pose all frames (cheap) → union framing
    posed = []
    for t in ts:
        p = ch.pose(f(t))
        G, X = PZ.pose_body(ch.body, p)
        posed.append((G, X))
    allX = np.vstack([X[:13380:25] for _, X in posed])
    cam = ch.camera_for(allX, yaw, W, H, pitch=pitch, fov=fov, margin=margin, frame=allX)
    fig_px = ch._fig_px(cam, posed[0][1])
    base_target = np.asarray(cam.target)
    seed = abs(hash(str(ch.body.params))) % 10007
    for i, (t, (G, X)) in enumerate(zip(ts, posed)):
        if yaw_end is not None:
            cam.yaw = yaw + (yaw_end - yaw) * smoother(t / max(1e-6, seconds))
        if follow:
            c = X[:13380].mean(0)
            cam.target = (c[0], base_target[1], c[2])
        parts = ch.parts(G, X)
        from .lineart import render_lines
        lf = render_lines(parts, cam, ch.style, extra=ch.face_strokes(G, X), ss=ss, fig_px=fig_px, seed=seed)
        rgba = draw_strokes(lf.strokes, W, H, Style(**{**ch.style.__dict__, "bg": None}))
        if svg_dir:
            with open(os.path.join(svg_dir, f"f{i:05d}.svg"), "w") as fh:
                fh.write(strokes_svg(lf.strokes, W, H, ch.style))
        yield i, rgba


def render_clip(character, motion, out, seconds=3.0, fps=24, yaw=30.0, W=720, H=1280, bg=None, **kw):
    """character: preset name, dict for Character(**dict) or a Character. out: .mov (ProRes 4444 alpha),
    .webm (VP9 alpha), .mp4 (needs bg, default white), or a directory (PNG sequence)."""
    import cv2
    ch = character if isinstance(character, Character) else \
        Character(**(preset(character) if isinstance(character, str) else character))
    ext = os.path.splitext(out)[1].lower()
    png_dir = out if ext == "" else tempfile.mkdtemp(prefix="hclip_")
    os.makedirs(png_dir, exist_ok=True)
    for i, rgba in render_frames(ch, motion, seconds, fps, yaw, W, H, **kw):
        img = rgba
        if bg is not None or ext == ".mp4":
            b = np.array(bg or (1, 1, 1), np.float32) * 255
            a = img[..., 3:4].astype(np.float32) / 255
            rgb = img[..., :3].astype(np.float32) + b * (1 - a)      # cairo output is premultiplied
            img = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), np.full(a.shape[:2], 255, np.uint8)])
        else:
            a = img[..., 3:4].astype(np.float32) / 255           # un-premultiply for straight-alpha PNG
            rgb = np.where(a > 0, img[..., :3] / np.maximum(a, 1e-6), 0)
            img = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), img[..., 3]])
        cv2.imwrite(os.path.join(png_dir, f"f{i:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    if ext == "":
        return out
    pat = os.path.join(png_dir, "f%05d.png")
    if ext == ".mov":
        args = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
    elif ext == ".webm":
        args = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "28", "-auto-alt-ref", "0"]
    else:
        args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat, *args, out], check=True)
    shutil.rmtree(png_dir, ignore_errors=True)
    return out
