"""
birthday_mocap — the storyboard ("little sister grows up, big brother is always there") as a line-art film where
EVERY body movement is real motion capture (CMU, Quaternius CC0, Mixamo samples) retargeted onto MakeHuman
characters and drawn by mograph.human.lineart. Only head look-at, a lean and one reach (arm around the shoulder)
are added on top of the captures.

    cd sdk && python3 examples/mocap/birthday_mocap.py OUT [--sheet [t1,t2,..]] [--jobs 2]

  shot 1  0.0- 6.4  childhood — she kneels over her toy box; he walks in, stops, watches; she looks up at him
  shot 2  6.0-13.0  years later — she sits; he walks over and sits down next to her (ONE real 2-person capture)
  shot 3 12.6-22.0  together — his arm around her shoulder, she leans in; candle; "Happy Birthday my little sister"
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

from mograph.human.body import BODY_N  # noqa: E402
from mograph.human.character import Character, preset  # noqa: E402
from mograph.human.lineart import Camera, Style, ribbon, render_lines  # noqa: E402
from mograph.mocap import props as PR  # noqa: E402
from mograph.mocap.director import Actor, Scene, smooth  # noqa: E402
from mograph.mocap.library import resolve  # noqa: E402
from mograph.mocap.retarget import MocapMotion  # noqa: E402

W, H, FPS, T_END = 1080, 1350, 24, 22.0
INK = (0.94, 0.93, 0.90)
BG = (0.035, 0.035, 0.04)
STYLE = Style(color=INK, width=3.0, inner=0.5, pressure=0.0, taper=0.2, min_len=7.0, smooth=2)


def lerp(a, b, u):
    return a + (b - a) * u


def keyed(t, keys):
    if t <= keys[0][0]:
        return dict(keys[0][1])
    for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            u = smooth((t - t0) / max(1e-6, t1 - t0))
            return {k: (tuple(lerp(np.asarray(a[k], float), np.asarray(b[k], float), u))
                        if isinstance(a[k], (tuple, list)) else lerp(a[k], b[k], u)) for k in a}
    return dict(keys[-1][1])


def make_cam(k):
    c = Camera(target=tuple(k["target"]), yaw=k["yaw"], pitch=k["pitch"], fov=k["fov"], W=W, H=H)
    c.dist = k["dist"]
    return c


def cast():
    girl = Character(**preset("girl", outfit=dict(dress="knee", sleeve=0.3, shoes="sneakers", hair="ponytail")),
                     face="minimal", style=STYLE)
    boy = Character(body=dict(gender=1, age=10, muscle=0.5, weight=0.45),
                    outfit=dict(top="longsleeve", bottom="trousers", shoes="sneakers", hair="side"),
                    face="minimal", style=STYLE)
    man = Character(body=dict(gender=1, age=26, muscle=0.55, weight=0.5, height=0.62),
                    outfit=dict(top="jacket", bottom="trousers", shoes="shoes", hair="side"), face="minimal",
                    style=STYLE)
    woman = Character(body=dict(gender=0, age=22, muscle=0.45, weight=0.42, height=0.42),
                      outfit=dict(top="jacket", bottom="jeans", shoes="sneakers", hair="bun"), face="minimal",
                      style=STYLE)
    return girl, boy, man, woman


class Film:
    def __init__(self):
        girl, boy, man, woman = cast()
        # ------------------------------------------------ shot 1: childhood
        g = Actor("girl", girl, [{"clip": "kneel_work", "t": 0.0, "at": [1.6, 0.4], "heading": -28,
                                  "in": 0.75, "out": 4.3, "speed": 0.6, "loop": False}],
                  look=[{"t": 3.9, "dur": 0.9, "target": "boy:head", "w": 0.95, "tilt": 6}])
        b = Actor("boy", boy, [{"clip": "slow_walk_stop", "t": 0.5, "in": 0.15, "loop": False, "heading": 72,
                                "end_at": [-4.6, -2.2]},
                               {"clip": "idle_breath", "t": 2.85, "fade": 0.8}],
                  look=[{"t": 2.4, "dur": 1.0, "target": "girl:head", "w": 0.9}],
                  lean=[{"t": 3.6, "dur": 1.2, "pitch": 6, "side": -3}])
        self.s1 = Scene([g, b])
        hs = []
        for t in np.linspace(0.8, 5.5, 10):
            G = g.base_matrices(t)
            hs.append((g.joint(G, "wrist.L") + g.joint(G, "wrist.R")) / 2)
        hc = np.mean(hs, 0)
        self.box, bs, P = PR.box((hc[0], 0.0, hc[2] + 0.4), (3.4, 1.25, 2.3), 90, yaw=-28)
        self.box_strokes = bs + PR.toys(P)
        # ------------------------------------------------ shot 2: one real two-person capture
        T2 = 6.0
        ps = MocapMotion(resolve("cmu:18_12")[0], body=man.body).scale
        common = {"t": T2, "anchor": "world", "rotate": 90, "at": [0.0, 0.0], "pos_scale": ps, "in": 0.2,
                  "loop": False}
        w2 = Actor("woman", woman, [dict(common, clip="cmu:18_12")],
                   look=[{"t": T2 + 3.6, "dur": 1.0, "target": "man:head", "w": 0.8}])
        m2 = Actor("man", man, [dict(common, clip="cmu:19_12")],
                   look=[{"t": T2 + 3.8, "dur": 1.0, "target": "woman:head", "w": 0.8}])
        self.s2 = Scene([w2, m2])
        seat = []
        for t in np.linspace(T2 + 1.0, T2 + 6.5, 8):
            Gw, Xw = w2.posed(t, self.s2)
            r = w2.joint(Gw, "root")
            B = Xw[:BODY_N]
            m = (np.abs(B[:, 0] - r[0]) < 1.6) & (np.abs(B[:, 2] - r[2]) < 1.6) & (B[:, 1] < r[1])
            if m.any():
                seat.append(B[m, 1].min())
        self.seat_h = float(np.median(seat)) - 0.05
        cw = w2.joint(w2.base_matrices(T2 + 5.0), "root")
        cm = m2.joint(m2.base_matrices(T2 + 6.5), "root")
        self.s2_center = (cw + cm) / 2
        lo, hi = min(cw[0], cm[0]) - 3.0, max(cw[0], cm[0]) + 3.0
        self.bench, self.bench_strokes = PR.bench(lo, hi, self.s2_center[2] - 0.4, self.seat_h)
        # ------------------------------------------------ shot 3: together
        T3 = 12.6
        m3 = Actor("man", man, [{"clip": "idle_breath", "t": T3, "at": [-2.55, 0.0], "heading": 16, "in": 0.3}],
                   reach=[{"t": T3 + 1.5, "dur": 1.5, "arm": "L", "target": "woman:shoulder.L",
                           "offset": [0.15, 0.55, -0.25], "palm": True, "pole": [0, -0.3, -1.0],
                           "arc": [0, 0.9, -0.6]}],
                   look=[{"t": T3 + 2.4, "dur": 1.1, "target": "woman:head", "w": 0.85, "tilt": 4}])
        w3 = Actor("woman", woman, [{"clip": "talk", "t": T3, "at": [2.05, 0.5], "heading": -16, "in": 0.4}],
                   look=[{"t": T3 + 3.2, "dur": 1.1, "target": "man:head", "w": 0.75, "tilt": -9}],
                   lean=[{"t": T3 + 3.0, "dur": 1.3, "side": 6, "pitch": -2}])
        self.s3 = Scene([m3, w3])
        self.candle = np.array([7.0, 0.0, 3.6])

    def camera(self, shot, t):
        if shot == 1:
            return make_cam(keyed(t, [(0.0, dict(target=(-0.4, 4.6, 0.0), yaw=6, pitch=7, fov=24, dist=68)),
                                      (3.0, dict(target=(-0.4, 4.8, 0.0), yaw=9, pitch=6, fov=24, dist=62)),
                                      (6.6, dict(target=(0.0, 4.4, 0.2), yaw=14, pitch=5, fov=24, dist=52))]))
        if shot == 2:
            c = self.s2_center
            return make_cam(keyed(t, [(6.0, dict(target=(c[0], 7.0, c[2]), yaw=-10, pitch=5, fov=24, dist=92)),
                                      (9.5, dict(target=(c[0], 7.0, c[2]), yaw=-6, pitch=4, fov=24, dist=84)),
                                      (13.2, dict(target=(c[0], 7.2, c[2]), yaw=0, pitch=3, fov=24, dist=72))]))
        return make_cam(keyed(t, [(12.6, dict(target=(0.8, 8.6, 0.6), yaw=-6, pitch=4, fov=24, dist=90)),
                                  (16.0, dict(target=(0.8, 8.9, 0.6), yaw=-2, pitch=4, fov=24, dist=80)),
                                  (22.0, dict(target=(1.2, 8.5, 0.9), yaw=4, pitch=5, fov=24, dist=88))]))

    def strokes(self, shot, t):
        sc = {1: self.s1, 2: self.s2, 3: self.s3}[shot]
        cam = self.camera(shot, t)
        parts, extra = [], []
        for k, a in enumerate(sc.actors.values()):
            G, X = a.posed(t, sc)
            ps = a.ch.parts(G, X)
            for p in ps:
                p.pid += 20 * k
            parts += ps
            extra += a.ch.face_strokes(G, X) + a.ch.fold_strokes(G, ps)
        if shot == 1:
            parts.append(self.box)
            extra += self.box_strokes + [PR.floor_line(-8.0, -45, 45, 1)]
        elif shot == 2:
            parts.append(self.bench)
            extra += self.bench_strokes + [PR.floor_line(self.s2_center[2] - 10.0, -70, 70, 2)]
        else:
            extra += [PR.floor_line(-7.0, -70, 70, 3)]
        lf = render_lines(parts, cam, STYLE, extra=extra, ss=1.25, fig_px=H * 0.55, seed=11)
        return lf.strokes, cam


def draw(ctx, strokes, alpha=1.0, reveal=1.0):
    """variable-width ink ribbons; reveal < 1 → the pen is still drawing (top → bottom, each from its start)."""
    ctx.set_source_rgba(*INK, alpha)
    for s in strokes:
        pts, w = s.pts, s.w
        if reveal < 1.0:
            u = (reveal * 1.5 - float(pts[:, 1].min()) / H * 0.5) / 0.6
            if u <= 0:
                continue
            m = max(2, int(len(pts) * min(1.0, u)))
            pts, w = pts[:m], w[:m]
        if len(pts) < 2:
            continue
        L, R = ribbon(pts, w)
        ctx.new_path()
        ctx.move_to(*L[0])
        for q in L[1:]:
            ctx.line_to(*q)
        for q in R[::-1]:
            ctx.line_to(*q)
        ctx.close_path()
        ctx.fill()
        for j in (0, -1):
            ctx.arc(pts[j][0], pts[j][1], w[j] / 2, 0, 2 * math.pi)
            ctx.fill()


_VIG = None


def finish(img, rng):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _VIG = np.clip(1 - 0.38 * (((xx - W / 2) / (W * 0.72)) ** 2 + ((yy - H / 2) / (H * 0.72)) ** 2),
                       0.55, 1)[..., None]
    out = img.astype(np.float32) * _VIG + rng.normal(0, 3.0, (H, W, 1)).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def frame(film, t, rng):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(*BG)
    ctx.paint()
    # transitions: the old shot dissolves out completely (0.5 s), the next one is DRAWN ON by the pen (1.1 s)
    if t < 6.2:
        a = 1 - smooth((t - 5.7) / 0.5) if t > 5.7 else 1.0
        st, _ = film.strokes(1, t)
        draw(ctx, st, a, reveal=smooth(min(1.0, t / 1.6)))
    if 6.1 <= t < 13.0:
        a = (1 - smooth((t - 12.5) / 0.5)) if t > 12.5 else 1.0
        st, _ = film.strokes(2, t)
        draw(ctx, st, a, reveal=smooth(min(1.0, (t - 6.1) / 1.1)))
    if t >= 12.9:
        a = (1 - smooth((t - 21.2) / 0.8)) if t > 21.2 else 1.0
        st, cam = film.strokes(3, t)
        draw(ctx, st, a, reveal=smooth(min(1.0, (t - 12.9) / 1.1)))
        ca = smooth((t - 15.2) / 0.8) * a
        if ca > 0:
            PR.candle(ctx, cam, film.candle, t, ca, INK)
        if t > 16.6:
            ctx.push_group()
            PR.script_title(ctx, t, 16.6, W * 0.62, H * 0.12, INK)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(a)
    surf.flush()
    img = np.ndarray((H, W, 4), np.uint8, surf.get_data())[..., [2, 1, 0]].copy()
    return finish(img, rng)


def worker(args):
    out, frames, wid = args
    import cv2
    film = Film()
    rng = np.random.default_rng(100 + wid)
    for i in frames:
        p = os.path.join(out, "frames", f"f{i:05d}.png")
        if not os.path.exists(p):
            cv2.imwrite(p, cv2.cvtColor(frame(film, i / FPS, rng), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return len(frames)


def sheet(out, ts):
    import cv2
    film = Film()
    rng = np.random.default_rng(0)
    tiles = []
    for t in ts:
        t0 = time.time()
        im = cv2.resize(frame(film, t, rng), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        print(f"t={t:5.2f} {time.time() - t0:.2f}s", flush=True)
        cv2.putText(im, f"{t:.1f}", (6, 18), 0, 0.5, (255, 190, 0), 1)
        tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(os.path.join(out, "sheet.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/home/user/render/bm"
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    if "--sheet" in sys.argv:
        i = sys.argv.index("--sheet")
        ts = [float(x) for x in sys.argv[i + 1].split(",")] if len(sys.argv) > i + 1 else \
            [0.8, 2.0, 3.2, 4.6, 5.8, 6.4, 7.5, 9.0, 10.5, 12.0, 13.0, 14.4, 15.6, 17.5, 19.5, 21.0]
        return sheet(out, ts)
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 2
    n = int(T_END * FPS)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(jobs) as pool:
        pool.map(worker, [(out, list(range(k, n, jobs)), k) for k in range(jobs)])
    print(f"frames {time.time() - t0:.0f}s", flush=True)
    from mograph.vecanim import audio
    wav = os.path.join(out, "music.wav")
    audio.music_box_score(wav, T_END, bpm=76, start=1.0)
    mp4 = os.path.join(out, "birthday_mocap.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i",
                    os.path.join(out, "frames", "f%05d.png"), "-i", wav, "-threads", "2", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "20", "-x264-params", "rc-lookahead=10:ref=2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
                    "-movflags", "+faststart", mp4], check=True)
    print(mp4)


if __name__ == "__main__":
    main()
