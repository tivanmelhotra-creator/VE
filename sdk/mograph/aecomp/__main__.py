"""
python3 -m mograph.aecomp <command>      — v17 AE-Comp: After-Effects shape-layer engine for agents

  list                                   generators + lower-third styles (name · category · what it does)
  sheet  [names…] [--cat C] [-o out.jpg] CONTACT SHEET: each element × 6 instants with its key (LOOK, then choose)
  lt-sheet [--title T --role R]          contact sheet of every lower-third style with your own text
  title-sheet [--title HERO --role KICKER]  contact sheet of every full-frame title style
  gif    <name> [--seed N] [-o out.gif]  animated preview (generator or lower-third style)
  mp4    <name|comp.json> [-o out.mp4] [--w 1080 --h 1080 --bg #000|alpha]   render a clip (alpha → .mov ProRes 4444)
  info   <name> [--seed N]               the comp JSON (copy it, edit it, pass it as "comp")
  frame  <comp.json> --t 0.5 [-o out.png]
  selftest                               numeric regression tests (< 10 s)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from . import generators as GEN
from . import lowerthirds as LT
from . import titles as TT
from .render import render_bgra, render_rgb


def _comp_for(name, seed=0, title="دکتر سارا احمدی", role="متخصص ارتودنسی"):
    if name.endswith(".json") and os.path.exists(name):
        return json.load(open(name))
    if name in LT.STYLES:
        return LT.make(name, title, role)
    if name in TT.STYLES:
        return TT.make(name, "مقدمه را آخر بنویس", "نکته نگارش")
    return GEN.make(name, seed=seed)


def _label(img, s, y=16):
    cv2.putText(img, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(img, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 160), 1)


def strip(comp, n=6, cell=200, label=""):
    d = float(comp.get("dur", 1.0))
    cells = []
    for t in np.linspace(0.06 * d, 0.94 * d, n):
        c = render_rgb(comp, float(t), cell, cell)[..., ::-1].copy()
        cv2.putText(c, f"{t:.2f}s", (6, cell - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
        cells.append(c)
    row = np.concatenate(cells, 1)
    _label(row, label)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mograph.aecomp")
    ap.add_argument("cmd")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--cat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="دکتر سارا احمدی")
    ap.add_argument("--role", default="متخصص ارتودنسی و زیبایی")
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--w", type=int, default=1080)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--bg", default="#111216")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        print("GENERATORS (\"gen\": name)")
        for k, (_, cat, doc) in GEN.GENERATORS.items():
            print(f"  {k:14s} {cat:11s} {doc}")
        print("TITLES (\"title_style\": style, \"hero\", \"kicker\") — full-frame, uses the 351 AE text animators")
        for k, fn in TT.STYLES.items():
            print(f"  {k:14s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        print("LOWER THIRDS (\"lower_third\": style) — box-free, RTL, Doran")
        for k, fn in LT.STYLES.items():
            print(f"  {k:14s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return
    if a.cmd == "sheet":
        names = a.names or GEN.names(a.cat)
        rows = [strip(GEN.make(n, seed=a.seed), label=f"{n}  [{GEN.GENERATORS[n][1]}]") for n in names]
        out = a.out or "/tmp/aecomp_sheet.jpg"
        cv2.imwrite(out, np.concatenate(rows, 0))
        print(out)
        return
    if a.cmd == "lt-sheet":
        rows = []
        for st in LT.styles():
            c = LT.make(st, a.title, a.role)
            cells = []
            for t in (0.25, 0.5, 0.8, 1.4, 3.0, 4.35, 4.65):
                im = render_rgb(c, t, 1080, 1920)[1280:1720, :, ::-1]
                cells.append(cv2.resize(np.ascontiguousarray(im), (324, 132)))
            row = np.concatenate(cells, 1)
            _label(row, st)
            rows.append(row)
        out = a.out or "/tmp/aecomp_lower_thirds.jpg"
        cv2.imwrite(out, np.concatenate(rows, 0))
        print(out)
        return
    if a.cmd == "title-sheet":
        rows = []
        for st in TT.styles():
            c = TT.make(st, a.title if a.title != "دکتر سارا احمدی" else "مقدمه را آخر بنویس", a.role if a.role != "متخصص ارتودنسی و زیبایی" else "نکته نگارش")
            cells = [cv2.resize(np.ascontiguousarray(render_rgb(c, t, 540, 960)[250:600, :, ::-1]), (270, 175))
                     for t in np.linspace(0.12, c["dur"] - 0.3, 7)]
            row = np.concatenate(cells, 1)
            _label(row, st)
            rows.append(row)
        out = a.out or "/tmp/aecomp_titles.jpg"
        cv2.imwrite(out, np.concatenate(rows, 0))
        print(out)
        return
    if a.cmd == "info":
        print(json.dumps(_comp_for(a.names[0], a.seed, a.title, a.role), ensure_ascii=False, indent=1))
        return
    if a.cmd == "frame":
        c = _comp_for(a.names[0], a.seed, a.title, a.role)
        out = a.out or "/tmp/aecomp_frame.png"
        cv2.imwrite(out, render_rgb(c, a.t, a.w, a.h, bg=a.bg)[..., ::-1])
        print(out)
        return
    if a.cmd in ("gif", "mp4"):
        c = _comp_for(a.names[0], a.seed, a.title, a.role)
        fps = float(c.get("fps", 30))
        n = int(round(float(c["dur"]) * fps))
        alpha = a.bg == "alpha"
        if a.cmd == "gif":
            from PIL import Image
            W = H = 320
            if a.names[0] in LT.STYLES or a.names[0] in TT.STYLES:
                W, H = 270, 480
            fr = [Image.fromarray(render_rgb(c, i / fps, W, H, bg=a.bg)) for i in range(0, n, 2)]
            out = a.out or f"/tmp/{os.path.basename(a.names[0]).replace('.json', '')}.gif"
            fr[0].save(out, save_all=True, append_images=fr[1:], duration=int(2000 / fps), loop=0)
            print(out)
            return
        W, H = a.w, a.h
        if (a.names[0] in LT.STYLES or a.names[0] in TT.STYLES) and a.h == 1080:
            W, H = 1080, 1920
        out = a.out or f"/tmp/{os.path.basename(a.names[0]).replace('.json', '')}.{'mov' if alpha else 'mp4'}"
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra" if alpha else "rgb24",
               "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
        cmd += (["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"] if alpha else
                ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]) + [out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(n):
            f = render_bgra(c, i / fps, W, H) if alpha else render_rgb(c, i / fps, W, H, bg=a.bg)
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
        p.wait()
        print(out)
        return
    if a.cmd == "selftest":
        from .selftest import run
        sys.exit(0 if run() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
