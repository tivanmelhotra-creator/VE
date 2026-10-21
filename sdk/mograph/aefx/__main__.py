"""
python3 -m mograph.aefx <command>

  list   [motion|text] [--cat C] [--sub S] [--dir in|out|fx]   catalogue (key, category, name, duration)
  find   <words...>                                            fuzzy search over both libraries
  info   <key>                                                 full JSON of one preset
  sheet  motion|text [--cat C] [--sub S] [--dir D] [--text "..."] [-o out.png]
                                                               CONTACT SHEET: every matching preset × 6 instants
                                                               (an agent LOOKS at this to choose a preset)
  gif    <key> [-o out.gif] [--text "..."]                     animated preview of one preset
  suggest <intent words…> [--dir in|out] [--n 8]               ranked presets for an INTENT, from measured data:
                                                               soft calm elegant · punchy energetic hype · spin ·
                                                               3d · blur dreamy · glitch tech · shake impact ·
                                                               slide travel · bounce overshoot · decode type …
  glow   in.png -o out.png [--radius 90 --exposure 0 --threshold 0.6 --tint #hex --aberration 0 --aspect 1]
  selftest                                                     numeric regression tests (< 10 s)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np


def _demo_plate(W=480, H=270):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (38, 28, 22)
    for x in range(0, W, 40):
        cv2.line(img, (x, 0), (x, H), (55, 45, 40), 1)
    cv2.rectangle(img, (W // 2 - 95, H // 2 - 55), (W // 2 + 95, H // 2 + 55), (60, 160, 255), -1)
    cv2.putText(img, "AE-FX", (W // 2 - 78, H // 2 + 16), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 3)
    return img


def _label(img, s):
    cv2.putText(img, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 120), 1)


def motion_strip(key, n=6, plate=None):
    from . import motion as AM
    p = AM.get(key)
    plate = _demo_plate() if plate is None else plate
    cells = []
    for t in np.linspace(0, p["dur"], n):
        c = AM.apply_preset(plate, key, t, motion_blur=180)
        cv2.putText(c, f"{t:.2f}s", (6, c.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cells.append(c)
    row = np.concatenate(cells, 1)
    _label(row, f"{key}   [{p['cat']} / {p['dir']}]  {p['dur']:.2f}s")
    return row


def text_strip(key, n=6, text="آموزش *ویدیو*", cell=(300, 150)):
    import cairo
    from ..v14 import glyphs as G
    from . import textanim as TA
    b = G.typeset(text, 74)
    an = TA.Animator(key)
    W, H = cell[0] * n, cell[1]
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c = cairo.Context(s)
    c.set_source_rgb(0.09, 0.09, 0.11)
    c.paint()
    for j, t in enumerate(np.linspace(0, an.dur * 1.02, n)):
        st = an.frame(b, t)
        c.save()
        c.rectangle(j * cell[0], 0, cell[0], H)
        c.clip()
        G.draw_block(c, b, j * cell[0] + cell[0] / 2, H * 0.58, motion=lambda g, _s=st: _s.apply(g, G.GS()), scale=0.5)
        c.restore()
    s.flush()
    a = np.ndarray((H, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :W, :3].copy()
    p = TA.get(key)
    _label(a, f"{key}  [{p['cat']}/{p['sub']}] {p['name']}")
    return a


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="mograph.aefx", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--cat")
    ap.add_argument("--sub")
    ap.add_argument("--dir")
    ap.add_argument("--text", default="آموزش *ویدیو*")
    ap.add_argument("-o", "--out")
    ap.add_argument("--max", type=int, default=40)
    for k, d in (("radius", 90.0), ("exposure", 0.0), ("threshold", 0.6), ("aberration", 0.0), ("aspect", 1.0)):
        ap.add_argument("--" + k, type=float, default=d)
    ap.add_argument("--tint")
    a = ap.parse_args(argv)
    from . import motion as AM
    from . import textanim as TA

    if a.cmd == "selftest":
        from .selftest import main as st
        sys.exit(st())
    if a.cmd == "list":
        which = a.args[0] if a.args else "motion"
        if which == "motion":
            for k, v in AM.library().items():
                if (not a.cat or v["cat"].lower() == a.cat.lower()) and (not a.dir or v["dir"] == a.dir):
                    print(f"{k:34s} {v['cat']:20s} {v['dir']:4s} {v['dur']:5.2f}s  fx={'+'.join(f['fx'] for f in v['fx'])}")
        else:
            for k, v in TA.library().items():
                if (not a.cat or v["cat"].lower() == a.cat.lower()) and (not a.sub or v["sub"].lower() == a.sub.lower()):
                    props = sorted({p for an in v["animators"] for p in an["props"]})
                    print(f"{k:8s} {v['cat']:14s} {v['sub']:13s} {v['name']:32s} {'+'.join(props)}")
        return
    if a.cmd == "find":
        q = " ".join(a.args).lower().split()
        for k, v in AM.library().items():
            s = f"{k} {v['cat']} {v['name']} {v['dir']}".lower()
            if all(w in s for w in q):
                print("motion ", k, "|", v["cat"], v["dir"], f"{v['dur']:.2f}s")
        for k, v in TA.library().items():
            s = f"{k} {v['cat']} {v['sub']} {v['name']}".lower()
            if all(w in s for w in q):
                print("text   ", k, "|", v["cat"], "/", v["sub"], "/", v["name"])
        return
    if a.cmd == "suggest":
        from .suggest import suggest
        for kind, k, sc, why in suggest(" ".join(a.args), dir_=a.dir, n=a.max if a.max != 40 else 8):
            print(f"{kind:6s} {k:30s} score {sc:5.1f}  {why}")
        return
    if a.cmd == "info":
        k = a.args[0]
        try:
            print(json.dumps(AM.get(k), ensure_ascii=False, indent=1))
        except KeyError:
            print(json.dumps(TA.get(k), ensure_ascii=False, indent=1))
        return
    if a.cmd == "sheet":
        which = a.args[0] if a.args else "motion"
        if which == "motion":
            keys = [k for k, v in AM.library().items() if (not a.cat or v["cat"].lower() == a.cat.lower())
                    and (not a.dir or v["dir"] == a.dir)]
            rows = [motion_strip(k) for k in keys[:a.max]]
        else:
            keys = [k for k, v in TA.library().items() if (not a.cat or v["cat"].lower() == a.cat.lower())
                    and (not a.sub or v["sub"].lower() == a.sub.lower())] if not a.args[1:] else a.args[1:]
            rows = [text_strip(k, text=a.text) for k in keys[:a.max]]
        sheet = np.concatenate(rows, 0)
        if sheet.shape[1] > 1800:
            sheet = cv2.resize(sheet, None, fx=1800 / sheet.shape[1], fy=1800 / sheet.shape[1],
                               interpolation=cv2.INTER_AREA)
        out = a.out or f"aefx_{which}_{(a.cat or 'all').replace(' ', '_')}.jpg"
        cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{len(rows)} presets → {out}")
        return
    if a.cmd == "gif":
        from PIL import Image
        k = a.args[0]
        frames = []
        try:
            p = AM.get(k)
            plate = _demo_plate()
            for i in range(int(p["dur"] * 25) + 12):
                frames.append(AM.apply_preset(plate, k, i / 25, 25, motion_blur=180)[:, :, ::-1])
        except KeyError:
            import cairo
            from ..v14 import glyphs as G
            b = G.typeset(a.text, 90)
            an = TA.Animator(k)
            for i in range(int(an.dur * 25) + 12):
                s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 480, 220)
                c = cairo.Context(s)
                c.set_source_rgb(0.09, 0.09, 0.11)
                c.paint()
                st = an.frame(b, i / 25)
                G.draw_block(c, b, 240, 128, motion=lambda g, _s=st: _s.apply(g, G.GS()), scale=0.8)
                s.flush()
                frames.append(np.ndarray((220, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :480, 2::-1].copy())
        out = a.out or f"{k}.gif"
        Image.fromarray(frames[0]).save(out, save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]],
                                        duration=40, loop=0)
        print(out)
        return
    if a.cmd == "glow":
        from .deepglow import deep_glow
        img = cv2.imread(a.args[0], cv2.IMREAD_UNCHANGED)
        o = deep_glow(img, radius=a.radius, exposure=a.exposure, threshold=a.threshold, tint=a.tint,
                      aberration=a.aberration, aspect=a.aspect)
        cv2.imwrite(a.out or "glow.png", o)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
