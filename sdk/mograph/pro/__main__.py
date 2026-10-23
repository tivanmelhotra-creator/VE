"""
python3 -m mograph.pro <command>      — v19 PRO EDITOR (colour · B-roll · delivery · lint · parallel render · doctor)

  doctor [--fix] [--no-tests]                 environment + caches + all 5 selftests (run FIRST after a reset)
  selftest                                    regression tests of the pro kit (< 5 s) — must say 12/12
  lint <edit.json>                            spec linter: types, typo'd keys, anchors, files, hero overlaps
  lint-all [root]                             lint every edit_*.json under root (default: projects/)
  looks [--spec edit.json | --src img|mp4] [-o sheet.jpg]   the 10 colour looks on YOUR footage (LOOK first)
  lut-export <look> [-o look.cube]            built-in look → .cube (Premiere / Resolve / FCP / CapCut)
  lut-info <file.cube>                        size + what it does to grey / skin / sky / grass
  timeline <edit.json>                        the "timeline panel": cuts, retime, layers, subs, sfx, camera, grade
  captions <edit.json> [--fmt srt|vtt|ass|all] [-o dir]   on-screen captions in screen time
  edl <edit.json> [-o edit.edl]               CMX3600 EDL of the cut (+ layer markers) for NLE round-trip
  chapters <edit.json>                        YouTube chapters
  thumbnail <edit.json> [--t auto|3.2] [-o thumb.jpg]     best frame, rendered with all layers
  frame <edit.json> --t 3.2 [-o f.jpg]        one composited frame (same renderer as v14 frame)
  platform <edit.json> [--to instagram_reels|youtube_shorts|tiktok|youtube|instagram_feed|linkedin]
  render <edit.json> [-j 2] [--chunk 4] [--preview] [--start a --end b] [--crf 17] [--profile master|web|social]
  reformat <in.mp4> --to 1:1|4:5|16:9|9:16|4:3 [--spec edit.json] [--mode crop|blur] [-o out.mp4]
  pack-cache <video> [-o pack_dir] / unpack-cache <video>   tracking cache ⇄ small committed pack
  style-bank [--sync | --list | --sheet] [-o sheet.jpg]   v21 inspiration bank (sdk/style_bank/references → catalog.json)
  originality <edit.json>                     v21 ANTI-COPY / ANTI-REPETITION audit (exit 2 = copy, 1 = repetition)
  --- v22 (Concat-inspired; models live in ~/.cache/mograph_models, never in git) ---
  stabilize <edit.json>                       analyse the used source ranges now + shake before/after
  enhance <img|mp4> [--factor 2] [--mix 0.6] [-o out] [--start a --end b]   tiled Real-ESRGAN restore
  cutout <img> [--quality fast|high] [-o out.png]          product/object background removal → RGBA
  models [name …]                             v22 model cache status / fetch
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"mograph.pro {cmd}")
    ap.add_argument("args", nargs="*")
    ap.add_argument("-o", "--out")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--src")
    ap.add_argument("--spec")
    ap.add_argument("--fmt", default="all")
    ap.add_argument("--t", default="auto")
    ap.add_argument("--to", default=None)
    ap.add_argument("--mode", default="crop")
    ap.add_argument("-j", "--jobs", type=int, default=2)
    ap.add_argument("--chunk", type=float, default=4.0)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--crf", type=int)
    ap.add_argument("--profile", default="master", choices=["master", "web", "social"])
    ap.add_argument("--x264-preset", dest="x264_preset")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--mix", type=float, default=0.6)
    ap.add_argument("--quality", default="fast")
    a = ap.parse_args(rest)
    A = a.args

    if cmd == "style-bank":
        from .. import style_bank as SB
        if a.sync or not (a.list or a.sheet):
            SB.sync()
        if a.list:
            SB.list_refs()
        if a.sheet:
            SB.sheet(a.out)
        return 0
    if cmd == "originality":
        from ..style_bank import run_originality
        return run_originality(A[0])
    if cmd == "stabilize":
        from ..v14.compositor import Renderer
        from .stabilize import install, report
        R = Renderer(A[0])
        if R.stab is None:
            R.spec["source"]["stabilize"] = True
            install(R)
        return report(R)
    if cmd == "enhance":
        from .enhance import cli as enh
        enh(A[0], a.out, a.factor, a.mix, a.start, a.end)
        return 0
    if cmd == "cutout":
        from .cutout import cutout_file
        print(cutout_file(A[0], a.out, a.quality))
        return 0
    if cmd == "models":
        from ..models_hub import main as mh
        return mh(A or ["--list"])
    if cmd == "selftest":
        from .selftest import main as st
        return st()
    if cmd == "doctor":
        from .doctor import doctor
        return 1 if doctor(a.fix, not a.no_tests) else 0
    if cmd == "lint":
        from .lint import run
        return run(A[0])
    if cmd == "lint-all":
        from .lint import lint_all
        return 1 if lint_all(os.path.abspath(A[0] if A else os.path.join(ROOT, "projects"))) else 0
    if cmd == "looks":
        import cv2
        from .color import looks_sheet
        if a.spec:
            from ..v14.compositor import Renderer
            R = Renderer(a.spec)
            img, _ = R.src_frame(R.tl.to_src(R.duration * 0.4))
        elif a.src:
            img = cv2.imread(a.src)
            if img is None:
                cap = cv2.VideoCapture(a.src)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.4))
                img = cap.read()[1]
        else:
            from .selftest import test_frame
            img = test_frame()
        print(looks_sheet(img, a.out or "/tmp/looks_sheet.jpg"))
        return 0
    if cmd == "lut-export":
        from .color import look_lut, write_cube
        print(write_cube(a.out or f"{A[0]}.cube", look_lut(A[0]), A[0]))
        return 0
    if cmd == "lut-info":
        import numpy as np
        from .color import read_cube, sample, resample
        L3, dmin, dmax = read_cube(A[0])
        print(f"{A[0]}: {L3.shape[0]}³  domain {dmin.tolist()} .. {dmax.tolist()}")
        L3 = resample(L3)
        for nm, c in (("black", [0, 0, 0]), ("grey 18%", [.18, .18, .18]), ("grey 50%", [.5, .5, .5]),
                      ("white", [1, 1, 1]), ("skin", [.85, .62, .50]), ("sky", [.45, .65, .90]), ("grass", [.3, .55, .2])):
            print(f"  {nm:<9} {c} → {np.round(sample(L3, np.array(c, np.float32)), 3).tolist()}")
        return 0
    if cmd in ("timeline", "captions", "edl", "chapters", "thumbnail", "platform", "frame"):
        from ..v14.compositor import Renderer
        from . import export as EX
        R = Renderer(A[0])
        base = os.path.dirname(os.path.abspath(A[0]))
        name = os.path.splitext(os.path.basename(A[0]))[0]
        if cmd == "timeline":
            print(EX.timeline(R))
        elif cmd == "captions":
            od = a.out or os.path.join(base, "out")
            os.makedirs(od, exist_ok=True)
            ph = EX.phrases(R)
            fns = {"srt": EX.write_srt, "vtt": EX.write_vtt, "ass": lambda p, q: EX.write_ass(p, q, R.W, R.H)}
            for f in (("srt", "vtt", "ass") if a.fmt == "all" else (a.fmt,)):
                print(fns[f](ph, os.path.join(od, f"{name}.{f}")))
        elif cmd == "edl":
            os.makedirs(os.path.join(base, "out"), exist_ok=True)
            print(EX.write_edl(R, a.out or os.path.join(base, "out", f"{name}.edl")))
        elif cmd == "chapters":
            print(EX.chapters(R))
        elif cmd in ("thumbnail", "frame"):
            import cv2
            t = EX.best_frame_time(R) if (a.t == "auto" and cmd == "thumbnail") else float(a.t if a.t != "auto" else 1.0)
            img, _ = R.render(t)
            os.makedirs(os.path.join(base, "out"), exist_ok=True)
            p = a.out or os.path.join(base, "out", f"{name}_{'thumb' if cmd == 'thumbnail' else f'f{t:.2f}'}.jpg")
            cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"{p}  (t={t:.2f}s)")
        elif cmd == "platform":
            for pf in ([a.to] if a.to else list(EX.PLATFORMS)):
                iss, P = EX.platform_check(R, pf)
                print(f"{'✓' if not iss else '⚠'} {pf:<16} {P['aspect']:<5} safe: {P['safe']}" +
                      "".join(f"\n     - {x}" for x in iss))
        return 0
    if cmd == "render":
        from .render import render
        render(A[0], a.jobs, a.chunk, a.preview, a.crf, a.start, a.end, profile=a.profile, preset=a.x264_preset)
        return 0
    if cmd == "reformat":
        from .export import reformat
        to = a.to or "1:1"
        out = a.out or os.path.splitext(A[0])[0] + f"_{to.replace(':', 'x')}.mp4"
        print(reformat(A[0], out, to, a.spec, a.mode))
        return 0
    if cmd == "pack-cache":
        from .doctor import pack_cache
        pack_cache(os.path.abspath(A[0]), a.out)
        return 0
    if cmd == "unpack-cache":
        from .doctor import unpack_cache
        return 0 if unpack_cache(os.path.abspath(A[0]), a.out) else 1
    print(f"unknown command '{cmd}'\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
