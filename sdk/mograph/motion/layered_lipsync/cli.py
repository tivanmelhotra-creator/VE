"""
CLI for the layered lip-sync engine:  python3 -m mograph.motion layered-lipsync <sub> …

  detect <img> [-o dbg.jpg]                 head / eye-pair / mouth landmarks (JSON + debug image)
  sheet <pose> [--visemes grid] [-o s.jpg]  all 10 Preston Blair mouths on a pose — LOOK before rendering
  render --bg B --poses P --audio A [--fg F] [--words w.json] [--visemes grid] [-o out.mp4]
  video <in.mp4> [--words w.json] [-o out]  lip-sync a character that moves (per-frame tracking + EMA)
  enhance-hd <img|mp4> [--size WxH] [-o o]  Full-HD ink-black / studio-white remaster
  contact <mp4> [-n 8] [-o sheet.jpg]       contact sheet for visual QC

Exit codes: 0 ok · 1 usage error · 2 rendered but QA reported failures (check the report JSON).
"""
from __future__ import annotations

import argparse
import json
import os

SUBS = ("detect", "sheet", "render", "video", "enhance-hd", "contact")


def _wh(v: str) -> tuple[int, int]:
    w, h = v.lower().split("x")
    return int(w), int(h)


def _grid(v: str) -> tuple[int, int]:
    r, c = v.lower().split("x")
    return int(r), int(c)


def _json_arg(v):
    if v is None:
        return None
    if os.path.exists(v):
        with open(v, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(v)


def main(rest: list[str]) -> int:
    import cv2

    from . import anatomy_summary, contact_sheet, debug_anatomy_image, enhance_lineart_hd, enhance_video_hd
    from . import render, render_video_lipsync, viseme_sheet
    from .media import load_image

    if not rest or rest[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if rest else 1
    sub = rest[0].replace("_", "-")
    if sub not in SUBS:
        print(f"unknown subcommand {rest[0]!r}\n" + __doc__)
        return 1
    ap = argparse.ArgumentParser(prog=f"python3 -m mograph.motion layered-lipsync {sub}")

    if sub == "detect":
        ap.add_argument("image")
        ap.add_argument("-o", default="anatomy_debug.jpg")
        a = ap.parse_args(rest[1:])
        im, _ = load_image(a.image)
        vis, info = debug_anatomy_image(im)
        cv2.imwrite(a.o, vis)
        print(json.dumps(dict(anatomy_summary(info), out=a.o)))
        return 0

    if sub == "sheet":
        ap.add_argument("pose")
        ap.add_argument("--visemes", default=None, help="optional drawn viseme grid image")
        ap.add_argument("--grid", type=_grid, default=(4, 4), help="ROWSxCOLS of the viseme grid")
        ap.add_argument("--no-remaster", action="store_true")
        ap.add_argument("-o", default="viseme_sheet.jpg")
        a = ap.parse_args(rest[1:])
        rep = viseme_sheet(a.pose, a.o, a.visemes, a.grid, remaster=not a.no_remaster)
        print(json.dumps(rep, ensure_ascii=False))
        return 0 if rep["qa_failures"] == 0 else 2

    if sub == "render":
        ap.add_argument("--bg", required=True)
        ap.add_argument("--poses", required=True, help="one image, comma-separated list, or a pose sheet")
        ap.add_argument("--audio", required=True)
        ap.add_argument("--fg", default=None, help="foreground props plate (desk, lamp …)")
        ap.add_argument("--words", default=None, help="words.json → Preston Blair visemes (Persian / English)")
        ap.add_argument("--visemes", default=None, help="drawn viseme grid (default: procedural vector mouths)")
        ap.add_argument("--grid", type=_grid, default=(4, 4))
        ap.add_argument("--viseme-map", default=None, help='JSON or file: {"AI": 5, ...} overrides auto mapping')
        ap.add_argument("--sheet", default=None, help="poses image is a pose sheet: 'auto' or a panel count")
        ap.add_argument("--body", default=None, help="JSON/file body timeline: per-frame list or [[t, pose], ...]")
        ap.add_argument("--mouth", default=None, help="JSON/file per-frame mouth timeline (viseme names / cell ids)")
        ap.add_argument("--ref", default=None, help="reference character image for eye registration")
        ap.add_argument("--align", default="auto", choices=("auto", "eyes", "none"))
        ap.add_argument("--occlusion", default="auto", help="auto | front | behind | <y px>")
        ap.add_argument("--fps", type=float, default=24.0)
        ap.add_argument("--size", type=_wh, default=(1080, 1920), help="WxH (default 1080x1920)")
        ap.add_argument("--no-remaster", action="store_true")
        ap.add_argument("--no-blink", action="store_true")
        ap.add_argument("--seed", type=int, default=7)
        ap.add_argument("--crf", type=int, default=18)
        ap.add_argument("--contact", default=None, help="also write a contact-sheet JPG")
        ap.add_argument("--hd", action="store_true", help="v29.2 flag (Full-HD is now the default)")
        ap.add_argument("-o", default="layered_lipsync.mp4")
        a = ap.parse_args(rest[1:])
        occ = a.occlusion
        occ = int(occ) if occ.lstrip("-").isdigit() else occ
        sheet = None if a.sheet is None else (a.sheet if a.sheet == "auto" else int(a.sheet))
        rep = render(bg=a.bg, poses=a.poses, audio=a.audio, out=a.o, fg=a.fg, words=a.words, visemes=a.visemes,
                     grid=a.grid, viseme_map=_json_arg(a.viseme_map), mouth_timeline=_json_arg(a.mouth),
                     body_timeline=_json_arg(a.body), pose_sheet=sheet, align=a.align, ref=a.ref, fps=a.fps,
                     wh=a.size, remaster=not a.no_remaster, occlusion=occ, blink=not a.no_blink, seed=a.seed,
                     crf=a.crf, contact=a.contact)
        rep.pop("poses", None)
        print(json.dumps(rep, ensure_ascii=False, default=str))
        return 0 if rep["qa_clean"] else 2

    if sub == "video":
        ap.add_argument("video")
        ap.add_argument("--audio", default=None, help="default: the video's own audio track")
        ap.add_argument("--words", default=None)
        ap.add_argument("--smooth", type=float, default=0.6, help="EMA factor for the mouth position, 0..0.95")
        ap.add_argument("--size", type=_wh, default=None)
        ap.add_argument("--remaster", action="store_true")
        ap.add_argument("-o", default="video_lipsync.mp4")
        a = ap.parse_args(rest[1:])
        rep = render_video_lipsync(a.video, a.o, audio=a.audio, words=a.words, smooth=a.smooth,
                                   remaster=a.remaster, wh=a.size)
        print(json.dumps(rep))
        return 0 if rep["qa_failures"] == 0 else 2

    if sub == "enhance-hd":
        ap.add_argument("target")
        ap.add_argument("--size", type=_wh, default=(1080, 1920))
        ap.add_argument("-o", default=None)
        a = ap.parse_args(rest[1:])
        if a.target.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
            print(json.dumps(enhance_video_hd(a.target, a.o or "enhanced_hd.mp4", a.size)))
        else:
            im, _ = load_image(a.target)
            out = a.o or "enhanced_hd.png"
            cv2.imwrite(out, enhance_lineart_hd(im, target_wh=a.size))
            print(json.dumps({"out": out, "resolution": f"{a.size[0]}x{a.size[1]}"}))
        return 0

    ap.add_argument("video")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("-o", default="contact.jpg")
    a = ap.parse_args(rest[1:])
    print(json.dumps({"out": contact_sheet(a.video, a.o, a.n, a.cols)}))
    return 0
