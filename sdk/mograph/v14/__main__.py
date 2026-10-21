"""
python3 -m mograph.v14 <command> spec.json [...]

  prep    spec.json                 offline pose + MODNet matte cache for the spec's source (run once)
  frame   spec.json --t 3.2 [...]   render exact frame(s) → out/<name>_t3.20.png   (< 2 s each)
  sheet   spec.json [--n 12]        contact sheet across the edit (every beat at a glance)
  words   spec.json                 print the edited-time transcript (for "w:" anchors)
  qa      spec.json                 resolve every anchor, check layers are on-screen, list timings
  render  spec.json [--preview]     final video (+ audio) → out/<name>.mp4 ; --preview = half res
  auto-edit words.json raw.mp4 -o edit_auto.json    v14.8 AUTOMATIC director: breath cuts + hero words + depth
                                                   rotation + camera/grade → a QA-clean spec (director.py)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("auto-edit", "auto_edit", "director"):
        from .director import main as director_main
        sys.exit(director_main(argv[1:]))
    ap = argparse.ArgumentParser(prog="mograph.v14")
    ap.add_argument("cmd")
    ap.add_argument("spec")
    ap.add_argument("--t", type=float, nargs="*", default=[])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args(argv)
    from .compositor import Renderer
    base = os.path.dirname(os.path.abspath(a.spec))
    name = os.path.splitext(os.path.basename(a.spec))[0]
    od = os.path.join(base, "out")
    os.makedirs(od, exist_ok=True)
    if a.cmd == "prep":
        import json
        from .track import prep_matte, prep_pose
        s = json.load(open(a.spec))
        src = os.path.join(base, s["source"]["src"])
        prep_pose(src)
        prep_matte(src)
        return
    R = Renderer(a.spec)
    if a.cmd == "words":
        for w in R.tl.words:
            print(f"{w['t']:6.2f}  {w['end']:6.2f}  {w['w']}")
        print(f"duration {R.duration:.2f}s  cuts {['%.2f' % c for c in R.tl.cuts]}")
        return
    if a.cmd == "qa":
        from .qa import run_qa
        sys.exit(run_qa(R))
    if a.cmd == "frame":
        for t in a.t or [R.duration / 2]:
            t0 = time.time()
            img, _ = R.render(t)
            p = os.path.join(od, f"{name}_t{t:05.2f}.png")
            cv2.imwrite(p, img)
            print(p, f"{time.time() - t0:.2f}s")
        return
    if a.cmd == "sheet":
        ts = a.t or list(np.linspace(0.3, R.duration - 0.3, a.n))
        ims = []
        for t in ts:
            img, _ = R.render(t)
            im = cv2.resize(img, (270, 480), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{t:.2f}", (8, 24), 0, 0.7, (0, 255, 255), 2)
            ims.append(im)
        cols = 6
        while len(ims) % cols:
            ims.append(np.zeros_like(ims[0]))
        rows = [np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]
        p = a.out or os.path.join(od, f"{name}_sheet.jpg")
        cv2.imwrite(p, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(p)
        return
    if a.cmd == "render":
        out = a.out or os.path.join(od, f"{name}{'_preview' if a.preview else ''}.mp4")
        silent = os.path.splitext(out)[0] + "_silent.mp4"
        R.render_video(silent, a.start, a.end, 0.5 if a.preview else 1.0)
        if a.no_audio:
            os.replace(silent, out)
            print(out)
            return
        from .audio import build_audio
        wav = build_audio(R, os.path.splitext(out)[0] + ".wav", a.start, a.end)
        os.system(f'ffmpeg -y -v error -i "{silent}" -i "{wav}" -c:v copy -c:a aac -b:a 192k -shortest "{out}"')
        os.remove(silent)
        print(out)


if __name__ == "__main__":
    main()
