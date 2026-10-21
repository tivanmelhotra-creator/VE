"""
python3 -m mograph.motion <command>      — v18 MOTION KIT (glossary · easing · retime · saber · character · lip sync)

  selftest                                   numeric regression tests (< 10 s) — must say 10/10
  resolve  "<user words>"                    Persian/English request → canonical concepts + params (JSON)
  glossary [--sync]                          print the glossary (--sync: glossary.yaml → glossary.json)
  ease list                                  every easing name (+ measured character)
  ease graph <spec> [<spec>…] [-o g.png]     AE-style VALUE + SPEED graph (bezier handles), e.g. 'cubic-bezier(.2,.8,.2,1)'
  ease measure <spec>                        overshoot / undershoot / peak speed / settle numbers
  retime <in.mp4> <out.mp4> --ops JSON [--speed 1] [--interpolate flow|blend|nearest] [--audio auto|varispeed|stretch|mute]
  retime-plan --ops JSON --dur 10            print the out-time map without rendering
  saber list                                 51 presets (+ capabilities → presets)
  saber sheet [--core shape|text|saber] [--text T] [--presets a,b] [-o s.jpg]   CONTACT SHEET of presets (LOOK, then choose)
  saber frame <spec.json> --t 1.0 [-o f.png] [--bg #101014]   render one saber spec (JSON = layer keys)
  saber mp4 <spec.json> [--dur 3] [-o out.mp4] [--w 1080 --h 1920]
  character sheet [--rig bean|blob|star] [--actions JSON] [-o c.jpg]   poses across the action timeline
  character mp4 <spec.json> [-o out.mp4] [--dur 6]
  lipsync track <words.json> [--audio a.wav] [-o visemes.json] [--fps 30]
  lipsync g2v <word …>                       grapheme → viseme list (Persian / English)
  lipsync chart [-o chart.png]               the Preston Blair mouth chart as drawn by the engine
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def _arg_json(v):
    if v is None:
        return None
    if os.path.exists(v):
        return json.load(open(v, encoding="utf-8"))
    return json.loads(v)


def _cairo_rgb(s):
    s.flush()
    return np.ndarray((s.get_height(), s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :s.get_width(), :3].copy()


def _pipe(out, W, H, fps):
    return subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out],
                            stdin=subprocess.PIPE)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "selftest":
        from .selftest import main as m
        return m()
    if cmd == "resolve":
        from .glossary import resolve
        print(json.dumps(resolve(" ".join(rest)), ensure_ascii=False, indent=1))
        return 0
    if cmd == "glossary":
        from . import glossary as G
        if "--sync" in rest:
            print(G.sync_json())
        else:
            print(json.dumps(G.load(), ensure_ascii=False, indent=1))
        return 0
    if cmd == "ease":
        from . import easing as E
        sub = rest[0] if rest else "list"
        ap = argparse.ArgumentParser()
        ap.add_argument("specs", nargs="*")
        ap.add_argument("-o", default="ease_graph.png")
        a = ap.parse_args(rest[1:])
        specs = [_spec(x) for x in a.specs]
        if sub == "list":
            for n in E.names():
                print(f"{n:<22} {E.character(n)}")
        elif sub == "graph":
            E.graph_image(specs or ["ease_out", "out_back"], a.o, labels=a.specs or None)
            print(a.o)
        elif sub == "measure":
            for s in specs:
                print(json.dumps(dict(E.measure(s), spec=str(s), character=E.character(s)), ensure_ascii=False))
        return 0
    if cmd in ("retime", "retime-plan"):
        from . import retime as RT
        ap = argparse.ArgumentParser()
        ap.add_argument("files", nargs="*")
        ap.add_argument("--ops", required=True)
        ap.add_argument("--speed", type=float, default=1.0)
        ap.add_argument("--interpolate", default="flow")
        ap.add_argument("--audio", default="auto")
        ap.add_argument("--dur", type=float, default=10.0)
        a = ap.parse_args(rest)
        ops = _arg_json(a.ops)
        if cmd == "retime-plan":
            print(json.dumps(RT.Retime(ops, a.dur, speed=a.speed).summary(), indent=1, default=float))
            return 0
        print(json.dumps(RT.retime_file(a.files[0], a.files[1], ops, a.speed, a.interpolate, a.audio), default=float))
        return 0
    if cmd == "saber":
        return _saber(rest)
    if cmd == "character":
        return _character(rest)
    if cmd == "lipsync":
        return _lipsync(rest)
    print(__doc__)
    return 1


def _spec(x):
    x = x.strip()
    if x[:1] in "[{":
        return json.loads(x)
    return x


def _saber(rest):
    import cv2
    from . import saber as SB
    sub = rest[0] if rest else "list"
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--w", type=int, default=1080)
    ap.add_argument("--h", type=int, default=1920)
    ap.add_argument("--bg", default="#101014")
    ap.add_argument("--core", default="shape")
    ap.add_argument("--text", default="سابر")
    ap.add_argument("--presets", default=None)
    ap.add_argument("--cell", type=int, default=240)
    ap.add_argument("-o", default=None)
    a = ap.parse_args(rest[1:])
    bgc = SB._hex(a.bg)[::-1] * 255
    if sub == "list":
        for n in SB.preset_names():
            p = SB.preset(n)
            caps = [c for c, ps in SB.GLOSSARY_CAPS.items() if n in ps]
            print(f"{n:<14} {p['color']}  dist={p['distortion']:<6} flicker={p['flicker_intensity']:.2f}"
                  f"{'  lightning' if p.get('lightning') else ''}  {SB.PRESET_TAGS.get(n, '')}  {' '.join(caps)}")
        return 0
    if sub == "sheet":
        names = a.presets.split(",") if a.presets else SB.preset_names()
        C = a.cell
        cells = []
        for n in names:
            sp = {"preset": n, "core": a.core, "seed": 3}
            if a.core == "shape":
                sp.update(shape="circle", w=C * 0.55, x=C / 2, y=C / 2, core_size=0.6)
            elif a.core == "text":
                sp.update(text=a.text, size=C * 0.28, x=C / 2, y=C / 2, core_size=0.5)
            else:
                sp.update(start=[C * 0.2, C * 0.8], end=[C * 0.8, C * 0.2], core_size=0.7)
            sp["scale"] = 0.5
            bg = np.empty((C, C, 3), np.uint8)
            bg[:] = bgc.astype(np.uint8)
            img = SB.render_over(bg, sp, a.t)
            cv2.putText(img, n, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
            cells.append(img)
        cols = 8
        while len(cells) % cols:
            cells.append(np.zeros_like(cells[0]))
        sheet = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
        out = a.o or "saber_sheet.jpg"
        cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(out)
        return 0
    spec = _arg_json(a.spec)
    bg = np.empty((a.h, a.w, 3), np.uint8)
    bg[:] = bgc.astype(np.uint8)
    if sub == "frame":
        out = a.o or "saber_frame.png"
        cv2.imwrite(out, SB.render_over(bg, spec, a.t))
        print(out)
        return 0
    if sub == "mp4":
        out = a.o or "saber.mp4"
        p = _pipe(out, a.w, a.h, 30)
        for i in range(int(a.dur * 30)):
            p.stdin.write(SB.render_over(bg, spec, i / 30).tobytes())
        p.stdin.close()
        p.wait()
        print(out)
        return 0
    return 1


def _character(rest):
    import cairo
    import cv2
    from . import character as CH
    sub = rest[0] if rest else "sheet"
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--rig", default="bean")
    ap.add_argument("--actions", default=None)
    ap.add_argument("--dur", type=float, default=6.0)
    ap.add_argument("-o", default=None)
    a = ap.parse_args(rest[1:])
    spec = _arg_json(a.spec) if a.spec else {"rig": a.rig, "color": "#ffcc33", "actions": _arg_json(a.actions) or [
        {"do": "enter", "t": 0}, {"do": "hop", "t": 0.9}, {"do": "wave", "t": 2.0}, {"do": "talk", "t": 3.5, "dur": 1.4},
        {"do": "surprise", "t": 5.0}, {"do": "point", "t": 6.0}, {"do": "celebrate", "t": 7.2}]}
    C = CH.prepare(spec, None, os.path.dirname(os.path.abspath(a.spec)) if a.spec else ".")
    if sub == "sheet":
        acts = C["_acts"]
        ts = []
        for x in acts:
            ts += [x["_t"] + x["_d"] * f for f in (0.15, 0.5)]
        cells = []
        for t in ts[:16]:
            s = cairo.ImageSurface(cairo.FORMAT_RGB24, 260, 360)
            c = cairo.Context(s)
            c.set_source_rgb(0.93, 0.95, 1)
            c.paint()
            CH.draw(c, C, t, 130, 330, 180)
            img = _cairo_rgb(s)
            cv2.putText(img, f"{t:.2f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
            cells.append(img)
        cols = 8
        while len(cells) % cols:
            cells.append(np.full_like(cells[0], 240))
        out = a.o or "character_sheet.jpg"
        cv2.imwrite(out, np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]))
        print(out, json.dumps(CH.principles_report(C)))
        return 0
    if sub == "mp4":
        W, H = int(spec.get("w", 1080)), int(spec.get("h", 1080))
        out = a.o or "character.mp4"
        p = _pipe(out, W, H, 30)
        for i in range(int(a.dur * 30)):
            s = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            c = cairo.Context(s)
            c.set_source_rgb(0.93, 0.95, 1)
            c.paint()
            CH.draw(c, C, i / 30, float(spec.get("x", W / 2)), float(spec.get("y", H * 0.85)), float(spec.get("size", H * 0.4)))
            p.stdin.write(np.ascontiguousarray(_cairo_rgb(s)).tobytes())
        p.stdin.close()
        p.wait()
        print(out)
        return 0
    return 1


def _lipsync(rest):
    from . import lipsync as LS
    sub = rest[0] if rest else "chart"
    if sub == "g2v":
        for w in rest[1:]:
            print(w, " ".join(LS.g2v(w)))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="?")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("-o", default=None)
    a = ap.parse_args(rest[1:])
    if sub == "chart":
        out = a.o or "mouth_chart.png"
        LS.chart_image(out)
        print(out)
        return 0
    if sub == "track":
        k = LS.track(a.words, a.audio, a.fps)
        s = json.dumps(k, ensure_ascii=False)
        if a.o:
            open(a.o, "w", encoding="utf-8").write(s)
            print(a.o, f"{len(k)} keys")
        else:
            print(s)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
