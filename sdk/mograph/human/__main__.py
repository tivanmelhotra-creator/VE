"""
python3 -m mograph.human <command> …

  setup                                   deps + EGL + MakeHuman CC0 assets (~40 s, after any sandbox reset)
  presets                                 list character presets / poses / motions / outfit options
  still  PRESET [--pose P] [--yaw Y] [--out f.png|f.svg] [--W 900 --H 1600] [--bg white|none]
         [--body k=v,…] [--outfit k=v,…] [--face none|minimal|simple] [--color r,g,b] [--width 3.2]
  clip   PRESET MOTION --out f.mov|f.webm|f.mp4|dir/ [--seconds 3] [--fps 24] [--yaw 70] [--yaw-end 20]
         [--W 1080 --H 1920] [--bg white] [--cycle 1.1]
  sheet  [--out sheet.png]                 contact sheet of every preset (QA)

Examples
  python3 -m mograph.human still office_woman --pose present --yaw 25 --out a.svg
  python3 -m mograph.human still casual_man --outfit top=jacket,hair=side --body weight=0.8 --out b.png
  python3 -m mograph.human clip business_man walk_cycle --seconds 2.2 --yaw 70 --out walk.mov
"""
from __future__ import annotations

import argparse
import sys


def _kv(s):
    out = {}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.split("=", 1)
        try:
            v = float(v) if "." in v or v.lstrip("-").isdigit() else v
        except ValueError:
            pass
        if v in ("true", "True"):
            v = True
        elif v in ("false", "False"):
            v = False
        out[k.strip()] = v
    return out


def _build(a):
    from .character import Character, preset
    from .lineart import Style
    p = preset(a.preset, body=_kv(a.body), outfit=_kv(a.outfit))
    st = Style(color=tuple(float(x) for x in a.color.split(",")), width=a.width)
    return Character(**p, face=a.face, style=st)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mograph.human", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("setup")
    sub.add_parser("presets")
    for name in ("still", "clip"):
        s = sub.add_parser(name)
        s.add_argument("preset")
        if name == "clip":
            s.add_argument("motion")
            s.add_argument("--seconds", type=float, default=3.0)
            s.add_argument("--fps", type=int, default=24)
            s.add_argument("--yaw-end", type=float, default=None)
            s.add_argument("--cycle", type=float, default=None)
        else:
            s.add_argument("--pose", default="stand_contra")
        s.add_argument("--yaw", type=float, default=25.0)
        s.add_argument("--W", type=int, default=1080)
        s.add_argument("--H", type=int, default=1920)
        s.add_argument("--out", required=True)
        s.add_argument("--bg", default="none")
        s.add_argument("--body", default="")
        s.add_argument("--outfit", default="")
        s.add_argument("--face", default="minimal")
        s.add_argument("--color", default="0.07,0.07,0.08")
        s.add_argument("--width", type=float, default=3.2)
    sh = sub.add_parser("sheet")
    sh.add_argument("--out", default="human_sheet.png")
    sc = sub.add_parser("showcase")
    sc.add_argument("--out", default="human_showcase.png")
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        from .assets import setup_env
        setup_env()
        return 0
    if a.cmd == "presets":
        from .character import PRESETS
        from .poses import LIBRARY
        from .garments import TOPS, BOTTOMS, SKIRTS, SHOE_STYLE
        print("presets :", ", ".join(PRESETS))
        print("poses   :", ", ".join(LIBRARY))
        print("motions : walk_cycle, walk_across, run_cycle, wave, idle, <pose>, {'keys': [(t, pose, kw)…]}")
        print("tops    :", ", ".join(TOPS), "| bottoms:", ", ".join(BOTTOMS), ", skirt, pencil_skirt")
        print("dress   :", ", ".join(SKIRTS), "| shoes:", ", ".join(SHOE_STYLE))
        print("hair    : short, buzz, side, bob, long, bun, ponytail, bald | face: none, minimal, simple")
        return 0
    bg = None if a.__dict__.get("bg", "none") in ("none", "transparent") else (1, 1, 1) if a.bg == "white" else \
        tuple(float(x) for x in a.bg.split(","))
    if a.cmd == "still":
        import cv2
        ch = _build(a)
        if a.out.endswith(".svg"):
            open(a.out, "w").write(ch.svg(pose=a.pose, yaw=a.yaw, W=a.W, H=a.H))
        else:
            im = ch.render(pose=a.pose, yaw=a.yaw, W=a.W, H=a.H, bg=bg)
            if bg is None:
                al = im[..., 3:4] / 255.0
                rgb = (im[..., :3] / (al + 1e-6)).clip(0, 255).astype("uint8")
                im = __import__("numpy").dstack([rgb, im[..., 3]])
            cv2.imwrite(a.out, cv2.cvtColor(im, cv2.COLOR_RGBA2BGRA))
        print(a.out)
        return 0
    if a.cmd == "clip":
        from .clip import render_clip
        ch = _build(a)
        render_clip(ch, a.motion, a.out, seconds=a.seconds, fps=a.fps, yaw=a.yaw, W=a.W, H=a.H, bg=bg,
                    yaw_end=a.yaw_end, cycle_s=a.cycle)
        print(a.out)
        return 0
    if a.cmd == "sheet":
        import cv2
        import numpy as np
        from .character import Character, PRESETS, preset
        tiles = []
        for n in PRESETS:
            im = Character(**preset(n)).render(pose="stand_contra", yaw=25, W=300, H=540, bg=(1, 1, 1))[..., :3].copy()
            cv2.putText(im, n, (6, 18), 0, 0.45, (0, 0, 200), 1)
            tiles.append(im)
        while len(tiles) % 8:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
        cv2.imwrite(a.out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
        print(a.out)
        return 0
    if a.cmd == "showcase":
        import cv2
        import numpy as np
        from .character import Character, preset
        jobs = [("casual_man", "walk", 70, {"phase": 0.1}), ("office_woman", "stand_contra", 20, {}),
                ("business_man", "present", 25, {}), ("heavy_man", "stand", 30, {}), ("old_woman", "stand", 35, {}),
                ("girl", "wave", 15, {}), ("boy", "run", 75, {"phase": 0.2}), ("dress_woman", "sit", 45, {}),
                ("athlete", "hands_on_hips", 10, {}), ("teen", "phone", 30, {}),
                ("coat_woman", "walk", 60, {"phase": 0.6}), ("old_man", "think", 30, {})]
        tiles = []
        for n, pose, yaw, kw in jobs:
            im = Character(**preset(n)).render(pose=pose, yaw=yaw, W=360, H=620, bg=(1, 1, 1), **kw)[..., :3].copy()
            cv2.putText(im, f"{n} / {pose}", (6, 18), 0, 0.45, (0, 0, 200), 1)
            tiles.append(im)
        rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
        cv2.imwrite(a.out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
        print(a.out)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
