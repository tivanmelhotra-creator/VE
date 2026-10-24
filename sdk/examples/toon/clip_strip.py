"""filmstrip of mocap clips on the toon rig:  python3 clip_strip.py "clipA+clipB" yaw t0,t1,... out.png"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import cairo, cv2, numpy as np
from mograph.human.lineart import Camera
from mograph.mocap.director import Actor, Scene
from mograph.toon import PAPER, ToonRig, compose, render_toon, toon_style
clips = sys.argv[1].split("+"); yaw = float(sys.argv[2]); ts = [float(x) for x in sys.argv[3].split(",")]
out = sys.argv[4] if len(sys.argv) > 4 else "/tmp/strip.png"
rows = []
for clip in clips:
    rig = ToonRig("guy"); toons = {"a": rig}
    A = Actor("a", rig, [{"clip": clip, "t": 0, "at": [0, 0], "heading": yaw, "root": "in_place"}], breath=0)
    S = Scene([A]); W, H = 420, 640; tiles = []
    for t in ts:
        cam = Camera(target=(0, 9.6, 0), yaw=0, pitch=3, fov=25, W=W, H=H); cam.dist = 62
        fr = render_toon(S, toons, t, cam, toon_style(4.4), ss=1.25, fig_px=H * 0.9)
        s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H); c = cairo.Context(s); c.set_source_rgb(*PAPER); c.paint()
        compose(c, fr, cam, toons, {"a": dict(expr="neutral", blink=False)}, W, H, t, boil_amp=0)
        s.flush(); a = np.ndarray((H, W, 4), np.uint8, s.get_data())[..., :3].copy()
        a = cv2.resize(a, (210, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(a, f"{clip[:16]} {t}", (4, 14), 0, 0.4, (0, 0, 0), 1); a[:, -1] = 255; tiles.append(a)
    rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows))
