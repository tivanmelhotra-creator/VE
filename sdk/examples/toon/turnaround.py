"""model sheet: front / 3-4 / side of the 3D ToonRig in a relaxed pose.  python3 turnaround.py out.png"""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import cairo, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from mograph.human.lineart import Camera
from mograph.mocap.director import axis_angle, roty
from mograph.toon import PAPER, ToonRig, compose, render_toon, toon_style
rig = ToonRig("guy"); toons = {"a": rig}
def pose(yaw):
    rot = {}
    for sd, sg in (("L", 1), ("R", -1)):
        rot[f"upperarm01.{sd}"] = axis_angle([0, 0, 1], math.radians(-30 * sg))
        rot[f"lowerarm01.{sd}"] = axis_angle([1, 0, 0], math.radians(-10))
    return rig.body.pose_matrices(rot, (0, 0, 0), roty(yaw)).astype(np.float64)
class A0:
    def __init__(s, yaw): s.yaw = yaw
    def matrices(s, t, scene): return pose(s.yaw)
class S0:
    def __init__(s, a): s.actors = {"a": a}
W, H = 700, 1300; tiles = []
for yaw in (0, 30, 90):
    cam = Camera(target=(0, 10.2, 0), yaw=0, pitch=1, fov=22, W=W, H=H); cam.dist = 62
    fr = render_toon(S0(A0(yaw)), toons, 0, cam, toon_style(4.4), ss=1.5, fig_px=H * 0.9)
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H); c = cairo.Context(s); c.set_source_rgb(*PAPER); c.paint()
    c.set_source_rgba(0.55, 0.48, 0.36, 0.6); c.set_line_width(2)
    for y in (180, 1180):
        c.move_to(0, y); c.line_to(W, y); c.stroke()
    compose(c, fr, cam, toons, {"a": dict(expr="neutral", blink=False)}, W, H, 0, boil_amp=0)
    s.flush(); tiles.append(np.ndarray((H, W, 4), np.uint8, s.get_data())[..., :3][..., ::-1].copy())
img = Image.fromarray(np.hstack(tiles)); d = ImageDraw.Draw(img)
f = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf", 58)
for k, lab in enumerate(("روبرو", "سه‌رخ", "نیم‌رخ")):
    d.text((W * k + W // 2, 95), lab, font=f, fill=(20, 18, 16), anchor="mm", direction="rtl", language="fa")
img.save(sys.argv[1] if len(sys.argv) > 1 else "/tmp/turnaround.png")
