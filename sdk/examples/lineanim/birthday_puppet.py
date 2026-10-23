"""Birthday panel — puppet-pin acting, code only (no video models).

Boy (back layer) smiles, leans in, lifts his arm and rests his hand on the
girl's shoulder; the girl (front layer) turns up to look at him and smiles.
Pins = AE Puppet Position Pins (MLS rigid).  Run:
  PYTHONPATH=sdk python3 sdk/examples/lineanim/birthday_puppet.py /tmp/sb/storyboard.png out.mp4
"""
import sys, subprocess, numpy as np, cv2, cairo
from PIL import Image
sys.path.insert(0, __file__.rsplit('/examples/', 1)[0])
from mograph.lineanim.panel_anim import tonal, soft_lines, geodesic_order
from mograph.lineanim.puppet import Track, warp_pins, poly_mask, smooth

SRC = sys.argv[1] if len(sys.argv) > 1 else '/tmp/sb/storyboard.png'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/sb/puppet.mp4'
PREVIEW = '--preview' in sys.argv
FPS, T = 30, 9.0
OW, OH = 1920, 1080

im = np.array(Image.open(SRC).convert('L'))
pan = im[666:896, 3:502]
A = tonal(pan, s=4)                       # 920 x 1996, panel coords below
H, W = A.shape
LINES = soft_lines(pan, s=4, lo=40, rng=30)

# ---------------------------------------------------------------- layers
GIRL_POLY = [(900, 188), (965, 193), (1025, 228), (1062, 298), (1088, 360), (1088, 452),
             (1112, 508), (1142, 600), (1162, 760), (1172, 920), (733, 920), (733, 700),
             (744, 560), (768, 500), (783, 420), (793, 320), (813, 240), (853, 198)]
gm = poly_mask((H, W), GIRL_POLY, feather=1.2)
girl_img, girl_a = A * gm, gm
boy_img = A * (1 - gm)                     # girl region -> black (revealed bg)
boy_a = np.ones_like(A)

def rot_pins(pts, piv, ang):
    c, s = np.cos(ang), np.sin(ang)
    return [(piv[0] + c * (x - piv[0]) - s * (y - piv[1]),
             piv[1] + s * (x - piv[0]) + c * (y - piv[1])) for x, y in pts]

# ---------------------------------------------------------------- tracks
boy_tilt  = Track([(2.6, 0.0), (3.8, np.radians(3.5), 'io'), (7.0, np.radians(3.0), 'io')])
boy_smile = Track([(2.6, 0.0), (3.4, 1.0, 'io')])
boy_arm   = Track([(3.4, 0.0), (4.6, 1.0, 'io')])
hand_in   = Track([(3.8, 0.0), (4.9, 1.0, "back")])
girl_turn = Track([(0.0, np.radians(7)), (4.7, np.radians(7)), (5.9, np.radians(-1.2), 'io'),
                   (6.7, np.radians(-0.6), 'io')])
girl_look = Track([(0.0, 1.0), (4.5, 1.0), (5.5, 0.0, 'io')])
girl_smile = Track([(0.0, 0.0), (5.2, 0.0), (6.2, 1.0, 'io')])
blink = lambda t, t0: max(0.0, 1 - abs(t - t0) / 0.13)   # 0..1 eyelid close

# ---------------------------------------------------------------- hand asset
SH_CURVE = [(738, 516), (760, 497), (790, 481), (830, 471), (858, 458), (880, 450)]

def _arr(S, w, h, scale):
    arr = np.frombuffer(S.get_data(), np.uint8).reshape(h * scale, S.get_stride() // 4, 4)[:, :w * scale]
    arr = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) / 255
    a = arr[..., 3]; return np.clip(arr[..., 2] / np.maximum(a, 1e-4), 0, 1), a

def draw_hand(scale=4, anchor=(1050, 464), rot=0.40, size=1.35):
    """Boy's hand on the girl's FAR shoulder (arm hidden behind her back):
    foreshortened back of hand on top of the shoulder, four fingers curling
    over the front. Local frame: x along knuckle line, y down the fingers."""
    x0, y0, w, h = anchor[0] - 110, anchor[1] - 80, 230, 200
    S = cairo.ImageSurface(cairo.FORMAT_ARGB32, w * scale, h * scale)
    c = cairo.Context(S); c.scale(scale, scale); c.translate(-x0, -y0)
    c.translate(*anchor); c.rotate(rot); c.scale(size, size)
    c.set_line_cap(cairo.LINE_CAP_ROUND); c.set_line_join(cairo.LINE_JOIN_ROUND)
    SKIN, WHITE, INK = (0.80, 0.80, 0.80), (1, 1, 1), (0.07, 0.07, 0.07)
    # fingers: (x on knuckle line, length, splay, width)
    F = [(-19, 24, 0.20, 10.0), (-6.5, 32, 0.07, 11.4), (6.5, 30, -0.04, 11.0), (18, 21, -0.16, 9.2)]
    def finger(x, L, sp, wd):
        dx, dy = np.sin(sp), np.cos(sp); nx, ny = dy, -dx
        r = wd / 2; tr = r * 0.78; tx, ty = x + dx * L - 1.5, dy * L
        def fn():
            c.new_path(); c.move_to(x - nx * r, -2 - ny * r)
            c.curve_to(x - nx * r + dx * L * .5, -ny * r + dy * L * .5 + 1,
                       tx - nx * tr, ty - ny * tr - 3, tx - nx * tr, ty - ny * tr)
            a0 = np.arctan2(-ny, -nx); c.arc_negative(tx, ty, tr, a0, a0 - np.pi)
            c.curve_to(tx + nx * tr, ty + ny * tr - 3, x + nx * r + dx * L * .5, ny * r + dy * L * .5 + 1,
                       x + nx * r, -2 + ny * r)
            c.close_path()
        return fn, (x, L, dx, dy, nx, ny)
    fns = [finger(*f) for f in F]
    def back():   # foreshortened back of hand, wrist going behind (up)
        c.new_path(); c.move_to(-30, 2); c.curve_to(-32, -14, -16, -26, 2, -27)
        c.curve_to(20, -27, 32, -16, 28, 2); c.close_path()
    # white silhouettes first (outer contour), then fills, then ink
    for fn, _ in fns: fn(); c.set_source_rgb(*WHITE); c.set_line_width(2.4); c.stroke()
    back(); c.set_source_rgb(*WHITE); c.set_line_width(2.4); c.stroke()
    back(); c.set_source_rgb(*SKIN); c.fill()
    for fn, _ in fns[::-1]:
        fn(); c.set_source_rgb(*SKIN); c.fill_preserve()
        c.set_source_rgb(*INK); c.set_line_width(0.95); c.stroke()
    c.set_source_rgb(*INK); c.set_line_width(1.0)
    for fn, (x, L, dx, dy, nx, ny) in fns:
        for k, ln in ((0.40, 3.0), (0.70, 2.4)):
            cx, cy = x + dx * L * k, dy * L * k
            c.move_to(cx - nx * ln, cy - ny * ln); c.line_to(cx + nx * ln * .7, cy + ny * ln * .7); c.stroke()
    c.set_source_rgba(*INK, 0.5); c.set_line_width(0.9)       # knuckle ridge + tendons
    c.move_to(-24, -4); c.curve_to(-8, -8, 8, -8, 24, -4); c.stroke()
    for xx in (-10, 4): c.move_to(xx, -8); c.line_to(xx - 2, -22); c.stroke()
    return x0, y0, *_arr(S, w, h, scale)

HX, HY, HAND_L, HAND_A = draw_hand()
HAND_L = cv2.GaussianBlur(HAND_L, (0, 0), 0.7); HAND_A = cv2.GaussianBlur(HAND_A, (0, 0), 0.7)
# entry mask: hand slides in from behind her hair/back (left), revealed by a ramp
XX = np.arange(W, dtype=np.float32)[None].repeat(H, 0)

# ---------------------------------------------------------------- per-frame
def boy_layer(t):
    img = np.dstack([boy_img, boy_a])
    # head tilt toward her (pivot at neck)
    ang = float(boy_tilt(t))
    head = [(640, 110), (560, 130), (575, 205), (672, 300), (712, 240), (700, 150), (610, 270)]
    fixed = [(560, 360), (700, 352), (625, 400), (520, 330), (760, 340)]
    if abs(ang) > 1e-4:
        img = warp_pins(img, head + fixed, rot_pins(head, (622, 330), ang) + fixed,
                        roi=(470, 30, 830, 430), border=50)
    # smile: lift back mouth corner, pull slightly
    k = float(boy_smile(t))
    if k > 1e-3:
        src = [(651, 257), (683, 266), (714, 240), (680, 296), (660, 230)]
        dst = [(649 - 1 * k, 257 - 3.2 * k), (683, 265.2 - 0.6 * k), (714, 240), (680, 296), (660, 230)]
        img = warp_pins(img, src, dst, roi=(615, 215, 735, 310), border=18)
    # arm lift (shoulder rises, elbow swings back toward her)
    k = float(boy_arm(t))
    if k > 1e-3:
        src = [(770, 345), (792, 470), (790, 600), (600, 500), (640, 760), (430, 600), (620, 330)]
        dst = [(776 + 3 * k, 345 - 6 * k), (800 + 8 * k, 466 - 6 * k), (800 + 12 * k, 595 - 10 * k),
               (600, 500), (640, 760), (430, 600), (620, 330)]
        img = warp_pins(img, src, dst, roi=(500, 300, 900, 820), border=60)
    return img[..., 0], img[..., 1]

def girl_layer(t):
    img = np.dstack([girl_img, girl_a])
    ang = float(girl_turn(t))
    head = [(900, 205), (820, 260), (812, 360), (870, 385), (935, 330), (1000, 260), (1060, 360),
            (960, 395), (850, 300)]
    fixed = [(870, 470), (940, 470), (800, 520), (1080, 500), (900, 520)]
    if abs(ang) > 1e-4:
        img = warp_pins(img, head + fixed, rot_pins(head, (905, 448), ang) + fixed,
                        roi=(760, 170, 1110, 560), border=50)
    # eyes: pupils look forward (down-right) -> up at him
    k = float(girl_look(t))
    if k > 1e-3:
        src = [(864, 293), (915, 301), (840, 280), (940, 280), (840, 315), (940, 318)]
        dst = [(864 + 2.5 * k, 293 + 3 * k), (915 + 2.5 * k, 301 + 3 * k)] + src[2:]
        img = warp_pins(img, src, dst, roi=(830, 265, 950, 325), border=14)
    # smile (original drawing = full smile; we start from a neutral version)
    k = 1 - float(girl_smile(t))
    if k > 1e-3:
        src = [(889, 345), (856, 338), (872, 350), (870, 320), (900, 365)]
        dst = [(888, 345 + 2.6 * k), (857, 338 + 1.2 * k), (872, 350), (870, 320), (900, 365)]
        img = warp_pins(img, src, dst, roi=(835, 318, 915, 372), border=14)
    # blink at 6.9s
    b = blink(t, 6.9)
    if b > 1e-3:
        src = [(864, 284), (915, 292), (864, 304), (915, 312), (840, 270), (945, 275)]
        dst = [(864, 284 + 8 * b), (915, 292 + 8 * b), (864, 304), (915, 312), (840, 270), (945, 275)]
        img = warp_pins(img, src, dst, roi=(830, 262, 950, 330), border=14)
    return img[..., 0], img[..., 1]

# far-shoulder top contour (hand drops onto it from behind)
FAR_SH = [(1000, 452), (1030, 462), (1056, 470), (1080, 486), (1100, 512)]

def hand_layer(t, base):
    k = float(hand_in(t))
    if k <= 1e-3: return base
    dx, dy = (1 - k) * 10, (1 - k) * -40             # fingers slide over the shoulder from behind
    M = np.float32([[1, 0, HX + dx], [0, 1, HY + dy]])
    L = cv2.warpAffine(HAND_L, M, (W, H)); Aa = cv2.warpAffine(HAND_A, M, (W, H))
    # appears from behind the shoulder: reveal line slides up as it lands
    m = np.zeros((H, W), np.float32)
    lift = 4 * min(1.0, k * 1.3)
    poly = [(x, y - lift) for x, y in FAR_SH] + [(1100, 620), (1000, 620)]
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1.0, cv2.LINE_AA)
    Aa = Aa * cv2.GaussianBlur(m, (0, 0), 2.0)
    return base * (1 - Aa) + L * Aa

def candle(t, a):
    if t < 2.0: return a
    fl = 1 + 0.07 * np.sin(t * 23) + 0.05 * np.sin(t * 37 + 1)
    xs, ys = 1360, 600
    sub = a[ys - 40:ys + 20, xs - 30:xs + 30].copy()
    sub = cv2.resize(sub, None, fx=1, fy=fl)[-60:]
    a[ys + 20 - sub.shape[0]:ys + 20, xs - 30:xs + 30] = sub
    g = np.zeros_like(a); cv2.circle(g, (xs, ys - 12), 60, 1, -1)
    return np.clip(a + cv2.GaussianBlur(g, (0, 0), 40) * 0.10 * fl * smooth(t - 2), 0, 1)

def text_layer(t, fr):
    if t < 6.2: return fr
    u = smooth((t - 6.2) / 1.3)
    S = cairo.ImageSurface(cairo.FORMAT_A8, OW, OH); c = cairo.Context(S)
    c.select_font_face('Dancing Script'); c.set_source_rgba(1, 1, 1, 1)
    c.set_font_size(64); c.move_to(1330, 330); c.show_text('Happy Birthday')
    c.set_font_size(48); c.move_to(1385, 395); c.show_text('my little sister')
    m = np.frombuffer(S.get_data(), np.uint8).reshape(OH, S.get_stride())[:, :OW].astype(np.float32)
    wx = int(1320 + 500 * u); ramp = np.clip((wx - np.arange(OW)) / 30, 0, 1)
    return np.maximum(fr, m * ramp[None] * 0.92)

# draw-on order from the start pose
order = geodesic_order(LINES, seed=(0, int(0.83 * H)))

def frame(t):
    bi, ba = boy_layer(t)
    gi, ga = girl_layer(t)
    a = bi * (1 - ga) + gi * ga
    a = hand_layer(t, a)
    if t < 2.8:
        prog = smooth(t / 2.6) * 1.08
        a = a * np.clip((prog - order) / 0.04, 0, 1)
    a = candle(t, a)
    br = 1 + 0.004 * np.sin(t * 2 * np.pi / 3.4)
    z = (1.0 + 0.06 * smooth(t / T)) * br
    sc = z * OW / W; cx, cy = W * 0.47, H * 0.5
    M = np.float32([[sc, 0, OW / 2 - cx * sc], [0, sc, OH / 2 - cy * sc + 40]])
    out = cv2.warpAffine(a, M, (OW, OH), flags=cv2.INTER_CUBIC)
    fr = np.clip(out * 0.94 * 255 + 9, 0, 255)
    fr = text_layer(t, fr)
    return fr.astype(np.uint8)

if __name__ == '__main__':
    if PREVIEW:
        ts = [1.2, 3.0, 4.2, 4.8, 5.5, 6.6, 8.0]
        for t in ts:
            cv2.imwrite(f'/tmp/sb/pv_{t:.1f}.png', frame(t))
        print('preview ok'); sys.exit()
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
                          '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                          '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium', OUT],
                         stdin=subprocess.PIPE)
    for f in range(int(T * FPS)):
        p.stdin.write(frame(f / FPS).tobytes())
    p.stdin.close(); p.wait(); print('ok', OUT)
