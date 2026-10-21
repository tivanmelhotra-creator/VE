import os
import math
import wave
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 720, 1280
FPS = 30
DURATION = 148.0
TOTAL_FRAMES = int(DURATION * FPS)

OUT_DIR = "/working_dir/c_4fc7259f35079930/artifacts/file_generation/ttl=63d/output"
os.makedirs(OUT_DIR, exist_ok=True)
VIDEO_OUT = os.path.join(OUT_DIR, "money_trial_10chapters_final_v2.mp4")

FONT_FA_BOLD = "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"
FONT_FA_REG = "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"
FONT_EN_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Colors (RGB)
C_BLACK = (0, 0, 0)
C_WHITE = (255, 255, 255)
C_GRAY_LIGHT = (155, 155, 155)
C_GRAY_MID = (95, 95, 95)
C_GRAY_DARK = (62, 62, 62)
C_GRAY_FAINT = (36, 36, 36)

# ============================================================================
# 1. EASING & MATH HELPERS
# ============================================================================
def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))

def prog(t, t0, dur):
    if dur <= 1e-5:
        return 1.0 if t >= t0 else 0.0
    return clamp((t - t0) / dur)

def ease_out(t):
    t = clamp(t)
    return 1.0 - (1.0 - t) ** 3

def ease_in_out(t):
    t = clamp(t)
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0

def spring(t, t0, dur=0.55, freq=6.0, decay=5.0):
    if t <= t0:
        return 0.0
    p = (t - t0) / max(dur, 1e-4)
    if p >= 2.5:
        return 1.0
    return 1.0 - math.exp(-decay * p) * math.cos(freq * p)


# ============================================================================
# 2. CACHED RTL PERSIAN TEXT & OPTICAL BLOOM SPRITE ENGINE
# ============================================================================
_FONT_CACHE = {}
_TEXT_SPRITE_CACHE = {}

def get_font(path, size):
    key = (path, int(size))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, int(size))
    return _FONT_CACHE[key]

def get_text_sprite(text, size=28, color=(255, 255, 255), bold=True, glow=False):
    """
    Renders shaped RTL Persian text once and caches the tight RGBA numpy array
    (including optional soft outer white optical bloom).
    """
    key = (text, int(size), tuple(color), bool(bold), bool(glow))
    if key in _TEXT_SPRITE_CACHE:
        return _TEXT_SPRITE_CACHE[key]

    is_ascii = all(ord(c) < 128 for c in text)
    font_path = FONT_EN_BOLD if is_ascii else (FONT_FA_BOLD if bold else FONT_FA_REG)
    font = get_font(font_path, size)
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    ddraw = ImageDraw.Draw(dummy)
    if is_ascii:
        bb = ddraw.textbbox((0, 0), text, font=font)
    else:
        bb = ddraw.textbbox((0, 0), text, font=font, direction="rtl", language="fa")
    tw = max(4, bb[2] - bb[0])
    th = max(4, bb[3] - bb[1])
    pad = 22 if glow else 8
    im = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    if is_ascii:
        draw.text((pad - bb[0], pad - bb[1]), text, font=font, fill=(*color, 255))
    else:
        draw.text((pad - bb[0], pad - bb[1]), text, font=font, fill=(*color, 255), direction="rtl", language="fa")
    arr = np.array(im, dtype=np.float32)

    if glow:
        alpha_blur = cv2.GaussianBlur(arr[:, :, 3], (21, 21), 6.5) * 0.55
        glow_rgb = np.array(color, dtype=np.float32)[None, None, :]
        base_a = arr[:, :, 3:4] / 255.0
        gl_a = (alpha_blur[:, :, None] / 255.0) * (1.0 - base_a)
        out_a = np.clip(base_a + gl_a, 0.0, 1.0)
        safe_a = np.maximum(out_a, 1e-4)
        out_rgb = (arr[:, :, :3] * base_a + glow_rgb * gl_a) / safe_a
        arr = np.dstack([np.clip(out_rgb, 0, 255), out_a * 255.0])

    _TEXT_SPRITE_CACHE[key] = arr.astype(np.uint8)
    return _TEXT_SPRITE_CACHE[key]

def blit_text(canvas, text, cx, cy, size=28, color=C_WHITE, bold=True, glow=False, opacity=1.0, align="center", angle_deg=0.0):
    if opacity <= 0.01 or not text:
        return (0, 0)
    sp = get_text_sprite(text, size=size, color=color, bold=bold, glow=glow)
    if abs(angle_deg) > 0.1:
        sh, sw = sp.shape[:2]
        M = cv2.getRotationMatrix2D((sw * 0.5, sh * 0.5), angle_deg, 1.0)
        sp = cv2.warpAffine(sp, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    sh, sw = sp.shape[:2]
    if align == "center":
        x0 = int(round(cx - sw * 0.5))
    elif align == "right":
        x0 = int(round(cx - sw))
    else: # left
        x0 = int(round(cx))
    y0 = int(round(cy - sh * 0.5))

    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + sw), min(H, y0 + sh)
    if cx1 <= cx0 or cy1 <= cy0:
        return (sw, sh)

    sub = sp[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    a = (sub[:, :, 3:4].astype(np.float32) / 255.0) * float(opacity)
    if np.max(a) <= 0.005:
        return (sw, sh)
    dst = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    canvas[cy0:cy1, cx0:cx1] = (dst * (1.0 - a) + sub[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    return (sw, sh)


# ============================================================================
# 3. MONOLINE VECTOR PRIMITIVES (TRIM-PATHS, CALLOUTS, STAMPS, ICONS)
# ============================================================================
def col_fade(color, opacity):
    op = clamp(opacity)
    return (int(color[0] * op), int(color[1] * op), int(color[2] * op))

def draw_trim_line(canvas, p1, p2, p=1.0, color=C_WHITE, thickness=2, dashed=False, dash_len=8, gap_len=6):
    p = clamp(p)
    if p <= 0.005:
        return
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    ex = x1 + (x2 - x1) * p
    ey = y1 + (y2 - y1) * p
    if not dashed:
        cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(ex)), int(round(ey))), color, thickness, cv2.LINE_AA)
    else:
        dist = math.hypot(ex - x1, ey - y1)
        if dist < 1.0:
            return
        ux, uy = (ex - x1) / dist, (ey - y1) / dist
        cur = 0.0
        step = dash_len + gap_len
        while cur < dist:
            d_end = min(dist, cur + dash_len)
            sx, sy = int(round(x1 + ux * cur)), int(round(y1 + uy * cur))
            tx, ty = int(round(x1 + ux * d_end)), int(round(y1 + uy * d_end))
            cv2.line(canvas, (sx, sy), (tx, ty), color, thickness, cv2.LINE_AA)
            cur += step

def draw_trim_polyline(canvas, pts, p=1.0, color=C_WHITE, thickness=2, closed=False, dashed=False):
    p = clamp(p)
    if p <= 0.005 or len(pts) < 2:
        return
    all_pts = list(pts) + ([pts[0]] if closed else [])
    segs = []
    total_len = 0.0
    for i in range(len(all_pts) - 1):
        d = math.hypot(all_pts[i+1][0] - all_pts[i][0], all_pts[i+1][1] - all_pts[i][1])
        segs.append((all_pts[i], all_pts[i+1], d))
        total_len += d
    if total_len < 1e-3:
        return
    rem = total_len * p
    for a, b, d in segs:
        if rem <= 0.0:
            break
        seg_p = min(1.0, rem / max(d, 1e-4))
        draw_trim_line(canvas, a, b, seg_p, color=color, thickness=thickness, dashed=dashed)
        rem -= d

def draw_trim_circle(canvas, cx, cy, r, p=1.0, color=C_WHITE, thickness=2, start_angle=-90, dashed=False):
    p = clamp(p)
    if p <= 0.005 or r <= 1:
        return
    if not dashed:
        end_angle = start_angle + 360.0 * p
        cv2.ellipse(canvas, (int(round(cx)), int(round(cy))), (int(round(r)), int(round(r))),
                    0, start_angle, end_angle, color, thickness, cv2.LINE_AA)
    else:
        n_dashes = max(8, int(r * 0.45))
        max_i = int(round(n_dashes * p))
        for i in range(max_i):
            a0 = start_angle + (360.0 / n_dashes) * i
            a1 = a0 + (360.0 / n_dashes) * 0.55
            cv2.ellipse(canvas, (int(round(cx)), int(round(cy))), (int(round(r)), int(round(r))),
                        0, a0, a1, color, thickness, cv2.LINE_AA)

def draw_trim_rect(canvas, cx, cy, w, h, p=1.0, color=C_WHITE, thickness=2, dashed=False, fill_black=True):
    p = clamp(p)
    if p <= 0.005:
        return
    x0, y0 = cx - w * 0.5, cy - h * 0.5
    x1, y1 = cx + w * 0.5, cy + h * 0.5
    if fill_black and p > 0.15:
        cv2.rectangle(canvas, (int(x0)+2, int(y0)+2), (int(x1)-2, int(y1)-2), (0, 0, 0), -1)
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    draw_trim_polyline(canvas, pts, p=p, color=color, thickness=thickness, closed=True, dashed=dashed)

def draw_arrow(canvas, p1, p2, p=1.0, color=C_WHITE, thickness=2, dashed=False, head_size=11):
    p = clamp(p)
    if p <= 0.01:
        return
    draw_trim_line(canvas, p1, p2, p=p, color=color, thickness=thickness, dashed=dashed)
    if p > 0.75:
        hp = (p - 0.75) / 0.25
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        ang = math.atan2(y2 - y1, x2 - x1)
        hs = head_size * hp
        lx = x2 - hs * math.cos(ang - 0.5)
        ly = y2 - hs * math.sin(ang - 0.5)
        rx = x2 - hs * math.cos(ang + 0.5)
        ry = y2 - hs * math.sin(ang + 0.5)
        cv2.line(canvas, (int(x2), int(y2)), (int(lx), int(ly)), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (int(x2), int(y2)), (int(rx), int(ry)), color, thickness, cv2.LINE_AA)

def draw_callout(canvas, anchor_pt, elbow_pt, end_pt, label, p=1.0, color=C_WHITE, text_size=21, glow=False):
    p = clamp(p)
    if p <= 0.01:
        return
    c = col_fade(color, min(1.0, p * 1.5))
    cv2.circle(canvas, (int(anchor_pt[0]), int(anchor_pt[1])), 4, c, -1, cv2.LINE_AA)
    draw_trim_polyline(canvas, [anchor_pt, elbow_pt, end_pt], p=min(1.0, p * 1.25), color=c, thickness=2)
    if p > 0.35:
        tp = (p - 0.35) / 0.65
        shelf_cx = clamp((elbow_pt[0] + end_pt[0]) * 0.5, 115, W - 115)
        shelf_cy = end_pt[1] - 22
        blit_text(canvas, label, shelf_cx, shelf_cy, size=text_size, color=color, bold=True, glow=glow, opacity=tp, align="center")

def draw_stamp(canvas, cx, cy, text, t, t0, angle_deg=-8.0, w=175, h=64):
    if t < t0:
        return
    sp = spring(t, t0, dur=0.38, freq=7.5, decay=5.5)
    scale = max(0.2, 1.85 - 0.85 * sp)
    op = clamp((t - t0) / 0.12)
    sw, sh = int(w * scale), int(h * scale)
    stamp_im = np.zeros((sh + 40, sw + 40, 3), dtype=np.uint8)
    ox, oy = 20, 20
    cv2.rectangle(stamp_im, (ox, oy), (ox + sw, oy + sh), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(stamp_im, (ox + 6, oy + 6), (ox + sw - 6, oy + sh - 6), (255, 255, 255), 1, cv2.LINE_AA)
    blit_text(stamp_im, text, ox + sw * 0.5, oy + sh * 0.5, size=int(28 * scale), color=C_WHITE, bold=True, glow=True, opacity=op)

    M = cv2.getRotationMatrix2D(((sw + 40) * 0.5, (sh + 40) * 0.5), angle_deg, 1.0)
    rot = cv2.warpAffine(stamp_im, M, (sw + 40, sh + 40), flags=cv2.INTER_LINEAR)
    x0 = int(round(cx - (sw + 40) * 0.5))
    y0 = int(round(cy - (sh + 40) * 0.5))
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + sw + 40), min(H, y0 + sh + 40)
    if cx1 > cx0 and cy1 > cy0:
        sub = rot[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        mask = (sub.max(axis=2) > 12)[:, :, None]
        # Clear interior black box slightly so stamp reads cleanly over lines
        canvas[cy0:cy1, cx0:cx1] = np.where(mask, np.maximum(canvas[cy0:cy1, cx0:cx1], sub), canvas[cy0:cy1, cx0:cx1])

def draw_money_core(canvas, cx, cy, r=44, p=1.0, t=0.0, color=C_WHITE, pulse=True):
    """
    Core Visual Symbol of the Narrator ('من' = Money):
    Concentric geometric double-circle coin with vertical currency axis & radial ticks.
    """
    p = clamp(p)
    if p <= 0.01:
        return
    c = col_fade(color, p)
    pr = r * (1.0 + (0.04 * math.sin(t * 4.2) if pulse else 0.0))
    draw_trim_circle(canvas, cx, cy, pr, p=p, color=c, thickness=2)
    draw_trim_circle(canvas, cx, cy, pr * 0.68, p=p, color=c, thickness=2, start_angle=90)
    # Inner vertical axis & stylized S/coin bar
    if p > 0.3:
        ip = (p - 0.3) / 0.7
        draw_trim_line(canvas, (cx, cy - pr * 0.48), (cx, cy + pr * 0.48), p=ip, color=c, thickness=2)
        draw_trim_circle(canvas, cx, cy, pr * 0.24, p=ip, color=c, thickness=2)
    # 8 radial tick marks rotating slowly
    if p > 0.5:
        tp = (p - 0.5) / 0.5
        rot_a = t * 0.45
        for i in range(8):
            ang = rot_a + i * (math.pi / 4.0)
            r0 = pr * 1.12
            r1 = pr * (1.12 + 0.16 * tp)
            x0, y0 = cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)
            x1, y1 = cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)
            cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), c, 2, cv2.LINE_AA)

def draw_human_icon(canvas, cx, cy, scale=1.0, p=1.0, color=C_WHITE, dashed=False, thickness=2):
    """
    Minimalist outline human silhouette matching the exact @ENSAN.BESHAM style:
    Head circle + rounded shoulder/torso arch open at the bottom.
    """
    p = clamp(p)
    if p <= 0.01:
        return
    c = col_fade(color, p)
    hr = int(round(13 * scale))
    hcy = int(round(cy - 18 * scale))
    draw_trim_circle(canvas, cx, hcy, hr, p=min(1.0, p * 1.4), color=c, thickness=thickness, dashed=dashed)
    if p > 0.25:
        bp = (p - 0.25) / 0.75
        bw = int(round(24 * scale))
        bh = int(round(26 * scale))
        bcy = int(round(cy + 20 * scale))
        if not dashed:
            cv2.ellipse(canvas, (int(round(cx)), bcy), (bw, bh), 0, 180, 180 + int(180 * bp), c, thickness, cv2.LINE_AA)
        else:
            for i in range(int(6 * bp)):
                a0 = 180 + i * 32
                cv2.ellipse(canvas, (int(round(cx)), bcy), (bw, bh), 0, a0, a0 + 18, c, thickness, cv2.LINE_AA)


# ============================================================================
# 4. PERSISTENT HUD ENGINE (HEADER NOTCH, PAGINATION DOTS, KARAOKE SUBTITLES)
# ============================================================================
CHAPTER_META = [
    (1, 0.0, 12.0, "من و شما"),
    (2, 12.0, 24.0, "کمبود"),
    (3, 24.0, 35.0, "گاوصندوق"),
    (4, 35.0, 52.0, "انسولین"),
    (5, 52.0, 72.0, "ارزش"),
    (6, 72.0, 90.0, "زمان"),
    (7, 90.0, 106.0, "طبیعت"),
    (8, 106.0, 120.0, "امضا"),
    (9, 120.0, 134.0, "آینه"),
    (10, 134.0, 148.0, "کارنامه"),
]

# Complete 10-Chapter Word-by-Word Karaoke Script (`(t_start, t_end, text_with_bold_markers)`)
SUBTITLES = [
    # Chapter 1 (00:00 - 00:12)
    (0.0, 4.0, "**شما** انسان‌ها می‌گین از **منِ** پول، همه‌چیز **خراب** شده."),
    (4.0, 6.0, "می‌گین **ریشه‌ی** همه‌ی فسادها **منم.**"),
    (6.0, 8.0, "ولی دلیلِ **من** علیه شما **مستندتره؛**"),
    (8.0, 12.0, "چون من فقط **«ممکنه»** وسوسه کنم، اما شما **انجام** **دادید.**"),
    # Chapter 2 (00:12 - 00:24)
    (12.0, 17.0, "**اولین** بهانه‌تون اینه که **«منابع** **کمه** و به همه **نمی‌رسه.»**"),
    (17.0, 19.0, "واقعاً **چرا** نمی‌رسه؟"),
    (19.0, 24.0, "چون **شما** انسان‌ها در کل تاریخ، دور هرچیزی **دیوار** کشیدین."),
    # Chapter 3 (00:24 - 00:35)
    (24.0, 29.0, "ثروتِ فقط **۱** **درصد** از شما، بیشتر از کلِ دارایی **۹۵** **درصد** بقیه‌ست."),
    (29.0, 32.0, "در حالی که من برای **همه‌ی** شما کافی بودم،"),
    (32.0, 35.0, "اما شما منو توی **گاوصندوق** حبس کردین."),
    # Chapter 4 (00:35 - 00:52)
    (35.0, 39.0, "**انسولین** یه کشف نجات‌بخش بود که مخترعش فقط به **یک** **دلار** فروخت."),
    (39.0, 42.0, "گفت دارو باید به دست **همه** برسه."),
    (42.0, 46.0, "**شما** چی کار کردین؟ همون رو تا **۳۰۰** **دلار** به بیمار فروختین."),
    (46.0, 52.0, "هزاران داروی حیاتی پشتِ **برچسب** **قیمتِ** شما از دسترس **خارج** **شدن.**"),
    # Chapter 5 (00:52 - 01:12)
    (52.0, 56.0, "**اتهام** **بعدی‌تون:** «پول جای **ارزش‌های** **انسانی** رو گرفته و کنترل‌مون می‌کنه.»"),
    (56.0, 59.0, "کی الان **ارزشِ** چیزها رو تعیین می‌کنه؟"),
    (59.0, 62.0, "کسی که روی **آبِ** **خوردن** و **سقفِ** **بالا** **سر** قیمت می‌ذاره."),
    (62.0, 65.0, "کسی که تصمیم می‌گیره **دارو** گرون‌تر باشه یا **سلاح.**"),
    (65.0, 68.0, "کسی که پای فیشِ **حقوقِ** **یه** **معلم** رو امضا می‌کنه."),
    (68.0, 72.0, "**همه‌شون** **انسانن.** من روی هیچ‌چیزی قیمت نذاشتم؛ **شما** گذاشتین."),
    # Chapter 6 (01:12 - 01:30)
    (72.0, 76.0, "**ترس** **سوم‌تون:** «پول، **عمر** **و** **تصمیم‌های** ما رو می‌بلعه.»"),
    (76.0, 79.0, "ساعتی که می‌گه **کِی** **بیدار** **شین،**"),
    (79.0, 81.5, "اجاره‌خونه‌ای که می‌گه **کجا** **زندگی** **کنین،**"),
    (81.5, 84.5, "تورمی که می‌گه **چی** **بخرین** و کِی **حق** **دارین** **استراحت** **کنین.**"),
    (84.5, 90.0, "**سال‌هاست** عمرتون معامله می‌شه؛ فقط اسمش رو گذاشتین **«زندگی** **عادی».**"),
    # Chapter 7 (01:30 - 01:46)
    (90.0, 94.0, "**بهانه‌ی** **چهارم‌تون:** «به خاطر پول، **زمین** داره نابود می‌شه.»"),
    (94.0, 97.0, "کسی که درخت رو می‌بُره، **آدمه.**"),
    (97.0, 101.0, "هر کارخونه و اره‌ای که ساختین، **دستِ** **یه** **انسان** روشنش کرده؛"),
    (101.0, 106.0, "من حتی از کاغذِ همون درختی ساخته شدم که **خودتون** بریدین."),
    # Chapter 8 (01:46 - 02:00)
    (106.0, 111.0, "هیچ اسکناسی **به‌خودیِ‌خود** جنگی راه ننداخته، مگه با **تصمیم** **یه** **انسان.**"),
    (111.0, 115.0, "تازه خیلی وقت‌ها هم که بهانه نبودم، باز هم **راهش** **رو** **پیدا** **کردین.**"),
    (115.0, 120.0, "هر قراردادی که زندگی‌ها رو زیر و رو کرده، یه **امضای** **انسانی** پاش بوده."),
    # Chapter 9 (02:00 - 02:14)
    (120.0, 123.5, "من از **طمع‌ها** **و** **ترس‌های** شما ساخته شدم."),
    (123.5, 128.0, "وقتی به من نگاه می‌کنین، دارین **صورت‌حسابِ** **خودتون** رو می‌خونین."),
    (128.0, 131.0, "**نمی‌ترسین** که پول بی‌رحم باشه؛"),
    (131.0, 134.0, "**می‌ترسین** ببینین وقتی قدرت دارین، چقدر **شبیهِ** **خودِ** **واقعیتون** می‌شین."),
    # Chapter 10 (02:14 - 02:24)
    (134.0, 139.0, "با این **کارنامه‌ای** که توی قیمت گذاشتن روی **دارو،** **نان،** **طبیعت** و **جونِ** **هم** دارین..."),
    (139.0, 144.2, "**واقعاً** فکر می‌کنین مشکل از **یه** **تکه** **کاغذه؟**"),
]

def draw_persistent_hud(canvas, t, ch_idx, ch_title):
    if t > 144.5:
        hud_op = clamp((146.5 - t) / 2.0)
    else:
        hud_op = 1.0
    if hud_op <= 0.01:
        return

    # 1. Top-Right Chapter Header Bar (`y = 78`) with 45-degree notch on the right
    c_line = col_fade(C_GRAY_LIGHT, hud_op * 0.85)
    # Horizontal line across left/center, bending down at 45 degrees on the right (`x=460 -> 495, y=78 -> 113 -> 675`)
    pts_header = [(45, 78), (455, 78), (490, 113), (675, 113)]
    draw_trim_polyline(canvas, pts_header, p=1.0, color=c_line, thickness=2)
    # Inside the right notch (`x=495..670, y=82`): `پرونده:` in gray + `[عنوان فصل]` in bold glowing white!
    blit_text(canvas, "پرونده:", 668, 82, size=21, color=C_GRAY_LIGHT, bold=False, glow=False, opacity=hud_op, align="right")
    blit_text(canvas, ch_title, 595, 82, size=25, color=C_WHITE, bold=True, glow=True, opacity=hud_op, align="right")

    # Subtle signature handle on mid-left (matching @ENSAN.BESHAM aesthetic)
    c_wm = col_fade(C_GRAY_MID, hud_op * 0.65)
    cv2.rectangle(canvas, (52, 990), (74, 1012), c_wm, 1, cv2.LINE_AA)
    cv2.circle(canvas, (63, 1001), 5, c_wm, 1, cv2.LINE_AA)
    blit_text(canvas, "@MONEY.TRIAL", 50, 1030, size=14, color=C_GRAY_MID, bold=True, glow=False, opacity=hud_op * 0.65, align="left")

    # 2. Pagination Progress Dots (`y = 1055` = 82.4% of H) — 10 chapters (RTL order: dot 1 on right, dot 10 on left)
    dot_y = 1055
    spacing = 24
    start_x = int(W * 0.5 + 4.5 * spacing)
    for i in range(1, 11):
        dx = start_x - (i - 1) * spacing
        if i == ch_idx:
            cv2.circle(canvas, (dx, dot_y), 7, col_fade(C_WHITE, hud_op), -1, cv2.LINE_AA)
            cv2.circle(canvas, (dx, dot_y), 11, col_fade(C_GRAY_MID, hud_op * 0.5), 1, cv2.LINE_AA)
        elif i < ch_idx:
            cv2.circle(canvas, (dx, dot_y), 4, col_fade(C_GRAY_LIGHT, hud_op * 0.7), -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (dx, dot_y), 4, col_fade(C_GRAY_DARK, hud_op * 0.7), 1, cv2.LINE_AA)

    # 3. Word-by-Word Karaoke Subtitles (`y = 1135` = 88.6% of H)
    active_sub = None
    for s0, s1, txt in SUBTITLES:
        if s0 <= t <= s1:
            active_sub = (s0, s1, txt)
            break
    if active_sub is not None:
        s0, s1, raw_txt = active_sub
        raw_tokens = raw_txt.split(" ")
        tokens = []
        for rt in raw_tokens:
            is_b = rt.startswith("**") and rt.endswith("**")
            clean_w = rt.replace("**", "")
            tokens.append((clean_w, is_b))

        # Split into 1 or 2 balanced lines if > 7 words
        if len(tokens) > 7:
            mid = len(tokens) // 2
            lines = [tokens[:mid], tokens[mid:]]
            line_ys = [1122, 1168]
        else:
            lines = [tokens]
            line_ys = [1140]

        sub_p = clamp((t - s0) / max(0.1, (s1 - s0) * 0.88))
        n_total = len(tokens)
        active_word_idx = int(math.floor(sub_p * n_total))

        w_counter = 0
        for line_tokens, ly in zip(lines, line_ys):
            # Measure total width of line in RTL
            word_sprites = []
            total_w = 0
            gap = 10
            for w_str, is_b in line_tokens:
                sp_dim = get_text_sprite(w_str, size=27, color=C_GRAY_DARK, bold=True, glow=False)
                # Tight text width is roughly sprite width minus padding (16px)
                adv = max(12, sp_dim.shape[1] - 14)
                word_sprites.append((w_str, is_b, adv))
                total_w += adv
            total_w += gap * max(0, len(line_tokens) - 1)

            # Draw from right to left (RTL)
            cur_right = W * 0.5 + total_w * 0.5
            for w_str, is_b, adv in word_sprites:
                w_cx = cur_right - adv * 0.5
                if w_counter <= active_word_idx:
                    # Spoken word -> Pure White (#FFFFFF), with optical glow if bold keyword!
                    blit_text(canvas, w_str, w_cx, ly, size=27, color=C_WHITE, bold=True, glow=is_b, opacity=hud_op)
                else:
                    # Unspoken word -> Dim Gray (#4A4A4A)
                    blit_text(canvas, w_str, w_cx, ly, size=27, color=C_GRAY_DARK, bold=True, glow=False, opacity=hud_op)
                cur_right -= (adv + gap)
                w_counter += 1


# ============================================================================
# 5. COMPLETE 10-CHAPTER VECTOR CHOREOGRAPHY ENGINE
# ============================================================================
def render_chapter_scene(canvas, t, ch_idx):
    # ------------------------------------------------------------------------
    # CHAPTER 1: پرونده: من و شما (00:00 - 00:12)
    # ------------------------------------------------------------------------
    if ch_idx == 1:
        st = t - 0.0
        if st < 6.0:
            # 00:00 - 00:06: Money Core at top + 7 human silhouettes at bottom (zooming to 3 at 00:04)
            z = 1.0 + 0.42 * ease_in_out(prog(st, 3.8, 1.4))
            fade_out = 1.0 - ease_in_out(prog(st, 5.4, 0.55))
            draw_money_core(canvas, W * 0.5, 310, r=48, p=prog(st, 0.1, 1.2) * fade_out, t=t)
            blit_text(canvas, "پول «من»", W * 0.5, 225, size=22, color=C_GRAY_LIGHT, opacity=prog(st, 0.6, 0.6) * fade_out)

            for i in range(7):
                offset_i = i - 3
                hx = W * 0.5 + offset_i * 78 * z
                hy = 690 + (z - 1.0) * 60
                if -40 <= hx <= W + 40:
                    hp = prog(st, 0.4 + i * 0.15, 0.6) * fade_out
                    # Highlight the center 3 figures when zooming in at 00:04
                    col = C_WHITE if (abs(offset_i) <= 1 or st < 3.8) else C_GRAY_DARK
                    draw_human_icon(canvas, hx, hy, scale=1.05 * z, p=hp, color=col)
            if st > 1.5:
                draw_arrow(canvas, (W * 0.5, 385), (W * 0.5, 610), p=prog(st, 1.5, 0.8) * fade_out, color=C_GRAY_MID, dashed=True)
                blit_text(canvas, "اتهام: ریشه فسادها", W * 0.5 + 115, 495, size=20, color=C_GRAY_LIGHT, opacity=prog(st, 2.0, 0.6) * fade_out)

        if st >= 5.6:
            # 00:06 - 00:12: Dossier Folder `انسان` with `[مستند]` stamp & split comparison `من` vs `شما`
            fp = prog(st, 5.6, 0.9)
            fx, fy, fw, fh = W * 0.5, 560, 540, 470
            # Folder tab at top-right
            tab_pts = [(fx + fw*0.5 - 160, fy - fh*0.5), (fx + fw*0.5 - 135, fy - fh*0.5 - 36),
                       (fx + fw*0.5 - 10, fy - fh*0.5 - 36), (fx + fw*0.5, fy - fh*0.5)]
            draw_trim_polyline(canvas, tab_pts, p=fp, color=C_WHITE, thickness=2)
            blit_text(canvas, "پرونده: انسان", fx + fw*0.5 - 82, fy - fh*0.5 - 18, size=20, color=C_WHITE, opacity=fp)
            draw_trim_rect(canvas, fx, fy, fw, fh, p=fp, color=C_WHITE, thickness=2)

            # Stamp `[مستند]` at 00:06.6
            draw_stamp(canvas, fx, fy - 165, "مستند", st, 6.5, angle_deg=-7.0, w=165, h=56)

            # Split view inside folder at 00:08
            if st > 7.8:
                sp_p = prog(st, 7.8, 0.8)
                draw_trim_line(canvas, (fx, fy - 95), (fx, fy + fh*0.5 - 25), p=sp_p, color=C_GRAY_MID, thickness=2, dashed=True)

                # Right Half: `من` (Money) -> dashed box `«ممکنه»` + `وسوسه`
                rx = fx + 135
                draw_money_core(canvas, rx, fy - 25, r=32, p=sp_p, t=t)
                blit_text(canvas, "من «پول»", rx, fy - 82, size=22, color=C_WHITE, opacity=sp_p)
                draw_trim_rect(canvas, rx, fy + 95, 185, 82, p=prog(st, 8.3, 0.6), color=C_GRAY_LIGHT, thickness=2, dashed=True)
                blit_text(canvas, "«ممکنه»", rx, fy + 95, size=28, color=C_WHITE, glow=True, opacity=prog(st, 8.5, 0.4))
                blit_text(canvas, "صرفاً وسوسه", rx, fy + 165, size=19, color=C_GRAY_LIGHT, opacity=prog(st, 8.7, 0.4))

                # Left Half: `شما` (Human) -> document lines + `[انجام دادید]` stamp!
                lx = fx - 135
                draw_human_icon(canvas, lx, fy - 25, scale=0.95, p=sp_p, color=C_WHITE)
                blit_text(canvas, "شما «انسان»", lx, fy - 82, size=22, color=C_WHITE, opacity=sp_p)
                doc_p = prog(st, 9.2, 0.6)
                draw_trim_rect(canvas, lx, fy + 105, 190, 120, p=doc_p, color=C_GRAY_MID, thickness=2)
                for r_i in range(4):
                    ly_doc = fy + 65 + r_i * 24
                    draw_trim_line(canvas, (lx - 70, ly_doc), (lx + 70, ly_doc), p=doc_p, color=C_GRAY_DARK, thickness=2)
                draw_stamp(canvas, lx, fy + 108, "انجام دادید", st, 9.8, angle_deg=6.0, w=195, h=60)

    # ------------------------------------------------------------------------
    # CHAPTER 2: پرونده: کمبود (00:12 - 00:24)
    # ------------------------------------------------------------------------
    elif ch_idx == 2:
        st = t - 12.0
        # Phase 1 (00:12 - 00:19): Top Money (`محدود`) -> Bottom Human (`فقر`) + Giant `?` at 00:17
        dim_f = 0.28 if st > 5.0 else 1.0
        shift_x = -195 * ease_in_out(prog(st, 6.8, 0.9))
        base_cx = W * 0.5 + shift_x
        if st < 7.6:
            c_base = col_fade(C_WHITE, dim_f)
            draw_money_core(canvas, base_cx, 310, r=42, p=prog(st, 0.1, 0.8), t=t, color=c_base)
            draw_callout(canvas, (base_cx + 44, 300), (base_cx + 95, 260), (base_cx + 185, 260),
                         "محدود", p=prog(st, 0.6, 0.7) * dim_f, color=c_base)
            draw_arrow(canvas, (base_cx, 370), (base_cx, 620), p=prog(st, 1.1, 0.8), color=c_base, dashed=True)
            draw_human_icon(canvas, base_cx, 680, scale=1.15, p=prog(st, 1.4, 0.7), color=c_base)
            draw_callout(canvas, (base_cx - 32, 680), (base_cx - 85, 640), (base_cx - 175, 640),
                         "فقر", p=prog(st, 1.8, 0.7) * dim_f, color=c_base)

        if 4.8 <= st <= 7.5:
            qp = prog(st, 4.8, 0.5) * (1.0 - prog(st, 6.8, 0.6))
            blit_text(canvas, "؟", W * 0.5, 495, size=180, color=C_WHITE, bold=True, glow=True, opacity=qp)

        # Phase 2 (00:19 - 00:24): Timeline `تاریخ` -> `الان` with 3 Walled-Off Columns (`غذا`, `زمین`, `ثروت`)
        if st > 6.8:
            tp = prog(st, 6.8, 0.9)
            ty = 835
            draw_arrow(canvas, (620, ty), (100, ty), p=tp, color=C_WHITE, thickness=2)
            blit_text(canvas, "تاریخ", 645, ty + 36, size=22, color=C_GRAY_LIGHT, opacity=tp)
            blit_text(canvas, "الان", 100, ty + 36, size=22, color=C_WHITE, glow=True, opacity=tp)

            cols = [
                (540, 7.2, "غذا", "wheat"),
                (360, 8.3, "زمین", "land"),
                (180, 9.4, "ثروت", "vault"),
            ]
            for cx_c, t_c, lbl, kind in cols:
                cp = prog(st, t_c, 0.8)
                if cp > 0.01:
                    draw_human_icon(canvas, cx_c, 340, scale=0.88, p=cp, color=C_WHITE)
                    blit_text(canvas, "انسان قوی‌تر", cx_c, 280, size=18, color=C_GRAY_LIGHT, opacity=cp)
                    draw_arrow(canvas, (cx_c, 395), (cx_c, 545), p=cp, color=C_GRAY_LIGHT, dashed=True)

                    # Resource icon at y=650
                    ry = 650
                    if kind == "wheat":
                        draw_trim_line(canvas, (cx_c, ry + 38), (cx_c, ry - 38), p=cp, color=C_WHITE, thickness=2)
                        for k in [-20, -4, 12]:
                            cv2.ellipse(canvas, (cx_c - 10, ry + k), (10, 5), -35, 0, 360, C_WHITE, 2, cv2.LINE_AA)
                            cv2.ellipse(canvas, (cx_c + 10, ry + k), (10, 5), 35, 0, 360, C_WHITE, 2, cv2.LINE_AA)
                    elif kind == "land":
                        pts_l = [(cx_c, ry - 32), (cx_c + 42, ry), (cx_c, ry + 32), (cx_c - 42, ry)]
                        draw_trim_polyline(canvas, pts_l, p=cp, color=C_WHITE, closed=True)
                        draw_trim_line(canvas, (cx_c - 21, ry - 16), (cx_c + 21, ry + 16), p=cp, color=C_GRAY_LIGHT)
                        draw_trim_line(canvas, (cx_c + 21, ry - 16), (cx_c - 21, ry + 16), p=cp, color=C_GRAY_LIGHT)
                    else:
                        draw_money_core(canvas, cx_c, ry, r=28, p=cp, t=t, pulse=False)

                    # Wall / Fence / Safe locking around the resource!
                    wp = prog(st, t_c + 0.55, 0.55)
                    if wp > 0.01:
                        draw_trim_rect(canvas, cx_c, ry, 118, 118, p=wp, color=C_WHITE, thickness=3, fill_black=False)
                        # Lock shackle at top of wall
                        cv2.circle(canvas, (cx_c + 44, ry - 44), 7, C_WHITE, 2, cv2.LINE_AA)
                    blit_text(canvas, lbl, cx_c, ry + 92, size=24, color=C_WHITE, glow=(wp > 0.5), opacity=cp)
                    cv2.circle(canvas, (cx_c, ty), 6, C_WHITE, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------------
    # CHAPTER 3: پرونده: گاوصندوق (00:24 - 00:35)
    # ------------------------------------------------------------------------
    elif ch_idx == 3:
        st = t - 24.0
        # 20 x 10 Data Grid of 200 dots (`۸ میلیارد نفر`)
        gp = prog(st, 0.1, 1.2)
        dim_95 = ease_in_out(prog(st, 2.3, 0.8))
        gx0, gy0 = 135, 250
        dx, dy = 23, 24
        for r in range(10):
            for c in range(20):
                idx = r * 20 + c
                if idx / 200.0 <= gp:
                    px, py = gx0 + c * dx, gy0 + r * dy
                    # Top-right 2 dots = 1% (stay bright white!), remaining 99% dim at st > 2.3s
                    is_top_1pct = (r == 0 and c >= 18)
                    if is_top_1pct:
                        cv2.circle(canvas, (px, py), 5 if dim_95 > 0.5 else 4, C_WHITE, -1, cv2.LINE_AA)
                    else:
                        c_dot = int(255 * (1.0 - 0.78 * dim_95))
                        cv2.circle(canvas, (px, py), 3, (c_dot, c_dot, c_dot), -1, cv2.LINE_AA)

        blit_text(canvas, "۸ میلیارد نفر", W * 0.5, 535, size=25, color=C_WHITE if dim_95 < 0.5 else C_GRAY_LIGHT, opacity=gp)
        if st > 2.4:
            draw_callout(canvas, (gx0 + 19 * dx, gy0), (gx0 + 19 * dx + 35, gy0 - 45), (gx0 + 19 * dx - 65, gy0 - 45),
                         "۱ درصد", p=prog(st, 2.4, 0.6), color=C_WHITE, glow=True)
            draw_callout(canvas, (gx0 + 4 * dx, gy0 + 5 * dy), (gx0 - 35, gy0 + 5 * dy), (gx0 - 45, gy0 + 5 * dy),
                         "۹۵ درصد بقیه", p=prog(st, 2.8, 0.6), color=C_GRAY_LIGHT, text_size=20)

        # 00:29 (st=5.0): Row of coins at bottom (`کافی برای همه`) with dashed rays to grid
        if st > 4.8:
            cp = prog(st, 4.8, 0.8)
            lock_p = prog(st, 7.8, 0.7)
            ray_op = cp * (1.0 - lock_p)
            if ray_op > 0.02:
                for target_x in [180, 270, 360, 450, 540]:
                    draw_trim_line(canvas, (W * 0.5, 720), (target_x, 505), p=ray_op, color=C_GRAY_MID, dashed=True)

            for k in range(-2, 3):
                draw_money_core(canvas, W * 0.5 + k * 56, 765, r=22, p=cp, t=t, pulse=(lock_p < 0.5))

            if lock_p < 0.5:
                draw_callout(canvas, (W * 0.5 + 135, 765), (W * 0.5 + 175, 725), (W * 0.5 + 255, 725),
                             "کافی برای همه", p=cp, color=C_WHITE)

            # 00:32 (st=7.8): Heavy double-walled Vault (`گاوصندوق`) slams around the coins!
            if lock_p > 0.01:
                draw_trim_rect(canvas, W * 0.5, 765, 340, 145, p=lock_p, color=C_WHITE, thickness=3, fill_black=False)
                draw_trim_rect(canvas, W * 0.5, 765, 320, 125, p=lock_p, color=C_WHITE, thickness=1, fill_black=False)
                # Rotating vault wheel handle on right side of safe
                wx, wy = int(W * 0.5 + 125), 765
                cv2.circle(canvas, (wx, wy), 22, C_WHITE, 2, cv2.LINE_AA)
                for s_i in range(4):
                    ang = lock_p * math.pi + s_i * (math.pi * 0.5)
                    cv2.line(canvas, (wx, wy), (int(wx + 20 * math.cos(ang)), int(wy + 20 * math.sin(ang))), C_WHITE, 2, cv2.LINE_AA)
                blit_text(canvas, "حبس در گاوصندوق", W * 0.5, 875, size=26, color=C_WHITE, glow=True, opacity=lock_p)

    # ------------------------------------------------------------------------
    # CHAPTER 4: پرونده: انسولین (00:35 - 00:52)
    # ------------------------------------------------------------------------
    elif ch_idx == 4:
        st = t - 35.0
        if st < 11.2:
            # 00:35 - 00:46: Insulin Vial (`۱ دلار` -> `۳۰۰ دلار`) + Patient (`[حذف‌شده]`)
            vp = prog(st, 0.1, 0.9)
            vx = int(round(W * 0.5 + 38.0 * ease_in_out(prog(st, 3.6, 0.6))))
            vy = 520
            # Draw Insulin Vial (cap + neck + body + label lines)
            draw_trim_rect(canvas, vx, vy - 68, 44, 16, p=vp, color=C_WHITE, thickness=2)
            draw_trim_rect(canvas, vx, vy - 48, 28, 24, p=vp, color=C_WHITE, thickness=2)
            draw_trim_rect(canvas, vx, vy + 18, 76, 110, p=vp, color=C_WHITE, thickness=2)
            draw_trim_rect(canvas, vx, vy + 18, 56, 54, p=vp, color=C_GRAY_LIGHT, thickness=1)
            blit_text(canvas, "انسولین", vx, vy + 18, size=14, color=C_WHITE, opacity=vp)

            draw_callout(canvas, (vx + 40, vy - 25), (vx + 95, vy - 75), (vx + 255, vy - 75),
                         "سال کشف: ۱۹۲۳", p=prog(st, 0.6, 0.7), color=C_GRAY_LIGHT, text_size=20)

            hike_p = prog(st, 7.2, 0.6)
            if hike_p < 0.5:
                draw_callout(canvas, (vx + 40, vy + 45), (vx + 95, vy + 95), (vx + 265, vy + 95),
                             "قیمت اختراع: ۱ دلار", p=prog(st, 1.2, 0.7), color=C_WHITE, glow=True, text_size=20)
            else:
                # Strikethrough `۱ دلار` and pop `قیمت شما: ۳۰۰ دلار`!
                draw_callout(canvas, (vx + 40, vy + 45), (vx + 95, vy + 95), (vx + 265, vy + 95),
                             "قیمت اختراع: ۱ دلار", p=1.0, color=C_GRAY_DARK, text_size=20)
                shelf_cx = clamp(vx + 180, 115, W - 115)
                draw_trim_line(canvas, (shelf_cx - 82, vy + 73), (shelf_cx + 82, vy + 73), p=hike_p, color=C_WHITE, thickness=2)
                blit_text(canvas, "قیمت شما: ۳۰۰ دلار", shelf_cx, vy + 132, size=23, color=C_WHITE, glow=True, opacity=hike_p)

            # Patient on left (`00:39`, st > 3.8)
            if st > 3.8:
                pp = prog(st, 3.8, 0.7)
                px, py = 165, 535
                draw_human_icon(canvas, px, py, scale=1.2, p=pp, color=C_GRAY_MID if hike_p > 0.5 else C_WHITE, dashed=(hike_p > 0.5))
                blit_text(canvas, "بیمار", px, py + 72, size=21, color=C_GRAY_LIGHT, opacity=pp)
                if hike_p < 0.3:
                    draw_arrow(canvas, (vx - 50, vy + 10), (px + 42, py + 10), p=pp, color=C_WHITE, dashed=True)
                    blit_text(canvas, "نجات جان", (vx + px) * 0.5, vy - 18, size=19, color=C_WHITE, opacity=pp)

            # Middleman / Price Wall stepping between Vial & Patient at `00:42` (`st > 6.8`)
            if st > 6.8:
                mp = prog(st, 6.8, 0.6)
                mx = 305
                draw_human_icon(canvas, mx, 520, scale=1.1, p=mp, color=C_WHITE)
                draw_trim_line(canvas, (mx - 42, 420), (mx - 42, 625), p=mp, color=C_WHITE, thickness=3)
                # Timeline at top: `هدف مخترع` -> `ورود بازار`
                draw_arrow(canvas, (540, 240), (180, 240), p=mp, color=C_GRAY_MID)
                blit_text(canvas, "هدف مخترع: همه", 540, 205, size=19, color=C_GRAY_LIGHT, opacity=mp)
                blit_text(canvas, "انحصار بازار", 180, 205, size=19, color=C_WHITE, opacity=mp)
                draw_stamp(canvas, 165, 535, "حذف‌شده", st, 7.8, angle_deg=-10.0, w=165, h=56)

        else:
            # 00:46 - 00:52 (`st >= 11.2`): Zoom-out to 7x7 Grid of Medical Icons + Diagonal Strikethroughs!
            gp = prog(st, 11.2, 0.9)
            blit_text(canvas, "میلیون‌ها بیمار در سال", W * 0.5, 215, size=30, color=C_WHITE, glow=True, opacity=gp)
            gx0, gy0 = 150, 310
            step = 70
            slash_p = prog(st, 12.2, 3.8)
            for r in range(7):
                for c in range(7):
                    idx = r * 7 + c
                    cx_i, cy_i = gx0 + c * step, gy0 + r * step
                    is_center = (r == 3 and c == 3)
                    col = C_WHITE if is_center else C_GRAY_MID
                    draw_trim_rect(canvas, cx_i, cy_i, 48, 48, p=gp, color=col, thickness=2 if is_center else 1)
                    # Draw mini medical cross / pill inside cell
                    cv2.line(canvas, (cx_i - 10, cy_i), (cx_i + 10, cy_i), col, 2, cv2.LINE_AA)
                    cv2.line(canvas, (cx_i, cy_i - 10), (cx_i, cy_i + 10), col, 2, cv2.LINE_AA)
                    if not is_center and slash_p > (idx / 49.0):
                        cv2.line(canvas, (cx_i - 20, cy_i + 20), (cx_i + 20, cy_i - 20), C_WHITE, 2, cv2.LINE_AA)
            draw_callout(canvas, (gx0 + 3 * step + 24, gy0 + 3 * step - 24),
                         (gx0 + 3 * step + 85, gy0 + 3 * step - 85),
                         (gx0 + 3 * step + 175, gy0 + 3 * step - 85),
                         "انسولین", p=gp, color=C_WHITE, glow=True)

    # ------------------------------------------------------------------------
    # CHAPTER 5: پرونده: ارزش (00:52 - 01:12)
    # ------------------------------------------------------------------------
    elif ch_idx == 5:
        st = t - 52.0
        if st < 4.0:
            # 00:52 - 00:56: Hand at bottom reaching for string attached to Money Core -> string snaps!
            hp = prog(st, 0.1, 0.8)
            snap = prog(st, 2.2, 0.5)
            draw_money_core(canvas, W * 0.5, 310, r=45, p=hp, t=t)
            if snap < 0.1:
                draw_trim_line(canvas, (W * 0.5, 360), (W * 0.5, 680), p=hp, color=C_WHITE, thickness=2)
            else:
                draw_trim_line(canvas, (W * 0.5, 360), (W * 0.5 - 18 * snap, 475), p=1.0, color=C_GRAY_LIGHT, thickness=2)
                draw_trim_line(canvas, (W * 0.5 + 18 * snap, 565), (W * 0.5, 680), p=1.0, color=C_GRAY_LIGHT, thickness=2)
                xc, yc, xr = int(W * 0.5), 520, int(10 * snap)
                cv2.line(canvas, (xc - xr, yc - xr), (xc + xr, yc + xr), C_WHITE, 2, cv2.LINE_AA)
                cv2.line(canvas, (xc - xr, yc + xr), (xc + xr, yc - xr), C_WHITE, 2, cv2.LINE_AA)
            draw_human_icon(canvas, W * 0.5, 745, scale=1.2, p=hp, color=C_WHITE)
        else:
            # 00:56 - 01:12: Marionette Cross-Bar & 3 Value Icons morphing into 3 Human Controllers!
            mp = prog(st, 4.0, 0.9)
            # Bottom controlled human
            bx, by = W * 0.5, 815
            draw_human_icon(canvas, bx, by, scale=1.35, p=mp, color=C_WHITE)
            # Raising arms attached to strings
            cv2.line(canvas, (int(bx - 32), int(by + 10)), (int(bx - 65), int(by - 20)), C_WHITE, 2, cv2.LINE_AA)
            cv2.line(canvas, (int(bx + 32), int(by + 10)), (int(bx + 65), int(by - 20)), C_WHITE, 2, cv2.LINE_AA)

            # Marionette cross-bar at y=465
            draw_trim_line(canvas, (140, 465), (580, 465), p=mp, color=C_GRAY_LIGHT, thickness=3)
            draw_trim_line(canvas, (360, 435), (360, 495), p=mp, color=C_GRAY_LIGHT, thickness=3)
            # Control strings down to human
            for sx_top, sx_bot in [(190, bx - 65), (360, bx), (530, bx + 65)]:
                draw_trim_line(canvas, (sx_top, 465), (sx_bot, by - 25), p=mp, color=C_GRAY_MID, dashed=True)

            # 3 Top Nodes (at x=540 Right, x=360 Center, x=180 Left)
            morph_human = prog(st, 14.2, 0.9) # At 01:06.2 (`st=14.2`), all 3 icons morph into humans!
            nodes = [
                (540, 7.0, "نان و مسکن", "house"),
                (360, 10.0, "بودجه و اولویت", "weapon"),
                (180, 13.0, "دستمزد و ارزش کار", "contract"),
            ]
            for nx, t_trig, lbl, kind in nodes:
                np_i = prog(st, t_trig, 0.7)
                if np_i > 0.01:
                    ny = 325
                    if morph_human < 0.5:
                        ic_op = np_i * (1.0 - morph_human * 2.0)
                        c_ic = col_fade(C_WHITE, ic_op)
                        draw_trim_circle(canvas, nx, ny, 44, p=ic_op, color=c_ic, thickness=2)
                        if kind == "house":
                            pts_h = [(nx - 20, ny + 5), (nx, ny - 18), (nx + 20, ny + 5), (nx + 15, ny + 5), (nx + 15, ny + 22), (nx - 15, ny + 22), (nx - 15, ny + 5)]
                            draw_trim_polyline(canvas, pts_h, p=ic_op, color=c_ic, closed=True)
                        elif kind == "weapon":
                            draw_trim_rect(canvas, nx - 12, ny, 14, 28, p=ic_op, color=c_ic, fill_black=False)
                            draw_arrow(canvas, (nx + 12, ny + 18), (nx + 12, ny - 20), p=ic_op, color=c_ic, thickness=2)
                        else:
                            draw_trim_rect(canvas, nx, ny, 34, 44, p=ic_op, color=c_ic, fill_black=False)
                            draw_trim_line(canvas, (nx - 10, ny + 10), (nx + 12, ny + 4), p=ic_op, color=c_ic, thickness=2)
                    else:
                        hp_m = (morph_human - 0.5) * 2.0
                        draw_human_icon(canvas, nx, ny, scale=1.15, p=hp_m, color=C_WHITE, thickness=3)
                    draw_trim_line(canvas, (nx, 372), (nx, 465), p=np_i, color=C_WHITE, thickness=2)
                    blit_text(canvas, lbl, nx, 245, size=20, color=C_GRAY_LIGHT if morph_human > 0.5 else C_WHITE, opacity=np_i)

            if morph_human > 0.2:
                blit_text(canvas, "تصمیم‌گیرنده: همه‌شون انسانن", W * 0.5, 185, size=28, color=C_WHITE, glow=True, opacity=morph_human)

    # ------------------------------------------------------------------------
    # CHAPTER 6: پرونده: زمان (01:12 - 01:30)
    # ------------------------------------------------------------------------
    elif ch_idx == 6:
        st = t - 72.0
        hp = prog(st, 0.1, 0.9)
        # Side-profile human head/bust on right (`x=535, y=565`) facing left
        hx, hy = 525, 565
        draw_trim_circle(canvas, hx, hy - 35, 48, p=hp, color=C_WHITE, thickness=2)
        # Eye dot looking left
        if hp > 0.5:
            cv2.circle(canvas, (hx - 26, hy - 42), 4, C_WHITE, -1, cv2.LINE_AA)
        # Shoulders
        cv2.ellipse(canvas, (hx + 10, hy + 75), (65, 65), 0, 180, 360, col_fade(C_WHITE, hp), 2, cv2.LINE_AA)

        # 5 Left Icons (`01:16 - 01:24`, st=4.0..12.0) connected by curved dashed neural lines
        items = [
            (295, 3.8, "کِی بیدار شین", "clock"),
            (415, 6.2, "کجا زندگی کنین", "home"),
            (535, 8.5, "چی بخرین", "cart"),
            (655, 10.0, "چی نخرین", "nocart"),
            (775, 11.4, "کِی استراحت کنین", "cal"),
        ]
        for iy, t_trig, lbl, kind in items:
            ip = prog(st, t_trig, 0.65)
            if ip > 0.01:
                ix = 215
                draw_trim_line(canvas, (hx - 48, hy - 35), (ix + 38, iy), p=ip, color=C_GRAY_MID, dashed=True)
                draw_trim_rect(canvas, ix, iy, 58, 58, p=ip, color=C_WHITE, thickness=2)
                if kind == "clock":
                    draw_trim_circle(canvas, ix, iy, 18, p=ip, color=C_WHITE, thickness=2)
                    cv2.line(canvas, (ix, iy), (ix, iy - 11), C_WHITE, 2, cv2.LINE_AA)
                    cv2.line(canvas, (ix, iy), (ix + 9, iy), C_WHITE, 2, cv2.LINE_AA)
                elif kind == "home":
                    pts_h = [(ix - 16, iy + 4), (ix, iy - 14), (ix + 16, iy + 4), (ix + 12, iy + 16), (ix - 12, iy + 16)]
                    draw_trim_polyline(canvas, pts_h, p=ip, color=C_WHITE, closed=True)
                elif kind == "cart":
                    draw_trim_rect(canvas, ix, iy, 26, 18, p=ip, color=C_WHITE, fill_black=False)
                elif kind == "nocart":
                    draw_trim_rect(canvas, ix, iy, 26, 18, p=ip, color=C_GRAY_LIGHT, fill_black=False)
                    cv2.line(canvas, (ix - 18, iy - 18), (ix + 18, iy + 18), C_WHITE, 2, cv2.LINE_AA)
                else:
                    draw_trim_rect(canvas, ix, iy + 2, 28, 26, p=ip, color=C_WHITE, fill_black=False)
                    cv2.line(canvas, (ix - 14, iy - 4), (ix + 14, iy - 4), C_WHITE, 2, cv2.LINE_AA)
                blit_text(canvas, lbl, ix - 42, iy, size=21, color=C_WHITE, align="right", opacity=ip)

        # 01:24.5 (`st > 12.5`): Reveal standing human controller behind the head (`سیستمِ انسانی`)
        if st > 12.5:
            sp_sys = prog(st, 12.5, 0.9)
            draw_human_icon(canvas, 615, 485, scale=1.45, p=sp_sys, color=C_GRAY_LIGHT, thickness=2)
            # Hand resting on back of head
            draw_trim_line(canvas, (585, 495), (568, 535), p=sp_sys, color=C_WHITE, thickness=2)
            draw_callout(canvas, (615, 405), (585, 255), (455, 255),
                         "سیستمِ انسانی", p=sp_sys, color=C_WHITE, glow=True, text_size=24)
            # Live Strikethrough at bottom: `̶ز̶ن̶د̶گ̶ی̶ ̶ع̶ا̶د̶ی̶` -> `معامله عمر`
            blit_text(canvas, "«زندگی عادی»", 460, 905, size=25, color=C_GRAY_MID, opacity=sp_sys)
            draw_trim_line(canvas, (385, 905), (535, 905), p=prog(st, 13.4, 0.5), color=C_WHITE, thickness=2)

    # ------------------------------------------------------------------------
    # CHAPTER 7: پرونده: طبیعت (01:30 - 01:46)
    # ------------------------------------------------------------------------
    elif ch_idx == 7:
        st = t - 90.0
        tp = prog(st, 0.1, 0.9)
        cut_p = prog(st, 4.0, 0.7)
        # Tree in center (`x=360, y=540`)
        tree_col = C_GRAY_DARK if cut_p > 0.5 else C_WHITE
        is_dashed = (cut_p > 0.5)
        draw_trim_line(canvas, (360, 660), (360, 480), p=tp, color=tree_col, thickness=3, dashed=is_dashed)
        draw_trim_circle(canvas, 360, 435, 58, p=tp, color=tree_col, thickness=2, dashed=is_dashed)
        draw_trim_circle(canvas, 320, 465, 40, p=tp, color=tree_col, thickness=2, dashed=is_dashed)
        draw_trim_circle(canvas, 400, 465, 40, p=tp, color=tree_col, thickness=2, dashed=is_dashed)
        draw_money_core(canvas, 525, 455, r=32, p=tp * (1.0 - prog(st, 6.8, 0.5)), t=t)

        # 01:34 (`st=4.0`): Human + Saw cutting the tree!
        if 3.6 <= st <= 7.5:
            sp_saw = prog(st, 3.6, 0.6)
            draw_human_icon(canvas, 210, 585, scale=1.05, p=sp_saw, color=C_WHITE)
            sx_off = int(math.sin(st * 18.0) * 14)
            draw_trim_rect(canvas, 320 + sx_off, 585, 110, 22, p=sp_saw, color=C_WHITE, thickness=2)
            blit_text(canvas, "عامل برش: انسان", 210, 665, size=21, color=C_WHITE, glow=True, opacity=sp_saw)

        # 01:37 (`st > 7.0`): Timeline `تاریخ` + `دست انسان` placing 3 Industrial Boxes
        if st > 7.0:
            hp_t = prog(st, 7.0, 0.8)
            ty = 830
            draw_arrow(canvas, (620, ty), (100, ty), p=hp_t, color=C_WHITE, thickness=2)
            blit_text(canvas, "تاریخ", 645, ty + 35, size=21, color=C_GRAY_LIGHT, opacity=hp_t)

            # Top Human Hand descending
            blit_text(canvas, "دستِ انسان", W * 0.5, 215, size=26, color=C_WHITE, glow=True, opacity=hp_t)
            draw_trim_rect(canvas, W * 0.5, 265, 140, 42, p=hp_t, color=C_WHITE, thickness=2)

            boxes = [
                (535, 7.5, "کارخانه به جای جنگل"),
                (360, 9.2, "نفت و معدن به جای رودخانه"),
                (185, 10.8, "برج و شهرسازی ..."),
            ]
            for bx, t_b, lbl in boxes:
                bp = prog(st, t_b, 0.7)
                if bp > 0.01:
                    by = int(755 - (1.0 - ease_out(bp)) * 180)
                    draw_arrow(canvas, (W * 0.5, 290), (bx, by - 65), p=bp, color=C_GRAY_MID, dashed=True)
                    draw_trim_rect(canvas, bx, by, 125, 110, p=bp, color=C_WHITE, thickness=2)
                    # Smokestack / chimney on top of industrial box
                    draw_trim_rect(canvas, bx - 25, by - 72, 22, 34, p=bp, color=C_WHITE, thickness=2)
                    draw_trim_rect(canvas, bx + 15, by - 64, 18, 20, p=bp, color=C_GRAY_LIGHT, thickness=2)
                    blit_text(canvas, lbl, bx, ty + 42 if bx != 360 else ty + 86, size=18, color=C_WHITE, opacity=bp)

    # ------------------------------------------------------------------------
    # CHAPTER 8: پرونده: امضا (01:46 - 02:00)
    # ------------------------------------------------------------------------
    elif ch_idx == 8:
        st = t - 106.0
        if st < 9.0:
            # 01:46 - 01:55: Command box (`دستور انسان`) -> shatters brick wall (`قانون اخلاقی`) -> Money Core
            cp = prog(st, 0.1, 0.8)
            draw_trim_rect(canvas, W * 0.5, 290, 260, 95, p=cp, color=C_WHITE, thickness=2)
            blit_text(canvas, "« دستور انسان »", W * 0.5, 290, size=25, color=C_WHITE, glow=True, opacity=cp)
            draw_money_core(canvas, W * 0.5, 780, r=46, p=cp, t=t)

            # Brick wall `قانون اخلاقی` at y=535
            wp = prog(st, 4.2, 0.6)
            smash_p = prog(st, 5.4, 0.8)
            if wp > 0.01:
                for r_i in range(2):
                    for c_i in range(5):
                        bx = 200 + c_i * 80 + (r_i % 2) * 25
                        by = 520 + r_i * 34
                        # Center bricks scatter outward when smashed at st > 5.4!
                        if smash_p > 0.01 and 1 <= c_i <= 3:
                            dx_s = (c_i - 2) * 95 * ease_out(smash_p) + 35 * smash_p
                            dy_s = (r_i - 0.5) * 110 * ease_out(smash_p)
                            col_b = C_GRAY_DARK
                        else:
                            dx_s, dy_s = 0, 0
                            col_b = C_WHITE
                        draw_trim_rect(canvas, bx + dx_s, by + dy_s, 72, 28, p=wp, color=col_b, thickness=2)
                blit_text(canvas, "قانون اخلاقی", 115, 535, size=20, color=C_GRAY_LIGHT, opacity=wp)

            # Dashed command beam shooting from top box down to Money Core
            beam_p = prog(st, 1.2, 1.0)
            draw_arrow(canvas, (W * 0.5, 342), (W * 0.5, 720), p=beam_p, color=C_WHITE, thickness=2, dashed=True)
        else:
            # 01:55 - 02:00 (`st >= 9.0`): Fan of 6 Contract Sheets + Live Handwritten `امضای انسانی`!
            fp = prog(st, 9.0, 0.9)
            cx, cy = W * 0.5, 560
            for i in range(5, -1, -1):
                ang = (i - 2.5) * 8.0 * ease_out(fp)
                rad = math.radians(ang)
                ox = cx + math.sin(rad) * 45
                oy = cy - (1.0 - math.cos(rad)) * 25
                col_c = C_WHITE if i == 0 else C_GRAY_DARK
                draw_trim_rect(canvas, ox, oy, 290, 380, p=fp, color=col_c, thickness=2)
                if i == 0:
                    blit_text(canvas, "قرارداد رسمی", ox, oy - 135, size=24, color=C_WHITE, opacity=fp)
                    for l_i in range(4):
                        ly = oy - 65 + l_i * 34
                        draw_trim_line(canvas, (ox - 105, ly), (ox + 105, ly), p=fp, color=C_GRAY_MID, thickness=2)
                    blit_text(canvas, "امضای انسانی:", ox + 105, oy + 78, size=21, color=C_WHITE, align="right", opacity=fp)
                    # Live cursive vector signature (`01:56.5`)
                    sig_p = prog(st, 10.2, 1.4)
                    sig_pts = [
                        (ox + 55, oy + 142), (ox + 15, oy + 108), (ox - 10, oy + 148),
                        (ox - 45, oy + 112), (ox - 80, oy + 140), (ox - 105, oy + 125), (ox + 10, oy + 156)
                    ]
                    draw_trim_polyline(canvas, sig_pts, p=sig_p, color=C_WHITE, thickness=3)

    # ------------------------------------------------------------------------
    # CHAPTER 9: پرونده: آینه (02:00 - 02:14)
    # ------------------------------------------------------------------------
    elif ch_idx == 9:
        st = t - 120.0
        if st < 7.8:
            # 02:00 - 02:07.8: 14 Orbiting Persian Words around Money Core -> Collapsing into `صورت‌حساب شما`
            words = ["رشوه", "جنگ", "انحصار", "تورم", "فقر", "تحریم", "ربا",
                     "اختلاس", "احتکار", "صدقه", "دیه", "معامله", "قرارداد", "ارث"]
            collapse_p = ease_in_out(prog(st, 5.2, 1.0))
            cx, cy = W * 0.5, 510
            if collapse_p < 0.85:
                draw_money_core(canvas, cx, cy, r=38, p=prog(st, 0.1, 0.7) * (1.0 - collapse_p), t=t)
                for idx, w_str in enumerate(words):
                    ring = 1 if idx < 7 else 2
                    r_orb = (145 if ring == 1 else 235) * (1.0 - 0.65 * collapse_p)
                    ang = t * (0.55 if ring == 1 else -0.38) + idx * (2.0 * math.pi / 7.0)
                    wx = cx + r_orb * math.cos(ang)
                    wy = cy + r_orb * math.sin(ang) * 0.85
                    wp = prog(st, 0.15 * idx, 0.6) * (1.0 - collapse_p)
                    blit_text(canvas, w_str, wx, wy, size=21 if ring == 1 else 18,
                              color=C_WHITE if ring == 1 else C_GRAY_LIGHT, opacity=wp)

            # Observer human at bottom-right (`st > 3.5`)
            if st > 3.5:
                op_h = prog(st, 3.5, 0.7)
                draw_human_icon(canvas, 585, 835, scale=1.1, p=op_h, color=C_WHITE)
                draw_trim_line(canvas, (555, 795), (425, 625), p=op_h, color=C_GRAY_MID, dashed=True)

            # Ledger card `صورت‌حساب شما` (`st > 5.4`)
            if collapse_p > 0.25:
                lp = (collapse_p - 0.25) / 0.75
                draw_trim_rect(canvas, cx, cy, 310, 220, p=lp, color=C_WHITE, thickness=2)
                blit_text(canvas, "صورت‌حسابِ خودِ شما", cx, cy - 65, size=26, color=C_WHITE, glow=True, opacity=lp)
                for r_i in range(3):
                    draw_trim_line(canvas, (cx - 110, cy + r_i * 35), (cx + 110, cy + r_i * 35), p=lp, color=C_GRAY_MID, dashed=True)
        else:
            # 02:07.8 - 02:14 (`st >= 7.8`): Standing Oval Mirror (`آینه قدی`) -> Reflection flips from Money to HUMAN!
            mp = prog(st, 7.8, 0.9)
            mx, my = W * 0.5, 495
            # Oval mirror frame & stand
            cv2.ellipse(canvas, (int(mx), int(my)), (145, 225), 0, 0, int(360 * mp), C_WHITE, 3, cv2.LINE_AA)
            cv2.ellipse(canvas, (int(mx), int(my)), (133, 213), 0, 0, int(360 * mp), C_GRAY_DARK, 1, cv2.LINE_AA)
            draw_trim_line(canvas, (mx - 155, my + 245), (mx + 155, my + 245), p=mp, color=C_WHITE, thickness=3)
            draw_trim_line(canvas, (mx, my + 225), (mx, my + 245), p=mp, color=C_WHITE, thickness=3)

            # Small Money Core standing in front of the mirror at bottom
            draw_money_core(canvas, mx, 835, r=32, p=mp, t=t)
            draw_trim_line(canvas, (mx, 790), (mx, 725), p=mp, color=C_GRAY_MID, dashed=True)

            # Inside Mirror Reflection:
            # Before `02:10.8` (`st < 10.8`): Reflection is Money Core
            # At `02:10.8` (`st >= 10.8`): Reflection transforms into glowing HUMAN silhouette (`خودِ واقعی شما`)!
            flip_p = prog(st, 10.8, 0.55)
            if flip_p < 0.5:
                draw_money_core(canvas, mx, my, r=56, p=mp * (1.0 - flip_p * 2.0), t=t, color=C_GRAY_LIGHT)
                blit_text(canvas, "ظاهر: پول", mx, my + 125, size=22, color=C_GRAY_LIGHT, opacity=mp * (1.0 - flip_p * 2.0))
            else:
                hp_ref = (flip_p - 0.5) * 2.0
                draw_human_icon(canvas, mx, my - 10, scale=2.1, p=hp_ref, color=C_WHITE, thickness=3)
                blit_text(canvas, "خودِ واقعی شما", mx, my + 135, size=26, color=C_WHITE, glow=True, opacity=hp_ref)

    # ------------------------------------------------------------------------
    # CHAPTER 10: پرونده: کارنامه (02:14 - 02:28)
    # ------------------------------------------------------------------------
    elif ch_idx == 10:
        st = t - 134.0
        if st < 10.8:
            sp_sc = prog(st, 0.1, 0.9)
            # 6 Icons dropping onto the right pan (`انسان`) between st=1.2s and st=5.2s
            drop_times = [1.2, 1.9, 2.6, 3.3, 4.0, 4.7]
            drop_labels = ["انسولین", "گاوصندوق", "نان و مسکن", "بریدن طبیعت", "امضای جنگ", "معامله عمر"]
            n_landed = sum(1 for dt in drop_times if st >= dt + 0.45)
            tilt_prog = ease_out(clamp((st - 1.5) / 4.0))
            # Tilt angle: right pan (`انسان`) goes DOWN (+22 deg), left pan (`پول`) goes UP!
            tilt_rad = math.radians(22.0 * tilt_prog)

            cx, cy = W * 0.5, 440
            # Central pillar & base of the Balance Scale
            draw_trim_line(canvas, (cx, cy - 40), (cx, 850), p=sp_sc, color=C_WHITE, thickness=3)
            draw_trim_line(canvas, (cx - 110, 850), (cx + 110, 850), p=sp_sc, color=C_WHITE, thickness=3)
            cv2.circle(canvas, (int(cx), int(cy)), 9, C_WHITE, -1, cv2.LINE_AA)

            # Tilted Beam (`arm_r = 210`)
            arm_r = 210
            lx = cx - arm_r * math.cos(tilt_rad)
            ly = cy - arm_r * math.sin(tilt_rad)
            rx = cx + arm_r * math.cos(tilt_rad)
            ry = cy + arm_r * math.sin(tilt_rad)
            draw_trim_line(canvas, (lx, ly), (rx, ry), p=sp_sc, color=C_WHITE, thickness=3)

            # Left Pan (`پول` + `«یه تکه کاغذ»` — rises high into the air!)
            l_pany = ly + 115
            draw_trim_line(canvas, (lx, ly), (lx - 55, l_pany), p=sp_sc, color=C_GRAY_LIGHT, thickness=2)
            draw_trim_line(canvas, (lx, ly), (lx + 55, l_pany), p=sp_sc, color=C_GRAY_LIGHT, thickness=2)
            draw_trim_line(canvas, (lx - 75, l_pany), (lx + 75, l_pany), p=sp_sc, color=C_WHITE, thickness=3)
            blit_text(canvas, "پول", lx, l_pany + 35, size=24, color=C_WHITE, opacity=sp_sc)
            # Feather-light dashed box `«یه تکه کاغذ»` on left pan
            paper_glow = (st > 5.2)
            draw_trim_rect(canvas, lx, l_pany - 32, 138, 48, p=sp_sc, color=C_WHITE, thickness=2, dashed=True)
            blit_text(canvas, "«یه تکه کاغذ»", lx, l_pany - 32, size=20, color=C_WHITE, glow=paper_glow, opacity=sp_sc)

            # Right Pan (`انسان` — weighed down heavily by the 6 icons!)
            r_pany = ry + 115
            draw_trim_line(canvas, (rx, ry), (rx - 65, r_pany), p=sp_sc, color=C_GRAY_LIGHT, thickness=2)
            draw_trim_line(canvas, (rx, ry), (rx + 65, r_pany), p=sp_sc, color=C_GRAY_LIGHT, thickness=2)
            draw_trim_line(canvas, (rx - 90, r_pany), (rx + 90, r_pany), p=sp_sc, color=C_WHITE, thickness=3)
            blit_text(canvas, "کارنامه انسان", rx, r_pany + 38, size=24, color=C_WHITE, glow=True, opacity=sp_sc)

            # Render falling & stacked evidence blocks on the Right Pan (`انسان`)
            for idx, (dt, dlbl) in enumerate(zip(drop_times, drop_labels)):
                if st >= dt:
                    dp = ease_out(prog(st, dt, 0.45))
                    col_offset = (idx % 2 - 0.5) * 92
                    row_idx = idx // 2
                    target_x = rx + col_offset
                    target_y = r_pany - 24 - row_idx * 44
                    cur_y = 140 + (target_y - 140) * dp
                    draw_trim_rect(canvas, target_x, cur_y, 86, 38, p=1.0, color=C_WHITE, thickness=2)
                    blit_text(canvas, dlbl, target_x, cur_y, size=15, color=C_WHITE, opacity=dp)
        else:
            # 02:24.8 - 02:28.0: Clean Minimalist Outro Emblem Pulse
            op_out = prog(st, 10.8, 0.8) * (1.0 - prog(st, 13.0, 0.9))
            draw_money_core(canvas, W * 0.5, H * 0.46, r=64, p=op_out, t=t)
            blit_text(canvas, "محاکمه انسان از زبانِ پول", W * 0.5, H * 0.58, size=28, color=C_WHITE, glow=True, opacity=op_out)


# ============================================================================
# 6. SELECTIVE OPTICAL BLOOM COMPOSITOR
# ============================================================================
def apply_optical_bloom(canvas):
    """
    Fast broadcast-grade monochrome optical glow on bright white strokes (>200).
    """
    bright = np.where(canvas > 210, canvas, 0)
    small = cv2.resize(bright, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (15, 15), 4.5)
    up = cv2.resize(blur, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(canvas.astype(np.uint16) + (up.astype(np.uint16) * 45) // 100, 0, 255).astype(np.uint8)

def render_frame(t):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    active_ch = CHAPTER_META[-1]
    for ch in CHAPTER_META:
        if ch[1] <= t <= ch[2]:
            active_ch = ch
            break
    ch_idx, t0, t1, ch_title = active_ch
    render_chapter_scene(canvas, t, ch_idx)
    canvas = apply_optical_bloom(canvas)
    draw_persistent_hud(canvas, t, ch_idx, ch_title)
    return canvas


# ============================================================================
# 7. PROCEDURAL DARK AMBIENT + PIANO + MICRO-SFX AUDIO SYNTHESIZER (148s)
# ============================================================================
def synthesize_audio_wav(wav_path):
    sr = 44100
    n_samples = int(DURATION * sr)
    t_arr = np.arange(n_samples, dtype=np.float32) / sr

    # 1. Dark Ambient Cinematic Drone (C2=65.4Hz, G2=98.0Hz, Eb3=155.6Hz, D3=146.8Hz)
    lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.08 * t_arr)
    drone = (
        0.11 * np.sin(2.0 * np.pi * 65.41 * t_arr) +
        0.07 * np.sin(2.0 * np.pi * 98.00 * t_arr + 0.3 * np.sin(2.0 * np.pi * 0.15 * t_arr)) +
        0.04 * lfo * np.sin(2.0 * np.pi * 155.56 * t_arr) +
        0.03 * (1.0 - lfo) * np.sin(2.0 * np.pi * 146.83 * t_arr)
    )

    audio = drone.copy()

    def add_tone(t_sec, freq, dur, amp=0.14, decay=4.5, harmonics=True):
        i0 = int(t_sec * sr)
        n = int(dur * sr)
        if i0 < 0 or i0 >= n_samples:
            return
        n = min(n, n_samples - i0)
        lt = np.arange(n, dtype=np.float32) / sr
        env = np.exp(-decay * lt) * np.minimum(1.0, lt / 0.012)
        sig = np.sin(2.0 * np.pi * freq * lt)
        if harmonics:
            sig += 0.35 * np.sin(2.0 * np.pi * (freq * 2.0) * lt) + 0.12 * np.sin(2.0 * np.pi * (freq * 3.0) * lt)
        audio[i0:i0 + n] += amp * env * sig

    def add_stamp_slam(t_sec):
        # Heavy 62Hz sub impact + crisp transient click
        add_tone(t_sec, 58.0, 0.45, amp=0.38, decay=9.0, harmonics=False)
        add_tone(t_sec, 145.0, 0.18, amp=0.22, decay=18.0, harmonics=True)

    def add_tick(t_sec, freq=2600.0, amp=0.07):
        add_tone(t_sec, freq, 0.035, amp=amp, decay=85.0, harmonics=False)

    # 2. Melancholic Sparse Piano Notes at Chapter Transitions & Turning Points
    piano_notes = [
        (0.1, 130.81), (4.0, 155.56), (6.0, 146.83), (8.0, 130.81),
        (12.0, 116.54), (17.0, 155.56), (19.0, 130.81),
        (24.0, 98.00), (29.0, 130.81), (32.0, 103.83),
        (35.0, 130.81), (39.0, 155.56), (42.0, 123.47), (46.0, 98.00),
        (52.0, 130.81), (56.0, 146.83), (68.0, 103.83),
        (72.0, 116.54), (76.0, 130.81), (84.5, 98.00),
        (90.0, 130.81), (94.0, 103.83), (101.0, 98.00),
        (106.0, 130.81), (111.0, 123.47), (115.0, 103.83),
        (120.0, 130.81), (123.5, 155.56), (130.8, 196.00), (131.0, 98.00),
        (134.0, 130.81), (139.0, 103.83), (144.8, 130.81),
    ]
    for tn, fn in piano_notes:
        add_tone(tn, fn, 2.6, amp=0.13, decay=1.8, harmonics=True)

    # 3. Synchronized Micro-SFX across all 10 Chapters
    for s0, s1, _ in SUBTITLES:
        add_tick(s0, freq=2200.0, amp=0.06)
        add_tick((s0 + s1) * 0.5, freq=2800.0, amp=0.04)

    # Stamp slams (`مستند`, `انجام دادید`, `حذف‌شده`, vault lock, wall shatter, scale drops)
    for ts in [6.5, 9.8, 20.5, 21.8, 23.0, 31.8, 42.8, 94.0, 111.4, 140.2]:
        add_stamp_slam(ts)

    # Scale icon drops in Chapter 10 (135.2s .. 138.7s)
    for idx, td in enumerate([135.2, 135.9, 136.6, 137.3, 138.0, 138.7]):
        add_tone(td + 0.4, 420.0 - idx * 32.0, 0.25, amp=0.16, decay=14.0, harmonics=True)

    # Fade in first 0.5s and fade out last 2.0s
    fade_in_n = int(0.5 * sr)
    fade_out_n = int(2.0 * sr)
    audio[:fade_in_n] *= np.linspace(0.0, 1.0, fade_in_n)
    audio[-fade_out_n:] *= np.linspace(1.0, 0.0, fade_out_n)

    audio_i16 = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    print(f"Synthesized 148s soundtrack to {wav_path}")


# ============================================================================
# 8. QA DIAGNOSTIC GRID & FULL 148-SECOND VIDEO RENDER PIPELINE
# ============================================================================
def render_qa_grid(out_jpg="/tmp/money_trial_10chapters_qa_grid.jpg"):
    # Sample the hero moment of all 10 chapters!
    hero_times = [10.8, 22.8, 33.8, 44.5, 69.5, 86.5, 103.5, 118.0, 132.5, 141.5]
    row1, row2 = [], []
    for idx, t in enumerate(hero_times):
        fr = render_frame(t)
        small = cv2.resize(fr, (270, 480), interpolation=cv2.INTER_AREA)
        if idx < 5:
            row1.append(small)
        else:
            row2.append(small)
    grid = np.vstack([np.hstack(row1), np.hstack(row2)])
    Image.fromarray(grid).save(out_jpg, quality=95)
    print(f"Saved 10-chapter QA grid to {out_jpg}")

def render_full_video():
    wav_path = "/tmp/money_trial_audio.wav"
    synthesize_audio_wav(wav_path)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-i", wav_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        VIDEO_OUT
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f_idx in range(TOTAL_FRAMES):
        t = f_idx / float(FPS)
        fr = render_frame(t)
        proc.stdin.write(fr.tobytes())
        if f_idx % 600 == 0:
            print(f"  Rendered frame {f_idx}/{TOTAL_FRAMES} (t={t:.1f}s)")
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg exited with code {proc.returncode}")
    print(f"Successfully rendered full 148s video to {VIDEO_OUT}")

if __name__ == "__main__":
    import sys
    if "--qa-only" in sys.argv:
        render_qa_grid()
    else:
        render_qa_grid()
        render_full_video()
