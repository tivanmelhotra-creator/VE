"""
vector_motion_library.py (V3 — Soft-Edged, Multi-Layered Complex Graphic Design Edition)
========================================================================================
موتور کامل و کتابخانه آماده موشن‌گرافیک وکتور استودیویی (به سبک @ENSAN.BESHAM و Motion.dev)
نسخه V3 با ۳ ارتقای گرافیکی اساسی:
1. لبه‌های نرم و گوشه‌های گرد خودکار (`Auto-Corner Filleting & Rounded Bezier Geometry`):
   - تبدیل خودکار گوشه‌های تیز ۹۰ درجه (`L / H / V`) به قوس‌های نرم بزیه (`Corner Radius = r`) با تابع `round_svg_path_corners`.
   - استفاده از مستطیل‌های کپسولی و گوشه‌گرد (`draw_rounded_rect` با `rx = 14..26px`) در تمام کارت‌ها، اسناد، مهرها، اسلایدرها و باکس‌ها.
2. طراحی گرافیکی چندلایه و پرجزئیات (`Multi-Layer Detailed Vector Art`):
   - هر آیکون و ویجت به جای یک خط ساده، از ۲ لایه تشکیل شده است:
     * لایه اول (`primary` با رنگ سفید درخشان `#FFFFFF` و ضخامت `2.4px`): فرم اصلی با منحنی‌های نرم و ارگانیک.
     * لایه دوم (`detail` با رنگ خاکستری نقره‌ای `#787878` و ضخامت `1.5px`): جزئیات ظریف داخلی، خطوط هاشور، بازتاب شیشه، مدارها، چرخ‌دنده‌ها و حلقه‌های هم‌مرکز که حس گرافیکی عمیق و حرفه‌ای به تصویر می‌دهند.
3. مجموعه کامل ۱۴ ویجت آماده سطح بالا (`High-Level Studio Widgets`) + ایمپورتر مستقیم `.svg` و `.json` (Lottie).
"""

import os
import glob
import json
import math
import re
import xml.etree.ElementTree as ET
import cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_FA_BOLD = "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"
FONT_FA_REG = "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"
FONT_EN_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

C_WHITE = (1.0, 1.0, 1.0, 1.0)
C_DETAIL = (0.52, 0.52, 0.55, 0.95)
C_GRAY_LIGHT = (0.68, 0.68, 0.70, 1.0)
C_GRAY_MID = (0.42, 0.42, 0.44, 1.0)
C_GRAY_DARK = (0.24, 0.24, 0.26, 1.0)

_FONT_CACHE = {}
_TEXT_SURFACE_CACHE = {}


# ============================================================================
# 1. MOTION.DEV STYLE PHYSICS, EASING & 2D CAMERA RIG
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

def spring(t, t0=0.0, dur=0.55, freq=6.2, decay=5.2):
    if t <= t0:
        return 0.0
    p = (t - t0) / max(dur, 1e-4)
    if p >= 2.5:
        return 1.0
    return 1.0 - math.exp(-decay * p) * math.cos(freq * p)

def stagger(t, t0, idx, step=0.12, dur=0.5):
    return ease_out(prog(t, t0 + idx * step, dur))

class Camera2D:
    def __init__(self, width=720, height=1280):
        self.w = width
        self.h = height
        self.cx = width * 0.5
        self.cy = height * 0.5
        self.zoom = 1.0

    def set_view(self, cx, cy, zoom=1.0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.zoom = float(zoom)

    def interpolate_views(self, view_a, view_b, p):
        ep = ease_in_out(p)
        self.cx = view_a[0] + (view_b[0] - view_a[0]) * ep
        self.cy = view_a[1] + (view_b[1] - view_a[1]) * ep
        self.zoom = view_a[2] + (view_b[2] - view_a[2]) * ep

    def apply(self, ctx):
        ctx.translate(self.w * 0.5, self.h * 0.5)
        ctx.scale(self.zoom, self.zoom)
        ctx.translate(-self.cx, -self.cy)


# ============================================================================
# 2. SOFT-CORNERED BEZIER GEOMETRY & AUTO-FILLET SVG PATH ENGINE
# ============================================================================
def rounded_rect_svg(x0, y0, w, h, r=14.0):
    """
    Generates a normalized SVG path string for a rectangle with smooth Cubic Bezier rounded corners.
    """
    r = min(float(r), w * 0.48, h * 0.48)
    x1, y1 = x0 + w, y0 + h
    k = 0.55228475 * r
    return (
        f"M {x0+r:.2f} {y0:.2f} "
        f"H {x1-r:.2f} C {x1-r+k:.2f} {y0:.2f} {x1:.2f} {y0+r-k:.2f} {x1:.2f} {y0+r:.2f} "
        f"V {y1-r:.2f} C {x1:.2f} {y1-r+k:.2f} {x1-r+k:.2f} {y1:.2f} {x1-r:.2f} {y1:.2f} "
        f"H {x0+r:.2f} C {x0+r-k:.2f} {y1:.2f} {x0:.2f} {y1-r+k:.2f} {x0:.2f} {y1-r:.2f} "
        f"V {y0+r:.2f} C {x0:.2f} {y0+r-k:.2f} {x0+r-k:.2f} {y0:.2f} {x0+r:.2f} {y0:.2f} Z"
    )

def draw_rounded_rect(ctx, cx, cy, w, h, r=18.0, progress=1.0, color=C_WHITE, stroke_w=2.3,
                      dashed=False, fill_black=True, double_border=False):
    """
    Draws a soft-cornered rectangle centered at `(cx, cy)` with progressive Trim-Path reveal.
    """
    if progress <= 0.005:
        return
    d_outer = rounded_rect_svg(cx - w * 0.5, cy - h * 0.5, w, h, r=r)
    draw_svg_path(ctx, d_outer, cx=50.0, cy=50.0, scale=1.0, progress=progress,
                  color=color, stroke_w=stroke_w, dashed=dashed, view_box=100.0,
                  fill_black=fill_black, auto_fillet=0.0)
    if double_border and progress > 0.35:
        inset = 6.0
        d_inner = rounded_rect_svg(cx - w * 0.5 + inset, cy - h * 0.5 + inset,
                                   w - inset * 2, h - inset * 2, r=max(4.0, r - inset * 0.8))
        draw_svg_path(ctx, d_inner, cx=50.0, cy=50.0, scale=1.0, progress=(progress - 0.35) / 0.65,
                      color=(color[0], color[1], color[2], color[3] * 0.55),
                      stroke_w=1.2, dashed=False, view_box=100.0, fill_black=False, auto_fillet=0.0)

def _tokenize_svg_path(d_str):
    return re.findall(r"[MLHVCSQTAZmlhvcsqtaz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d_str)

def _build_filleted_poly(pts, closed, radius=4.5):
    """
    Takes a list of polygon vertices `pts` and replaces every sharp corner with a smooth Quadratic/Cubic Bezier arc!
    """
    n = len(pts)
    if n < 3 or radius <= 0.2:
        return pts, []
    curves = []
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            curves.append(("pt", pts[i]))
            continue
        p_prev = pts[(i - 1) % n]
        p_cur = pts[i]
        p_next = pts[(i + 1) % n]
        v1x, v1y = p_prev[0] - p_cur[0], p_prev[1] - p_cur[1]
        v2x, v2y = p_next[0] - p_cur[0], p_next[1] - p_cur[1]
        d1 = math.hypot(v1x, v1y)
        d2 = math.hypot(v2x, v2y)
        if d1 < 1e-3 or d2 < 1e-3:
            curves.append(("pt", p_cur))
            continue
        r_eff = min(radius, d1 * 0.38, d2 * 0.38)
        start_pt = (p_cur[0] + (v1x / d1) * r_eff, p_cur[1] + (v1y / d1) * r_eff)
        end_pt = (p_cur[0] + (v2x / d2) * r_eff, p_cur[1] + (v2y / d2) * r_eff)
        curves.append(("fillet", start_pt, p_cur, end_pt))
    return pts, curves

def draw_svg_path(ctx, d_str, cx=50.0, cy=50.0, scale=1.0, progress=1.0,
                  color=C_WHITE, stroke_w=2.2, dashed=False, view_box=100.0,
                  fill_black=False, auto_fillet=3.8):
    """
    Renders any SVG path string (`M, L, H, V, C, Q, Z`) centered at `(cx, cy)`.
    When `auto_fillet > 0`, any sharp `L / H / V` corners are automatically softened with Bezier fillets!
    """
    if progress <= 0.002:
        return
    ctx.save()
    half = view_box * 0.5
    ctx.translate(cx, cy)
    ctx.scale(scale, scale)
    ctx.translate(-half, -half)

    ctx.new_path()
    tokens = _tokenize_svg_path(d_str)
    # Check if path is pure polygonal (`M, L, H, V, Z` without `C/Q`) so we can auto-fillet sharp corners!
    has_curves = any(t in ("C", "c", "Q", "q") for t in tokens)

    if not has_curves and auto_fillet > 0.1:
        # Parse into sub-polylines and apply automatic corner rounding!
        subpaths = []
        cur_poly = []
        i = 0
        cmd = "M"
        cur_x, cur_y = 0.0, 0.0
        while i < len(tokens):
            tok = tokens[i]
            if tok.isalpha():
                cmd = tok
                i += 1
                if cmd in ("Z", "z"):
                    if cur_poly:
                        subpaths.append((cur_poly, True))
                        cur_poly = []
                    continue
            if i >= len(tokens):
                break
            if cmd == "M":
                if cur_poly:
                    subpaths.append((cur_poly, False))
                cur_x, cur_y = float(tokens[i]), float(tokens[i+1])
                cur_poly = [(cur_x, cur_y)]
                cmd = "L"
                i += 2
            elif cmd == "L":
                cur_x, cur_y = float(tokens[i]), float(tokens[i+1])
                cur_poly.append((cur_x, cur_y))
                i += 2
            elif cmd == "H":
                cur_x = float(tokens[i])
                cur_poly.append((cur_x, cur_y))
                i += 1
            elif cmd == "V":
                cur_y = float(tokens[i])
                cur_poly.append((cur_x, cur_y))
                i += 1
            else:
                i += 1
        if cur_poly:
            subpaths.append((cur_poly, False))

        for poly, is_closed in subpaths:
            if len(poly) < 2:
                continue
            if len(poly) == 2:
                ctx.move_to(poly[0][0], poly[0][1])
                ctx.line_to(poly[1][0], poly[1][1])
            else:
                _, items = _build_filleted_poly(poly, is_closed, radius=auto_fillet)
                first = items[0]
                if first[0] == "pt":
                    ctx.move_to(first[1][0], first[1][1])
                else:
                    ctx.move_to(first[1][0], first[1][1])
                    p0, ctrl, p1 = first[1], first[2], first[3]
                    ctx.curve_to(p0[0] + (2/3)*(ctrl[0]-p0[0]), p0[1] + (2/3)*(ctrl[1]-p0[1]),
                                 p1[0] + (2/3)*(ctrl[0]-p1[0]), p1[1] + (2/3)*(ctrl[1]-p1[1]), p1[0], p1[1])
                for it in items[1:]:
                    if it[0] == "pt":
                        ctx.line_to(it[1][0], it[1][1])
                    else:
                        p0, ctrl, p1 = it[1], it[2], it[3]
                        ctx.line_to(p0[0], p0[1])
                        ctx.curve_to(p0[0] + (2/3)*(ctrl[0]-p0[0]), p0[1] + (2/3)*(ctrl[1]-p0[1]),
                                     p1[0] + (2/3)*(ctrl[0]-p1[0]), p1[1] + (2/3)*(ctrl[1]-p1[1]), p1[0], p1[1])
                if is_closed:
                    ctx.close_path()
    else:
        i = 0
        cmd = "M"
        cur_x, cur_y = 0.0, 0.0
        start_x, start_y = 0.0, 0.0
        while i < len(tokens):
            tok = tokens[i]
            if tok.isalpha():
                cmd = tok
                i += 1
                if cmd in ("Z", "z"):
                    ctx.close_path()
                    cur_x, cur_y = start_x, start_y
                    continue
            if i >= len(tokens):
                break
            if cmd == "M":
                x, y = float(tokens[i]), float(tokens[i+1])
                ctx.move_to(x, y)
                cur_x, cur_y = x, y
                start_x, start_y = x, y
                cmd = "L"
                i += 2
            elif cmd == "m":
                x, y = cur_x + float(tokens[i]), cur_y + float(tokens[i+1])
                ctx.move_to(x, y)
                cur_x, cur_y = x, y
                start_x, start_y = x, y
                cmd = "l"
                i += 2
            elif cmd == "L":
                x, y = float(tokens[i]), float(tokens[i+1])
                ctx.line_to(x, y)
                cur_x, cur_y = x, y
                i += 2
            elif cmd == "l":
                x, y = cur_x + float(tokens[i]), cur_y + float(tokens[i+1])
                ctx.line_to(x, y)
                cur_x, cur_y = x, y
                i += 2
            elif cmd == "H":
                x = float(tokens[i])
                ctx.line_to(x, cur_y)
                cur_x = x
                i += 1
            elif cmd == "h":
                x = cur_x + float(tokens[i])
                ctx.line_to(x, cur_y)
                cur_x = x
                i += 1
            elif cmd == "V":
                y = float(tokens[i])
                ctx.line_to(cur_x, y)
                cur_y = y
                i += 1
            elif cmd == "v":
                y = cur_y + float(tokens[i])
                ctx.line_to(cur_x, y)
                cur_y = y
                i += 1
            elif cmd == "C":
                x1, y1 = float(tokens[i]), float(tokens[i+1])
                x2, y2 = float(tokens[i+2]), float(tokens[i+3])
                x, y = float(tokens[i+4]), float(tokens[i+5])
                ctx.curve_to(x1, y1, x2, y2, x, y)
                cur_x, cur_y = x, y
                i += 6
            elif cmd == "c":
                x1, y1 = cur_x + float(tokens[i]), cur_y + float(tokens[i+1])
                x2, y2 = cur_x + float(tokens[i+2]), cur_y + float(tokens[i+3])
                x, y = cur_x + float(tokens[i+4]), cur_y + float(tokens[i+5])
                ctx.curve_to(x1, y1, x2, y2, x, y)
                cur_x, cur_y = x, y
                i += 6
            elif cmd == "Q":
                x1, y1 = float(tokens[i]), float(tokens[i+1])
                x, y = float(tokens[i+2]), float(tokens[i+3])
                cx1 = cur_x + (2.0 / 3.0) * (x1 - cur_x)
                cy1 = cur_y + (2.0 / 3.0) * (y1 - cur_y)
                cx2 = x + (2.0 / 3.0) * (x1 - x)
                cy2 = y + (2.0 / 3.0) * (y1 - y)
                ctx.curve_to(cx1, cy1, cx2, cy2, x, y)
                cur_x, cur_y = x, y
                i += 4
            else:
                i += 1

    ctx.restore()
    ctx.save()
    if fill_black and progress > 0.25:
        ctx.set_source_rgba(0.0, 0.0, 0.0, 0.94)
        ctx.fill_preserve()

    ctx.set_source_rgba(*color)
    ctx.set_line_width(stroke_w)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)

    if dashed:
        ctx.set_dash([7.0, 6.0], 0.0)
    elif progress < 0.999:
        approx_len = 460.0 * max(0.4, scale)
        ctx.set_dash([approx_len * clamp(progress), approx_len * 2.5], 0.0)
    else:
        ctx.set_dash([], 0.0)

    ctx.stroke()
    ctx.restore()


# ============================================================================
# 3. MULTI-LAYERED, SOFT-EDGED COMPLEX GRAPHIC ICONS (PRIMARY + DETAIL LAYERS)
# ============================================================================
# Each icon can define:
# - Primary contours (drawn in bright `#FFFFFF` with `stroke_w`)
# - Detail contours (prefixed with `"DETAIL:"` — drawn in `#7A7A80` with `0.65 * stroke_w` for rich graphic depth!)
VECTOR_ICONS = {
    # --- CATEGORY A: PEOPLE, HANDS & SOCIETY ---
    "human": [
        "M 50 14 C 60 14 67 21 67 31 C 67 41 60 48 50 48 C 40 48 33 41 33 31 C 33 21 40 14 50 14 Z",
        "M 20 86 C 20 64 32 54 50 54 C 68 54 80 64 80 86",
        "DETAIL: M 42 24 C 45 20 52 19 57 22 M 32 84 C 32 70 40 63 50 63 C 60 63 68 70 68 84"
    ],
    "human_profile": [
        "M 58 14 C 38 14 28 26 28 42 C 28 46 23 52 20 56 C 19 58 22 61 27 62 V 70 C 27 75 32 78 38 78 H 44 V 88",
        "M 58 14 C 74 14 82 26 82 46 C 82 60 74 70 66 76 V 88",
        "DETAIL: M 38 40 C 41 40 43 42 43 45 C 43 48 41 50 38 50 C 35 50 33 48 33 45 C 33 42 35 40 38 40 Z",
        "DETAIL: M 48 30 C 56 26 66 30 68 40 C 70 48 64 56 56 58"
    ],
    "crowd_three": [
        "M 50 18 C 58 18 64 24 64 32 C 64 40 58 46 50 46 C 42 46 36 40 36 32 C 36 24 42 18 50 18 Z",
        "M 26 86 C 26 66 36 56 50 56 C 64 56 74 66 74 86",
        "DETAIL: M 22 28 C 28 28 32 32 32 38 C 32 44 28 48 22 48 C 16 48 12 44 12 38 C 12 32 16 28 22 28 Z M 6 82 C 6 66 13 58 24 58",
        "DETAIL: M 78 28 C 84 28 88 32 88 38 C 88 44 84 48 78 48 C 72 48 68 44 68 38 C 68 32 72 28 78 28 Z M 94 82 C 94 66 87 58 76 58"
    ],
    "hand": [
        "M 26 54 V 28 C 26 23 33 23 33 28 V 46",
        "M 33 46 V 18 C 33 13 41 13 41 18 V 46",
        "M 41 46 V 15 C 41 10 49 10 49 15 V 46",
        "M 49 46 V 22 C 49 17 57 17 57 22 V 54",
        "M 57 46 L 68 36 C 73 32 78 38 74 44 L 58 70 C 53 79 45 85 35 85 C 24 85 19 75 19 60 V 42 C 19 37 26 37 26 42",
        "DETAIL: M 30 58 C 36 56 44 58 50 64 M 32 68 C 38 68 44 72 46 76"
    ],
    "hand_click": [
        "M 42 52 V 22 C 42 16 51 16 51 22 V 46",
        "M 51 43 C 51 38 59 38 59 43 V 49 C 59 44 67 44 67 49 V 55 C 67 51 75 51 75 58 V 71 C 75 82 64 88 49 88 C 35 88 28 78 28 65 L 22 54 C 19 49 25 44 30 48 L 42 52",
        "DETAIL: M 32 20 C 37 10 55 10 60 20 M 24 26 L 31 26 M 69 26 L 62 26"
    ],
    "eye": [
        "M 8 50 C 20 26 35 18 50 18 C 65 18 80 26 92 50 C 80 74 65 82 50 82 C 35 82 20 74 8 50 Z",
        "M 50 32 C 60 32 68 40 68 50 C 68 60 60 68 50 68 C 40 68 32 60 32 50 C 32 40 40 32 50 32 Z",
        "DETAIL: M 50 42 C 55 42 58 45 58 50 C 58 55 55 58 50 58 C 45 58 42 55 42 50 M 22 36 C 30 26 40 23 50 23"
    ],
    "brain": [
        "M 50 18 V 82",
        "M 50 24 C 36 16 20 24 22 40 C 12 46 14 64 26 68 C 26 80 38 86 50 78",
        "M 50 24 C 64 16 80 24 78 40 C 88 46 86 64 74 68 C 74 80 62 86 50 78",
        "DETAIL: M 22 40 C 32 42 38 36 38 28 M 26 68 C 34 64 40 56 36 48 M 78 40 C 68 42 62 36 62 28 M 74 68 C 66 64 60 56 64 48"
    ],

    # --- CATEGORY B: MONEY, ECONOMY & COMMERCE ---
    "money_core": [
        "M 50 12 C 71 12 88 29 88 50 C 88 71 71 88 50 88 C 29 88 12 71 12 50 C 12 29 29 12 50 12 Z",
        "M 50 28 V 72 M 58 36 C 58 31 42 31 42 39 C 42 48 58 48 58 58 C 58 66 42 66 42 60",
        "DETAIL: M 50 21 C 66 21 79 34 79 50 C 79 66 66 79 50 79 C 34 79 21 66 21 50 C 21 34 34 21 50 21 Z"
    ],
    "banknote": [
        rounded_rect_svg(10, 24, 80, 52, r=12),
        "M 50 36 C 58 36 64 42 64 50 C 64 58 58 64 50 64 C 42 64 36 58 36 50 C 36 42 42 36 50 36 Z",
        "DETAIL: " + rounded_rect_svg(17, 31, 66, 38, r=8),
        "DETAIL: M 24 44 V 56 M 76 44 V 56"
    ],
    "vault": [
        rounded_rect_svg(14, 14, 72, 70, r=14),
        "M 54 34 C 63 34 70 41 70 50 C 70 59 63 66 54 66 C 45 66 38 59 38 50 C 38 41 45 34 54 34 Z",
        "M 54 28 V 72 M 32 50 H 76",
        "DETAIL: " + rounded_rect_svg(22, 22, 56, 54, r=9),
        "DETAIL: M 24 84 V 91 M 76 84 V 91 M 22 36 H 28 M 22 64 H 28"
    ],
    "smart_card": [
        rounded_rect_svg(10, 22, 80, 56, r=12),
        "M 10 38 H 90",
        "DETAIL: " + rounded_rect_svg(20, 48, 20, 16, r=4),
        "DETAIL: M 50 52 H 78 M 50 62 H 68 M 50 70 H 60"
    ],
    "cart": [
        "M 12 20 H 22 C 25 20 27 22 28 26 L 35 58 C 36 62 39 64 44 64 H 74 C 78 64 81 61 82 57 L 87 34 H 29",
        "M 40 72 C 44 72 47 75 47 79 C 47 83 44 86 40 86 C 36 86 33 83 33 79 C 33 75 36 72 40 72 Z",
        "M 70 72 C 74 72 77 75 77 79 C 77 83 74 86 70 86 C 66 86 63 83 63 79 C 63 75 66 72 70 72 Z",
        "DETAIL: M 44 42 H 74 M 48 52 H 70"
    ],
    "price_tag": [
        "M 16 50 L 46 20 C 49 17 53 16 57 16 H 78 C 82 16 84 18 84 22 V 43 C 84 47 83 51 80 54 L 50 84 C 45 89 38 89 33 84 L 16 67 C 11 62 11 55 16 50 Z",
        "M 67 28 C 71 28 74 31 74 35 C 74 39 71 42 67 42 C 63 42 60 39 60 35 C 60 31 63 28 67 28 Z",
        "DETAIL: M 34 56 L 52 38 M 42 64 L 58 48"
    ],
    "chart_up": [
        "M 16 16 V 76 C 16 81 19 84 24 84 H 86",
        "M 26 66 C 34 62 38 46 46 46 C 54 46 56 56 64 52 C 70 48 76 34 82 24 M 66 24 H 82 V 40",
        "DETAIL: M 28 84 V 74 M 46 84 V 66 M 64 84 V 68 M 82 84 V 52"
    ],

    # --- CATEGORY C: POLITICS, LAW, GOVERNANCE & WAR ---
    "government": [
        "M 12 86 H 88 M 16 78 H 84",
        "M 16 40 H 84 L 54 16 C 51 14 49 14 46 16 L 16 40 Z",
        "M 24 40 V 78 M 38 40 V 78 M 62 40 V 78 M 76 40 V 78",
        "DETAIL: M 50 24 C 54 24 57 27 57 31 C 57 35 54 38 50 38 C 46 38 43 35 43 31 C 43 27 46 24 50 24 Z M 46 78 V 56 C 46 52 54 52 54 56 V 78"
    ],
    "scale_balance": [
        "M 50 14 V 84 M 26 84 H 74",
        "M 16 32 C 30 32 38 28 50 28 C 62 28 70 32 84 32",
        "M 18 32 L 8 56 C 8 64 28 64 28 56 Z M 82 32 L 72 56 C 72 64 92 64 92 56 Z",
        "DETAIL: M 50 18 C 54 18 57 21 57 25 C 57 29 54 32 50 32 C 46 32 43 29 43 25 C 43 21 46 18 50 18 Z M 34 78 H 66"
    ],
    "contract": [
        rounded_rect_svg(20, 12, 60, 76, r=10),
        "M 32 72 C 38 62 42 78 48 68 C 54 58 58 76 68 68",
        "DETAIL: M 32 30 H 68 M 32 42 H 68 M 32 54 H 54 M 62 24 C 66 24 68 26 68 30"
    ],
    "folder": [
        "M 12 30 C 12 24 16 20 22 20 H 38 C 42 20 45 22 48 26 L 52 32 H 78 C 84 32 88 36 88 42 V 76 C 88 82 84 86 78 86 H 22 C 16 86 12 82 12 76 Z",
        "DETAIL: M 14 44 H 86 M 26 64 H 54"
    ],
    "gavel": [
        rounded_rect_svg(44, 16, 36, 20, r=6),
        "M 54 36 L 22 68 C 19 71 19 75 22 78",
        "M 14 84 H 56",
        "DETAIL: M 48 22 H 76 M 20 78 H 50"
    ],
    "missile": [
        "M 50 10 C 62 24 64 44 64 66 H 36 C 36 44 38 24 50 10 Z",
        "M 36 52 L 22 68 C 20 71 22 74 26 74 H 36 M 64 52 L 78 68 C 80 71 78 74 74 74 H 64",
        "DETAIL: M 50 32 C 54 32 57 35 57 39 C 57 43 54 46 50 46 C 46 46 43 43 43 39 C 43 35 46 32 50 32 Z M 44 66 L 40 84 M 56 66 L 60 84"
    ],
    "wall_bricks": [
        rounded_rect_svg(12, 22, 76, 56, r=8),
        "M 12 40 H 88 M 12 58 H 88",
        "DETAIL: M 38 22 V 40 M 64 22 V 40 M 26 40 V 58 M 52 40 V 58 M 76 40 V 58 M 38 58 V 78 M 64 58 V 78"
    ],
    "marionette_cross": [
        rounded_rect_svg(14, 16, 72, 12, r=6),
        "M 50 8 V 36",
        "DETAIL: M 24 28 C 24 48 28 62 32 82 M 50 36 V 76 M 76 28 C 76 48 72 62 68 82"
    ],

    # --- CATEGORY D: MEDICINE, HEALTH & SCIENCE ---
    "vial": [
        rounded_rect_svg(34, 12, 32, 10, r=4),
        "M 40 22 V 32 L 30 42 C 27 45 26 49 26 54 V 78 C 26 84 30 88 36 88 H 64 C 70 88 74 84 74 78 V 54 C 74 49 73 45 70 42 L 60 32 V 22",
        "DETAIL: " + rounded_rect_svg(34, 50, 32, 24, r=5),
        "DETAIL: M 50 56 V 68 M 44 62 H 56"
    ],
    "pill": [
        "M 26 52 L 52 26 C 61 17 75 17 83 25 C 91 33 91 47 82 56 L 56 82 C 47 91 33 91 25 83 C 17 75 17 61 26 52 Z",
        "M 39 39 L 69 69",
        "DETAIL: M 56 28 C 62 23 70 23 75 28 M 30 62 C 26 68 26 74 30 78"
    ],
    "syringe": [
        "M 68 16 L 84 32 M 76 24 L 62 38",
        "M 48 26 L 74 52 L 42 84 L 16 58 Z M 29 71 L 12 88",
        "DETAIL: M 42 42 L 52 52 M 34 50 L 44 60 M 26 58 L 36 68"
    ],
    "heart_pulse": [
        "M 50 84 C 50 84 14 62 14 36 C 14 21 26 14 38 18 C 44 20 48 26 50 31 C 52 26 56 20 62 18 C 74 14 86 21 86 36 C 86 62 50 84 50 84 Z",
        "M 22 48 H 37 L 44 34 L 54 62 L 61 48 H 78",
        "DETAIL: M 26 26 C 32 22 38 24 42 28"
    ],

    # --- CATEGORY E: NATURE, ENERGY, FUEL & INDUSTRY ---
    "tree": [
        "M 50 88 V 44 M 50 66 L 36 54 M 50 58 L 64 46",
        "M 50 14 C 34 14 23 26 27 40 C 16 44 18 62 32 64 H 68 C 82 62 84 44 73 40 C 77 26 66 14 50 14 Z",
        "DETAIL: M 36 26 C 42 20 52 18 60 24 M 34 88 H 66"
    ],
    "wheat": [
        "M 50 88 V 16",
        "M 50 70 C 34 70 28 58 33 47 C 44 47 50 58 50 70 Z M 50 70 C 66 70 72 58 67 47 C 56 47 50 58 50 70 Z",
        "M 50 50 C 34 50 28 38 33 27 C 44 27 50 38 50 50 Z M 50 50 C 66 50 72 38 67 27 C 56 27 50 38 50 50 Z",
        "M 50 30 C 39 22 43 10 50 8 C 57 10 61 22 50 30 Z",
        "DETAIL: M 36 44 L 26 36 M 64 44 L 74 36 M 36 24 L 28 16 M 64 24 L 72 16"
    ],
    "drop": [
        "M 50 12 C 50 12 22 46 22 65 C 22 81 35 90 50 90 C 65 90 78 81 78 65 C 78 46 50 12 50 12 Z",
        "DETAIL: M 34 62 C 34 73 41 80 51 80 M 36 48 C 40 38 46 28 50 22"
    ],
    "factory": [
        "M 14 84 V 48 L 36 60 V 48 L 58 60 V 34 C 58 30 60 28 64 28 H 70 C 74 28 76 30 76 34 V 60 H 86 V 84 Z",
        "M 20 48 V 28 C 20 25 22 24 25 24 H 30 C 33 24 35 25 35 28 V 55",
        "DETAIL: M 64 20 C 64 12 74 14 78 8 C 82 12 88 8 92 4 M 26 18 C 26 12 34 12 38 8",
        "DETAIL: " + rounded_rect_svg(22, 66, 12, 12, r=3) + " " + rounded_rect_svg(44, 66, 12, 12, r=3) + " " + rounded_rect_svg(66, 66, 12, 12, r=3)
    ],
    "fuel_pump": [
        rounded_rect_svg(20, 18, 44, 66, r=8),
        rounded_rect_svg(28, 26, 28, 20, r=4),
        "M 64 34 H 72 C 76 34 80 38 80 44 V 70 C 80 76 72 76 72 70 V 52 H 64 M 14 84 H 70",
        "DETAIL: M 32 58 H 52 M 32 66 H 46"
    ],
    "oil_derrick": [
        "M 26 84 L 44 18 H 56 L 74 84 M 14 84 H 86",
        "M 42 18 C 42 12 58 12 58 18",
        "DETAIL: M 38 42 H 62 M 32 64 H 68 M 38 42 L 68 64 M 62 42 L 32 64"
    ],
    "saw": [
        "M 28 40 H 86 C 88 40 88 44 86 48 L 80 58 L 72 48 L 64 58 L 56 48 L 48 58 L 40 48 L 32 58 V 40 Z",
        rounded_rect_svg(12, 34, 18, 28, r=5),
        "DETAIL: " + rounded_rect_svg(17, 40, 8, 16, r=3)
    ],
    "pickaxe": [
        "M 20 28 C 38 14 62 14 80 28 C 66 28 56 32 50 38 C 44 32 34 28 20 28 Z",
        "M 50 22 V 86",
        "DETAIL: M 46 22 H 54 M 47 78 H 53"
    ],

    # --- CATEGORY F: EVERYDAY LIFE, HOUSING, TIME & TRANSPORT ---
    "home": [
        "M 14 48 L 45 18 C 48 15 52 15 55 18 L 86 48",
        "M 24 42 V 78 C 24 82 27 85 32 85 H 68 C 73 85 76 82 76 78 V 42",
        "DETAIL: M 42 85 V 60 C 42 55 58 55 58 60 V 85 M 44 34 C 44 30 56 30 56 34 C 56 38 44 38 44 34 Z"
    ],
    "bread": [
        "M 20 46 C 12 32 22 20 50 20 C 78 20 88 32 80 46 L 76 76 C 75 81 71 84 66 84 H 34 C 29 84 25 81 24 76 Z",
        "DETAIL: M 36 32 C 40 38 40 44 36 50 M 50 30 C 54 36 54 42 50 48 M 64 32 C 68 38 68 44 64 50"
    ],
    "car": [
        "M 10 64 C 10 56 14 52 18 48 L 30 34 C 33 31 38 30 44 30 H 62 C 68 30 73 32 77 38 L 84 48 C 88 50 90 54 90 60 V 66 C 90 69 88 71 84 71 H 16 C 12 71 10 68 10 64 Z",
        "M 28 62 C 34 62 38 66 38 72 C 38 78 34 82 28 82 C 22 82 18 78 18 72 C 18 66 22 62 28 62 Z",
        "M 72 62 C 78 62 82 66 82 72 C 82 78 78 82 72 82 C 66 82 62 78 62 72 C 62 66 66 62 72 62 Z",
        "DETAIL: M 34 36 L 24 48 H 48 V 36 Z M 54 36 V 48 H 76 L 68 36 Z"
    ],
    "clock": [
        "M 50 14 C 70 14 86 30 86 50 C 86 70 70 86 50 86 C 30 86 14 70 14 50 C 14 30 30 14 50 14 Z",
        "M 50 28 V 50 L 66 58",
        "DETAIL: M 50 20 V 24 M 50 76 V 80 M 20 50 H 24 M 76 50 H 80"
    ],
    "calendar": [
        rounded_rect_svg(16, 24, 68, 62, r=12),
        "M 16 42 H 84 M 34 14 V 30 M 66 14 V 30",
        "DETAIL: " + rounded_rect_svg(28, 52, 12, 10, r=3) + " " + rounded_rect_svg(46, 52, 12, 10, r=3) + " " + rounded_rect_svg(62, 52, 12, 10, r=3),
        "DETAIL: " + rounded_rect_svg(28, 68, 12, 10, r=3) + " " + rounded_rect_svg(46, 68, 12, 10, r=3)
    ],
    "moon": [
        "M 64 20 C 46 22 34 38 36 56 C 38 72 54 82 70 78 C 56 88 34 84 24 68 C 14 52 18 30 34 20 C 44 14 54 14 64 20 Z",
        "DETAIL: M 70 28 L 73 34 L 79 37 L 73 40 L 70 46 L 67 40 L 61 37 L 67 34 Z"
    ],
    "mirror": [
        "M 50 12 C 68 12 80 26 80 47 C 80 68 68 82 50 82 C 32 82 20 68 20 47 C 20 26 32 12 50 12 Z",
        "M 22 88 H 78 M 50 82 V 88",
        "DETAIL: M 50 18 C 64 18 73 30 73 47 C 73 64 64 76 50 76 C 36 76 27 64 27 47 C 27 30 36 18 50 18 Z M 36 32 L 46 22"
    ],
    "book_open": [
        "M 50 26 C 38 18 24 18 12 24 V 76 C 24 70 38 70 50 78 C 62 70 76 70 88 76 V 24 C 76 18 62 18 50 26 Z",
        "M 50 26 V 78",
        "DETAIL: M 20 38 C 28 35 36 35 42 38 M 20 50 C 28 47 36 47 42 50 M 58 38 C 64 35 72 35 80 38 M 58 50 C 64 47 72 47 80 50"
    ],

    # --- CATEGORY G: TECH, AI & SYMBOLS ---
    "ai_monogram": [
        "M 36 24 H 64 M 50 24 V 76 M 36 76 H 64",
        "M 50 10 C 72 10 90 28 90 50 C 90 72 72 90 50 90 C 28 90 10 72 10 50 C 10 28 28 10 50 10 Z",
        "DETAIL: M 50 17 C 68 17 83 32 83 50 C 83 68 68 83 50 83 C 32 83 17 68 17 50 C 17 32 32 17 50 17 Z"
    ],
    "lock": [
        rounded_rect_svg(24, 44, 52, 42, r=10),
        "M 34 44 V 30 C 34 18 41 12 50 12 C 59 12 66 18 66 30 V 44",
        "M 50 58 C 53 58 56 60 56 64 C 56 67 53 69 50 70 V 76",
        "DETAIL: " + rounded_rect_svg(30, 50, 40, 30, r=6)
    ],
    "question": [
        "M 36 34 C 36 20 46 16 54 18 C 65 20 68 34 56 42 C 50 46 50 52 50 60",
        "M 50 72 C 53 72 55 74 55 77 C 55 80 53 82 50 82 C 47 82 45 80 45 77 C 45 74 47 72 50 72 Z",
        "DETAIL: M 50 10 C 72 10 90 28 90 50 C 90 72 72 90 50 90 C 28 90 10 72 10 50 C 10 28 28 10 50 10 Z"
    ],
    "check": [
        "M 20 52 L 42 74 L 80 26"
    ],
    "cross": [
        "M 26 26 L 74 74 M 74 26 L 26 74"
    ],
    "warning": [
        "M 45 18 C 47 14 53 14 55 18 L 86 76 C 88 80 85 84 80 84 H 20 C 15 84 12 80 14 76 Z",
        "M 50 38 V 58 M 50 68 V 72"
    ],
    "search": [
        "M 44 18 C 58 18 70 30 70 44 C 70 58 58 70 44 70 C 30 70 18 58 18 44 C 18 30 30 18 44 18 Z",
        "M 63 63 L 84 84",
        "DETAIL: M 32 34 C 36 28 44 26 50 28"
    ],
    "cpu_chip": [
        rounded_rect_svg(26, 26, 48, 48, r=10),
        rounded_rect_svg(38, 38, 24, 24, r=5),
        "DETAIL: M 38 14 V 26 M 50 14 V 26 M 62 14 V 26 M 38 74 V 86 M 50 74 V 86 M 62 74 V 86",
        "DETAIL: M 14 38 H 26 M 14 50 H 26 M 14 62 H 26 M 74 38 H 86 M 74 50 H 86 M 74 62 H 86"
    ],
}

def draw_icon(ctx, icon_name, cx, cy, scale=1.0, progress=1.0, color=C_WHITE,
              stroke_w=2.3, dashed=False, fill_black=False, show_details=True):
    """
    Renders a multi-layered icon from `VECTOR_ICONS`:
    - Primary paths are drawn with `color` and `stroke_w`.
    - `"DETAIL:"` paths are drawn with a softer gray/alpha and thinner stroke (`0.65 * stroke_w`) for rich visual depth!
    """
    paths = VECTOR_ICONS.get(icon_name, VECTOR_ICONS["question"])
    n = len(paths)
    for idx, raw_d in enumerate(paths):
        is_detail = raw_d.startswith("DETAIL:")
        if is_detail and not show_details:
            continue
        d_str = raw_d.replace("DETAIL:", "").strip()
        sub_p = clamp((progress * n - idx * 0.28) / max(1.0, n - idx * 0.28))
        if is_detail:
            det_col = (color[0] * 0.58, color[1] * 0.58, color[2] * 0.60, color[3] * 0.88)
            draw_svg_path(ctx, d_str, cx=cx, cy=cy, scale=scale, progress=sub_p,
                          color=det_col, stroke_w=max(1.2, stroke_w * 0.62), dashed=dashed, fill_black=False)
        else:
            draw_svg_path(ctx, d_str, cx=cx, cy=cy, scale=scale, progress=sub_p,
                          color=color, stroke_w=stroke_w, dashed=dashed, fill_black=(fill_black and idx == 0))


# ============================================================================
# 4. EXTERNAL SVG & LOTTIE JSON IMPORTERS
# ============================================================================
def load_external_svg_paths(svg_file_path):
    tree = ET.parse(svg_file_path)
    root = tree.getroot()
    vb = root.attrib.get("viewBox", "0 0 24 24").split()
    vb_size = float(vb[2]) if len(vb) >= 3 else 24.0
    paths = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "path" and "d" in elem.attrib:
            paths.append(elem.attrib["d"])
    return paths, vb_size

def find_system_svg(keyword):
    matches = glob.glob(f"/usr/share/icons/Adwaita/scalable/**/*{keyword}*.svg", recursive=True)
    return sorted(matches)

def load_lottie_json_paths(lottie_json_path):
    with open(lottie_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    svg_paths = []
    def _walk_shapes(items):
        for it in items:
            if it.get("ty") == "gr" and "it" in it:
                _walk_shapes(it["it"])
            elif it.get("ty") == "sh" and "ks" in it:
                ks = it["ks"].get("k", {})
                if isinstance(ks, list) and len(ks) > 0 and "s" in ks[0]:
                    ks = ks[0]["s"][0]
                if isinstance(ks, dict) and "v" in ks:
                    verts, ins, outs = ks["v"], ks["i"], ks["o"]
                    closed = ks.get("c", False)
                    if len(verts) >= 2:
                        parts = [f"M {verts[0][0]:.2f} {verts[0][1]:.2f}"]
                        for idx in range(len(verts) - 1):
                            p0, p1 = verts[idx], verts[idx + 1]
                            c1 = [p0[0] + outs[idx][0], p0[1] + outs[idx][1]]
                            c2 = [p1[0] + ins[idx + 1][0], p1[1] + ins[idx + 1][1]]
                            parts.append(f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p1[0]:.2f} {p1[1]:.2f}")
                        if closed:
                            p0, p1 = verts[-1], verts[0]
                            c1 = [p0[0] + outs[-1][0], p0[1] + outs[-1][1]]
                            c2 = [p1[0] + ins[0][0], p1[1] + ins[0][1]]
                            parts.append(f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p1[0]:.2f} {p1[1]:.2f} Z")
                        svg_paths.append(" ".join(parts))
    for layer in data.get("layers", []):
        if "shapes" in layer:
            _walk_shapes(layer["shapes"])
    return svg_paths


# ============================================================================
# 5. DUAL-FONT PERSIAN/ASCII TEXT ENGINE WITH CACHED GLOW SURFACES
# ============================================================================
def _get_font(path, size):
    k = (path, int(size))
    if k not in _FONT_CACHE:
        _FONT_CACHE[k] = ImageFont.truetype(path, int(size))
    return _FONT_CACHE[k]

def get_text_surface(text, size=22, color_rgb=(255, 255, 255), bold=True, glow=False):
    key = (text, int(size), tuple(color_rgb), bool(bold), bool(glow))
    if key in _TEXT_SURFACE_CACHE:
        return _TEXT_SURFACE_CACHE[key]

    has_ascii_sym = any(c in "()[]%$@#&*+=<>/✓✕✗✔★☆" or ('a' <= c.lower() <= 'z') for c in text)
    f_path = FONT_EN_BOLD if has_ascii_sym else (FONT_FA_BOLD if bold else FONT_FA_REG)
    font = _get_font(f_path, size)
    is_pure_ascii = all(ord(c) < 128 for c in text)

    dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dummy)
    if is_pure_ascii:
        bb = dd.textbbox((0, 0), text, font=font)
    else:
        bb = dd.textbbox((0, 0), text, font=font, direction="rtl", language="fa")
    tw, th = max(4, bb[2] - bb[0]), max(4, bb[3] - bb[1])
    pad = 18 if glow else 8
    im = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    if is_pure_ascii:
        draw.text((pad - bb[0], pad - bb[1]), text, font=font, fill=(*color_rgb, 255))
    else:
        draw.text((pad - bb[0], pad - bb[1]), text, font=font, fill=(*color_rgb, 255), direction="rtl", language="fa")

    arr = np.array(im, dtype=np.float32)
    if glow:
        blur_a = cv2.GaussianBlur(arr[:, :, 3], (19, 19), 5.5) * 0.55
        base_a = arr[:, :, 3:4] / 255.0
        gl_a = (blur_a[:, :, None] / 255.0) * (1.0 - base_a)
        out_a = np.clip(base_a + gl_a, 0.0, 1.0)
        safe_a = np.maximum(out_a, 1e-4)
        c_arr = np.array(color_rgb, dtype=np.float32)[None, None, :]
        out_rgb = (arr[:, :, :3] * base_a + c_arr * gl_a) / safe_a
        arr = np.dstack([np.clip(out_rgb, 0, 255), out_a * 255.0])

    bgra = np.zeros_like(arr, dtype=np.uint8)
    a_norm = arr[:, :, 3:4] / 255.0
    bgra[:, :, 0] = np.clip(arr[:, :, 2:3] * a_norm, 0, 255)[:, :, 0].astype(np.uint8)
    bgra[:, :, 1] = np.clip(arr[:, :, 1:2] * a_norm, 0, 255)[:, :, 0].astype(np.uint8)
    bgra[:, :, 2] = np.clip(arr[:, :, 0:1] * a_norm, 0, 255)[:, :, 0].astype(np.uint8)
    bgra[:, :, 3] = np.clip(arr[:, :, 3], 0, 255).astype(np.uint8)

    _TEXT_SURFACE_CACHE[key] = (bgra, tw, th, pad)
    return _TEXT_SURFACE_CACHE[key]

def draw_pango_text(ctx, text, cx, cy, size=22, color=C_WHITE, bold=True, glow=False, align="center"):
    if not text or color[3] <= 0.01:
        return (0, 0)
    c_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
    bgra, tw, th, pad = get_text_surface(text, size=size, color_rgb=c_rgb, bold=bold, glow=glow)
    sh, sw = bgra.shape[:2]
    surf = cairo.ImageSurface.create_for_data(bgra, cairo.FORMAT_ARGB32, sw, sh)

    if align == "center":
        x0 = cx - sw * 0.5
    elif align == "right":
        x0 = cx - sw + pad
    else:
        x0 = cx - pad
    y0 = cy - sh * 0.5

    ctx.save()
    ctx.set_source_surface(surf, x0, y0)
    ctx.paint_with_alpha(clamp(color[3]))
    ctx.restore()
    return (tw, th)


# ============================================================================
# 6. SOFT-EDGED, MULTI-LAYERED @ENSAN.BESHAM STUDIO WIDGETS
# ============================================================================
def widget_callout(ctx, anchor_pt, elbow_pt, shelf_w, label, progress=1.0, color=C_WHITE, size=19, glow=False):
    if progress <= 0.01:
        return
    ctx.save()
    ctx.set_source_rgba(*color)
    ctx.arc(anchor_pt[0], anchor_pt[1], 3.8, 0, 2 * math.pi)
    ctx.fill()
    ctx.arc(anchor_pt[0], anchor_pt[1], 7.0, 0, 2 * math.pi)
    ctx.set_line_width(1.2)
    ctx.stroke()

    end_x = elbow_pt[0] + shelf_w
    d_str = f"M {anchor_pt[0]} {anchor_pt[1]} L {elbow_pt[0]} {elbow_pt[1]} L {end_x} {elbow_pt[1]}"
    draw_svg_path(ctx, d_str, cx=50, cy=50, scale=1.0, progress=min(1.0, progress * 1.3),
                  color=color, stroke_w=1.8, view_box=100.0, auto_fillet=8.0)
    if progress > 0.35:
        tp = (progress - 0.35) / 0.65
        tc = (color[0], color[1], color[2], color[3] * tp)
        draw_pango_text(ctx, label, (elbow_pt[0] + end_x) * 0.5, elbow_pt[1] - 18, size=size, color=tc, bold=True, glow=glow)
    ctx.restore()

def widget_stamp(ctx, cx, cy, label, t, t0, angle_deg=-7.0, w=172, h=56):
    """
    Soft-cornered double-bordered vintage verdict stamp (`[مستند]`, `[صفر]`, `[حذف‌شده]`).
    """
    if t < t0:
        return
    sp = spring(t, t0, dur=0.40, freq=7.2, decay=5.5)
    scale = max(0.2, 1.75 - 0.75 * sp)
    op = clamp((t - t0) / 0.12)
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(math.radians(angle_deg))
    ctx.scale(scale, scale)
    draw_rounded_rect(ctx, 0, 0, w, h, r=14.0, progress=1.0, color=(1, 1, 1, op),
                      stroke_w=2.6, fill_black=True, double_border=True)
    draw_pango_text(ctx, label, 0, 0, size=24, color=(1, 1, 1, op), bold=True, glow=True)
    ctx.restore()

def widget_live_strikethrough(ctx, cx, cy, false_word, true_word, strike_prog, true_prog, size=25):
    dim = 1.0 - 0.58 * clamp(strike_prog)
    tw, _ = draw_pango_text(ctx, false_word, cx, cy, size=size, color=(dim, dim, dim, 1.0), bold=True)
    if strike_prog > 0.01:
        hw = (tw * 0.5 + 14) * ease_out(strike_prog)
        ctx.save()
        ctx.set_source_rgba(1, 1, 1, 1)
        ctx.set_line_width(2.5)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(cx + tw * 0.5 + 12, cy)
        ctx.line_to(cx + tw * 0.5 + 12 - hw * 2.0, cy)
        ctx.stroke()
        ctx.restore()
    if true_prog > 0.01:
        ep = ease_out(true_prog)
        ty = cy - 46 - (1.0 - ep) * 12
        # Soft rounded pill around the verified truth!
        tw_t, _ = draw_pango_text(ctx, true_word, cx - 14, ty,
                                  size=size, color=(1, 1, 1, ep), bold=True, glow=True)
        draw_icon(ctx, "check", cx - 14 + tw_t * 0.5 + 22, ty, scale=0.34, progress=ep, color=(1, 1, 1, ep), stroke_w=2.8)

def widget_laser_scanner(ctx, cx, cy, w, h, title, clause_text, scan_prog, open_prog=1.0):
    """
    Soft-cornered legal document tablet (`rx=22px`) with inner border, header divider,
    rounded skeleton lines, and a glowing horizontal laser scanner beam.
    """
    if open_prog <= 0.01:
        return
    draw_rounded_rect(ctx, cx, cy, w, h, r=22.0, progress=open_prog, color=C_WHITE,
                      stroke_w=2.4, fill_black=True, double_border=True)
    draw_pango_text(ctx, title, cx, cy - h * 0.5 + 36, size=24, color=(1, 1, 1, open_prog), bold=True, glow=True)

    # Header underline
    ctx.save()
    ctx.set_source_rgba(0.45, 0.45, 0.48, open_prog)
    ctx.set_line_width(1.5)
    ctx.move_to(cx - w * 0.38, cy - h * 0.5 + 62)
    ctx.line_to(cx + w * 0.38, cy - h * 0.5 + 62)
    ctx.stroke()
    ctx.restore()

    for i in range(5):
        ly = cy - h * 0.5 + 90 + i * 28
        lw = w * (0.68 if i % 2 == 0 else 0.52)
        ctx.save()
        ctx.set_source_rgba(0.30, 0.30, 0.33, open_prog)
        ctx.set_line_width(2.2)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(cx - lw * 0.5, ly)
        ctx.line_to(cx + lw * 0.5, ly)
        ctx.stroke()
        ctx.restore()

    if 0.01 < scan_prog < 0.99:
        beam_y = (cy - h * 0.5 + 68) + (h - 96) * ease_in_out(scan_prog)
        ctx.save()
        ctx.set_source_rgba(1, 1, 1, 0.95)
        ctx.set_line_width(3.2)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(cx - w * 0.44, beam_y)
        ctx.line_to(cx + w * 0.44, beam_y)
        ctx.stroke()
        ctx.restore()

    if scan_prog > 0.38:
        cp = clamp((scan_prog - 0.38) / 0.35)
        draw_rounded_rect(ctx, cx, cy + 6, w * 0.86, 48, r=14.0, progress=cp,
                          color=(0.75, 0.75, 0.78, cp), stroke_w=1.4, fill_black=True)
        draw_pango_text(ctx, clause_text, cx, cy + 6, size=19, color=(1, 1, 1, cp), bold=True, glow=True)

def widget_formula_pill(ctx, cx, cy, keyword, claim_text, open_prog=1.0, lock_morph=0.0):
    """
    Soft stadium capsule `[ همه می‌دونن | + هر ادعایی ... ]` with curved top arrow and `? -> 🔒` morphing badge.
    """
    if open_prog <= 0.01:
        return
    w, h = 310, 52
    draw_rounded_rect(ctx, cx, cy, w, h, r=26.0, progress=open_prog, color=C_WHITE, stroke_w=2.2, fill_black=True)
    # Highlighted right sub-pill for keyword
    draw_rounded_rect(ctx, cx + w * 0.24, cy, w * 0.48, h - 8, r=22.0, progress=open_prog,
                      color=(0.8, 0.8, 0.85, open_prog), stroke_w=1.5, fill_black=False)
    draw_pango_text(ctx, keyword, cx + w * 0.24, cy, size=18, color=(1, 1, 1, open_prog), bold=True, glow=True)
    draw_pango_text(ctx, claim_text, cx - w * 0.23, cy, size=16, color=(0.72, 0.72, 0.75, open_prog), bold=True)

    # Bottom morphing badge (`?` -> `lock`)
    by = cy + 62
    if lock_morph < 0.5:
        draw_icon(ctx, "question", cx, by, scale=0.45, progress=open_prog * (1.0 - lock_morph * 2), color=C_WHITE)
    else:
        draw_icon(ctx, "lock", cx, by, scale=0.45, progress=(lock_morph - 0.5) * 2, color=C_WHITE)

def widget_everyday_story_hud(ctx, cx, cy, prog_all=1.0, t=0.0):
    if prog_all <= 0.01:
        return
    gx, gy, gr = cx - 58, cy - 55, 48
    # Soft outer decorative ring
    ctx.save()
    ctx.set_source_rgba(0.32, 0.32, 0.35, prog_all)
    ctx.set_line_width(1.2)
    ctx.arc(gx, gy, gr + 9, 0, 2 * math.pi)
    ctx.stroke()
    ctx.restore()

    for s_i in range(8):
        a0 = math.radians(-90 + s_i * 45 + 5)
        a1 = math.radians(-90 + (s_i + 1) * 45 - 5)
        ctx.save()
        ctx.set_source_rgba(1, 1, 1, prog_all if s_i < int(prog_all * 8.5) else 0.25)
        ctx.set_line_width(4.8)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.arc(gx, gy, gr, a0, a1)
        ctx.stroke()
        ctx.restore()
    draw_pango_text(ctx, "۸", gx, gy - 2, size=34, color=(1, 1, 1, prog_all), bold=True, glow=True)
    draw_pango_text(ctx, "ساعت کار", gx, gy + 70, size=17, color=(0.78, 0.78, 0.80, prog_all), bold=True)

    needs = [("نون", "bread", 0.58), ("اجاره", "home", 0.40), ("دارو", "pill", 0.24)]
    for idx, (nlbl, nicon, fill_ratio) in enumerate(needs):
        ny = cy - 95 + idx * 42
        draw_icon(ctx, nicon, cx + 148, ny, scale=0.30, progress=prog_all, color=C_GRAY_LIGHT, stroke_w=1.9)
        draw_pango_text(ctx, nlbl, cx + 112, ny, size=15, color=(0.75, 0.75, 0.78, prog_all), align="right")
        # Rounded stadium capsule slider!
        bx, bw, bh = cx + 28, 54, 12
        draw_rounded_rect(ctx, bx, ny, bw, bh, r=6.0, progress=prog_all, color=(0.4, 0.4, 0.42, prog_all), stroke_w=1.4, fill_black=True)
        fw = max(6.0, (bw - 4) * fill_ratio * prog_all)
        draw_rounded_rect(ctx, bx - (bw - 4) * 0.5 + fw * 0.5, ny, fw, bh - 4, r=4.0,
                          progress=prog_all, color=C_WHITE, stroke_w=3.0, fill_black=False)

    # Road horizon & 3 soft-edged aerodynamic cars
    ctx.save()
    ctx.set_source_rgba(0.5, 0.5, 0.52, prog_all)
    ctx.set_line_width(1.8)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.move_to(cx - 175, cy + 98)
    ctx.line_to(cx + 175, cy + 98)
    ctx.stroke()
    ctx.restore()
    for c_i, ox in enumerate([-112, 0, 112]):
        col_c = C_WHITE if c_i == 1 else C_GRAY_MID
        draw_icon(ctx, "car", cx + ox, cy + 72, scale=0.66, progress=prog_all, color=col_c, stroke_w=2.4 if c_i == 1 else 1.8)