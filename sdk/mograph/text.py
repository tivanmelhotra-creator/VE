"""
mograph.text — fully VECTOR Persian/English text engine.

HarfBuzz shaping → glyph outlines → cairo paths.  Consequences:
  * text is a real vector: stays razor-sharp under any camera zoom/rotation
    (v3 blitted raster sprites → blurry when the camera zoomed in);
  * exact per-word geometry → exact karaoke highlighting, per-word animation,
    and exact bounding boxes for automatic overlap QA;
  * mixed Persian + English + digits + symbols in ONE string, with a small
    built-in bidi resolver (digits stay LTR inside RTL, parentheses mirror,
    '@handle' does not jump to the wrong side);
  * per-character font fallback (Vazirmatn → DejaVu → Noto Symbols2), and
    `missing_glyphs()` so QA can guarantee "no ▯ tofu boxes" before rendering.

Public API
----------
layout_text(text, size, weight="bold", max_width=None, line_height=1.55, align="center")
draw_text(ctx, text, x, y, size=40, color="#fff", weight="bold", align="center",
          valign="middle", alpha=1.0, max_width=None, word_style=None, reveal=None)
measure_text(text, size, weight="bold", max_width=None) -> (w, h)
text_bbox(text, x, y, ...) -> (x0, y0, x1, y1)
missing_glyphs(text, weight="bold") -> set of chars
parse_markup("**bold** word") -> (clean_text, {emphasised word indices})
to_fa_digits("2024") -> "۲۰۲۴"
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from functools import lru_cache

import cairo
try:
    import uharfbuzz as hb
    _HAS_UHARFBUZZ = True
    HB_BACKEND = "uharfbuzz"
except ImportError:
    # v9: no wheel → the native C shaper through ctypes (libharfbuzz.so.0) + fontTools outlines.
    # Same API surface, same 100 % vector path (exact ink boxes, karaoke geometry, outline/extrude styles).
    from . import hbc as _hbc
    if _hbc.available() and os.environ.get("MOGRAPH_SHAPER", "") != "pil":
        hb = _hbc
        _HAS_UHARFBUZZ = True
        HB_BACKEND = "ctypes:libharfbuzz " + (_hbc.version() or "")
    else:
        hb = None
        _HAS_UHARFBUZZ = False
        HB_BACKEND = "pil-raqm (raster fallback)"
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .core import clamp, color as _color

_HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(os.path.dirname(_HERE), "fonts")

WEIGHTS = {
    "regular": "Peyda-Regular.ttf",
    "medium": "Peyda-Medium.ttf",
    "semibold": "Peyda-SemiBold.ttf",
    "bold": "Peyda-Bold.ttf",
    "extrabold": "Peyda-ExtraBold.ttf",
    "black": "KalamehFaNum-Black.ttf",
    "title": "KalamehFaNum-Black.ttf",
    "peyda": "Peyda-Black.ttf",
    "kalameh": "KalamehFaNum-Black.ttf",
    "pelak": "PelakFA-ExtraBold.ttf",
    "doran": "DoranFaNum-ExtraBold.ttf",
    "aria": "AriaFaNum-Heavy.ttf",
    "handwritten": "DigiGhasedak-Bold.ttf",
    "delbar": "Delbar-Bold.ttf",
    "shoko": "Shoko-Bold.ttf",
    "anil": "DigiAnil-Bold.ttf",
    "nastaliq": "Miremad.ttf",
    "vazir": "Vazirmatn-Bold.ttf",
}
FALLBACKS = [
    os.path.join(FONT_DIR, "AriaFaNum-Heavy.ttf"),       # v7: Persian display font WITH arrows/dashes → symbols stay on-brand
    os.path.join(FONT_DIR, "Vazirmatn-Bold.ttf"),        # Persian fallback (digits/letters missing in display fonts)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
]


# Semantic typography roles → concrete weights. Elements accept "font": "<role>" (or any WEIGHTS key).
ROLES = {
    "body": "bold",            # Peyda Bold — neutral UI text, labels
    "title": "title",          # Kalameh Black — hooks, chapter titles, stamps, pills
    "subtitle": "aria",        # Aria Heavy — karaoke subtitles (very readable on mobile)
    "counter": "pelak",        # Pelak ExtraBold — numbers, counters, rings, bars
    "document": "doran",       # Doran ExtraBold — legal clauses, quotes, invoices
    "handwritten": "handwritten",  # DigiGhasedak — margin notes, callouts, speech bubbles
    "note": "delbar",          # Delbar — softer handwriting
    "nastaliq": "nastaliq",    # Miremad — poetic/philosophical twist, signatures
    "ui": "vazir",             # Vazirmatn — widest glyph coverage (symbols, mixed scripts)
    "ui_title": "bold",        # HUD header title (Peyda Bold)
    "ui_label": "medium",      # HUD prefix / handle / small labels (Peyda Medium)
}
DEFAULT_ROLES = dict(ROLES)


def apply_style_roles(overrides):
    """Per-video typography: meta.fonts = {"title": "peyda", "body": "medium", ...} (role → weight key / path).
    Called by Scene; resets to defaults first so videos don't leak into each other."""
    use_roles(overrides)


_ROLES_KEY = None


def use_roles(fonts):
    """v8: switch the active font roles WITHOUT clearing the shaping caches (caches are keyed by the
    concrete font file, so chapters with different typography never poison each other).
    Called by the renderer before every chapter / the HUD."""
    global _ROLES_KEY
    key = tuple(sorted((fonts or {}).items()))
    if key == _ROLES_KEY:
        return
    ROLES.clear(); ROLES.update(DEFAULT_ROLES)
    for k, v in (fonts or {}).items():
        ROLES[k] = v
    _ROLES_KEY = key


def concrete_font(weight):
    """role / weight key / path → absolute font path (what is actually painted)."""
    return _font(weight).path


def _shape_cached_clear():
    f = globals().get("_shape_cached")
    if f is not None and hasattr(f, "cache_clear"):
        f.cache_clear()


def resolve_font(font=None, weight=None, default="bold"):
    """'font' (role or weight key) wins over 'weight'. Unknown → default."""
    for k in (font, weight, default):
        if not k:
            continue
        if k in ROLES:
            return ROLES[k]
        if k in WEIGHTS or os.path.isabs(str(k)):
            return k
    return "bold"


def register_weight(name: str, path: str):
    """Add your own font: register_weight('title', '/path/Font.ttf')."""
    WEIGHTS[name] = path


# ---------------------------------------------------------------------------
# font objects
# ---------------------------------------------------------------------------

class _Font:
    def __init__(self, path):
        self.path = path
        if _HAS_UHARFBUZZ:
            blob = hb.Blob.from_file_path(path)
            self.face = hb.Face(blob)
            self.hb = hb.Font(self.face)
            self.upem = self.face.upem
            ext = self.hb.get_font_extents("ltr")
            self.asc, self.desc = ext.ascender, ext.descender
            self._cmap = None
        else:
            tt = TTFont(path)
            self.upem = tt["head"].unitsPerEm
            self.asc = tt["hhea"].ascent
            self.desc = tt["hhea"].descent
            self._cmap = set(tt.getBestCmap().keys())
            tt.close()
        self._paths = {}
        self._pil_fonts = {}

    def pil_font(self, size_px):
        k = max(4, int(round(size_px)))
        if k not in self._pil_fonts:
            self._pil_fonts[k] = ImageFont.truetype(self.path, k)
        return self._pil_fonts[k]

    def has(self, ch):
        if self._cmap is None:
            self._cmap = set(self.face.unicodes)
        return ord(ch) in self._cmap

    def glyph_path(self, gid):
        p = self._paths.get(gid)
        if p is None:
            p = _outline_to_cairo(self.hb, gid)
            self._paths[gid] = p
        return p


_FONTS: dict[str, _Font] = {}


def _font(path_or_weight) -> _Font:
    path = path_or_weight
    for _ in range(3):                           # roles are accepted anywhere a weight is (may chain)
        if path in ROLES and path not in WEIGHTS:
            path = ROLES[path]
        elif path in ROLES and ROLES[path] != path and path not in ("bold", "medium", "regular", "black"):
            path = ROLES[path]
        else:
            break
    if path in WEIGHTS:
        path = WEIGHTS[path]
    if not os.path.isabs(path):
        path = os.path.join(FONT_DIR, path)
    f = _FONTS.get(path)
    if f is None:
        f = _Font(path)
        _FONTS[path] = f
    return f


_SCR = cairo.Context(cairo.ImageSurface(cairo.FORMAT_A8, 1, 1))


from fontTools.pens.basePen import BasePen as _BasePen


class _Pen(_BasePen):
    """fontTools pen → cairo path (y flipped).  BasePen decomposes quadratic runs, implied on-curve points
    and all-off-curve TrueType contours, so every glyph (glyf or CFF) becomes exact cubic Béziers."""

    def __init__(self, ctx):
        super().__init__(None)
        self.c = ctx

    def _moveTo(self, p):
        self.c.move_to(p[0], -p[1])

    def _lineTo(self, p):
        self.c.line_to(p[0], -p[1])

    def _curveToOne(self, a, b, p):
        self.c.curve_to(a[0], -a[1], b[0], -b[1], p[0], -p[1])

    def _closePath(self):
        self.c.close_path()

    def _endPath(self):
        pass


def _outline_to_cairo(hbfont, gid):
    _SCR.identity_matrix()
    _SCR.new_path()
    try:
        hbfont.draw_glyph_with_pen(gid, _Pen(_SCR))
    except Exception:
        pass
    p = _SCR.copy_path()
    _SCR.new_path()
    return p


# ---------------------------------------------------------------------------
# helpers: digits, markup, bidi classes
# ---------------------------------------------------------------------------

_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def to_fa_digits(s) -> str:
    return str(s).translate(_FA_DIGITS)


def parse_markup(text: str):
    """'**شما** انسان‌ها' → ('شما انسان‌ها', {0}). Emphasis is per space-separated word."""
    words = text.split()
    clean, emph = [], set()
    open_ = False
    for i, w in enumerate(words):
        starts = w.startswith("**")
        ends = w.endswith("**") and (len(w) > 2 or not starts)
        if starts:
            open_ = True
        if open_:
            emph.add(i)
        if ends:
            open_ = False
        clean.append(w.replace("**", ""))
    return " ".join(clean), emph


def _is_rtl_char(c):
    o = ord(c)
    if 0x066A <= o <= 0x066C or o == 0x060C and False:
        return False  # ٪ ٫ ٬ behave like number punctuation
    return (0x0590 <= o <= 0x08FF or 0xFB1D <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF) and not _is_digit(c)


def _is_digit(c):
    o = ord(c)
    return 0x30 <= o <= 0x39 or 0x660 <= o <= 0x669 or 0x6F0 <= o <= 0x6F9


_NUM_JOIN = set(".,:/٫٬")
_NUM_AFFIX = set("%٪$€£#+-−°")


def _classes(text):
    """Per char: 'R', 'L', 'N' (number, rendered LTR), or None (neutral)."""
    cls = []
    for c in text:
        if _is_digit(c):
            cls.append("N")
        elif _is_rtl_char(c):
            cls.append("R")
        elif unicodedata.category(c)[0] in "LM" and c not in "\u200c\u200d":
            cls.append("L")
        else:
            cls.append(None)
    n = len(text)
    # number separators between digits and affixes adjacent to digits join the number
    for i, c in enumerate(text):
        if cls[i] is None:
            if c in _NUM_JOIN and 0 < i < n - 1 and cls[i - 1] == "N" and cls[i + 1] == "N":
                cls[i] = "N"
    for i, c in enumerate(text):
        if cls[i] is None and c in _NUM_AFFIX:
            if (i > 0 and cls[i - 1] == "N") or (i < n - 1 and cls[i + 1] == "N"):
                cls[i] = "N"
    return cls


def _resolve_levels(text, base_rtl):
    cls = _classes(text)
    n = len(text)
    strong = ["R" if c == "N" else c for c in cls]  # numbers act as R for neutrals (N1)
    if not base_rtl:
        strong = [("L" if c == "N" else c) for c in cls]
    levels = [0] * n
    base = 1 if base_rtl else 0
    for i in range(n):
        c = cls[i]
        if c is None:
            # neutral: look left/right for strong
            l = next((strong[j] for j in range(i - 1, -1, -1) if strong[j]), None)
            r = next((strong[j] for j in range(i + 1, n) if strong[j]), None)
            d = l if (l is not None and l == r) else ("R" if base_rtl else "L")
            levels[i] = (1 if d == "R" else 0) if not base_rtl else (1 if d == "R" else 2)
        elif c == "R":
            levels[i] = 1
        elif c == "L":
            levels[i] = 2 if base_rtl else 0
        else:  # number
            levels[i] = 2 if base_rtl or (i > 0 and levels[i - 1] == 1) else 0
    if not base_rtl:
        # neutral resolution for LTR base yields 0/1 already
        pass
    return levels, base


def fallback_chars(text, weight="body"):
    """Visible chars the chosen font lacks → they would render in a fallback font (off-brand)."""
    f = _font(weight)
    out = set()
    for c in text:
        if c.isspace() or unicodedata.category(c) in ("Cf", "Mn") or f.has(c):
            continue
        out.add(c)
    return out


def missing_glyphs(text, weight="bold"):
    chain = [_font(weight)] + [_font(p) for p in FALLBACKS if os.path.exists(p)]
    out = set()
    for c in text:
        if c.isspace() or unicodedata.category(c) in ("Cf", "Mn"):
            continue
        if not any(f.has(c) for f in chain):
            out.add(c)
    return out


# ---------------------------------------------------------------------------
# shaping & layout
# ---------------------------------------------------------------------------

class Line:
    __slots__ = ("glyphs", "width", "words", "text", "word_offset", "_ink")

    def __init__(self):
        self._ink = None
        self.glyphs = []   # (font, gid, x, y) in font-independent units of EM=1
        self.width = 0.0
        self.words = {}    # word_idx -> [x0, x1] (EM units)
        self.text = ""
        self.word_offset = 0


class Layout:
    __slots__ = ("lines", "size", "line_height", "align", "width", "height", "n_words")

    def size_px(self):
        return self.width * self.size, self.height * self.size


def _font_for(c, chain):
    if c in "\u200c\u200d\u200e\u200f":       # joiners are invisible: never switch font for them
        return chain[0]
    for f in chain:
        if f.has(c):
            return f
    return chain[0]



# ---------------------------------------------------------------------------
# Pillow + HarfBuzz/raqm Fallback when uharfbuzz C-extension is not installed
# ---------------------------------------------------------------------------
_PIL_SURF_CACHE = {}

def _pil_measure_word(w_str, f_obj, ref_px=100):
    pf = f_obj.pil_font(ref_px)
    dummy = Image.new("L", (8, 8), 0)
    dd = ImageDraw.Draw(dummy)
    is_rtl = any(_is_rtl_char(c) for c in w_str)
    bb = dd.textbbox((0, 0), w_str, font=pf, direction="rtl" if is_rtl else "ltr", language="fa" if is_rtl else None)
    return max(2.0, float(bb[2] - bb[0])) / float(ref_px)

def _shape_line_pil(text, weight, word_offset, base_rtl=None):
    line = Line()
    line.text = text
    line.word_offset = word_offset
    if not text:
        return line
    f = _font(weight)
    words = text.split()
    if not words:
        return line
    if base_rtl is None:
        first = next((c for c in text if _is_rtl_char(c) or (unicodedata.category(c)[0] == "L")), "")
        base_rtl = bool(first) and _is_rtl_char(first)
    gap_em = 0.26
    w_ems = [_pil_measure_word(w, f) for w in words]
    total_em = sum(w_ems) + gap_em * max(0, len(words) - 1)
    line.width = total_em
    # Store word spans in visual left-to-right order (for RTL, word 0 is on the right!)
    if base_rtl:
        cur_right = total_em
        for idx, (w_str, w_em) in enumerate(zip(words, w_ems)):
            x0 = cur_right - w_em
            x1 = cur_right
            line.words[word_offset + idx] = [x0, x1]
            line.glyphs.append((f, w_str, x0, 0.0, word_offset + idx))
            cur_right -= (w_em + gap_em)
    else:
        cur_left = 0.0
        for idx, (w_str, w_em) in enumerate(zip(words, w_ems)):
            x0 = cur_left
            x1 = cur_left + w_em
            line.words[word_offset + idx] = [x0, x1]
            line.glyphs.append((f, w_str, x0, 0.0, word_offset + idx))
            cur_left += (w_em + gap_em)
    return line

def _get_pil_word_surface(f_obj, w_str, px_size, col_rgba):
    r8, g8, b8 = int(round(col_rgba[0]*255)), int(round(col_rgba[1]*255)), int(round(col_rgba[2]*255))
    key = (f_obj.path, w_str, int(round(px_size)), r8, g8, b8)
    if key in _PIL_SURF_CACHE:
        return _PIL_SURF_CACHE[key]
    if any(not f_obj.has(c) for c in w_str if not c.isspace() and unicodedata.category(c) not in ("Cf", "Mn")):
        from .text import FALLBACKS, _font
        for fb_p in FALLBACKS:
            if os.path.exists(fb_p) and all(_font(fb_p).has(c) for c in w_str if not c.isspace() and unicodedata.category(c) not in ("Cf", "Mn")):
                f_obj = _font(fb_p)
                break
    pf = f_obj.pil_font(px_size)
    is_rtl = any(_is_rtl_char(c) for c in w_str)
    dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dummy)
    bb = dd.textbbox((0, 0), w_str, font=pf, anchor="ls", direction="rtl" if is_rtl else "ltr", language="fa" if is_rtl else None)
    tw, th = max(2, bb[2] - bb[0]), max(2, bb[3] - bb[1])
    pad = 8
    im = Image.new("RGBA", (tw + pad*2, th + pad*2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((pad - bb[0], pad - bb[1]), w_str, font=pf, fill=(b8, g8, r8, 255),
              anchor="ls", direction="rtl" if is_rtl else "ltr", language="fa" if is_rtl else None)
    arr = np.array(im, dtype=np.uint8)
    af = arr[:, :, 3:4].astype(np.float32) / 255.0
    arr[:, :, :3] = (arr[:, :, :3].astype(np.float32) * af).astype(np.uint8)
    _PIL_SURF_CACHE[key] = (arr, pad - bb[0], pad - bb[1])
    return _PIL_SURF_CACHE[key]

def _shape_line(text, weight, word_offset, base_rtl=None):
    if not _HAS_UHARFBUZZ:
        return _shape_line_pil(text, weight, word_offset, base_rtl)
    line = Line()
    line.text = text
    line.word_offset = word_offset
    if not text:
        return line
    chain = [_font(weight)] + [_font(p) for p in FALLBACKS if os.path.exists(p)]
    if base_rtl is None:
        first = next((c for c in text if _is_rtl_char(c) or (unicodedata.category(c)[0] == "L")), "")
        base_rtl = bool(first) and _is_rtl_char(first)
    levels, base = _resolve_levels(text, base_rtl)

    # word index per char (logical)
    widx, w, prev_space = [], 0, True
    for c in text:
        if c.isspace():
            widx.append(-1); prev_space = True
        else:
            if prev_space and widx and any(i >= 0 for i in widx):
                w += 1
            prev_space = False
            widx.append(w)

    # itemise into runs (same level + same font)
    runs = []
    i = 0
    n = len(text)
    while i < n:
        lv = levels[i]
        f = _font_for(text[i], chain) if not text[i].isspace() else None
        j = i + 1
        while j < n and levels[j] == lv:
            fj = _font_for(text[j], chain) if not text[j].isspace() else None
            if f is None:
                f = fj
            elif fj is not None and fj is not f:
                break
            j += 1
        runs.append([i, j, lv, f or chain[0]])
        i = j

    # L2: reverse runs from highest level to lowest odd level
    max_lv = max(r[2] for r in runs)
    min_odd = min((r[2] for r in runs if r[2] % 2 == 1), default=None)
    order = list(range(len(runs)))
    if min_odd is not None:
        for lv in range(max_lv, min_odd - 1, -1):
            k = 0
            while k < len(order):
                if runs[order[k]][2] >= lv:
                    s = k
                    while k < len(order) and runs[order[k]][2] >= lv:
                        k += 1
                    order[s:k] = order[s:k][::-1]
                else:
                    k += 1

    x = 0.0
    for ri in order:
        s, e, lv, f = runs[ri]
        seg = text[s:e]
        buf = hb.Buffer()
        buf.add_str(seg)
        buf.guess_segment_properties()
        buf.direction = "rtl" if lv % 2 == 1 else "ltr"
        hb.shape(f.hb, buf, {"kern": True, "liga": True})
        inv = 1.0 / f.upem
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            ci = s + info.cluster
            gx = x + pos.x_offset * inv
            gy = -pos.y_offset * inv
            adv = pos.x_advance * inv
            wi = widx[ci] if ci < len(widx) else -1
            line.glyphs.append((f, info.codepoint, gx, gy, wi + word_offset if wi >= 0 else -1))
            if wi >= 0:
                k = wi + word_offset
                if k in line.words:
                    line.words[k][0] = min(line.words[k][0], x)
                    line.words[k][1] = max(line.words[k][1], x + adv)
                else:
                    line.words[k] = [x, x + adv]
            x += adv
    line.width = x
    return line


@lru_cache(maxsize=4096)
def _shape_cached(text, weight, word_offset):
    return _shape_line(text, weight, word_offset)


def layout_text(text: str, size: float, weight="bold", max_width=None, line_height=1.55,
                align="center") -> Layout:
    """Shape + wrap. Lines are wrapped by words in LOGICAL order (correct for RTL)."""
    return _layout_cached(text, size, concrete_font(weight), max_width, line_height, align)


@lru_cache(maxsize=2048)
def _layout_cached(text, size, weight, max_width, line_height, align):
    paragraphs = text.split("\n")
    lines = []
    wo = 0
    for para in paragraphs:
        words = para.split()
        if max_width is None or not words:
            ln = _shape_cached(" ".join(words), weight, wo)
            lines.append(ln)
            wo += len(words)
            continue
        cur = []
        for wd in words:
            trial = " ".join(cur + [wd])
            if cur and _shape_cached(trial, weight, wo).width * size > max_width:
                lines.append(_shape_cached(" ".join(cur), weight, wo))
                wo += len(cur)
                cur = [wd]
            else:
                cur.append(wd)
        if cur:
            lines.append(_shape_cached(" ".join(cur), weight, wo))
            wo += len(cur)
    L = Layout()
    L.lines, L.size, L.line_height, L.align = lines, size, line_height, align
    L.width = max((ln.width for ln in lines), default=0.0)
    L.height = line_height * (len(lines) - 1) + 1.0 if lines else 0.0
    L.n_words = wo
    return L


def measure_text(text, size, weight="bold", max_width=None, line_height=1.55):
    L = layout_text(text, size, weight, max_width, line_height)
    return L.width * size, L.height * size


# vertical metrics (EM units): Persian body sits ~0.0..0.62 above baseline
_BASELINE_FROM_CENTER = 0.30


def _line_origin(L, idx, x, y, align, valign):
    n = len(L.lines)
    lh = L.line_height
    block_h = lh * (n - 1)
    if valign == "middle":
        top_baseline = y - block_h * L.size / 2 + _BASELINE_FROM_CENTER * L.size
    elif valign == "top":
        top_baseline = y + 0.82 * L.size
    else:  # bottom / baseline of last line
        top_baseline = y - block_h * L.size
    ln = L.lines[idx]
    w = ln.width * L.size
    if align == "center":
        ox = x - w / 2
    elif align == "right":
        ox = x - w
    else:
        ox = x
    return ox, top_baseline + idx * lh * L.size


# ---------------------------------------------------------------------------
# AUDIT (used by QA): while AUDIT is a list, every glyph group draw_text paints is checked for
#   * clip  — real ink outside the active clip rect (dots / tails cut by a mask, text cut by its slab)
#   * contrast — WCAG ratio of the glyph colour vs the pixels already under it
# AUDIT_TAG names the element being drawn (set by render.draw_world / hud).
# ---------------------------------------------------------------------------
AUDIT = None
AUDIT_TAG = ""
AUDIT_MUTE = None      # v8: None | "clip" (intentional clip: odometer wheels) | "all" (region transition frame)


def _audit_group(ctx, col, a, text, ink_dev, quiet=False):
    import numpy as _np
    x0, y0, x1, y1 = ink_dev
    if x1 - x0 < 2 or y1 - y0 < 2 or AUDIT_MUTE == "all":
        return
    cx0, cy0, cx1, cy1 = ctx.clip_extents()
    d0x, d0y = ctx.user_to_device(cx0, cy0); d1x, d1y = ctx.user_to_device(cx1, cy1)
    cx0, cx1 = min(d0x, d1x), max(d0x, d1x); cy0, cy1 = min(d0y, d1y), max(d0y, d1y)
    surf = ctx.get_target()
    W, H = surf.get_width(), surf.get_height()
    tol = 1.5
    cut = max(cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1)
    on_screen = x1 > 0 and y1 > 0 and x0 < W and y0 < H
    # ignore clips that are just the canvas itself, and text that is clipped because it leaves the frame
    canvas_clip = cx0 <= 0.5 and cy0 <= 0.5 and cx1 >= W - 0.5 and cy1 >= H - 0.5
    if on_screen and not canvas_clip and cut > tol and AUDIT_MUTE != "clip":
        if a > 0.3:
            AUDIT.append({"kind": "clip", "tag": AUDIT_TAG, "text": text, "px": round(cut, 1),
                          "y": round(y0, 1)})
    if a < 0.95 or not on_screen or not isinstance(surf, cairo.ImageSurface):
        return
    from .contrast import ratio
    surf.flush()
    stride = surf.get_stride()
    buf = _np.frombuffer(surf.get_data(), _np.uint8).reshape(H, stride // 4, 4)
    X0, Y0, X1, Y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    reg = buf[Y0:Y1:2, X0:X1:2, :3].reshape(-1, 3)
    if not len(reg):
        return
    b, g, r = _np.median(reg, axis=0) / 255.0
    AUDIT.append({"kind": "contrast", "tag": AUDIT_TAG, "text": text, "quiet": bool(quiet),
                  "ratio": round(ratio(col, (r, g, b, 1.0)), 2)})


# ---------------------------------------------------------------------------
# REAL ink extents (dots, kasra, descending ی/ن tails …) — never guess with fixed ratios
# ---------------------------------------------------------------------------
_INK_CTX = None


def line_ink(ln):
    """(x0, top, x1, bottom) of the real glyph outlines of a shaped line, EM units, relative to the
    line origin / baseline (y down).  Cached on the line."""
    global _INK_CTX
    if ln._ink is not None:
        return ln._ink
    if not ln.glyphs:
        ln._ink = (0.0, -0.72, ln.width, 0.38); return ln._ink
    if not _HAS_UHARFBUZZ:
        f0 = ln.glyphs[0][0]
        pf = f0.pil_font(100)
        dd = ImageDraw.Draw(Image.new("L", (8, 8), 0))
        is_rtl = any(_is_rtl_char(c) for c in ln.text)
        bb = dd.textbbox((0, 0), ln.text, font=pf, anchor="ls", direction="rtl" if is_rtl else "ltr", language="fa" if is_rtl else None)
        ln._ink = (bb[0] / 100.0, bb[1] / 100.0, max(ln.width, bb[2] / 100.0), bb[3] / 100.0)
        return ln._ink
    if _INK_CTX is None:
        _INK_CTX = cairo.Context(cairo.ImageSurface(cairo.FORMAT_A8, 1, 1))
    c = _INK_CTX
    c.new_path(); c.identity_matrix()
    for (f, gid, gx, gy, _wi) in ln.glyphs:
        c.save(); c.translate(gx * 1000, gy * 1000); k = 1000 / f.upem; c.scale(k, k)
        c.append_path(f.glyph_path(gid)); c.restore()
    x0, y0, x1, y1 = c.fill_extents() if c.has_current_point() else (0, -720, ln.width * 1000, 380)
    c.new_path()
    ln._ink = (x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000)
    return ln._ink


def ink_box(text, x, y, size=40, weight="bold", align="center", valign="middle", max_width=None,
            line_height=1.55):
    """Canvas bbox of the actual painted glyphs (what the eye sees)."""
    L = layout_text(text, size, weight, max_width, line_height)
    if not L.lines:
        return (x, y, x, y)
    x0 = y0 = 1e9; x1 = y1 = -1e9
    for i, ln in enumerate(L.lines):
        ox, by = _line_origin(L, i, x, y, align, valign)
        a, t, b, btm = line_ink(ln)
        x0 = min(x0, ox + a * size); x1 = max(x1, ox + b * size)
        y0 = min(y0, by + t * size); y1 = max(y1, by + btm * size)
    return (x0, y0, x1, y1)


def mask_rect(text, x, y, size=40, weight="bold", align="center", valign="middle", max_width=None,
              line_height=1.55, pad=0.14, full_width=True):
    """Clip rectangle for a 'rise from a mask' reveal that NEVER cuts dots or descenders:
    real ink box + pad·size vertically (horizontally the whole canvas unless full_width=False).
    Returns (x, y, w, h)."""
    b = ink_box(text, x, y, size, weight, align, valign, max_width, line_height)
    p = pad * size
    if full_width:
        return (-4000, b[1] - p, 9000, (b[3] - b[1]) + 2 * p)
    return (b[0] - p, b[1] - p, (b[2] - b[0]) + 2 * p, (b[3] - b[1]) + 2 * p)


def text_bbox(text, x, y, size=40, weight="bold", align="center", valign="middle", max_width=None,
              line_height=1.55):
    L = layout_text(text, size, weight, max_width, line_height)
    if not L.lines:
        return (x, y, x, y)
    x0 = y0 = 1e9; x1 = y1 = -1e9
    for i, ln in enumerate(L.lines):
        ox, by = _line_origin(L, i, x, y, align, valign)
        a, t, b, btm = line_ink(ln)
        x0 = min(x0, ox, ox + a * size); x1 = max(x1, ox + ln.width * size, ox + b * size)
        y0 = min(y0, by - 0.72 * size, by + t * size); y1 = max(y1, by + 0.38 * size, by + btm * size)
    return (x0, y0, x1, y1)


def word_boxes(text, x, y, size=40, weight="bold", align="center", valign="middle", max_width=None,
               line_height=1.55):
    """{word_idx: (x0,y0,x1,y1)} in canvas coords — use to point callouts at a specific word."""
    L = layout_text(text, size, weight, max_width, line_height)
    out = {}
    for i, ln in enumerate(L.lines):
        ox, by = _line_origin(L, i, x, y, align, valign)
        for k, (a, b) in ln.words.items():
            out[k] = (ox + a * size, by - 0.72 * size, ox + b * size, by + 0.38 * size)
    return out


def draw_text(ctx, text, x, y, size=40, color="#ffffff", weight="body", align="center",
              valign="middle", alpha=1.0, max_width=None, line_height=1.55,
              word_style=None, reveal=None, stroke=None, stroke_width=0.0, quiet=False,
              style=None, kashida=None):
    """
    Draw vector text.  `word_style(i) -> dict(color=?, alpha=?, dy=?, scale=?)` styles words
    individually (karaoke, per-word pop-in).  `reveal=(mode, p)` with mode in
      'wipe'  : RTL-aware clip wipe (right→left for Persian),
      'words' : words appear one by one (logical order) with a soft rise,
      'fade'  : simple alpha.
    v9: `style` = None | "solid" | "outline" | "extrude_3d" | "knockout" | dict (see text_style()),
        `kashida` = {"px": stretch_px, "word": "کش", "after": None} — live Persian tatweel stretch.
    Returns the bbox (x0,y0,x1,y1).
    """
    if not text or alpha <= 0.002:
        return (x, y, x, y)
    tsty = text_style(style, size)
    if kashida and _HAS_UHARFBUZZ:
        text = kashida_text(text, kashida.get("word"), kashida.get("after"))
    L = layout_text(text, size, weight, max_width, line_height, align)
    kpx = float(kashida.get("px", 0.0)) if (kashida and _HAS_UHARFBUZZ) else 0.0
    kword = _kashida_word_index(text, kashida.get("word")) if kpx > 0.01 else None
    base_col = _color(color)
    s = size
    rtl = any(_is_rtl_char(c) for c in text)
    mode, rp = (reveal if reveal else (None, 1.0))
    if mode == "fade":
        alpha *= clamp(rp)
    if alpha <= 0.002:
        return (x, y, x, y)

    bx0 = by0 = 1e9; bx1 = by1 = -1e9
    ctx.save()
    if mode == "wipe" and rp < 0.999:
        bb = text_bbox(text, x, y, size, weight, align, valign, max_width, line_height)
        w = (bb[2] - bb[0] + 8) * clamp(rp)
        if rtl:
            ctx.rectangle(bb[2] + 4 - w, bb[1] - size, w, bb[3] - bb[1] + 2 * size)
        else:
            ctx.rectangle(bb[0] - 4, bb[1] - size, w, bb[3] - bb[1] + 2 * size)
        ctx.clip()

    knock = tsty is not None and tsty["kind"] == "knockout" and _HAS_UHARFBUZZ
    if knock:
        _knockout_begin(ctx, text, x, y, size, weight, align, valign, max_width, line_height, tsty, base_col, alpha,
                        kpx)
    n_words = max(1, L.n_words)
    for li, ln in enumerate(L.lines):
        ox, by = _line_origin(L, li, x, y, align, valign)
        lglyphs, extra = ln.glyphs, 0.0
        if kpx > 0.01 and _HAS_UHARFBUZZ:
            max_allow_em = max(0.0, ((max_width or 990.0) / s) - ln.width)
            eff_em = min(kpx / s, max_allow_em) if (ln.width * s < 990.0) else (kpx / s)
            lglyphs, extra = _kashida_glyphs(ln, eff_em, kword)
            ox -= {"center": extra / 2, "right": extra}.get(align, 0.0) * s
        bx0 = min(bx0, ox); bx1 = max(bx1, ox + (ln.width + extra) * s)
        by0 = min(by0, by - 0.72 * s); by1 = max(by1, by + 0.38 * s)
        # group glyphs by style so we fill once per group (fast)
        groups = {}
        for g in lglyphs:
            f, gid, gx, gy, wi = g[:5]
            if word_style is not None or mode == "words":
                key = wi
            else:
                key = 0
            groups.setdefault(key, []).append((f, gid, gx, gy, g[5] if len(g) > 5 else 1.0))
        for key, glyphs in groups.items():
            col, a, dy, sc = base_col, alpha, 0.0, 1.0
            _quiet = quiet
            if mode == "words" and key >= 0:
                wp = clamp(rp * (n_words + 1.5) - key)
                a *= wp
                dy += (1 - wp) * 0.25 * s
            if word_style is not None and key >= 0:
                st = word_style(key) or {}
                if "color" in st:
                    col = _color(st["color"])
                a *= st.get("alpha", 1.0)
                dy += st.get("dy", 0.0)
                sc = st.get("scale", 1.0)
                _quiet = st.get("quiet", _quiet)
            a *= col[3]
            if a <= 0.002:
                continue
            ctx.save()
            if sc != 1.0 and key in ln.words:
                wx0, wx1 = ln.words[key]
                pcx = ox + (wx0 + wx1) / 2 * s
                ctx.translate(pcx, by - 0.3 * s)
                ctx.scale(sc, sc)
                ctx.translate(-pcx, -(by - 0.3 * s))
            if not _HAS_UHARFBUZZ:
                for (f, w_str, gx, gy, _sx) in glyphs:
                    arr, x_anc, y_anc = _get_pil_word_surface(f, w_str, s * 2.0, col)
                    sh, sw = arr.shape[:2]
                    wsurf = cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_ARGB32, sw, sh)
                    ctx.save()
                    ctx.translate(ox + gx * s - x_anc * 0.5, by + gy * s - y_anc * 0.5 + dy)
                    ctx.scale(0.5, 0.5)
                    ctx.set_source_surface(wsurf, 0, 0)
                    ctx.paint_with_alpha(clamp(a))
                    ctx.restore()
            else:
                ctx.new_path()
                for (f, gid, gx, gy, gsx) in glyphs:
                    ctx.save()
                    ctx.translate(ox + gx * s, by + gy * s + dy)
                    k = s / f.upem
                    if gsx > 1.02 and gid == _tatweel_gid(f):
                        # Draw a flat, non-tapered baseline bar matching the exact vertical ink bounds of uni0640
                        # so stretching by 10x-20x never stretches rounded end-caps into V-notches!
                        ym0, ym1 = _tatweel_ybounds(f, gid)
                        adv_u = _adv_em(f, gid) * f.upem
                        pad_u = 18.0
                        ctx.scale(k, k)
                        ctx.rectangle(-pad_u, ym0, adv_u * gsx + pad_u * 2.0, ym1 - ym0)
                    else:
                        ctx.scale(k * gsx, k)
                        ctx.append_path(f.glyph_path(gid))
                    ctx.restore()
                if AUDIT is not None:
                    ex = ctx.fill_extents()
                    p0 = ctx.user_to_device(ex[0], ex[1]); p1 = ctx.user_to_device(ex[2], ex[3])
                    _ws = text.split()
                    _wtxt = _ws[key] if (word_style is not None or mode == "words") and 0 <= key < len(_ws) else text
                    _audit_group(ctx, col, a, _wtxt, (min(p0[0], p1[0]), min(p0[1], p1[1]),
                                                      max(p0[0], p1[0]), max(p0[1], p1[1])), _quiet)
                if tsty is not None and tsty["kind"] != "solid":
                    _paint_styled(ctx, tsty, col, a)
                    ctx.restore()
                    continue
                ctx.set_source_rgba(col[0], col[1], col[2], a)
                if stroke is not None and stroke_width > 0:
                    ctx.fill_preserve()
                    sc_ = _color(stroke)
                    ctx.set_source_rgba(sc_[0], sc_[1], sc_[2], sc_[3] * a)
                    ctx.set_line_width(stroke_width)
                    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                    ctx.stroke()
                else:
                    ctx.fill()
            ctx.restore()
    if knock:
        ctx.pop_group_to_source()
        ctx.paint()
    ctx.restore()
    if tsty is not None and tsty["kind"] == "extrude_3d":
        bx1 += tsty["dx"] * tsty["depth"]; by1 += tsty["dy"] * tsty["depth"]
    return (bx0, by0, bx1, by1)



# ---------------------------------------------------------------------------
# v9 — typography treatments: solid | outline | extrude_3d | knockout
# ---------------------------------------------------------------------------
def text_style(style, size):
    """Normalise a text style → dict or None (solid fast path).
    "outline"              hollow vector stroke     keys: stroke_w, reveal (0..1 trim-path), fill_alpha
    "extrude_3d"           directional block relief keys: depth (px), angle (deg, 45 = down-right), extrude
                           (colour of the far side), shade (0..1 darkening toward the back), face_stroke
    "knockout"             slab with the glyphs CUT OUT (background shows through)  keys: pad, radius
    """
    if style is None or style == "solid":
        return None
    if isinstance(style, str):
        style = {"kind": style}
    k = style.get("kind") or style.get("style") or "solid"
    if k == "solid":
        return None
    out = {"kind": k}
    out["stroke_w"] = float(style.get("stroke_w", max(2.0, size * 0.035)))
    out["reveal"] = float(style.get("reveal", 1.0))
    out["fill_alpha"] = float(style.get("fill_alpha", 0.0))
    out["fill"] = style.get("fill")
    d = float(style.get("depth", size * 0.14))
    ang = math.radians(float(style.get("angle", 45)))
    out["depth"] = max(0.0, d) * float(style.get("depth_p", 1.0))
    out["dx"], out["dy"] = math.cos(ang), math.sin(ang)
    out["extrude"] = _color(style.get("extrude", "#000000"))
    out["shade"] = float(style.get("shade", 0.45))
    out["face_stroke"] = style.get("face_stroke")
    out["pad"] = float(style.get("pad", size * 0.22))
    out["radius"] = float(style.get("radius", size * 0.12))
    return out


def _flat_subpaths(ctx):
    """Current path → list of flattened sub-paths [(points, closed)] (device-independent user coords)."""
    subs, cur, closed = [], [], False
    for kind, pts in ctx.copy_path_flat():
        if kind == cairo.PATH_MOVE_TO:
            if len(cur) > 1:
                subs.append((cur, closed))
            cur, closed = [pts], False
        elif kind == cairo.PATH_LINE_TO:
            cur.append(pts)
        elif kind == cairo.PATH_CLOSE_PATH:
            closed = True
    if len(cur) > 1:
        subs.append((cur, closed))
    return subs


def _stroke_reveal(ctx, p):
    """Trim-path on glyph contours: every contour draws simultaneously from its start point (pen feel)."""
    subs = _flat_subpaths(ctx)
    ctx.new_path()
    for pts, closed in subs:
        if closed:
            pts = pts + [pts[0]]
        L = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        if L <= 1e-3:
            continue
        ctx.move_to(*pts[0])
        for q in pts[1:]:
            ctx.line_to(*q)
        ctx.set_dash([max(1e-3, L * p), L * 2 + 10], 0)
        ctx.stroke()
    ctx.set_dash([], 0)


def _paint_styled(ctx, st, col, a):
    """The current path is one glyph group; paint it with the v9 style."""
    k = st["kind"]
    if k == "outline":
        path = ctx.copy_path()
        if st["fill_alpha"] > 0:
            fc = _color(st["fill"]) if st.get("fill") else col
            ctx.set_source_rgba(fc[0], fc[1], fc[2], a * st["fill_alpha"] * fc[3])
            ctx.fill_preserve()
        ctx.set_source_rgba(col[0], col[1], col[2], a)
        ctx.set_line_width(st["stroke_w"])
        ctx.set_line_join(cairo.LINE_JOIN_ROUND); ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        if st["reveal"] < 0.999:
            _stroke_reveal(ctx, clamp(st["reveal"]))
        else:
            ctx.stroke()
        ctx.new_path()
        del path
        return
    if k == "extrude_3d":
        path = ctx.copy_path()
        ctx.new_path()
        d = st["depth"]
        n = max(1, int(d / 1.6))
        e = st["extrude"]
        for i in range(n, 0, -1):          # back → front: far layers darker
            f = i / n
            sh = 1 - st["shade"] * f
            ctx.save()
            ctx.translate(st["dx"] * d * f, st["dy"] * d * f)
            ctx.append_path(path)
            ctx.set_source_rgba(e[0] * sh, e[1] * sh, e[2] * sh, a * e[3])
            ctx.fill()
            ctx.restore()
        ctx.append_path(path)
        ctx.set_source_rgba(col[0], col[1], col[2], a)
        if st.get("face_stroke"):
            fs = _color(st["face_stroke"])
            ctx.fill_preserve()
            ctx.set_source_rgba(fs[0], fs[1], fs[2], a * fs[3]); ctx.set_line_width(st["stroke_w"] * 0.6)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND); ctx.stroke()
        else:
            ctx.fill()
        return
    if k == "knockout":                    # glyphs punch a hole in the slab painted by _knockout_begin
        op = ctx.get_operator()
        ctx.set_operator(cairo.OPERATOR_DEST_OUT)
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.fill()
        ctx.set_operator(op)
        return
    ctx.set_source_rgba(col[0], col[1], col[2], a)
    ctx.fill()


def _knockout_begin(ctx, text, x, y, size, weight, align, valign, max_width, line_height, st, col, alpha, kpx):
    b = ink_box(text, x, y, size, weight, align, valign, max_width, line_height)
    p, r = st["pad"], st["radius"]
    x0, y0, x1, y1 = b[0] - p - kpx / 2, b[1] - p * 0.8, b[2] + p + kpx / 2, b[3] + p * 0.8
    ctx.push_group()
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    ctx.new_sub_path()
    ctx.arc(x1 - r, y0 + r, r, -math.pi / 2, 0); ctx.arc(x1 - r, y1 - r, r, 0, math.pi / 2)
    ctx.arc(x0 + r, y1 - r, r, math.pi / 2, math.pi); ctx.arc(x0 + r, y0 + r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()
    ctx.set_source_rgba(col[0], col[1], col[2], col[3] * alpha)
    ctx.fill()


# ---------------------------------------------------------------------------
# v9 — live Persian kashida / tatweel stretch
# ---------------------------------------------------------------------------
_TATWEEL = "\u0640"
_NO_LEFT_JOIN = set("اآأإٱدذرزژوؤةۀ\u200c ")


def _is_ar_letter(c):
    return "\u0620" <= c <= "\u064A" or "\u066E" <= c <= "\u06D3" or c in "پچژکگی"


def kashida_text(text, word=None, after=None):
    """Insert ONE tatweel at the first legal join of `word` (or of the first word that has one) unless the
    text already contains tatweels.  after = char index inside the word to stretch after."""
    if _TATWEEL in text:
        return text
    words = text.split(" ")
    from .scene import _norm_word
    idx = None
    if word:
        nw = _norm_word(word)
        idx = next((i for i, w in enumerate(words) if _norm_word(w) == nw), None)
        if idx is None:
            idx = next((i for i, w in enumerate(words) if nw and nw in _norm_word(w)), None)
    cand = [idx] if idx is not None else list(range(len(words)))
    for i in cand:
        w = words[i]
        js = [j for j in range(len(w) - 1) if _is_ar_letter(w[j]) and w[j] not in _NO_LEFT_JOIN
              and _is_ar_letter(w[j + 1])]
        if not js:
            continue
        j = js[0]
        if after is not None:
            j = min(js, key=lambda q: abs(q - int(after)))
        words[i] = w[:j + 1] + _TATWEEL + w[j + 1:]
        return " ".join(words)
    return text


def _kashida_word_index(text, word):
    ws = text.split()
    tw = [i for i, w in enumerate(ws) if _TATWEEL in w]
    if word:
        from .scene import _norm_word
        nw = _norm_word(word.replace(_TATWEEL, ""))
        for i in tw:
            if _norm_word(ws[i].replace(_TATWEEL, "")) == nw:
                return i
    return tw[0] if tw else None


def _tatweel_ybounds(f, gid):
    yb = getattr(f, "_tatweel_yb", None)
    if yb is not None:
        return yb
    _SCR.new_path()
    _SCR.append_path(f.glyph_path(gid))
    ex = _SCR.path_extents()
    _SCR.new_path()
    yb = (ex[1], ex[3]) if (ex[3] > ex[1]) else (-0.18 * f.upem, 0.0)
    f._tatweel_yb = yb
    return yb


def _tatweel_gid(f):
    g = getattr(f, "_tatweel", "unset")
    if g == "unset":
        try:
            g = f.hb.get_nominal_glyph(0x0640)
        except Exception:  # noqa: BLE001
            g = None
        f._tatweel = g
    return g


def _kashida_glyphs(ln, S, word_idx):
    """Shaped line → glyph list where the tatweel(s) of word `word_idx` are stretched by S em in total:
    the tatweel bar is scaled horizontally (it is a pure baseline bar, so the letters on both sides stay razor
    sharp) and every glyph to its LEFT is pushed left. Returns (glyphs, extra_em)."""
    tw = [k for k, g in enumerate(ln.glyphs) if g[1] == _tatweel_gid(g[0]) and g[1] is not None
          and (word_idx is None or g[4] == word_idx)]
    if not tw:
        return ln.glyphs, 0.0
    per = S / len(tw)
    txs = [ln.glyphs[q][2] for q in tw]
    out = []
    for k, g in enumerate(ln.glyphs):
        f, gid, gx, gy, wi = g[:5]
        # every stretched bar that starts LEFT of this glyph pushes it right; the bar itself grows in place
        n_left = sum(1 for q, tx in zip(tw, txs) if q != k and tx < gx - 1e-6)
        sx = 1.0
        if k in tw:
            adv = _adv_em(f, gid)
            sx = (adv + per) / max(1e-6, adv)
        out.append((f, gid, gx + per * n_left, gy, wi, sx))
    return out, S


def _adv_em(f, gid):
    c = getattr(f, "_adv", None)
    if c is None:
        c = f._adv = {}
    v = c.get(gid)
    if v is None:
        try:
            v = f.hb.advance(gid) / f.upem if hasattr(f.hb, "advance") else f.hb.get_glyph_h_advance(gid) / f.upem
        except Exception:  # noqa: BLE001
            v = 0.3
        c[gid] = v
    return v
