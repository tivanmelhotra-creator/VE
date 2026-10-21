"""
mograph.v14.glyphs — VECTOR typography you can deform in real 3D.

v13.9 audit: titles were one flat block; curved text was a RASTER warped by cv2.remap (blurry, no 3D); "3D" was a
2D scale+rotate (the dry-bread look).  v14 shapes Persian with HarfBuzz, pulls every glyph OUTLINE as polygons and
pushes every point through a 3D pipeline each frame:

   outline ─► per-glyph local 3D transform (spring pop / flip / jelly)  ─► global Field (arc / cylinder / wave /
   jelly / bulge) ─► block plane rotation ─► perspective projection ─► cairo path  (+ extrusion slices = real depth)

Cursive joins survive every deformation (joins are shared outline edges).

MARKUP (one string any LLM can write):   "آمپول‌های|*لاغری*"     '|' = new line
   plain → thin (light, 1.0×, ink)   *w* → hero (black, 2.0×, accent, glow)   ~w~ → callig (Delbar, 2.1×, accent)
   _w_ → bold (1.25×, ink)           ^w^ → tiny (0.62×, mute)                  !w! → alert (1.6×, red, glow)
   markers may wrap several words: "از آدامس‌های|*بدون قند*|^برای تحریک بزاق^"
Override per layer: "styles": {"hero": {"font": "Morabba-Heavy.ttf", "scale": 2.4, "color": "#FF2E42"}}
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache

import cairo
import numpy as np

from .. import text as _T
try:
    import uharfbuzz as hb
except ImportError:
    from .. import hbc as hb

BASE_STYLES = {
    "thin":   {"font": "YekanBakh-Light.ttf", "scale": 1.0, "color": "@ink", "glow": 0.0},
    "hero":   {"font": "Morabba-Heavy.ttf", "scale": 2.0, "color": "@accent", "glow": 0.6},
    "callig": {"font": "Delbar-Bold.ttf", "scale": 2.1, "color": "@accent", "glow": 0.6},
    "bold":   {"font": "YekanBakh-Heavy.ttf", "scale": 1.25, "color": "@ink", "glow": 0.0},
    "tiny":   {"font": "YekanBakh-Light.ttf", "scale": 0.62, "color": "@mute", "glow": 0.0},
    "alert":  {"font": "Morabba-Heavy.ttf", "scale": 1.6, "color": "@red", "glow": 0.6},
    "body":   {"font": "YekanBakh-Medium.ttf", "scale": 1.0, "color": "@ink", "glow": 0.0},
}
MARKERS = {"*": "hero", "~": "callig", "_": "bold", "^": "tiny", "!": "alert", "=": "hero_white"}
PALETTE = {"ink": "#F6F2EA", "accent": "#FBBF24", "mute": "#B9BDC7", "red": "#FF2E42", "teal": "#2DD4BF",
           "dark": "#121418", "white": "#FFFFFF", "gold": "#FBBF24", "amber": "#F59E0B", "black": "#0B0B0D",
           "accent2": "#F59E0B", "alert": "#FF2E42", "glow": "#FF8A00", "stage": "#120C08", "grid": "#D9C9A8",
           "hg0": "#FFD36B", "hg1": "#FFAE1F", "hg2": "#F07A0A", "ag0": "#FF8A8A", "ag1": "#FF2E42", "ag2": "#C8101E"}


def set_palette(pal):
    """Install a designed palette globally (Renderer does this from meta.palette / palette.auto_palette)."""
    p = dict(pal)
    for i, c in enumerate(p.pop("accent_grad", [])[:3]):
        p[f"hg{i}"] = c
    for i, c in enumerate(p.pop("alert_grad", [])[:3]):
        p[f"ag{i}"] = c
    PALETTE.update(p)


def font_path(fp):
    return fp if os.path.isabs(fp) else os.path.join(_T.FONT_DIR, fp)


def rgba(c, pal=None):
    pal = pal or PALETTE
    if isinstance(c, str) and c.startswith("@"):
        c = pal.get(c[1:], PALETTE.get(c[1:], "#ffffff"))
    if isinstance(c, str):
        h = c.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255,
                int(h[6:8], 16) / 255 if len(h) == 8 else 1.0)
    return tuple(c) if len(c) == 4 else (c[0], c[1], c[2], 1.0)


_SC = cairo.Context(cairo.ImageSurface(cairo.FORMAT_A8, 1, 1))
_SC.set_tolerance(0.15)


@lru_cache(maxsize=4096)
def glyph_contours(fp, gid):
    f = _T._font(fp)
    _SC.identity_matrix()
    _SC.new_path()
    _SC.append_path(f.glyph_path(gid))
    flat = _SC.copy_path_flat()
    _SC.new_path()
    conts, cur = [], []
    for kind, pts in flat:
        if kind == cairo.PATH_MOVE_TO:
            if len(cur) > 2:
                conts.append(np.array(cur, np.float32))
            cur = [pts]
        elif kind == cairo.PATH_LINE_TO:
            cur.append(pts)
        elif kind == cairo.PATH_CLOSE_PATH:
            if len(cur) > 2:
                conts.append(np.array(cur, np.float32))
            cur = []
    if len(cur) > 2:
        conts.append(np.array(cur, np.float32))
    return tuple(c / float(f.upem) for c in conts)


def _is_neutral_ltr(ch):
    o = ord(ch)
    return (0x06F0 <= o <= 0x06F9 or 0x0660 <= o <= 0x066C or o < 0x0590 or ch in "%٪+-−.,/:")


def _shape_single(word, fp):
    f = _T._font(fp)
    buf = hb.Buffer()
    buf.add_str(word)
    buf.guess_segment_properties()
    if all(_is_neutral_ltr(ch) for ch in word):
        buf.direction = "ltr"
    hb.shape(f.hb, buf, {"kern": True, "liga": True})
    inv = 1.0 / f.upem
    out, x = [], 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        out.append((info.codepoint, x + pos.x_offset * inv, -pos.y_offset * inv))
        x += pos.x_advance * inv
    return out, x, buf.direction == "rtl"


def _shape(word, fp):
    if "\u200c" in word:
        parts = [p for p in word.split("\u200c") if p]
        if len(parts) > 1:
            # In RTL, HarfBuzz returns glyphs ordered left-to-right (visual order: last sub-word first)
            sub_res = [_shape_single(p, fp) for p in parts]
            rtl = any(r[2] for r in sub_res)
            ordered = list(reversed(sub_res)) if rtl else sub_res
            out, x = [], 0.0
            zwnj_gap = 0.04
            for idx, (gls, adv, _) in enumerate(ordered):
                for gid, gx, gy in gls:
                    out.append((gid, x + gx, gy))
                x += adv + (zwnj_gap if idx + 1 < len(ordered) else 0.0)
            return out, x, rtl
    return _shape_single(word, fp)


class Glyph:
    __slots__ = ("conts", "cx", "cy", "x0", "x1", "y0", "y1", "word", "line", "order", "style", "sname", "size",
                 "wpos", "nwg", "u", "px", "py", "worder")


class Block:
    def __init__(self):
        self.glyphs, self.words, self.lines = [], [], []
        self.bbox = (0, 0, 0, 0)

    @property
    def w(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self):
        return self.bbox[3] - self.bbox[1]


_TOK = re.compile(r"([*~_^!=])(.+?)\1")


def parse_markup(line):
    out, pos = [], 0
    for m in _TOK.finditer(line):
        out += [(w, "thin") for w in line[pos:m.start()].split()]
        out += [(w, MARKERS[m.group(1)]) for w in m.group(2).split()]
        pos = m.end()
    return out + [(w, "thin") for w in line[pos:].split()]


# ---------------------------------------------------------------------------------------------------------------
# FONT SYSTEMS (research-based: one SUPERFAMILY per lockup, contrast through weight × size × colour, never two
# characterful families fighting).  Every theme maps the markup roles to weights of ONE family; the only
# allowed "second voice" is `callig` (a calligraphic accent word), used max once per video section.
# Hero colour = vertical gold gradient (light top → amber bottom) like the reference; thin = warm white.
# ---------------------------------------------------------------------------------------------------------------
GOLD = ["#FFD36B", "#FFAE1F", "#F07A0A"]
RED = ["#FF8A8A", "#FF2E42", "#C8101E"]
THEMES = {
    # name: (thin, light, bold, hero/black)        character
    "pelak":     ("Pelak-Thin.ttf", "Pelak-light.ttf", "Pelak-Bold.ttf", "Pelak-Black.ttf"),          # modern geometric, wide range (default)
    "doran":     ("Doran-Thin.ttf", "Doran-Light.ttf", "Doran-Bold.ttf", "Doran-ExtraBold.ttf"),      # soft humanist, warm, medical/edu
    "yekanbakh": ("YekanBakh-Hairline.ttf", "YekanBakh-Light.ttf", "YekanBakh-Bold.ttf", "YekanBakh-Fat.ttf"),  # neutral UI, compact
    "morabba":   ("Morabba-UltraLight.ttf", "Morabba-Light.ttf", "Morabba-Bold.ttf", "Morabba-Heavy.ttf"),     # square, punchy, youthful
    "abar":      ("AbarMid-Thin.ttf", "AbarMid-Light.ttf", "AbarMid-Bold.ttf", "AbarMid-ExtraBlack.ttf"),      # editorial, elegant
    "pinar":     ("PINAR-LIGHT.TTF", "PINAR-REGULAR.TTF", "PINAR-BOLD.TTF", "PINAR-DS3-BLACK.TTF"),            # rounded display
}
CALLIG = {"pelak": "Delbar-Bold.ttf", "doran": "FONT_2417.ttf", "yekanbakh": "Delbar-Bold.ttf", "morabba": "Delbar-Bold.ttf",
          "abar": "FONT_2417.ttf", "pinar": "Delbar-Bold.ttf"}       # FONT_2417 = IranNastaliq Bold


FONT_LOCK = {"doran": True}           # user law: Doran ONLY — no second family anywhere in a Doran video
STRETCH = {"doran": 1.12}             # reference Doran reads a little extended (wider) → +12 % horizontal


def theme_styles(theme, over=None):
    """Role → style for one superfamily.  Ratios measured on the @novadent reference frames:
       hero ≈ 1.75× thin (not 2×+), thin line = REGULAR/Light weight at full presence, tiny ≈ 0.6× in Light,
       hero colour gold gradient with an orange outer glow; on red/glass backgrounds use "hero_white"."""
    th, li, bo, he = THEMES.get(theme, THEMES["doran"])
    reg = {"doran": "Doran-Regular.ttf", "pelak": "Pelak-Regular.ttf", "yekanbakh": "YekanBakh-Regular.ttf",
           "morabba": "Morabba-Regular.ttf", "abar": "AbarMid-Regular.ttf", "pinar": "PINAR-REGULAR.TTF"}.get(theme, li)
    st = {"thin": {"font": reg, "color": "@ink", "scale": 1.0},
          "tiny": {"font": li, "color": "@ink", "scale": 0.60, "alpha": 0.86},
          "bold": {"font": bo, "color": "#FFFFFF", "scale": 1.15},
          "hero": {"font": he, "scale": 1.75, "color": "@accent", "gradient": ["@hg0", "@hg1", "@hg2"], "glow": 0.75,
                   "glow_color": "@glow"},
          "hero_white": {"font": he, "scale": 1.75, "color": "#FFFFFF", "glow": 0.35, "glow_color": "#FFFFFF"},
          "alert": {"font": he, "scale": 1.6, "color": "@alert", "gradient": ["@ag0", "@ag1", "@ag2"], "glow": 0.7,
                    "glow_color": "@alert"},
          "callig": {"font": CALLIG.get(theme, "Delbar-Bold.ttf"), "scale": 2.0, "color": "@accent",
                     "gradient": ["@hg0", "@hg1", "@hg2"], "glow": 0.6, "glow_color": "@glow"}}
    if FONT_LOCK.get(theme):
        # ONE family only (user law): callig/accent words become the family's heaviest weight in gold
        st["callig"].update({"font": he, "scale": 1.9})
    stretch = STRETCH.get(theme, 1.0)
    for v in st.values():
        v.setdefault("stretch", stretch)
    for k, v in (over or {}).items():
        v = dict(v)
        if FONT_LOCK.get(theme) and "font" in v and not v["font"].lower().startswith(theme):
            v.pop("font")                 # a layer may NOT smuggle in another family
        st.setdefault(k, dict(st["thin"])).update(v)
    return st


def lock_font(theme, role="thin"):
    """Font file for a role in the theme (for code paths that need a raw font: subtitles, badges, labels)."""
    return theme_styles(theme)[role]["font"]


def merged_styles(over=None):
    st = {k: dict(v) for k, v in BASE_STYLES.items()}
    for k, v in (over or {}).items():
        st.setdefault(k, dict(BASE_STYLES["thin"])).update(v)
    return st


def plain(text):
    """Markup → plain text (for QA / subtitles)."""
    return _TOK.sub(lambda m: m.group(2), text).replace("|", " ")


@lru_cache(maxsize=96)
def _typeset_cached(text, size, styles_key, leading, align, word_gap, tracking, lines_key, lockup, tuck):
    import json
    styles = json.loads(styles_key) if styles_key else None
    lines_cfg = json.loads(lines_key) if lines_key else None
    return _typeset(text, size, styles, leading, align, word_gap, tracking, lines_cfg, lockup, tuck)


DEFAULT_THEME = [None]        # set by the Renderer from meta.theme → EVERY typeset call (subs, disc, badges) obeys it


def typeset(text, size=60, styles=None, leading=0.10, align="center", word_gap=0.28, tracking=0.0, lines=None,
            lockup=True, tuck=0.04, theme=None):
    import json
    theme = theme or DEFAULT_THEME[0]
    if theme:
        styles = theme_styles(theme, styles)
    return _typeset_cached(text, float(size), json.dumps(styles, sort_keys=True) if styles else "",
                           float(leading), align, float(word_gap), float(tracking),
                           json.dumps(lines) if lines else "", bool(lockup), float(tuck))


def _typeset(text, size, styles, leading, align, word_gap, tracking, lines_cfg, lockup=True, tuck=0.18):
    st = merged_styles(styles)
    blk = Block()
    order = wi = 0
    L_all = []
    for li, ln in enumerate(text.split("|")):
        cfg = (lines_cfg[li] if lines_cfg and li < len(lines_cfg) else None) or {}
        lsize = float(cfg.get("size", size))
        placed = []
        for w, sname in parse_markup(ln):
            s = st.get(sname, st["thin"])
            px = lsize * float(s.get("scale", 1.0))
            sx_ = float(s.get("stretch", 1.0))
            gl, adv, rtl = _shape(w, s["font"])
            trk = tracking * px * max(0, len(gl) - 1)
            fp = font_path(s["font"])
            ix0, ix1 = 0.0, adv
            for gid, gx, gy in gl:          # ink overhang beyond the advance box (final ه / ی tails) → real gaps
                for c in glyph_contours(fp, gid):
                    ix0, ix1 = min(ix0, gx + float(c[:, 0].min())), max(ix1, gx + float(c[:, 0].max()))
            placed.append(dict(text=w, sname=sname, style=s, px=px, font=fp, gl=gl, sx=sx_,
                               adv=adv * px * sx_ + trk, rtl=rtl, trk=tracking * px,
                               ovl=max(0.0, -ix0) * px * sx_, ovr=max(0.0, ix1 - adv) * px * sx_))
        line_rtl = any(p["rtl"] for p in placed) or not placed
        gap = word_gap * lsize
        gaps = []
        for a_, b_ in zip(placed, placed[1:]):
            ov = (a_["ovl"] + b_["ovr"]) if line_rtl else (a_["ovr"] + b_["ovl"])
            gaps.append(gap + ov)
        gaps.append(0.0)
        total = sum(p["adv"] for p in placed) + sum(gaps[:-1])
        cur = total if line_rtl else 0.0
        gls, ly0, ly1, lwords = [], 1e9, -1e9, []
        for pi_, p in enumerate(placed):
            x0 = cur - p["adv"] if line_rtl else cur
            cur = x0 - gaps[pi_] if line_rtl else cur + p["adv"] + gaps[pi_]
            n = len(p["gl"])
            objs = []
            for v, (gid, gx, gy) in enumerate(p["gl"]):
                conts = glyph_contours(p["font"], gid)
                if not conts:
                    continue
                ox = x0 + gx * p["px"] * p["sx"] + p["trk"] * v
                scl = np.array([p["px"] * p["sx"], p["px"]], np.float32)
                pts = [c * scl + np.array([ox, gy * p["px"]], np.float32) for c in conts]
                a = np.concatenate(pts)
                g = Glyph()
                g.conts = pts
                g.x0, g.y0, g.x1, g.y1 = float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())
                g.cx, g.cy = (g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2
                g.word, g.line, g.style, g.sname, g.size = wi, li, p["style"], p["sname"], p["px"]
                g.wpos = (n - 1 - v) if p["rtl"] else v
                objs.append(g)
            for k, g in enumerate(sorted(objs, key=lambda g: g.wpos)):
                g.order, g.wpos, g.nwg = order, k, len(objs)
                order += 1
            gls += objs
            wy0 = min((g.y0 for g in objs), default=-lsize * .7)
            wy1 = max((g.y1 for g in objs), default=lsize * .2)
            ly0, ly1 = min(ly0, wy0), max(ly1, wy1)
            wd = dict(text=p["text"], sname=p["sname"], line=li, i=wi, x0=x0, x1=x0 + p["adv"], y0=wy0, y1=wy1,
                      px=p["px"])
            blk.words.append(wd)
            lwords.append(wd)
            wi += 1
        if ly0 > ly1:
            ly0, ly1 = -lsize * .7, lsize * .2
        L_all.append(dict(glyphs=gls, w=total, y0=ly0, y1=ly1, cfg=cfg, size=lsize, words=lwords))
    maxw = max((L["w"] for L in L_all), default=0)
    # ---- LOCKUP composition (the @novadent stagger): thin lines tuck into the hero line's corners
    #      line before the hero → top-RIGHT (RTL reading start), line after → bottom-LEFT, overlapping the
    #      hero's ascender / descender zone instead of sitting in a centred stack.
    for L in L_all:
        hp = [w["px"] for w in L["words"] if w["sname"] in ("hero", "callig", "alert", "hero_white")]
        L["hero"] = bool(hp)
        L["hpx"] = max(hp) if hp else L["size"]
        if L["hero"]:                       # optical body of a heavy Persian word (ignore alef / kaf tops, ی tails)
            L["oy0"] = max(L["y0"], -0.60 * L["hpx"])
            L["oy1"] = min(L["y1"], 0.16 * L["hpx"])
        else:
            L["oy0"], L["oy1"] = L["y0"], L["y1"]
    hi = next((i for i, L in enumerate(L_all) if L["hero"]), None)
    use_lock = lockup and hi is not None and len(L_all) > 1

    def _xalign(li, L):
        al = L["cfg"].get("align", align)
        if use_lock and not L["hero"] and "align" not in L["cfg"]:
            al = "right" if li < hi else "left"
        inset = (maxw - L["w"]) * 0.06 if use_lock and not L["hero"] else 0.0
        return {"center": (maxw - L["w"]) / 2, "right": maxw - L["w"] - inset}.get(al, inset) + float(L["cfg"].get("dx", 0))

    def _profile(L, dx, bottom, x0, x1, nb=48):
        """Ink extent (min y for top profile / max y for bottom) of a line in columns x0..x1 (line-local y)."""
        cols = np.full(nb, np.nan)
        for g in L["glyphs"]:
            for c in g.conts:
                xs = c[:, 0] + dx
                idx = np.clip(((xs - x0) / max(1e-6, x1 - x0) * nb).astype(int), 0, nb - 1)
                ok = (xs >= x0) & (xs <= x1)
                for k, yv in zip(idx[ok], c[ok, 1]):
                    if np.isnan(cols[k]) or (yv > cols[k] if bottom else yv < cols[k]):
                        cols[k] = yv
        return cols

    for li, L in enumerate(L_all):
        L["_dx"] = _xalign(li, L)
        if li == 0:
            base = -L["oy0"]
        else:
            P = L_all[li - 1]
            gap = leading * max(P["size"], L["size"])
            base = P["_b"] + P["y1"] + gap - L["y0"]
            if use_lock and (P["hero"] != L["hero"]):
                # pull the lines together as far as the REAL ink allows (column-wise collision) + clearance
                x0, x1 = -10.0, maxw + 10.0
                pb = _profile(P, P["_dx"], True, x0, x1)
                lt = _profile(L, L["_dx"], False, x0, x1)
                both = ~np.isnan(pb) & ~np.isnan(lt)
                clear = float(tuck) * min(P["size"], L["size"])
                if both.any():
                    need = float(np.max((P["_b"] + pb[both]) - lt[both])) + clear
                    base = min(base, need)
        base += float(L["cfg"].get("dy", 0))
        L["_b"] = base
        dx = L["_dx"]
        sh = np.array([dx, base], np.float32)
        for g in L["glyphs"]:
            g.conts = [c + sh for c in g.conts]
            g.x0 += dx; g.x1 += dx; g.cx += dx; g.y0 += base; g.y1 += base; g.cy += base
        for w in L["words"]:
            w["x0"] += dx; w["x1"] += dx; w["y0"] += base; w["y1"] += base
        blk.glyphs += L["glyphs"]
    if blk.glyphs:
        x0 = min(g.x0 for g in blk.glyphs); x1 = max(g.x1 for g in blk.glyphs)
        y0 = min(g.y0 for g in blk.glyphs); y1 = max(g.y1 for g in blk.glyphs)
    else:
        x0 = y0 = x1 = y1 = 0.0
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
    for g in blk.glyphs:
        g.conts = [k - c for k in g.conts]
        g.x0 -= c[0]; g.x1 -= c[0]; g.cx -= c[0]; g.y0 -= c[1]; g.y1 -= c[1]; g.cy -= c[1]
    W = max(1.0, x1 - x0)
    for g in blk.glyphs:
        g.u = (x1 - c[0] - g.cx) / W          # 0 = right edge (RTL start) … 1 = left edge
    for w in blk.words:
        for k in ("x0", "x1"):
            w[k] -= c[0]
        for k in ("y0", "y1"):
            w[k] -= c[1]
        w["cx"], w["cy"] = (w["x0"] + w["x1"]) / 2, (w["y0"] + w["y1"]) / 2
    for g in blk.glyphs:                         # default pivot = WORD centre (Persian is cursive)
        w = blk.words[g.word]
        g.px, g.py = w["cx"], w["cy"]
        g.worder = g.word
    blk.bbox = (x0 - c[0], y0 - c[1], x1 - c[0], y1 - c[1])
    blk.lines, blk.n, blk.nwords = L_all, len(blk.glyphs), wi
    return blk


# ------------------------------------------------------------------------------------------------- 3D
def rot3(rx=0.0, ry=0.0, rz=0.0):
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    return Rz @ Ry @ Rx


class Field:
    """Global deformation of block points (N×3, block-centred px, y down, +z away from camera)."""

    def __init__(self, kind="none", **kw):
        self.kind, self.kw = kind, kw

    def __call__(self, P):
        k, kw = self.kind, self.kw
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        if k in ("arc_top", "arc_bottom"):
            R = float(kw.get("radius", 300))
            a = x / R + math.radians(float(kw.get("angle", 0.0)))
            if k == "arc_top":
                r = R - y
                return np.stack([r * np.sin(a), -r * np.cos(a), z], 1)
            r = R + y
            return np.stack([r * np.sin(a), r * np.cos(a), z], 1)
        if k == "cylinder":
            R = float(kw.get("radius", 300))
            a = x / R + math.radians(float(kw.get("angle", 0.0)))
            rr = R - z
            out = np.stack([rr * np.sin(a), y, -rr * np.cos(a)], 1)
            if kw.get("tilt"):
                out = out @ rot3(float(kw["tilt"]), 0, 0).T
            return out
        if k == "wave":
            A, L, ph = float(kw.get("amp", 12)), float(kw.get("length", 260)), float(kw.get("phase", 0))
            return np.stack([x, y + A * np.sin(2 * math.pi * x / L + ph), z], 1)
        if k == "jelly":
            A, L, ph = float(kw.get("amp", 10)), float(kw.get("length", 220)), float(kw.get("phase", 0))
            return np.stack([x + 0.6 * A * np.sin(2 * math.pi * y / L + ph), y + A * np.sin(2 * math.pi * x / L + 1.3 * ph), z], 1)
        if k == "bulge":
            cx, cy, R, A = float(kw.get("cx", 0)), float(kw.get("cy", 0)), float(kw.get("radius", 200)), float(kw.get("amp", .25))
            dx, dy = x - cx, y - cy
            f = 1 + A * np.exp(-2 * (dx * dx + dy * dy) / (R * R))
            return np.stack([cx + dx * f, cy + dy * f, z], 1)
        if k == "taper":         # perspective-like taper toward one side (fake 3D lean)
            A = float(kw.get("amp", 0.3)); W = float(kw.get("width", 400))
            f = 1 + A * x / W
            return np.stack([x, y * f, z], 1)
        if k == "bezier_path":   # Ported from EffectCraft `crates/text/src/path_text.rs` (`PathMeasure`)
            pts = np.asarray(kw.get("points", [[-300, 0], [0, -90], [300, 0]]), dtype=np.float32)
            if len(pts) < 2:
                return P
            diffs = np.diff(pts, axis=0)
            segl = np.hypot(diffs[:, 0], diffs[:, 1])
            cum = np.concatenate([[0.0], np.cumsum(segl)])
            tot = float(cum[-1])
            if tot <= 1e-5:
                return P
            s_raw = x + tot * 0.5 + float(kw.get("margin", 0.0))
            s_q = np.clip(s_raw, 0.0, tot)
            idx = np.clip(np.searchsorted(cum, s_q, side="right") - 1, 0, len(diffs) - 1)
            tx = diffs[idx, 0] / np.maximum(1e-5, segl[idx])
            ty = diffs[idx, 1] / np.maximum(1e-5, segl[idx])
            extra = s_raw - s_q
            px = np.interp(s_q, cum, pts[:, 0]) + extra * tx
            py = np.interp(s_q, cum, pts[:, 1]) + extra * ty
            if kw.get("perp", True):
                nx, ny = -ty, tx
                return np.stack([px + y * nx, py + y * ny, z], 1)
            return np.stack([px, py + y, z], 1)
        return P


def eval_selector(u: float, t: float, cfg: dict) -> float:
    """
    Ported from EffectCraft `crates/text/src/selectors.rs`:
    Evaluates After Effects Range (`square`, `ramp_up`, `ramp_down`, `triangle`, `round`, `smooth`)
    and `wiggly` text selector weight in [-1, 1] for unit position `u in [0, 1]` at time `t`.
    """
    kind = cfg.get("shape", "smooth")
    if kind == "wiggly":
        freq = float(cfg.get("wiggles_per_sec", 3.0))
        corr = float(cfg.get("correlation", 0.5))
        phase = t * freq * 2.0 * math.pi + u * (1.0 - corr) * 19.7
        return float(math.sin(phase) * 0.65 + math.cos(phase * 1.618 + 1.3) * 0.35)
    st = float(cfg.get("start", 0.0)) + float(cfg.get("offset", 0.0))
    en = float(cfg.get("end", 1.0)) + float(cfg.get("offset", 0.0))
    span = max(1e-5, en - st)
    r = (u - st) / span
    if r < 0.0 or r > 1.0:
        return 0.0
    if kind == "ramp_up":
        v = r
    elif kind == "ramp_down":
        v = 1.0 - r
    elif kind == "triangle":
        v = 1.0 - abs(2.0 * r - 1.0)
    elif kind == "round":
        v = math.sqrt(max(0.0, 1.0 - (2.0 * r - 1.0) ** 2))
    elif kind == "smooth":
        v = 0.5 * (1.0 - math.cos(2.0 * math.pi * r))
    else:
        v = 1.0
    return float(v * float(cfg.get("amount", 1.0)))


def project(P, f=1600.0):
    s = f / np.maximum(f * 0.08, f + P[:, 2])
    return P[:, :2] * s[:, None]


class GS:
    """Per-glyph animation state."""
    __slots__ = ("tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "shear", "alpha", "color", "glow", "visible",
                 "fill_alpha", "stroke")

    def __init__(self):
        self.tx = self.ty = self.tz = self.rx = self.ry = self.rz = self.shear = 0.0
        self.sx = self.sy = self.alpha = self.fill_alpha = 1.0
        self.color = self.glow = self.stroke = None
        self.visible = True


def _local(g, gs, extra_z=0.0):
    A = rot3(gs.rx, gs.ry, gs.rz) @ np.array([[gs.sx, gs.shear * gs.sx, 0], [0, gs.sy, 0], [0, 0, 1]], np.float32)
    c = np.array([g.px, g.py, 0], np.float32)
    t = np.array([gs.tx, gs.ty, gs.tz], np.float32)
    out = []
    for k in g.conts:
        P = np.concatenate([k, np.full((len(k), 1), extra_z, np.float32)], 1) - c
        out.append(P @ A.T + c + t)
    return out


def _poly(ctx, q, x, y):
    ctx.move_to(q[0, 0] + x, q[0, 1] + y)
    for i in range(1, len(q)):
        ctx.line_to(q[i, 0] + x, q[i, 1] + y)
    ctx.close_path()


def draw_block(ctx, blk, x, y, motion=None, field=None, plane=None, f=1600.0, pal=None, ectx=None, depth=None,
               extrude=None, scale=1.0, alpha=1.0, stroke=None, shadow=None, mblur=None):
    """Draw block at screen (x, y).
    motion(g) → GS   per-glyph animation        field   Field                 plane {rx, ry, rz, z}
    extrude {depth, color, steps, light}         depth "back" | "front"      (z > 0 = behind the speaker)
    ectx  emission context: glowing glyphs are ALSO painted there (→ bloom, light-wrap, light spill).
    mblur v14.8 {"motion": motion_prev, "plane": plane_prev, "x": x_prev, "y": y_prev, "cfg": {...}} — the block's
          state one SHUTTER interval earlier; glyphs whose projected displacement exceeds cfg["threshold"] px are
          drawn with a 1-D directional streak along their own velocity (motionblur.py, 180° shutter)."""
    pal = pal or PALETTE
    Rp = rot3(plane.get("rx", 0), plane.get("ry", 0), plane.get("rz", 0)) if plane else None
    pz = float(plane.get("z", 0)) if plane else 0.0

    def world(cs, _Rp=Rp, _pz=pz):
        if scale != 1.0:
            cs = [c * scale for c in cs]
        if field is not None:
            cs = [field(c) for c in cs]
        if _Rp is not None:
            cs = [c @ _Rp.T for c in cs]
        if _pz:
            cs = [c + np.array([0, 0, _pz], np.float32) for c in cs]
        return cs

    mb_prev = None
    if mblur and mblur.get("motion") is not None and mblur.get("cfg"):
        pp = mblur.get("plane")
        mb_Rp = rot3(pp.get("rx", 0), pp.get("ry", 0), pp.get("rz", 0)) if pp else None
        mb_pz = float(pp.get("z", 0)) if pp else 0.0
        mb_prev = (mblur["motion"], mb_Rp, mb_pz, float(mblur.get("x", x)), float(mblur.get("y", y)), mblur["cfg"])

    def _paint_glyph(c_, g, gs, cs, ectx_, emission_only=False):
        """One glyph: extrusion, shadow, fill/gradient/stroke on c_ (+ emission on ectx_).  Returns fill extents."""
        st = g.style
        col = rgba(gs.color or st.get("color", "@ink"), pal)
        a = gs.alpha * alpha * col[3] * float(st.get("alpha", 1.0))
        e = (0.0, 0.0, 0.0, 0.0)
        if not emission_only:
            ex = st.get("extrude") or extrude
            if ex and float(ex.get("depth", 0)) > 0.5:
                dep = float(ex["depth"])
                steps = int(ex.get("steps", max(5, min(18, dep / 2.0))))
                ec = rgba(ex["color"], pal) if ex.get("color") else (col[0] * .42, col[1] * .32, col[2] * .22, 1)
                for si in range(steps, 0, -1):
                    cc = world(_local(g, gs, extra_z=dep * si / steps / max(scale, 1e-6)))
                    sh = 0.5 + 0.5 * (1 - si / steps) * float(ex.get("light", 0.6))
                    c_.new_path()
                    for c in cc:
                        _poly(c_, project(c, f), x, y)
                    c_.set_source_rgba(ec[0] * sh, ec[1] * sh, ec[2] * sh, a)
                    c_.fill()
            c_.new_path()
            for c in cs:
                _poly(c_, project(c, f), x, y)
            e = c_.fill_extents()
            if shadow:
                c_.save()
                c_.translate(float(shadow.get("dx", 0)), float(shadow.get("dy", 5)))
                c_.set_source_rgba(0, 0, 0, a * float(shadow.get("alpha", .5)))
                c_.fill_preserve()
                c_.restore()
            fa = gs.fill_alpha * float(st.get("fill_alpha", 1.0))
            grad = st.get("gradient")
            if grad:
                lg = cairo.LinearGradient(0, e[1], 0, e[3])
                for i, gc in enumerate(grad):
                    rc = rgba(gc, pal)
                    lg.add_color_stop_rgba(i / max(1, len(grad) - 1), rc[0], rc[1], rc[2], a * fa)
                c_.set_source(lg)
            else:
                c_.set_source_rgba(col[0], col[1], col[2], a * fa)
            stk = gs.stroke or st.get("stroke") or stroke
            if stk:
                if fa > 0.003:
                    c_.fill_preserve()
                sc = rgba(stk.get("color", "@ink"), pal)
                c_.set_source_rgba(sc[0], sc[1], sc[2], a * sc[3])
                c_.set_line_width(float(stk.get("width", 2.0)))
                c_.set_line_join(cairo.LINE_JOIN_ROUND)
                c_.stroke()
            else:
                c_.fill()
        glow = gs.glow if gs.glow is not None else float(st.get("glow", 0.0))
        if ectx_ is not None and glow > 0.003:
            ectx_.new_path()
            for c in cs:
                _poly(ectx_, project(c, f), x, y)
            gc = rgba(st.get("glow_color", gs.color or st.get("color", "@ink")), pal)
            ectx_.set_source_rgba(gc[0], gc[1], gc[2], a * min(1.0, glow))
            ectx_.fill()
        return e

    items = []
    for g in blk.glyphs:
        gs = motion(g) if motion else GS()
        if not gs.visible or gs.alpha * alpha <= 0.003:
            continue
        cs = world(_local(g, gs))
        zc = float(np.mean([c[:, 2].mean() for c in cs]))
        if (depth == "back" and zc <= 0) or (depth == "front" and zc > 0):
            continue
        vec = None
        if mb_prev is not None:
            mprev, pRp, ppz, px_, py_, mcfg = mb_prev
            gp = mprev(g)
            P1 = np.concatenate([project(c, f) for c in cs]) + np.array([x, y], np.float32)
            if gp.visible and gp.alpha > 0.003:
                cp = world(_local(g, gp), pRp, ppz)
                P0 = np.concatenate([project(c, f) for c in cp]) + np.array([px_, py_], np.float32)
                if len(P0) == len(P1):
                    from .motionblur import principal_displacement
                    dvx, dvy = principal_displacement(P0, P1)
                    if math.hypot(dvx, dvy) >= float(mcfg["threshold"]):
                        vec = (dvx, dvy, float(mcfg["max"]), P1)
        items.append((zc, g, gs, cs, vec))
    items.sort(key=lambda it: -it[0])
    bb = [1e9, 1e9, -1e9, -1e9]
    for zc, g, gs, cs, vec in items:
        if vec is not None:
            P1 = vec[3]
            ext = 8.0 + (float((g.style.get("extrude") or extrude or {}).get("depth", 0)) * 1.2) + \
                (abs(float(shadow.get("dy", 5))) + abs(float(shadow.get("dx", 0))) if shadow else 0.0)
            gbox = (float(P1[:, 0].min()) - ext, float(P1[:, 1].min()) - ext,
                    float(P1[:, 0].max()) + ext, float(P1[:, 1].max()) + ext)
            from .motionblur import paint_blurred
            paint_blurred(ctx, lambda c, _g=g, _gs=gs, _cs=cs: _paint_glyph(c, _g, _gs, _cs, None),
                          gbox, vec[0], vec[1], vec[2])
            bb = [min(bb[0], gbox[0]), min(bb[1], gbox[1]), max(bb[2], gbox[2]), max(bb[3], gbox[3])]
            if ectx is not None:
                _paint_glyph(None, g, gs, cs, ectx, emission_only=True)
            continue
        e = _paint_glyph(ctx, g, gs, cs, ectx)
        bb = [min(bb[0], e[0]), min(bb[1], e[1]), max(bb[2], e[2]), max(bb[3], e[3])]
    return tuple(bb)
