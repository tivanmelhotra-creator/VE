"""
mograph.aecomp.lowerthirds — BOX-FREE, RTL-Persian lower thirds (name / role / handle) as aecomp comps.

Anatomy learned from the studied pack (50 lower thirds, 10 s comps, 30 fps), NOT its look:
    IN   (0 → ~1.3 s) an accent line draws (trim End leads, Start holds), text slides OUT FROM BEHIND the line through
         a track matte (range-selector offset 0→100 % over ≈0.45–0.85 s, Easy Ease), secondary line follows ~0.3 s later
    HOLD everything still (tiny drift allowed)
    OUT  (last ~0.9 s) the exact reverse, secondary first
Measured: title in ≈ 0.44–0.70 s, role delay ≈ 0.31 s, out ≈ 0.62 s; text 2.6:1 title:role size ratio.

User laws obeyed: NO flat rectangle/pill behind text (zero-box), Doran family only (+12 % stretch through the v14
theme), right-aligned RTL reading, accents = thin strokes / dots / arcs only.

    from mograph.aecomp import lowerthirds as LT
    comp = LT.make("line_reveal", title="دکتر سارا احمدی", role="متخصص ارتودنسی", dur=5.0, accent="#ffb21f")
    LT.styles()  → list of style names
"""
from __future__ import annotations

from .generators import kf

E33 = 1 / 3


def _txt(name, text, size, x, y, font, color, align, anim_in, anim_out, t_in, t_out):
    return {"name": name, "type": "text", "in": 0, "out": 999,
            "tr": {"pos": [x, y]},
            "text": {"text": text, "size": size, "font": font, "color": color, "align": align,
                     "reveal": {"t_in": t_in, "t_out": t_out, "in": anim_in, "out": anim_out}}}


def _line(name, x0, x1, y, width, color, t0, d, t_out, d_out, cap="round", lead_right=True):
    a, b = (x1, x0) if lead_right else (x0, x1)              # RTL: draws from the right edge leftwards
    return {"name": name, "type": "shape", "in": 0, "out": 999, "tr": {},
            "shapes": [{"ty": "gr", "it": [
                {"ty": "sh", "path": {"v": [[a, y], [b, y]], "i": [[0, 0], [0, 0]], "o": [[0, 0], [0, 0]], "c": False}},
                {"ty": "tm", "end": kf((t0, 0), (t0 + d, 100), (t_out, 100), (t_out + d_out, 100), ease=(0.75, 0.75)),
                 "start": kf((t0, 0), (t_out, 0), (t_out + d_out, 100), ease=(0.75, 0.75))},
                {"ty": "st", "color": color, "width": width, "cap": cap}]}]}


def _dot(name, x, y, r, color, t0, t_out):
    return {"name": name, "type": "shape", "in": 0, "out": 999, "tr": {"pos": [x, y]},
            "shapes": [{"ty": "gr", "it": [{"ty": "el", "size": [2 * r, 2 * r]}, {"ty": "fl", "color": color}],
                        "tr": {"scale": kf((t0, [0, 0]), (t0 + 0.28, [118, 118]), (t0 + 0.42, [100, 100]),
                                           (t_out, [100, 100]), (t_out + 0.3, [0, 0]))}}]}


def _matte_rect(name, x0, y0, x1, y1):
    """invisible matte (used only as a track matte → text appears from behind an edge; never drawn)."""
    return {"name": name, "type": "shape", "in": 0, "out": 999, "tr": {},
            "shapes": [{"ty": "gr", "it": [{"ty": "rc", "size": [x1 - x0, y1 - y0],
                                            "pos": [(x0 + x1) / 2, (y0 + y1) / 2]}, {"ty": "fl", "color": "#ffffff"}]}]}


# text reveal animations (applied by render._draw_text through mograph.aefx.textanim-style per-block offset):
SLIDE_UP = {"dy": 1.1}          # moves up from below its matte edge (fraction of font size)
SLIDE_DOWN = {"dy": -1.6}
SLIDE_RTL = {"dx": -0.6, "fade": True}


def line_reveal(title, role="", w=1080, h=1920, x=None, y=None, size=78, accent="#ffb21f", ink="#f6f2ea",
                dur=5.0, font="Doran-ExtraBold.ttf", font2="Doran-Regular.ttf"):
    """Accent line draws right→left; title rises from behind it, role drops from below it."""
    x = w - 90 if x is None else x
    y = h * 0.78 if y is None else y
    L = 0.62 * w
    t_out = dur - 0.9
    lay = [
        _txt("title", title, size, x, y - 0.62 * size, font, ink, "right", SLIDE_UP, SLIDE_UP, 0.30, t_out + 0.18),
        _matte_rect("m_title", x - L - 40, y - 3 * size, x + 40, y),
        _txt("role", role, size / 2.6, x, y + 0.62 * size, font2, accent, "right", SLIDE_DOWN, SLIDE_DOWN, 0.61, t_out),
        _matte_rect("m_role", x - L - 40, y + 4, x + 40, y + 2 * size),
        _line("line", x - L, x, y, 5, accent, 0.0, 0.7, t_out + 0.2, 0.6),
        _dot("dot", x + 26, y, 9, accent, 0.05, t_out + 0.45),
    ]
    lay[0]["matte"] = {"layer": "m_title", "mode": "alpha"}
    lay[2]["matte"] = {"layer": "m_role", "mode": "alpha"}
    return {"name": "lt_line_reveal", "w": w, "h": h, "dur": dur, "fps": 30, "controls": {}, "layers": lay}


def bracket(title, role="", w=1080, h=1920, x=None, y=None, size=74, accent="#2dd4bf", ink="#f6f2ea", dur=5.0,
            font="Doran-ExtraBold.ttf", font2="Doran-Light.ttf"):
    """A vertical accent bar draws at the right (RTL start) with short serifs; text slides out leftwards from it."""
    x = w - 90 if x is None else x
    y = h * 0.78 if y is None else y
    hh = 1.9 * size
    t_out = dur - 0.9
    bar = {"name": "bar", "type": "shape", "in": 0, "out": 999, "tr": {}, "shapes": [{"ty": "gr", "it": [
        {"ty": "sh", "path": {"v": [[x - 24, y - hh / 2], [x, y - hh / 2], [x, y + hh / 2], [x - 24, y + hh / 2]],
                              "i": [[0, 0]] * 4, "o": [[0, 0]] * 4, "c": False}},
        {"ty": "tm", "end": kf((0, 50), (0.55, 100), (t_out + 0.3, 100), (t_out + 0.8, 50), ease=(0.8, 0.8)),
         "start": kf((0, 50), (0.55, 0), (t_out + 0.3, 0), (t_out + 0.8, 50), ease=(0.8, 0.8))},
        {"ty": "st", "color": accent, "width": 6, "cap": "round", "join": "round"}]}]}
    lay = [
        _txt("title", title, size, x - 40, y - 0.28 * size, font, ink, "right", SLIDE_RTL, SLIDE_RTL, 0.35, t_out + 0.15),
        _txt("role", role, size / 2.5, x - 40, y + 0.55 * size, font2, accent, "right", SLIDE_RTL, SLIDE_RTL, 0.62, t_out),
        _matte_rect("m", 0, y - hh, x - 30, y + hh),
        bar]
    lay[0]["matte"] = {"layer": "m", "mode": "alpha"}
    lay[1]["matte"] = {"layer": "m", "mode": "alpha"}
    return {"name": "lt_bracket", "w": w, "h": h, "dur": dur, "fps": 30, "controls": {}, "layers": lay}


def orbit_dot(title, role="", w=1080, h=1920, x=None, y=None, size=76, accent="#ff4f6d", ink="#f6f2ea", dur=5.0,
              font="Doran-ExtraBold.ttf", font2="Doran-Regular.ttf"):
    """A small ring + dot spins in at the right, then a hair-line underline runs left; text wipes in."""
    x = w - 110 if x is None else x
    y = h * 0.78 if y is None else y
    t_out = dur - 0.9
    ring = {"name": "ring", "type": "shape", "in": 0, "out": 999, "tr": {"pos": [x + 40, y - 0.1 * size]},
            "shapes": [{"ty": "gr", "it": [
                {"ty": "el", "size": [46, 46]},
                {"ty": "tm", "end": kf((0, 0), (0.6, 100), (t_out + 0.35, 100), (t_out + 0.8, 0)), "offset": kf((0, -90), (dur, 270))},
                {"ty": "st", "color": accent, "width": 4}]},
                {"ty": "gr", "it": [{"ty": "el", "size": [14, 14]}, {"ty": "fl", "color": accent}],
                 "tr": {"scale": kf((0.3, [0, 0]), (0.55, [120, 120]), (0.7, [100, 100]), (t_out + 0.4, [100, 100]),
                                    (t_out + 0.7, [0, 0]))}}]}
    lay = [
        _txt("title", title, size, x, y - 0.1 * size, font, ink, "right", SLIDE_RTL, SLIDE_RTL, 0.45, t_out + 0.15),
        _txt("role", role, size / 2.6, x, y + 0.82 * size, font2, ink, "right", SLIDE_UP, SLIDE_UP, 0.8, t_out),
        _matte_rect("m", 0, y - 1.6 * size, x + 6, y + 0.45 * size),
        _matte_rect("m2", 0, y + 0.47 * size, x + 6, y + 1.6 * size),
        _line("under", x - 0.55 * w, x, y + 0.46 * size, 2, accent, 0.5, 0.6, t_out + 0.1, 0.5),
        ring]
    lay[0]["matte"] = {"layer": "m", "mode": "alpha"}
    lay[1]["matte"] = {"layer": "m2", "mode": "alpha"}
    return {"name": "lt_orbit_dot", "w": w, "h": h, "dur": dur, "fps": 30, "controls": {}, "layers": lay}


def split_lines(title, role="", w=1080, h=1920, x=None, y=None, size=80, accent="#ffb21f", ink="#f6f2ea", dur=5.0,
                font="Doran-ExtraBold.ttf", font2="Doran-Regular.ttf"):
    """Two lines open from a centre seam (top line goes up, bottom goes down) revealing the title between them."""
    x = w - 90 if x is None else x
    y = h * 0.78 if y is None else y
    t_out = dur - 0.9
    gap = 0.78 * size
    Lw = 0.6 * w

    def mv(name, sign):
        return {"name": name, "type": "shape", "in": 0, "out": 999,
                "tr": {"pos": kf((0.25, [0, 0]), (0.75, [0, sign * gap]), (t_out, [0, sign * gap]), (t_out + 0.5, [0, 0]),
                                 ease=(0.8, 0.8))},
                "shapes": _line(name, x - Lw, x, y, 3, accent, 0.0, 0.45, t_out + 0.4, 0.5)["shapes"]}
    top, bot = mv("top", -1), mv("bot", 1)
    lay = [
        _txt("title", title, size, x, y + 0.02 * size, font, ink, "right", {"scale_y": True}, {"scale_y": True}, 0.3,
             t_out),
        _txt("role", role, size / 2.7, x, y + gap + 0.55 * size, font2, accent, "right", SLIDE_DOWN, SLIDE_DOWN, 0.75,
             t_out - 0.1),
        _matte_rect("m2", 0, y + gap + 4, w, y + gap + 1.5 * size),
        top, bot]
    lay[1]["matte"] = {"layer": "m2", "mode": "alpha"}
    return {"name": "lt_split_lines", "w": w, "h": h, "dur": dur, "fps": 30, "controls": {}, "layers": lay}


STYLES = {"line_reveal": line_reveal, "bracket": bracket, "orbit_dot": orbit_dot, "split_lines": split_lines}


def styles():
    return list(STYLES)


def make(style, title, role="", **kw):
    c = STYLES[style](title, role, **kw)
    c["gen"] = {"lower_third": style}
    return c
