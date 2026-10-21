"""
mograph.aecomp.titles — full-frame TITLE templates (hero line + kicker + shape accents) as aecomp comps.

Anatomy learned from the studied Titles pack (194 projects: Minimal, Modern, Bigtypo, Kinetic, Creative…), not its look:
  · 2–3 text lines: a small KICKER (≈0.35× hero, wide tracking) + a HERO line + optional small footer
  · lines arrive STAGGERED (≈0.15–0.3 s apart); per-letter/word delay 0.02–0.09 s × index (we use the 351 AE text
    animators of mograph.aefx: "chars" / "words" / "lines" based range selectors, Persian-aware cursive pieces)
  · a shape accent frames the type: lines that draw (trim), a rule under the hero, corner brackets, a disc burst
  · hold → exit = the entrance played backwards (or a faster opacity-only out)
User laws: zero-box (no plates behind text), Doran family only, RTL.

    from mograph.aecomp import titles as TT
    comp = TT.make("minimal_rule", hero="مقدمه را آخر بنویس", kicker="نکته نگارش", dur=4.0, accent="#ffb21f")
"""
from __future__ import annotations

from .generators import kf, burst


def _text(name, text, size, x, y, font, color, preset, t_in, dur_in, t_out, dur_out=0.45, align="center",
          tracking=0.0, out_preset=None):
    return {"name": name, "type": "text", "in": t_in, "out": t_out + dur_out,
            "tr": {"pos": [x, y]},
            "text": {"text": text, "size": size, "font": font, "color": color, "align": align, "tracking": tracking,
                     "preset": preset, "preset_dur": dur_in, "anim_t": t_in,
                     "reveal": {"t_in": t_in, "t_out": t_out, "dur_in": 0.01, "dur_out": dur_out,
                                "in": {}, "out": {"fade": True, "dy": -0.25}}}}


def _rule(name, x0, x1, y, width, color, t0, d, t_out, d_out=0.4, from_centre=False):
    if from_centre:
        cx = (x0 + x1) / 2
        segs = [[[cx, y], [x1, y]], [[cx, y], [x0, y]]]
    else:
        segs = [[[x1, y], [x0, y]]]
    it = []
    for v in segs:
        it.append({"ty": "sh", "path": {"v": v, "i": [[0, 0], [0, 0]], "o": [[0, 0], [0, 0]], "c": False}})
    it += [{"ty": "tm", "end": kf((t0, 0), (t0 + d, 100), ease=(0.8, 0.8)),
            "start": kf((t_out, 0), (t_out + d_out, 100), ease=(0.8, 0.8)), "mode": 2},
           {"ty": "st", "color": color, "width": width, "cap": "round"}]
    return {"name": name, "type": "shape", "in": 0, "out": 999, "tr": {}, "shapes": [{"ty": "gr", "it": it}]}


def _corners(name, cx, cy, w, h, arm, width, color, t0, t_out):
    """four L-shaped corner brackets that draw in (AE trim) — a frame without a box."""
    it = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        x, y = cx + sx * w / 2, cy + sy * h / 2
        it.append({"ty": "gr", "it": [
            {"ty": "sh", "path": {"v": [[x, y - sy * arm], [x, y], [x - sx * arm, y]], "i": [[0, 0]] * 3,
                                  "o": [[0, 0]] * 3, "c": False}},
            {"ty": "tm", "end": kf((t0, 50), (t0 + 0.5, 100), (t_out, 100), (t_out + 0.4, 50), ease=(0.8, 0.8)),
             "start": kf((t0, 50), (t0 + 0.5, 0), (t_out, 0), (t_out + 0.4, 50), ease=(0.8, 0.8))},
            {"ty": "st", "color": color, "width": width, "cap": "round", "join": "round"}]})
    return {"name": name, "type": "shape", "in": 0, "out": 999, "tr": {}, "shapes": it}


def fit_size(text, size, max_w, font="Doran-ExtraBold.ttf"):
    """largest size <= size so the line is at most max_w px wide (measured with the real glyph engine)."""
    from ..v14 import glyphs as G
    st = {"thin": {"font": font, "scale": 1.0, "stretch": G.STRETCH.get(font.split("-")[0].lower(), 1.0)}}
    b = G.typeset(text, size=float(size), styles=st)
    wpx = float(b.bbox[2] - b.bbox[0]) or 1.0
    return float(size) * min(1.0, max_w / wpx)


def fit_w(text, size, font):
    from ..v14 import glyphs as G
    st = {"thin": {"font": font, "scale": 1.0, "stretch": G.STRETCH.get(font.split("-")[0].lower(), 1.0)}}
    b = G.typeset(text, size=float(size), styles=st)
    return float(b.bbox[2] - b.bbox[0])


def _comp(name, w, h, dur, layers):
    return {"name": name, "w": w, "h": h, "dur": dur, "fps": 30, "controls": {}, "layers": layers}


def minimal_rule(hero, kicker="", w=1080, h=1920, y=None, size=120, accent="#ffb21f", ink="#f6f2ea", dur=4.0,
                 font="Doran-ExtraBold.ttf", font2="Doran-Regular.ttf"):
    """Kicker fades up, hero types on by connected pieces, a rule draws open from the centre under it."""
    size = fit_size(hero, size, 0.86 * w, font)
    y = h * 0.42 if y is None else y
    to = dur - 0.6
    rw = min(0.7 * w, fit_w(hero, size, font) * 0.8)
    return _comp("title_minimal_rule", w, h, dur, [
        _text("kicker", kicker, size * 0.36, w / 2, y - size * 0.95, font2, accent, "atp_100", 0.0, 0.6, to - 0.1,
              tracking=80),
        _text("hero", hero, size, w / 2, y, font, ink, "atp_3", 0.2, 0.9, to),
        _rule("rule", w / 2 - rw / 2, w / 2 + rw / 2, y + size * 0.75, 4, accent, 0.55, 0.6, to - 0.05,
              from_centre=True)])


def bracket_frame(hero, kicker="", w=1080, h=1920, y=None, size=110, accent="#2dd4bf", ink="#f6f2ea", dur=4.0,
                  font="Doran-ExtraBold.ttf", font2="Doran-Light.ttf"):
    """Corner brackets draw around the title area, hero rises line-by-line, kicker below."""
    size = fit_size(hero, size, 0.86 * w, font)
    y = h * 0.42 if y is None else y
    to = dur - 0.6
    bw = min(0.94 * w, fit_w(hero, size, font) + size * 0.8)
    return _comp("title_bracket_frame", w, h, dur, [
        _text("hero", hero, size, w / 2, y, font, ink, "atp_23", 0.25, 0.8, to),
        _text("kicker", kicker, size * 0.4, w / 2, y + size * 1.05, font2, accent, "atp_13", 0.5, 0.8, to - 0.1),
        _corners("frame", w / 2, y + size * 0.2, bw, size * 2.6, size * 0.45, 5, accent, 0.0, to)])


def burst_hero(hero, kicker="", w=1080, h=1920, y=None, size=130, accent="#ff4f6d", ink="#ffffff", dur=3.5,
               font="Doran-ExtraBold.ttf", font2="Doran-Regular.ttf"):
    """Punchy: a radial burst fires behind the hero which scales in by words; kicker slides after."""
    size = fit_size(hero, size, 0.86 * w, font)
    y = h * 0.42 if y is None else y
    to = dur - 0.5
    b = burst(seed=4, rays=12, radius=520, inner=160, dots=True)
    shapes = b["layers"][0]["shapes"]
    for g in shapes:                       # colour the burst from the accent
        for it in g["it"]:
            if it.get("ty") == "st":
                it["color"] = accent
                it["width"] = 8
            if it.get("ty") == "fl":
                it["color"] = ink
    return _comp("title_burst_hero", w, h, dur, [
        _text("hero", hero, size, w / 2, y, font, ink, "atp_16", 0.1, 0.7, to),
        _text("kicker", kicker, size * 0.36, w / 2, y + size * 1.0, font2, accent, "atp_100", 0.45, 0.5, to - 0.1,
              tracking=60),
        {"name": "burst", "type": "shape", "in": 0.05, "out": 1.1, "tr": {"pos": [w / 2, y - size * 0.1],
                                                                          "scale": [130, 130]},
         "shapes": shapes}])


def kinetic_stack(hero, kicker="", w=1080, h=1920, y=None, size=118, accent="#ffb21f", ink="#f6f2ea", dur=4.0,
                  font="Doran-ExtraBold.ttf", font2="Doran-Regular.ttf"):
    """Two thin rules slide in from opposite sides, hero lands between them (blur-in by chars), kicker on top."""
    size = fit_size(hero, size, 0.86 * w, font)
    y = h * 0.42 if y is None else y
    to = dur - 0.6
    rw = 0.62 * w
    top, bot = y - size * 0.85, y + size * 0.7

    def slide(name, yy, sign, t0):
        return {"name": name, "type": "shape", "in": 0, "out": 999,
                "tr": {"pos": kf((t0, [sign * w * 0.6, 0]), (t0 + 0.55, [0, 0]), (to, [0, 0]), (to + 0.45, [-sign * w * 0.6, 0]),
                                 ease=(0.85, 0.85))},
                "shapes": [{"ty": "gr", "it": [
                    {"ty": "sh", "path": {"v": [[w / 2 - rw / 2, yy], [w / 2 + rw / 2, yy]], "i": [[0, 0], [0, 0]],
                                          "o": [[0, 0], [0, 0]], "c": False}},
                    {"ty": "st", "color": accent, "width": 3, "cap": "round"}]}]}
    return _comp("title_kinetic_stack", w, h, dur, [
        _text("kicker", kicker, size * 0.34, w / 2, top - size * 0.35, font2, accent, "atp_100", 0.35, 0.5, to - 0.1,
              tracking=80),
        _text("hero", hero, size, w / 2, y, font, ink, "atp_116", 0.25, 0.8, to),
        slide("r1", top, 1, 0.0), slide("r2", bot, -1, 0.08)])


STYLES = {"minimal_rule": minimal_rule, "bracket_frame": bracket_frame, "burst_hero": burst_hero,
          "kinetic_stack": kinetic_stack}


def styles():
    return list(STYLES)


def make(style, hero, kicker="", **kw):
    c = STYLES[style](hero, kicker, **kw)
    c["gen"] = {"title": style}
    return c
