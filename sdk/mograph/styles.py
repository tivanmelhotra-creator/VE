"""
mograph.styles — three house styles as presets, so the agent writes ONE word ("meta.style") and gets a
coherent look + motion + typography + sound palette.  Everything stays overridable in the spec.

  lineart  (default)  noir vector line-art: black, white strokes, bloom, grain, pen sounds
  swiss               International Typographic Style: strict grid, flat colour fields (paper/red/black),
                      huge grotesque type flush to the grid, rules, numbers as graphics, hard structural
                      motion (wipe / slide / mask), no glow, crisp UI sounds
  kinetic             kinetic typography: the narration's words ARE the picture — slam / stack / scale /
                      scroll on the spoken word, bold contrast, camera punches, punchy hits

Use:  "meta": {"style": "swiss", "palette": "red"}      (palette: red | blue | yellow | mono)
Elements from every style can be mixed in one video; the preset only sets defaults.
"""
from __future__ import annotations

PALETTES = {
    # paper, ink, accent, mute
    "red":    {"paper": "#efece6", "ink": "#111111", "accent": "#e3262e", "mute": "#8c8a86"},
    "blue":   {"paper": "#eeeeea", "ink": "#121212", "accent": "#1f4fd8", "mute": "#8a8a88"},
    "yellow": {"paper": "#111111", "ink": "#f2f0ea", "accent": "#ffd21f", "mute": "#77756f"},
    "mono":   {"paper": "#f2f2f0", "ink": "#0e0e0e", "accent": "#0e0e0e", "mute": "#9a9a98"},
    "noir":   {"paper": "#000000", "ink": "#ffffff", "accent": "#ffffff", "mute": "#8a8a90"},
    "ink":    {"paper": "#000000", "ink": "#ffffff", "accent": "#ffd21f", "mute": "#5c5c62"},
    "blood":  {"paper": "#000000", "ink": "#ffffff", "accent": "#ff3b30", "mute": "#5c5c62"},
    # v9 promo worlds — extra tokens: glow/edge (radial_spotlight), studio/vignette (studio_vignette),
    # vivid/vivid_edge (crimson_vivid), panel (UI mockup cards), bg_mode default for the palette
    "crimson": {"paper": "#070103", "ink": "#ffffff", "accent": "#ff1e42", "mute": "#8a6a70",
                "glow": "#5e0818", "edge": "#070103", "mid": "#2a0309", "studio": "#fafafa", "vignette": "#b8b8c0",
                "vivid": "#ff1e42", "vivid_edge": "#9e0820", "panel": "#140508", "bg_mode": "radial_spotlight"},
    "silver":  {"paper": "#e9e9ec", "ink": "#111114", "accent": "#e8112d", "mute": "#77777f",
                "glow": "#ffffff", "edge": "#b8b8c0", "studio": "#fafafa", "vignette": "#b8b8c0",
                "vivid": "#ff1e42", "vivid_edge": "#9e0820", "panel": "#ffffff", "bg_mode": "studio_vignette"},
    "cyber":   {"paper": "#04060f", "ink": "#eaf6ff", "accent": "#21e6ff", "mute": "#5a6a85",
                "glow": "#0b2a4a", "edge": "#02030a", "mid": "#061430", "studio": "#f4fbff", "vignette": "#a9c2d6",
                "vivid": "#21e6ff", "vivid_edge": "#08607a", "panel": "#0a1022", "bg_mode": "radial_spotlight"},
    "emerald": {"paper": "#020805", "ink": "#f2fff8", "accent": "#18e08a", "mute": "#5f7d6d",
                "glow": "#073d24", "edge": "#010402", "mid": "#04200f", "studio": "#f6fffa", "vignette": "#b4cbbd",
                "vivid": "#18e08a", "vivid_edge": "#06683c", "panel": "#06140c", "bg_mode": "radial_spotlight"},
    # v10 ui_physics worlds — Apple / Linear / Rube-Goldberg interactive UI style
    "apple_blue": {"paper": "#0d0d0f", "ink": "#ffffff", "accent": "#0a84ff", "mute": "#8e8e98",
                   "panel": "#1c1c21", "border": "#2c2c34", "dot": "#1e1e24", "switch_off": "#3a3a42", "track_bg": "#32323a"},
    "emerald_os": {"paper": "#080d0b", "ink": "#f2fff9", "accent": "#0b9e6d", "mute": "#7a9287",
                   "panel": "#131e19", "border": "#23352c", "dot": "#16241e", "switch_off": "#2d3d35", "track_bg": "#24332c"},
    "violet_saas": {"paper": "#0a0812", "ink": "#f6f3ff", "accent": "#8b5cf6", "mute": "#8880a0",
                    "panel": "#171326", "border": "#2a2342", "dot": "#1c172e", "switch_off": "#342d4a", "track_bg": "#2b253d"},
    "macos_light": {"paper": "#f5f5f7", "ink": "#1d1d1f", "accent": "#007aff", "mute": "#86868b",
                    "panel": "#ffffff", "border": "#d2d2d7", "dot": "#dcdce2", "switch_off": "#d1d1d6", "track_bg": "#d8d8dc"},
}

PRESETS = {
    "lineart": {
        "meta": {"bg": "#000000", "bloom": 0.45, "vignette": 0.35, "grain": 0.014, "sound": "paper"},
        "fonts": {},
        "hud": {},
    },
    "swiss": {
        "meta": {"bloom": 0.0, "vignette": 0.0, "grain": 0.0, "camera_drift": 0, "slam_pulse": False,
                 "sound": "ui", "transition": {"type": "wipe", "dur": 0.55}},
        # grotesque hierarchy: Peyda Black display, Peyda Bold body, Peyda Medium labels, Pelak numbers
        "fonts": {"title": "peyda", "body": "bold", "ui_title": "bold", "ui_label": "medium",
                  "counter": "pelak", "subtitle": "bold"},
        "hud": {"header": {"show": False}, "dots": {"style": "numbers", "on": "@ink", "done": "@mute", "off": "@mute_soft"},
                "subs": {"spoken": "@ink", "emph": "@accent", "upcoming": "@mute_soft", "size": 46}},
    },
    "kinetic": {
        "meta": {"bg": "#000000", "bloom": 0.25, "vignette": 0.25, "grain": 0.01, "sound": "punchy",
                 "transition": {"type": "cut", "dur": 0.0}},
        "fonts": {"title": "title", "body": "peyda", "subtitle": "aria"},
        "hud": {"header": {"show": False}, "dots": {"show": False}, "subs": {"show": False}},
    },
    # v9 — high-energy kinetic promo / SaaS UI showcase: radial spotlights, bokeh dust, 3D type, whip blur,
    # Dutch-tilt camera, UI mock widgets.  Backdrop follows the palette (crimson → radial_spotlight,
    # silver → studio_vignette); override per chapter with "bg_mode".
    "promo": {
        "meta": {"bloom": 0.32, "vignette": 0.22, "grain": 0.012, "sound": "punchy", "slam_pulse": True,
                 "transition": {"type": "whip_pan", "dur": 0.42, "dir": "left"}},
        "fonts": {"title": "title", "body": "peyda", "subtitle": "aria", "ui": "vazir", "counter": "pelak"},
        "hud": {"header": {"show": False}, "dots": {"show": False},
                "subs": {"size": 46, "y": 0.9}},
    },
    "ui_physics": {
        "meta": {"bg": "#0d0d0f", "bloom": 0.08, "vignette": 0.10, "grain": 0.0, "camera_drift": 0.0,
                 "slam_pulse": False, "sound": "ui", "ambient": "dot_matrix",
                 "transition": {"type": "cut", "dur": 0.0}},
        "fonts": {"title": "peyda", "body": "vazir", "ui_title": "bold", "ui_label": "medium",
                  "counter": "pelak", "subtitle": "vazir"},
        "hud": {"header": {"show": False}, "dots": {"show": False}, "subs": {"show": False}},
    },
}


def palette(spec):
    m = spec.get("meta", {})
    if m.get("style") == "swiss":
        return dict(PALETTES.get(m.get("palette", "red"), PALETTES["red"]))
    if m.get("style") == "kinetic":
        return dict(PALETTES.get(m.get("palette", "ink"), PALETTES["ink"]))
    if m.get("style") == "promo":
        return dict(PALETTES.get(m.get("palette", "crimson"), PALETTES["crimson"]))
    if m.get("style") == "ui_physics":
        return dict(PALETTES.get(m.get("palette", "apple_blue"), PALETTES["apple_blue"]))
    return dict(PALETTES.get(m.get("palette", "noir"), PALETTES["noir"]))


def _deep_default(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_default(dst[k], v)
        elif k not in dst:
            dst[k] = v


def resolve_color(v, pal):
    """'@accent' / '@ink' / '@paper' / '@mute' → hex; anything else unchanged."""
    if isinstance(v, str) and v.startswith("@"):
        key = v[1:]
        if key == "mute_soft":
            return pal.get("mute")
        return pal.get(key, v)
    return v


DEFAULT_PALETTE = {"lineart": "noir", "swiss": "red", "kinetic": "ink", "promo": "crimson", "ui_physics": "apple_blue"}
LOOK_KEYS = ("bloom", "vignette", "grain", "slam_pulse")
WORLD_KEYS = ("style", "palette", "bg", "fonts", "sound", "hud", "bg_mode", "ambient") + LOOK_KEYS   # v8/v9 per-chapter


def _walk_tokens(o, pal):
    """Resolve @accent/@ink/@paper/@mute tokens in place (never touches time / target keys)."""
    if isinstance(o, dict):
        for k, v in list(o.items()):
            if isinstance(v, str) and v.startswith("@") and k not in ("t", "out", "anchor", "target", "from", "to"):
                o[k] = resolve_color(v, pal)
            else:
                _walk_tokens(v, pal)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, str) and v.startswith("@") and len(v) < 12:
                o[i] = resolve_color(v, pal)
            else:
                _walk_tokens(v, pal)


def world_for(user_meta, user_hud, ch=None):
    """v8 — the complete 'style world' of one chapter: style, fitted palette, bg, font roles, sound palette,
    post-FX and HUD.  Resolution order: chapter keys > meta keys (only when the chapter keeps the meta style)
    > the chapter style's preset.  A chapter that switches style gets that style's full look, not the
    leftovers of the base style (no bloom on paper, no header in a kinetic chapter...)."""
    import copy as _cp
    ch = ch or {}
    base = user_meta.get("style", "lineart")
    st = ch.get("style", base)
    if st not in PRESETS:
        st = "lineart"
    same = st == base
    pre = PRESETS[st]
    pname = ch.get("palette") or (user_meta.get("palette") if same else None) or DEFAULT_PALETTE[st]
    pal = dict(PALETTES.get(pname, PALETTES[DEFAULT_PALETTE[st]]))
    look = {k: pre["meta"].get(k) for k in LOOK_KEYS if k in pre["meta"]}
    look.setdefault("slam_pulse", True)
    if same:
        look.update({k: user_meta[k] for k in LOOK_KEYS if k in user_meta})
    look.update({k: ch[k] for k in LOOK_KEYS if k in ch})
    bg = ch.get("bg") or (user_meta.get("bg") if same else None) or pre["meta"].get("bg") or pal["paper"]
    bg = resolve_color(bg, pal)
    # v9: procedural backdrop — chapter > meta (same style) > palette default (promo only) > solid
    bg_mode = ch.get("bg_mode") or (user_meta.get("bg_mode") if same else None) or \
        (pal.get("bg_mode") if st == "promo" else None) or "solid"
    if bg_mode != "solid" and not ch.get("bg"):
        # contrast decisions use the spotlight CENTRE (where the content sits), the canvas paints the edge
        from .apex import _stops
        stops = _stops(bg_mode, pal)
        if stops:
            centre = stops[0]
            if bg_mode == "radial_spotlight":
                from .contrast import mix as _mix
                centre = _mix(stops[0], stops[1], 0.35)
            bg = centre
    pal = fit_palette(pal, bg)
    from .contrast import is_light
    if is_light(bg) and "bloom" not in ch:      # optical bloom only glows on dark grounds (paper = flat print)
        look["bloom"] = 0.0
        look["vignette"] = min(float(look.get("vignette") or 0), 0.08)
    fonts = dict(pre["fonts"])
    if same:
        fonts.update(user_meta.get("fonts") or {})
    fonts.update(ch.get("fonts") or {})
    hud = _cp.deepcopy(user_hud or {})
    _deep_default(hud, _cp.deepcopy(pre["hud"]))
    for k, v in (ch.get("hud") or {}).items():          # chapter hud wins
        if isinstance(v, dict) and isinstance(hud.get(k), dict):
            hud[k].update(v)
        else:
            hud[k] = v
    _walk_tokens(hud, pal)
    tr = (user_meta.get("transition") if same else None) or (pre["meta"].get("transition") if not same else None)
    ambient = ch.get("ambient") if "ambient" in ch else (user_meta.get("ambient") if same else None)
    return {"style": st, "palette_name": pname, "palette": pal, "bg": bg, "fonts": fonts, "bg_mode": bg_mode,
            "ambient": ambient,
            "sound": ch.get("sound") or (user_meta.get("sound") if same else None) or pre["meta"]["sound"],
            "hud": hud, "transition": tr, **look}


def apply(spec):
    """Mutates spec in place (before Scene compiles it): fills style defaults, resolves @colour tokens
    PER CHAPTER (v8: each chapter may set style / palette / bg / fonts / sound / hud) and stores the
    per-chapter worlds in meta._worlds."""
    import copy as _cp
    m = spec.setdefault("meta", {})
    user_meta = _cp.deepcopy(m)
    user_hud = _cp.deepcopy(spec.get("hud") or {})
    style = m.get("style", "lineart")
    pre = PRESETS.get(style, PRESETS["lineart"])
    _deep_default(m, pre["meta"])
    w0 = world_for(user_meta, user_hud, None)
    m["bg"] = w0["bg"]
    fonts = dict(pre["fonts"]); fonts.update(m.get("fonts") or {})
    m["fonts"] = fonts
    hud = spec.setdefault("hud", {})
    _deep_default(hud, pre["hud"])
    _walk_tokens(hud, w0["palette"])
    worlds = []
    for ch in spec.get("chapters", []):
        w = world_for(user_meta, user_hud, ch)
        _walk_tokens(ch, w["palette"])
        if "bg" in ch:
            ch["bg"] = w["bg"]
        worlds.append(w)
    m["_palette"] = w0["palette"]
    m["_worlds"] = worlds
    return spec


def fit_palette(pal, bg):
    """Background awareness for the palette itself: if the real bg disagrees with the palette's paper
    (e.g. a dark-ground palette on a cream bg), re-derive ink / accent / mute so every token stays legible.
    Text tokens get >= 4.5:1 (ink) / 3:1 (accent, bold display) / 2.2:1 (mute)."""
    from .contrast import dim, ensure, ratio
    if not bg:
        return pal
    out = dict(pal)
    out["paper"] = bg if ratio(pal.get("paper", bg), bg) > 1.25 else pal.get("paper", bg)
    def hx(c):
        return "#%02x%02x%02x" % tuple(int(round(max(0, min(1, v)) * 255)) for v in c[:3])
    out["ink"] = hx(ensure(pal["ink"], bg, 4.5, prefer=("#111111", "#ffffff")))
    out["accent"] = hx(ensure(pal["accent"], bg, 3.0))          # keeps the hue (yellow → deep gold), never drops the brand
    m = ensure(pal["mute"], bg, 2.2)
    if ratio(m, bg) > 4.0:            # mute must stay recessive, not become a second ink
        m = dim(out["ink"], bg, 2.6)
    out["mute"] = hx(m)
    return out
