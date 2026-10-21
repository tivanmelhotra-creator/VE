"""
mograph.motion.glossary — map the user's words (Persian / English / slang) onto engine vocabulary.

    from mograph.motion.glossary import resolve
    resolve("یه اسلوموشن نرم با ease out و افکت نئون روی متن")
    → [{"canonical": "retime", "op": "slow_motion", ...}, {"canonical": "easing", "type": "ease_out"},
       {"canonical": "saber", "preset": "neon", "target": "text_layer"}]

The glossary lives in glossary.yaml (user-authored, + "spec"/"cli" hints); glossary.json is the
dependency-free mirror (regenerate with `python3 -m mograph.motion glossary --sync`).
Sub-terms (SUBTERMS) turn a phrase into a concrete parameter: "ease out" → type ease_out, "تایم‌لپس" →
op fast_motion, "صاعقه" → saber preset electric …  Longest match wins, each span is used once.
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(HERE, "glossary.yaml")
JSON_PATH = os.path.join(HERE, "glossary.json")

# phrase → (canonical, {param: value})       (phrases are matched after normalise())
SUBTERMS = {
    # ---- easing types
    "ease in out": ("easing", {"type": "ease_in_out"}), "ease-in-out": ("easing", {"type": "ease_in_out"}),
    "easy ease": ("easing", {"type": "ease_in_out"}), "ایزی ایز": ("easing", {"type": "ease_in_out"}),
    "ease in": ("easing", {"type": "ease_in"}), "ease-in": ("easing", {"type": "ease_in"}),
    "ease out": ("easing", {"type": "ease_out"}), "ease-out": ("easing", {"type": "ease_out"}),
    "کاهش سرعت": ("easing", {"type": "ease_out"}), "شتاب": ("easing", {"type": "ease_in"}),
    "نرم": ("easing", {"type": "ease_in_out"}), "اینرسی": ("easing", {"type": "ease_out"}),
    "cubic": ("easing", {"type": "cubic"}), "quart": ("easing", {"type": "quart"}),
    "expo": ("easing", {"type": "expo"}), "back": ("easing", {"type": "back"}),
    "overshoot": ("easing", {"type": "back"}), "اورشوت": ("easing", {"type": "back"}),
    "elastic": ("easing", {"type": "elastic"}), "الاستیک": ("easing", {"type": "elastic"}),
    "فنری": ("easing", {"type": "elastic"}), "spring": ("easing", {"type": "spring"}),
    "bounce": ("easing", {"type": "bounce"}), "بانس": ("easing", {"type": "bounce"}),
    "linear": ("easing", {"type": "linear"}), "خطی": ("easing", {"type": "linear"}),
    "بزیه": ("easing", {"tool": "cubic-bezier"}), "bezier": ("easing", {"tool": "cubic-bezier"}),
    "cubic-bezier": ("easing", {"tool": "cubic-bezier"}),
    "graph editor": ("easing", {"tool": "graph_editor"}), "گراف ادیتور": ("easing", {"tool": "graph_editor"}),
    # ---- retime ops
    "slow motion": ("retime", {"op": "slow_motion"}), "اسلوموشن": ("retime", {"op": "slow_motion"}),
    "اسلو موشن": ("retime", {"op": "slow_motion"}), "slowmo": ("retime", {"op": "slow_motion"}),
    "slow-mo": ("retime", {"op": "slow_motion"}), "اسلو": ("retime", {"op": "slow_motion"}),
    "کند": ("retime", {"op": "slow_motion"}),
    "fast motion": ("retime", {"op": "fast_motion"}), "تایم لپس": ("retime", {"op": "fast_motion", "speed": 8.0}),
    "timelapse": ("retime", {"op": "fast_motion", "speed": 8.0}),
    "time lapse": ("retime", {"op": "fast_motion", "speed": 8.0}), "تند": ("retime", {"op": "fast_motion"}),
    "speed ramp": ("retime", {"op": "speed_ramp"}), "ریمپ": ("retime", {"op": "speed_ramp"}),
    "اسپید رمپ": ("retime", {"op": "speed_ramp"}),
    "freeze frame": ("retime", {"op": "freeze_frame"}), "freeze": ("retime", {"op": "freeze_frame"}),
    "فریز": ("retime", {"op": "freeze_frame"}), "استاپ": ("retime", {"op": "freeze_frame"}),
    "reverse": ("retime", {"op": "reverse"}), "rewind": ("retime", {"op": "reverse"}),
    "ریورس": ("retime", {"op": "reverse"}), "برعکس": ("retime", {"op": "reverse"}),
    "interpolation": ("retime", {"op": "frame_interpolation"}),
    "اینترپولیشن": ("retime", {"op": "frame_interpolation"}),
    "optical flow": ("retime", {"op": "frame_interpolation"}), "twixtor": ("retime", {"op": "frame_interpolation"}),
    # ---- character
    "rubber hose": ("character_animation", {"style": "rubber_hose"}),
    "رابر هوز": ("character_animation", {"style": "rubber_hose"}),
    "خرفه ای": ("character_animation", {"style": "rubber_hose"}),
    "bouncy": ("character_animation", {"principle": "squash_and_stretch"}),
    "squash": ("character_animation", {"principle": "squash_and_stretch"}),
    "stretch": ("character_animation", {"principle": "squash_and_stretch"}),
    "اسکواش": ("character_animation", {"principle": "squash_and_stretch"}),
    "کش و فشار": ("character_animation", {"principle": "squash_and_stretch"}),
    "جلی": ("character_animation", {"principle": "squash_and_stretch"}),
    "anticipation": ("character_animation", {"principle": "anticipation"}),
    "پیش حرکت": ("character_animation", {"principle": "anticipation"}),
    "follow through": ("character_animation", {"principle": "follow_through"}),
    "on twos": ("character_animation", {"timing": "twos"}), "on threes": ("character_animation", {"timing": "threes"}),
    "دوتایی": ("character_animation", {"timing": "twos"}),
    "cutout": ("character_animation", {"technique": "cutout"}), "کات اوت": ("character_animation", {"technique": "cutout"}),
    "puppet": ("character_animation", {"technique": "puppet_pin"}), "پاپت": ("character_animation", {"technique": "puppet_pin"}),
    "rig": ("character_animation", {"technique": "rigging"}), "ریگ": ("character_animation", {"technique": "rigging"}),
    # ---- saber presets / targets
    "شمشیر نوری": ("saber", {"preset": "default"}), "lightsaber": ("saber", {"preset": "default"}),
    "light saber": ("saber", {"preset": "default"}),
    "نئون": ("saber", {"preset": "neon"}), "neon": ("saber", {"preset": "neon"}),
    "رعد": ("saber", {"preset": "electric"}), "صاعقه": ("saber", {"preset": "electric"}),
    "برق": ("saber", {"preset": "electric"}), "الکتریک": ("saber", {"preset": "electric"}),
    "lightning": ("saber", {"preset": "electric"}), "electric": ("saber", {"preset": "electric"}),
    "آتش": ("saber", {"preset": "fire"}), "fire": ("saber", {"preset": "fire"}), "اتش": ("saber", {"preset": "fire"}),
    "دود": ("saber", {"preset": "soft_smoke"}), "smoke": ("saber", {"preset": "soft_smoke"}),
    "پورتال": ("saber", {"preset": "portal"}), "portal": ("saber", {"preset": "portal"}),
    "انرژی": ("saber", {"preset": "energy"}), "energy": ("saber", {"preset": "energy"}),
    "افکت انرژی": ("saber", {"preset": "energy"}),
    "لیزر": ("saber", {"preset": "thin"}), "laser": ("saber", {"preset": "thin"}),
    "پرتو": ("saber", {"preset": "tractor_beam"}), "beam": ("saber", {"preset": "tractor_beam"}),
    "روی متن": ("saber", {"target": "text_layer"}), "دور متن": ("saber", {"target": "text_layer"}),
    "on text": ("saber", {"target": "text_layer"}), "دور شکل": ("saber", {"target": "shape_layer"}),
    "ماسک": ("saber", {"target": "mask"}), "mask": ("saber", {"target": "mask"}),
    # ---- lip sync
    "لیپ سینک": ("lip_sync", {}), "lip sync": ("lip_sync", {}), "lipsync": ("lip_sync", {}),
}

_AR2FA = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی", "ة": "ه", "أ": "ا", "إ": "ا", "ۀ": "ه", "ؤ": "و",
                        "\u200c": " ", "\u200d": "", "\u064b": "", "\u064c": "", "\u064d": "", "\u064e": "",
                        "\u064f": "", "\u0650": "", "\u0651": "", "\u0652": "", "ـ": ""})


def normalise(s: str) -> str:
    """lower-case, Arabic→Persian letters, ZWNJ → space, punctuation → space, collapse spaces."""
    s = str(s).translate(_AR2FA).lower()
    s = re.sub(r"[_\.\,،؛;:!؟?\"'«»()\[\]{}/\\|]+", " ", s)
    s = s.replace("-", " ") if "cubic bezier" not in s else s
    return re.sub(r"\s+", " ", s).strip()


def load():
    """glossary dict {key: entry}.  YAML when PyYAML is present, else the JSON mirror."""
    try:
        import yaml  # noqa: WPS433
        with open(YAML_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)["motion_glossary"]
    except Exception:  # noqa: BLE001
        with open(JSON_PATH, encoding="utf-8") as f:
            return json.load(f)["motion_glossary"]


def sync_json():
    import yaml
    with open(YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return JSON_PATH


def _phrases():
    """[(normalised phrase, canonical, params)] longest first."""
    out = {}
    for key, e in load().items():
        can = e["canonical"]
        for a in list(e.get("aliases", [])) + [can, key]:
            out.setdefault(normalise(a), (can, {}))
        for ty in e.get("types", []) + e.get("operations", []) + e.get("principles", []) + e.get("techniques", []):
            k = "type" if ty in e.get("types", []) else "op" if ty in e.get("operations", []) else \
                "principle" if ty in e.get("principles", []) else "technique"
            out[normalise(ty.replace("_", " "))] = (can, {k: ty})
    for p, v in SUBTERMS.items():
        out[normalise(p)] = v
    return sorted(((p, c, d) for p, (c, d) in out.items() if p), key=lambda r: -len(r[0]))


_CACHE = {}


# single generic words that only count when ANOTHER concept word of the same family is present (avoid
# "go back home" → easing back, "نوری که از پنجره میاد" → saber)
WEAK = {"back", "نور", "نرم", "تند", "کند", "سرعت", "speed", "glow", "beam", "fire", "smoke", "دود", "آتش", "اتش",
        "energy", "انرژی", "برق", "mouth", "دهان", "rig", "ریگ", "puppet", "curve", "منحنی", "اسلو", "mask", "ماسک"}


def resolve(text: str, strict=True):
    """free text → list of {canonical, matched, **params} (merged per canonical, order of first appearance)."""
    if "p" not in _CACHE:
        _CACHE["p"] = _phrases()
    s = " " + normalise(text) + " "
    used = [False] * len(s)
    hits = []
    for p, can, params in _CACHE["p"]:
        start = 0
        pat = " " + p + " " if re.match(r"^[a-z0-9 ]+$", p) else p      # latin words need word boundaries
        while True:
            i = s.find(pat, start)
            if i < 0:
                break
            j = i + len(pat)
            if not any(used[i:j]):
                for k in range(i, j):
                    used[k] = True
                hits.append((i, can, p, params))
            start = j
    hits.sort()
    if strict:                                    # drop weak words whose concept has no strong word in the text
        strong = {can for _, can, p, _ in hits if p not in WEAK}
        hits = [h for h in hits if h[2] not in WEAK or h[1] in strong]
    merged, order, best = {}, [], {}
    for i, can, p, params in hits:
        if can not in merged:
            merged[can] = {"canonical": can, "matched": []}
            order.append(can)
        merged[can]["matched"].append(p)
        for k, v in params.items():               # the most explicit (longest) phrase wins a parameter …
            if len(p) > best.get((can, k), -1):
                best[(can, k)] = len(p)
                merged[can][k] = v
            if k in ("op", "principle", "preset"):   # … and sequences keep every value in spoken order
                lst = merged[can].setdefault(k + "s", [])
                if v not in lst:
                    lst.append(v)
    for c in list(merged.values()):
        for k in ("ops", "principles", "presets"):
            if len(c.get(k, [])) < 2:
                c.pop(k, None)
    cols = [(i, v) for w, v in COLOURS.items() for i in [s.find(" " + w + " ")] if i >= 0]
    if cols and "saber" in merged:
        merged["saber"]["color"] = sorted(cols)[0][1]
    elif cols:
        merged["_color"] = {"canonical": "color", "matched": [], "color": sorted(cols)[0][1]}
        order.append("_color")
    return [merged[c] for c in order]


COLOURS = {"قرمز": "#ff2a2a", "red": "#ff2a2a", "آبی": "#2a7dff", "blue": "#2a7dff", "سبز": "#27ff6a",
           "green": "#27ff6a", "زرد": "#ffd21f", "yellow": "#ffd21f", "نارنجی": "#ff7a1a", "orange": "#ff7a1a",
           "بنفش": "#a43cff", "purple": "#a43cff", "صورتی": "#ff3fb4", "pink": "#ff3fb4", "سفید": "#ffffff",
           "white": "#ffffff", "طلایی": "#ffb627", "gold": "#ffb627", "فیروزه ای": "#18e6e0", "cyan": "#18e6e0",
           "سرخابی": "#ff1f8f", "magenta": "#ff1f8f"}


def describe(canonical: str):
    for key, e in load().items():
        if e["canonical"] == canonical or key == canonical:
            return dict(e, key=key)
    return None
