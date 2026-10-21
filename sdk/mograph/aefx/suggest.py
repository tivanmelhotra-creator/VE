"""
mograph.aefx.suggest — intent → ranked presets, from the MEASURED catalogue (presets/catalog.json).

    from mograph.aefx.suggest import suggest
    suggest("soft elegant title", dir_="in")      → [("motion"|"text", key, score, why), ...]

Intent words (English or Persian) map to wanted / unwanted tags and numeric targets (energy, scale, rotation …).
Clean-title safety: IN presets that do not end at identity ("settles-offset") are penalised unless asked for.
"""
from __future__ import annotations

import json
import os

_CAT = None
VOCAB = {   # word → (wanted tags, unwanted tags, energy target or None)
    "soft": ({"soft"}, {"punchy", "distort", "spin"}, 2.0), "calm": ({"soft"}, {"punchy", "distort"}, 2.0),
    "elegant": ({"soft"}, {"punchy", "distort", "random"}, 2.5), "luxury": ({"soft", "blur"}, {"distort"}, 2.5),
    "punchy": ({"punchy"}, {"soft"}, 12.0), "energetic": ({"punchy", "overshoot"}, {"soft"}, 12.0),
    "hype": ({"punchy", "overshoot", "spin"}, {"soft"}, 14.0), "spin": ({"spin"}, set(), None),
    "3d": ({"3d"}, set(), None), "blur": ({"blur"}, set(), None), "dreamy": ({"blur", "soft"}, {"punchy"}, 3.0),
    "glitch": ({"distort", "random"}, {"soft"}, None), "tech": ({"distort", "decode", "random"}, set(), None),
    "shake": ({"punchy"}, set(), None), "impact": ({"punchy", "overshoot"}, {"soft"}, 13.0),
    "slide": ({"travel"}, set(), None), "travel": ({"travel"}, set(), None),
    "bounce": ({"overshoot"}, set(), None), "overshoot": ({"overshoot"}, set(), None),
    "decode": ({"decode"}, set(), None), "type": ({"chars"}, set(), None), "typewriter": ({"chars"}, set(), None),
    "word": ({"words"}, set(), None), "line": ({"lines"}, set(), None), "random": ({"random"}, set(), None),
    "wiggle": ({"wiggly"}, set(), None),
    # Persian
    "نرم": ({"soft"}, {"punchy", "distort"}, 2.0), "لوکس": ({"soft", "blur"}, {"distort"}, 2.5),
    "پرانرژی": ({"punchy", "overshoot"}, {"soft"}, 12.0), "ضربه": ({"punchy", "overshoot"}, {"soft"}, 13.0),
    "چرخش": ({"spin"}, set(), None), "سه‌بعدی": ({"3d"}, set(), None), "بلور": ({"blur"}, set(), None),
    "گلیچ": ({"distort", "random"}, set(), None), "تکنولوژی": ({"distort", "decode"}, set(), None),
    "لرزش": ({"punchy"}, set(), None), "حرکت": ({"travel"}, set(), None), "تایپ": ({"chars"}, set(), None),
}
CAT_HINT = {"shake": "Shake", "glitch": "Glitch", "flicker": "Flicker", "blur": "Blur", "spin": "Rotation",
            "slide": "Position", "3d": "3D Rotation", "decode": "Coding", "type": "Coding", "film": "Static",
            "لرزش": "Shake", "گلیچ": "Glitch", "تایپ": "Coding"}


def catalog():
    global _CAT
    if _CAT is None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets", "catalog.json"),
                  encoding="utf-8") as f:
            _CAT = json.load(f)
    return _CAT


def suggest(intent, dir_=None, n=8, kinds=("motion", "text")):
    words = [w.strip().lower() for w in intent.replace(",", " ").split() if w.strip()]
    want, avoid, etarget, cats = set(), set(), [], set()
    for w in words:
        if w in VOCAB:
            a, b, e = VOCAB[w]
            want |= a
            avoid |= b
            if e is not None:
                etarget.append(e)
        if w in CAT_HINT:
            cats.add(CAT_HINT[w].lower())
    et = sum(etarget) / len(etarget) if etarget else None
    title = any(w in ("title", "clean", "تیتر", "عنوان") for w in words)
    out = []
    C = catalog()
    for kind in kinds:
        for k, v in C[kind].items():
            if kind == "motion" and dir_ and v["dir"] != dir_:
                continue
            tags = set(v["tags"])
            sc = 3.0 * len(tags & want) - 3.0 * len(tags & avoid)
            catname = (v.get("cat", "") + " " + v.get("sub", "")).lower()
            sc += 4.0 * sum(1 for c in cats if c in catname)
            if et is not None and "energy" in v:
                sc -= abs(v["energy"] - et) * 0.4
            if "settles-offset" in tags and (title or "soft" in want):
                sc -= 6
            if "stays-visible" in tags:
                sc -= 2
            if sc <= 0 and (want or cats):
                continue
            why = " ".join(sorted(tags & want)) or v.get("cat", "")
            out.append((kind, k, round(sc, 2), f"{v.get('cat', '')}/{v.get('dir', v.get('sub', ''))} [{' '.join(v['tags'])}] {why}"))
    out.sort(key=lambda r: -r[2])
    return out[:n]
