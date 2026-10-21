"""
mograph.v14.director — AUTOMATIC BEAT-SHEET DIRECTOR (v14.8; resolves SKILL_TOOLBOX backlog #3).

    python3 -m mograph.v14 auto-edit words.json media/raw.mp4 -o edit_auto.json [--theme doran] [--no-qa]

Input : a word-timed transcript (words.json: {"words":[{"text","start","end"}]} | [{"w","t","end"}] | [[w,t,end]])
        + the raw talking-head video.
Output: a complete, QA-clean v14 spec (same schema as hand-made edit_v14.json) — ready for `frame` / `render`.

PIPELINE (every step is deterministic → same input = same edit):
  1  BREATH CUTS     pauses between words > 0.32 s are removed → source.segments (0.10 s pre-roll, 0.14 s tail,
                     very short islands are merged with their neighbour).
  2  WORD SCORING    every word gets an "impact" score from cheap Persian linguistics (no model):
                       + content length (letters), + numbers / number words / ordinals (اولین، دومین، ۳، صد…)
                       + emphasis lexicon (آخر، اول، مهم، دقیقا، هرگز، نه، فقط، بهترین…)
                       + topic repetition (a noun said 2-3× is the TOPIC → strong first time, weak after)
                       + prosody: phrase-final words (before a pause / punctuation) and slowly spoken words
                       − stop words, pronouns, prepositions, auxiliaries, conjugated verbs (می‌/ن… prefixes, -ید/-ین…)
  3  BEAT SHEET      one *hero* word every 2.5 – 3.5 s (greedy best-score inside each window; first beat = hook).
  4  DEPTH ROTATION  beats cycle through the depth recipe (never two neighbours on the same plane):
                       kinetic_title BACK → orbit_ring (wraps the head) → frame_box + object_3d → disc (flip_at)
                       → halo BACK → glass_shape FRONT (+ subtitles.hide windows for chest elements)
                     the closing beat is always the FRONT glass badge (the take-away card).
                     texts are built from the spoken context (stop words trimmed), sizes from text length.
  5  CAMERA / GRADE  push-in on every beat (alternating silk / expo / jelly), punch-zoom + shake on cuts,
                     background dim/tint keys on the big chest/halo beats, flash on every cut, bokeh dust bed.
  6  QA FIX LOOP     the spec is loaded by the real Renderer, `qa_warnings` is run and each warning is repaired
                     automatically (too-short → drop, frame edge → shrink, face → push away, subtitle → re-phrase)
                     until QA is clean (max 5 rounds).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

PAUSE = 0.32
PRE_ROLL, TAIL = 0.10, 0.14
BEAT_MIN, BEAT_MAX = 2.5, 3.5

STOP = set("""
که و از به با را رو در تا هم یا اگر چون پس این اون آن همین همون اینو اونو ما من تو او شما ایشون آنها اونا
خودم خودت خودش خودمون خودتون یه یک برای باید هست هستش است بود شد شده میشه میشید نیست نمی ولی اما چه چی
چیزی کسی هر خیلی دیگه دیگر مثل مثلا یعنی وقتی موقع اینکه اونکه کنین کنید کنیم کرد کردن کنه داره دارید داریم
بشه بشید باشه باشید بر روی زیر بین بعد قبل الان حالا فقط خب خوب آره بله درسته
""".split())
EMPH = {"آخر": 2.2, "اول": 2.0, "اولین": 1.4, "آخرین": 1.6, "مهم": 2.0, "مهمترین": 2.4, "دقیقا": 1.2, "هرگز": 2.4,
        "نه": 2.2, "حتما": 1.6, "بهترین": 2.2, "بدترین": 2.2, "اشتباه": 2.2, "راز": 2.4, "شروع": 1.8, "پایان": 1.8,
        "نکته": 2.0, "همیشه": 1.6, "هیچ": 1.4, "رایگان": 2.4, "سریع": 1.4, "قیمت": 1.8, "پول": 1.8, "خطر": 2.4}
NUM_WORDS = {"یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "ده", "صد", "هزار", "میلیون", "میلیارد", "درصد",
             "دومین", "سومین", "چهارمین", "پنجمین", "نصف", "دوبرابر"}
VERB_PREFIX = re.compile(r"^(می|نمی|بی)\u200c")
VERB_SUFFIX = re.compile(r"(ید|ین|یم|ند|یدن)$")
DIGITS = re.compile(r"[0-9۰-۹]")
# object_3d asset picked from the hero word (bundled sdk/mograph/assets_3d) — generic fallback = pin
OBJ_MAP = [(("لپ", "کامپیوتر", "سیستم"), "laptop_3d.png"), (("پول", "قیمت", "هزینه", "تومن", "دلار", "درآمد"), "cash_3d.png"),
           (("هشدار", "خطر", "اشتباه", "مراقب"), "warn_3d.png"), (("امن", "محافظ", "بیمه", "سپر", "شروع"), "shield_3d.png"),
           (("سفر", "پرواز", "هواپیما"), "plane_3d.png"), (("نمره", "عالی", "قبول", "رتبه"), "grade_a_3d.png")]
ROTATION = ["kinetic_back", "orbit_ring", "frame_box", "disc", "halo", "glass_shape"]
CHEST = {"disc", "glass_shape"}
FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"


# =================================================================================================== 1 cuts
def build_segments(words, src_dur, pause=PAUSE):
    """Speech islands separated by pauses > `pause` → [[a, b], ...] in source seconds."""
    if not words:
        return [[0.0, src_dur]]
    groups, cur = [], [words[0]]
    for w in words[1:]:
        if w["t"] - cur[-1]["end"] > pause:
            groups.append(cur)
            cur = [w]
        else:
            cur.append(w)
    groups.append(cur)
    segs = []
    for g in groups:
        a = max(0.0, g[0]["t"] - PRE_ROLL)
        b = min(src_dur, g[-1]["end"] + TAIL)
        if segs and (a - segs[-1][1] < 0.12 or (b - a) < 0.45):    # tiny island / overlap → merge
            segs[-1][1] = max(segs[-1][1], round(b, 3))
        else:
            segs.append([round(a, 3), round(b, 3)])
    return segs


# =================================================================================================== 2 scoring
def _clean(w):
    from .compositor import norm_word
    return norm_word(w)


def score_words(tw):
    """tw = Timeline.words (output time).  Returns list of floats (impact score per word)."""
    counts = {}
    for w in tw:
        counts[w["n"]] = counts.get(w["n"], 0) + 1
    seen, sc = {}, []
    for i, w in enumerate(tw):
        n, raw = w["n"], w["w"]
        if not n:
            sc.append(-9.0)
            continue
        s = 0.0
        letters = len(n)
        if n in STOP:
            s -= 3.0
        s += min(2.0, 0.42 * max(0, letters - 2))
        if DIGITS.search(n) or n in NUM_WORDS:
            s += 2.6
        s += EMPH.get(n, 0.0)
        if VERB_PREFIX.match(raw):
            s -= 2.2
        elif letters >= 4 and VERB_SUFFIX.search(n) and n not in EMPH and n not in NUM_WORDS:
            s -= 0.6
        k = seen.get(n, 0)
        if counts[n] >= 2 and n not in STOP:
            s += 1.6 if k == 0 else (0.5 if k == 1 else -0.4)
        seen[n] = k + 1
        nxt = tw[i + 1] if i + 1 < len(tw) else None
        if raw.rstrip()[-1:] in "،,.!?؟" or nxt is None or nxt["t"] - w["end"] > 0.18:
            s += 0.8                                               # phrase-final stress (Persian)
        dur = w["end"] - w["t"]
        s += min(1.0, max(0.0, (dur / max(1, letters) - 0.07) * 9))  # slowly spoken = stressed
        sc.append(round(s, 3))
    return sc


# =================================================================================================== 3 beats
def pick_beats(tw, scores, duration, bmin=BEAT_MIN, bmax=BEAT_MAX):
    """Greedy: one hero word per [last + bmin, last + bmax] window (first beat = best word in the opening)."""
    beats = []
    cand = [(i, w) for i, w in enumerate(tw) if scores[i] > 0.8] or list(enumerate(tw))
    first = [c for c in cand if c[1]["t"] <= min(2.6, duration * 0.3)]
    if first:
        beats.append(max(first, key=lambda c: scores[c[0]] - 0.15 * c[1]["t"])[0])
    last_t = tw[beats[0]]["t"] if beats else -bmin
    used = {tw[b]["n"] for b in beats}
    while True:
        lo, hi = last_t + bmin, last_t + bmax
        if lo > duration - 0.9:
            break
        win = [c for c in cand if lo <= c[1]["t"] <= hi and c[1]["t"] < duration - 0.6]
        if not win:
            later = [c for c in cand if hi < c[1]["t"] < duration - 0.6]
            if not later:
                break
            win = [later[0]]
        i, w = max(win, key=lambda c: scores[c[0]] - (1.5 if c[1]["n"] in used else 0.0))
        beats.append(i)
        used.add(w["n"])
        last_t = w["t"]
    return beats


# =================================================================================================== helpers
def _anchor(tl, w, end=False, off=0.0):
    """'w:word#k' anchor for this exact word instance (falls back to seconds when the anchor is ambiguous)."""
    exact = [x for x in tl.words if x["n"] == w["n"]]
    k = next((j for j, x in enumerate(exact) if x is w), None)
    base = w["end"] if end else w["t"]
    if k is None or not w["n"]:
        return round(base + off, 3)
    a = f"{'we' if end else 'w'}:{w['n']}" + (f"#{k + 1}" if k else "")
    try:
        if abs(tl.resolve(a) - base) > 1e-6:
            return round(base + off, 3)
    except KeyError:
        return round(base + off, 3)
    if off:
        a += f"{off:+.2f}"
    return a


def _txt(w):
    return w["w"].strip("،,.!?؟:;«»\"'()")


def _ctx(tw, i, before=3, after=4, seg_of=None):
    """Context words around the hero inside the same speech segment."""
    pre, post = [], []
    j = i - 1
    while j >= 0 and len(pre) < before and (seg_of is None or seg_of[j] == seg_of[i]) and tw[i]["t"] - tw[j]["t"] < 1.6:
        pre.insert(0, _txt(tw[j]))
        j -= 1
    j = i + 1
    while j < len(tw) and len(post) < after and (seg_of is None or seg_of[j] == seg_of[i]) and tw[j]["t"] - tw[i]["end"] < 1.2:
        post.append(_txt(tw[j]))
        j += 1
    return pre, post


def _trim(ws):
    """Drop stop words / verbs at both ends of a context phrase («و اولین بخشی که» → «اولین بخشی»)."""
    def weak(t):
        n = _clean(t)
        return n in STOP or len(n) <= 1 or bool(VERB_PREFIX.match(t))
    ws = list(ws)
    while ws and weak(ws[0]):
        ws.pop(0)
    while ws and weak(ws[-1]):
        ws.pop()
    return ws


def _fa_num(n):
    return "".join(FA_DIGITS[int(c)] for c in str(n))


def _obj_for(hero):
    for keys, src in OBJ_MAP:
        if any(k in hero for k in keys):
            return src
    return "pin_3d.png"


# =================================================================================================== 4 layers
def build_layers(tl, tw, scores, beats, duration, seg_of):
    layers, hides, camera, shakes, gkeys = [], [], [[0.0, 1.08, 0.5, 0.42]], [], []
    eases = ["silk", "expo", "jelly"]
    for bi, i in enumerate(beats):
        w = tw[i]
        hero = _txt(w)
        kind = ROTATION[bi % len(ROTATION)]
        if bi == len(beats) - 1 and bi >= 3 and kind != "glass_shape":
            kind = "glass_shape"                  # closing beat = front glass badge (the take-away card)
        nxt_t = tw[beats[bi + 1]]["t"] if bi + 1 < len(beats) else duration
        start_w = tw[i - 1] if bi > 0 and i > 0 and w["t"] - tw[i - 1]["t"] < 0.8 else w
        t0 = _anchor(tl, start_w, off=-0.08) if start_w is not w else (_anchor(tl, w, off=-0.08) if bi else 0.05)
        t1v = min(duration - 0.15, nxt_t - 0.12)
        t1 = round(t1v, 3)
        pre, post = _ctx(tw, i, 3, 4, seg_of)
        pre, post = _trim(pre), _trim(post)
        lid = f"b{bi + 1}_{kind}"
        hero_hit = _anchor(tl, w)
        ln = len(hero)
        if kind == "kinetic_back":
            ctx = " ".join(pre[-2:])
            layers.append({"type": "kinetic_title", "id": lid, "t": t0, "out": t1, "anchor": "head_top",
                           "offset": [0, -0.62], "size": 78 if ln <= 6 else 66, "pass": "back",
                           "text": (f"{ctx}|*{hero}*" if ctx else f"*{hero}*"), "enter": "drop_bounce", "exit": "pop_out",
                           "idle": "float", "plane_in": {"rx": -70}, "extrude": {"depth": 10, "light": 0.7},
                           "hero_hit": hero_hit, "stagger": 0.03,
                           # v15.1 AE-style selector: a light wave runs through the line once after the drop
                           "selectors": [{"shape": "smooth", "start": 0.0, "end": 0.35,
                                          "offset": [0.55, 1.25, -0.4, 1.05],
                                          "props": {"ty": -14, "glow": 0.9, "scale": 0.08}}]})
        elif kind == "orbit_ring":
            tail = " ".join(_trim(post[:3]) or pre[-2:])
            layers.append({"type": "orbit_ring", "id": lid, "t": t0, "out": t1, "anchor": "head_top",
                           "offset": [0, 0.15], "size": 80 if ln <= 6 else 70, "radius": 1.4, "tilt": -14, "spin": 48,
                           "phase": 160, "text": (f"*{hero}* = {tail}" if tail else f"*{hero}*"),
                           "styles": {"hero": {"scale": 1.45}}, "extrude": {"depth": 7, "light": 0.6},
                           "max_copies": 2, "sep": 1.0, "max_radius": 1.6})
        elif kind == "frame_box":
            ctx = " ".join(pre[-1:])
            layers.append({"type": "frame_box", "id": lid, "t": t0, "out": t1, "anchor": "head_top",
                           "offset": [0, -0.62], "size": 84 if ln <= 6 else 72, "w": 2.3, "h": 1.15, "rot": -7,
                           "pass": "back", "text": (f"{ctx}|*{hero}*" if ctx else f"*{hero}*"), "styles": {},
                           "badge_text": _fa_num(bi + 1), "extrude": {"depth": 6}})
            layers.append({"type": "object_3d", "id": lid + "_obj", "t": _anchor(tl, w, off=0.1), "out": t1,
                           "src": _obj_for(hero), "geometry": "auto", "anchor": "shoulder_r",
                           "offset": [0.05, 0.3], "size": 0.8, "motion": "turntable_osc", "yaw_amp": 28,
                           "pitch": 12, "enter": "spring_pop"})
        elif kind == "disc":
            top = " ".join(pre[-3:])
            bot = " ".join(_trim(post[:3]))
            if len(top) < 6:                       # weak lead-in → read the sentence across both arcs
                top, bot = bot or hero, " ".join(_trim(post[3:6])) or top or hero
            mid = tw[min(len(tw) - 1, i + max(1, len(post) // 2 + 1))]
            flip = _anchor(tl, mid, off=-0.1) if mid["t"] < t1v - 0.3 else round((w["t"] + t1v) / 2, 3)
            layers.append({"type": "disc", "id": lid, "t": t0, "out": t1, "anchor": "chest", "offset": [0, 0.15],
                           "radius": 0.92, "flip_at": flip, "neon": "@alert", "spill": 1.2,
                           "center_text": f"*{hero}*", "top_text": top or hero, "bottom_text": bot or hero,
                           "arc_size": 0.15, "follow": 0.6})
            gkeys.append([t0, {"dim": 0.6, "blur": 11, "tint": "#1C0B0B"}])
            gkeys.append([round(t1v + 0.1, 3), {"dim": 0.44, "blur": 7, "tint": "#2A1C12"}])
        elif kind == "halo":
            ctx = " ".join(pre[-1:])
            layers.append({"type": "halo", "id": lid, "t": t0, "out": t1, "pass": "back",
                           "text": (f"^{ctx}^|={hero}=" if ctx else f"={hero}="),
                           "styles": {"hero_white": {"scale": 1.8, "glow": 0.45}, "tiny": {"scale": 0.75}},
                           "color": "@alert", "peek": True, "peek_glyphs": 1, "text_mode": "center",
                           "size": 96 if ln <= 5 else 80})
            gkeys.append([t0, {"dim": 0.7, "blur": 12, "tint": "#160606", "desat": 0.65}])
            gkeys.append([round(t1v + 0.1, 3), {"dim": 0.48, "blur": 8, "tint": "#22180E", "desat": 0.4}])
        else:  # glass_shape
            ctx = " ".join(pre[-2:])
            layers.append({"type": "glass_shape", "id": lid, "t": t0, "out": t1, "shape": "hept", "anchor": "chest",
                           "offset": [0, 0.55], "w": 1.6, "h": 1.42, "size": 56 if ln <= 6 else 48,
                           "text": (f"^{ctx}^|*{hero}*" if ctx else f"*{hero}*"),
                           "styles": {"hero": {"scale": 1.7}, "tiny": {"scale": 0.7}}, "style": "nested", "gap": 7})
        if kind in CHEST:
            hides.append([t0, t1])
        z = [1.16, 1.12, 1.2, 1.06, 1.24, 1.05][bi % 6]
        camera.append([hero_hit, z, 0.48, 0.47 if kind in CHEST else 0.41, eases[bi % 3]])
        if kind in ("disc", "halo", "glass_shape") or bi == 0:
            shakes.append([hero_hit, 0.3, 8])
    for ci, c in enumerate(tl.cuts):          # cut accents
        layers.append({"type": "flash", "id": f"cut{ci + 1}", "t": round(c - 0.02, 3), "out": round(c + 0.32, 3),
                       "dur": 0.16, "amount": 0.4})
        camera.append([round(c - 0.01, 3), None, None, None, "step"])
        camera.append([round(c, 3), "punch", None, None, "step"])
        shakes.append([round(c, 3), 0.22, 7])
    layers.insert(0, {"type": "progress", "id": "prog", "t": 0, "h": 6, "color": "@accent"})
    layers.insert(1, {"type": "ambient_bokeh", "id": "dust", "t": 0, "out": round(duration, 3), "back": 12,
                      "front": 6, "alpha": 0.9})
    return layers, hides, _resolve_camera(tl, camera), shakes, gkeys


def _resolve_camera(tl, keys):
    """Fill 'step' / 'punch' placeholders for cut keys from the preceding regular key (sorted by time)."""
    def T(k):
        return tl.resolve(k[0])
    reg = sorted([k for k in keys if k[1] not in (None, "punch")], key=T)
    out = list(reg)
    for k in keys:
        if k[1] is None or k[1] == "punch":
            t = T(k)
            prev = [r for r in reg if T(r) <= t]
            p = prev[-1] if prev else reg[0]
            z = p[1] if k[1] is None else round(p[1] * 1.1, 3)
            out.append([k[0], z, p[2], p[3], "step"])
    out.sort(key=T)
    return out


# =================================================================================================== 6 qa loop
def _qa_fix(spec, base_dir, rounds=5, log=print):
    from .compositor import Renderer
    from .qa import qa_warnings
    for r in range(rounds):
        warn = qa_warnings(Renderer(copy.deepcopy(spec), base_dir))
        if not warn:
            log(f"[director] QA clean ✓ (round {r + 1})")
            return spec, []
        log(f"[director] QA round {r + 1}: {len(warn)} warning(s) → auto-fix")
        by_id = {L.get("id"): L for L in spec["layers"]}
        drop = set()
        for w in warn:
            log("   ⚠ " + w)
            if w.startswith("subtitle"):
                s = spec["subtitles"]
                s["max_words"] = max(2, int(s.get("max_words", 4)) - 1)
                continue
            lid = w.split(":", 1)[0]
            L = by_id.get(lid)
            if L is None:
                continue
            if "very short" in w or "starts after the end" in w:
                drop.add(lid)
            elif "touches frame edge" in w:
                for kk in ("size", "radius", "w", "h"):
                    if isinstance(L.get(kk), (int, float)):
                        L[kk] = round(L[kk] * 0.88, 3)
            elif "FACE" in w:
                off = L.setdefault("offset", [0, 0])
                if L.get("anchor") in ("head_top", "face", "nose"):
                    off[1] = round(off[1] - 0.18, 3)
                elif str(L.get("anchor", "")).startswith("shoulder") or L["type"] == "object_3d":
                    off[0] = round(off[0] + (0.25 if off[0] >= 0 else -0.25), 3)
                    off[1] = round(off[1] + 0.15, 3)
                else:
                    off[1] = round(off[1] + 0.18, 3)
                if L["type"] == "kinetic_title":
                    L["size"] = round(L.get("size", 72) * 0.92, 2)
        if drop:
            spec["layers"] = [L for L in spec["layers"] if L.get("id") not in drop]
    return spec, qa_warnings(Renderer(copy.deepcopy(spec), base_dir))


# =================================================================================================== main
def auto_edit(words_path, video_path, out_path, theme="doran", qa=True, log=print):
    import cv2
    from .compositor import Timeline, load_words
    out_path = os.path.abspath(out_path)
    base = os.path.dirname(out_path)
    os.makedirs(base, exist_ok=True)
    words = load_words(words_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise SystemExit(f"[director] cannot read video: {video_path}")
    segs = build_segments(words, n / fps)
    tl = Timeline(segs, words)
    tw = tl.words
    seg_of = [sum(1 for c in tl.cuts if c <= w["t"] + 1e-6) for w in tw]
    scores = score_words(tw)
    beats = pick_beats(tw, scores, tl.duration)
    log(f"[director] {len(words)} words, {len(segs)} segments (pauses > {PAUSE}s cut) → {tl.duration:.2f}s edit")
    log("[director] beats: " + "  ".join(f"{tw[b]['t']:.2f}s *{_txt(tw[b])}* ({scores[b]:.1f})" for b in beats))
    layers, hides, cam, shakes, gkeys = build_layers(tl, tw, scores, beats, tl.duration, seg_of)
    heroes = [_txt(tw[b]) for b in beats]
    rel = lambda p: os.path.relpath(os.path.abspath(p), base)   # noqa: E731
    spec = {
        "_about": "v15.1 AUTO-EDIT (mograph.v14.director): breath cuts + scored hero words + depth rotation. "
                  "Edit freely — every time is a word anchor.",
        "meta": {"w": 1080, "h": 1920, "fps": 30, "theme": theme, "palette": "auto"},
        "source": {"src": rel(video_path), "words": rel(words_path), "segments": segs,
                   "morph_cuts": {"dur": 0.24, "max_motion": 60}},
        "grade": {"bg": {"blur": 7, "dim": 0.42, "desat": 0.22, "tint": "#3A2414", "tint_amt": 0.24,
                         "vignette_bg": 0.35},
                  "fg": {"contrast": 1.07, "warm": 0.035, "sat": 1.04, "lift": 0.0},
                  "keys": gkeys},
        "camera": {"keys": cam, "shakes": shakes},
        "subtitles": {"anchor": "chest", "offset": [0, 0.95], "size": 44, "max_words": 4, "emph": heroes,
                      "hide": hides},
        "layers": layers,
        "post": {"bloom": 1.0, "spill": 1.3, "wrap_bg": 0.5, "wrap_glow": 1.8, "vignette": 0.3, "grain": 0.016,
                 "fade": [0.0, 0.35], "glass": {"refract": 4.0, "chroma": 1.5, "caustic": 0.55, "fresnel": 0.35},
                 "decontaminate": 0.75,
                 "motion_blur": {"shutter": 180}},
        "audio": {"voice_gain": 1.0, "music": {"style": "lofi", "bpm": 92, "key": "D", "scale": "minor", "gain": 0.16},
                  "sfx_gain": 0.55, "lufs": -14, "denoise": "auto", "multiband": True, "vocal_booth_reverb": 0.04},
    }
    remaining = []
    if qa:
        spec, remaining = _qa_fix(spec, base, log=log)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
    log(f"[director] wrote {out_path}  ({len(spec['layers'])} layers, {len(beats)} beats, "
        f"{'QA clean' if not remaining else str(len(remaining)) + ' QA warning(s) left'})")
    return out_path, remaining


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mograph.v14 auto-edit", description="v14.8 automatic beat-sheet director")
    ap.add_argument("words")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="edit_auto.json")
    ap.add_argument("--theme", default="doran")
    ap.add_argument("--no-qa", action="store_true")
    a = ap.parse_args(argv)
    _, rem = auto_edit(a.words, a.video, a.out, a.theme, qa=not a.no_qa)
    return 1 if rem else 0


if __name__ == "__main__":
    sys.exit(main())
