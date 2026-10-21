"""
mograph.motion.lipsync — words (+ audio) → a timed VISEME track on the Preston Blair mouth chart.

Preston Blair / Hanna-Barbera 10-mouth set (the classic animation chart; letters = Papagayo / Rhubarb names):
    rest  X   closed, relaxed                 MBP   A   lips pressed (m b p)
    AI    D   wide open (a, ā, i in "father") E     C   mid open, teeth (e, eh, ae)
    O     E   round open (o, aw)              U     F   small round / pucker (u, oo, w, و)
    FV    G   lower lip under teeth (f v)     L     H   tongue up behind teeth (l, th, d-n "la")
    WQ    F   pucker (w, q, و consonant)      etc   B   slightly open teeth (c d g k n r s t y z ش ...)

Persian: words are in Persian script (no short vowels written) → a rule-based grapheme→phoneme pass:
    long vowels ا/آ → AI, و → U (or WQ before a vowel), ی → E (ee); consonants by place of articulation;
    unwritten short vowels are inserted between consonant clusters as /a/ (AI, reduced) or /e/ (E) — the
    mouth needs SOME opening between closures or it looks like a ventriloquist.
English: a compact letter→sound rule set (digraphs ch sh th ph oo ee ou ow …); good enough for mouth shapes.

Timing (what animators do, not phonetics):
    * each word's span is split over its visemes weighted by class (vowels 1.6, closures 0.7, others 1.0)
    * closures (MBP / FV) get ≥ 2 frames so they READ; anything < 1 frame is merged into its neighbour
    * anticipation: the mouth shape leads the sound by `lead` (default 1 frame @30) — the eye expects it
    * gaps > 0.18 s between words → rest;   0.06–0.18 s → hold a relaxed 'etc'
    * audio (optional): RMS envelope scales the jaw OPEN amount per frame (0 = silent → shut), so loud = wide

Output:  [{"t": 0.00, "v": "rest", "open": 0.0}, {"t": 0.12, "v": "MBP", "open": 0.0}, ...]   (step function)
API:  track(words, audio=None, fps=30) · at(track, t) → (viseme, open, blend_from, blend_p) · MOUTHS (shape table)
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess

import numpy as np

VISEMES = ["rest", "MBP", "etc", "E", "AI", "O", "U", "FV", "L", "WQ"]
RHUBARB = {"rest": "X", "MBP": "A", "etc": "B", "E": "C", "AI": "D", "O": "E", "U": "F", "WQ": "F", "FV": "G", "L": "H"}

# mouth geometry per viseme (normalised, mouth width = 1):  w width, h opening, round, teeth (upper/lower), tongue,
# lip_press (MBP), bite (FV)  — drawn by draw_mouth() and used by any rig
MOUTHS = {
    "rest": dict(w=0.80, h=0.03, round=0.0, teeth=0.0, tongue=0.0),
    "MBP":  dict(w=0.74, h=0.00, round=0.0, teeth=0.0, tongue=0.0, press=1.0),
    "etc":  dict(w=0.86, h=0.20, round=0.1, teeth=0.8, tongue=0.0),
    "E":    dict(w=1.00, h=0.32, round=0.0, teeth=1.0, tongue=0.25),
    "AI":   dict(w=0.94, h=0.72, round=0.2, teeth=0.6, tongue=0.55),
    "O":    dict(w=0.62, h=0.62, round=1.0, teeth=0.2, tongue=0.4),
    "U":    dict(w=0.40, h=0.30, round=1.0, teeth=0.0, tongue=0.0),
    "WQ":   dict(w=0.34, h=0.22, round=1.0, teeth=0.0, tongue=0.0),
    "FV":   dict(w=0.84, h=0.14, round=0.0, teeth=1.0, tongue=0.0, bite=1.0),
    "L":    dict(w=0.88, h=0.46, round=0.15, teeth=0.8, tongue=1.0, tongue_up=1.0),
}
WEIGHT = {"AI": 1.6, "O": 1.6, "E": 1.4, "U": 1.4, "MBP": 0.75, "FV": 0.8, "WQ": 0.9, "L": 0.9, "etc": 1.0, "rest": 1.0}
VOWELS = {"AI", "O", "E", "U"}

# ------------------------------------------------------------------------------------------- Persian G2P
FA_CONS = {
    "ب": "MBP", "پ": "MBP", "م": "MBP",
    "ف": "FV", "و_c": "WQ",
    "ل": "L", "ت": "L", "ط": "L", "د": "L", "ن": "L", "ث": "etc", "ذ": "etc",
    "ج": "etc", "چ": "etc", "ح": "etc", "خ": "etc", "ر": "etc", "ز": "etc", "ژ": "etc", "س": "etc", "ش": "etc",
    "ص": "etc", "ض": "etc", "ظ": "etc", "ع": "etc", "غ": "etc", "ق": "etc", "ک": "etc", "گ": "etc", "ه": "etc",
    "ء": "etc", "ئ": "E", "ؤ": "U", "أ": "AI", "إ": "E",
}
FA_LONG = {"ا": "AI", "آ": "AI", "ی": "E", "و": "U"}
FA_DIAC = {"\u064e": "AI", "\u0650": "E", "\u064f": "O"}     # fatha kasra damma when present


def g2v_fa(word):
    """Persian word → viseme list (short vowels inferred)."""
    w = word.replace("ي", "ی").replace("ك", "ک").replace("\u200c", "").replace("ـ", "")
    w = re.sub(r"[^\u0600-\u06FF]", "", w)
    out = []
    chars = list(w)
    n = len(chars)
    for i, ch in enumerate(chars):
        if ch in FA_DIAC:
            out.append(FA_DIAC[ch])
            continue
        if ch in ("\u0651", "\u0652", "\u064b", "\u064c", "\u064d"):
            continue
        nxt = chars[i + 1] if i + 1 < n else ""
        prv = out[-1] if out else None
        if ch == "آ":
            out.append("AI")
        elif ch == "ا":
            out.append("AI" if i > 0 else ("E" if nxt == "ی" else "AI"))   # initial alef = vowel carrier
        elif ch == "و":
            if i == 0 or (nxt in FA_LONG or nxt in FA_DIAC):
                out.append("WQ")                    # consonant v/w
            elif prv in VOWELS:
                out.append("WQ")
            else:
                out.append("U")
        elif ch == "ی":
            if i == 0 and nxt in FA_LONG:
                out.append("etc")                   # consonant y
            else:
                out.append("E")
        elif ch == "ه" and i == n - 1 and n > 1:
            out.append("E")                         # silent final he = /e/
        else:
            v = FA_CONS.get(ch, "etc")
            run = 0                                   # consonants since the last vowel
            for q in reversed(out):
                if q in VOWELS:
                    break
                run += 1
            # unwritten short vowel: Persian syllables are CV, CVC, CVCC → after a word-initial consonant, or
            # when a 3rd consonant would pile up
            if v not in VOWELS and ((run == 1 and len(out) == 1) or run >= 2):
                out.append("AI" if len(out) == 1 else "E")
            out.append(v)
    if out and out[0] not in VOWELS and len(out) == 1:
        out.append("AI")
    return out or ["etc"]


# ------------------------------------------------------------------------------------------- English G2P
EN_DIGRAPH = [("tion", ["etc", "U", "L"]), ("ough", ["O"]), ("igh", ["AI"]), ("ch", ["etc"]), ("sh", ["etc"]),
              ("th", ["L"]), ("ph", ["FV"]), ("wh", ["WQ"]), ("qu", ["WQ"]), ("ck", ["etc"]), ("ng", ["etc"]),
              ("oo", ["U"]), ("ee", ["E"]), ("ea", ["E"]), ("ou", ["O"]), ("ow", ["O"]), ("oa", ["O"]),
              ("ai", ["E"]), ("ay", ["E"]), ("au", ["O"]), ("aw", ["O"]), ("oi", ["O", "E"]), ("oy", ["O", "E"]),
              ("ie", ["AI"]), ("ue", ["U"]), ("ew", ["U"])]
EN_CHAR = {"a": "AI", "e": "E", "i": "AI", "o": "O", "u": "U", "y": "E", "b": "MBP", "m": "MBP", "p": "MBP",
           "f": "FV", "v": "FV", "w": "WQ", "l": "L", "t": "L", "d": "L", "n": "L"}


def g2v_en(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]                                   # silent final e
    out, i = [], 0
    while i < len(w):
        for dg, vs in EN_DIGRAPH:
            if w.startswith(dg, i):
                out += vs
                i += len(dg)
                break
        else:
            ch = w[i]
            if not (i > 0 and w[i - 1] == ch):       # double letters → one sound
                out.append(EN_CHAR.get(ch, "etc"))
            i += 1
    return out or ["etc"]


def g2v(word):
    return g2v_fa(word) if re.search(r"[\u0600-\u06FF]", word) else g2v_en(word)


# ------------------------------------------------------------------------------------------- audio envelope
def envelope(audio_path, fps=30, sr=16000):
    """per-frame normalised loudness (0..1) of the voice — drives jaw opening."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", audio_path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    if not len(x):
        return np.zeros(1, np.float32)
    try:
        from scipy.signal import butter, sosfilt
        x = sosfilt(butter(2, [200 / (sr / 2), 3400 / (sr / 2)], "band", output="sos"), x)   # speech band
    except Exception:  # noqa: BLE001
        pass
    hop = int(sr / fps)
    n = len(x) // hop
    e = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(e + 1e-9)
    hi, lo = np.percentile(db, 97), np.percentile(db, 30)
    v = np.clip((db - lo) / max(6.0, hi - lo), 0, 1)
    k = np.array([0.25, 0.5, 0.25])
    return np.convolve(v, k, "same").astype(np.float32)


# ------------------------------------------------------------------------------------------- track
def load_words(w):
    """words.json ({"words": [{start,end,text}]}) / list of dicts / list [t0, t1, text] → [(t0, t1, text)]."""
    if isinstance(w, str):
        with open(w, encoding="utf-8") as f:
            w = json.load(f)
    if isinstance(w, dict):
        w = w.get("words", [])
    out = []
    for it in w:
        if isinstance(it, dict):
            out.append((float(it.get("start", it.get("t"))), float(it.get("end")), str(it.get("text", it.get("w")))))
        else:
            out.append((float(it[0]), float(it[1]), str(it[2])))
    return sorted(out)


def track(words, audio=None, fps=30.0, lead=None, offset=0.0, min_frames_closure=2):
    """→ list of keys {"t", "v", "open"} (step function, sorted).  `offset` shifts all words (e.g. layer start)."""
    W = load_words(words)
    lead = (1.0 / fps) if lead is None else float(lead)
    env = envelope(audio, fps) if audio and os.path.exists(str(audio)) else None
    fr = 1.0 / fps
    keys = [{"t": 0.0, "v": "rest"}]
    prev_end = -1.0
    for (t0, t1, text) in W:
        t0, t1 = t0 + offset, t1 + offset
        gap = t0 - prev_end
        if prev_end >= 0 and gap > 0.18:
            keys.append({"t": prev_end + 0.04, "v": "rest"})
        elif prev_end >= 0 and gap > 0.06:
            keys.append({"t": prev_end, "v": "etc"})
        vs = g2v(text)
        wts = np.array([WEIGHT[v] for v in vs], np.float64)
        dur = max(fr, t1 - t0)
        ds = wts / wts.sum() * dur
        # closures must read: ≥ min_frames_closure frames (stolen from the longest vowel)
        for i, v in enumerate(vs):
            need = min_frames_closure * fr
            if v in ("MBP", "FV") and ds[i] < need and dur > need * 2:
                j = int(np.argmax(ds))
                take = min(need - ds[i], ds[j] - fr)
                if take > 0:
                    ds[i] += take
                    ds[j] -= take
        t = t0
        for v, d in zip(vs, ds):
            keys.append({"t": max(0.0, t - lead), "v": v})
            t += d
        prev_end = t1
    if W:
        keys.append({"t": prev_end + 0.05, "v": "rest"})
    # merge sub-frame keys + identical neighbours
    keys.sort(key=lambda k: k["t"])
    out = []
    for k in keys:
        if out and k["t"] - out[-1]["t"] < fr * 0.999:
            if k["v"] in ("MBP", "FV") or out[-1]["v"] in ("rest", "etc"):
                out[-1] = dict(k, t=out[-1]["t"])   # closures win (they're what the eye checks)
            continue
        if out and out[-1]["v"] == k["v"]:
            continue
        out.append(dict(k))
    for k in out:
        base = MOUTHS[k["v"]]["h"]
        if env is not None:
            i = min(len(env) - 1, max(0, int(round((k["t"] + lead) * fps))))
            loud = float(env[max(0, i - 1):i + 3].max())
            k["open"] = round(base * (0.45 + 0.75 * loud) if k["v"] not in ("MBP", "rest") else 0.0, 3)
        else:
            k["open"] = round(base, 3)
        k["t"] = round(float(k["t"]), 4)
        k["open"] = float(k["open"])
    return out


def at(keys, t, blend=0.045):
    """(viseme, open, prev_viseme, p) at time t;  p 0→1 over `blend` s after a key (shape interpolation)."""
    if not keys:
        return "rest", 0.0, "rest", 1.0
    lo, hi = 0, len(keys) - 1
    if t < keys[0]["t"]:
        return keys[0]["v"], keys[0].get("open", 0.0), keys[0]["v"], 1.0
    while lo < hi:
        m = (lo + hi + 1) // 2
        if keys[m]["t"] <= t:
            lo = m
        else:
            hi = m - 1
    k = keys[lo]
    pv = keys[lo - 1]["v"] if lo > 0 else k["v"]
    p = min(1.0, (t - k["t"]) / blend) if blend > 0 else 1.0
    return k["v"], k.get("open", MOUTHS[k["v"]]["h"]), pv, p


def shape_at(keys, t, blend=0.045):
    """interpolated mouth geometry dict at t (smooth transitions between chart mouths)."""
    v, op, pv, p = at(keys, t, blend)
    a, b = MOUTHS[pv], MOUTHS[v]
    s = p * p * (3 - 2 * p)
    g = {k: a.get(k, 0.0) + (b.get(k, 0.0) - a.get(k, 0.0)) * s for k in set(a) | set(b)}
    base = b["h"] or 1e-6
    if b["h"] > 0:
        g["h"] = a["h"] + (op - a["h"]) * s if op is not None else g["h"]
    g["v"] = v
    return g


# ------------------------------------------------------------------------------------------- drawing
def draw_mouth(ctx, g, x, y, w, ink=(0.12, 0.05, 0.06), lip=(0.80, 0.25, 0.30), inner=(0.30, 0.04, 0.08),
               teeth=(0.98, 0.97, 0.94), tongue=(0.93, 0.42, 0.46), line_w=None, style="toon"):
    """cartoon mouth from geometry g (shape_at) centred at x, y, width w (cairo)."""
    import cairo  # noqa: F401
    mw = w * g["w"]
    mh = w * 0.5 * g["h"]
    lw = line_w or max(1.5, w * 0.045)
    r = g.get("round", 0.0)
    ctx.save()
    ctx.translate(x, y)
    if mh < w * 0.02:                                   # closed: a line (pressed lips bulge a little)
        press = g.get("press", 0.0)
        ctx.move_to(-mw / 2, 0)
        ctx.curve_to(-mw / 4, w * 0.03 * (1 + press), mw / 4, w * 0.03 * (1 + press), mw / 2, 0)
        ctx.set_source_rgb(*ink)
        ctx.set_line_width(lw * (1 + 0.5 * press))
        ctx.set_line_cap(1)
        ctx.stroke()
        if press > 0.5:                                  # MBP: little lip lines above/below
            for sy in (-1, 1):
                ctx.move_to(-mw * 0.18, sy * lw * 1.6)
                ctx.line_to(mw * 0.18, sy * lw * 1.6)
            ctx.set_line_width(lw * 0.5)
            ctx.stroke()
        ctx.restore()
        return
    # outer shape: superellipse-ish between a wide smile (r=0) and an O (r=1)
    top_k = 0.35 + 0.35 * r
    ctx.move_to(-mw / 2, 0)
    ctx.curve_to(-mw / 2, -mh * top_k * 1.6, -mw * 0.25 * (1 - r * 0.3), -mh, 0, -mh * (0.85 + 0.15 * r))
    ctx.curve_to(mw * 0.25 * (1 - r * 0.3), -mh, mw / 2, -mh * top_k * 1.6, mw / 2, 0)
    ctx.curve_to(mw / 2, mh * 1.5, -mw / 2, mh * 1.5, -mw / 2, 0)
    ctx.close_path()
    path = ctx.copy_path()
    ctx.set_source_rgb(*inner)
    ctx.fill_preserve()
    ctx.save()
    ctx.clip()
    if g.get("teeth", 0) > 0.05:                          # upper teeth band
        tb = mh * (0.42 + 0.33 * g["teeth"])
        ctx.rectangle(-mw / 2, -mh * 1.2, mw, tb)
        ctx.set_source_rgb(*teeth)
        ctx.fill()
        if g.get("bite", 0) > 0.5:                         # FV: lower lip tucked under the teeth
            ctx.rectangle(-mw / 2, -mh * 1.2 + tb, mw, mh * 3)
            ctx.set_source_rgb(*lip)
            ctx.fill()
    if g.get("tongue", 0) > 0.05:
        ty = mh * (0.75 - 0.9 * g.get("tongue_up", 0.0))
        ctx.arc(0, ty + mh * 0.62, min(mw * 0.3, mh * 0.9), math.pi, 2 * math.pi)
        ctx.set_source_rgb(*tongue)
        ctx.fill()
    ctx.restore()
    ctx.new_path()
    ctx.append_path(path)
    ctx.set_source_rgb(*ink)
    ctx.set_line_width(lw)
    ctx.set_line_join(1)
    ctx.stroke()
    ctx.restore()


def chart_image(out=None, cell=220):
    """the Preston Blair chart as rendered by draw_mouth (for LOOK / docs)."""
    import cairo
    import cv2
    cols = 5
    rows = (len(VISEMES) + cols - 1) // cols
    s = cairo.ImageSurface(cairo.FORMAT_RGB24, cell * cols, cell * rows)
    c = cairo.Context(s)
    c.set_source_rgb(0.97, 0.94, 0.86)
    c.paint()
    for i, v in enumerate(VISEMES):
        x, y = (i % cols) * cell + cell / 2, (i // cols) * cell + cell / 2 - 10
        g = dict(MOUTHS[v], v=v)
        draw_mouth(c, g, x, y, cell * 0.6)
        c.set_source_rgb(0.2, 0.2, 0.2)
        c.select_font_face("Sans")
        c.set_font_size(18)
        c.move_to(x - cell * 0.4, (i // cols) * cell + cell - 18)
        c.show_text(f"{v}  ({RHUBARB[v]})")
    s.flush()
    img = np.ndarray((s.get_height(), s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :s.get_width(), :3].copy()
    if out:
        cv2.imwrite(out, img)
    return img
