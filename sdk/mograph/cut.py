"""
mograph.cut — v12 "Studio Cut": edit decision list (EDL) for a RAW talking-head take.

  "edit": {
    "src": "media/raw.mp4",          raw take (footage elements with "edit": true play the EDITED timeline)
    "words": "words.json",           word timestamps of the take  [[w, t0, t1], ...] | [{"w","t","end"}]
    "segments": [[0.45, 8.2], ...]   OR explicit source ranges; omitted → automatic silence/jump cut
    "pad_pre": 0.10, "pad_post": 0.14, "max_gap": 0.35,
    "shots": "auto" | [9.0, 15.6],   camera changes in the raw (segments never straddle one)
    "bpm": 100, "snap": 0.5, "snap_tol": 0.12,   cuts land on the music grid
    "transitions": "whip" | {"shot": "zoom_blur", "jump": "punch"},  "transition_dur": 0.3,
    "emph": ["سود", ...],            emphasised words (captions / kinetic)
    "start": 0.0, "tail": 0.6, "max_words": 4,
    "enhance": true | {"h": 1920, "denoise": 1.5, "sharpen": 0.55}   cached upscale+denoise+sharpen proxy
  }
Scene receives: subtitles in EDITED time (so "w:word" anchors work), spec._edit (segment table),
spec._cuts, spec._cutfx (transition events), meta.duration.
CLI: python3 -m mograph cut spec.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess

_END = re.compile(r"[.!?؟…]$")
_SOFT = re.compile(r"[،,;:]$")


def load_words(src, base_dir="."):
    if isinstance(src, str):
        p = src if os.path.isabs(src) else os.path.join(base_dir, src)
        with open(p, encoding="utf-8") as f:
            src = json.load(f)
        if isinstance(src, dict):
            src = src.get("words") or []
    out = []
    for w in src or []:
        if isinstance(w, (list, tuple)):
            out.append({"w": str(w[0]), "t": float(w[1]), "end": float(w[2])})
        else:
            out.append({"w": str(w.get("w") or w.get("text") or w.get("word")),
                        "t": float(w.get("t", w.get("start", 0.0))), "end": float(w.get("end", 0.0))})
    return sorted(out, key=lambda w: w["t"])


def norm(w):
    import unicodedata
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    return re.sub(r"[\s\u200c.,!?؟،:;«»\"'()\-]", "", w).replace("ي", "ی").replace("ك", "ک").lower()


def detect_shots(path, thresh=0.3):
    cache = path + ".shots.json"
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-vf", f"select='gt(scene,{thresh})',showinfo",
                        "-f", "null", "-"], capture_output=True, text=True)
    ts = sorted({round(float(m), 3) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr)})
    with open(cache, "w") as f:
        json.dump(ts, f)
    return ts


def media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def plan_segments(words, src_dur, pad_pre=0.10, pad_post=0.14, max_gap=0.35, shots=()):
    if not words:
        return [[0.0, src_dur]]
    runs, cur = [], [words[0]]
    for w in words[1:]:
        crosses = any(cur[-1]["end"] <= s <= w["t"] for s in shots)
        if w["t"] - cur[-1]["end"] > max_gap or crosses:
            runs.append(cur); cur = [w]
        else:
            cur.append(w)
    runs.append(cur)
    segs = []
    for run in runs:
        a, b = run[0]["t"] - pad_pre, run[-1]["end"] + pad_post
        lo = max([0.0] + [s + 0.04 for s in shots if s <= run[0]["t"]])
        hi = min([src_dur] + [s - 0.04 for s in shots if s >= run[-1]["end"]])
        a, b = max(a, lo), min(b, hi)
        if segs and a <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([round(a, 3), round(b, 3)])
    return segs


def snap_segments(segs, grid, tol, start=0.0, last_word=None):
    out, e = [], start
    for i, (a, b) in enumerate(segs):
        end = e + (b - a)
        d = round(end / grid) * grid - end
        if abs(d) <= tol:
            nb = b + d
            if last_word is not None:
                nb = max(nb, last_word[i] + 0.04)
            b = nb
        out.append([round(a, 4), round(b, 4)])
        e += b - a
    return out


def edit_table(segs, start=0.0):
    tab, e = [], start
    for a, b in segs:
        tab.append([float(a), float(b), round(e, 4)])
        e += b - a
    return tab, e


def to_edit_time(tab, st):
    for a, b, e0 in tab:
        if a - 1e-6 <= st <= b + 1e-6:
            return e0 + (st - a)
    return None


def to_src_time(tab, t):
    """Edited time → source time (before the first segment: its start; holds the last frame)."""
    if not tab:
        return t
    i = 0
    for k, row in enumerate(tab):
        if t >= row[2]:
            i = k
    a, b, e0 = tab[i]
    if i == len(tab) - 1:                       # last segment keeps playing into the tail (no frozen frame)
        return a + max(0.0, t - e0)
    return min(b, a + max(0.0, t - e0))


def edited_subtitles(words, tab, emph=(), max_words=4):
    """Short caption groups (reel style: 2–4 words) in edited time; sentence punctuation always breaks."""
    em = {norm(e) for e in emph}
    ws = []
    for w in words:
        t0, t1 = to_edit_time(tab, w["t"]), to_edit_time(tab, w["end"])
        if t0 is None:
            continue
        t1 = t1 if t1 is not None else t0 + (w["end"] - w["t"])
        nw = norm(w["w"])
        ws.append({"w": w["w"].strip(), "t": round(t0, 3), "end": round(t1, 3),
                   "emph": bool(nw) and any(nw == e or (len(e) > 2 and e in nw) for e in em)})
    subs, cur = [], []
    for i, w in enumerate(ws):
        cur.append(w)
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        if (nxt is None or _END.search(w["w"]) or _SOFT.search(w["w"]) or nxt["t"] - w["end"] > 0.4
                or len(cur) >= max_words):
            end = cur[-1]["end"] + 0.1
            if nxt is not None:
                end = min(end, nxt["t"] - 0.01)
            subs.append({"t": cur[0]["t"], "end": round(max(end, cur[-1]["end"]), 3),
                         "words": [dict(x, w=x["w"].rstrip(".،,")) for x in cur]})
            cur = []
    for s in subs:
        s["text"] = " ".join(w["w"] for w in s["words"])
    return subs


def plan_cutfx(cuts, kinds, cfg):
    tr = cfg.get("transitions")
    if not tr:
        return []
    d = float(cfg.get("transition_dur", 0.3))
    out = []
    for i, (t, kind) in enumerate(zip(cuts, kinds)):
        typ = tr if isinstance(tr, str) else (tr.get(kind) or tr.get("default")) if isinstance(tr, dict) else \
            next((x.get("type") for x in tr if int(x.get("at", -1)) == i + 1), None)
        extra = {}
        if isinstance(tr, list):
            extra = next((dict(x) for x in tr if int(x.get("at", -1)) == i + 1), {})
            extra.pop("at", None); extra.pop("type", None)
        if typ and typ != "none":
            ev = {"t": round(t, 4), "type": typ, "dur": d, "kind": kind}
            ev.update(extra)
            out.append(ev)
    return out


def enhance_video(path, opts=True, log=print):
    """Cached proxy: temporal denoise → lanczos upscale → contrast-adaptive sharpen (fixes 480p phone footage)."""
    o = {} if opts is True else dict(opts or {})
    h = int(o.get("h", 1920)); dn = float(o.get("denoise", 1.5)); sh = float(o.get("sharpen", 0.55))
    key = hashlib.md5(json.dumps([os.path.getsize(path), h, dn, sh]).encode()).hexdigest()[:8]
    out = f"{path}.enh_{key}.mp4"
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return out
    vf = ([f"hqdn3d={dn:.2f}:{dn:.2f}:{dn * 3:.2f}:{dn * 3:.2f}"] if dn > 0 else []) + \
         [f"scale=-2:{h}:flags=lanczos"] + ([f"cas={sh:.2f}"] if sh > 0 else [])
    log(f"[enhance] {os.path.basename(path)} → {h}p (denoise {dn}, sharpen {sh}) …")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vf", ",".join(vf), "-an", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-g", "10", out], check=True)
    return out


def _walk(els):
    for el in els or []:
        yield el
        yield from _walk(el.get("children"))


def apply_edit(spec, base_dir=".", log=print):
    ed = spec.get("edit")
    if not ed or spec.get("_edit"):
        return spec
    src = ed.get("src")
    sp = src if (src is None or os.path.isabs(src)) else os.path.join(base_dir, src)
    has = bool(sp) and os.path.exists(sp)
    dur = media_duration(sp) if has else 1e9
    words = load_words(ed["words"], base_dir) if ed.get("words") else []
    shots = ed.get("shots", "auto")
    shots = [float(s) for s in ((detect_shots(sp) if has else []) if shots == "auto" else (shots or []))]
    if ed.get("segments"):
        segs = [[float(a), float(b)] for a, b in ed["segments"]]
    else:
        segs = plan_segments(words, dur, float(ed.get("pad_pre", 0.10)), float(ed.get("pad_post", 0.14)),
                             float(ed.get("max_gap", 0.35)), shots)
    start = float(ed.get("start", 0.0))
    if ed.get("bpm"):
        grid = 60.0 / float(ed["bpm"]) * float(ed.get("snap", 0.5))
        lw = [max([w["end"] for w in words if a <= w["t"] <= b] or [a]) for a, b in segs]
        segs = snap_segments(segs, grid, float(ed.get("snap_tol", 0.12)), start, lw)
    tab, end = edit_table(segs, start)
    cuts = [row[2] for row in tab[1:]]
    kinds = ["shot" if any(b0 - 0.08 <= s <= a1 + 0.08 for s in shots) else "jump"
             for (a0, b0, _), (a1, b1, _) in zip(tab, tab[1:])]
    spec["_edit"] = {"segs": tab, "src": sp, "end": end, "cut_kinds": kinds, "shots": shots}
    spec["_cuts"] = sorted(set(list(spec.get("_cuts") or []) + cuts))
    spec["_cutfx"] = plan_cutfx(cuts, kinds, ed) + list(spec.get("cut_fx") or [])
    if words and not spec.get("subtitles"):
        spec["subtitles"] = edited_subtitles(words, tab, ed.get("emph") or [], int(ed.get("max_words", 4)))
    m = spec.setdefault("meta", {})
    if m.get("duration") is None:
        m["duration"] = round(end + float(ed.get("tail", 0.5)), 3)
    for ch in spec.get("chapters") or []:
        for el in _walk(ch.get("elements")):
            if el.get("type") == "footage" and el.get("edit"):
                el["segments"] = tab
                el.setdefault("src", src)
                if ed.get("enhance") and "enhance" not in el:
                    el["enhance"] = ed["enhance"]
    log(f"[edit] {len(tab)} segments, {len(cuts)} cuts ({', '.join(kinds) or '-'}), "
        f"{sum(b - a for a, b, _ in tab):.2f}s kept of {dur if dur < 1e8 else 0:.2f}s")
    return spec


def report(spec):
    ed = spec.get("_edit") or {}
    lines = ["#  source            → edited"]
    for i, (a, b, e0) in enumerate(ed.get("segs", [])):
        lines.append(f"{i + 1:<2} {a:6.2f} – {b:6.2f}  → {e0:6.2f} – {e0 + b - a:6.2f}  ({b - a:4.2f}s)")
    for c in spec.get("_cutfx", []):
        lines.append(f"cut {c['t']:6.2f}s  {c.get('kind', ''):5s} → {c['type']}")
    for s in spec.get("subtitles") or []:
        lines.append(f"{s['t']:6.2f}–{s['end']:6.2f}  " + " ".join(("*" + w["w"] + "*") if w.get("emph") else w["w"]
                                                              for w in s["words"]))
    return "\n".join(lines)
