"""
mograph.edit — v9 surgical voice & video disfluency editor.

Real recordings have dead pauses, slips of the tongue (تپق) and repeated fillers («مثلاً... مثلاً...»).
Everything here works on (pcm float32 @SR, words [{w,t,end}]) and returns an EditMap so that every
downstream time (chapters, camera keys, numeric element times, omni windows, talking-head video) moves
through the same remap(t).  Word anchors ("w:…") need nothing: they re-resolve against the new words.

spec["voice"] keys (all optional):
  "max_pause": 0.22        clamp every silence longer than this down to it (mid-sentence AND between lines)
  "min_pause": 0.06        never squeeze a gap below this
  "cut_ranges": [[t0,t1]]  excise ranges (seconds of the SOURCE clip / line audio; per clip: clips[i].cut_ranges)
  "dedupe_fillers": true   remove consecutive duplicate words and filler tokens (also CLI --dedupe-fillers)
  "fillers": ["اِم","اوم","آم","ام","مثلاً","یعنی","um","uh"]   (only removed when repeated / standalone fillers)
  "xfade_ms": 12           raised-cosine de-click crossfade at every splice (zero-crossing snapped)

Video jump-cut polisher (mograph.media / talking_head):  every splice time is exported as scene.cuts → a talking-
head element applies an editorial punch-in (1.0 → 1.12 zoom alternating per cut) or the motion scene covers it.
"""
from __future__ import annotations

import numpy as np

SR = 44100
DEFAULT_FILLERS = ["اِم", "اوم", "آم", "ام", "اممم", "مثلاً", "مثلا", "یعنی", "خب", "um", "uh", "erm", "like"]


class EditMap:
    """Piecewise map source-time → edited-time built from removed intervals (sorted, non-overlapping)."""

    def __init__(self, removed):
        self.removed = sorted([(float(a), float(b)) for a, b in removed if b > a])

    def __call__(self, t):
        shift = 0.0
        for a, b in self.removed:
            if t >= b:
                shift += b - a
            elif t > a:                    # inside a removed span → collapses onto the splice point
                shift += t - a
                break
            else:
                break
        return t - shift

    def splices(self):
        """Edited-time positions of every splice (jump cuts)."""
        out, shift = [], 0.0
        for a, b in self.removed:
            out.append(round(a - shift, 4))
            shift += b - a
        return out

    @property
    def total_removed(self):
        return sum(b - a for a, b in self.removed)


def _zero_cross(x, i, win=220):
    """Nearest zero-crossing to sample i (within ±win samples)."""
    a, b = max(1, i - win), min(len(x) - 1, i + win)
    seg = x[a:b]
    if len(seg) < 2:
        return i
    zc = np.nonzero(np.signbit(seg[:-1]) != np.signbit(seg[1:]))[0]
    if not len(zc):
        return i
    return int(a + zc[np.argmin(np.abs(zc + a - i))])


def splice(pcm, removed, xfade_ms=12.0, sr=SR):
    """Remove [t0,t1] intervals from pcm with raised-cosine crossfades at zero-crossings (de-click)."""
    if not removed:
        return pcm
    f = max(2, int(xfade_ms / 1000 * sr))
    keep = []
    cur = 0
    for a, b in sorted(removed):
        ia, ib = _zero_cross(pcm, int(a * sr)), _zero_cross(pcm, int(b * sr))
        if ib <= ia:
            continue
        keep.append((cur, ia))
        cur = ib
    keep.append((cur, len(pcm)))
    out = pcm[keep[0][0]:keep[0][1]].astype(np.float32).copy()
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f, dtype=np.float32))     # raised cosine 0→1
    for s0, s1 in keep[1:]:
        nxt = pcm[s0:s1].astype(np.float32)
        k = min(f, len(out), len(nxt))
        if k >= 2:
            ww = w[:k] if k == f else (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k, dtype=np.float32)))
            mixed = out[-k:] * (1 - ww) + nxt[:k] * ww
            out = np.concatenate([out[:-k], mixed, nxt[k:]])
        else:
            out = np.concatenate([out, nxt])
    return out


def _norm(w):
    from .scene import _norm_word
    return _norm_word(w)


def plan_cuts(words, dur, cfg):
    """→ removed intervals [(t0,t1)] (source time) from max_pause / cut_ranges / dedupe_fillers."""
    removed = [tuple(map(float, r)) for r in (cfg.get("cut_ranges") or [])]
    # disfluency: consecutive duplicates (keep the LAST take — the fluent one) and standalone fillers
    if cfg.get("dedupe_fillers"):
        fillers = {_norm(f) for f in (cfg.get("fillers") or DEFAULT_FILLERS)}
        i = 0
        while i < len(words) - 1:
            a, b = words[i], words[i + 1]
            if _norm(a["w"]) and _norm(a["w"]) == _norm(b["w"]):
                removed.append((a["t"] - 0.02, b["t"] - 0.01))       # drop the first take incl. its gap
            elif _norm(a["w"]) in fillers and (i == 0 or words[i - 1]["end"] < a["t"] - 0.12) and \
                    b["t"] - a["end"] > 0.12:
                removed.append((a["t"] - 0.02, b["t"] - 0.01))       # isolated filler between pauses
            i += 1
    mp = cfg.get("max_pause")
    if mp is not None:
        mp = float(mp)
        mn = float(cfg.get("min_pause", 0.06))
        keep = max(mn, mp)
        edges = [(0.0, words[0]["t"])] if words else []
        edges += [(a["end"], b["t"]) for a, b in zip(words, words[1:])]
        for a, b in edges:
            if b - a > keep:
                pad = keep / 2
                removed.append((a + pad, b - pad))
    removed = [(max(0.0, a), min(dur, b)) for a, b in removed if b - a > 0.005]
    removed.sort()
    merged = []
    for a, b in removed:                      # merge overlaps
        if merged and a <= merged[-1][1] + 0.003:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def edit_clip(pcm, words, cfg, sr=SR):
    """Apply the voice edit to one clip. Returns (pcm_edited, words_edited, EditMap)."""
    dur = len(pcm) / sr
    removed = plan_cuts(words, dur, cfg)
    em = EditMap(removed)
    if not removed:
        return pcm, words, em
    out = splice(pcm, removed, float(cfg.get("xfade_ms", 12.0)), sr)
    nw = []
    for w in words:
        mid = (w["t"] + w["end"]) / 2
        if any(a <= mid <= b for a, b in removed):
            continue                       # excised word
        nw.append(dict(w, t=round(em(w["t"]), 3), end=round(em(w["end"]), 3)))
    return out, nw, em


def report(removed):
    if not removed:
        return "no cuts"
    return f"{len(removed)} cuts, {sum(b - a for a, b in removed):.2f}s removed: " + \
        ", ".join(f"[{a:.2f}–{b:.2f}]" for a, b in removed[:8]) + (" …" if len(removed) > 8 else "")
