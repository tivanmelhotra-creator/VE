"""
mograph.vad — v22 VOICE ACTIVITY DETECTION (Silero VAD v5, 2.3 MB ONNX, MIT; energy fallback offline).

Why (lesson from Concat's own audit, finding H8/N3): without VAD, a transcriber glues the start of a new segment to
the END of the previous one across a silence, so subtitles / "w:word" anchors appear 1–7 s before the speech.
VE fixes this at the source: every word timing that comes from a transcriber or a TTS engine is SNAPPED to real
speech onsets.

    speech_regions(pcm, sr)           → [(t0, t1), …] seconds of speech
    snap_words(words, regions)        → words whose start lies in silence are moved to the next speech onset,
                                        ends clamped to the speech region (in-place safe copy)
    words_in_regions(tokens, regions) → evenly-weighted word timings distributed over the speech only
                                        (used by the offline TTS provider, which returns no timestamps)

CLI:  python3 -m mograph.vad audio.wav|video.mp4 [--words words.json --out fixed.json]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

SR16 = 16000
CHUNK, CTX = 512, 64           # Silero v5 @16 kHz: 512-sample window + 64-sample context


def load_audio(path, sr=SR16):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(sr), "pipe:1"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, "<i2").astype(np.float32) / 32768.0


def _resample(x, sr_in, sr_out=SR16):
    if sr_in == sr_out:
        return x.astype(np.float32)
    n = int(len(x) * sr_out / sr_in)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def probabilities(pcm, sr=SR16):
    """speech probability per 32 ms window (Silero) or None if the model is unavailable."""
    from .models_hub import session
    s = session("silero_vad", threads=1)
    if s is None:
        return None
    x = _resample(pcm, sr)
    state = np.zeros((2, 1, 128), np.float32)
    ctx = np.zeros((1, CTX), np.float32)
    srt = np.array(SR16, np.int64)
    out = []
    for i in range(0, len(x), CHUNK):
        ch = x[i:i + CHUNK]
        if len(ch) < CHUNK:
            ch = np.pad(ch, (0, CHUNK - len(ch)))
        inp = np.concatenate([ctx, ch[None]], axis=1).astype(np.float32)
        p, state = s.run(None, {"input": inp, "state": state, "sr": srt})
        ctx = inp[:, -CTX:]
        out.append(float(p[0, 0]))
    return np.array(out, np.float32)


def _energy_prob(pcm, sr):
    x = _resample(pcm, sr)
    n = len(x) // CHUNK
    if n == 0:
        return np.zeros(0, np.float32)
    e = 20 * np.log10(np.sqrt((x[:n * CHUNK].reshape(n, CHUNK) ** 2).mean(1)) + 1e-6)
    floor = np.percentile(e, 10)
    return np.clip((e - floor - 9) / 9, 0, 1).astype(np.float32)


def speech_regions(pcm, sr=SR16, thresh=0.5, min_speech=0.12, min_silence=0.18, pad=0.03):
    """→ list of (t0, t1) seconds.  Hysteresis (thresh / thresh-0.15), short gaps merged, short blips dropped."""
    pr = probabilities(pcm, sr)
    if pr is None:
        pr = _energy_prob(pcm, sr)
    step = CHUNK / SR16
    regs, on, t0 = [], False, 0.0
    for i, p in enumerate(pr):
        t = i * step
        if not on and p >= thresh:
            on, t0 = True, t
        elif on and p < thresh - 0.15:
            on = False
            regs.append([t0, t])
    if on:
        regs.append([t0, len(pr) * step])
    merged = []
    for a, b in regs:
        if merged and a - merged[-1][1] < min_silence:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    dur = len(pcm) / sr
    return [(max(0.0, a - pad), min(dur, b + pad)) for a, b in merged if b - a >= min_speech]


def snap_words(words, regions, max_shift=8.0):
    """move word starts that fall in silence to the next speech onset (≤ max_shift s); clamp ends to speech."""
    if not regions:
        return [dict(w) for w in words]
    out = []
    for w in words:
        w = dict(w)
        if "t" not in w and "start" in w:            # words.json {"start","end","text"} format
            w["t"] = w["start"]
        t, e = float(w["t"]), float(w.get("end", w["t"] + 0.2))
        inside = next((r for r in regions if r[0] - 0.02 <= t <= r[1]), None)
        if inside is None:
            nxt = next((r for r in regions if r[0] > t), None)
            if nxt is not None and nxt[0] - t <= max_shift:
                d = e - t
                t = nxt[0]
                e = min(max(e, t + min(d, 0.6)), nxt[1])
                inside = nxt
        if inside is not None:
            e = min(max(e, t + 0.05), inside[1])
        w["t"], w["end"] = round(t, 3), round(max(e, t + 0.05), 3)
        if "start" in w:
            w["start"] = w["t"]
        out.append(w)
    for i in range(1, len(out)):                     # keep order monotonic
        if out[i]["t"] < out[i - 1]["t"]:
            out[i]["t"] = out[i - 1]["t"]
            out[i]["end"] = max(out[i]["end"], out[i]["t"] + 0.05)
    return out


def words_in_regions(tokens, regions, weight=None):
    """distribute tokens over the speech regions only (by letter weight) → [{w, t, end}]."""
    if not tokens:
        return []
    if not regions:
        return []
    wf = weight or (lambda s: max(1, len(s.replace("\u200c", ""))) + 1.5)
    W = [wf(t) for t in tokens]
    total_speech = sum(b - a for a, b in regions)
    # assign tokens to regions proportionally to region length
    out, k, acc_w = [], 0, 0.0
    totw = sum(W)
    bounds, c = [], 0.0
    for a, b in regions:
        c += (b - a) / total_speech
        bounds.append(c * totw)
    ri = 0
    for tok, w in zip(tokens, W):
        mid = acc_w + w / 2
        while ri < len(regions) - 1 and mid > bounds[ri]:
            ri += 1
        out.append((ri, tok, w))
        acc_w += w
    res = []
    for r, (a, b) in enumerate(regions):
        grp = [(t, w) for rr, t, w in out if rr == r]
        if not grp:
            continue
        gw = sum(w for _, w in grp)
        t = a
        for tok, w in grp:
            d = (b - a) * w / gw
            res.append({"w": tok, "t": round(t, 3), "end": round(t + d * 0.94, 3)})
            t += d
        k += len(grp)
    return res


def selftest():
    """synthetic: 0.6 s silence · 1.0 s 'speech' (harmonic + noise, syllabic AM) · 1.2 s silence · 0.8 s speech."""
    sr = SR16
    rng = np.random.default_rng(1)

    def voice(d):
        t = np.arange(int(d * sr)) / sr
        f0 = 140 + 20 * np.sin(2 * np.pi * 3 * t)
        s = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
        return (0.25 * s * (0.55 + 0.45 * np.sin(2 * np.pi * 4.5 * t)) + 0.02 * rng.standard_normal(len(t)))

    sil = lambda d: 0.002 * rng.standard_normal(int(d * sr))  # noqa: E731
    pcm = np.concatenate([sil(0.6), voice(1.0), sil(1.2), voice(0.8), sil(0.4)]).astype(np.float32)
    regs = speech_regions(pcm, sr)
    from .models_hub import have
    eng = "silero" if have("silero_vad") else "energy"
    ok_r = len(regs) == 2 and abs(regs[0][0] - 0.6) < 0.2 and abs(regs[1][0] - 2.8) < 0.3
    # the Concat bug: a transcriber puts word 3 at the end of word 2 (inside the silence) → snapped to 2.8 s
    ws = [{"w": "a", "t": 0.62, "end": 1.0}, {"w": "b", "t": 1.0, "end": 1.55}, {"w": "c", "t": 1.9, "end": 3.2}]
    sn = snap_words(ws, regs)
    ok_s = abs(sn[2]["t"] - regs[1][0]) < 1e-6 and sn[0]["t"] == 0.62
    wr = words_in_regions(["سلام", "دنیا", "خوب"], regs)
    ok_w = len(wr) == 3 and all(any(a - 1e-6 <= w["t"] <= b for a, b in regs) for w in wr)
    return ok_r and ok_s and ok_w, (f"{eng}: regions {[(round(a, 2), round(b, 2)) for a, b in regs]}, early word "
                                    f"1.90 → {sn[2]['t']:.2f} s (snapped to the onset), TTS words only inside speech {ok_w}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] == "selftest":
        ok, msg = selftest()
        print(("PASS " if ok else "FAIL ") + msg)
        return 0 if ok else 1
    pcm = load_audio(argv[0])
    regs = speech_regions(pcm)
    for a, b in regs:
        print(f"  speech {a:7.2f} → {b:7.2f}")
    if "--words" in argv:
        wp = argv[argv.index("--words") + 1]
        d = json.load(open(wp, encoding="utf-8"))
        ws = d["words"] if isinstance(d, dict) else d
        fixed = snap_words(ws, regs)
        moved = sum(1 for a, b in zip(ws, fixed) if abs(float(a.get("t", a.get("start"))) - b["t"]) > 0.05)
        out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.splitext(wp)[0] + ".vad.json"
        json.dump({"words": fixed} if isinstance(d, dict) else fixed, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"{moved} word(s) moved onto speech onsets → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
