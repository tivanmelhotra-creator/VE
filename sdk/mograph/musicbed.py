"""
mograph.musicbed — v22 MUSIC-BED DETECTOR: does the source already carry background music?

Why: auto-edit jump-cuts every pause > 0.35 s.  If the clip already has a piano / pad under the voice, each cut
makes the melody jump.  This looks only at the GAPS between words (no speech there), so any steady tonal
energy is music, not voice:

  * level        gap RMS relative to the speech RMS (room tone sits ≈ -45…-60 dB under speech, a bed -15…-30 dB)
  * tonality     1 - spectral flatness in 80 Hz…5 kHz (noise / room tone is flat, notes have peaks)
  * persistence  share of gap frames that are tonal AND loud

    detect(audio_path, words)  → {"music": bool, "confidence": 0..1, "gap_db": …, "tonality": …, "frames": n}
    python3 -m mograph.musicbed media/raw.mp4 words.json

Used by v14.director.auto_edit: music → no breath cuts (segments = the whole clip, `"music_bed": true` in meta),
so the edit keeps the original music continuous.  Override: `--cuts force|off|auto`.
"""
from __future__ import annotations

import json
import sys

import numpy as np

SR = 16000
FRAME, HOP = 1024, 512


def _frames(x):
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(FRAME)[None, :].astype(np.float32)


def _gap_mask(n_frames, words, pad=0.12):
    """True for frames that lie in a pause between words (+ pad away from each word)."""
    t = (np.arange(n_frames) * HOP + FRAME / 2) / SR
    m = np.ones(n_frames, bool)
    for w in words:
        a, b = float(w.get("t", w.get("start", 0))) - pad, float(w.get("end", 0)) + pad
        m &= ~((t >= a) & (t <= b))
    return m


def analyse(pcm, words, sr=SR):
    if sr != SR:
        pcm = np.interp(np.linspace(0, len(pcm) - 1, int(len(pcm) * SR / sr)), np.arange(len(pcm)), pcm)
    pcm = pcm.astype(np.float32)
    F = _frames(pcm)
    if len(F) < 8:
        return {"music": False, "confidence": 0.0, "frames": 0, "reason": "too short"}
    spec = np.abs(np.fft.rfft(F, axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    band = (freqs >= 80) & (freqs <= 5000)
    S = spec[:, band]
    rms = np.sqrt((F ** 2).mean(1)) + 1e-9
    flat = np.exp(np.log(S).mean(1)) / S.mean(1)                 # spectral flatness 0 (tonal) … 1 (noise)
    gap = _gap_mask(len(F), words)
    speech = ~gap
    if gap.sum() < 10 or speech.sum() < 10:
        return {"music": False, "confidence": 0.0, "frames": int(gap.sum()), "reason": "not enough pauses"}
    sp_db = 20 * np.log10(np.percentile(rms[speech], 75))
    g_db = 20 * np.log10(rms[gap]) - sp_db                        # gap level relative to speech
    tonal = 1.0 - flat[gap]
    loud = g_db > -34.0
    persist = float(np.mean(loud & (tonal > 0.45)))
    med_db = float(np.median(g_db))
    med_ton = float(np.median(tonal))
    conf = float(np.clip((persist - 0.2) / 0.35, 0, 1) * np.clip((med_db + 42) / 14, 0, 1) *
                 np.clip((med_ton - 0.36) / 0.08, 0, 1))
    return {"music": conf >= 0.5, "confidence": round(conf, 3), "gap_db": round(med_db, 1),
            "tonality": round(med_ton, 3), "persistence": round(persist, 3), "frames": int(gap.sum())}


def detect(audio_path, words):
    from .vad import load_audio
    return analyse(load_audio(audio_path, SR), words, SR)


def _load_words(p):
    d = json.load(open(p, encoding="utf-8"))
    ws = d["words"] if isinstance(d, dict) else d
    return [{"t": float(w.get("t", w.get("start"))), "end": float(w["end"])} for w in ws]


def selftest():
    """synthetic voice bursts with pauses: (a) dry room tone → no music, (b) + soft piano chord bed → music."""
    rng = np.random.default_rng(3)
    dur = 12.0
    t = np.arange(int(dur * SR)) / SR
    voice = np.zeros_like(t, dtype=np.float32)
    words = []
    for k in range(10):                      # 0.7 s "words" with 0.5 s pauses
        a = 0.3 + k * 1.2
        b = a + 0.7
        m = (t >= a) & (t < b)
        f0 = 150 + 30 * np.sin(2 * np.pi * 3 * t[m])
        voice[m] = 0.3 * sum(np.sin(2 * np.pi * f0 * h * t[m]) / h for h in range(1, 6)) * \
            (0.6 + 0.4 * np.sin(2 * np.pi * 5 * t[m])) + 0.02 * rng.standard_normal(m.sum())
        words.append({"t": a, "end": b})
    room = 0.002 * rng.standard_normal(len(t)).astype(np.float32)
    env = 0.5 + 0.5 * np.cos(2 * np.pi * (t % 2.0) / 2.0 * 0.5)  # slow note decays
    piano = 0.035 * env * sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0, 523.3))
    a = analyse(voice + room, words)
    b = analyse(voice + room + piano.astype(np.float32), words)
    ok = (not a["music"]) and b["music"]
    return ok, f"dry voice → music={a['music']} (conf {a['confidence']}), with piano bed → music={b['music']} " \
               f"(conf {b['confidence']}, gap {b['gap_db']} dB, tonality {b['tonality']})"


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "selftest":
        ok, msg = selftest()
        print(("PASS " if ok else "FAIL ") + msg)
        sys.exit(0 if ok else 1)
    print(json.dumps(detect(sys.argv[1], _load_words(sys.argv[2])), indent=1))
