"""
mograph.audio — soundtrack builder.

* SFX are AUTOMATIC: every element's enter time triggers its type's default
  sound (icon→draw, stamp→thud, callout→tick, grid→ticks, counter→roll…).
  Override per element with "sfx": "stamp" | "none" | ...   (no manual cue lists).
* Procedural dark-ambient bed (drone + sparse low piano on chapter changes)
  OR a music file  ("audio": {"music": "bed.mp3"}).
* Voice-over ("audio": {"vo": "narration.wav"}) is mixed on top and music/SFX
  are side-chain DUCKED under the voice automatically.
* Loudness normalised to -14 LUFS (Instagram / YouTube target).

spec["audio"] = {"vo": "vo.wav", "music": null, "ambient": true, "sfx": true,
                 "music_gain": 1.0, "sfx_gain": 0.8, "duck": 0.6, "lufs": -14}
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import wave

import numpy as np

from .elements import REGISTRY

SR = 44100


def _env(n, attack=0.004, decay=8.0):
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-decay * t)


def _noise(n, seed):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def _band(x, k):
    """cheap band-pass: difference of two box blurs"""
    k1, k2 = max(1, k), max(2, k * 4)
    return np.convolve(x, np.ones(k1) / k1, "same") - np.convolve(x, np.ones(k2) / k2, "same")


def _tl(d):
    return np.arange(int(d * SR)) / SR


def _ticks(n_ticks, total, f, gain, curve, seed=0, decay=120):
    n = int(total * SR)
    s = np.zeros(n, np.float32)
    rng = np.random.default_rng(seed)
    m = int(0.03 * SR)
    for k in range(n_ticks):
        i0 = int(curve(k / n_ticks) * (total - 0.05) * SR)
        ff = f + rng.uniform(-250, 450)
        seg = np.sin(2 * np.pi * ff * np.arange(m) / SR) * _env(m, 0.0005, decay) * gain
        e = min(m, n - i0)
        if e > 0:
            s[i0:i0 + e] += seg[:e]
    return s


_SFX_CACHE: dict = {}


def sfx(name, seed=0):
    key = (name, seed % 4)
    if key in _SFX_CACHE:
        return _SFX_CACHE[key]
    if name == "tick":
        s = np.sin(2 * np.pi * 2400 * _tl(0.05)) * _env(int(0.05 * SR), 0.001, 90) * 0.35
    elif name == "draw":
        n = int(0.5 * SR)
        s = _band(_noise(n, seed), 3) * np.sin(np.pi * np.linspace(0, 1, n)) ** 2 * 0.22
    elif name == "whoosh":
        n = int(0.45 * SR)
        s = _band(_noise(n, seed + 7), 6) * np.sin(np.pi * np.linspace(0, 1, n) ** 0.7) ** 3 * 0.4
    elif name == "stamp":
        n = int(0.6 * SR); tt = _tl(0.6)
        s = (np.sin(2 * np.pi * (70 - 30 * tt) * tt) * _env(n, 0.002, 9) * 0.9
             + _band(_noise(n, seed), 2) * _env(n, 0.0005, 60) * 0.6)
    elif name == "pop":
        n = int(0.18 * SR); tt = _tl(0.18)
        s = np.sin(2 * np.pi * (520 + 900 * np.exp(-tt * 40)) * tt) * _env(n, 0.001, 22) * 0.35
    elif name == "grid":
        s = _ticks(14, 0.9, 1800, 0.16, lambda p: p ** 1.4, seed)
    elif name == "counter":
        s = _ticks(18, 1.2, 3000, 0.14, lambda p: 1 - (1 - p) ** 2, seed, 160)
    elif name == "grow":
        tt = _tl(0.9)
        s = np.sin(2 * np.pi * (180 + 260 * tt) * tt) * np.sin(np.pi * tt / 0.9) ** 2 * 0.12
    elif name == "strike":
        n = int(0.35 * SR)
        s = _band(_noise(n, seed + 3), 2) * np.linspace(1, 0, n) ** 2 * 0.3
    elif name == "lock":
        n = int(0.4 * SR)
        s = np.zeros(n, np.float32)
        m = int(0.12 * SR)
        a = np.arange(m) / SR
        for d0, f in ((0.0, 1300), (0.09, 900)):
            i0 = int(d0 * SR)
            seg = (np.sin(2 * np.pi * f * a) + 0.5 * np.sin(2 * np.pi * f * 2.7 * a)) * _env(m, 0.0005, 45) * 0.3
            s[i0:i0 + m] += seg[: n - i0]
    elif name == "impact":
        n = int(1.4 * SR); tt = _tl(1.4)
        s = (np.sin(2 * np.pi * 45 * tt) * _env(n, 0.003, 3.5) * 0.9
             + _band(_noise(n, seed), 8) * _env(n, 0.001, 12) * 0.3)
    else:
        s = np.zeros(1, np.float32)
    s = np.asarray(s, np.float32)
    _SFX_CACHE[key] = s
    return s


SFX_NAMES = ["tick", "draw", "whoosh", "stamp", "pop", "grid", "counter", "grow", "strike", "lock", "impact"]


def ambient_bed(duration, chapter_times, seed=1, key_hz=65.41):
    n = int(duration * SR)
    tt = np.arange(n, dtype=np.float32) / SR
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.06 * tt)
    bed = (0.10 * np.sin(2 * np.pi * key_hz * tt)
           + 0.06 * np.sin(2 * np.pi * key_hz * 1.5 * tt + 0.4 * np.sin(2 * np.pi * 0.11 * tt))
           + 0.035 * lfo * np.sin(2 * np.pi * key_hz * 2.38 * tt)
           + 0.03 * (1 - lfo) * np.sin(2 * np.pi * key_hz * 2.245 * tt))
    bed = bed + np.convolve(_noise(n, seed), np.ones(64) / 64, "same") * 0.012
    notes = [key_hz * 2 * r for r in (1.0, 1.189, 1.335, 1.122, 0.891, 1.498)]
    for i, ct in enumerate(chapter_times):
        i0 = int(ct * SR)
        m = min(int(3.0 * SR), n - i0)
        if m <= 0:
            continue
        f = notes[i % len(notes)]
        lt = np.arange(m) / SR
        tone = np.sin(2 * np.pi * f * lt) + 0.35 * np.sin(4 * np.pi * f * lt) + 0.12 * np.sin(6 * np.pi * f * lt)
        bed[i0:i0 + m] += 0.12 * tone * np.minimum(1, lt / 0.01) * np.exp(-1.6 * lt)
    fade = min(n // 2, int(2.0 * SR))
    if fade > 0:
        bed[:fade] *= np.linspace(0, 1, fade)
        bed[-fade:] *= np.linspace(1, 0, fade)
    return bed.astype(np.float32)


def spectral_carve(bed, env, cfg=None):
    """v9 spectral voice-over carve (HyperFrames /hyperframes-audio): split the bed with zero-phase Butterworth
    filters into the VOCAL PRESENCE band (300–3400 Hz) and the OUT-OF-BAND warmth (< 300 Hz sub + > 3.4 kHz air);
    while speech_env is active, the presence band drops by carve_db (−11 dB) and the rest only by keep_db (−2 dB).
    The three bands sum back to the original exactly when env = 0 (linear-phase split: low + band + high = x)."""
    from scipy.signal import butter, sosfiltfilt
    cfg = cfg or {}
    lo, hi = float(cfg.get("carve_lo", 300.0)), float(cfg.get("carve_hi", 3400.0))
    sos_lo = butter(4, lo, "lowpass", fs=SR, output="sos")
    sos_hi = butter(4, hi, "highpass", fs=SR, output="sos")
    low = sosfiltfilt(sos_lo, bed).astype(np.float32)
    high = sosfiltfilt(sos_hi, bed).astype(np.float32)
    band = (bed - low - high).astype(np.float32)          # complementary → perfect reconstruction
    g_band = 10 ** (float(cfg.get("carve_db", -11.0)) / 20)
    g_keep = 10 ** (float(cfg.get("keep_db", -2.0)) / 20)
    e = np.clip(env, 0, 1).astype(np.float32)
    return (band * (1 - (1 - g_band) * e) + (low + high) * (1 - (1 - g_keep) * e)).astype(np.float32)


def read_audio(path, duration=None):
    """Decode any audio/video file to mono float32 @ SR via ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], check=True, capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.float32).copy()
    if duration is not None:
        n = int(duration * SR)
        a = np.pad(a, (0, max(0, n - len(a))))[:n]
    return a


def collect_cues(scene):
    cues = []
    for ch in scene.chapters:
        for el in ch["elements"]:
            if el.get("_bad"):
                continue
            name = el.get("sfx", "auto")
            if name in (None, "none", False):
                continue
            if name == "auto":
                name = REGISTRY[el["type"]]["sfx"]
                if name == "none":
                    continue
            cues.append((float(el["t"]) + float(el.get("sfx_offset", 0.0)), name, float(el.get("sfx_gain", 1.0))))
            if el.get("type") == "strike":
                cues.append((float(el["t"]) + 0.55, "pop", 0.6))
            for dr in el.get("drops", []) if el.get("type") == "scale_rig" else []:
                cues.append((float(dr["t"]) + 0.55, "lock", 0.8))
            for hl in el.get("highlights", []) if el.get("type") == "quote_rig" else []:
                cues.append((float(hl["t"]), "draw", 0.7))
            for r in el.get("rows", []) if el.get("type") == "invoice" else []:
                if "t" in r:
                    cues.append((float(r["t"]), "tick", 0.9))
            if el.get("type") == "invoice" and isinstance(el.get("stamp"), dict) and "t" in el["stamp"]:
                cues.append((float(el["stamp"]["t"]), "stamp", 1.0))
            if el.get("scan_at") is not None:
                cues.append((float(el["scan_at"]), "whoosh", 0.8))
            if el.get("drill_at") is not None:
                cues.append((float(el["drill_at"]), "grow", 1.0))
                cues.append((float(el["drill_at"]) + 2.5, "pop", 0.8))
            if el.get("morph_at") is not None:
                cues.append((float(el["morph_at"]), "whoosh", 0.7))
            for tk in el.get("_chain_t", [])[1:]:
                cues.append((float(tk), "whoosh", 0.7))
            for f in (el.get("flips") or [])[1:]:
                cues.append((float(f[0]), "pop", 0.7))
    return sorted(cues)


def build_soundtrack(scene, out_wav, log=print):
    cfg = scene.spec.get("audio", {}) or {}
    base = scene.base_dir
    D = scene.duration
    n = int(D * SR)
    music = np.zeros(n, np.float32)
    if cfg.get("music") and os.path.exists(os.path.join(base, cfg["music"])):
        music = read_audio(os.path.join(base, cfg["music"]), D) * float(cfg.get("music_gain", 0.55))
    elif cfg.get("beat"):                        # v12: procedural edit-synced beat (mograph.beat)
        from .beat import beat_path
        bcfg = cfg["beat"] if isinstance(cfg["beat"], dict) else {}
        music = read_audio(beat_path(scene), D) * float(bcfg.get("gain", cfg.get("music_gain", 0.5)))
    elif cfg.get("ambient", True):
        music = ambient_bed(D, [c["t"] for c in scene.chapters]) * float(cfg.get("music_gain", 1.0))
    # v7: event-locked stereo SFX bus (mograph.sound); legacy cue engine via audio.engine="legacy"
    if cfg.get("sfx", True) and cfg.get("engine", "v7") != "legacy":
        from .sound import render_bus, collect
        fxL, fxR = render_bus(scene, n)
        cues = collect(scene)
    else:
        fxL = np.zeros(n, np.float32)
        cues = collect_cues(scene)
        if cfg.get("sfx", True):
            last = {}
            for i, (t0, name, g) in enumerate(cues):
                if t0 - last.get(name, -9) < 0.12:
                    continue
                last[name] = t0
                s_ = sfx(name, i)
                i0 = int(t0 * SR)
                if i0 >= n:
                    continue
                m = min(len(s_), n - i0)
                fxL[i0:i0 + m] += s_[:m] * g
        fxR = fxL.copy()
    sg = float(cfg.get("sfx_gain", 0.8))
    fxL *= sg; fxR *= sg
    vo = np.zeros(n, np.float32)
    if cfg.get("vo"):
        vo = read_audio(os.path.join(base, cfg["vo"]), D) * float(cfg.get("vo_gain", 1.0))
        win = int(0.05 * SR)
        rms = np.sqrt(np.convolve(vo ** 2, np.ones(win) / win, "same"))
        env = np.clip(rms / (rms.max() + 1e-9) * 6, 0, 1)
        k = int(0.25 * SR)
        env = np.convolve(env, np.ones(k) / k, "same")
        if cfg.get("carve", "spectral") == "spectral":
            music = spectral_carve(music, env, cfg)            # v9: carve the vocal pocket, keep warmth & air
        else:
            music *= 1 - float(cfg.get("duck", 0.6)) * env
        fxL *= 1 - 0.35 * env; fxR *= 1 - 0.35 * env
    try:                                       # v11: footage elements with "audio": true (time-remapped)
        from .fx import footage_audio
        clip = footage_audio(scene, n, SR)
        if np.abs(clip).max() > 1e-5:
            if not cfg.get("vo"):
                win = int(0.05 * SR)
                rms = np.sqrt(np.convolve(clip ** 2, np.ones(win) / win, "same"))
                env = np.convolve(np.clip(rms / (rms.max() + 1e-9) * 6, 0, 1), np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")
                music = spectral_carve(music, env, cfg) if cfg.get("carve", "spectral") == "spectral" else music * (1 - 0.6 * env)
            vo = vo + clip
    except Exception as e:  # noqa: BLE001
        log(f"[audio] footage audio skipped: {e}")
    mix = np.stack([music + fxL + vo, music + fxR + vo], axis=1)
    peak = float(np.abs(mix).max()) + 1e-9
    if peak > 0.98:
        mix *= 0.98 / peak
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    lufs = cfg.get("lufs", -14)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-af",
                        f"loudnorm=I={lufs}:TP=-1.5:LRA=11,alimiter=limit=0.70:attack=1:release=50:level=false", "-ar", str(SR), "-ac", "2", out_wav], check=True)
        os.remove(tmp)
    except Exception:
        os.replace(tmp, out_wav)
    log(f"[audio] {len(cues)} event-locked SFX, vo={'yes' if cfg.get('vo') else 'no'} → {out_wav}")
    return out_wav


# ---------------------------------------------------------------------------
# subtitle import helpers
# ---------------------------------------------------------------------------

def subtitles_from_srt(srt_path):
    """.srt → sentence subtitles [{"t","end","text"}] (words are auto-timed inside each block)."""
    txt = open(srt_path, encoding="utf-8-sig").read()
    out = []

    def ts(s):
        h, m, rest = s.strip().split(":")
        sec, ms = rest.replace(".", ",").split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

    for b in re.split(r"\n\s*\n", txt.strip()):
        lines = [l for l in b.splitlines() if l.strip()]
        k = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if k is None:
            continue
        a, b_ = lines[k].split("-->")
        text = " ".join(lines[k + 1:]).strip()
        if text:
            out.append({"t": round(ts(a), 3), "end": round(ts(b_), 3), "text": text})
    return out


def subtitles_from_words(words, max_words=7, max_gap=0.6):
    """Word timestamps (e.g. whisper output [{"w"|"text"|"word", "t"|"start", "end"}]) →
    word-level subtitle sentences grouped by pauses/length. Gives frame-exact karaoke."""
    norm = []
    for w in words:
        txt = (w.get("w") or w.get("text") or w.get("word") or "").strip()
        if not txt:
            continue
        t0 = float(w.get("t", w.get("start", 0)))
        norm.append({"w": txt, "t": t0, "end": float(w.get("end", t0 + 0.3)), "emph": bool(w.get("emph", False))})
    out, cur = [], []
    for w in norm:
        if cur and (len(cur) >= max_words or w["t"] - cur[-1]["end"] > max_gap
                    or cur[-1]["w"][-1:] in ".!?؟،؛"):
            out.append({"t": cur[0]["t"], "end": cur[-1]["end"] + 0.15, "words": cur})
            cur = []
        cur.append(w)
    if cur:
        out.append({"t": cur[0]["t"], "end": cur[-1]["end"] + 0.15, "words": cur})
    return out
