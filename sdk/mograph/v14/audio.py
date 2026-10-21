"""
mograph.v14.audio — soundtrack for a v14 edit, built from the SAME timing maths as the picture.

  voice   NOISE-CANCELLED (denoise.py: RNNoise neural + presence/air restore, auto strength from measured SNR),
          then cut with the EDL (12 ms raised-cosine splices), high-pass 80 Hz, gentle de-ess,
          compressor-ish level riding, normalised
  sfx     event-locked: each layer's entrance → its sound (whoosh / pop / jelly / boom / tick / scribble / hit),
          hero_hit → hit, disc flip → whoosh, glass pulse → jelly, subtitle phrase change → soft jelly
          (pitch from the capsule width change), camera shakes → low thump.  Stereo pan by layer x.
  music   procedural bed from mograph/beat.py (lofi / trap_pop / cinematic), ducked under the voice
  master  sum → soft limiter → integrated loudness to audio.lufs (default −14)
"""
from __future__ import annotations

import math
import os
import subprocess
import wave

import numpy as np

from ..sound import syn, SR


def _read_src(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def _hp(x, fc=80.0):
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x)
    prev_x = prev_y = 0.0
    # vectorised one-pole HP via lfilter-like recursion (scipy if available)
    try:
        from scipy.signal import lfilter
        return lfilter([(1 + a) / 2, -(1 + a) / 2], [1, -a], x).astype(np.float32)
    except Exception:  # noqa: BLE001
        for i, v in enumerate(x):
            prev_y = a * prev_y + (1 + a) / 2 * (v - prev_x)
            prev_x = v
            y[i] = prev_y
        return y


def _level(x, win=0.4, target=0.12, maxgain=3.0):
    """Slow level rider (compressor-like) → consistent voice."""
    n = int(win * SR)
    e = np.sqrt(np.convolve(x * x, np.ones(n) / n, "same") + 1e-8)
    g = np.clip(target / e, 0.4, maxgain)
    g = np.convolve(g, np.ones(n) / n, "same")
    return (x * g).astype(np.float32)


def voice_track(R, n):
    dn = R.spec.get("audio", {}).get("denoise", "auto")
    src = None
    if dn not in (None, False, "off", 0):
        try:
            from .denoise import clean_voice
            cfg = dn if isinstance(dn, dict) else {}
            stg = cfg.get("strength", dn if isinstance(dn, (int, float)) and not isinstance(dn, bool) else "auto")
            src = _read_src(clean_voice(R.src, cfg.get("model", "lq"), stg))
        except Exception as e:  # noqa: BLE001
            print(f"[audio] denoise failed ({e}) → raw voice")
    if src is None:
        src = _read_src(R.src)
    out = np.zeros(n, np.float32)
    xf = int(0.012 * SR)
    for i, (a, b) in enumerate(R.tl.seg):
        s0, s1 = int(a * SR), int(b * SR)
        seg = src[s0:s1].copy()
        if len(seg) <= 2 * xf:
            continue
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf))
        seg[:xf] *= w
        seg[-xf:] *= w[::-1]
        o = int(R.tl.cum[i] * SR)
        m = min(len(seg), n - o)
        if m > 0:
            out[o:o + m] += seg[:m]
    out = _hp(out, 85)
    if R.spec.get("audio", {}).get("dehum", False):
        out = dehum_comb(out, SR)
    out = _level(out)
    if R.spec.get("audio", {}).get("multiband", True):
        out = multiband_compress_4band(out, SR)
    vb_wet = float(R.spec.get("audio", {}).get("vocal_booth_reverb", 0.045))
    if vb_wet > 0:
        out = vocal_booth_reverb(out, SR, wet=vb_wet)
    pk = np.percentile(np.abs(out), 99.95) + 1e-9
    return out * (0.6 / pk)


SFX_OF = {"kinetic_title": "whoosh_pop", "orbit_ring": "whoosh", "disc": "whoosh", "glass_shape": "jelly",
          "halo": "boom", "frame_box": "pop", "strike": "scratch", "counter_badge": "tick", "flash": "hit",
          "object_3d": "whoosh_pop"}


def _kind_sound(kind, seed):
    if kind == "whoosh_pop":
        a = syn("whoosh", 0.35, seed)
        b = syn("pop", 0.16, seed, pitch=0.9)
        out = np.zeros(len(a) + len(b), np.float32)
        out[:len(a)] += a * 0.7
        out[int(0.24 * SR):int(0.24 * SR) + len(b)] += b
        return out, 0.24
    if kind == "boom":
        a = syn("impact", 0.7, seed) * 0.8
        return a, 0.0
    if kind == "whoosh":
        return syn("whoosh", 0.45, seed) * 0.8, 0.30
    return syn(kind, 0.3, seed), 0.0


def sfx_events(R):
    ev = []          # (t, kind, gain, pan)
    for i, L in enumerate(R.layers):
        kind = L.get("sfx", SFX_OF.get(L["type"]))
        if L["type"] == "progress" or kind in (None, "none"):
            continue
        x = float(L.get("x", R.W / 2))
        pan = (x / R.W - 0.5) * 0.6
        ev.append((L["_t0"], kind, float(L.get("sfx_gain", 1.0)), pan))
        if L.get("_hero_hit") is not None:
            ev.append((L["_hero_hit"], "hit", 0.8, pan))
        if L.get("_flip_at") is not None:
            ev.append((L["_flip_at"], "whoosh", 0.9, 0.0))
        if L.get("_pulse_at") is not None:
            for t in (L["_pulse_at"] if isinstance(L["_pulse_at"], list) else [L["_pulse_at"]]):
                ev.append((t, "jelly", 0.9, 0.0))
    if R.subs:
        P = R.subs["_phr"]
        for k, p in enumerate(P):
            hidden = any(a <= p["t0"] <= b for a, b in R.subs["_hide"])
            if hidden:
                continue
            dw = (p["w"] - (P[k - 1]["w"] if k else 0)) / 300.0
            ev.append((p["t0"], ("jelly", float(np.clip(1.15 - dw * 0.25, 0.8, 1.5))), 0.35, 0.0))
    for s in R.cam.shakes:
        ev.append((s[0], "thump", 0.7, 0.0))
    return ev


def sfx_track(R, n, gain=0.55):
    L = np.zeros(n, np.float32)
    Rr = np.zeros(n, np.float32)
    for j, (t, kind, g, pan) in enumerate(sfx_events(R)):
        pitch = 1.0
        if isinstance(kind, tuple):
            kind, pitch = kind
        if kind == "thump":
            tt = np.arange(int(0.3 * SR)) / SR
            s = (np.sin(2 * np.pi * (70 - 30 * tt) * tt) * np.exp(-tt * 14) * 0.5).astype(np.float32)
            lead = 0.0
        elif kind == "jelly":
            s, lead = syn("jelly", 0.22, j, pitch=pitch), 0.0
        else:
            s, lead = _kind_sound(kind, j)
        i0 = int((t - lead) * SR)
        if i0 < 0:
            s = s[-i0:]
            i0 = 0
        m = min(len(s), n - i0)
        if m <= 0:
            continue
        gl, gr = g * gain * (1 - pan), g * gain * (1 + pan)
        L[i0:i0 + m] += s[:m] * gl
        Rr[i0:i0 + m] += s[:m] * gr
    return L, Rr


def music_track(R, n, cfg):
    from ..beat import render_beat
    secs = cfg.get("sections") or [{"t": 0.0, "part": "intro"}, {"t": 1.2, "part": "verse"}]
    secs = [dict(s, t=R.tl.resolve(s["t"])) for s in secs]
    y = render_beat(cfg, R.duration + 2.5, secs, int(cfg.get("seed", 3)))
    if cfg.get("remix", True):
        from .remix import remix_music_to_duration
        y = remix_music_to_duration(y, SR, R.duration, float(cfg.get("bpm", 92.0)))
    out = np.zeros(n, np.float32)
    m = min(n, len(y))
    out[:m] = y[:m]
    return out


def build_audio(R, out_wav, start=0.0, end=None):
    A = R.spec.get("audio", {})
    end = R.duration if end is None else end
    n = int(R.duration * SR) + SR
    v = voice_track(R, n) * float(A.get("voice_gain", 1.0))
    sl, sr = sfx_track(R, n, float(A.get("sfx_gain", 0.55)))
    mus = np.zeros(n, np.float32)
    mc = A.get("music")
    if mc:
        mus = music_track(R, n, mc) * float(mc.get("gain", 0.16))
        # duck under the voice
        w = int(0.25 * SR)
        env = np.sqrt(np.convolve(v * v, np.ones(w) / w, "same"))
        duck = 1 - 0.55 * np.clip(env / (env.max() * 0.35 + 1e-9), 0, 1)
        mus *= np.convolve(duck, np.ones(w) / w, "same")
    Lc = v + sl + mus
    Rc = v + sr + mus
    st = np.stack([Lc, Rc], 1)[int(start * SR):int(end * SR)]
    pk = np.abs(st).max() + 1e-9
    st = soft_clip(st / pk * 0.96, 0.92)
    tmp = out_wav.replace(".wav", "_pre.wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(st, -1, 1) * 32767).astype(np.int16).tobytes())
    lufs = float(A.get("lufs", -14))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-af", f"loudnorm=I={lufs}:TP=-1.0:LRA=9",
                    "-ar", str(SR), out_wav])
    os.remove(tmp)
    return out_wav


# ====================================================================================================
# v15 (FilmCraft multiband.rs, reverbs.rs, restoration.rs):
# 4-Band Linkwitz-Riley Multiband Compressor, Vocal Booth IR Reverb, DeHum & Soft Clipper
# ====================================================================================================
def soft_clip(x: np.ndarray, t: float = 0.92) -> np.ndarray:
    """Ported from FilmCraft `crates/audio-dsp/src/effects/multiband.rs`: cubic soft clipper with ceiling `t`."""
    k = 0.5 * t
    a = np.abs(x)
    u = np.minimum(1.0, np.maximum(0.0, (a - k) / max(1e-6, 2.0 * (t - k))))
    y = k + (t - k) * (2.0 * u - u * u)
    return np.where(a <= k, x, np.sign(x) * y).astype(np.float32)


def dehum_comb(x: np.ndarray, sr: int = SR, f0: float = 50.0, harmonics: int = 6, q: float = 35.0) -> np.ndarray:
    """Ported from FilmCraft `crates/audio-dsp/src/effects/restoration.rs` (`DeHum`): harmonic notch comb."""
    from scipy.signal import iirnotch, lfilter
    out = x.copy()
    for h in range(1, harmonics + 1):
        freq = f0 * h
        if freq >= sr * 0.48:
            break
        b, a = iirnotch(freq, q * h, sr)
        out = lfilter(b, a, out).astype(np.float32)
    return out


def multiband_compress_4band(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Ported from FilmCraft `crates/audio-dsp/src/effects/multiband.rs`:
    4-band Linkwitz-Riley (4th-order Butterworth squared) crossover at 160 Hz, 800 Hz, 4500 Hz
    with independent soft-knee envelope compression per band for broadcast vocal warmth & presence.
    """
    from scipy.signal import butter, sosfilt
    nyq = 0.5 * sr
    # Band 1 (<160 Hz), Band 2 (160..800 Hz), Band 3 (800..4500 Hz), Band 4 (>4500 Hz)
    b1 = sosfilt(butter(2, 160.0 / nyq, btype="low", output="sos"), x)
    b2 = sosfilt(butter(2, [160.0 / nyq, 800.0 / nyq], btype="bandpass", output="sos"), x)
    b3 = sosfilt(butter(2, [800.0 / nyq, 4500.0 / nyq], btype="bandpass", output="sos"), x)
    b4 = sosfilt(butter(2, 4500.0 / nyq, btype="high", output="sos"), x)

    bands_cfg = [
        (b1, 0.045, 2.8, 1.08),  # Low warmth
        (b2, 0.055, 2.4, 0.96),  # Low-mid mud control
        (b3, 0.050, 3.2, 1.18),  # Vocal presence & articulation
        (b4, 0.025, 3.0, 1.12),  # High air & de-ess ceiling
    ]
    win = max(16, int(0.018 * sr))
    k_win = np.ones(win, dtype=np.float32) / win
    out = np.zeros_like(x, dtype=np.float32)
    for b_sig, thresh, ratio, makeup in bands_cfg:
        env = np.sqrt(np.convolve(b_sig * b_sig, k_win, mode="same") + 1e-9)
        over = np.maximum(1.0, env / thresh)
        gain = (over ** (1.0 / ratio - 1.0)) * makeup
        out += (b_sig * gain).astype(np.float32)
    return out


def vocal_booth_reverb(x: np.ndarray, sr: int = SR, wet: float = 0.055) -> np.ndarray:
    """
    Ported from FilmCraft `crates/audio-dsp/src/effects/reverbs.rs` (`IrSpec` "Vocal Booth"):
    rt60 = 0.30s, er_ms = 15.0ms, er_count = 8, hf_decay = 8.0, onset_ms = 4.0ms.
    Adds subtle acoustic studio early reflections so denoised voice never sounds overly dry.
    """
    if wet <= 0.002:
        return x
    from scipy.signal import fftconvolve, butter, lfilter
    rt60 = 0.30
    n_ir = int(rt60 * sr)
    t_ir = np.arange(n_ir, dtype=np.float32) / sr
    rng = np.random.default_rng(16)
    ir = np.zeros(n_ir, dtype=np.float32)
    # 8 Early reflections in 4..19ms
    for i in range(8):
        delay_s = 0.004 + (i / 8.0) * 0.015 + float(rng.uniform(0.0, 0.0015))
        idx = min(n_ir - 1, int(delay_s * sr))
        ir[idx] += ((-1.0) ** i) * (0.45 * math.exp(-delay_s * 18.0))
    # Diffuse late tail with HF decay
    tail = rng.standard_normal(n_ir).astype(np.float32) * np.exp(-6.9078 * t_ir / rt60) * 0.08
    b_lp, a_lp = butter(1, 3800.0 / (0.5 * sr), btype="low")
    tail = lfilter(b_lp, a_lp, tail).astype(np.float32)
    ir += tail
    rev = fftconvolve(x, ir, mode="full")[:len(x)].astype(np.float32)
    return ((1.0 - wet) * x + wet * rev).astype(np.float32)
