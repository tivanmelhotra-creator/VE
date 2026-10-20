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
    out = _level(out)
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
    y = render_beat(cfg, R.duration, secs, int(cfg.get("seed", 3)))
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
    st = np.tanh(st / pk * 1.3) / np.tanh(1.3) * 0.89
    tmp = out_wav.replace(".wav", "_pre.wav")
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(st, -1, 1) * 32767).astype(np.int16).tobytes())
    lufs = float(A.get("lufs", -14))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-af", f"loudnorm=I={lufs}:TP=-1.0:LRA=9",
                    "-ar", str(SR), out_wav])
    os.remove(tmp)
    return out_wav
