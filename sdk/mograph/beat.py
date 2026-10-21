"""
mograph.beat — v12 procedural music bed (no external generator needed): trap-pop / lofi / cinematic beats that
follow the EDIT — every section restarts on a cut, risers land exactly on the drop, the ending hits on the last frame.

  "audio": {"beat": {"style": "trap_pop" | "lofi" | "cinematic", "bpm": 100, "key": "A", "scale": "minor",
                     "progression": [0, 5, 3, 4]   (scale degrees per bar → i–VI–iv–v),
                     "sections": [{"t": 0, "part": "intro"}, {"t": 2.4, "part": "verse"},
                                  {"t": 7.8, "part": "build"}, {"t": "abs:12.3", "part": "riser"},
                                  {"t": 13.5, "part": "drop"}, {"t": 16.8, "part": "outro"}],
                     "gain": 0.55, "seed": 3}}
  sections default: verse from 0, drop on the 2nd edit cut, outro at the edit end.
Parts: intro (filtered pluck + hats) · verse (kick, hats, 808, pluck) · build (+clap, +pad) · riser (snare roll,
noise sweep, kick cut) · drop (full, 16th hats, hard 808) · outro (pad + pluck tail, final hit).
Rendered once (≈0.3 s CPU for 20 s), cached next to the spec as <spec>.beat_<hash>.wav.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import wave

import numpy as np

SR = 44100
NOTES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
SCALES = {"minor": [0, 2, 3, 5, 7, 8, 10], "major": [0, 2, 4, 5, 7, 9, 11], "dorian": [0, 2, 3, 5, 7, 9, 10]}


def _hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def _rng(seed):
    return np.random.default_rng(seed)


def _env(n, a, d, sustain=0.0, curve=1.0):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4)) * (sustain + (1 - sustain) * np.exp(-t / max(d, 1e-4)) ** curve)
    return e.astype(np.float32)


def _onepole(x, fc):
    k = math.exp(-2 * math.pi * fc / SR)
    from scipy.signal import lfilter
    return lfilter([1 - k], [1, -k], x).astype(np.float32)


def _hp(x, fc):
    return (x - _onepole(x, fc)).astype(np.float32)


def kick(dur=0.45, punch=1.0):
    n = int(dur * SR); t = np.arange(n) / SR
    f = 48 + 120 * np.exp(-t * 32)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * _env(n, 0.001, 0.16 * punch) * 1.1
    s[: int(0.004 * SR)] += _rng(1).standard_normal(int(0.004 * SR)) * 0.25
    return (np.tanh(s * 1.25) * 0.9).astype(np.float32)


def clap(seed=0):
    n = int(0.35 * SR)
    no = _rng(seed).standard_normal(n).astype(np.float32)
    e = np.zeros(n, np.float32)
    for k, off in enumerate((0, 0.009, 0.018, 0.028)):
        i = int(off * SR)
        e[i:] += _env(n - i, 0.0005, 0.012 if k < 3 else 0.12)
    return _hp(_onepole(no * e, 4200), 900) * 1.4


def hat(open_=False, seed=0):
    n = int((0.22 if open_ else 0.05) * SR)
    no = _rng(seed + 11).standard_normal(n).astype(np.float32)
    x = _onepole(_onepole(_hp(no, 6000), 8500), 11000)
    return x * _env(n, 0.0005, 0.09 if open_ else 0.012) * 0.18


def snare(seed=0):
    n = int(0.22 * SR); t = np.arange(n) / SR
    body = np.sin(2 * np.pi * 190 * t) * _env(n, 0.001, 0.05)
    no = _hp(_rng(seed + 5).standard_normal(n).astype(np.float32), 1800) * _env(n, 0.001, 0.07)
    return (body * 0.5 + no * 0.7).astype(np.float32)


def bass808(f0, dur, glide_from=None):
    n = int(dur * SR); t = np.arange(n) / SR
    f = np.full(n, f0, np.float32)
    if glide_from:
        g = np.exp(-t * 18); f = f0 + (glide_from - f0) * g
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.06 * np.sin(2 * ph)
    e = _env(n, 0.004, dur * 0.7, 0.0)
    rel = min(n, int(0.04 * SR)); e[-rel:] *= np.linspace(1, 0, rel)
    return (np.tanh(s * e * 1.15) * 0.75).astype(np.float32)


def pluck(freq, dur=0.35, bright=1.0):
    n = int(dur * SR); t = np.arange(n) / SR
    s = np.zeros(n, np.float32)
    for h in range(1, 7):
        s += (np.sin(2 * np.pi * freq * h * t + h) / h) * np.exp(-t * (6 + h * 5 / bright))
    s *= _env(n, 0.002, dur * 0.4)
    return s.astype(np.float32) * 0.35


def pad(freqs, dur, cutoff=1800):
    n = int(dur * SR); t = np.arange(n) / SR
    s = np.zeros(n, np.float32)
    for f in freqs:
        for det in (-0.12, 0.0, 0.13):
            ph = 2 * np.pi * f * (1 + det / 100 * 12) * t
            s += (2 * ((ph / (2 * np.pi)) % 1) - 1) * 0.12
    s = _onepole(_onepole(s, cutoff), cutoff)
    a = min(n // 2, int(0.35 * SR))
    e = np.ones(n, np.float32); e[:a] = np.linspace(0, 1, a); e[-a:] *= np.linspace(1, 0, a)
    return s * e


def riser(dur, seed=0):
    n = int(dur * SR); t = np.arange(n) / SR
    no = _rng(seed + 21).standard_normal(n).astype(np.float32)
    out = np.zeros(n, np.float32)
    blk = 512
    for i in range(0, n, blk):
        q = i / n
        fc = 400 + 9000 * q ** 2
        seg = no[i:i + blk]
        out[i:i + blk] = _hp(seg, fc * 0.5) if len(seg) > 2 else seg
    tone = np.sin(2 * np.pi * np.cumsum(220 + 880 * (t / dur) ** 2) / SR) * 0.15
    return (out * 0.35 + tone) * (t / dur) ** 1.6


def impact(seed=0):
    n = int(1.6 * SR); t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(38 + 90 * np.exp(-t * 9)) / SR) * np.exp(-t * 2.2)
    no = _onepole(_rng(seed + 31).standard_normal(n).astype(np.float32), 2500) * np.exp(-t * 3.5) * 0.4
    return np.tanh((boom + no) * 1.1).astype(np.float32) * 0.6


def _place(buf, s, t, g=1.0):
    i = int(round(t * SR))
    if i >= len(buf) or i + len(s) <= 0:
        return
    a = max(0, -i); i = max(0, i)
    m = min(len(s) - a, len(buf) - i)
    buf[i:i + m] += s[a:a + m] * g


def _resolve(v, spec_t):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and v.startswith("abs:"):
        return float(v[4:])
    return float(v)


def default_sections(scene):
    cuts = list(getattr(scene, "cuts", None) or [])
    D = scene.duration
    end = (scene.spec.get("_edit") or {}).get("end") or D
    secs = [{"t": 0.0, "part": "verse"}]
    if len(cuts) >= 2:
        secs += [{"t": cuts[0], "part": "build"}, {"t": max(cuts[0] + 1, cuts[1] - 1.2), "part": "riser"},
                 {"t": cuts[1], "part": "drop"}]
    elif cuts:
        secs += [{"t": max(0.5, cuts[0] - 1.2), "part": "riser"}, {"t": cuts[0], "part": "drop"}]
    if end < D - 0.3:
        secs.append({"t": end, "part": "outro"})
    return secs


def render_beat(cfg, duration, sections, seed=3):
    bpm = float(cfg.get("bpm", 100))
    beat = 60.0 / bpm
    root = 45 + NOTES.get(str(cfg.get("key", "A")).upper(), 9)          # A2
    sc = SCALES.get(cfg.get("scale", "minor"), SCALES["minor"])
    prog_ = cfg.get("progression", [0, 5, 3, 4])
    n = int((duration + 2.0) * SR)
    drums = np.zeros(n, np.float32); bass = np.zeros(n, np.float32); keys = np.zeros(n, np.float32)
    fx = np.zeros(n, np.float32)
    K = kick(); Kd = kick(0.5, 1.4); C = clap(seed); S = snare(seed)
    HC = [hat(False, i) for i in range(4)]; HO = hat(True)
    secs = sorted(sections, key=lambda s: s["t"])
    bar_i = 0
    sweep = []
    kick_times = []
    swing = float(cfg.get("swing", 0.08))
    hrng = _rng(seed + 3)

    def chord(deg):
        notes = [sc[(deg + k) % 7] + 12 * ((deg + k) // 7) for k in (0, 4, 6, 9)]
        return [root + 12 + x for x in notes], root + sc[deg % 7]

    for si, sec in enumerate(secs):
        t0 = sec["t"]; t1 = secs[si + 1]["t"] if si + 1 < len(secs) else duration + 1.2
        part = sec["part"]
        nb = int(math.ceil((t1 - t0) / beat - 1e-6))
        if part == "riser":
            _place(fx, riser(t1 - t0, seed), t0, 0.8)
            steps = int((t1 - t0) / (beat / 4))
            rr = _rng(seed + 77)
            for k in range(steps):
                q = k / max(1, steps)
                div = 1 if q < 0.4 else (2 if q < 0.75 else 4)
                if k % (4 // div) == 0:
                    _place(drums, S, t0 + k * beat / 4, (0.08 + 0.42 * q ** 1.5) * rr.uniform(0.8, 1.0))
            sweep.append((t0, t1))
            continue
        if part == "drop":
            _place(fx, impact(seed), t0, 1.0)
        for b in range(nb):
            tb = t0 + b * beat
            if tb >= t1 - 1e-4:
                break
            bar = (bar_i + b // 4); pos = b % 4
            ch, bn = chord(int(prog_[bar % len(prog_)]))
            full = part in ("verse", "build", "drop")
            if part != "intro" and part != "outro":
                kp = [0, 2.5] if part == "verse" else [0, 0.75, 2.5]
                if part == "drop":
                    kp = [0, 0.75, 1.5, 2.5, 3.25]
                for kk in kp:
                    if int(kk) == pos:
                        _place(drums, Kd if part == "drop" else K, tb + (kk - pos) * beat, 0.95)
                        kick_times.append(tb + (kk - pos) * beat)
                if part in ("build", "drop") and pos in (1, 3):
                    _place(drums, C, tb, 0.7)
                # 808 on the downbeat (+ 3rd beat in drop)
                if pos == 0 or (part == "drop" and pos == 2):
                    L = beat * (2 if pos == 0 else 1.5)
                    _place(bass, bass808(_hz(bn - 12), min(L, t1 - tb), _hz(bn) if part == "drop" and pos == 2 else None),
                           tb, 0.9)
            # hats
            div = 4 if part == "drop" else 2
            for h in range(div):
                if part == "intro" and h % 2:
                    continue
                g = 0.35 if h % 2 else 0.5
                if part == "drop" and pos == 3 and h >= 2:
                    g *= 1.2
                sw = swing * beat / div if h % 2 else 0.0
                _place(drums, HC[(b + h) % 4], tb + h * beat / div + sw,
                       g * hrng.uniform(0.7, 1.0) * (0.6 if part in ("intro", "outro") else 1))
            if pos == 3 and part in ("build", "drop"):
                _place(drums, HO, tb + beat / 2, 0.35)
            # pluck arp (8ths)
            arp = [0, 2, 1, 3, 2, 1, 3, 2]
            for e8 in range(2):
                note = ch[arp[(pos * 2 + e8) % 8]] + (12 if part == "drop" and e8 else 0)
                _place(keys, pluck(_hz(note), 0.4, 0.6 if part == "intro" else 1.0), tb + e8 * beat / 2,
                       0.5 if part != "intro" else 0.35)
            if pos == 0 and part in ("build", "drop", "outro", "intro"):
                _place(keys, pad([_hz(x) for x in ch], min(beat * 4, t1 - tb + 0.5),
                                 900 if part == "intro" else 2200), tb, 0.55 if part != "drop" else 0.4)
        bar_i += max(1, nb // 4)
        if part == "outro":
            _place(fx, impact(seed + 1), t0, 0.55)
    keys = keys + 0.22 * np.concatenate([np.zeros(int(beat * 0.75 * SR), np.float32), keys])[:n]   # dotted delay
    keys = _hp(keys, 220)
    # vocal pocket: dip 1–4 kHz of the melodic bus (~ -3 dB) so speech sits in front
    band = _hp(_onepole(keys, 4000), 1000)
    keys = keys - 0.3 * band
    # sidechain: kick-only envelope (fast attack, ~60 ms release) ducks bass + keys
    kenv = np.zeros(n, np.float32)
    kn = int(0.06 * SR)
    shape = np.exp(-np.arange(kn) / (0.05 * SR)).astype(np.float32)
    for kt in kick_times:
        i = int(kt * SR)
        if 0 <= i < n:
            m = min(kn, n - i); kenv[i:i + m] = np.maximum(kenv[i:i + m], shape[:m])
    duck = 1 - 0.7 * kenv
    music = keys * 0.42 * duck + bass * 0.7 * duck
    # riser: high-pass sweep on the music bus right before the drop
    for a0, a1 in sweep:
        i0, i1 = int(a0 * SR), min(n, int(a1 * SR))
        if i1 > i0:
            seg = music[i0:i1].copy()
            hp = _hp(seg, 1200)
            q = np.linspace(0, 1, len(seg)) ** 1.5
            music[i0:i1] = seg * (1 - q) + hp * q
    mix = drums * 0.8 + music + fx * 0.5
    for a0, a1 in sweep:                       # 1/16 of silence right before the drop
        i1 = int(a1 * SR); g = int(beat / 4 * SR)
        if 0 < i1 - g < n:
            mix[i1 - g:i1] *= np.linspace(1, 0, g) ** 3
    pk = np.percentile(np.abs(mix), 99.9) + 1e-9
    mix = mix * (0.7 / pk)                     # ~ -3 dB headroom before the soft limiter
    mix = np.tanh(mix * 1.1) / 1.1 * 0.89
    mix = mix[: int(duration * SR)]
    f = int(0.04 * SR)
    mix[-f:] *= np.linspace(1, 0, f)
    return mix.astype(np.float32)


def beat_path(scene):
    cfg = (scene.spec.get("audio") or {}).get("beat")
    if not cfg:
        return None
    cfg = {} if cfg is True else dict(cfg)
    secs = cfg.get("sections")
    if secs:
        secs = [dict(s, t=scene.resolve_time(s["t"], 0.0, scene.duration)) for s in secs]
    else:
        secs = default_sections(scene)
    key = hashlib.md5(json.dumps([cfg, secs, scene.duration], sort_keys=True, default=str).encode()).hexdigest()[:8]
    out = os.path.join(scene.base_dir, f".beat_{key}.wav")
    if not os.path.exists(out):
        y = render_beat(cfg, scene.duration, secs, int(cfg.get("seed", 3)))
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
            wf.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return out
