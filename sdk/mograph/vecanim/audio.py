"""Audio for vector animation: onset (transient) detection + a small procedural score.

`detect_transients` is a Python port of SoundCraft's `dsp/src/offline.rs detect_transients`
(https://github.com/storytold/soundcraft, Apache-2.0): vibrato-suppressed spectral flux on
24 log bands per octave (each band compared with the loudest of its neighbours in the previous
frame), peaks that stand out from the local mean (0.1 s before / 0.07 s after) by a FIXED delta
(so quiet hits count), a 30 ms minimum gap, and the onset placed where its attack starts.

`music_box_score` writes a gentle "Happy Birthday" music-box arrangement (additive synthesis,
no samples) -> keyframes can be snapped to its detected hits.
"""
import numpy as np, wave


def _log_bands(n, sr):
    """Triangular bands, 24 per octave from 30 Hz, centres de-duplicated (offline.rs log_bands).
    Returns a (B, n/2+1) weight matrix, each row summing to 1."""
    bin_hz = sr / n
    top = min(20000.0, sr * 0.45)
    centres, f = [], 30.0
    while f <= top:
        k = int(round(f / bin_hz))
        if not centres or k > centres[-1]:
            centres.append(k)
        f *= 2 ** (1 / 24)
    Wm = []
    for lo, mid, hi in zip(centres, centres[1:], centres[2:]):
        row = np.zeros(n // 2 + 1)
        for k in range(lo, hi + 1):
            v = (k - lo) / max(mid - lo, 1) if k <= mid else (hi - k) / max(hi - mid, 1)
            if v > 0:
                row[k] = v
        if row.sum() > 0:
            Wm.append(row / row.sum())
    return np.array(Wm)


def _attack_start(x, s, n):
    """First point where the short envelope reaches 20 % of its rise inside the frame."""
    seg = np.abs(x[s:s + n])
    if len(seg) < 8:
        return None
    k = max(1, len(seg) // 48)
    env = np.convolve(seg, np.ones(k) / k, 'same')
    lo, hi = env[:max(2, len(env) // 8)].min(), env.max()
    if hi < 1e-5 or hi < lo * 2:
        return None
    return s + int(np.argmax(env > lo + 0.2 * (hi - lo)))


def detect_transients(x, sr, sensitivity=0.5, min_gap=0.03):
    """x: mono float array.  Returns onset times (seconds), ascending.
    min_gap: SoundCraft uses 30 ms; for animation keys use ~0.15 s so room echoes / flams merge."""
    x = np.asarray(x, np.float32)
    n = int(2 ** np.ceil(np.log2(np.clip(sr * 0.012, 64, 8192))))
    hop = n // 4
    if len(x) < n:
        return np.array([])
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    frames = (len(x) - n) // hop + 1
    idx = np.arange(n)[None] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1))
    energy = (spec ** 2).sum(1)
    gain = 1e4 / (n / 4)
    cur = np.log10(1 + gain * spec @ _log_bands(n, sr).T)
    prev = np.vstack([cur[:1], cur[:-1]])
    nb = np.maximum(np.maximum(prev, np.roll(prev, 1, 1)), np.roll(prev, -1, 1))   # vibrato suppression
    flux = np.maximum(cur - nb, 0).sum(1); flux[0] = 0
    fo = lambda s: max(1, int(s * sr) // hop)
    near, before, after, gap = fo(0.03), fo(0.1), fo(0.07), fo(min_gap)
    delta = 2.5 * 2 ** ((0.5 - np.clip(sensitivity, 0, 1)) * 4)
    out, last = [], None
    for m in range(frames):
        c = flux[m]
        peak = flux[max(0, m - near):m + near + 1].max()
        mean = flux[max(0, m - before):m + after + 1].mean()
        if c <= 0 or c < peak or c < mean + delta or (last is not None and m - last < gap):
            continue
        grows = (energy[m:m + 2] > energy[max(m - 1, 0)]).any()
        at = _attack_start(x, m * hop, n)
        if not grows and at is None:
            continue
        last = m
        out.append((at if at is not None else m * hop) / sr)
    return np.array(sorted(set(np.round(out, 4))))


HB = [('G4', .75), ('G4', .25), ('A4', 1), ('G4', 1), ('C5', 1), ('B4', 2),
      ('G4', .75), ('G4', .25), ('A4', 1), ('G4', 1), ('D5', 1), ('C5', 2),
      ('G4', .75), ('G4', .25), ('G5', 1), ('E5', 1), ('C5', 1), ('B4', 1), ('A4', 2),
      ('F5', .75), ('F5', .25), ('E5', 1), ('C5', 1), ('D5', 1), ('C5', 3)]


def _hz(name):
    st = {'C': -9, 'D': -7, 'E': -5, 'F': -4, 'G': -2, 'A': 0, 'B': 2}[name[0]]
    return 440.0 * 2 ** ((st + 12 * (int(name[-1]) - 4)) / 12)


def music_box_score(path, dur, sr=44100, bpm=84, start=0.6):
    """Write a mono WAV; return the note-onset times written (ground truth for the detector)."""
    N = int(dur * sr)
    y = np.zeros(N, np.float32)
    beat, t, times = 60 / bpm, start, []
    rng = np.random.default_rng(3)
    for note, b in HB:
        if t >= dur - 0.3:
            break
        f = _hz(note)
        L = int(min(2.6, dur - t) * sr)
        tt = np.arange(L) / sr
        tone = (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * f * 2.01 * tt) * np.exp(-tt * 6)
                + 0.12 * np.sin(2 * np.pi * f * 4.2 * tt) * np.exp(-tt * 12))
        env = np.exp(-tt * 2.2) * np.minimum(1, tt / 0.004)
        i0 = int(t * sr)
        y[i0:i0 + L] += 0.22 * (tone * env)[:N - i0] * (0.9 + 0.2 * rng.random())
        times.append(round(t, 4))
        t += b * beat
    tt = np.arange(N) / sr
    pad = sum(np.sin(2 * np.pi * _hz(n) / 2 * tt + k) for k, n in enumerate(['C4', 'G4', 'E4']))
    y += 0.035 * pad * np.clip(tt / 2.5, 0, 1) * np.clip((dur - tt) / 1.5, 0, 1)
    for d, g in ((0.11, 0.25), (0.23, 0.12)):
        k = int(d * sr); y[k:] += g * y[:-k]
    y = y / max(1e-6, np.abs(y).max()) * 0.85
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    return np.array(times)


def read_wav(path):
    with wave.open(path) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        a = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768
    return (a.reshape(-1, ch).mean(1) if ch > 1 else a), sr
