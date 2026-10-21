"""
mograph.v14.remix — BEAT-GRID MUSIC REMIX to an exact duration (idea from FilmCraft `audio-dsp/src/remix.rs`).

    y2 = remix_music_to_duration(y, sr, target_dur, bpm=None, beats_per_bar=4, downbeat=None)

v15.1 rewrite (the v15.0 version moved the second splice point OFF the bar grid — measured on a click track:
beat intervals of 0.36 s / 0.52 s instead of 0.652 s at 92 BPM → audible "hiccup" in the groove):
  1  GRID     bpm (given, or estimated by onset-envelope autocorrelation 60–180 BPM) + first downbeat
              (given, or the strongest onset phase) → bar boundaries b_k = downbeat + k·bar.
  2  FEATURES per bar: 12 log-spaced spectral band energies (55 Hz – 8 kHz) of 3 frames, L2-normalised.
  3  SHORTEN  keep intro bars [0, k) and outro bars [j, end) with j − k = bars to drop; both splice points are
              bar boundaries → the beat grid is continuous.  Score = cosine(bar k−1 , bar j−1) (what the ear
              expects after bar k−1 is bar k; we play bar j instead, so compare their predecessors) +
              cosine(bar k, bar j)  − a small penalty for very short intros/outros.
  4  EXTEND   jump back from bar j to an earlier similar bar i (loop) until long enough, then shorten.
  5  SPLICE   equal-power (raised-cosine) crossfade of `xfade` seconds centred ON the bar boundary.
  6  END      if the result still has to be trimmed inside a bar, a 0.35 s cosine fade-out (or the natural tail
              when the outro ends exactly at the target).
All numpy; < 30 ms for a 60 s bed.
"""
from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------------------------- analysis
def onset_envelope(y: np.ndarray, sr: int, hop: int = 512) -> np.ndarray:
    """Spectral-flux onset strength (half-wave rectified log-magnitude increase), one value per hop."""
    n_fft = 1024
    if len(y) < n_fft + hop:
        return np.zeros(1, np.float32)
    n = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    fr = y[idx] * np.hanning(n_fft).astype(np.float32)
    mag = np.log1p(np.abs(np.fft.rfft(fr, axis=1)) * 10.0)
    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    return np.concatenate([[0.0], flux]).astype(np.float32)


def estimate_bpm(y: np.ndarray, sr: int, lo: float = 60.0, hi: float = 180.0, hop: int = 512) -> float:
    """Tempo from the autocorrelation of the onset envelope (with a mild preference for 80–130 BPM)."""
    env = onset_envelope(y, sr, hop)
    env = env - env.mean()
    if len(env) < 8 or not np.any(env):
        return 100.0
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    fps = sr / hop
    lags = np.arange(len(ac))
    bpm_of = np.where(lags > 0, 60.0 * fps / np.maximum(lags, 1), 0)
    ok = (bpm_of >= lo) & (bpm_of <= hi)
    if not ok.any():
        return 100.0
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm_of, 1) / 105.0) / 0.9) ** 2)
    score = np.where(ok, ac * prior, -np.inf)
    lag = int(np.argmax(score))
    # parabolic refinement of the peak
    if 1 <= lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        den = a - 2 * b + c
        lag = lag + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)
    return float(60.0 * fps / lag)


def fit_grid(y: np.ndarray, sr: int, bpm: float, beats_per_bar: int = 4, hop: int = 256, refine: bool = True):
    """(bpm, downbeat_s): phase (and, if refine, tempo ±2 %) of the bar grid that collects the most onset energy
    on its beats (downbeats weighted ×2).  Tempo refinement removes the slow drift of the autocorrelation estimate."""
    env = onset_envelope(y, sr, hop)
    env = np.maximum(env, cv_max3(env))
    fps = sr / hop
    best = (-1.0, bpm, 0.0)
    for b in (np.linspace(bpm * 0.98, bpm * 1.02, 41) if refine else [bpm]):
        beat_f = 60.0 / b * fps
        bar_f = beat_f * beats_per_bar
        for ph in np.arange(0.0, bar_f, 1.0):
            pos = np.arange(ph, len(env) - 1, beat_f)
            if len(pos) < 2:
                continue
            v = env[np.round(pos).astype(int)]
            w = np.where((np.arange(len(v)) % beats_per_bar) == 0, 2.0, 1.0)
            sc = float((v * w).sum()) / len(pos)
            if sc > best[0]:
                best = (sc, float(b), float(ph / fps))
    return best[1], best[2]


def cv_max3(e: np.ndarray) -> np.ndarray:
    """3-tap running max (tolerates ±1 hop of onset jitter)."""
    p = np.pad(e, 1, mode="edge")
    return np.maximum(np.maximum(p[:-2], p[1:-1]), p[2:])


def estimate_downbeat(y: np.ndarray, sr: int, bpm: float, beats_per_bar: int = 4) -> float:
    return fit_grid(y, sr, bpm, beats_per_bar, refine=False)[1]


def bar_features(y: np.ndarray, sr: int, starts: np.ndarray, bar_n: int) -> np.ndarray:
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(55.0, min(8000.0, sr * 0.45), 13)
    band = np.digitize(freqs, edges) - 1
    feats = np.zeros((len(starts), 12), np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    for i, s0 in enumerate(starts):
        acc = np.zeros(12, np.float32)
        for q in (0.125, 0.45, 0.75):
            a = int(s0 + q * bar_n)
            seg = y[a:a + n_fft]
            if len(seg) < n_fft:
                continue
            mag = np.abs(np.fft.rfft(seg * win))
            for b in range(12):
                m = band == b
                if m.any():
                    acc[b] += np.log1p(float(mag[m].mean()))
        feats[i] = acc / (np.linalg.norm(acc) + 1e-6)
    return feats


# --------------------------------------------------------------------------------------------- splicing
def _splice(a: np.ndarray, b: np.ndarray, at_a: int, at_b: int, xf: int) -> np.ndarray:
    """a[:at_a] + b[at_b:] with an equal-power crossfade of 2·xf samples that ENDS on the bar line: the downbeat
    transient comes only from `b` (no doubled kick), and the sample count is exactly at_a + len(b) − at_b."""
    n = int(min(2 * xf, at_a, at_b))
    if n < 4:
        return np.concatenate([a[:at_a], b[at_b:]]).astype(np.float32)
    ph = np.linspace(0.0, math.pi / 2, n, dtype=np.float32)
    mix = a[at_a - n:at_a] * np.cos(ph) + b[at_b - n:at_b] * np.sin(ph)
    return np.concatenate([a[:at_a - n], mix, b[at_b:]]).astype(np.float32)


def remix_music_to_duration(y: np.ndarray, sr: int, target_dur: float, bpm: float | None = None,
                            beats_per_bar: int = 4, downbeat: float | None = None, xfade: float = 0.03,
                            fade_out: float = 0.35, return_info: bool = False):
    y = np.asarray(y, np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    target_n = int(round(target_dur * sr))
    info = {"bpm": None, "cuts": [], "loops": []}
    if target_n <= 0 or len(y) == 0:
        out = np.zeros(max(1, target_n), np.float32)
        return (out, info) if return_info else out
    given = bool(bpm)
    bpm = float(bpm) if given else estimate_bpm(y, sr)
    if downbeat is None:
        bpm, downbeat = fit_grid(y, sr, bpm, beats_per_bar, refine=not given)
    info["bpm"] = round(bpm, 3)
    info["downbeat"] = round(float(downbeat), 3)
    bar_n = int(round(60.0 / bpm * beats_per_bar * sr))
    off = int(round(float(downbeat) * sr)) % bar_n
    xf = int(xfade * sr)

    def grid(sig):
        return np.arange(off, len(sig) - bar_n // 2, bar_n)

    # ---- EXTEND by looping (jump from a late bar j back to a similar earlier bar i)
    guard = 0
    while len(y) < target_n and guard < 12:
        guard += 1
        st = grid(y)
        if len(st) < 4:                          # too short to analyse → plain tiling on the grid
            seg = y[off:off + max(bar_n, (len(y) - off) // bar_n * bar_n)]
            y = np.concatenate([y, seg])
            continue
        F = bar_features(y, sr, st, bar_n)
        nb = len(st)
        best, bi, bj = -9.0, 1, nb - 1
        for j in range(max(2, nb // 2), nb):
            for i in range(1, j - 1):
                s = float(F[j - 1] @ F[i - 1]) + float(F[j] @ F[i])
                s += 0.02 * (j - i)              # prefer longer loops (less repetition)
                if s > best:
                    best, bi, bj = s, i, j
        info["loops"].append([round(float(st[bj]) / sr, 3), round(float(st[bi]) / sr, 3)])
        y = _splice(y, y, int(st[bj]), int(st[bi]), xf)

    # ---- SHORTEN: drop bars [k, j) — both cut points on the grid
    out = y
    if len(y) > target_n + int(0.05 * sr):
        st = grid(y)
        nb = len(st)
        drop_n = len(y) - target_n
        drop_bars = int(math.ceil(drop_n / bar_n))         # remove whole bars (then trim the tail < 1 bar)
        best, bk = -9.0, None
        F = bar_features(y, sr, st, bar_n) if nb >= 3 else None
        for k in range(1, nb):
            j = k + drop_bars
            if j >= nb:
                break
            s = 0.0
            if F is not None:
                s = float(F[k - 1] @ F[j - 1]) + float(F[k] @ F[j])
            s -= 0.15 * (k < 2) + 0.15 * (nb - j < 2)
            if s > best:
                best, bk = s, k
        if bk is not None:
            out = _splice(y, y, int(st[bk]), int(st[bk + drop_bars]), xf)
            info["cuts"].append([round(float(st[bk]) / sr, 3), round(float(st[bk + drop_bars]) / sr, 3)])
    res = np.zeros(target_n, np.float32)
    m = min(target_n, len(out))
    res[:m] = out[:m]
    if len(out) > target_n + int(0.02 * sr):                # trimmed inside a bar → fade out
        fo = min(m, int(fade_out * sr))
        res[m - fo:m] *= 0.5 * (1.0 + np.cos(np.linspace(0, math.pi, fo, dtype=np.float32)))
    else:
        fo = min(m, int(0.012 * sr))
        res[m - fo:m] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return (res, info) if return_info else res


def load_music(path: str, sr: int) -> np.ndarray:
    """Any audio file → mono float32 at `sr` (ffmpeg)."""
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()
