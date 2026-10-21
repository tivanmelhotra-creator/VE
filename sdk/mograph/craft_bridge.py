"""
mograph/craft_bridge.py — FilmCraft + EffectCraft Native Studio Bridge (v14 Master)
===================================================================================
Ports the core mathematical, color, 3D geometry, layer-style, and broadcast audio DSP
engines from `storytold/filmcraft` and `storytold/effectcraft` into `mograph`.
"""

from __future__ import annotations
import math
import os
import subprocess
import tempfile
import cairo
import cv2
import numpy as np
from scipy import signal


def _make_octagonal_bokeh_kernel(radius: int, sides: int = 8, roundness: float = 0.35, fringe: float = 0.45) -> np.ndarray:
    r = max(2, int(radius))
    size = 2 * r + 1
    y, x = np.ogrid[-r:r+1, -r:r+1]
    dist = np.hypot(x, y) / float(r)
    ang = np.arctan2(y, x)
    n = float(max(3, sides))
    seg = (2.0 * math.pi) / n
    rel = np.mod(ang, seg) - seg * 0.5
    poly = math.cos(math.pi / n) / np.maximum(np.cos(rel), 1e-4)
    bound = poly + (1.0 - poly) * float(np.clip(roundness, 0.0, 1.0))
    norm_d = dist / np.maximum(bound, 1e-4)
    inside = np.clip((1.05 - norm_d) * 6.0, 0.0, 1.0)
    rim = np.exp(-((norm_d - 0.86) ** 2) / 0.018) * float(fringe)
    k = inside * (1.0 + rim)
    s = np.sum(k)
    return (k / s).astype(np.float32) if s > 1e-6 else np.ones((size, size), dtype=np.float32) / (size * size)

_BOKEH_KERNEL_CACHE: dict[tuple[int, int], np.ndarray] = {}

def effectcraft_iris_bokeh(bgr: np.ndarray, radius: int = 11, highlight_gain: float = 0.75,
                           highlight_thresh: float = 0.62, sides: int = 8) -> np.ndarray:
    if radius <= 1:
        return bgr
    key = (int(radius), int(sides))
    if key not in _BOKEH_KERNEL_CACHE:
        _BOKEH_KERNEL_CACHE[key] = _make_octagonal_bokeh_kernel(radius, sides=sides)
    kern = _BOKEH_KERNEL_CACHE[key]

    h, w = bgr.shape[:2]
    r_eff = max(2, min(5, int(radius // 2)))
    key_eff = (r_eff, int(sides))
    if key_eff not in _BOKEH_KERNEL_CACHE:
        _BOKEH_KERNEL_CACHE[key_eff] = _make_octagonal_bokeh_kernel(r_eff, sides=sides)
    kern_eff = _BOKEH_KERNEL_CACHE[key_eff]
    small = cv2.resize(bgr, (w // 5, h // 5), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    lum = 0.114 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.299 * small[:, :, 2]
    hi_mask = np.clip((lum - highlight_thresh) / max(1e-3, 1.0 - highlight_thresh), 0.0, 1.0)[:, :, None]
    boosted = small * (1.0 + hi_mask * float(highlight_gain))
    blurred = cv2.filter2D(boosted, -1, kern_eff, borderType=cv2.BORDER_REFLECT)
    out = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


_SPOT_CACHE: dict[tuple[int, int], np.ndarray] = {}

def filmcraft_lumetri_grade(
    bgr: np.ndarray,
    person_alpha: np.ndarray,
    bg_dim: float = 0.42,
    bg_tint_bgr: tuple[int, int, int] = (28, 22, 14),
    bokeh_radius: int = 11,
    dof_speaker_blur: float = 0.0,
    draw_dna_grid: bool = True,
    accent_bgr: tuple[int, int, int] = (36, 191, 251),
) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    bg_bokeh = effectcraft_iris_bokeh(bgr, radius=bokeh_radius, highlight_gain=0.85, highlight_thresh=0.58)
    # Compute background grade at 1/4 resolution then upsample (16x faster, zero loss on bokeh background!)
    hs, ws = h // 4, w // 4
    bg_s = cv2.resize(bg_bokeh, (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum_s = (0.114 * bg_s[:, :, 0] + 0.587 * bg_s[:, :, 1] + 0.299 * bg_s[:, :, 2])[:, :, None]
    bg_desat = bg_s * 0.45 + lum_s * 0.55
    tint_arr = np.array(bg_tint_bgr, dtype=np.float32)[None, None, :]
    bg_g_s = bg_desat * (1.0 - bg_dim) + tint_arr * bg_dim
    if (hs, ws) not in _SPOT_CACHE:
        yy, xx = np.ogrid[0:hs, 0:ws]
        r2 = ((xx - ws * 0.5) / (ws * 0.52)) ** 2 + ((yy - hs * 0.36) / (hs * 0.42)) ** 2
        _SPOT_CACHE[(hs, ws)] = np.exp(-r2 * 1.35).astype(np.float32)[:, :, None]
    spot_s = _SPOT_CACHE[(hs, ws)]
    acc_arr = np.array(accent_bgr, dtype=np.float32)[None, None, :]
    bg_g_s = np.clip(bg_g_s * (0.72 + 0.38 * spot_s) + acc_arr * (spot_s * 0.11), 0, 255).astype(np.uint8)
    bg_graded = cv2.resize(bg_g_s, (w, h), interpolation=cv2.INTER_LINEAR)

    if draw_dna_grid:
        step = 120
        grid_col = (
            min(255, int(bg_tint_bgr[0] + 26)),
            min(255, int(bg_tint_bgr[1] + 28)),
            min(255, int(bg_tint_bgr[2] + 30)),
        )
        for gx in range(60, w, step):
            cv2.line(bg_graded, (gx, 0), (gx, h), grid_col, 1, cv2.LINE_AA)
        for gy in range(60, h, step):
            cv2.line(bg_graded, (0, gy), (w, gy), grid_col, 1, cv2.LINE_AA)
        tick_col = (
            min(255, int(accent_bgr[0] * 0.45 + 40)),
            min(255, int(accent_bgr[1] * 0.45 + 40)),
            min(255, int(accent_bgr[2] * 0.45 + 40)),
        )
        for gx in range(60, w, step * 2):
            for gy in range(60, int(h * 0.65), step * 2):
                cv2.line(bg_graded, (gx - 5, gy), (gx + 5, gy), tick_col, 1, cv2.LINE_AA)
                cv2.line(bg_graded, (gx, gy - 5), (gx, gy + 5), tick_col, 1, cv2.LINE_AA)

    if not hasattr(filmcraft_lumetri_grade, "_fg_lut"):
        x = np.arange(256, dtype=np.float32) / 255.0
        c = x - 0.11 * np.sin(2.0 * math.pi * x)
        m = 4.0 * c * (1.0 - c)
        lut = np.zeros((1, 256, 3), dtype=np.uint8)
        lut[0, :, 0] = np.clip((c - 0.025 * m) * 255.0, 0, 255).astype(np.uint8)
        lut[0, :, 1] = np.clip((c + 0.018 * m) * 255.0, 0, 255).astype(np.uint8)
        lut[0, :, 2] = np.clip((c + 0.045 * m) * 255.0, 0, 255).astype(np.uint8)
        filmcraft_lumetri_grade._fg_lut = lut
    fg_u8 = cv2.LUT(bgr, filmcraft_lumetri_grade._fg_lut)

    if dof_speaker_blur > 0.02:
        k_blur = max(3, int(dof_speaker_blur * 28) | 1)
        fg_u8 = effectcraft_iris_bokeh(fg_u8, radius=max(3, k_blur // 2), highlight_gain=0.45, highlight_thresh=0.68)
        fg_u8 = (fg_u8.astype(np.float32) * (1.0 - 0.32 * dof_speaker_blur)).astype(np.uint8)
    else:
        blur_detail = cv2.GaussianBlur(fg_u8, (0, 0), 1.4)
        fg_u8 = cv2.addWeighted(fg_u8, 1.22, blur_detail, -0.22, 0)

    return bg_graded, fg_u8


def composite_speaker_with_auto_light_wrap(
    canvas_bgra: np.ndarray,
    fg_bgr: np.ndarray,
    person_alpha: np.ndarray,
    wrap_strength: float = 0.55,
    defringe: float = 0.35,
) -> None:
    h, w = fg_bgr.shape[:2]
    hs, ws = h // 4, w // 4
    pa_s = cv2.resize(person_alpha, (ws, hs), interpolation=cv2.INTER_AREA)
    pa_blur_s = cv2.GaussianBlur(pa_s, (9, 9), 2.6)
    rim_s = np.clip((1.0 - pa_blur_s) * pa_s * 2.2, 0.0, 1.0)[:, :, None]

    behind_s = cv2.resize(canvas_bgra[:, :, :3], (ws, hs), interpolation=cv2.INTER_AREA).astype(np.float32)
    b_blur_s = cv2.GaussianBlur(behind_s, (11, 11), 3.6)
    wrap_add_s = np.clip((b_blur_s - 28.0) * (float(wrap_strength) * 0.85), 0.0, 255.0) * rim_s
    wrap_add = cv2.resize(wrap_add_s.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    fg_lit = cv2.add(fg_bgr, wrap_add)

    pa_u16 = (np.clip(person_alpha, 0.0, 1.0) * 256.0).astype(np.uint16)[:, :, None]
    inv_u16 = 256 - pa_u16
    canvas_bgra[:, :, :3] = ((fg_lit.astype(np.uint16) * pa_u16 + canvas_bgra[:, :, :3].astype(np.uint16) * inv_u16) >> 8).astype(np.uint8)
    canvas_bgra[:, :, 3] = 255


def _design_peaking_biquad(sr: float, freq: float, q: float, gain_db: float):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

def _design_high_shelf_biquad(sr: float, freq: float, q: float, gain_db: float):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    sqA = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + sqA)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - sqA)
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + sqA
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - sqA
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1.0, a1/a0, a2/a0])

def filmcraft_compressor_curve_db(x_db: np.ndarray, threshold_db: float = -21.0,
                                  ratio: float = 3.5, knee_db: float = 6.0) -> np.ndarray:
    d = x_db - threshold_db
    out = np.where(d > 0.0, threshold_db + d / ratio, x_db)
    if knee_db > 0.0:
        in_knee = (2.0 * np.abs(d)) <= knee_db
        knee_val = x_db + (1.0 / ratio - 1.0) * ((d + knee_db * 0.5) ** 2) / (2.0 * knee_db)
        out = np.where(in_knee, knee_val, out)
    return out

def filmcraft_expander_curve_db(x_db: np.ndarray, threshold_db: float = -35.0,
                                ratio: float = 2.2, range_db: float = 10.0) -> np.ndarray:
    atten = np.maximum((x_db - threshold_db) * (ratio - 1.0), -range_db)
    return np.where(x_db >= threshold_db, x_db, x_db + atten)

def _splice_segments_zero_crossing(stereo: np.ndarray, sr: int, segments: list[tuple[float, float]], crossfade_ms: float = 16.0) -> np.ndarray:
    cf_len = max(32, int(sr * (crossfade_ms / 1000.0)))
    pieces = []
    for idx, (t0, t1) in enumerate(segments):
        i0 = max(0, min(len(stereo) - 1, int(round(t0 * sr))))
        i1 = max(i0 + cf_len * 2, min(len(stereo), int(round(t1 * sr))))
        win_z = min(200, i0)
        if win_z > 4:
            mono_slice = np.mean(stereo[i0 - win_z:i0 + win_z], axis=1)
            zc = np.where(np.diff(np.signbit(mono_slice)))[0]
            if len(zc) > 0:
                best = zc[np.argmin(np.abs(zc - win_z))]
                i0 = i0 - win_z + int(best)
        seg = stereo[i0:i1].copy()
        if idx == 0:
            fi = min(len(seg), int(0.025 * sr))
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi, fi)))[:, None]
            seg[:fi] *= ramp
        if idx == len(segments) - 1:
            fo = min(len(seg), int(0.060 * sr))
            ramp = 0.5 * (1.0 + np.cos(np.linspace(0, math.pi, fo)))[:, None]
            seg[-fo:] *= ramp
        pieces.append(seg)

    if len(pieces) == 1:
        return pieces[0]

    out = pieces[0]
    for nxt in pieces[1:]:
        ov = min(cf_len, len(out) // 4, len(nxt) // 4)
        t_lin = np.linspace(0.0, math.pi, ov)[:, None]
        w_out = 0.5 * (1.0 + np.cos(t_lin))
        w_in = 0.5 * (1.0 - np.cos(t_lin))
        blended = out[-ov:] * w_out + nxt[:ov] * w_in
        out = np.vstack([out[:-ov], blended, nxt[ov:]])
    return out


def synthesize_studio_music_bed(duration_s: float, sr: int = 44100, bpm: float = 118.0) -> np.ndarray:
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    chord_freqs = [
        (0.0, 4.0, [146.83, 220.00, 261.63, 329.63, 440.00]),
        (4.0, 8.0, [116.54, 174.61, 220.00, 293.66, 349.23]),
        (8.0, 12.0, [98.00, 146.83, 174.61, 220.00, 293.66]),
        (12.0, duration_s + 0.5, [110.00, 164.81, 196.00, 261.63, 329.63]),
    ]
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    for t0, t1, freqs in chord_freqs:
        i0, i1 = max(0, int(t0 * sr)), min(n, int(t1 * sr))
        if i1 <= i0:
            continue
        tc = t[i0:i1] - t0
        dur_c = max(0.2, t1 - t0)
        env = np.sin(np.clip(tc / dur_c, 0.0, 1.0) * math.pi) ** 0.45
        for f_i, f in enumerate(freqs):
            detune = 1.0 + (f_i - 2) * 0.0012
            sig_l = np.sin(2.0 * math.pi * f * tc) + 0.25 * np.sin(2.0 * math.pi * (f * 2.0) * tc) * np.exp(-tc * 0.8)
            sig_r = np.sin(2.0 * math.pi * (f * detune) * tc + 0.35) + 0.25 * np.sin(2.0 * math.pi * (f * 2.0 * detune) * tc) * np.exp(-tc * 0.8)
            pan_l = 0.65 - 0.1 * (f_i % 3)
            pan_r = 0.35 + 0.1 * (f_i % 3)
            left[i0:i1] += sig_l * env * pan_l * 0.045
            right[i0:i1] += sig_r * env * pan_r * 0.045

    beat_dt = 60.0 / bpm
    beat_t = 0.0
    while beat_t < duration_s:
        bi = int(beat_t * sr)
        blen = min(int(0.28 * sr), n - bi)
        if blen > 10:
            tb = np.arange(blen, dtype=np.float32) / float(sr)
            f_sub = 73.42 * (1.0 + 0.45 * np.exp(-tb * 28.0))
            sub_wave = np.sin(2.0 * math.pi * f_sub * tb) * np.exp(-tb * 11.0) * 0.11
            left[bi:bi+blen] += sub_wave
            right[bi:bi+blen] += sub_wave
        beat_t += beat_dt

    return np.column_stack([left, right])


def filmcraft_master_reel_audio(
    raw_mp4_path: str,
    segments: list[tuple[float, float]],
    sfx_events: list[tuple[float, str, float, float]],
    out_wav_path: str,
    sr: int = 44100,
) -> float:
    from mograph.sound import syn

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_raw_wav = tmp_in.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_mp4_path, "-vn", "-ac", "2", "-ar", str(sr), tmp_raw_wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    from scipy.io import wavfile
    _, raw_data = wavfile.read(tmp_raw_wav)
    os.remove(tmp_raw_wav)

    if raw_data.ndim == 1:
        raw_data = np.column_stack([raw_data, raw_data])
    vo = raw_data.astype(np.float32) / 32768.0

    vo = _splice_segments_zero_crossing(vo, sr, segments, crossfade_ms=16.0)
    n_samples = len(vo)
    dur_s = n_samples / float(sr)

    sos_hp = signal.butter(2, 85.0 / (sr * 0.5), btype="highpass", output="sos")
    vo = signal.sosfilt(sos_hp, vo, axis=0)

    eq_bands = [
        _design_peaking_biquad(sr, 195.0, 1.1, +1.8),
        _design_peaking_biquad(sr, 320.0, 1.6, -2.6),
        _design_peaking_biquad(sr, 1850.0, 1.2, +1.5),
        _design_peaking_biquad(sr, 3350.0, 1.0, +3.2),
        _design_high_shelf_biquad(sr, 10200.0, 0.707, +2.5),
    ]
    for b_coeff, a_coeff in eq_bands:
        vo = signal.lfilter(b_coeff, a_coeff, vo, axis=0)

    sos_sib = signal.butter(2, [5200.0 / (sr * 0.5), 8800.0 / (sr * 0.5)], btype="bandpass", output="sos")
    sib_band = signal.sosfilt(sos_sib, vo, axis=0)
    sib_env = np.sqrt(signal.lfilter([0.005], [1.0, -0.995], np.mean(sib_band ** 2, axis=1)))
    deess_gain = np.where(sib_env > 0.018, 0.018 / np.maximum(sib_env, 1e-5), 1.0)[:, None]
    vo = vo - sib_band * (1.0 - deess_gain) * 0.65

    mono_pow = np.mean(vo ** 2, axis=1)
    alpha_rms = 1.0 - math.exp(-1.0 / (0.010 * sr))
    rms_sq = signal.lfilter([alpha_rms], [1.0, -(1.0 - alpha_rms)], mono_pow)
    env_db = 10.0 * np.log10(np.maximum(rms_sq, 1e-10))

    exp_out_db = filmcraft_expander_curve_db(env_db, threshold_db=-36.0, ratio=2.0, range_db=9.0)
    comp_out_db = filmcraft_compressor_curve_db(exp_out_db, threshold_db=-22.0, ratio=3.4, knee_db=6.0)
    target_gr_db = comp_out_db - env_db
    alpha_gr = 1.0 - math.exp(-1.0 / (0.015 * sr))
    smooth_gr_db = signal.lfilter([alpha_gr], [1.0, -(1.0 - alpha_gr)], target_gr_db)
    vo_gain = (10.0 ** ((smooth_gr_db + 5.5) / 20.0))[:, None]
    vo = vo * vo_gain

    bed = synthesize_studio_music_bed(dur_s, sr=sr, bpm=118.0)
    speech_act = np.clip((env_db - (-33.0)) / 14.0, 0.0, 1.0)
    win_duck = int(0.18 * sr)
    speech_env = np.convolve(speech_act, np.ones(win_duck) / win_duck, mode="same")[:, None]

    sos_mid = signal.butter(2, [300.0 / (sr * 0.5), 3400.0 / (sr * 0.5)], btype="bandpass", output="sos")
    bed_mid = signal.sosfilt(sos_mid, bed, axis=0)
    bed_out = bed - bed_mid
    bed_ducked = bed_mid * (1.0 - 0.72 * speech_env) + bed_out * (1.0 - 0.28 * speech_env)

    sfx_bus = np.zeros_like(vo)
    for t_ev, kind, gain, pitch_mult in sfx_events:
        if t_ev < 0.0 or t_ev >= dur_s:
            continue
        wave_mono = syn(kind, dur=0.28, pitch=pitch_mult)
        if len(wave_mono) == 0:
            continue
        i0 = int(round(t_ev * sr))
        i1 = min(n_samples, i0 + len(wave_mono))
        seg_w = wave_mono[:i1 - i0] * float(gain)
        sfx_bus[i0:i1, 0] += seg_w * 0.95
        sfx_bus[i0:i1, 1] += seg_w * 1.05

    mix = vo + bed_ducked + sfx_bus
    mix = np.tanh(mix * 0.92) / 0.92
    mix_i16 = np.clip(mix * 32767.0, -32767, 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_m:
        tmp_pre_lufs = tmp_m.name
    wavfile.write(tmp_pre_lufs, sr, mix_i16)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", tmp_pre_lufs,
            "-af", "loudnorm=I=-14.0:TP=-1.0:LRA=9.0",
            "-ar", str(sr), "-ac", "2", out_wav_path,
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    os.remove(tmp_pre_lufs)
    return dur_s
