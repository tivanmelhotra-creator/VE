"""
mograph.v14.denoise — automatic broadcast voice cleanup (noise cancelling) for talking-head footage.

No knobs needed: it MEASURES the take and picks the strength itself.

  1  measure      noise floor (quietest 10 % of 50 ms windows) vs speech level → SNR
  2  ENGINE       DeepFilterNet3 (sdk/mograph/bin/deep-filter, static binary, model embedded) — default;
                  attenuation cap from SNR (noisy 42 dB … clean 18 dB).  Fallback when the binary is missing:
     rnnoise      neural speech denoiser (RNNoise, ffmpeg `arnndn`, models in mograph/models/rnnoise:
                  lq = voice vs general noise (default, best in QC listening test), bd, sh, cb, mp).
                  The dry/wet mix is chosen from the SNR (noisy take → 0.72, clean studio → 0.40) so clean
                  voices never get the "underwater" artefact.
  3  residual     OPTIONAL spectral gate (noisereduce) — off by default (QC: phasey on outdoor takes)
  4  polish       de-click, high-pass 90 Hz, mud cut (-2 dB @ 250 Hz), presence (+2.5 dB @ 4 kHz),
                  air (+2.5 dB shelf @ 9.5 kHz), light de-ess.  NO gate/expander (QC: pumping, eaten endings).
  5  report       before/after noise floor & SNR → <cache>/denoise_report.json

Use:   from mograph.v14.denoise import clean_voice ;  wav = clean_voice("media/raw.mp4")   (cached)
CLI:   python3 -m mograph.v14.denoise media/raw.mp4  [--model bd|sh|cb|lq|mp] [--strength auto|0..1]
Spec:  "audio": {"denoise": "auto"}   ("off" | "auto" | 0..1 strength | {"model": "sh", "strength": 0.8})
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

SR = 48000
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "rnnoise")


def _read(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def _write(path, x, sr=SR):
    import wave
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def _ff(x, af, sr=SR):
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-af", af,
                        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-"], input=x.astype(np.float32).tobytes(),
                       capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(p.stderr.decode()[-400:])
    y = np.frombuffer(p.stdout, np.float32).copy()
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def measure(x, sr=SR):
    """Noise floor = RMS of the quietest 10 % of 50 ms windows; speech = loudest 30 %."""
    n = int(0.05 * sr)
    k = len(x) // n
    if k < 10:
        return {"noise_db": -90.0, "speech_db": -20.0, "snr_db": 70.0}
    r = np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(1) + 1e-12)
    r = np.sort(r)
    nz = r[: max(1, k // 10)].mean()
    sp = r[-max(1, int(k * 0.3)):].mean()
    db = lambda v: float(20 * np.log10(v + 1e-12))
    return {"noise_db": round(db(nz), 1), "speech_db": round(db(sp), 1), "snr_db": round(db(sp) - db(nz), 1)}


DFN_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "deep-filter")


def _dfn(x, atten_db, sr=SR, pf=False):
    """DeepFilterNet3 (Rikorose/DeepFilterNet, MIT/Apache) — prebuilt static Rust binary, model embedded,
    CPU only, ~0.2× real-time.  Full-band 48 kHz deep filtering: removes non-stationary noise (streams,
    fountains, traffic, wind, crowd) that RNNoise / spectral gates cannot."""
    import tempfile
    import shutil
    d = tempfile.mkdtemp()
    try:
        inp = os.path.join(d, "in.wav")
        _write(inp, x, sr)
        cmd = [DFN_BIN, "-D", "-a", str(int(atten_db)), "-o", os.path.join(d, "o"), inp]
        if pf:
            cmd.insert(1, "--pf")
        r = subprocess.run(cmd, capture_output=True)
        outp = os.path.join(d, "o", "in.wav")
        if r.returncode != 0 or not os.path.exists(outp):
            raise RuntimeError(r.stderr.decode()[-300:])
        y = _read(outp, sr)
        return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def clean(x, sr=SR, model="lq", strength="auto", residual=False, log=print, engine="auto"):
    """Engine order: DeepFilterNet3 (strong, natural) → RNNoise fallback.
    Lesson (v14.2→v14.3): RNNoise at partial mix left the river audible (user complaint) — a WEAK denoiser can't
    be fixed by mixing; you need a model that separates speech from non-stationary noise.  DFN keeps the speech
    spectrum within ~2-3 dB (measured) while removing ~50 dB of stream noise; we cap the attenuation from the
    measured SNR (very noisy → 42 dB, clean → 18 dB) and restore the small HF loss with EQ."""
    m0 = measure(x, sr)
    snr = m0["snr_db"]
    if engine in ("auto", "dfn") and not os.path.exists(DFN_BIN):   # binary not in git → fetch once
        try:
            from .. import fetch_assets
            fetch_assets.try_ensure("deep-filter", log=log)
        except Exception:  # noqa: BLE001
            pass
    use_dfn = engine in ("auto", "dfn") and os.path.exists(DFN_BIN)
    y = _ff(x, "adeclick=w=55:o=75,highpass=f=80")
    if use_dfn:
        att = float(strength) * 50 if strength != "auto" else float(np.clip(42 - (snr - 12) * 0.9, 18, 42))
        try:
            y = _dfn(y, att, sr)
            eng = f"deepfilternet3 atten {att:.0f} dB"
        except Exception as e:  # noqa: BLE001
            log(f"[denoise] DFN failed ({e}) → RNNoise")
            use_dfn = False
    if not use_dfn:
        s = float(np.clip(0.85 - (snr - 12) / 23 * 0.40, 0.45, 0.85)) if strength == "auto" else float(strength)
        mpath = os.path.join(MODELS, f"{model}.rnnn")
        y = _ff(y, f"arnndn=m='{mpath}':mix={s:.2f}") if os.path.exists(mpath) else \
            _ff(y, f"afftdn=nr={6 + 12 * s:.0f}:nf=-50:tn=1")
        eng = f"rnnoise {model} mix {s:.2f}"
    polish = ("equalizer=f=250:t=q:w=1.0:g=-1.5,"      # mud
              "equalizer=f=3500:t=q:w=0.9:g=2.0,"      # presence (measured −2 dB @ 2–5 kHz after DFN)
              "highshelf=f=7000:g=2.5,"                # air      (measured −3 dB @ 5–10 kHz)
              "deesser=i=0.35:m=0.5:f=0.45")
    y = _ff(y, polish)
    if use_dfn:   # "room tone": a whisper of soft pink air (-62 dBFS, voice-ducked) so pauses aren't a dead vacuum
        rng = np.random.default_rng(7)
        pink = _ff(rng.standard_normal(len(y)).astype(np.float32) * 0.02, "lowpass=f=6000,highpass=f=150")
        env = np.sqrt(np.convolve(y * y, np.ones(4800) / 4800, "same"))
        duck = 1 - np.clip(env / (env.max() * 0.15 + 1e-9), 0, 1) * 0.7
        y = y + pink / (np.sqrt((pink ** 2).mean()) + 1e-9) * 10 ** (-62 / 20) * duck
    m1 = measure(y, sr)
    return y, {"before": m0, "after": m1, "engine": eng}


def clean_voice(src, model="lq", strength="auto", log=print, residual=False):
    """Cached: returns path of a 48 kHz mono cleaned WAV next to the media cache."""
    cdir = src + ".v14"
    os.makedirs(cdir, exist_ok=True)
    key = hashlib.md5(f"{os.path.getsize(src)}|{model}|{strength}|{residual}|v5dfn".encode()).hexdigest()[:8]
    out = os.path.join(cdir, f"voice_clean_{key}.wav")
    if os.path.exists(out):
        return out
    x = _read(src)
    y, rep = clean(x, SR, model, strength, residual, log)
    _write(out, y)
    with open(os.path.join(cdir, "denoise_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    log(f"[denoise] {rep['before']} → {rep['after']}  ({rep['engine']})")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--model", default="lq")
    ap.add_argument("--strength", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    st = a.strength if a.strength == "auto" else float(a.strength)
    p = clean_voice(a.src, a.model, st)
    if a.out:
        import shutil
        shutil.copy(p, a.out)
        p = a.out
    print(p)
    sys.exit(0)
