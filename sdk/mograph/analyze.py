"""
mograph.analyze — v9 reference video analyzer (OpenMontage "reference-driven creation", BMTS-lite beats).

python3 -m mograph analyze reference.mp4 -o blueprint.json [--spec spec.json]

Extracts with ffprobe + cv2 + scipy only (offline):
  meta       duration, fps, w×h, aspect
  cuts       hard scene-cut timestamps (HSV-histogram distance + adaptive threshold) and cuts/min cadence
  shots      [{t, end, dur}] between cuts
  palette    dominant 4 colours (k-means on sampled frames) mapped to @paper (most common, darkest/lightest
             ground), @ink (max contrast vs paper), @accent (most saturated), @mute (remaining)
  motion     mean optical-flow magnitude per second (energy curve) + overall energy class
  audio      integrated LUFS (ffmpeg ebur128), onsets (spectral flux peaks), tempo (BPM, autocorrelation of the
             onset envelope) and beats [...] (tempo grid phase-locked to the strongest onsets)
  suggest    a ready "meta" block (style / palette tokens) + a chapter skeleton aligned to the cuts

--spec spec.json  writes the beats into spec["beats"] so  "t": "b:12"  (12th beat, 1-based) /  "b:12+0.05"
                  resolves in scene.resolve_time (also "bar:3" = first beat of bar 3 in 4/4).
"""
from __future__ import annotations

import json
import os
import re
import subprocess

import cv2
import numpy as np

SR = 22050


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json", path],
                         capture_output=True, text=True).stdout
    d = json.loads(out or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = any(s.get("codec_type") == "audio" for s in d.get("streams", []))
    num, den = (v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / max(1.0, float(den))
    return {"duration": float(d.get("format", {}).get("duration", 0) or 0), "fps": round(fps, 3),
            "w": int(v.get("width", 0) or 0), "h": int(v.get("height", 0) or 0), "has_audio": a}


def _aspect(w, h):
    if not w or not h:
        return None
    r = w / h
    return min({"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0, "4:5": 0.8}.items(), key=lambda kv: abs(kv[1] - r))[0]


def scan_video(path, step_fps=10.0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / step_fps)))
    hists, times, flow, samples = [], [], [], []
    prev = None
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            sm = cv2.resize(fr, (160, int(160 * fr.shape[0] / fr.shape[1]) or 90))
            hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
            hists.append(cv2.normalize(h, h).flatten())
            times.append(i / fps)
            g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                f = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                flow.append(float(np.linalg.norm(f, axis=2).mean()))
            else:
                flow.append(0.0)
            prev = g
            if len(times) % max(1, int(step_fps / 2)) == 0:
                samples.append(sm.reshape(-1, 3))
        i += 1
    cap.release()
    return np.array(times), hists, np.array(flow), samples


def detect_cuts(times, hists, min_gap=0.25):
    if len(hists) < 3:
        return []
    d = np.array([0.0] + [cv2.compareHist(hists[k - 1], hists[k], cv2.HISTCMP_BHATTACHARYYA)
                          for k in range(1, len(hists))])
    med = np.median(d); mad = np.median(np.abs(d - med)) + 1e-6
    thr = max(0.32, med + 6 * mad)
    cuts = []
    for k in range(1, len(d)):
        if d[k] > thr and (not cuts or times[k] - cuts[-1] >= min_gap):
            cuts.append(round(float(times[k]), 3))
    return cuts


def palette(samples, k=4):
    if not samples:
        return {}
    px = np.concatenate(samples).astype(np.float32)
    if len(px) > 60000:
        px = px[np.random.default_rng(0).choice(len(px), 60000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _c, lab, cen = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.flatten(), minlength=k)
    cols = []
    for i in range(k):                                                     # RGB 0..1, merge near-duplicates
        c = cen[i][::-1] / 255.0
        hit = next((j for j, (cc, _n) in enumerate(cols) if np.abs(cc - c).max() < 0.08), None)
        if hit is None:
            cols.append((c, int(cnt[i])))
        else:
            cols[hit] = (cols[hit][0], cols[hit][1] + int(cnt[i]))

    def hx(c):
        return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)

    def sat(c):
        return (max(c) - min(c)) / (max(c) + 1e-6)

    def lum(c):
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    order = sorted(range(len(cols)), key=lambda i: -cols[i][1])
    paper = cols[order[0]]
    rest = order[1:] or order[:1]
    ai = max(rest, key=lambda i: sat(cols[i][0]) * (0.4 + lum(cols[i][0])))
    rest = [i for i in rest if i != ai]
    accent = cols[ai]
    ii = max(rest, key=lambda i: abs(lum(cols[i][0]) - lum(paper[0]))) if rest else ai
    rest = [i for i in rest if i != ii]
    ink = cols[ii]
    mute = cols[rest[0]] if rest else ink
    if abs(lum(ink[0]) - lum(paper[0])) < 0.35:      # weak ink → pure black / white for legibility
        ink = (np.array([1, 1, 1.0]) if lum(paper[0]) < 0.5 else np.array([0.07, 0.07, 0.07]), 0)
    return {"paper": hx(paper[0]), "ink": hx(ink[0]), "accent": hx(accent[0]), "mute": hx(mute[0]),
            "shares": {hx(c): round(n / max(1, sum(cnt)), 3) for c, n in cols}}


def _audio(path):
    try:
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                             capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    return np.frombuffer(raw, np.float32).copy()


def lufs(path):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter_complex", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"I:\s+(-?\d+(?:\.\d+)?) LUFS", p.stderr)
    return float(m[-1]) if m else None


def beats_from_audio(x, sr=SR):
    """Onsets (spectral flux) + tempo (autocorrelation 70–180 BPM) + phase-locked beat grid."""
    from scipy.signal import find_peaks, stft
    if x is None or len(x) < sr:
        return [], None, []
    f, t, Z = stft(x, fs=sr, nperseg=1024, noverlap=1024 - 256)
    M = np.log1p(np.abs(Z) * 30)
    flux = np.maximum(0, np.diff(M, axis=1)).sum(axis=0)
    flux = np.concatenate([[0], flux])
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    hop = 256 / sr
    pk, _ = find_peaks(flux, height=1.0, distance=max(1, int(0.1 / hop)))
    onsets = [round(float(t[k]), 3) for k in pk]
    env = np.clip(flux, 0, None)
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    lags = np.arange(len(ac)) * hop
    sel = (lags >= 60 / 180) & (lags <= 60 / 70)
    if not sel.any() or ac[sel].max() <= 0:
        return onsets, None, onsets
    period = lags[sel][np.argmax(ac[sel])]
    bpm = 60.0 / period
    # phase: choose the offset that maximises the onset energy on the grid
    best, best_ph = -1, 0.0
    for ph in np.arange(0, period, hop):
        idx = ((np.arange(ph, t[-1], period)) / hop).astype(int)
        idx = idx[idx < len(env)]
        sc = env[idx].sum()
        if sc > best:
            best, best_ph = sc, ph
    beats = [round(float(b), 3) for b in np.arange(best_ph, t[-1], period)]
    return onsets, round(float(bpm), 1), beats


def analyze(path, out=None, spec_path=None, log=print):
    meta = _probe(path)
    meta["aspect"] = _aspect(meta["w"], meta["h"])
    log(f"[analyze] {os.path.basename(path)}  {meta['w']}x{meta['h']} {meta['fps']}fps {meta['duration']:.2f}s")
    times, hists, flow, samples = scan_video(path)
    cuts = detect_cuts(times, hists)
    D = meta["duration"] or (times[-1] if len(times) else 0)
    bounds = [0.0] + cuts + [D]
    shots = [{"t": round(a, 3), "end": round(b, 3), "dur": round(b - a, 3)} for a, b in zip(bounds, bounds[1:]) if b > a]
    pal = palette(samples)
    sec = {}
    for tt, fl in zip(times, flow):
        sec.setdefault(int(tt), []).append(fl)
    energy = [round(float(np.mean(v)), 3) for _k, v in sorted(sec.items())]
    e_mean = float(np.mean(flow)) if len(flow) else 0.0
    audio = {"lufs": None, "bpm": None, "beats": [], "onsets": []}
    if meta["has_audio"]:
        x = _audio(path)
        on, bpm, beats = beats_from_audio(x)
        audio.update({"lufs": lufs(path), "bpm": bpm, "beats": beats, "onsets": on[:400]})
    cpm = round(len(cuts) / max(1e-6, D / 60), 1)
    style = "promo" if cpm > 18 or e_mean > 1.2 else ("kinetic" if cpm > 8 else "swiss" if pal and
                                                      _light(pal.get("paper")) else "lineart")
    bp = {"source": os.path.basename(path), "meta": meta, "cuts": cuts, "cuts_per_min": cpm, "shots": shots,
          "palette": pal, "motion": {"energy_per_s": energy, "mean": round(e_mean, 3),
                                     "class": "high" if e_mean > 1.2 else "medium" if e_mean > 0.4 else "calm"},
          "audio": audio,
          "suggest": {"meta": {"style": style, "aspect": meta["aspect"] or "9:16",
                               "palette_tokens": {k: pal.get(k) for k in ("paper", "ink", "accent", "mute")}},
                      "chapters": [{"t": s["t"], "dur": s["dur"]} for s in shots[:40]],
                      "note": "copy palette_tokens into meta.palette_custom; lock hits with \"t\": \"b:N\""}}
    out = out or os.path.splitext(path)[0] + ".blueprint.json"
    json.dump(bp, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"[analyze] {len(cuts)} cuts ({cpm}/min), palette {[pal.get(k) for k in ('paper','ink','accent','mute')]}, "
        f"LUFS {audio['lufs']}, {audio['bpm']} BPM, {len(audio['beats'])} beats → {out}")
    if spec_path:
        sp = json.load(open(spec_path, encoding="utf-8"))
        sp["beats"] = audio["beats"]
        json.dump(sp, open(spec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        log(f"[analyze] beats written into {spec_path} (use \"t\": \"b:N\")")
    return bp


def _light(h):
    if not h:
        return False
    from .contrast import is_light
    return is_light(h)


def beat_grid(bpm, duration, offset=0.0):
    """Synthetic beat list for specs without a reference: spec["beats"] = {"bpm": 120, "offset": 0.1}."""
    p = 60.0 / float(bpm)
    return [round(offset + k * p, 4) for k in range(int((duration - offset) / p) + 1)]
