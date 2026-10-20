"""
mograph.voice — natural narration (VO) from the spec's own subtitles, with frame-exact word timing.

Providers (REST only, no SDKs):
  elevenlabs  eleven_v3 (Persian, audio tags [whispers] [sarcastic] …) → character timestamps → exact words
  gemini      Google AI Studio TTS (gemini-3.1-flash-tts-preview …, 30 voices, style prompt) → words aligned
              by whisper if installed, otherwise by syllable weight inside the trimmed clip
  draft       offline, free: speech-shaped placeholder audio with realistic durations (timing preview)

Spec:
  "voice": {"provider": "elevenlabs", "voice": "<voice_id>", "model": "eleven_v3", "lang": "fa",
            "style": "calm, ironic documentary narrator",   # gemini: delivery prompt
            "stability": 0.5, "speed": 1.0, "pause": 0.25, "gain": 1.0}
  subtitle line extras:  "say": "spoken text with [tags] / phonetic spelling"   (display stays "text")
                         "style": "whispered"  (per-line delivery, gemini)

CLI:  python -m mograph voice spec.json [--provider draft] [--out spec.voiced.json]
→ voice/<hash>.wav per line (cached: re-runs only re-synthesise changed lines), voice/vo.wav,
  and spec.voiced.json: word-level subtitles at the REAL spoken times, chapters retimed
  (piecewise-linear map old→new), audio.vo set.  Word anchors (w:...) keep everything in sync.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import wave

import numpy as np

from .keys import get_key
from .text import parse_markup

SR = 44100
_TAG = re.compile(r"\[[^\]]*\]|<[^>]*>")


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _post_json(url, body, headers, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="ignore")[:600]
        raise RuntimeError(f"HTTP {e.code} from {url.split('?')[0]}: {msg}") from None


def _decode_to_pcm(data: bytes, hint=""):
    """any audio bytes (mp3/wav/raw l16@24k) → float32 mono @ SR"""
    if hint.startswith(("l16", "pcm")):
        rate = int(re.findall(r"\d+", hint)[-1]) if re.findall(r"\d+", hint) else 24000
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
        return _resample(x, rate)
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(SR),
                        "pipe:1"], input=data, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype="<i2").astype(np.float32) / 32768


def _resample(x, rate):
    if rate == SR:
        return x
    n = int(len(x) * SR / rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


# ---------------------------------------------------------------------------
# providers → (pcm float32 @SR, words [{w,t,end}] or None)
# ---------------------------------------------------------------------------
def tts_elevenlabs(text, cfg, prev_text=None, next_text=None, project_dir=None):
    key = get_key("ELEVENLABS_API_KEY", project_dir)
    vid = cfg.get("voice") or "JBFqnCBsd6RMkjVDRZzb"
    base = cfg.get("base_url", "https://api.elevenlabs.io")
    url = f"{base}/v1/text-to-speech/{vid}/with-timestamps?output_format=mp3_44100_128"
    body = {"text": text, "model_id": cfg.get("model", "eleven_v3"),
            "voice_settings": {"stability": cfg.get("stability", 0.5), "similarity_boost": cfg.get("similarity", 0.75),
                               "style": cfg.get("style_exaggeration", 0.0), "speed": cfg.get("speed", 1.0)}}
    if cfg.get("lang", "fa"):
        body["language_code"] = cfg.get("lang", "fa")
    if cfg.get("seed") is not None:
        body["seed"] = int(cfg["seed"])
    ctx = {k: v for k, v in (("previous_text", prev_text), ("next_text", next_text)) if v}
    try:
        r = _post_json(url, {**body, **ctx}, {"xi-api-key": key})
    except RuntimeError as e:
        if "HTTP 400" not in str(e) or not ctx:
            raise
        r = _post_json(url, body, {"xi-api-key": key})   # some models reject continuity context
    pcm = _decode_to_pcm(base64.b64decode(r["audio_base64"]))
    al = r.get("alignment") or r.get("normalized_alignment")
    words = _chars_to_words(al["characters"], al["character_start_times_seconds"],
                            al["character_end_times_seconds"]) if al else None
    return pcm, words


def tts_gemini(text, cfg, style=None, project_dir=None):
    key = get_key("GEMINI_API_KEY", project_dir)
    base = cfg.get("base_url", "https://generativelanguage.googleapis.com")
    model = cfg.get("model", "gemini-3.1-flash-tts-preview")
    voice = cfg.get("voice", "Charon")
    style = style or cfg.get("style")
    hdr = {"x-goog-api-key": key}
    # 1) Interactions API (current docs)
    content = {"type": "text", "text": text}
    if style:
        content["annotations"] = [{"type": "speech_metadata", "style": style}]
    body = {"model": model, "input": [{"type": "user_input", "content": [content]}],
            "response_format": {"type": "audio", "mime_type": "audio/wav"},
            "generation_config": {"speech_config": [{"voice": voice}]}}
    try:
        r = _post_json(f"{base}/v1beta/interactions", body, hdr, timeout=240)
        data, mime = None, ""
        for st in r.get("steps", []):
            if st.get("type") == "model_output":
                for c in st.get("content", []):
                    if c.get("type") == "audio" and c.get("data"):
                        data, mime = c["data"], c.get("mime_type", "")
        if not data and r.get("output_audio"):
            data, mime = r["output_audio"].get("data"), r["output_audio"].get("mime_type", "")
        if not data:
            raise RuntimeError("no audio in interactions response")
        raw = base64.b64decode(data)
        return _decode_to_pcm(raw, "l16" if "l16" in mime else ""), None
    except RuntimeError as e:
        if not any(c in str(e) for c in ("HTTP 404", "HTTP 400", "no audio")):
            raise
    # 2) legacy generateContent (PCM s16le 24 kHz)
    prompt = f"Say in this style ({style}): {text}" if style else text
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    r = _post_json(f"{base}/v1beta/models/{model}:generateContent", body, hdr, timeout=240)
    part = r["candidates"][0]["content"]["parts"][0]["inlineData"]
    return _decode_to_pcm(base64.b64decode(part["data"]), "pcm_24000"), None


def tts_draft(text, cfg, **_):
    """Free offline placeholder with realistic Persian pacing (~ 2.6 words/s) — for timing previews."""
    toks = [w for w in _TAG.sub(" ", text).split() if w]
    rate = 0.115 / float(cfg.get("speed", 1.0))       # seconds per letter
    words, pcm, t = [], [], 0.0
    rng = np.random.default_rng(len(text))
    for w in toks:
        d = max(0.18, len(w.replace("\u200c", "")) * rate)
        n = int(d * SR)
        tt = np.arange(n) / SR
        f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * (t + tt))
        syl = 0.5 - 0.5 * np.cos(2 * np.pi * np.clip(tt / d, 0, 1) * max(1, len(w) // 2))
        s = (0.22 * np.sin(2 * np.pi * f0 * tt) + 0.04 * rng.standard_normal(n)) * syl
        pcm.append(s.astype(np.float32))
        words.append({"w": w, "t": t, "end": t + d})
        t += d
        gap = 0.32 if w[-1:] in "،,؛." else 0.07
        pcm.append(np.zeros(int(gap * SR), np.float32))
        t += gap
    return np.concatenate(pcm) if pcm else np.zeros(SR // 2, np.float32), words


PROVIDERS = {"elevenlabs": tts_elevenlabs, "gemini": tts_gemini, "draft": tts_draft}


# ---------------------------------------------------------------------------
# alignment helpers
# ---------------------------------------------------------------------------
def _chars_to_words(chars, starts, ends):
    words, cur, t0, t1, in_tag = [], "", None, None, False
    for c, s, e in zip(chars, starts, ends):
        if c in "[<":
            in_tag = True
        if in_tag:
            if c in "]>":
                in_tag = False
            continue
        if c.isspace():
            if cur:
                words.append({"w": cur, "t": t0, "end": t1}); cur = ""
            continue
        if not cur:
            t0 = s
        cur += c; t1 = e
    if cur:
        words.append({"w": cur, "t": t0, "end": t1})
    return words


def _trim(pcm, thresh_db=-42, pad=0.04):
    a = np.abs(pcm)
    k = max(1, int(0.01 * SR))
    env = np.convolve(a, np.ones(k) / k, "same")
    idx = np.where(env > 10 ** (thresh_db / 20))[0]
    if not len(idx):
        return pcm, 0.0
    i0, i1 = max(0, idx[0] - int(pad * SR)), min(len(pcm), idx[-1] + int(pad * SR))
    return pcm[i0:i1], i0 / SR


def _weight_words(display_words, dur):
    from .hud import _weight
    tot = sum(_weight(w) for w in display_words) or 1
    out, acc = [], 0.0
    for w in display_words:
        d = dur * _weight(w) / tot
        out.append({"w": w, "t": acc, "end": acc + d * 0.92})
        acc += d
    return out


def _whisper_words(pcm):
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return None
    import tempfile
    from .align import transcribe_words
    fd, p = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    _write_wav(p, pcm)
    try:
        return transcribe_words(p)
    finally:
        os.remove(p)


def _write_wav(path, pcm, rate=SR):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())


def _read_wav(path):
    with wave.open(path, "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def synth_line(line, cfg, provider, cache_dir, prev_text=None, next_text=None, project_dir=None, log=print):
    """One subtitle line → (pcm trimmed, words with DISPLAY spelling, times relative to clip start)."""
    from .align import snap_to_script
    display = line.get("text") or " ".join(w["w"] for w in line.get("words", []))
    spoken = line.get("say") or parse_markup(display)[0]
    style = line.get("style")
    h = hashlib.sha1(json.dumps([provider, spoken, style, {k: v for k, v in cfg.items() if k not in ("pause", "gain")}],
                                ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    wav, meta = os.path.join(cache_dir, f"{h}.wav"), os.path.join(cache_dir, f"{h}.json")
    if os.path.exists(wav) and os.path.exists(meta):
        pcm = _read_wav(wav)
        words = json.load(open(meta, encoding="utf-8"))["words"]
        cached = True
    else:
        fn = PROVIDERS[provider]
        if provider == "elevenlabs":
            pcm, words = fn(spoken, cfg, prev_text=prev_text, next_text=next_text, project_dir=project_dir)
        elif provider == "gemini":
            pcm, words = fn(spoken, cfg, style=style, project_dir=project_dir)
        else:
            pcm, words = fn(spoken, cfg)
        pcm, off = _trim(pcm)
        if words:
            words = [{"w": w["w"], "t": max(0.0, w["t"] - off), "end": max(0.0, w["end"] - off)} for w in words]
        else:
            words = _whisper_words(pcm) or None
        disp_tokens = parse_markup(display)[0].split()
        if words:
            words = snap_to_script(words, display)          # spoken tokens → display spelling (+emph)
        else:
            words = _weight_words(disp_tokens, len(pcm) / SR)
            for j in parse_markup(display)[1]:
                if j < len(words):
                    words[j]["emph"] = True
        os.makedirs(cache_dir, exist_ok=True)
        _write_wav(wav, pcm)
        json.dump({"words": words, "spoken": spoken, "provider": provider}, open(meta, "w", encoding="utf-8"),
                  ensure_ascii=False)
        cached = False
    log(f"  [{'cache' if cached else provider}] {len(pcm) / SR:5.2f}s  {display[:48]}")
    return pcm, words


# ---------------------------------------------------------------------------
# "file" provider — narration produced OUTSIDE the SDK (agent-platform TTS tool, studio recording,
# a voice actor…) + its word timestamps (any transcriber).  One clip may cover several lines.
#   "voice": {"provider": "file", "clips": [
#       {"audio": "voice_src/act1.mp3", "words": "voice_src/act1.words.json", "lines": [0, 2]}, ...]}
#   words json: {"words":[{"start","end","text"}]}  |  [{"w","t","end"}]  |  [{"word","start","end"}]
#   lines: inclusive [first, last] subtitle index (sorted by t) spoken in that clip
# ---------------------------------------------------------------------------
def _load_words(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("words") or d.get("segments") or []
    out = []
    for w in d:
        txt = w.get("w") or w.get("text") or w.get("word") or ""
        t0 = w.get("t", w.get("start")); t1 = w.get("end")
        if txt.strip() and t0 is not None:
            out.append({"w": txt.strip(), "t": float(t0), "end": float(t1 if t1 is not None else t0 + 0.2)})
    return out


def file_lines(cfg, subs, base):
    """→ {line_index: (pcm, words_relative)} cut from the external clips at word boundaries."""
    from .align import snap_to_script
    res = {}
    for c in cfg.get("clips") or []:
        ap = os.path.join(base, c["audio"]); wp = os.path.join(base, c["words"])
        pcm = _decode_to_pcm(open(ap, "rb").read())
        a, b = c.get("lines", [0, len(subs) - 1])
        idx = list(range(int(a), int(b) + 1))
        texts = [subs[i].get("text") or " ".join(w["w"] for w in subs[i].get("words", [])) for i in idx]
        words = snap_to_script(_load_words(wp), " ".join(texts))
        k = 0
        groups = []
        for tx in texts:
            n = len(parse_markup(tx)[0].split())
            groups.append(words[k:k + n]); k += n
        for j, (i, g) in enumerate(zip(idx, groups)):
            if not g:
                continue
            prev_end = groups[j - 1][-1]["end"] if j and groups[j - 1] else 0.0
            next_t = groups[j + 1][0]["t"] if j + 1 < len(groups) and groups[j + 1] else len(pcm) / SR
            s0 = max((prev_end + g[0]["t"]) / 2, g[0]["t"] - 0.08, 0.0)
            s1 = min((g[-1]["end"] + next_t) / 2, g[-1]["end"] + 0.18, len(pcm) / SR)
            seg = pcm[int(s0 * SR):int(s1 * SR)].copy()
            f = min(len(seg), int(0.012 * SR))          # de-click the cut points
            if f > 1:
                seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
            res[i] = (seg, [dict(w, t=round(w["t"] - s0, 3), end=round(w["end"] - s0, 3)) for w in g])
    missing = [i for i in range(len(subs)) if i not in res]
    if missing:
        raise SystemExit(f"[voice:file] subtitle lines without audio: {missing} — check clips[].lines")
    return res


def voice_spec(spec_path, provider=None, out_path=None, log=print, dedupe_fillers=None):
    spec = json.load(open(spec_path, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(spec_path))
    cfg = dict(spec.get("voice") or {})
    if dedupe_fillers:
        cfg["dedupe_fillers"] = True
    provider = provider or cfg.get("provider", "draft")
    if provider not in PROVIDERS and provider != "file":
        raise SystemExit(f"unknown voice provider '{provider}' (elevenlabs | gemini | file | draft)")
    subs = sorted(spec.get("subtitles") or [], key=lambda s: float(s["t"]))
    if not subs:
        raise SystemExit("spec has no subtitles to speak")
    cache = os.path.join(base, "voice")
    log(f"[voice] {provider}: {len(subs)} lines")
    min_pause = float(cfg.get("pause", 0.25))
    max_gap = float(cfg["max_pause"]) if cfg.get("max_pause") is not None else None   # v9: between-line clamp
    if max_gap is not None:
        min_pause = min(min_pause, max_gap)
    from .edit import edit_clip, report as _erep
    edit_cfg = {k: cfg[k] for k in ("max_pause", "min_pause", "dedupe_fillers", "fillers", "xfade_ms") if k in cfg}
    line_cuts = {int(k): v for k, v in (cfg.get("line_cut_ranges") or {}).items()}
    cut_log, splices_abs, line_maps = [], [], []
    clips, new_subs, old_t, new_t = [], [], [], []
    cursor = float(subs[0]["t"])
    pre = file_lines(cfg, subs, base) if provider == "file" else None
    for i, line in enumerate(subs):
        disp = lambda s: parse_markup(s.get("text") or "")[0]
        if pre is not None:
            pcm, words = pre[i]
            log(f"  [file] {len(pcm) / SR:5.2f}s  {(line.get('text') or '')[:48]}")
        else:
          pcm, words = synth_line(line, cfg, provider, cache,
                                prev_text=disp(subs[i - 1]) if i else None,
                                next_text=disp(subs[i + 1]) if i + 1 < len(subs) else None,
                                project_dir=base, log=log)
        ecfg = dict(edit_cfg, cut_ranges=line_cuts.get(i, []))
        em = None
        if any(ecfg.get(k) for k in ("max_pause", "dedupe_fillers", "cut_ranges")) and words:
            pcm, words, em = edit_clip(pcm, words, ecfg)
            if em.removed:
                cut_log.append(f"line {i}: {_erep(em.removed)}")
        if i:
            gap = float(line["t"]) - float(subs[i - 1]["end"])     # keep the author's dramatic pauses
            if max_gap is not None:
                gap = min(gap, max_gap)                             # …unless max_pause clamps them
            cursor = max(cursor + min_pause, cursor + gap)
        start = round(cursor, 3)
        old_t.append(float(line["t"])); new_t.append(start)
        if em is not None and em.removed:
            splices_abs += [round(start + sp, 3) for sp in em.splices()]
            line_maps.append((float(line["t"]), float(line["end"]), start, em, len(pcm) / SR))
        clips.append((start, pcm))
        ws = [{"w": w["w"], "t": round(start + w["t"], 3), "end": round(start + w["end"], 3),
               **({"emph": True} if w.get("emph") else {})} for w in words]
        new_subs.append({"t": start, "end": round(start + len(pcm) / SR + 0.12, 3), "words": ws,
                         "text": line.get("text", "")})
        cursor = start + len(pcm) / SR
    total = cursor + 0.6
    vo = np.zeros(int(total * SR) + SR, np.float32)
    for st, pcm in clips:
        i0 = int(st * SR)
        vo[i0:i0 + len(pcm)] += pcm * float(cfg.get("gain", 1.0))
    os.makedirs(cache, exist_ok=True)
    vo_name = os.path.splitext(os.path.basename(spec_path))[0].replace(".voiced", "") + ".vo.wav"
    _write_wav(os.path.join(cache, vo_name), vo)     # one VO per spec (several specs may share voice/)

    def remap(t):
        # v9: inside an edited line, follow that line's own cut map (words that survived keep their beats)
        for (lt0, lt1, st, em_, ln) in line_maps:
            if lt0 <= t <= lt1:
                rel = (t - lt0) / max(1e-6, lt1 - lt0) * (ln + em_.total_removed)
                return st + min(ln, em_(rel))
        return _remap_lin(t)

    def _remap_lin(t):      # old absolute time → new absolute time (piecewise linear on line starts)
        if t <= old_t[0]:
            return t + (new_t[0] - old_t[0])
        for k in range(1, len(old_t)):
            if t <= old_t[k]:
                a, b = old_t[k - 1], old_t[k]
                return new_t[k - 1] + (t - a) / max(1e-6, b - a) * (new_t[k] - new_t[k - 1])
        return t + (new_t[-1] - old_t[-1])

    out = copy.deepcopy(spec)
    out["subtitles"] = new_subs
    _retime_chapters(out.get("chapters", []), remap)
    for j in out.get("omni") or []:          # omni windows are absolute times → same map
        for k in ("t", "end"):
            if isinstance(j.get(k), (int, float)):
                j[k] = round(remap(float(j[k])), 3)
    out.setdefault("audio", {})["vo"] = "voice/" + vo_name
    m = out.setdefault("meta", {})
    if m.get("duration"):
        m["duration"] = round(max(total, remap(float(m["duration"]))), 2)
    if splices_abs:
        out["_cuts"] = splices_abs              # v9: jump-cut times (talking_head punch-in / cover)
    for c in cut_log:
        log(f"[voice:edit] {c}")
    out["_voice"] = {"provider": provider, "duration": round(total, 2),
                     "note": "generated by `mograph voice` — edit the source spec, not this file"}
    out_path = out_path or os.path.splitext(spec_path)[0] + ".voiced.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"[voice] VO {total:.1f}s → voice/{vo_name} ; spec → {out_path}")
    return out_path


def _retime_chapters(chs, remap):
    """Move chapter starts AND every numeric chapter-local time through the same old→new map,
    so beats authored in seconds stay on the words they were written against.
    Word anchors (w:/we:) need nothing — they re-resolve against the new word timings."""
    from .scene import _TIME_KEYS, _NESTED_TIME_LISTS
    old_starts = [float(c["t"]) if isinstance(c.get("t"), (int, float)) else None for c in chs]
    for i, ch in enumerate(chs):
        ot = old_starts[i]
        if ot is None:
            continue
        nt = round(remap(ot), 3)

        def f(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return round(remap(ot + float(v)) - nt, 3)
            return v
        for k in ch.get("camera") or []:
            k[0] = f(k[0])
        for el in ch.get("elements", []):
            for k in _TIME_KEYS:
                if k in el and not (el.get("type") == "morph" and k in ("from", "to")):
                    el[k] = f(el[k])
            for k in _NESTED_TIME_LISTS:
                for it in (el.get(k) if isinstance(el.get(k), list) else []):
                    if isinstance(it, dict) and "t" in it:
                        it["t"] = f(it["t"])
            for k in ("stamp", "total"):
                if isinstance(el.get(k), dict) and "t" in el[k]:
                    el[k]["t"] = f(el[k]["t"])
            if isinstance(el.get("at"), list):
                el["at"] = [f(v) for v in el["at"]]
            for fl in el.get("flips") or []:
                fl[0] = f(fl[0])
            for kv in (el.get("tracks") or {}).values():
                for kk in kv:
                    kk[0] = f(kk[0])
        ch["t"] = nt


def list_voices(provider, project_dir=None):
    if provider == "gemini":
        return ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
                "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
                "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird",
                "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"]
    if provider == "elevenlabs":
        key = get_key("ELEVENLABS_API_KEY", project_dir)
        req = urllib.request.Request("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key})
        with urllib.request.urlopen(req, timeout=60) as r:
            vs = json.loads(r.read().decode())["voices"]
        return [f"{v['voice_id']}  {v['name']}  {v.get('labels', {})}" for v in vs]
    return ["draft (offline placeholder)"]
