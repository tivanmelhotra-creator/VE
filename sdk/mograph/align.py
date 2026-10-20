"""
mograph.align — narration audio → word-level subtitles (frame-exact karaoke + word anchors).

Uses faster-whisper if installed (pip install faster-whisper; CPU 'small' model is fine for Persian).
If a script text is given, recognised words are snapped to the SCRIPT's spelling (so word anchors
like "w:مستندتره؛" match exactly), keeping whisper's timing.

CLI:  python -m mograph align narration.wav --text script.txt --out subs.json
Then paste subs.json into spec["subtitles"] (word form) and set spec["audio"]["vo"].
"""
from __future__ import annotations

import difflib
import json

from .audio import subtitles_from_words


def transcribe_words(audio_path, language="fa", model_size="small"):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit("faster-whisper not installed:  pip install faster-whisper") from e
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio_path, language=language, word_timestamps=True, vad_filter=True)
    words = []
    for sg in segs:
        for w in sg.words or []:
            words.append({"w": w.word.strip(), "t": round(w.start, 3), "end": round(w.end, 3)})
    return words


def snap_to_script(words, script_text):
    """Replace recognised tokens by the script's tokens (sequence alignment), interpolating timing
    for script words whisper missed. Emphasis markers **x** in the script are preserved as emph."""
    from .text import parse_markup
    clean, emph = parse_markup(script_text)
    sw = clean.split()
    rw = [w["w"] for w in words]
    sm = difflib.SequenceMatcher(a=rw, b=sw, autojunk=False)
    out = [None] * len(sw)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                out[j1 + k] = dict(words[i1 + k], w=sw[j1 + k])
            if tag == "replace" and (j2 - j1) > n and i2 > i1:   # spread remaining script words
                t0, t1 = words[i2 - 1]["t"], words[i2 - 1]["end"]
                for k in range(n, j2 - j1):
                    f = (k - n + 1) / (j2 - j1 - n + 1)
                    out[j1 + k] = {"w": sw[j1 + k], "t": t0 + (t1 - t0) * f, "end": t1}
    # fill gaps by interpolation between neighbours
    for j in range(len(sw)):
        if out[j] is None:
            prev = next((out[k] for k in range(j - 1, -1, -1) if out[k]), None)
            nxt = next((out[k] for k in range(j + 1, len(sw)) if out[k]), None)
            t0 = prev["end"] if prev else (nxt["t"] - 0.3 if nxt else 0)
            t1 = nxt["t"] if nxt else t0 + 0.3
            out[j] = {"w": sw[j], "t": round(t0, 3), "end": round(max(t0 + 0.08, t1), 3)}
    for j in emph:
        if j < len(out):
            out[j]["emph"] = True
    return out


def align_cli(audio_path, script_path=None, out_path=None):
    words = transcribe_words(audio_path)
    if script_path:
        words = snap_to_script(words, open(script_path, encoding="utf-8").read())
    subs = subtitles_from_words(words)
    js = json.dumps(subs, ensure_ascii=False, indent=1)
    if out_path:
        open(out_path, "w", encoding="utf-8").write(js)
        print(out_path)
    else:
        print(js)
    return 0
