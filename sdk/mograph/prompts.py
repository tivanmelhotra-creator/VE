"""
mograph.prompts — v11: turn a spec into the generation jobs an agent must run with its OWN platform tools
(native image generator, music model, TTS), and report which media files are still missing.

    python3 -m mograph prompts spec.json            → prints a plan + writes <spec>.prompts.json
    python3 -m mograph prompts spec.json --out p.json

Sources of jobs
  1. any element whose src file does not exist and that has a  "gen": {"prompt": "...", "aspect": "9:16"}
     (footage / image_card / auto_trace / depth_plate / sprite_rig / planar_insert)
     → image job (for footage: a video job; the agent may also substitute a still + depth_plate).
     Type-specific prompt suffixes are added automatically (line art on black for auto_trace, a 3×2 parts sheet on
     green for sprite_rig, clear foreground/background separation for depth_plate …).
  2. "audio": {"music": "bed.mp3", "music_gen": {"mood": "...", "bpm": 92}} with a missing file → music job
     (BPM / duration / key / mood derived from the spec: beats, duration, style).
  3. "voice" block without a rendered VO → TTS job (see INTEGRATIONS.md, provider "file").
After generating, save the files under the given paths and re-run `mograph qa`.
"""
from __future__ import annotations

import json
import os

SUFFIX = {
    "auto_trace": "clean continuous line art, thick uniform strokes, pure black background, no fill, no shading, "
                  "no text, vector style",
    "sprite_rig": "character parts sheet for 2D puppet animation on a flat pure green (#00ff00) background, grid of "
                  "3 columns x 2 rows: row 1 head, torso (no arms/legs), left arm; row 2 right arm, left leg, right "
                  "leg; each part centred with wide margins, parts not touching, no text, no shadows",
    "depth_plate": "strong depth separation: clear foreground subject, mid-ground, distant background/sky, "
                   "cinematic lighting, no text",
    "image_card": "high quality photo, centred subject, no text, no watermark",
    "planar_insert": "flat front-facing screen/label artwork, full bleed, no device frame, no text unless asked",
    "footage": "",
}
STYLE_MOOD = {"lineart": "calm, minimal, cinematic ambient with soft piano and sub bass",
              "swiss": "clean modern corporate, light percussion, plucks",
              "kinetic": "punchy energetic hip-hop / trap drums",
              "promo": "high-energy electronic trailer, risers and impacts",
              "ui_physics": "playful minimal tech, marimba and clicks"}


def _aspect(spec):
    return spec.get("meta", {}).get("aspect") or "9:16"


def _walk(spec):
    for ci, ch in enumerate(spec.get("chapters") or []):
        stack = list(ch.get("elements") or [])
        while stack:
            el = stack.pop(0)
            yield ci, el
            stack.extend(el.get("children") or [])
            stack.extend(el.get("behind") if isinstance(el.get("behind"), list) else [])


def plan(spec_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(spec_path))
    jobs, missing = [], []
    for ci, el in _walk(spec):
        src = el.get("src")
        if not src or not isinstance(src, str) or src.startswith("#"):
            continue
        p = src if os.path.isabs(src) else os.path.join(base, src)
        if os.path.exists(p):
            continue
        g = el.get("gen")
        if not g:
            missing.append({"chapter": ci + 1, "id": el.get("id"), "type": el.get("type"), "path": src})
            continue
        kind = "video" if el.get("type") == "footage" or src.lower().endswith((".mp4", ".mov")) else "image"
        prompt = g.get("prompt", "")
        sfx = SUFFIX.get(el.get("type"), "")
        if sfx and g.get("suffix", True):
            prompt = f"{prompt}. {sfx}"
        jobs.append({"kind": kind, "tool": "native image generator" if kind == "image" else "video generator",
                     "save_as": src, "aspect": g.get("aspect", _aspect(spec) if el.get("type") in
                                                     ("footage", "depth_plate") else "1:1"),
                     "prompt": prompt, "reference_images": g.get("refs", []),
                     "for": f"ch{ci + 1}:{el.get('id', el.get('type'))} ({el.get('type')})"})
    a = spec.get("audio") or {}
    if a.get("music") and not os.path.exists(os.path.join(base, a["music"])):
        mg = a.get("music_gen") or {}
        beats = spec.get("beats")
        bpm = mg.get("bpm") or (beats.get("bpm") if isinstance(beats, dict) else None) or 96
        dur = spec.get("meta", {}).get("duration") or 30
        style = spec.get("meta", {}).get("style", "lineart")
        jobs.append({"kind": "music", "tool": "music model", "save_as": a["music"],
                     "prompt": f"{mg.get('mood') or STYLE_MOOD.get(style, 'modern background music')}, "
                               f"instrumental, {bpm} BPM, {mg.get('key', 'minor key')}, clear downbeats, "
                               f"{int(round(float(dur)))} seconds, no vocals, loopable ending",
                     "after": f"python3 -m mograph analyze {a['music']} --into {os.path.basename(spec_path)}"})
    if spec.get("voice") and not (a.get("vo") and os.path.exists(os.path.join(base, a["vo"]))):
        jobs.append({"kind": "voice", "tool": "TTS (provider file) or mograph voice",
                     "prompt": " ".join(s.get("text", "") for s in spec.get("subtitles") or [])[:4000],
                     "after": f"python3 -m mograph voice {os.path.basename(spec_path)}"})
    return {"spec": os.path.basename(spec_path), "jobs": jobs, "missing_without_gen": missing}


def run(spec_path, out=None, log=print):
    p = plan(spec_path)
    out = out or os.path.splitext(spec_path)[0] + ".prompts.json"
    json.dump(p, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not p["jobs"] and not p["missing_without_gen"]:
        log("[prompts] nothing to generate — every media file exists ✓")
    for j in p["jobs"]:
        log(f"[{j['kind']:5s}] → {j.get('save_as', '-')}  ({j.get('aspect', '')})  {j.get('for', '')}\n"
            f"         {j['prompt'][:220]}")
    for m in p["missing_without_gen"]:
        log(f"[MISSING] ch{m['chapter']}:{m['id']} {m['type']} → {m['path']} (add \"gen\": {{\"prompt\": …}} or provide it)")
    log(f"[prompts] {len(p['jobs'])} job(s) → {out}")
    return 1 if p["missing_without_gen"] else 0
