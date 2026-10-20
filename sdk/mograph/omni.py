"""
mograph.omni — Gemini Omni Flash bridge: realistic shots where vector code is the wrong tool.

The engine stays the director: it renders its OWN frames (exact layout/camera/timing) and hands them
to Omni as the input; Omni returns a realistic clip that is laid back onto the timeline as a PLATE.
Vector HUD, karaoke subtitles and post-FX stay crisp on top.  If the clip is missing, the vector
version renders unchanged (safe fallback) — QA tells you.

Spec (top level, times are absolute seconds or word anchors):
  "omni": [{
     "id": "coin_real",
     "t": "w:سکه", "end": "w:سکه+5",          # 3..10 s
     "task": "edit",          # edit        : engine frames t..end → Omni restyles them (keeps motion/layout)
                              # first_last  : engine frame at t + at end → Omni animates between them
                              # from_frame  : engine frame at t → Omni continues it (image_to_video)
                              # text        : prompt only (b-roll)
     "prompt": "Turn the white line-art coin into a real heavy gold coin spinning on black velvet ...",
     "mode": "replace",       # replace: footage over the world | under: footage behind the vector layer
     "grade": "noir",         # none | mono | noir  (match the noir style)
     "fade": 0.25, "resolution": "720p"
  }]

CLI:
  python -m mograph omni spec.json            # DRY RUN: shows the plan + cost-relevant facts (no API call)
  python -m mograph omni spec.json --yes      # generate missing/changed clips → media/omni/<id>.mp4
  python -m mograph omni spec.json --yes --job coin_real --provider draft   # offline stand-in for tests
Rules that make Omni reliable (from Google's guidance + gbro-collage-broll):
  * prompts in ENGLISH, short, one change; end edits with "Keep everything else the same."
  * never ask Omni to write Persian text — keep text in the vector layer
  * input clips ≤ 10 s; one idea per clip; locked camera unless motion is the point
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

from .keys import get_key

MODEL = "gemini-omni-1.1-flash"
BASE = "https://generativelanguage.googleapis.com"
GUARD_EDIT = (" Keep the exact framing, layout, timing and camera motion of the input video."
              " Keep everything else the same. No text, no letters, no logos, no subtitles, no dialogue, no music.")
GUARD_GEN = " Locked vertical 9:16 frame. No text, no letters, no logos, no subtitles, no dialogue, no music."


# ---------------------------------------------------------------------------
# REST client (no SDK needed)
# ---------------------------------------------------------------------------
def _req(method, url, key, body=None, headers=None, timeout=600, raw=False):
    data = None if body is None else (body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode())
    h = {"x-goog-api-key": key, **({"Content-Type": "application/json"} if body is not None and not raw else {}),
         **(headers or {})}
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
            return (payload, dict(r.headers)) if raw else json.loads(payload.decode() or "{}")
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="ignore")[:800]
        hint = ""
        if e.code in (400, 403) and "region" in msg.lower():
            hint = "  (video edit/extend is unavailable in EEA/CH/UK — use task first_last/from_frame)"
        raise RuntimeError(f"Omni HTTP {e.code}: {msg}{hint}") from None


def upload_file(path, key, base=BASE, mime="video/mp4"):
    """Gemini Files API resumable upload → file uri (waits until ACTIVE)."""
    size = os.path.getsize(path)
    _, hdr = _req("POST", f"{base}/upload/v1beta/files", key, body=json.dumps({"file": {"display_name": os.path.basename(path)}}).encode(),
                  headers={"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                           "X-Goog-Upload-Header-Content-Length": str(size),
                           "X-Goog-Upload-Header-Content-Type": mime, "Content-Type": "application/json"}, raw=True)
    up = {k.lower(): v for k, v in hdr.items()}.get("x-goog-upload-url")
    if not up:
        raise RuntimeError("Files API did not return an upload URL")
    payload, _ = _req("POST", up, key, body=open(path, "rb").read(), raw=True,
                      headers={"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize",
                               "Content-Length": str(size)})
    f = json.loads(payload.decode())["file"]
    for _ in range(90):
        if f.get("state", "ACTIVE") == "ACTIVE":
            return f["uri"]
        if f.get("state") == "FAILED":
            raise RuntimeError("uploaded file processing FAILED")
        time.sleep(2)
        f = _req("GET", f"{base}/v1beta/{f['name']}", key)
    raise RuntimeError("uploaded file never became ACTIVE")


def _part(path, kind, key, base):
    mime = "video/mp4" if kind == "video" else ("image/png" if path.endswith(".png") else "image/jpeg")
    size = os.path.getsize(path)
    if kind == "video" and size > 14 * 1024 * 1024:
        return {"type": "video", "uri": upload_file(path, key, base, mime), "mime_type": mime}
    return {"type": kind, "data": base64.b64encode(open(path, "rb").read()).decode(), "mime_type": mime}


def interact(prompt, key, images=(), video=None, task=None, aspect="9:16", resolution="720p", duration=None,
             previous_id=None, model=MODEL, base=BASE, log=print):
    """One Interactions API call → (mp4 bytes, interaction_id)."""
    inp = [_part(p, "image", key, base) for p in images]
    if video:
        inp.append(_part(video, "video", key, base))
    inp.append({"type": "text", "text": prompt})
    body = {"model": model, "input": inp,
            "response_format": {"type": "video", "aspect_ratio": aspect, "resolution": resolution, "delivery": "uri"}}
    if duration:
        body["response_format"]["duration"] = f"{int(round(duration))}s"
    if task:
        body["generation_config"] = {"video_config": {"task": task}}
    if previous_id:
        body["previous_interaction_id"] = previous_id
    r = _req("POST", f"{base}/v1beta/interactions", key, body)
    t0 = time.time()
    while r.get("status") in ("in_progress", "running", "pending", "queued") and time.time() - t0 < 900:
        time.sleep(8)
        r = _req("GET", f"{base}/v1beta/interactions/{r['id']}", key)
    vid = None
    for st in r.get("steps", []) or []:
        if st.get("type") == "model_output":
            for c in st.get("content", []):
                if c.get("type") == "video":
                    vid = c
    vid = vid or r.get("output_video")
    if not vid:
        raise RuntimeError(f"Omni returned no video (status={r.get('status')}). Safety block or bad prompt? "
                           f"Simplify the prompt, keep it in English.")
    if vid.get("data"):
        return base64.b64decode(vid["data"]), r.get("id")
    uri = vid["uri"]
    data, _ = _req("GET", uri + ("&" if "?" in uri else "?") + "alt=media", key, raw=True)
    return data, r.get("id")


# ---------------------------------------------------------------------------
# engine side: export inputs, build jobs, cache, plates
# ---------------------------------------------------------------------------
def _scene_frame_png(scene, t, path, scale=2 / 3):
    from .render import render_frame
    render_frame(scene, t, scale=scale, hud=False, plates=False).write_to_png(path)
    return path


def _scene_clip(scene, t0, t1, path, fps=24, scale=2 / 3):
    from .render import render_frame, _as_array
    W, H = int(round(scene.W * scale)), int(round(scene.H * scale))
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                          "-pix_fmt", "yuv420p", "-an", path], stdin=subprocess.PIPE)
    n = int(round((t1 - t0) * fps))
    for i in range(n):
        surf = render_frame(scene, t0 + i / fps, scale=scale, hud=False, plates=False)
        p.stdin.write(np.ascontiguousarray(_as_array(surf)).tobytes())
    p.stdin.close(); p.wait()
    return path


def resolve_jobs(scene):
    jobs = []
    for j in scene.spec.get("omni") or []:
        j = dict(j)
        j["t0"] = scene.resolve_time(j["t"], 0.0, None)
        j["t1"] = scene.resolve_time(j["end"], 0.0, None) if "end" in j else j["t0"] + float(j.get("duration", 5))
        j.setdefault("task", "edit")
        j.setdefault("mode", "replace")
        j["out"] = os.path.join(scene.base_dir, "media", "omni", f"{j['id']}.mp4")
        jobs.append(j)
    return jobs


def _job_hash(scene, j):
    from .render import render_frame
    sig = [j.get(k) for k in ("task", "prompt", "resolution", "model")] + [round(j["t0"], 3), round(j["t1"], 3)]
    if j["task"] != "text":   # include what the engine shows (layout changes → regenerate)
        for tt in (j["t0"], (j["t0"] + j["t1"]) / 2, j["t1"] - 0.05):
            a = np.frombuffer(render_frame(scene, tt, scale=0.125, hud=False, bloom=0, plates=False).get_data(), np.uint8)
            sig.append(hashlib.md5(a.tobytes()).hexdigest()[:10])
    return hashlib.sha1(json.dumps(sig, ensure_ascii=False).encode()).hexdigest()[:16]


def plan(scene):
    rows = []
    for j in resolve_jobs(scene):
        span = j["t1"] - j["t0"]
        h = _job_hash(scene, j)
        side = j["out"] + ".json"
        fresh = os.path.exists(j["out"]) and os.path.exists(side) and json.load(open(side)).get("hash") == h
        rows.append((j, span, h, fresh))
    return rows


def run(spec_path, yes=False, only=None, provider="gemini", log=print):
    from .scene import load_spec
    scene = load_spec(spec_path)
    rows = plan(scene)
    if not rows:
        log("[omni] spec has no \"omni\" jobs"); return 0
    todo = [r for r in rows if not r[3] and (only is None or r[0]["id"] == only)]
    for j, span, h, fresh in rows:
        flag = "up-to-date" if fresh else ("TODO" if (only is None or j["id"] == only) else "skip")
        log(f"[omni] {j['id']:14s} {j['task']:10s} {j['t0']:6.2f}-{j['t1']:6.2f}s ({span:.1f}s) {j['mode']:7s} {flag}")
        log(f"        prompt: {j.get('prompt', '')[:110]}")
        if not 3.0 <= span <= 10.0:
            log(f"        ✗ span {span:.1f}s — Omni needs 3..10 s"); return 1
    if not todo:
        log("[omni] nothing to generate"); return 0
    if not yes:
        log(f"[omni] DRY RUN — {len(todo)} paid generation(s) with {MODEL}. Show this plan to the user, get their OK, "
            f"then re-run with --yes"); return 0
    key = get_key("GEMINI_API_KEY", scene.base_dir) if provider == "gemini" else None
    base = (scene.spec.get("omni_cfg") or {}).get("base_url", BASE)
    for j, span, h, _ in todo:
        work = os.path.join(scene.base_dir, "media", "omni", "_work", j["id"])
        os.makedirs(work, exist_ok=True)
        images, video, task = [], None, None
        prompt = j.get("prompt", "").strip()
        if j["task"] == "edit":
            video = _scene_clip(scene, j["t0"], j["t1"], os.path.join(work, "input.mp4")); task = "edit"
            prompt += GUARD_EDIT
        elif j["task"] == "first_last":
            images = [_scene_frame_png(scene, j["t0"], os.path.join(work, "first.png")),
                      _scene_frame_png(scene, j["t1"] - 0.04, os.path.join(work, "last.png"))]
            prompt = ("Use Image 1 as the exact first frame and Image 2 as the exact last frame. " + prompt + GUARD_GEN)
        elif j["task"] == "from_frame":
            images = [_scene_frame_png(scene, j["t0"], os.path.join(work, "first.png"))]; task = "image_to_video"
            prompt = "Start exactly from this image. " + prompt + GUARD_GEN
        else:
            task = "text_to_video"; prompt += GUARD_GEN
        log(f"[omni] generating {j['id']} ({provider}) …")
        t0 = time.time()
        if provider == "draft":
            data = _draft(scene, j, work, video, images)
            iid = "draft"
        else:
            data, iid = interact(prompt, key, images=images, video=video, task=task,
                                 resolution=j.get("resolution", "720p"),
                                 duration=None if j["task"] == "edit" else max(3, min(10, round(span))),
                                 model=j.get("model", MODEL), base=base, log=log)
        raw = os.path.join(work, "raw.mp4")
        open(raw, "wb").write(data)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-an", "-c:v", "copy", j["out"]], check=True)
        json.dump({"hash": h, "interaction_id": iid, "prompt": prompt, "task": j["task"], "provider": provider,
                   "t0": j["t0"], "t1": j["t1"], "secs": round(time.time() - t0, 1)},
                  open(j["out"] + ".json", "w"), ensure_ascii=False, indent=1)
        log(f"[omni] ✓ {j['out']}  ({time.time() - t0:.0f}s) — interaction {iid} (reuse with previous_interaction_id "
            f"for a follow-up edit)")
    return 0


def _draft(scene, j, work, video, images):
    """Offline stand-in: engine clip + noir 'photographic' grade, so the plate pipeline can be tested free."""
    src = video or _scene_clip(scene, j["t0"], j["t1"], os.path.join(work, "input.mp4"))
    out = os.path.join(work, "draft.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf",
                    "gblur=sigma=1.2,eq=contrast=1.25:brightness=0.02,noise=alls=14:allf=t,vignette=PI/4",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", out], check=True)
    return open(out, "rb").read()


def load_plates(scene):
    """Spec omni jobs whose clip exists → scene.plates (used by render_frame)."""
    plates, missing = [], []
    try:
        jobs = resolve_jobs(scene)
    except Exception:
        return [], []
    for j in jobs:
        if os.path.exists(j["out"]):
            plates.append({"id": j["id"], "src": j["out"], "t0": j["t0"], "t1": j["t1"], "mode": j["mode"],
                           "grade": j.get("grade", "none"), "fade": float(j.get("fade", 0.25)),
                           "blend": j.get("blend", "normal")})
        else:
            missing.append(j["id"])
    return plates, missing
