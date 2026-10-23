# V22 GUIDE — Concat-inspired kit (MCP · RVM · offline voice · VAD · enhance · stabilize · SDF masks · rational time)

Source of the ideas: the analysis of `jub0t/concat` v0.2.8 (Rust NLE).  Nothing was copied: every module below is
an own Python implementation sized for **2 CPU cores / 1 GB RAM / no GPU**.

## 0. Size rule (repo < 100 MB)
All v22 models live in `~/.cache/mograph_models/` (override `MOGRAPH_MODELS`), are sha256-pinned in
`mograph/models_hub.py` and are fetched on first use or by `pro doctor --fix`.  Git grew by ~0.1 MB (code only).
Every feature has an offline fallback, so `MOGRAPH_OFFLINE=1` still passes `v22 selftest 10/10`.

| model | MB | used by | fallback |
|---|---|---|---|
| rvm | 15.0 | `source.matte_engine: "rvm"` | MODNet |
| esrgan | 4.9 | `pro enhance`, broll/source `enhance` | unsharp mask |
| u2netp | 4.6 | `pro cutout`, broll `cutout` | GrabCut |
| isnet (opt-in) | 178.6 | `cutout {"quality":"high"}` | u2netp |
| silero_vad | 2.3 | `mograph.vad`, voice timings | energy VAD |
| piper_fa (+ gyro opt-in) | 20.7 | voice provider `local` | draft beeps |

```bash
python3 -m mograph.models_hub --list        # or: python3 -m mograph.pro models
python3 -m mograph.v22 selftest             # 10/10 (also run by pro doctor → 81 tests in total)
```

## 1. MCP server — warm session for agents  (`mograph/mcp`)
```bash
cd sdk && python3 -m mograph.mcp            # stdio MCP (JSON-RPC 2.0), pure stdlib
python3 -m mograph.mcp --list
python3 -m mograph.mcp --call ve_lint_spec '{"spec":"projects/maghale/edit_v14.json"}'
```
Claude Code / Cursor: `{"mcpServers":{"ve":{"command":"python3","args":["-m","mograph.mcp"],"cwd":"/home/user/webapp/sdk"}}}`
Tools: `ve_doctor ve_resolve_glossary ve_lint_spec ve_qa_audit ve_check_originality ve_timeline ve_words
ve_preview_frame ve_render_sheet ve_spec_read ve_spec_patch ve_models ve_vad ve_cutout ve_enhance`.
The Renderer stays in memory (rebuilt only when the spec's mtime changes): a warm frame costs about 0.8 s instead of
a cold Python start plus a Renderer build.  `ve_spec_patch` = JSON merge patch; layers carrying an `"id"` merge by id,
`{"id":"x","_delete":true}` removes one, a backup goes to `.bak`, and the patch is reverted if lint reports errors.

## 2. RVM matte engine (temporal)  — `"source": {"matte_engine": "rvm"}`
`python3 -m mograph.v14.track media/raw.mp4 --engine rvm` → `<video>.v14/matte_rvm/`.
Measured on maghale (537 frames, 1080×1920): **about 110 s vs about 9 min with MODNet**, mean |Δ| against MODNet ≈ 1.5/255.
Recurrent state, so no 3-frame median is needed.  The committed MODNet pack stays the default (stabilize also reads it).

## 3. Offline Persian voice  — `"voice": {"provider": "local", "voice": "amir", "speed": 1.0}`
sherpa-onnx + Piper VITS int8 (≈ 160 MB RAM, ≈ 1 s per sentence).  Word times = VAD speech regions + letter weights,
so no word ever lands in a pause.  The transcript of a test sentence matched the input exactly.
Not voice cloning (Chatterbox needs > 1 GB): for overdub-style fixes keep using ElevenLabs.

## 4. VAD — Concat audit lesson H8/N3  (`mograph/vad.py`)
Transcriber word starts that sit inside a silence are snapped to the next speech onset (≤ 8 s).  Applied
automatically to whisper timings in `voice.py`.  CLI: `python3 -m mograph.vad raw.mp4 --words words.json --out fixed.json`
(on the maghale words.json: 0 words moved, as expected for a clean transcript).

## 5. Tiled Real-ESRGAN  (`pro/enhance.py`)
256² tiles + 16 px margin (288² window, ~60 MB peak), `mix` 0.6 keeps skin texture.  ≈ 1.3 s per tile on 2 cores →
for stills, logos and punch-in windows, not whole videos.
- `python3 -m mograph.pro enhance logo.png --factor 2 --mix 0.6`
- broll: `"enhance": {"factor": 2, "mix": 0.6}` (stills, cached as `*.enh2_60.png`)
- source: `"enhance": {"mix": 0.5, "punch_in": 1.3}` → only the visible crop, only while the camera zoom ≥ 1.3, disk cache.

## 6. Range-aware stabilisation  — `"source": {"stabilize": true}`
Only the used segments ± 1.5 s are analysed (360 px grey, LK + RANSAC similarity, **speaker masked out with the
matte**); the cache `<video>.v14/stab.json` is incremental.  The correction is folded into `Renderer.matrix`, so the
plate, matte, anchors and tracked layers stay locked together.  Options: `strength 0.85 · smooth 0.8 s · crop 1.05`.
`python3 -m mograph.pro stabilize edit.json` → maghale: shake 0.35 → 0.08 px (79 % less; tripod footage).

## 7. Product cutout  (`pro/cutout.py`)
`python3 -m mograph.pro cutout product.jpg` → `product.cut.png` (u2netp, edge decontamination).  broll: `"cutout": true`.

## 8. SDF masks + diagonal split  (`pro/masks.py`)
broll `"mask": {"shape": "linear|mirror|circle|rectangle|heart|star", "angle", "feather", "roundness", "invert"}` —
values may be `[from, to]` (animated iris / wipe).  New mode `"mode": "split_diag", "diag": {"angle": -14, "side": "top"}`
gives comparison frames (USER_TASTE law 3) without a flat box.  Coverage drift between 480p and 1920p = 0.

## 9. Rational time  (`pro/rtime.py`)
`pro render` computes every frame↔time conversion and chunk border with `Fraction`; ffmpeg gets `-r 30000/1001` for
29.97 footage.  Fixes the float off-by-one (33.4 s at 29.97 → frame 1001; `int()` gave 1000).

## Not done (and why)
- Voice cloning (Chatterbox / Pocket-TTS): model + runtime > 1 GB RAM → not viable in this sandbox.
- SlimSAM click-to-select: needs an interactive UI; the MCP `ve_cutout` covers the agent use case.
- Whisper inside the engine: `faster-whisper` stays optional; VAD snapping is already wired for when it is installed.

---
# v22.1 — fixes from the 77-second field report (`Untitled-2.md`), each verified against the real code

| report item | verdict | what changed | measured |
|---|---|---|---|
| §1.1 Haar fallback anatomy (no MediaPipe) | **real** | `track._prep_pose_offline_fallback`: Haar box = position only; head width, shoulder width and shoulder line come from the person-matte **silhouette** (calibrated vs MediaPipe: hair/hijab ≈ 1.81 × ear width, body ≈ 1.49 × shoulder width, shoulder line where the silhouette ≥ 1.85 × head); the report's clamps are kept as the no-matte fallback | head 256→247 px, shoulder 607→583, chest y 1423→1395 (MediaPipe → fallback, 2–4 %) |
| §1.2 guard throws chest/shoulder titles over the head | **real** | `stage.guard_mode`: default direction follows the anchor (chest/belly/neck/shoulders/hands → down, head → up); an explicit `"guard"` still wins | chest title: y 1285 → 527 (old, over the head) → 1329 (now, just below the chin) |
| §1.3 `fade_out` returns 1 at the end | **not in this repo** (`v14/elements/base.py` doesn't exist) | verified by rendering 6 layer types around their `out`: none brightens on exit | probe: every exit decays monotonically to 0 |
| §1.4 subtitles over `lower_third` | partly done in v21 (aecomp lower thirds + chest heroes) | `subs._in_lower_third`: ANY screen-placed layer at y ≥ 72 % and low PiP cards also auto-hide subtitles; `"subs_hide": true/false` overrides; full cutaways keep subtitles (normal editing) | selftest `lower_hide` |
| §2 matte median (3 imread + np.median) | **real** | raw-frame cache + exact closed-form `median3 = a+b+c−min−max` | 115 → 16 ms per frame |
| §2 compositor wide blurs / NumPy blend | **real** | `wide_blur`: σ ≥ 19 runs at half res (≤ 2/255 error); speaker composite via `cv2.blendLinear` | 671 → 498 ms/frame; output vs old mean 0.02/255 |
| §2 glyph path building | (profiled) | `_poly` vectorised (`.tolist()` instead of 2·n numpy scalar reads) | ~2× path building, −3…5 % frame |
| §2 subtitle RGBA cache | **measured: not worth it** | prototype gave 0 % hit rate on this clip (the jelly pop and the capsule spring keep the text moving almost the whole phrase); reverted to keep the code simple | — |
| §2 `three_stage.py` Chromium daemon | **not in this repo** (no such file) | — | — |
| §3.1 music-bed detector before breath cuts | **done** | `mograph/musicbed.py` (persistent peaks in the 20th-percentile spectrum + gap tonality); `auto-edit --cuts auto|force|off` → background music detected = continuous source, no lofi bed added | dry / room noise / mains hum → 0.0; held chord 0.6; piano bed 0.89 → no cuts |
| §3.2 bundle a small ASR model | **tested, rejected** | Whisper-base int8 via sherpa-onnx (154 MB) output on the Persian clip was unusable ("از از از …"); Whisper-small (640 MB) doesn't fit 1 GB RAM. Keep the external transcript + `mograph.vad` snapping | — |
| §3.3 static overlap check in lint | **done** | `pro lint` samples the real placement (`layers.pos` + `element_radius`, guard included) every 0.25 s without rendering → "spatial collision A × B t0–t1 (IoU)"; same `"group"` = exempt | 0.4 s for a spec, catches a chest×chest collision IoU 0.73 |
| §3.4 render size | **done** | `pro render --profile master|web|social` (+ `--x264-preset`), encoded once in the chunks | 3 s test: master 5.4 MB · web 2.8 MB (52 %) · social 2.2 MB (41 %), SSIM vs master 0.99 |
