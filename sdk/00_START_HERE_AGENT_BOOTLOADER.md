# 🧊 v20 PROCEDURAL THREE.JS 3D ENGINE & FAST PIPELINE — true 360° procedural 3D + 6–8× faster renders → read `V14_3D_GUIDE.md`
> **Never use flat 2D sprite slab extrusion (`obj3d.py` / `extract3d`) for turning 3D objects!** Use `mograph.three3d`:
> `{"type": "object_3d", "three_model": "hourglass_3d" | "microblading_pen_3d" | "emerald_tree_3d" | "star_trophy_3d", "motion": "spin_360", "explode": 0.0}`
> Or author custom `.js` Three.js generator functions in `mograph/three_models.js` (`"three_script": "three_models.js:buildHourglass3D"`).
> CLI: `python3 -m mograph three3d-list` · `python3 -m mograph three3d-sheet hourglass_3d --out sheet.png` · `python3 -m mograph.three3d selftest` (7/7).

# 🎞 v19 PRO EDITOR — colour · B-roll · delivery → read `PRO_EDITOR_GUIDE.md` (and `../AGENT_INDEX.md`: request → key)
> After ANY reset: `python3 -m mograph.pro doctor --fix` (deps, binaries, fonts, footage, tracking cache in seconds, 60 tests).
> `"grade": {"look": "teal_orange" | "lut": "x.cube", "grain": .2}` · `{"type": "broll", "mode": "full|pip|behind|split_top", "in": "whip_left"}`
> `python3 -m mograph.pro lint edit.json` before every render · `python3 -m mograph.pro render edit.json -j 2` (parallel, resumable)
> captions / edl / chapters / thumbnail / reformat --to 1:1 / platform → `python3 -m mograph.pro --help`

# 🎬 v18 MOTION KIT — easing · retime · Saber light · cartoon characters · lip sync → read `MOTION_GUIDE.md`
> FIRST map the user's words: `python3 -m mograph.motion resolve "اسلوموشن با نئون روی متن"` → canonical keys + params.
> `{"type": "saber", "preset": "neon", "core": "text", …}` (51 Saber presets: `saber sheet` → LOOK) · `"fx": [{"saber": …}]`
> `"source": {"retime": [{"op": "speed_ramp"|"slow_motion"|"fast_motion"|"freeze_frame"|"reverse", …}]}` (v14, optical flow)
> `"ease"` everywhere now takes `cubic-bezier(…)`, `[x1,y1,x2,y2]`, `{"type": "elastic"…}`, Persian names (`ease graph`).
> `{"type": "character", "rig": "bean", "actions": [...], "lip_sync": {...}, "timing": "twos"}` · `{"type": "mouth"}`
> `python3 -m mograph.motion selftest` must say 10/10.

# 🔷 v17 AE-Comp — After-Effects SHAPE-LAYER engine + box-free lower thirds → read `AECOMP_GUIDE.md`
> 12 own shape elements (burst, confetti, arcs, arrow trace, slides, wipes, loops…) + 4 RTL lower thirds + 4 full-frame titles (`"title_style"`) (zero-box,
> Doran) via `{"type": "aecomp", "gen": "burst"}` / `{"type": "aecomp", "lower_third": "line_reveal", "title": …}` in
> v5–13 scenes AND v14 edits. Full AE shape semantics (trim, repeater, merge, zigzag, twist, mattes, AE keyframes).
> LOOK first: `python3 -m mograph.aecomp sheet` · `lt-sheet --title "…"` · `title-sheet --title "…"` · `python3 -m mograph.aecomp selftest` = 11/11.

# 🎛 v16 AE-FX — After-Effects-grade presets & effects for EVERY layer → read `AEFX_GUIDE.md`
> 351 AE text animators (`"preset": "atp_23"` on v14 titles, Persian-aware) · 352 in/out/fx motion presets
> (`"fx": [{"motion": "scale_10_in"}]` on any v14 layer or v5–13 element) · `deep_glow` · `path_blur` · cut presets.
> Choose from MEASURED data: `python3 -m mograph.aefx suggest punchy impact --dir in` → `sheet` → LOOK → apply.
> `python3 -m mograph.aefx selftest` must say 10/10.

# 🧠 00_START_HERE — Motion Creator Brain SDK **v14 Studio Depth**

> ## 🚀 FOR ANY REAL TALKING-HEAD FOOTAGE → USE `mograph.v14` (read `V14_GUIDE.md` first, then `V14_AUDIT.md`)
> v14 = physics springs + vector 3D typography + zero-lag offline tracking + MODNet depth matte + true glass +
> light wrap/spill + event-locked sound. One JSON spec (`projects/maghale/edit_v14.json` is the reference edit).
> ```bash
> bash tools/setup_env.sh
> PYTHONPATH=sdk python3 -m mograph.v14 prep|words|qa|sheet|frame|render  edit.json
> bash tools/render_chunks.sh edit.json        # resumable final render
> python3 -m mograph.v14.showcase after.mp4 raw.mp4 showcase.mp4 --segments "a-b,c-d"
> ```
> The v5–v13 engine below is still valid for pure vector / infographic videos (no footage).
> ⚠ Work ONLY inside /home/user/webapp (sandbox resets wipe everything else). Commit after every step.

---

# 🧠 00_START_HERE — Motion Creator Brain SDK **v13.5 Visual-DNA & Zero-Box Edition** (`mograph`)

> 📸 **v14.6 PHOTOREAL 3D (use this first for real objects):** `pbr3d` element / v14 `object_3d` with `"model"` — scanned CC0 PBR models (Poly Haven: `python3 -m mograph pbr3d-search food` → `pbr3d-fetch <id>`) or any .glb/.gltf, rendered by the bundled EffectCraft CLI (`sdk/bin/effectcraft-cli`) with HDRI lighting + real cast shadow. Read `V14_3D_GUIDE.md` §0 and `CRAFT_REPOS_ANALYSIS.md`.
> 🧊 **v14.4 3D:** real turning 3D objects → read `V14_3D_GUIDE.md` (`image_to_3d`, `sculpt3d`, v14 `object_3d`, `turntable3d` / `sculpt-check` / `sculpt-compare`).

> 🧭 **v14.3 FIRST READS:** `USER_TASTE_GUIDE.md` (how to work with this user) → `SKILL_TOOLBOX.md` (every tool, what exists/missing) → `SKILL_DEV_LOOP.md` (mandatory: log lessons & extend the SDK while editing) → `V14_GUIDE.md`.

> 🚨 **CRITICAL FIRST STEP FOR ANY NEW SESSION — READ `USER_TASTE_AND_DNA_LAWS.md`, `CREATIVE_REELS_PLAYBOOK.md` & `VISUAL_DNA_ARCHITECTURE.md`!**
> Before designing or rendering any video or reel for the User, you **MUST** obey the User's permanent aesthetic constitution in **`USER_TASTE_AND_DNA_LAWS.md`**:
> 1. **STRICT ZERO-BOX LAW:** NEVER place titles, counters (`+۸ میلیون`), brand logos, or stickers inside flat 2D rectangular boxes or pills! Rectangular shapes are ONLY permitted if they are **(a) true 3D tactile blocks** (`grade_a_3d.png`, `tactile_3d`, `iso_cube`) or **(b) game-style segmented loading bars (`@element("hud_loading_bar")`)**.
> 2. **BEHIND-SPEAKER COORDINATE GRID (`"behind": "clip"`):** Never put a box behind text for contrast; use `behind: "clip"` so `reelfx.py` transforms the entire background behind the segmented speaker into a color-graded studio coordinate grid (`_draw_dna_grid_environment`) with floating 3D multi-font typography and 3D props!
> 3. **FULL-FRAME $f/2.8$ SPEAKER DEFOCUS (`"dof_blur"` on `footage`):** For hero warnings/statements (`نکته خیلی مهم!`), smoothly blur the entire speaker frame (`dof_blur`) and stage giant 3D multi-font typography across the center of the full screen!
> 4. **TACTILE 3D ASSETS & 3D PERSPECTIVE SPIN (`mograph/assets_3d/` & `extract3d`):** Use the 14 bundled 3D studio sprites in `mograph/assets_3d/` (or generate new 2×2 3D sheets on `#000000` black background via `image_gen:generate_image` and slice them in 1 second with `python3 -m mograph extract3d sheet.jpg --names a,b,c,d --grid 2x2`).

---


**To the AI agent:** if this package was uploaded to you, **don't write drawing code from scratch.**
In v5 you write the scenario as **data (a JSON spec)**. The engine handles drawing, timing, camera, subtitles, sound, QA, and parallel rendering.
That way all your reasoning goes into **directing and the scenario**, not into the coordinates of lines.

---

## ⚡ The first 60 seconds
```bash
unzip motion_creator_brain_sdk_v11.zip -d sdk && cd sdk
pip install pycairo fonttools opencv-contrib-python-headless numpy pillow onnxruntime mediapipe shapely   # if missing (shapely: exact Merge Paths; optional)
apt-get install -y libegl1   # only for hand/pose tracking (mediapipe)
python3 -m mograph qa examples/demo_spec.json        # must say "QA clean ✓"
python3 -m mograph sheet examples/demo_spec.json     # look at the output image
python3 -m mograph qa examples/demo_v5_spec.json     # 56 s showcase: all 6 widgets, morph, thread, font roles
```
Then read these three files: `SPEC_REFERENCE.md` (all elements and keys), `DIRECTOR_WORKFLOW.md` (how to think through the scenario), and `catalog/icons_catalog.png` (the icons that exist).

---

## 🔁 The mandatory working loop (fast → slow)
| Step | Command | Time | Purpose |
|---|---|---|---|
| 1. Plan | write a Beat Sheet (see `DIRECTOR_WORKFLOW.md`) | — | reason through the story before any code |
| 2. Spec | `project/spec.json` | — | turn the plan into data |
| 3. QA | `python3 -m mograph qa spec.json` | **< 1 s** | glyph ▯, overlaps, out-of-frame, broken word anchors |
| 4. Single frame | `python3 -m mograph frame spec.json --t 12.4` | **< 1 s** | check one exact moment |
| 5. Contact sheet | `python3 -m mograph sheet spec.json` | ~3 s | overview of every chapter |
| 6. Preview | `python3 -m mograph preview spec.json` | ~10 s per 30 s | half resolution, 15 fps, for motion and timing |
| 7. Final | `python3 -m mograph render spec.json` | ~1.2× real-time | 1080×1920, 30 fps, audio −14 LUFS |
| 8. Review | `python3 -m mograph review final.mp4 --out spec.json` | ~5 s | checks the REAL video: black/frozen frames, LUFS, peaks, silence + frame sheet (runs automatically after render) |
| Partial fix | `render spec.json --from 35 --to 52` | | re-render one chapter only |
| VO timing | `python3 -m mograph align vo.wav --text script.txt` | | whisper word timestamps snapped to your script spelling → exact `words` subtitles |

> **Rule:** do the final render only once. Every fix is checked with steps 3–6. A single frame costs about 25 ms.
> Use a new output filename every time (the CLI adds a timestamp automatically) so the player doesn't show a cached version.

---

## ✂️ New in v12 — Studio Cut: RAW footage → finished reel (details: `SPEC_REFERENCE.md` → "v12")
Workflow for a raw talking-head take (see `projects/raw01/laptop_reel.json`):
1. Transcribe with word timestamps (`audio_transcribe` elevenlabs_scribe_v2 or `mograph align`) → `words.json`.
2. Top-level `"edit": {"src", "words", "transitions", "emph", "enhance"}` → jump cuts, shot-change transitions,
   captions in edited time. Check with `python3 -m mograph cut spec.json`.
3. One `footage` with `"edit": true, "audio": true, "audio_enhance": true, "auto_grade": {...}, "skin": {...}` +
   `frame` keys (virtual camera: push-ins on key words).
4. `hud.subs.style: "reel"` captions; motion kit `title_card`, `logo_chip`, `route_arc`, `price_tag`, `grade_scale`,
   `sticker`, `cta_arrow`, `light_leak`, `progress_bar`, `vignette_pulse` timed with `w:word` anchors.
5. Music: `audio.music` (real track) or `audio.beat` (procedural, drop on the big cut).
6. `qa → frame → render --workers 1`, then get an external review and iterate.

## 🏛 New in v11 — Studio Pantheon: real-footage editing + tracking + AI-image engines
(details: `SPEC_REFERENCE.md` → "v11 Studio Pantheon additions"; per-reference recipes: `REFERENCE_ANALYSIS.md`)
- **`footage`** = a whole edit suite in one element: speed ramp / time remap / freeze / beat ramp, animated grade
  (`grade_keys`), selective colour, **hue shift keyframes**, halftone, looks, flash, matte + background swap,
  **edge glow (saber rim)**, virtual camera (`frame`, `reframe: face`, punch-ins, zooms, beat pump, shake), clip audio + enhance.
- **Tracking:** `"trackers": {"hand": {"kind": "hand", "point": "index_tip"}}` inside a footage, then on ANY element
  `"follow": "clip.hand"` → motion tracking / point track. Kinds: point box planar hand pose face color.
  `planar_insert` = screen / box-art replacement. `tracked_callout` = leader line glued to a moving point.
- **Depth masking:** `"behind": "clip"` on any element → it passes behind the speaker (auto matte).
- **Energy:** `saber`, `flash_pulse`, `glow_ring`, `light_rays`, `orbit_trail`, `emitter`; generic keys `glow`, `shadow`,
  `shiver`; scene `shake`, `beat_pump`, `flashes`; caption slab `hud.subs.box`.
- **Editor-reel kit:** `device_frame` (container with `children`, authored full-screen and scaled in), `backdrop`
  (paper/tiles/smoke/streaks/studio), `nle_timeline`, `badge_button`, `handle_tag`, `social_outro`, `image_card`,
  `redact_text`, `radial_formula`.
- **AI images → motion:** `auto_trace` (line art → live strokes, source colours), `depth_plate` (one image → 2.5D
  parallax), `sprite_rig` (parts sheet → puppet). Put `"gen": {"prompt": …}` on elements whose file is missing and run
  `python3 -m mograph prompts spec.json` → list of image/music/voice jobs for YOUR platform generators.
- **Templates 11–19** = one recipe per reference video (ref01…ref09). Start from the closest one, swap `src`/text.
- Workflow for a new footage video: `mograph track spec.json` (caches + debug sheet) → `qa` → `frame` → `render --workers 1`.

## 🔺 New in v9 — Studio Apex (details: `SPEC_REFERENCE.md` → "v9 Studio Apex additions")
- **No `uharfbuzz` needed:** the system `libharfbuzz.so.0` is driven through ctypes → full vector text. Check: `python3 -c "from mograph import text; print(text.HB_BACKEND)"`.
  Only `pip install pycairo` is required if cairo is missing.
- **Style `promo`** for kinetic promo / SaaS-UI videos: radial spotlights, bokeh, 3D/outline/knockout type, kashida stretch, whip blur, Dutch tilt.
- **Widgets:** `liquid_blob iso_cube starburst prompt_box timeline_ui bezier_graph viewfinder_hud filmstrip sine_wave` — reach for them before drawing by hand.
- **Real speaker:** `talking_head` (offline .onnx matte) — put the climax word in `behind` so it sits behind the head; `pip_at` shrinks the speaker to a circle.
- **Clean narration:** `voice.max_pause: 0.3`, `voice.dedupe_fillers: true` → dead air and «مثلاً… مثلاً» disappear, every anchor follows.
- **Music sync:** `mograph analyze ref.mp4 --into spec.json` then `"t": "b:8"`. Aspect: `meta.aspect: "16:9"`.
- Showcase: `examples/v9_apex_showcase.json` → `demo_v9_apex.mp4`.

## 🌅 New in v8 — Studio Horizon (details: `SPEC_REFERENCE.md` → "v8 additions")
- **One spec, many styles:** per-chapter `style` / `palette` / `bg` / `fonts` / `sound` / `hud`. Bridge worlds with
  `"transition": {"type": "portal", "from": "<id>"}` (iris or `"shape": "slab"`) or `wipe` (each side keeps its own paper).
- **Swiss data widgets:** `waffle_grid`, `swiss_bars`, `versus_table` (grid-aware, cut-out type, contrast-safe).
- **Kinetic:** `mode: "masonry"` (bento rows locked to one width), `mode: "highlight_sweep"` (marker slab + inverted glyphs);
  `"roll": true` on `big_number` / `counter` = odometer digits.
- **2.5D parallax:** `"z"` on any element (−0.35 background … +0.25 foreground) — depth from existing camera moves.
- **Audio-reactive:** `"idle": "vo_pulse"`, element `waveform`, `scene.vo_env(t)`.
- Showcase: `examples/v8_trilogy_master.json` (41.7 s, lineart → swiss → kinetic → lineart finale, real voice):
  `python3 -m mograph voice examples/v8_trilogy_master.json && python3 -m mograph render examples/v8_trilogy_master.voiced.json`
- Low-RAM sandboxes (≈1 GB): `render … --workers 1` (≈ 12 fps at 1080×1920).

## 🛡 v7.1 — text can no longer be clipped or invisible
- All masks are built from the **real glyph outlines** (`text.mask_rect`), so dots/kasra/tails are never cut.
- HUD colours are chosen from the **real pixels behind them** (WCAG) → no white words on light backgrounds.
- `mograph qa` now reports `glyph clipped` and `low contrast` — treat both as bugs to fix before rendering.
- Writing a new element? use `mask_rect` for any text clip and `contrast.ensure` for any text colour you invent.

## 🆕 New in v7 — three styles, synced sound, guaranteed fonts, keys file
- **Pick a style first** (`meta.style`), each is a separate system with its own defaults:
  | style | look | elements | sound palette |
  |---|---|---|---|
  | `lineart` (default) | black, white line icons, bloom/grain | all v5 widgets, icons, morph | `paper` |
  | `swiss` | paper background, accent colour, 6×12 grid, hard wipes | `block` `headline` `rule` `big_number` `index` `disc` + `col`/`row` | `ui` |
  | `kinetic` | pure typography synced to the narration | `kinetic` (mode slam / stack / scale / scroll / type) | `punchy` |
  Colours by token: `@accent @ink @paper @mute` from `meta.palette` (red, blue, yellow, mono, noir, ink, blood).
- **Sound = event-locked.** Every SFX is placed on the exact frame of the visual event (digit change,
  stamp contact, line draw-on speed, wipe edge), panned by x position. Check with
  `python3 -m mograph cues spec.json` → list of `time  kind  reason`. Never place SFX by hand.
- **Fonts:** only `fonts/` are used (roles; `meta.fonts` overrides per style). QA warns if a character
  would fall back. Never rely on a system default font.
- **Keys:** the user puts keys in a file (`mograph_keys.txt`) and uploads it → `mograph keys import FILE --yes`
  (also auto-detected in uploads / AI Drive). See `INTEGRATIONS.md §0`. Never ask for a key in the chat.
- **Voice from outside:** `voice.provider: "file"` = use narration made by any TTS tool your platform has
  (+ its word timestamps). Same retiming and word lock as ElevenLabs. See `INTEGRATIONS.md §1b`.
- Examples: `examples/v7_act1_lineart.json`, `v7_act2_swiss.json`, `v7_act3_kinetic.json` (real voice).

## 🎙 New in v6 — real voice + realistic shots (optional, read `INTEGRATIONS.md` only when needed)
- **Narrator:** add `"voice": {"provider": "elevenlabs" | "gemini" | "draft", ...}` → `mograph voice spec.json` →
  render `spec.voiced.json`. Word-exact subtitles, everything retimed to the real voice, cached per line.
- **Realistic shots:** `"omni": [{"id","t","end","task":"edit","prompt"}]` → `mograph omni spec.json` (dry run =
  show the plan to the user) → `--yes`. Engine frames go in, a realistic plate comes back; HUD/subs stay vector.
- **Footage:** element `video` (blend screen, noir grade).
- **Keys:** `mograph keys`. Missing key → command exits 3 with a ready Persian message: **send it to the user and wait.**
- Workflow: iterate with `provider: draft` (free) → approve timing → real voice → Omni last (paid, per-shot).

## 🚀 v5 — think in sequences, not primitives
1. **Widgets** (`quote_rig flow_loop laser_scan invoice strata scale_rig`): one element = a whole 4–8 s sequence with its own internal beats and SFX. Reach for a widget first; build from primitives only when nothing fits.
2. **Morph**: `{"type":"morph","chain":["drop","money_core","question"],"at":[2.4,4.6]}` — one object transforms instead of cut/replace.
3. **Value tracks**: one counter changes value in place (`tracks.value`) with a punch. Never stack two counters.
4. **Auto-cascade `out` + `group`**: leader lines follow their target out; beats leave together.
5. **Font roles**: `"font":"title|document|handwritten|nastaliq|counter|..."` — typography carries tone (legal = document, twist = nastaliq, margin note = handwritten).
6. **Thread transition**: `{"type":"thread","from":"<id>","to":"<id>"}` — a glowing line physically wires chapter N to N+1.
7. **Post-FX**: `meta.vignette / grain / slam_pulse` — cinematic finish at ~0 extra cost.
8. **Strict QA + `review`**: typo'd keys, orphan callouts, cross-fade overlaps, dead stretches, and post-render video/audio checks.

## 🧩 Key v4 concepts (still valid)
1. **Word anchors:** timing is locked to narration words: `"t": "w:انجام"`, `"t": "we:دلار+0.2"`, `"out": "end-0.5"`.
   If the voice-over timing changes, you only change the subtitles and **every animation stays in sync automatically.**
2. **Chapters with local coordinates:** each chapter is its own 1080×1920 screen. `transition: pan` lays chapters out on one large canvas and the camera slides along it (the @ENSAN.BESHAM style, no black cut between chapters).
3. **Fully vector text:** HarfBuzz + Persian font set (Peyda, Kalameh, Pelak, Doran, Aria, DigiGhasedak, Miremad nastaliq, Vazirmatn fallback). Sharp under camera zoom, correct bidi for Persian, English, digits, and `%`/`@`, plus automatic fallback for `✓ ✕ →`. QA reports any missing glyph **before** rendering.
4. **Exact Trim Path:** paths are measured at their true length, so drawing speed is uniform across all icons.
5. **Automatic sound:** every element has a default SFX (icon → pen stroke, stamp → thud, callout → tick, grid → ticks, counter → roll). Narration (`audio.vo`) automatically ducks the music.
6. **Growable library:** 80 icons with Persian tags (`python3 -m mograph icons دارو`); adding one: `icons.add_icon(..., save=True)` or `import_svg_icon(path, name, save=True)`.
7. **Plugins:** if you need a new element, write it in a `.py` file with `@element("name")` and put it in `"plugins": ["my.py"]` in the spec. No engine edits needed.

---

## 🎨 Style 1: monochrome noir vector (@ENSAN.BESHAM)
- Pure black background, white `#FFFFFF` for active elements, gray `#555` for inactive ones, details in silver.
- HUD: angled 45° header `پرونده: …` at the top right, chapter dots at `y=82%`, word-by-word karaoke subtitles at `y=88.5%` (all automatic, via `hud`).
- Don't clear the screen: use `dim_at` to push old elements into the background, and `pan` between chapters.
- The default content area is `y ≈ 180 … 1500`. Below that is HUD territory, and QA warns if you go there.

## 🎬 Style 2: 3D hybrid with the speaker video
Still lives in `legacy_v3/pipelines/render_3d_hybrid_commercial_v3.py` plus the rules in `memory_and_prompts/03_…`.
In v4 you can use the `image` element (transparent PNG) with `tracks` for 2.5D motion, and stack layers on top of the speaker.

---

## 📈 Evolution protocol
At the end of every session:
1. Save new icons with `save=True` in `mograph/icons.json`.
2. Put new elements in `mograph/elements.py` (or keep them as plugins under `plugins/`).
3. Write lessons learned in `memory_and_prompts/04_CUMULATIVE_EVOLUTION_LOG.md`.
4. Run `python3 tools/package.py v11` to build a new zip (`motion_creator_brain_sdk_vX.zip`) for the user.
