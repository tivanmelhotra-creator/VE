# CONTINUE_HERE — VE v20.1 three3d + three_anim (resume point after any sandbox reset)

## Restore in 1 command (replaces the old 3-step / 9-minute restore)
```bash
cd /home/user/webapp/sdk && python3 -m mograph.pro doctor --fix   # deps, binaries, fonts, media copy, tracking cache
                                                                 # from work/input/raw.mp4.v14pack (seconds), 67 tests
```
Then read `/home/user/webapp/AGENT_INDEX.md` (request → file → key).

## v20.1 — animation INSIDE 3D objects (mograph/three_anim.py) — see sdk/V14_3D_GUIDE.md §00.1
- `object_3d` + `"three_model":"hourglass_3d","anim":"hourglass_loop","flips_at":["w:…"]` (pour → flip → pour), `laptop_open`,
  `assemble`, `explode_at` on any model.  Test edit: projects/maghale/edit_v20_anim.json.  LOOK: `python3 -m mograph three3d-anim-sheet`
- v20 = mograph/three3d.py (procedural Three.js engine, review report V20_PR_REVIEW.md)
- next ideas: more stateful models (pen click-extend, trophy shine sweep), motion blur on the flip, sand grain sparkle

## v19 — Pro Editor (mograph/pro) — see sdk/PRO_EDITOR_GUIDE.md
- colour: `grade.look` (10 looks) / `grade.lut` (.cube) / wb / lift_gamma_gain / protect_skin / grain / halation /
  sharpen / letterbox  (pro/color.py, wired in v14/compositor.py: plate LUT + final finish)
- `{"type": "broll"}` full / pip / behind / split_top / split_bottom, 12 transitions, ken burns, chroma key, own grade
  (pro/broll.py, registered via v14/layers.PREPARE + DRAW)
- saber blade on tracked points: `start_anchor` / `end_anchor` / `start_offset` / `end_offset` / `angle` / `length`
- time anchors: plain numbers & `"e:"` = edited (mapped through retime, fixes the v18 "hex/na vanish" bug),
  `"s:"` = raw screen seconds, `"src:"` = seconds in the source file
- tools: `pro lint` · `pro timeline` · `pro render -j 2` (parallel, resumable, frame-exact) · `pro captions` · `pro edl`
  · `pro chapters` · `pro thumbnail` · `pro reformat --to 1:1` · `pro platform` · `pro looks` · `pro lut-export`
- test edit: projects/maghale/edit_v19_pro.json (QA clean, lint clean)
- merged: the other agent's v18.1 omni-polish patch (retime numeric anchors, subtitle 3.4 s cap, halo/frame_box edge
  guards, character-vs-subtitle guard, qa keys) — all 9 maghale edits QA clean + lint clean (pbr3d obj_tooth edge fixed)
- next ideas: lip sync from audio only (whisper word timings), character rigs from user PNG parts (puppet_arap),
  QA for optical-flow ghosting, Saber shared-noise speed-up

---
# (older) v14 Studio Depth notes

## Restore in 3 commands
```bash
bash /home/user/webapp/sdk/tools/setup_env.sh                       # pip deps + libegl + MODNet model
mkdir -p /home/user/webapp/projects/maghale/media && cp /home/user/webapp/work/input/raw.mp4 /home/user/webapp/projects/maghale/media/raw.mp4
cd /home/user/webapp/sdk && nohup python3 -u -m mograph.v14.track ../projects/maghale/media/raw.mp4 > /tmp/prep.log 2>&1 &   # ~9 min, cache
```
Then: `cd /home/user/webapp/projects/maghale && PYTHONPATH=/home/user/webapp/sdk python3 -m mograph.v14 frame edit_v14.json --t 1 3.6 8 12 15`

## Layout
- `sdk/` = mograph SDK (v13.9 + user font pack: 139 fonts, incl. YekanBakh, Morabba, Pelak, Doran, Abar, Pinar, Gohar…)
- `sdk/mograph/v14/` = NEW engine: `physics.py` `glyphs.py` (vector 3D type) `track.py` (MODNet + zero-lag pose)
  `compositor.py` (depth sandwich, true glass, light wrap/spill, bloom) `layers.py` `subs.py` `__main__.py` (CLI)
- `projects/maghale/edit_v14.json` = the edit of the user's raw video (words.json = transcript, edited time 16.02 s)
- `work/input/` = raw.mp4, ref_novadent.mp4, brief_v14.md (the user's inputs)

## v18 — Motion Kit (mograph/motion) — see sdk/MOTION_GUIDE.md
- FIRST after any reset: `cd sdk && python3 -m mograph.motion selftest` → 10/10 (glossary, easing, retime, flow, saber,
  saber_time, lipsync, character, bridges).  Also aefx 9/9 · aecomp 13/13 · v14 9/9 still pass.
- user's brief = sdk/mograph/motion/glossary.yaml (5 concepts) — `python3 -m mograph.motion resolve "…"` maps words → keys
- reference: Video Copilot Saber 1.0.40 (parameter list + 52 preset names read from the plugin binary; NOT in repo;
  rar password www.bitgraph.ir, needs 7zz ≥ 21).  Our saber.py is an own implementation of that parameter set.
- tests: sdk/tests/v18_motion.json (v5 scene) · projects/maghale/edit_v18_retime.json (v14 retime)
- next: saber on tracked hands (anchor hand_r → blade start/end), viseme from audio only (no transcript),
  more rigs (cutout from user PNG parts via puppet_arap), retime in v5 footage element

## v17 — AE-Comp (mograph/aecomp) — see sdk/AECOMP_GUIDE.md
- FIRST after any reset: `cd sdk && python3 -m mograph.aecomp selftest` → 12/12 (keyframes, ellipse, trim, fill, merge,
  merge_line, repeater, trim_anim, matte, generators, lower3rds, titles)
- spec: `{"type": "aecomp", "gen": NAME}` / `{"type": "aecomp", "lower_third": STYLE, "title", "role"}` / `{"type": "aecomp", "title_style": STYLE, "hero", "kicker"}` / `{"comp": json}`
  in v5–13 scenes (elements registry) and v14 (layers.DRAW).  Test: `python3 -m mograph frame tests/v17_aecomp.json --t 1`
- LOOK: `python3 -m mograph.aecomp sheet` · `lt-sheet --title …` · `title-sheet --title …` · gallery tab shape/lower_third (`tools/build_aefx_gallery.py --only aecomp`)
- dev only (reference packs NOT in repo; re-download to /home/user/refs if needed, password www.bitgraph.ir):
  `pip install py_aep shapely` → `tools/import_aep.py x.aep -o x.json` → `tools/validate_aecomp.py x.json preview.mp4 --sheet s.jpg`
  RAR5 needs 7-Zip ≥ 21: `curl -sL -o 7z.tar.xz https://github.com/ip7z/7zip/releases/download/24.08/7z2408-linux-x64.tar.xz
  && tar xf 7z.tar.xz 7zz` → `./7zz x -pwww.bitgraph.ir pack.rar`  (apt p7zip/unrar-free FAIL on RAR5)
  whole pack: `tools/batch_validate_aecomp.py <NM_Library_Shapes_AE> -o /tmp/b.jsonl` (resumable) → `--report /tmp/b.jsonl`
- fixed with the batch (v17.1): Repeater + operators below (Trim on every copy; Explosions_49 F3 0.21→0.97), exact
  Merge Paths outline (shapely), content("…").prop expression refs (Arcs_43 0.25→0.79), Twist strongest at centre
- measured numbers: `sdk/mograph/aecomp/presets/aecomp_stats.json`
- fidelity v17.1 (all 452 elements, numbers only): `sdk/tests/aecomp_validation_report.txt` → F3 median 0.916,
  397/452 ≥ 0.6; weakest = Slides 0.68, Blow_Ups 0.82 (see "worst" list there for the next fixes)
- next: AE effects on shape layers (turbulent displace / radial wipe), text animators on imported comps, more
  generators per category (titles from Titles Factory study: per-char delay, venetian blinds, minimax "outspace")

## v16 — AE-FX (mograph/aefx) — see sdk/AEFX_GUIDE.md
- FIRST after any reset: `cd sdk && python3 -m mograph.aefx selftest` → 10/10 (keyframes, prproj fit, libraries, selector,
  textanim, cursive, deepglow, pathblur, bridge)
- presets: aefx/presets/{motion_presets,text_presets,catalog}.json + CATALOG.md (measured).  Pick with
  `python3 -m mograph.aefx suggest …` / `sheet …`; apply with `"fx": [{"motion": KEY}]`, `"preset": KEY` (v14 titles),
  `post.aefx`, `source.cut_presets`.  Demo spec: sdk/tests/v16_aefx.json
- gallery: `python3 sdk/tools/build_aefx_gallery.py` → public/aefx/index.html (media git-ignored)
- ⚠ sandbox RAM: v14.track (MediaPipe+MODNet) takes ~55 % RAM → do NOT run builds/renders in parallel with it.

## v15.1 — Craft-Omni (audited merge of the v15.0 patch) — see sdk/V15_REVIEW.md + sdk/V14_GUIDE.md "v15.1 additions"
- FIRST after any reset: `cd sdk && python3 -m mograph.v14 selftest` → 10/10 pass (multiband, remix, ARAP, morph, Oklab,
  decontam, path text, varstroke, selectors)
- new spec keys: source.morph_cuts · grade.match · post.decontaminate · post.glass.fresnel · audio.multiband ·
  audio.music.file (bar-grid remix) · "selectors" on text layers · field bezier_path
- new layers: shatter_text · puppet_sticker · morph_shape · vegas_comet · god_rays   (v14/craftlayers.py, layers.py)
- visual test: projects/maghale/edit_v15_craft.json (QA clean)

## v14.8 — Studio Optical & Auto-Director — see sdk/V14_GUIDE.md "v14.8 additions"
- refractive glass (compositor._glass_optics) · 180° shutter motion blur (v14/motionblur.py) · bokeh dust (v14/bokeh.py,
  layer `ambient_bokeh` or post.bokeh_dust) · AUTO EDIT: `cd projects/maghale && PYTHONPATH=../../sdk python3 -m mograph.v14 auto-edit words.json media/raw.mp4 -o edit_auto.json`
- generated example: projects/maghale/edit_auto.json (QA clean) · patch: ve_v14_8.patch (repo root)

## v14.6 — pbr3d (PHOTOREAL scanned PBR 3D via bundled EffectCraft CLI) — see sdk/V14_3D_GUIDE.md §0 + sdk/CRAFT_REPOS_ANALYSIS.md
- `sdk/bin/effectcraft-cli`, `sdk/mograph/bin/deep-filter`, `sdk/mograph/models/modnet.onnx` are NOT in git (repo < 100 MB) →
  `setup_env.sh` / `python3 -m mograph.fetch_assets` downloads them (also auto on first use). v14.2 zip/mp4 → release `v14.2-deliverables`. Models: sdk/mograph/pbr_assets/models/
- test: `cd sdk && python3 -m mograph frame tests/v146_pbr3d.json --t 2` · footage test: projects/maghale/edit_v14_pbr3d.json
- demo: work/renders/demo_pbr3d_v14_6.mp4 · compare: work/renders/3d_compare_v14_6.jpg
- cache lives in ~/.cache/mograph_pbr3d (NEVER /tmp: tmpfs = RAM)

## v14.5 — real3d (true 3D textured models: soda_can, pizza_slice, growth_card) — see sdk/V14_3D_GUIDE.md §0
- reference: work/input/three_objects/ref.jpg · demo: work/renders/demo_real3d_green.mp4 (render script: sdk/tools/real3d_demo.py)

## v14.4 — 3D merged from the v13.9 3D branch + rewritten
- obj3d.py (extruded sprite / multiview), sculpt3d.py (+zraster numba z-buffer, MATERIALS, extrude, validate,
  fit_pose, comparison_sheet), sculpts/dental_implant.json, v14/object3d.py (`object_3d` layer)
- CLI: turntable3d · sculpt-check · sculpt-compare · img2mesh.   Guide: sdk/V14_3D_GUIDE.md
- test edit: projects/maghale/edit_v14_3d.json (main edit_v14.json untouched)

## v14.3
- DeepFilterNet3 denoise default (chmod +x sdk/mograph/bin/deep-filter after restore)
- ring monotonic spin-in; halo = 2 discs above head, text inside bright disc (center/arc), VFX
- docs: USER_TASTE_GUIDE.md, SKILL_TOOLBOX.md, SKILL_DEV_LOOP.md, TOOLS_INDEX.md (tools/build_tools_index.py)

## v14.2 — automatic editor brains — automatic editor brains
- denoise.py (RNNoise models in sdk/mograph/models/rnnoise, QC-tuned), palette.py (colours designed from the
  footage), stage.py (layout guard + scrim/grid/light), ring_fit (40 % coverage + facing fade), band_fit (disc),
  Doran font lock + 1.12 stretch, subtitle smart-merge, QA face-coverage check.  See sdk/V14_GUIDE.md §6.

## Status
- [x] physics, glyphs (tested: arcs, cylinder, extrude, mixed-font markup, non-1000-upem fonts fixed)
- [x] track, compositor, layers, subs, CLI — first frames render (~2 s/frame)
- [x] Doran theme, lockup, nested hept glass, tracked_tag on shoulders, env light, ink-aware gaps
- [x] audio.py (voice EDL + event SFX + lofi bed, −14 LUFS)
- [x] qa.py
- [ ] final render + before/after split showcase
- [x] docs V14_GUIDE.md, V14_AUDIT.md, bootloader updated
- render: `cd projects/maghale && bash ../../sdk/tools/render_chunks.sh edit_v14.json 4` (resumable; chunks in out/chunks)
