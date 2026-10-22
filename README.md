# VE — mograph: automatic video editor & motion-graphics SDK (v19 Pro Editor)

> **Agents: read [`AGENT_INDEX.md`](AGENT_INDEX.md) first** (one page: request → file → key). After any reset:
> `cd sdk && python3 -m mograph.pro doctor --fix` → `ALL GOOD ✓` (53 selftests).

An agent-driven engine for editing **Persian talking-head footage** and making **vector motion graphics**,
designed to run inside a small sandbox (2 vCPU, ~1 GB RAM, no GPU).
The agent writes one JSON spec; the engine does the drawing, word-locked timing, tracking, depth matte,
3D objects, sound and QA.

## v21 — Style Bank & Creative Synthesis
**Copying and repetition are forbidden.** References in `sdk/style_bank/` are style studies (principles, never
templates). `python3 -m mograph.pro originality edit.json` audits every spec. The auto-director picks forms from the
meaning, using pure glow typography as the backbone. Subtitles come in 4 styles and auto-hide under lower thirds.
The layout guard keeps titles off the forehead. Fonts can be paired freely. See `sdk/style_bank/STYLE_BANK_PLAYBOOK.md`
and `sdk/USER_TASTE_GUIDE.md`.

## What's inside
| Path | Contents |
|---|---|
| `sdk/mograph/` | v5–v13 engine (vector scenes, typography, widgets, footage, tracking, audio, QA) |
| `sdk/mograph/v14/` | v14 "Studio Depth" footage engine: physics, vector 3D type, MODNet matte, glass/light compositor, subtitles |
| `sdk/mograph/v14/{remix,flow_cut,craftlayers,selftest}.py`, `sdk/mograph/{varstroke,autotrace,puppet_arap,shatter3d}.py` | **v15.1** Craft engines (bar-grid music remix, morph cuts, ARAP puppet, shape morph, shatter, tapered strokes, Oklab look match) — `python3 -m mograph.v14 selftest` |
| `sdk/mograph/aefx/` | **v16 AE-FX**: AE keyframes (speed/influence), 351 AE text animators (range/wiggly selectors, Persian-aware), 352 Premiere-style in/out/fx motion presets, deep_glow, path_blur, `"fx"` stack on any layer, measured catalogue + `suggest` — `python3 -m mograph.aefx selftest` · guide `sdk/AEFX_GUIDE.md` |
| `sdk/mograph/motion/` | **v18 Motion Kit**: glossary resolver (Persian/English → keys), unified easing (cubic-bezier / AE speed-influence / spring / steps + graph editor), retime (slow/fast motion, speed ramp, freeze, reverse, optical-flow in-betweens, audio remap), Saber-style light (51 presets, blade/path/shape/text/alpha cores, glow, flicker, smoke/fluid/energy distortion, lightning), rubber-hose characters (animation principles, on twos) and Preston-Blair lip sync — `python3 -m mograph.motion selftest` · guide `sdk/MOTION_GUIDE.md` |
| `sdk/mograph/pro/` | **v19 Pro Editor**: colour looks (10) + `.cube` LUTs + skin protect + grain/halation/letterbox · B-roll layer (cutaway / PiP / background replace / split screen / chroma key, 12 transitions, Ken Burns) · spec linter · parallel resumable render · captions SRT/VTT/ASS · EDL · chapters · thumbnail · reformat 1:1/4:5/16:9 · doctor (restore after reset in seconds) — `python3 -m mograph.pro selftest` · guide `sdk/PRO_EDITOR_GUIDE.md` |
| `sdk/mograph/aecomp/` | **v17 AE-Comp**: After-Effects shape-layer engine (trim, repeater, merge, zigzag, twist, round corners, pucker, offset, mattes, parenting, AE keyframes/loopOut/wiggle/controls) + 12 own shape elements + 4 box-free RTL lower thirds + 4 full-frame titles; `{"type": "aecomp", ...}` in v5–13 and v14 — `python3 -m mograph.aecomp selftest` · guide `sdk/AECOMP_GUIDE.md` |
| `sdk/mograph/pbr3d.py` | **v14.6** photoreal 3D: scanned CC0 PBR models rendered by the bundled EffectCraft CLI (HDRI + real shadow) |
| `sdk/mograph/real3d.py`, `sculpt3d.py`, `obj3d.py` | earlier procedural 3D engines (fallback) |
| `sdk/fonts/` | 139 Persian/Latin fonts (YekanBakh, Morabba, Doran, Pelak, Abar, Pinar, Gohar…) |
| `sdk/tests/`, `sdk/examples/`, `sdk/templates/` | test specs, example videos, reference templates |
| `projects/maghale/` | reference edit of a real video (`edit_v14.json`, `edit_v14_3d.json`, `edit_v14_pbr3d.json`) |
| `work/input/` | raw footage + reference inputs |
| `sdk/*.md` | docs — start with `sdk/00_START_HERE_AGENT_BOOTLOADER.md` |

## Quick start
```bash
bash sdk/tools/setup_env.sh                          # pip deps, MODNet, EffectCraft CLI, fonts

# vector / infographic engine (v5–v13)
cd sdk && python3 -m mograph frame tests/v11_showcase.json --t 1.5
python3 -m mograph render examples/demo_spec.json

# photoreal 3D (v14.6)
python3 -m mograph pbr3d-search food
python3 -m mograph frame tests/v146_pbr3d.json --t 2

# talking-head footage engine (v14)
mkdir -p ../projects/maghale/media && cp ../work/input/raw.mp4 ../projects/maghale/media/
python3 -m mograph.v14.track ../projects/maghale/media/raw.mp4      # ~9 min, cached
cd ../projects/maghale && PYTHONPATH=../../sdk python3 -m mograph.v14 frame edit_v14.json --t 1 8 15
bash ../../sdk/tools/render_chunks.sh edit_v14.json 4                # resumable final render
```

## Docs
- `sdk/MOTION_GUIDE.md` — **v18** Motion Kit: glossary, easing, retime, saber, character, lip sync
- `sdk/AECOMP_GUIDE.md` — **v17** AE-Comp: shape elements, lower thirds, comp JSON, measured timing rules
- `sdk/AEFX_GUIDE.md` — **v16** AE-FX: text/motion presets, glow, path blur, fx stacks, preset gallery (`public/aefx/`)
- `CONTINUE_HERE.md` — resume point / changelog v14.2 → v15.1
- `sdk/V15_REVIEW.md` — audit of the v15.0 patch: measured bugs and their fixes
- `sdk/V14_GUIDE.md`, `sdk/V14_3D_GUIDE.md`, `sdk/V14_AUDIT.md`
- `sdk/CRAFT_REPOS_ANALYSIS.md` — evaluation of storytold EffectCraft / VectorCraft / FilmCraft
- `sdk/USER_TASTE_GUIDE.md`, `sdk/SKILL_TOOLBOX.md`, `sdk/TOOLS_INDEX.md`

## Status (tested 2026-10-22 — v21, doctor 71/71 ✓)
- 66/66 modules import · 12 test specs + 4 examples render · v14 footage edit renders (~2 s/frame), QA clean
- pbr3d photoreal objects render standalone and on footage

## Notes
- **Repo kept < 100 MB.** Large binaries are NOT in git; they are downloaded once (sha256-verified, byte-identical)
  by `bash sdk/tools/setup_env.sh` → `python3 -m mograph.fetch_assets`, and also on first use automatically:
  | file | size | source (fallback: release `assets-mirror`) |
  |---|---|---|
  | `sdk/bin/effectcraft-cli` | 75 MB | storytold/effectcraft v0.11.1 release |
  | `sdk/mograph/bin/deep-filter` | 36 MB | Rikorose/DeepFilterNet v0.5.6 release |
  | `sdk/mograph/models/modnet.onnx` | 25 MB | huggingface Xenova/modnet |
  Check: `cd sdk && python3 -m mograph.fetch_assets --check`.
- Old v14.2 deliverables (`mograph_v14_2_sdk.zip`, `maghale_v14_2_final.mp4`, `showcase_v14_2.mp4`) moved to the
  GitHub Release `v14.2-deliverables` → `cd sdk && python3 -m mograph.fetch_assets --deliverables` (→ `work/deliver/`).
- API keys are never stored in the repo (`python3 -m mograph keys …` keeps them outside git).
- `src/` (Hono) only serves the AE-FX / AE-Comp preset gallery: `python3 sdk/tools/build_aefx_gallery.py && npm run build && npx wrangler pages dev dist` → `/aefx/`. Gallery previews (`public/aefx/media/`, ~8 MB) are git-ignored and regenerated on demand.
