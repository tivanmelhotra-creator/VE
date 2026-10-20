# VE — mograph: automatic video editor & motion-graphics SDK (v14.6)

An agent-driven engine for editing **Persian talking-head footage** and making **vector motion graphics**,
designed to run inside a small sandbox (2 vCPU, ~1 GB RAM, no GPU).
The agent writes one JSON spec; the engine does the drawing, word-locked timing, tracking, depth matte,
3D objects, sound and QA.

## What's inside
| Path | Contents |
|---|---|
| `sdk/mograph/` | v5–v13 engine (vector scenes, typography, widgets, footage, tracking, audio, QA) |
| `sdk/mograph/v14/` | v14 "Studio Depth" footage engine: physics, vector 3D type, MODNet matte, glass/light compositor, subtitles |
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
- `CONTINUE_HERE.md` — resume point / changelog v14.2 → v14.6
- `sdk/V14_GUIDE.md`, `sdk/V14_3D_GUIDE.md`, `sdk/V14_AUDIT.md`
- `sdk/CRAFT_REPOS_ANALYSIS.md` — evaluation of storytold EffectCraft / VectorCraft / FilmCraft
- `sdk/USER_TASTE_GUIDE.md`, `sdk/SKILL_TOOLBOX.md`, `sdk/TOOLS_INDEX.md`

## Status (tested 2026-10-20)
- 66/66 modules import · 12 test specs + 4 examples render · v14 footage edit renders (~2 s/frame), QA clean
- pbr3d photoreal objects render standalone and on footage

## Notes
- Large binaries are committed: `sdk/bin/effectcraft-cli` (75 MB, EffectCraft 0.11.1 headless),
  `sdk/mograph/bin/deep-filter` (DeepFilterNet3), ONNX models in `sdk/mograph/models/`.
- API keys are never stored in the repo (`python3 -m mograph keys …` keeps them outside git).
- The `src/`, `wrangler.jsonc`, `package.json` files are the unused Hono web template of the sandbox.
