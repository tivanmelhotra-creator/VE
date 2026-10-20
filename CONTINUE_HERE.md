# CONTINUE_HERE — v14 Studio Depth (resume point after any sandbox reset)

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

## v14.6 — pbr3d (PHOTOREAL scanned PBR 3D via bundled EffectCraft CLI) — see sdk/V14_3D_GUIDE.md §0 + sdk/CRAFT_REPOS_ANALYSIS.md
- `sdk/bin/effectcraft-cli` is NOT in git (75 MB) → `setup_env.sh` downloads it. Models: sdk/mograph/pbr_assets/models/
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
