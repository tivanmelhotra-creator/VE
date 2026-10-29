# SKILL_TOOLBOX — what this project HAS, when to use it, and what is still MISSING

> Read this right after `00_START_HERE_AGENT_BOOTLOADER.md`. It is the curated map; the complete auto-generated
> inventory (every module, function, CLI, model, font) is `TOOLS_INDEX.md` — regenerate it with
> `python3 sdk/tools/build_tools_index.py` whenever you add or change a tool.
> Rule: **never write a new script before checking here and in TOOLS_INDEX.md** — most problems already have a
> tool. If you DO add one, register it here (table + "how to use") in the same commit.

## 0. Setup / survival (sandbox resets often)
| need | tool | command |
|---|---|---|
| restore deps after reset | `tools/setup_env.sh` | `bash sdk/tools/setup_env.sh` (pip + libegl + MODNet check) |
| media + caches | — | `mkdir -p projects/<p>/media && cp work/input/raw.mp4 projects/<p>/media/` then prep (below) |
| matte + pose cache | `mograph.v14.track` | `cd sdk && nohup python3 -u -m mograph.v14.track ../projects/<p>/media/raw.mp4 > /tmp/prep.log &` (~8 min / 18 s 1080p) |
| resumable final render | `tools/render_chunks.sh` | `cd projects/<p> && bash ../../sdk/tools/render_chunks.sh edit.json 4` (4-s chunks, OOM-safe on 1 GB) |
| regenerate inventory | `tools/build_tools_index.py` | `python3 sdk/tools/build_tools_index.py` |

## 1. The v14 pipeline (real talking-head footage)
| step | module | command / key | notes |
|---|---|---|---|
| transcript words | (external) `audio_transcribe` elevenlabs → `words.json` | fix mis-heard words by hand | word anchors `w:word`, `w:word#2`, `we:word+0.2` |
| edit spec | `edit_v14.json` | see `V14_GUIDE.md` §1–3 | layers anchored to tracked body points |
| check timeline | `mograph.v14 words` | `python3 -m mograph.v14 words edit.json` | edited-time word list + cuts |
| QA | `mograph.v14.qa` | `python3 -m mograph.v14 qa edit.json` | face coverage, edges, short layers, flashing subs |
| look | `mograph.v14 sheet / frame` | `… sheet edit.json --t 2 4 8 --out x.jpg` | ALWAYS look at frames before rendering |
| render | `mograph.v14 render` / `render_chunks.sh` | `--preview` = half res | |
| before/after | `mograph.v14.showcase` | `python3 -m mograph.v14.showcase after.mp4 raw.mp4 out.mp4 --segments "a-b,c-d"` | |

## 2. Automatic "editor brains" (they decide so a weaker model can't get it wrong)
| brain | module | decides | override in spec |
|---|---|---|---|
| **Noise cancelling** | `v14/denoise.py` + `mograph/bin/deep-filter` (DeepFilterNet3 static binary) | measures SNR → attenuation 18–42 dB, HF restore EQ, room tone; RNNoise fallback | `"audio": {"denoise": "off" / 0.6 / {"model": "sh"}}` · CLI `python3 -m mograph.v14.denoise media/raw.mp4 --out clean.wav` |
| **Colour palette** | `v14/palette.py` | samples background + clothes → accent from curated cinematic families by colour theory, alert, ink, stage, grid; WCAG ≥ 4.5 | `"meta": {"palette": "auto" / {"accent": "#..."}}` · CLI `python3 -m mograph.v14.palette media/raw.mp4` (+ swatch png) |
| **Font system** | `v14/glyphs.py` THEMES / FONT_LOCK / STRETCH | one superfamily; Doran = Doran only, +12 % width | `"meta": {"theme": "doran"}` |
| **Ring solver** | `v14/layers.ring_fit` | text on a 3D ring ≤ 40 % of circumference, facing fade, monotonic spin-in | `max_copies`, `max_radius`, `spin_in` |
| **Band fit** | `v14/layers.band_fit` | disc texts inside the band | `band_span` |
| **Halo (v14.3 anatomy)** | `v14/layers.draw_halo` + `HALO_DEFAULT` | outer translucent disc + inner BRIGHT disc, centred ABOVE the head (head_top, offset −0.30); VFX: shockwave, neon flicker, breathing glow, light sweep, rim bloom/spill | `radius`, `inner_ratio`, `outer_alpha`, `sweep`, `shock`, `pulse` |
| **Halo text** | `v14/layers._halo_text` | text INSIDE the bright disc: `text_mode` center (stacked lockup) / arc (bent along the edge) / auto; optional `arc_text` line on the top edge; ≤ 35 % behind the head; first hero glyph peeks in front | `text_mode`, `arc_text`, `arc_r`, `max_hidden`, `peek_glyphs` |
| **Layout guard** | `v14/stage.guard` | no element over the face; safe frame | `"guard": false / "down" / "up" / "side"` |
| **Stage light** | `v14/stage.draw_stage` | scrim + world-locked grid + coloured light behind elements (colour = hero word) | `"stage": false / {...}` |
| **Layered 2D Lip-Sync** | `motion/layered_lipsync.py` | topological eye-pair & mouth-contour detector (`detect_face_anatomy`), auto base-mouth erasure (`erase_base_mouth`), 16-viseme grid extractor (`extract_viseme_grid_sprites`), 3-layer Z-depth + dual-track renderer (`render_layered_lipsync`), and `qa_mouth_placement` guard | CLI `python3 -m mograph.motion layered-lipsync detect/render` |
| **Subtitles** | `v14/subs.py` | phrase grouping + smart merge of < 0.45 s phrases, jelly capsule, karaoke | `subtitles.max_words`, `hide`, `emph` |

## 3. Visual building blocks (layers) — see V14_GUIDE.md §3 for params
`kinetic_title` · `orbit_ring` · `disc` · `glass_shape` (nested badge) · `halo` · `frame_box` · `tracked_tag` ·
`strike` · `counter_badge` · `flash` · `progress` · `sweep_light` · `subtitles`.

## 4. QC tools (use them — your own eyes/ears are not in the loop otherwise)
| what | how |
|---|---|
| frames | `sheet` / `frame` CLI then Read the jpg |
| motion over time | render 6–8 frames 0.07 s apart into one image (see EXPERIENCE_LOG "ring jerk") |
| audio listening QC | upload wav (UploadFileWrapper) → `analyze_media_content` with `gemini-3.1-pro-preview`; ask ONE file per call (multi-file calls confuse it) and back it with numbers (`denoise.measure`, band energies) — the listener is inconsistent |
| loudness | `ffmpeg -i x.mp4 -af ebur128 -f null - 2>&1 \| grep I:` (target −14 LUFS) |

## 5. Known gaps / backlog (add yours; remove when done)
- [ ] Matte prep is slow (~1.5 fps on CPU) — could run MODNet at 384 px or reuse every 2nd frame with optical-flow warp.
- [ ] Glass/stage are per-frame CPU (≈2 s/frame) — a numba/vectorised glass blur would halve render time.
- [x] ~~No automatic "edit director"~~ → v14.8 `python3 -m mograph.v14 auto-edit words.json raw.mp4 -o edit_auto.json`
      (director.py). Next: semantic scoring with an LLM pass, B-roll / object picking beyond the bundled assets_3d.
- [ ] Ring text only supports a horizontal cylinder; tilted orbit around shoulders not implemented.
- [ ] The music bed is procedural (beat.py) — a licensed-music picker would sound better.
- [ ] DeepFilterNet on extreme noise still leaves a slightly thin voice; a voice-restoration model (e.g. resemble-enhance)
      could add body back if a light CPU build is found.

## 6. How to add a tool (checklist — do all of it in ONE commit)
1. Module docstring: purpose, method, CLI usage, spec keys, lessons.  2. Put it in `sdk/mograph/v14/` (or `tools/`).
3. Wire it into the pipeline with a sensible automatic default (weak agents never call optional tools).
4. Add a row here (§1–4) and run `python3 sdk/tools/build_tools_index.py`.
5. Log what triggered it in `memory_and_prompts/05_EXPERIENCE_LOG.md` (see `SKILL_DEV_LOOP.md`).

## v14.4 — 3D objects (see V14_3D_GUIDE.md)
| Tool | File | What | Key params |
|---|---|---|---|
| **image_to_3d** | `sam3d_bridge.py` → `obj3d.py` | any RGBA image as a solid turning object | `geometry` slab/badge/cushion/multiview/sphere/capsule, `depth`, `round`, `bulge`, `motion` |
| **sculpt3d** | `sculpt3d.py` + `zraster.py` | procedural part-tree model (z-buffer, Gouraud) | `preset`, `tree` (dict / json), `motion`, `open_dur` |
| **object_3d** (v14 layer) | `v14/object3d.py` | 3D object glued to a tracked point on footage | `anchor`, `offset`, `size` (head widths), `turn_on` |
| **sam3d_object** | `sam3d_bridge.py` | `.ply`/`.obj` point-cloud render | `src`, `scale`, `yaw_speed` |
| turntable3d / sculpt-check / sculpt-compare / img2mesh | `cli.py` | 3D QA sheet · strict gate · IoU + camera match · image→ply | |
