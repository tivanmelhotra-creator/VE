# Motion Creator Brain SDK v11 Studio Pantheon — `mograph`

Declarative engine for producing Persian vector motion graphics (@ENSAN.BESHAM style) for an LLM agent.
The agent writes the scenario as JSON; the engine handles drawing, word-locked timing, camera, karaoke subtitles, sound, QA, and parallel rendering.

## Quick start
```bash
pip install pycairo fonttools opencv-contrib-python-headless numpy pillow onnxruntime mediapipe
python3 -m mograph qa      examples/demo_spec.json   # under 1 s
python3 -m mograph sheet   examples/demo_spec.json   # contact sheet
python3 -m mograph preview examples/demo_spec.json   # fast preview
python3 -m mograph render  examples/demo_spec.json   # final 1080x1920@30 + audio
python3 -m mograph render  examples/demo_v5_spec.json   # 56 s v5 showcase → auto review
python3 -m mograph voice   spec.json --provider draft  # narrator (elevenlabs|gemini|draft) → spec.voiced.json
python3 -m mograph cues    spec.json                   # v7: readable sound score (every SFX + why)
python3 -m mograph keys import mograph_keys.txt --yes  # v7: keys from a file, never via chat
python3 -m mograph join out/a.mp4 out/b.mp4 --out final.mp4   # v7: join acts of different styles
python3 -m mograph omni    spec.json                   # realistic shots (dry run; --yes to generate)
python3 -m mograph voice   examples/v8_trilogy_master.json   # v8 showcase: one spec, 3 styles, real voice
python3 -m mograph render  examples/v8_trilogy_master.voiced.json --workers 1
python3 -m mograph track   templates/15_ref05_tiles_phone_clip_timeline_tracking.json   # v11: trackers + debug sheet
python3 -m mograph prompts templates/19_ref09_neon_lineart_story_genimages.json       # v11: image/music/voice jobs
python3 -m mograph frame   templates/16_ref06_before_after_neon_saber.json --t 2       # v11 footage + saber
python3 tools/package.py v11                            # build the zip for the next session
```

## New in v11 Studio Pantheon
- **Real-footage editing:** `footage` element — speed ramps / time remap / freeze, grade keyframes, selective colour,
  hue-shift keys, halftone, looks, flash, matte + bg swap, edge glow, virtual camera, clip audio + enhance.
- **Tracking:** point / box / planar / hand / pose / face / colour trackers (cached) + `follow` on any element,
  `planar_insert`, `tracked_callout`. **Depth masking:** `behind: "clip"`.
- **Energy FX:** saber, flash_pulse, glow_ring, light_rays, orbit_trail, emitter; keys glow / shadow / shiver; scene
  shake / beat_pump / flashes.
- **Editor-reel widgets:** device_frame (container), backdrop, nle_timeline, badge_button, handle_tag, social_outro,
  image_card, redact_text, radial_formula.
- **AI images → motion:** auto_trace, depth_plate, sprite_rig + `mograph prompts`.
- **Templates 11–19** recreate the 9 reference reels — see `REFERENCE_ANALYSIS.md`.

## New in v9 Studio Apex
- **Zero-wheel vector shaping:** `libharfbuzz.so.0` via ctypes (`mograph/hbc.py`) + fontTools outlines → exact vector Persian text even without `uharfbuzz`.
- **Typography:** `text_style` = `outline` (+`stroke_reveal`) / `extrude_3d` / `knockout`, live **kashida** stretch (`"kashida": {"px":260,"word":"کش"}`).
- **4th style `promo`** (crimson / silver / cyber / emerald): `bg_mode` radial_spotlight · studio_vignette · crimson_vivid, `ambient` bokeh · warp_stars · grid,
  camera roll `[x,y,zoom,angle]`, `enter: whip_blur`, transition `whip_pan`.
- **9 widgets:** `liquid_blob` (knockout inversion), `iso_cube`, `starburst`, `prompt_box`, `timeline_ui`, `bezier_graph`, `viewfinder_hud`, `filmstrip`, `sine_wave`.
- **Voice surgery:** `voice.max_pause`, `dedupe_fillers`, `line_cut_ranges` with 12 ms raised-cosine splices → `_cuts` → talking-head punch-in.
- **Speaker matting:** bundled `mograph/models/portrait_seg.onnx` (462 KB) → `talking_head` 3-layer Z-sandwich (climax text BEHIND the speaker, rim light, PIP morph).
- **Spectral VO carve** (−11 dB 300–3400 Hz, −2 dB elsewhere) · **`mograph analyze ref.mp4`** (cuts, palette, LUFS, BPM, beats) + `"t": "b:N"` · **`meta.aspect`** 9:16 / 16:9 / 1:1 / 4:5.
- Showcase: `examples/v9_apex_showcase.json` (49.7 s, lineart → swiss → kinetic → promo → swiss → promo, real voice) → `demo_v9_apex.mp4`.
```bash
python3 -m mograph voice  examples/v9_apex_showcase.json          # real VO (file provider) + pause clamp + dedupe
python3 -m mograph render examples/v9_apex_showcase.voiced.json --workers 1
python3 -m mograph analyze reference.mp4 -o blueprint.json --into my_spec.json
python3 tools/package.py v9
```

## Structure
| Path | Contents |
|---|---|
| `00_START_HERE_AGENT_BOOTLOADER.md` | entry point for the agent |
| `SPEC_REFERENCE.md` | every key and element |
| `INTEGRATIONS.md` | v6: ElevenLabs / Gemini TTS voice, Gemini Omni realistic shots, video element, keys |
| `mograph/hbc.py`, `apex.py`, `promo.py` | v9: ctypes HarfBuzz · backdrops / ambient / whip / aspect · promo + UI widgets |
| `mograph/edit.py`, `matte.py`, `analyze.py` | v9: disfluency editor · onnx/chroma/grabcut matte · reference analyzer + beats |
| `mograph/models/portrait_seg.onnx` | v9: MediaPipe Selfie Segmentation (Apache-2.0, offline) |
| `tests/v9_*.json` | v9 widget / UI / viewfinder / talking-head test specs (QA clean) |
| `mograph/horizon.py` | v8: waffle_grid, swiss_bars, versus_table, waveform, odometer roll, vo ring |
| `examples/v8_trilogy_master.json` | v8: 41.7 s, lineart → swiss → kinetic → lineart in ONE spec (portal / wipe / slab), all v8 features |
| `tests/v8_*.json` | v8 widget / kinetic / parallax test specs (QA clean) |
| `mograph/contrast.py` | v7.1: WCAG contrast + background sampling (scene-aware HUD colours) |
| `tests/stress_v71.json` | v7.1: worst-case glyphs (پ چ ژ ی ِ) + wrong palette on light bg — must stay QA clean |
| `mograph/styles.py`, `typo.py`, `sound.py` | v7: style presets + palettes, Swiss grid elements + kinetic typography, event-locked stereo SFX |
| `examples/v7_act*.json` | v7: line-art / Swiss / kinetic acts with real narration (`voice_src/`) |
| `DIRECTOR_WORKFLOW.md` | three-pass Director → Designer → Engineer workflow + checklist |
| `mograph/` | engine: core, geometry (+morph), text (+font roles), icons(.json), elements, **widgets**, hud, scene (+thread), render (+post-FX), audio, qa (+review), align, **keys, voice, omni, media**, cli, catalog |
| `examples/demo_spec.json` | 3-chapter sample (من و شما / گاوصندوق / انسولین) |
| `examples/demo_v5_spec.json` | 7-chapter 56 s showcase: 6 widgets, morph chains, value track, thread transitions, font roles, post-FX |
| `examples/widgets/` | minimal one-screen spec for each widget pair (copy-paste starting points) |
| `plugins/` | sample plugin (speech_bubble) |
| `catalog/icons_catalog.png` | catalog of 80 icons |
| `fonts/` | 24 Persian fonts by role (Peyda, Kalameh, Pelak, Doran, Aria, DigiGhasedak, Delbar, Miremad, Vazirmatn …) |
| `legacy_v3/`, `reference_v3/` | v3 code (3D hybrid pipeline still lives here) |
| `memory_and_prompts/` | master prompts + evolution log |

## Performance (2-core / 1 GB sandbox)
- Single frame 1080×1920: ~17–25 ms incl. bloom + vignette + grain (post-FX 144 → 23 ms in v5) (v3 was ~64 ms at 720×1280)
- QA for a 28 s video: 0.2 s · Preview: ~10 s · Final: ~35 s (with audio)
