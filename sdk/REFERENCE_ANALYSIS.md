# REFERENCE_ANALYSIS — the 9 reference videos → v11 techniques → templates

How to use: find the reference closest to the requested video, open its template, swap media (`src`), texts and
timing, then `qa → frame → render --workers 1`. Every technique below is one or two spec keys; no new code needed.
(Proxies of the references: `reference_videos/ref0N.mp4` in the webapp workspace; contact sheets in `sheets/`.)

| ref | look / genre | key techniques seen | v11 keys & elements | template |
|---|---|---|---|---|
| ref01 | talking head on grey studio sweep, giant words, photo cards | depth masking (words behind speaker), selective colour (keep orange), B&W + cyan-negative look, halftone hook, octagon/rounded photo cards, big stat, logo watermark, punch-in zooms | `footage` + `behind`, `selective`, `looks`, `halftone`, `zooms`/`punch`, `image_card shape:octagon`, `big_number` | `11_ref01_talking_head_depth_beauty` |
| ref02 | editor portfolio, teal paper, Raw vs Edited phones | two `device_frame`s (small raw + big edited), word behind the speaker inside a frame, light infographic (alert pulse, counter), 3D button, IG handle + outro | `backdrop paper`, `device_frame children`, `behind`, `glow_ring`, `counter`, `badge_button`, `handle_tag`, `social_outro`, `hud.subs.box` | `12_ref02_editor_showcase_raw_vs_edited` |
| ref03 | Storyboard vs finished motion | hand-drawn storyboard drawn on stroke by stroke beside the finished phone: notification badges, falling broken hearts, toggle, megaphone bubbles | `auto_trace` (sketch), `device_frame`, `emitter shape:heart`, `ui_toggle`, `ui_notification_stack`, `icon` | `13_ref03_storyboard_vs_motion` |
| ref04 | edit-process explainer | same clip through steps: raw → enhance audio → colour correction → text animation → SFX; segment timeline + red playhead | `footage grade_keys`, `audio_enhance`, `kinetic`, `nle_timeline mode:segments`, `pill`, `social_outro` | `14_ref04_editing_process_timeline` |
| ref05 | dark tiles, phone, layered timeline | coloured clip timeline under I-beam, **text tracked to hands**, speed ramp on the gesture, kinetic words behind the speaker, B&W ending | `backdrop tiles`, `trackers hand` + `follow`, `ramps`, `behind`, `looks mono`, `nle_timeline mode:clips` | `15_ref05_tiles_phone_clip_timeline_tracking` |
| ref06 | BEFORE/AFTER on purple smoke, neon | saber/edge glow around the speaker, neon blue words, depth-plate illustration, glowing brain with orbit, light streaks | `backdrop smoke/streaks`, `edge_glow`, `saber`, `glow`, `depth_plate`, `orbit_trail`, `kinetic` | `16_ref06_before_after_neon_saber` |
| ref07 | @ENSAN noir line-art essay | white strokes on black, podium boxes, growing bars, dashed callouts, pulsing AI eye, crowd grid, chain pan, karaoke subs | core `lineart` style: `box`, `bars`, `callout`, `icon idle:pulse`, `glow_ring`, `icon_grid`, `transition pan` | `17_ref07_noir_lineart_comparison_story` (+ 03/04) |
| ref08 | viral-formula wheel | octagon hub + 8 nodes rotating node by node, red progress arc, ghost step number, redacted keyword reveals, dark open-loop interludes with orbiting red dot, REC viewfinder | `radial_formula`, `big_number roll decor`, `redact_text`, `orbit_trail`, `viewfinder_hud`, `backdrop studio` | `18_ref08_radial_viral_formula` |
| ref09 | neon line-art storytelling | AI illustrations drawn on in source colours with glow, shaking icons under stress, puppet character, light rays, big central captions | `auto_trace color_mode:source` + `glow`, `shiver`, `sprite_rig`, `light_rays`, `depth_plate`, `gen` + `mograph prompts`, `hud.subs.box` | `19_ref09_neon_lineart_story_genimages` |

## Extra techniques requested by the user (not all visible in the references)
| technique | how |
|---|---|
| Motion tracking / point track | footage `trackers: {name: {kind: point, at:[u,v]}}` → any element `follow: "clip.name"` |
| Planar tracking (screen / box-art replacement) | `trackers: {scr: {kind: planar, quad: [...]}}` + `planar_insert follow: "clip.scr"` |
| Rotoscoping / object mask | footage `matte` (person, ONNX) or `roto: {mode: color, at: [u,v]}` |
| Edge glow / saber / flash | footage `edge_glow`; elements `saber`, `flash_pulse`; footage `flash`; scene `flashes` |
| Selective hue shift / colour keyframing | footage `hue_shift: [{t, from, to, within: person, back}]`, `selective`, `grade_keys` |
| Tracked callout box | `tracked_callout anchor: "clip.hand"` (line glued to the point, box keeps its spot / lags) |
| Depth masking (text behind speaker) | `behind: "clip"` on any element |
| Speed ramping / time remapping | footage `ramps`, `remap`, `freeze`, `beat_ramp` (audio follows the remap) |

## Asset generation for new videos
Elements with missing `src` + `"gen": {"prompt": ...}` → `python3 -m mograph prompts spec.json` writes
`<spec>.prompts.json` with image jobs (native image generator → jpg into the workspace), a music job
(`audio.music_gen`, BPM/duration derived from the spec), and a voice job. Sound effects come from `sound.py`
(event-locked scores for every v11 element).
