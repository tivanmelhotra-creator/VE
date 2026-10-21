# 05_EXPERIENCE_LOG — v14 era (newest at the bottom). Format: see SKILL_DEV_LOOP.md

## 2026-10-20 — v13.9 could not reproduce the @novadent reference
- Trigger: user brief v14 (audit why fine details were lost).
- Root cause: raster text, laggy online tracking, no depth matte, fake glass, no light interaction.
- Fix: v14 engine (vector 3D glyphs, offline zero-lag pose, MODNet matte depth sandwich, true glass, emission→spill).
- Lesson: details live in the compositor (depth, light, glass), not in the layer list.

## 2026-10-20 — fonts rendered 2× too big / digits reversed / cursive joins breaking
- Root cause: outlines not normalised by each font's unitsPerEm; neutral LTR runs shaped RTL; per-glyph pivots.
- Fix: upem normalisation, `_is_neutral_ltr`, word-pivot motion. Lesson: test every new font at 2 sizes + digits.

## 2026-10-20 — "font pairing is random / bad"
- Fix: THEMES = one superfamily, contrast via weight × size × colour; user confirmed Doran ≈ reference.
- Later (v14.2): user: "only Doran, a bit wider" → FONT_LOCK strips other families + STRETCH 1.12.
- Lesson: never mix families unless the user asks; one family, many weights.

## 2026-10-20 — heptagon looked like a solid glass blob
- Fix: glassbadge.py nested anatomy (outline → ~7 px clean gap → inset glass body). Lesson: zoom into the reference.

## 2026-10-20 — render killed mid-way (OOM on 1 GB sandbox)
- Fix: bounded lru caches, x264 `-threads 2 rc-lookahead=10`, chunked renderer. Also `.replace(".mp4")` matched
  twice in `c02.mp4.tmp.mp4` → use `os.path.splitext`.

## 2026-10-20 — orbit text collided with itself ("last letter in the first")
- Root cause: no relation between text length and ring circumference; on a cylinder the two halves converge at
  the silhouettes.
- Fix: `ring_fit` (≤ 40 % coverage, shrink → grow radius), facing fade (|cos| alpha). ONE phrase, solver repeats.

## 2026-10-20 — disc texts outside the band, ugly bright background behind elements, elements over the mouth
- Fix: `band_fit`; stage.py (scrim + world-locked grid + coloured light, user's own "grid" taste); layout guard
  (face ellipse + safe frame, smoothed); QA face-coverage check.

## 2026-10-20 — colours copied from the reference didn't suit the new plate
- User: the goal is better LOGIC, not parroting the reference.
- Fix: palette.py designs colours from the footage (curated cinematic families, avoid clothing hues, WCAG).
  Warm park → teal accent + red alert. Lesson: every visual constant that depends on the plate must be measured.

## 2026-10-20 — v14.3: "noise cancelling doesn't work, the river is still in my head"
- Root cause: RNNoise (2018 models) at partial wet mix (chosen to avoid artefacts) leaves non-stationary water noise.
- Fix: DeepFilterNet3 static musl binary (36 MB, model embedded, CPU 3.7 s for 18 s) in `sdk/mograph/bin/`,
  default engine; attenuation cap from SNR; presence/air EQ (measured −2…−3 dB HF after DFN); −62 dB room tone.
  Measured: noise floor −41 → −78 dB, speech level −28 → −29 dB. Listener: "no water noise anywhere", 8/10 reel.
- Lesson: a weak model can't be rescued by mixing; find the strongest light-weight OSS model. Verify with numbers.

## 2026-10-20 — v14.3: ring entrance "moves, goes back, moves again"
- Root cause: angle = spin·t − (1 − spring)·140 → spring overshoot (1.32) reverses angular velocity.
- Fix: angle = spin·t + boost·τ·(1 − e^(−t/τ)) (strictly monotonic), scale uses "smooth" spring, softer jelly.
- Verified: 8-frame strip at 0.07 s. Lesson: never feed an overshooting spring into a position that must not reverse.

## 2026-10-20 — v14.3: halo text "outside the circle is strange"
- User idea: bigger halo, «نقطهٔ شروع» inside it, part of «شروع» hidden behind her head except «ش».
- Fix: `_halo_text` fits text into the disc chord; occlusion rule (≤ 35 % of the hero height behind the head);
  "peek" = first hero glyph re-drawn in the front pass. Lesson: text that interacts with depth reads as 3D.

## 2026-10-20 — meta: user wants the project to document itself
- Added SKILL_TOOLBOX.md (curated map + backlog), TOOLS_INDEX.md (auto inventory via tools/build_tools_index.py),
  SKILL_DEV_LOOP.md (mandatory improve-while-editing loop), USER_TASTE_GUIDE.md (how to work with this user).

## 2026-10-20 — v14.3b: halo reference image (two discs above the head)
- User ref: small BRIGHT red disc + bigger TRANSLUCENT disc, both a bit ABOVE the head, text centred in the bright
  disc («اصلاً / سخت / نیست» lockup), head occludes the bottom, soft VFX. Off-centre text must bend along the edge.
- Fix: halo rewritten — HALO_DEFAULT (anchor head_top, offset −0.30, inner 0.74, outer_alpha 0.30), text_mode
  center/arc/auto, arc_text, VFX (shock ring, flicker, breathing glow, light sweep, rim bloom/spill).
- Lesson: when the user sends a reference crop, list its ANATOMY (layers, alpha, position vs. anchors, where the text
  sits, effects) as a checklist and encode each item as a default — not as per-spec numbers.


## 2026-10-20 — v14.7: Offline Sandbox Resilience, Doran ZWNJ Fix, Chest-Hero 3D Guard & 7x Compositor Speedup
- Trigger: Auditing VE (v14.6) in an offline sandbox on `projects/maghale/edit_v14.json` and `edit_v14_3d.json`, plus merging `FilmCraft` + `EffectCraft` (`craft_bridge.py`).
- Root causes found & fixed:
  1. `v14/glyphs.py` hard-imported `uharfbuzz` instead of falling back to `mograph/hbc.py` (`ctypes.CDLL("libharfbuzz.so.0")`).
  2. `Doran-Regular.ttf` broke medial cursive joins (`م` and `ی`) whenever a word contained `\u200c` (ZWNJ, e.g. `می‌خونید`, `می‌نویسید`). Fixed `_shape()` in `v14/glyphs.py` to split on `\u200c`, shape each sub-word with full cursive joining, and place them with a `0.04 em` half-space.
  3. `v14/track.py` lacked an offline fallback when `mediapipe` was absent, and needed anatomical left/right ordering (`eye_l.x > eye_r.x`, `sh_l.x > sh_r.x`) + `[-12°, +12°]` roll clamping so titles never invert.
  4. `v14/stage.py` (`guard`) only checked `face_zone`, allowing `side` 3D objects (`object_3d`) at chest height to overlap centered chest hero widgets (`disc`, `glass_shape`). Added chest-hero collision avoidance to `guard()`.
  5. `v14/compositor.py` allocated 22 full-res float32 arrays and ran `sigma=38..60` blurs at half-res; optimized light transport to quarter-res (`270x480`) + 256-entry `cv2.LUT` + `uint16` alpha blend (`7x` faster).
  6. `zraster.py` and `pbr3d.py` upgraded with vectorized NumPy triangle bbox rasterizers (`_raster_vec_np`, `55ms` without `numba`) and an offline pure-Python glTF PBR + HDRI software fallback (`_render_gltf_software_fallback`).
  7. `v14/subs.py` pre-pop word alpha raised from `0.22` to `0.72` for full phrase legibility inside dark glass capsules.
- Verified: `qa` clean, 6-frame sheets `/tmp/ve_v14_6frames_grid_fixed.jpg` & `/tmp/ve_v14_3d_guard_verified.jpg`, full renders `maghale_v14_7_studio_depth_master.mp4` & `maghale_v14_7_before_after_showcase.mp4`.


## 2026-10-21 — v14.8: Studio Optical (refractive glass, shutter motion blur, bokeh dust) & Auto-Director
- Built on v14.7 (c7eb7a0). 4 upgrades from the co-engineering brief:
  1. `compositor.glass()` → `_glass_optics`: distance-to-rim band, inward normal from the blurred shape mask, cv2.remap
     lens bend (4 px) + R/B split (1.5 px) + top-left caustic. Cost ≈3–6 ms per glass element (crop-local only).
  2. `motionblur.py` + `glyphs.draw_block(mblur=…)`: per-glyph displacement over a 180° shutter (motion evaluated at
     t − 1/60 s through a `_ShiftT` frame proxy) → line-kernel streak on a premultiplied offscreen ROI. Lesson: never
     re-evaluate `pos()` for the earlier instant — the layout guard keeps per-frame smoothing state; only `t` shifts.
     `object3d` refactored into `_motion_state` (pure) + `_blur_vector` (yaw rim speed + drop/fall + spring scale).
  3. `bokeh.py`: depth-sandwich dust — BACK slot after the back pass (speaker occludes it), FRONT slot before bloom,
     parallax relative to the t=0 camera matrix (0.7× / 1.3×), front discs faded over `stage.face_zone`.
  4. `director.py` (`auto-edit` CLI): breath cuts, linguistic word scoring (stop/verb penalties, numbers, emphasis
     lexicon, topic repetition, phrase-final + slow-speech stress), greedy 2.5–3.5 s beats, depth rotation, camera
     keys with punch-in on cuts, grade keys, subtitles.hide, and a QA auto-fix loop (`qa.qa_warnings`).
- Verified: QA clean on edit_v14 / edit_v14_3d / edit_auto (edit_v14_pbr3d keeps its pre-existing obj_tooth edge
  warning from v14.7). A/B frames v14.7 vs v14.8 at the same t: streaks on drop_bounce / swing_3d only, glass rims refract.


## 2026-10-21 — v15.0 Craft-Omni Master: Absorbing 18 Hidden EffectCraft & FilmCraft Engines into mograph
- Trigger: Deep line-by-line source audit of `storytold/effectcraft` and `storytold/filmcraft` (`CRAFT_EXTRACTION_BLUEPRINT_AND_RANK1_PROMPT_V15.md`).
- Implemented across `sdk/mograph/` and `sdk/mograph/v14/`:
  1. **True Height-Field Snell Refractive Glass + Chromatic Aberration + Fresnel Bevel Specular (`v14/compositor.py` `glass`)**: Ported from `effectcraft/crates/effects/src/stylize3.rs`. Rasterizes the shape mask in ROI, computes a smooth bevel height field $H$ and Sobel surface normal $N = (-\partial_x H, -\partial_y H)$, warps the frosted backdrop via `cv2.remap` with 1.6px Red/Blue chromatic dispersion and bevel specular highlight.
  2. **RotoBrush 2.0 Foreground Edge Color Decontamination (`v14/compositor.py` `decontaminate_foreground_edge`)**: Ported from `effectcraft/crates/track/src/roto/refine.rs`. Strips background color spill from semi-transparent hair/hijab/shoulder boundary pixels ($0.03 < \alpha < 0.94$) before applying light wrap/spill.
  3. **Oklab 3-Range Perceptual Color Match with <= 8° Skin-Tone Guard (`v14/palette.py` `match_reference_grade`)**: Ported from `filmcraft/crates/render/src/color_match.rs` (`oklab`, shadows/midtones/highlights smoothstep weights, x0.2 skin-tone down-weighting, and <= 8° skin hue shift clamp).
  4. **Automatic Beat-Sheet Director (`v14/director.py` & CLI `python3 -m mograph.v14 auto-edit`)**: Resolves Backlog #3 in `SKILL_TOOLBOX.md`. Excises dead pauses (> 0.34s), segments transcript into semantic beats, scores Persian hero keywords, alternates layer depth (`kinetic_title` -> `orbit_ring` -> `frame_box` -> `disc` -> `halo` -> `glass_shape`), populates `subtitles.hide`, and outputs a `QA clean ✓` spec.
  5. **4-Band Linkwitz-Riley Multiband Compressor + Vocal Booth IR Convolution Reverb + DeHum + Soft Clipper (`v14/audio.py`)**: Ported from `filmcraft/crates/audio-dsp/src/effects/{multiband.rs, reverbs.rs, restoration.rs}`.
  6. **Beat-Similarity Matrix Automatic Music Remix (`v14/remix.py`)**: Ported from `filmcraft/crates/audio-dsp/src/remix.rs`. Retimes any music bed to exact `R.duration` by splicing at the highest-similarity bar boundary with a beat-locked raised-cosine crossfade.
  7. **Bidirectional Optical-Flow Morph Cut, Gaussian/L1 Stabilizer & Planar Homography Tracker (`v14/flow_cut.py`)**: Ported from `filmcraft/crates/render/src/flow.rs` and `effectcraft/crates/track/src/{stabilize.rs, klt.rs}`.
  8. **Variable-Width Tapered & Wave Brush Strokes + Per-Vertex Variable Mask Feather (`mograph/varstroke.py` & `v14/layers.py` `draw_strike`, `draw_vegas_comet`, `draw_god_rays`)**: Ported from `effectcraft/crates/path/src/{varstroke.rs, feather.rs}` and `crates/effects/src/{generate2.rs, generate3.rs}`.
  9. **Sub-Pixel Marching-Squares + Douglas-Peucker + Newton Cubic-Bezier Auto-Trace & Belongie Shape-Context + 2D Procrustes Morph (`mograph/autotrace.py`)**: Ported from `effectcraft/crates/path/src/{trace.rs, fit.rs, interp.rs}`.
  10. **SIGGRAPH 2005 As-Rigid-As-Possible (`ARAP`) 2D Puppet Mesh Solver (`mograph/puppet_arap.py`) & 3D Extruded Shatter Physics (`mograph/shatter3d.py`)**: Ported from `effectcraft/crates/effects/src/puppet/arap.rs` and `sim2.rs`.
  11. **Composable AE Range & Wiggly Text Selectors + Arbitrary Bezier Path Text (`v14/glyphs.py` `eval_selector` & `Field("bezier_path")`)**: Ported from `effectcraft/crates/text/src/{selectors.rs, path_text.rs}`.
- Verified: All 20 modules pass `py_compile`, `auto-edit` produces `QA clean ✓` on `maghale`, and 6-panel visual audit sheet `/tmp/v15_craft_omni_audit_sheet.jpg` verified via `view_file`.
