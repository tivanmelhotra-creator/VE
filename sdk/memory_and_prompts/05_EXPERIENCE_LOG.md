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
