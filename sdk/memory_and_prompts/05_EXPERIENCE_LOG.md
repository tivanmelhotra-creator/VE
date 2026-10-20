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
