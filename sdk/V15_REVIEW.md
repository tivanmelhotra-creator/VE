# V15 REVIEW — audit of `ve_v15_0_craft_omni_master.patch` and the v15.1 fixes

Reviewer verdict: **the patch had good ideas and clean structure, but it could not be merged as delivered.**
Of the 11 engines, 4 had mathematical bugs (measured, not suspected), 5 were dead code that no spec could reach,
and the patch was built on an old base, so it silently reverted the v14.8 release. v15.1 fixes all of this.
Every claim below is checked by `python3 -m mograph.v14 selftest` (9/9 pass).

## 0. Integration problem (blocking)
* The patch was made against `89a4ee8`, but `main` already had **v14.7 + v14.8** (PR #2). Applying it directly
  failed on 12 files. Forcing it would have deleted `bokeh.py`, `motionblur.py`, `craft_bridge.py`, the v14.8
  `director.py` (QA auto-fix loop, linguistic scoring) and the v14.8 glass optics.
* Resolution: apply on its own base → 3-way merge into `main`. Kept: v14.8 director / optics / bokeh / motion
  blur. Added: every v15 module. Its v14.7 part was already identical to `main` (no diff).
* The v15 `glass()` was a second height-field refraction that duplicated v14.8 `_glass_optics` (two optical
  models stacked). Kept one model and added its only new idea, Fresnel rim reflectance (Schlick, F0 = 0.04),
  as the `post.glass.fresnel` parameter.

## 1. Bugs found and fixed (measured before → after)
| engine | v15.0 bug | measured v15.0 | v15.1 |
|---|---|---|---|
| multiband (`audio.py`) | separate Butterworth LP/BP/HP bands are not phase-aligned, so they cancel at each crossover; gain applied at audio rate with no attack/release | −20 dB @160 Hz, −14 dB @800 Hz, −18 dB @4.5 kHz | LR4 **subtractive** tree (Σ bands = x, error 4e-9) + dB-domain attack 6 ms / release 90 ms → 0.00 dB |
| remix (`remix.py`) | the second splice point was not on the bar grid; beat length assumed to be 2 beats; no tempo/downbeat detection | beat intervals 0.36 / 0.52 s instead of 0.652 s (a hiccup in the groove) | bpm (autocorrelation) + grid fit (tempo ±2 %, phase), both cuts on bar lines, crossfade ends on the downbeat, looping to extend → error ≤ 1.3 ms |
| ARAP (`puppet_arap.py`) | cold start + Python loop; does not converge | pure 90° rotation: 103 px error (1 iteration), 78 px (4) | similarity warm start + `splu` factorised once + vectorised RHS → 2.6e-13 px, 7 ms for a 20×20 mesh |
| Procrustes morph (`autotrace.py`) | rotation sign read with the column convention for row vectors; correspondence search not rotation-invariant | at t = .25: 7° instead of 15°; area +9 % (60°) / +19 % (−80°); wrong pairing at 150° | θ = atan2(R01, R00), FFT cross-correlation (rotation-invariant), geometric scale → area 0.00 %, angle ±0.05° |
| decontaminate (`compositor.py`) | weight (0.94−α)·min(1,8α) ≤ 0.85 → only ~30 % un-mix | edge error 27.9 → 19.9 | true un-mix (edge = α·F + (1−α)·B → F), push-pull fill, bbox-local → 27.9 → 0.8 at strength 1, 3× faster (53 ms) |
| skin guard (`palette.py`) | assumed hue shift ∝ cast strength; no limit on ΔL or saturation | 10.7° on a strong cast; midtone ΔL −0.12 (face crushed); sat ×1.45 | exact bisection (≤ 8°), \|ΔL\| ≤ 0.05, saturation follows the guard, clamped to [0.8, 1.3] |
| path text (`glyphs.py`) | linear polyline through the points → tangent jumps at every knot | 10 px of text → 27 px gap at the apex | Catmull-Rom or exact Bézier chain, dense arc-length table (cached) → spacing variation 0.6 % |
| varstroke | width evaluated only at the input vertices; averaged normals | stepped taper on coarse paths; corners pinched | arc-length resampling (original corners kept) + miter normals (≤ 2.2 r) + winding fill |
| vegas_comet | head jumps from vertex to vertex | comet = 1–2 straight segments on a heptagon | travels by arc length (`polyline_slice`) |

## 2. Dead code → now usable from the spec
| v15.0 function | problem | v15.1 spec key |
|---|---|---|
| `match_reference_grade` | computed numbers, nothing applied them | `"grade": {"match": {"ref": "...mp4/jpg", "strength": 0.8}}` (64³ delta LUT) |
| `flow_cut.morph_cut_frame` | never called | `"source": {"morph_cuts": {"dur": 0.24, "max_motion": 60}}`: flows cached per cut, matte blended too, large-motion cuts skipped |
| `eval_selector` | never called | `"selectors": [{"shape", "start", "end", "offset": [t0, t1, o0, o1], "props": {...}}]` on every text layer |
| `shatter3d` | never called | layer `shatter_text` (`break_at`) |
| `puppet_arap` | never called | layer `puppet_sticker` (pins with `wiggle` / `to`) |
| `shape_context_procrustes_morph` | never called | layer `morph_shape` (`shapes`, `at`) |
| remix | only for the procedural bed (which is already composed to fit the edit) | `"music": {"file": "media/song.mp3"}`: a real track, remixed automatically |

## 3. Left as-is (fine)
`soft_clip`, `dehum_comb`, `vocal_booth_reverb`, `douglas_peucker`, `trace_scalar_field_to_beziers`,
`stabilize_camera_trajectory`, `track_planar_homography`, `variable_feather_mask`, Oklab conversion
(exact round trip). The brief's claim of "Newton cubic-Bézier fitting" is not accurate: the fit is a
least-squares fit with fixed tangents (fine for tracing).
