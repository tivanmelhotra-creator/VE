# TOOLS_INDEX — auto-generated inventory of the SDK
_generated 2026-10-21 by `python3 sdk/tools/build_tools_index.py` — do not edit by hand; edit docstrings instead. The curated guide is `SKILL_TOOLBOX.md`._


## v14 engine — talking-head footage (USE THIS FOR REAL VIDEO)

### `mograph.v14.__main__`  (sdk/mograph/v14/__main__.py)  · CLI: `python3 -m mograph.v14.__main__ …`
python3 -m mograph.v14 <command> spec.json [...]

<details><summary>details</summary>

```
prep    spec.json                 offline pose + MODNet matte cache for the spec's source (run once)
  frame   spec.json --t 3.2 [...]   render exact frame(s) → out/<name>_t3.20.png   (< 2 s each)
  sheet   spec.json [--n 12]        contact sheet across the edit (every beat at a glance)
  words   spec.json                 print the edited-time transcript (for "w:" anchors)
  qa      spec.json                 resolve every anchor, check layers are on-screen, list timings
  render  spec.json [--preview]     final video (+ audio) → out/<name>.mp4 ; --preview = half res
  selftest                          v15.1 numeric regression tests of every Craft engine (< 15 s, no footage)
  auto-edit words.json raw.mp4 -o edit_auto.json    v14.8 AUTOMATIC director: breath cuts + hero words + depth
                                                   rotation + camera/grade → a QA-clean spec (director.py)
```
</details>

| function / class | what |
|---|---|
| `main(argv)` |  |

### `mograph.v14.audio`  (sdk/mograph/v14/audio.py)
mograph.v14.audio — soundtrack for a v14 edit, built from the SAME timing maths as the picture.

<details><summary>details</summary>

```
voice   NOISE-CANCELLED (denoise.py: RNNoise neural + presence/air restore, auto strength from measured SNR),
          then cut with the EDL (12 ms raised-cosine splices), high-pass 80 Hz, gentle de-ess,
          compressor-ish level riding, normalised
  sfx     event-locked: each layer's entrance → its sound (whoosh / pop / jelly / boom / tick / scribble / hit),
          hero_hit → hit, disc flip → whoosh, glass pulse → jelly, subtitle phrase change → soft jelly
          (pitch from the capsule width change), camera shakes → low thump.  Stereo pan by layer x.
  music   procedural bed from mograph/beat.py (lofi / trap_pop / cinematic), ducked under the voice
  master  sum → soft limiter → integrated loudness to audio.lufs (default −14)
```
</details>

| function / class | what |
|---|---|
| `voice_track(R, n)` |  |
| `sfx_events(R)` |  |
| `sfx_track(R, n, gain)` |  |
| `music_track(R, n, cfg)` | Music bed.  cfg = {"style": "lofi", "bpm": 92, ...}  → procedural beat (mograph/beat.py) |
| `build_audio(R, out_wav, start, end)` |  |
| `soft_clip(x, t)` | Ported from FilmCraft `crates/audio-dsp/src/effects/multiband.rs`: cubic soft clipper with ceiling `t`. |
| `dehum_comb(x, sr, f0, harmonics, q)` | Ported from FilmCraft `crates/audio-dsp/src/effects/restoration.rs` (`DeHum`): harmonic notch comb. |
| `split_bands(x, sr, xovers)` | v15.1 PERFECT-RECONSTRUCTION band split (sum of bands == x to float precision). |
| `multiband_compress_4band(x, sr, bands_cfg, attack_ms, release_ms)` | 4-band broadcast voice compressor (FilmCraft `multiband.rs` idea) — v15.1 rewrite: |
| `vocal_booth_reverb(x, sr, wet)` | Ported from FilmCraft `crates/audio-dsp/src/effects/reverbs.rs` (`IrSpec` "Vocal Booth"): |

### `mograph.v14.bokeh`  (sdk/mograph/v14/bokeh.py)
mograph.v14.bokeh — depth-sandwich 3D BOKEH DUST (v14.8, CPU port of EffectCraft three_d/bokeh.rs).

<details><summary>details</summary>

```
Out-of-focus floating dust in a real lens: every point light far from the focal plane becomes a disc shaped like the
iris (octagonal aperture blades), slightly brighter at the rim (spherical aberration "soap bubble"), and moves with
a parallax that depends on its distance to the camera.  We render it as TWO layers of the depth sandwich:

  BACK   12 small/medium octagonal discs BEHIND the speaker (drawn on the background before the speaker composite,
         so she occludes them).  Tinted by @accent mixed with warm white.  Parallax 0.7× of the camera.
  FRONT   6 LARGE very soft discs IN FRONT of everything (drawn after the front pass, before bloom), parallax
         1.3× of the camera → they slide faster than the speaker on every push-in / pan = real depth.
         Front discs fade out over the protected face ellipse (never fog the eyes / mouth).
```
</details>

| function / class | what |
|---|---|
| `resolve(R, t)` | Active config at time t (merging post.bokeh_dust and ambient_bokeh layers) or None. |
| `draw(F, arr, kind, face_zone)` | Draw the `kind` ("back" / "front") dust onto BGR uint8 `arr` (in place). |

### `mograph.v14.compositor`  (sdk/mograph/v14/compositor.py)
mograph.v14.compositor — the v14 "Studio Depth" frame pipeline for talking-head reels.

<details><summary>details</summary>

```
WHY A NEW PIPELINE (v13.9 audit, see V14_AUDIT.md):
  * v13.9 drew everything on ONE cairo canvas → no true depth, glass could not blur what is under it, glows
    never touched the speaker, light spill was a fake gradient, behind-speaker text had halo edges.
  * v14 renders every frame as a physically ordered sandwich:

     1  plate       source frame → virtual camera (warpAffine) → speaker matte (MODNet, cached)
     2  background  mask-normalised DOF blur + grade (dim / desat / tint, keyframable) – no person ghost
     3  BACK pass   layers with pass "back" (titles behind the head, halos, domes, back half of 3D rings)
                    + their EMISSION (glow) on a half-res light buffer
     4  speaker     graded, LIGHT-WRAPPED by what is behind her, LIGHT-SPILLED by every glowing layer
```
</details>

| function / class | what |
|---|---|
| `norm_word(w)` |  |
| `Timeline` | EDL: output time ↔ source time, and word anchors in OUTPUT time. |
| `load_words(path)` |  |
| `Camera` | Virtual camera on the source plate. keys: [t, zoom, cx, cy, ease?, rot?]  (cx, cy in source uv). |
| `Frame` | Everything a layer needs for one frame. |
| `surface_array(surf)` |  |
| `decontaminate_foreground_edge(fg_u8, alpha_f32, strength)` | Ported from EffectCraft `crates/track/src/roto/refine.rs` (`decontaminate`): |
| `glass(ctx, F, path_fn, bbox, blur, tint, …)` | TRUE frosted glass: blurs the real composite under the shape (not a flat fill). |
| `Renderer` |  |

### `mograph.v14.craftlayers`  (sdk/mograph/v14/craftlayers.py)
mograph.v14.craftlayers — v15.1 spec-level layers for the v15 Craft engines (so the agent uses them from JSON,

<details><summary>details</summary>

```
not from Python).  Registered into layers.DRAW on import.

  shatter_text   a kinetic text block that holds, then EXPLODES into 3D shards at `break_at` (shatter3d.py)
      {"type": "shatter_text", "t": "w:مقاله", "out": 4.2, "text": "*مقاله*", "size": 90, "anchor": "head_top",
       "offset": [0, -0.6], "break_at": "w:نوشتن", "rows": 4, "cols": 7, "force": 380, "gravity": 900}
  puppet_sticker an RGBA sticker (assets_3d / project path) bent by ARAP pins (puppet_arap.py) — a waving flag,
                 a wobbling cartoon, a bending arrow.  Pins: normalised [u, v] on the image + motion.
      {"type": "puppet_sticker", "src": "warn_3d.png", "anchor": "shoulder_r", "size": 0.9,
       "pins": [{"uv": [0.5, 1.0]}, {"uv": [0.5, 0.0], "wiggle": [26, 0], "freq": 1.6}], "grid": 10}
  morph_shape    a filled shape that MORPHS between outlines (autotrace.shape_context_procrustes_morph):
                 shapes = names (circle | hept | hex | star | heart | diamond | rsquare) or point lists (block px)
```
</details>

| function / class | what |
|---|---|
| `draw_shatter_text(ctx, L, F)` |  |
| `draw_puppet_sticker(ctx, L, F)` |  |
| `draw_morph_shape(ctx, L, F)` |  |
| `register(DRAW)` |  |

### `mograph.v14.denoise`  (sdk/mograph/v14/denoise.py)  · CLI: `python3 -m mograph.v14.denoise …`
mograph.v14.denoise — automatic broadcast voice cleanup (noise cancelling) for talking-head footage.

<details><summary>details</summary>

```
No knobs needed: it MEASURES the take and picks the strength itself.

  1  measure      noise floor (quietest 10 % of 50 ms windows) vs speech level → SNR
  2  ENGINE       DeepFilterNet3 (sdk/mograph/bin/deep-filter, static binary, model embedded) — default;
                  attenuation cap from SNR (noisy 42 dB … clean 18 dB).  Fallback when the binary is missing:
     rnnoise      neural speech denoiser (RNNoise, ffmpeg `arnndn`, models in mograph/models/rnnoise:
                  lq = voice vs general noise (default, best in QC listening test), bd, sh, cb, mp).
                  The dry/wet mix is chosen from the SNR (noisy take → 0.72, clean studio → 0.40) so clean
                  voices never get the "underwater" artefact.
  3  residual     OPTIONAL spectral gate (noisereduce) — off by default (QC: phasey on outdoor takes)
```
</details>

| function / class | what |
|---|---|
| `measure(x, sr)` | Noise floor = RMS of the quietest 10 % of 50 ms windows; speech = loudest 30 %. |
| `clean(x, sr, model, strength, residual, log, …)` | Engine order: DeepFilterNet3 (strong, natural) → RNNoise fallback. |
| `clean_voice(src, model, strength, log, residual)` | Cached: returns path of a 48 kHz mono cleaned WAV next to the media cache. |

### `mograph.v14.director`  (sdk/mograph/v14/director.py)  · CLI: `python3 -m mograph.v14.director …`
mograph.v14.director — AUTOMATIC BEAT-SHEET DIRECTOR (v14.8; resolves SKILL_TOOLBOX backlog #3).

<details><summary>details</summary>

```
python3 -m mograph.v14 auto-edit words.json media/raw.mp4 -o edit_auto.json [--theme doran] [--no-qa]

Input : a word-timed transcript (words.json: {"words":[{"text","start","end"}]} | [{"w","t","end"}] | [[w,t,end]])
        + the raw talking-head video.
Output: a complete, QA-clean v14 spec (same schema as hand-made edit_v14.json) — ready for `frame` / `render`.

PIPELINE (every step is deterministic → same input = same edit):
  1  BREATH CUTS     pauses between words > 0.32 s are removed → source.segments (0.10 s pre-roll, 0.14 s tail,
                     very short islands are merged with their neighbour).
  2  WORD SCORING    every word gets an "impact" score from cheap Persian linguistics (no model):
```
</details>

| function / class | what |
|---|---|
| `build_segments(words, src_dur, pause)` | Speech islands separated by pauses > `pause` → [[a, b], ...] in source seconds. |
| `score_words(tw)` | tw = Timeline.words (output time).  Returns list of floats (impact score per word). |
| `pick_beats(tw, scores, duration, bmin, bmax)` | Greedy: one hero word per [last + bmin, last + bmax] window (first beat = best word in the opening). |
| `build_layers(tl, tw, scores, beats, duration, seg_of)` |  |
| `auto_edit(words_path, video_path, out_path, theme, qa, log)` |  |
| `main(argv)` |  |

### `mograph.v14.flow_cut`  (sdk/mograph/v14/flow_cut.py)
mograph.v14.flow_cut — BIDIRECTIONAL OPTICAL-FLOW MORPH CUT, GAUSSIAN/L1 STABILIZER & PLANAR TRACKER

<details><summary>details</summary>

```
(Ported from FilmCraft `crates/render/src/flow.rs` & EffectCraft `crates/track/src/{stabilize.rs, klt.rs}`).

1. `morph_cut_frame(frame_a, frame_b, alpha)`:
   Synthesizes an occlusion-aware optical-flow intermediate frame at `alpha in (0, 1)` between two
   adjacent clip frames across a jump-cut boundary (`R.tl.cuts`).
2. `stabilize_camera_trajectory(transforms, sigma=4.0, max_scale=1.08)`:
   Matsushita (PAMI 2006) Gaussian viewpoint smoothing + Grundmann (CVPR 2011) maximum-scale crop relaxation.
3. `track_planar_homography(prev_bgr, cur_bgr, quad_pts)`:
   Pyramidal Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`) + RANSAC 8-DOF homography (`cv2.findHomography`)
   for Mocha-style 4-corner planar surface tracking.
```
</details>

| function / class | what |
|---|---|
| `morph_cut_frame(frame_a, frame_b, alpha, work_w)` | Ported from FilmCraft `crates/render/src/flow.rs`: |
| `stabilize_camera_trajectory(dx_dy_da, sigma, max_shift_px)` | Ported from EffectCraft `crates/track/src/stabilize.rs`: |
| `track_planar_homography(prev_bgr, cur_bgr, quad_pts)` | Ported from EffectCraft `crates/track/src/{klt.rs, fit.rs, solve.rs}`: |
| `MorphCuts` | Optical-flow MORPH CUT on every jump cut of the EDL (Premiere "Morph Cut" idea) — spec: |

### `mograph.v14.glassbadge`  (sdk/mograph/v14/glassbadge.py)
mograph.v14.glassbadge — NESTED glass badge (the @novadent heptagon, measured from the reference frames).

<details><summary>details</summary>

```
Anatomy (outside → in):
  ① outline   thin ACCENT stroke (palette) (≈2.6 px) + soft orange emission that lights the speaker/clothes
  ② GAP       `gap` px (default 7) of UNTOUCHED footage between the outline and the glass body
  ③ body      inset copy of the shape: frosted blur of what is under it + DARK RADIAL CORE
              (centre ≈ 82 % dark → rim ≈ 15 %) so text reads while the rim stays glassy
  ④ hairline  1 px white @ 16 % on the body edge (glass thickness) + top-left sheen
  ⑤ specular  warm bloom at the head of the outline while it draws on, then slowly orbiting
Entrance: outline trims on (0.45 s silk) while the body jelly-springs from the centre; text pops last.
v14.8: the body is REFRACTIVE glass — lens bend at the rim (`refract` px), R/C dispersion (`chroma` px) and a
specular caustic (`caustic` 0..1) from compositor._glass_optics; set any of them to 0 per layer to disable.
```
</details>

| function / class | what |
|---|---|
| `shape_path(shape, x, y, w, h, rad, …)` |  |
| `draw_glass_badge(ctx, L, F, pos, glyph_motion, blk, …)` |  |

### `mograph.v14.glyphs`  (sdk/mograph/v14/glyphs.py)
mograph.v14.glyphs — VECTOR typography you can deform in real 3D.

<details><summary>details</summary>

```
v13.9 audit: titles were one flat block; curved text was a RASTER warped by cv2.remap (blurry, no 3D); "3D" was a
2D scale+rotate (the dry-bread look).  v14 shapes Persian with HarfBuzz, pulls every glyph OUTLINE as polygons and
pushes every point through a 3D pipeline each frame:

   outline ─► per-glyph local 3D transform (spring pop / flip / jelly)  ─► global Field (arc / cylinder / wave /
   jelly / bulge) ─► block plane rotation ─► perspective projection ─► cairo path  (+ extrusion slices = real depth)

Cursive joins survive every deformation (joins are shared outline edges).

MARKUP (one string any LLM can write):   "آمپول‌های|*لاغری*"     '|' = new line
```
</details>

| function / class | what |
|---|---|
| `set_palette(pal)` | Install a designed palette globally (Renderer does this from meta.palette / palette.auto_palette). |
| `font_path(fp)` |  |
| `rgba(c, pal)` |  |
| `glyph_contours(fp, gid)` |  |
| `Glyph` |  |
| `Block` |  |
| `parse_markup(line)` |  |
| `theme_styles(theme, over)` | Role → style for one superfamily.  Ratios measured on the @novadent reference frames: |
| `lock_font(theme, role)` | Font file for a role in the theme (for code paths that need a raw font: subtitles, badges, labels). |
| `merged_styles(over)` |  |
| `plain(text)` | Markup → plain text (for QA / subtitles). |
| `typeset(text, size, styles, leading, align, word_gap, …)` |  |
| `rot3(rx, ry, rz)` |  |
| `Field` | Global deformation of block points (N×3, block-centred px, y down, +z away from camera). |
| `eval_selector(u, t, cfg)` | Ported from EffectCraft `crates/text/src/selectors.rs`: |
| `project(P, f)` |  |
| `GS` | Per-glyph animation state. |
| `draw_block(ctx, blk, x, y, motion, field, …)` | Draw block at screen (x, y). |

### `mograph.v14.layers`  (sdk/mograph/v14/layers.py)
mograph.v14.layers — the v14 motion-design component library (every layer = physics + 3D + light).

<details><summary>details</summary>

```
Each layer is a JSON dict.  Common keys:
  type      layer type (below)               id      name (for QA)
  t, out    in / out time ("w:word", "we:word+0.1", seconds)
  pass      "back" (behind the speaker) | "front" (default) | "both" (3D layers split themselves by depth)
  anchor    tracked point the layer is glued to: head_top face nose chin neck chest belly shoulder_l/r hand_l/r hands
            or "screen" (fixed).  "offset": [dx, dy] in units of HEAD WIDTH (so it scales with the speaker/zoom)
            "x","y" = screen px when anchor == "screen".   "follow": 0..1 how much it follows (default 1)
  size      base font px (scaled by "head_scale" × head width when anchored, default true)
  text      MARKUP (see glyphs.py):  "آمپول‌های|*لاغری*"
  styles    per-layer style overrides       palette  {"accent": "#FF2E42", ...}
```
</details>

| function / class | what |
|---|---|
| `prepare(L, R)` |  |
| `in_pass(L, p)` |  |
| `pos(L, F)` |  |
| `element_radius(L, F, hw)` | Approximate bounding radius (px) of a layer — used by the layout guard and the stage light. |
| `blk(L, size_mul, text_key, size_key)` |  |
| `draw(ctx, L, F)` |  |
| `draw_kinetic_title(ctx, L, F)` |  |
| `ring_fit(text_w, R, sep, min_scale, max_span, max_R, …)` | GEOMETRY SOLVER for text on a 3D ring — the agent never has to think about it. |
| `draw_orbit_ring(ctx, L, F)` | Text on a TRUE 3D cylinder around the head (back half drawn behind her, front half in front). |
| `band_fit(blk_w, blk_h, r_in, r_out, span_deg, want)` | Fit a text block INTO a circular band (r_in..r_out). Returns (scale, radius_of_text_centre). |
| `draw_band_text(ctx, F, L, b, x, y, …)` |  |
| `draw_disc(ctx, L, F)` | @novadent vinyl: dark translucent glass body, thin rim, world-locked NEON half-rim on top that spills red |
| `draw_glass_shape(ctx, L, F)` |  |
| `draw_halo(ctx, L, F)` |  |
| `draw_frame_box(ctx, L, F)` |  |
| `draw_strike(ctx, L, F)` | v15 Variable-Width Tapered Brush Strike / Underline / Circle (using `mograph.varstroke`). |
| `draw_flash(ctx, L, F)` |  |
| `draw_progress(ctx, L, F)` |  |
| `draw_counter_badge(ctx, L, F)` |  |
| `draw_sweep_light(ctx, L, F)` |  |
| `draw_tracked_tag(ctx, L, F)` | MOTION-TRACKING SHOWCASE: a small glass chip glued to a tracked point (hand / shoulder) with a curved |
| `draw_object_3d(ctx, L, F)` |  |
| `draw_ambient_bokeh(ctx, L, F)` | Rendered by the compositor (bokeh.py) in the two depth slots; nothing to draw on the layer canvases. |
| `draw_vegas_comet(ctx, L, F)` | Ported from EffectCraft `crates/effects/src/generate3.rs` (`vegas`): |
| `draw_god_rays(ctx, L, F)` | Ported from EffectCraft `crates/effects/src/generate2.rs` (`light_rays` / `light_burst`): |

### `mograph.v14.motionblur`  (sdk/mograph/v14/motionblur.py)
mograph.v14.motionblur — velocity-driven 180° shutter DIRECTIONAL motion blur (v14.8).

<details><summary>details</summary>

```
WHY: at 30 fps a spring entrance (drop_bounce, swing_3d, flip_3d, object_3d spin-in) moves 40-120 px between two
frames.  Without blur the eye sees discrete copies ("stroboscopic stepping").  A real camera with a 180° shutter
integrates the motion over half the frame interval → a 1-D streak along the velocity vector.

HOW (exact, cheap, local):
  1  the caller knows the element at time t and at t − Δ (Δ = shutter/360 / fps) → screen displacement vector d
     (glyphs: projected contour points; object_3d: rim speed of the yaw spin + spring translation)
  2  if |d| < threshold (default 2.5 px) nothing happens — static / settled frames cost zero
  3  otherwise the element is painted into a small OFFSCREEN premultiplied-ARGB surface (its device bbox + |d|),
     convolved with a normalised LINE kernel of length |d| along d (cv2.filter2D — premultiplied alpha makes the
```
</details>

| function / class | what |
|---|---|
| `config(L, F)` | Resolved motion-blur config for a layer, or None when disabled. |
| `shutter_dt(cfg, fps)` |  |
| `line_kernel(dx, dy)` | Normalised 1-D line kernel of length /d/ along (dx, dy) (anti-aliased endpoints). |
| `principal_displacement(P0, P1)` | Dominant screen displacement (dx, dy) between two (N,2) point sets (translation, or the main axis of a |
| `paint_blurred(ctx, draw_fn, bbox, dx, dy, max_len)` | Draw `draw_fn(c)` (user-space coords of ctx) into an offscreen surface, directional-blur by (dx, dy) |

### `mograph.v14.object3d`  (sdk/mograph/v14/object3d.py)
mograph.v14.object3d — `object_3d` layer: a REAL turning 3D object inside the v14 talking-head engine.

<details><summary>details</summary>

```
{"type": "object_3d", "t": "w:لپ‌تاپ", "out": "we:قیمت+0.3",
     "model": "croissant",                   # v14.6 PHOTOREAL scanned PBR model (pbr3d) — preferred — or —
     "src": "laptop_3d.png",                 # any RGBA image (assets_3d / project path)  — or —
     "preset": "laptop_pro_3d",              # procedural sculpt3d model (laptop_pro_3d | ozempic_pen_3d | trophy_3d)
     "geometry": "auto",                     # auto | slab | badge | cushion | multiview (image sources)
     "anchor": "hand_r", "offset": [0.2, -0.6],   # or "anchor": "screen", "x": 760, "y": 640
     "size": 1.4,                            # diameter in HEAD WIDTHS when anchored, px when anchor == screen
     "motion": "hover",                      # turntable_osc | spin_360 | spring_entry | hover | none
     "yaw": -20, "pitch": 12,                # base angles;  "turn_on": "w:…" → one 360° spin at that word
     "enter": "spring_pop" | "drop" | "fade",  "exit": "pop_out" | "fall" | "fade",
```
</details>

| function / class | what |
|---|---|
| `draw_object_3d(ctx, L, F, pos, out_lt)` |  |

### `mograph.v14.palette`  (sdk/mograph/v14/palette.py)  · CLI: `python3 -m mograph.v14.palette …`
mograph.v14.palette — AUTOMATIC colour design from the footage (no reference needed).

<details><summary>details</summary>

```
The reference's gold/red works because ITS background is warm-dark wood.  A different plate needs different
colours.  This module looks at the actual graded background + the speaker and designs a palette with real
colour-theory rules, so even a weak agent gets harmonious, readable elements:

  1  sample      background pixels (matte < 0.2) and speaker pixels (clothes / skin) over the whole clip
  2  measure     dominant background hue (circular mean, weighted by chroma), background luminance,
                 speaker clothing hues (to AVOID: an accent that matches her clothes disappears)
  3  design      accent  = complementary-ish to the background hue (≈ +150..210°), pushed away from
                           clothing hues, warm bias (warm accents read as "light"),
                 accent2 = split-complement (+30° from accent) for secondary emphasis,
```
</details>

| function / class | what |
|---|---|
| `luminance(rgb)` |  |
| `contrast(a, b)` |  |
| `sample(video, matte_dir, n)` |  |
| `design(bg_px, fg_px, mode)` |  |
| `auto_palette(video, log)` |  |
| `swatch(pal, path)` |  |
| `bgr_to_oklab(bgr_u8)` | Convert BGR uint8 image to perceptual Oklab (L, a, b) float32 (Björn Ottosson 2020). |
| `oklab_to_bgr(lab)` | Convert Oklab (L, a, b) float32 image back to BGR uint8. |
| `match_reference_grade(src_bgr, ref_bgr, person_alpha, max_skin_deg, max_dL)` | Ported from FilmCraft `crates/render/src/color_match.rs`: |
| `grade_match_lut(m, strength, n)` | Colour-only correction → a DELTA 3D LUT (n³ × 3 int16, indexed by BGR >> shift).  Built once per match. |
| `apply_grade_match(bgr, m, strength, work)` | v15.1: APPLY a match_reference_grade() result (v15.0 only computed it — nothing used the numbers). |
| `reference_frame(path, at)` | A still from an image or a video (at `at` × duration) for grade matching. |

### `mograph.v14.physics`  (sdk/mograph/v14/physics.py)
mograph.v14.physics — motion that feels ALIVE (the cure for "text moving like dry bread").

<details><summary>details</summary>

```
v13.9 audit: every animation used one scalar p (0→1) driving opacity/scale.  Real motion designers animate with
physics.  This module gives every layer:
  * closed-form damped spring with initial VELOCITY  (overshoot, settle)
  * squash & stretch from the motion derivative (volume preserved: sx·sy = 1)
  * secondary motion: per-glyph delay → wave / ripple / follow-through
  * jelly wobble: decaying scale + shear oscillation after an impact
All functions are PURE functions of time → frames render in any order / in parallel.
```
</details>

| function / class | what |
|---|---|
| `clamp(v, lo, hi)` |  |
| `spring(t, k, c, m, v0)` | Damped spring 0→1 for t ≥ 0 (initial velocity v0).  Overshoots when damping c is small. |
| `sp(t, preset, v0)` |  |
| `sp_vel(t, preset, dt)` |  |
| `bezier(x1, y1, x2, y2)` |  |
| `ease_in_out(p)` |  |
| `ease_out(p)` |  |
| `ease_in(p)` |  |
| `jelly(t, amp, freq, decay)` | Decaying wobble after an impact at t=0 → (sx, sy, shear); volume preserved. |
| `osc(t, freq, amp, seed)` |  |
| `env(t, t_in, t_out, fin, fout)` |  |

### `mograph.v14.qa`  (sdk/mograph/v14/qa.py)
mograph.v14.qa — fast checks before any render (< 5 s):

<details><summary>details</summary>

```
* every time / word anchor resolved (prepare() already raises on a missing word)
  * every layer: duration > 0.25 s, inside the edit, sampled at 3 instants → ink bbox on screen (warn if clipped)
  * overlapping FRONT layers at the same instant in the same screen region (clutter)
  * subtitle phrases: none longer than 3.5 s, none shorter than 0.35 s
Exit code 0 = clean, 1 = warnings.
```
</details>

| function / class | what |
|---|---|
| `run_qa(R)` |  |
| `qa_warnings(R, verbose)` | All QA warnings as a list of strings (used by run_qa and by the v14.8 auto-director's fix loop). |

### `mograph.v14.remix`  (sdk/mograph/v14/remix.py)
mograph.v14.remix — BEAT-GRID MUSIC REMIX to an exact duration (idea from FilmCraft `audio-dsp/src/remix.rs`).

<details><summary>details</summary>

```
y2 = remix_music_to_duration(y, sr, target_dur, bpm=None, beats_per_bar=4, downbeat=None)

v15.1 rewrite (the v15.0 version moved the second splice point OFF the bar grid — measured on a click track:
beat intervals of 0.36 s / 0.52 s instead of 0.652 s at 92 BPM → audible "hiccup" in the groove):
  1  GRID     bpm (given, or estimated by onset-envelope autocorrelation 60–180 BPM) + first downbeat
              (given, or the strongest onset phase) → bar boundaries b_k = downbeat + k·bar.
  2  FEATURES per bar: 12 log-spaced spectral band energies (55 Hz – 8 kHz) of 3 frames, L2-normalised.
  3  SHORTEN  keep intro bars [0, k) and outro bars [j, end) with j − k = bars to drop; both splice points are
              bar boundaries → the beat grid is continuous.  Score = cosine(bar k−1 , bar j−1) (what the ear
              expects after bar k−1 is bar k; we play bar j instead, so compare their predecessors) +
```
</details>

| function / class | what |
|---|---|
| `onset_envelope(y, sr, hop)` | Spectral-flux onset strength (half-wave rectified log-magnitude increase), one value per hop. |
| `estimate_bpm(y, sr, lo, hi, hop)` | Tempo from the autocorrelation of the onset envelope (with a mild preference for 80–130 BPM). |
| `fit_grid(y, sr, bpm, beats_per_bar, hop, refine)` | (bpm, downbeat_s): phase (and, if refine, tempo ±2 %) of the bar grid that collects the most onset energy |
| `cv_max3(e)` | 3-tap running max (tolerates ±1 hop of onset jitter). |
| `estimate_downbeat(y, sr, bpm, beats_per_bar)` |  |
| `bar_features(y, sr, starts, bar_n)` |  |
| `remix_music_to_duration(y, sr, target_dur, bpm, beats_per_bar, downbeat, …)` |  |
| `load_music(path, sr)` | Any audio file → mono float32 at `sr` (ffmpeg). |

### `mograph.v14.selftest`  (sdk/mograph/v14/selftest.py)
mograph.v14.selftest — numeric regression tests for every v15 Craft engine (no footage needed, < 15 s).

<details><summary>details</summary>

```
cd sdk && python3 -m mograph.v14 selftest          # exit 0 = all pass

Each test measures the property that v15.0 got wrong (see sdk/V15_REVIEW.md), so a regression is caught at once:
  multiband   Σ bands == input (perfect reconstruction) and flat response at low level (±2.5 dB)
  remix       beat grid preserved through every splice (click track, 92 & 120 BPM, shorten + extend)
  arap        a pure 90° rotation of two pins is reproduced exactly (rigid) ; pins hit their targets
  morph       rotated rectangle: area constant (±1 %) and orientation linear in t (±2°)
  oklab       round trip exact (≤ 1 LSB) ; grade match on a strong cast keeps skin hue within 8°
  decontam    composite error on a synthetic green-spill edge reduced ≥ 3×
  pathtext    glyph spacing across a path knot stays uniform (±15 %)
```
</details>

| function / class | what |
|---|---|
| `main()` |  |

### `mograph.v14.showcase`  (sdk/mograph/v14/showcase.py)  · CLI: `python3 -m mograph.v14.showcase …`
mograph.v14.showcase — the @novadent "BEFORE / AFTER" editor-showcase wrapper.

<details><summary>details</summary>

```
Two tilted rounded cards on a heavily blurred, darkened copy of the AFTER video:
  BEFORE (raw take, small, tilted back, desaturated) behind  ·  AFTER (the edit, big, front, teal rim)
Cards float (slow 3D sway), the AFTER card punches to full screen on "full_at" windows (ease in/out).

python3 -m mograph.v14.showcase after.mp4 raw.mp4 out.mp4 --segments "1.12-7.14,7.44-12.06,12.24-17.62"         --full "6.0-6.8,9.9-10.6"
```
</details>

| function / class | what |
|---|---|
| `build(after, raw, out, segments, full, W, …)` |  |

### `mograph.v14.stage`  (sdk/mograph/v14/stage.py)
mograph.v14.stage — two automatic "editor brains" every layer goes through (a weak agent gets them for free).

<details><summary>details</summary>

```
A) LAYOUT GUARD  (never cover the face / mouth, never leave the safe frame)
   protected zone = an ellipse around the tracked face, from forehead to below the chin, + a margin, computed
   per frame from the pose.  Before a layer is drawn its bounding circle/box is tested; if it overlaps the zone
   it is MOVED along the layer's preferred escape direction (down for chest elements, up for head titles,
   sideways for side tags) — smoothly (the shift itself is spring-smoothed over time so nothing jumps) — and
   also clamped to the 9:16 safe frame (top 140 px UI bar, bottom 260 px caption zone, 40 px sides).
   Layers opt out with "guard": false (e.g. halo, orbit ring and full-screen flashes are designed around her).

B) STAGE LIGHT  (the user's taste: a translucent engineering grid, low opacity, that DARKENS the plate locally,
   with a centre glow in the element's colour that fades with distance)
```
</details>

| function / class | what |
|---|---|
| `face_zone(F, margin)` | Protected ellipse (cx, cy, rx, ry) in screen px: forehead … below the chin. |
| `guard(L, F, x, y, r, kind)` | Return a safe (x, y) for an element of radius r.  Smooth over time (no jumps). |
| `plate_brightness(F, cx, cy, R)` |  |
| `draw_stage(F, L, cx, cy, size, color, …)` | Drawn on the BACK canvas (behind the speaker) — called by the renderer's stage pass. |
| `element_alpha(L, F)` | Envelope (0..1) of a layer's presence, for fading its stage with it. |

### `mograph.v14.subs`  (sdk/mograph/v14/subs.py)
mograph.v14.subs — STATEFUL jelly glass subtitles (the @novadent capsule, done properly).

<details><summary>details</summary>

```
v13.9 audit: the capsule wobbled around the CURRENT phrase width only, so it popped between phrases.
v14: the capsule keeps physical state across phrases:
  * width springs from the previous phrase's width to the new one (under-damped, k=380 c=14)
  * height conserves area (h ∝ w^-0.38) → shrinking = plumps up, growing = stretches thin
  * real frosted glass (blurs the speaker's clothes), teal crescent rim highlights on both end caps
  * karaoke: words pop in on their exact timestamp (jelly spring); the active word gets the accent colour;
    emphasised words ("emph" list or *markup*) use the hero font → inline weight contrast inside every phrase
  * a soft "jelly" SFX is scheduled on each phrase change (pitch from the width delta) — see audio.py

"subtitles": {"anchor": "chest", "offset": [0, 0.35], "size": 46, "max_words": 4, "emph": ["مقدمه", ...],
```
</details>

| function / class | what |
|---|---|
| `prepare_subs(cfg, R)` |  |
| `draw_subs(ctx, S, F)` |  |

### `mograph.v14.track`  (sdk/mograph/v14/track.py)  · CLI: `python3 -m mograph.v14.track …`
mograph.v14.track — offline, ZERO-LAG speaker analysis (computed once per clip, cached next to the video).

<details><summary>details</summary>

```
v13.9 audit: live one-euro tracking is causal (text trails the speaker) and the 256 px selfie matte has halo edges
("behind the speaker" looks pasted).  v14 analyses the whole clip OFFLINE:
  matte  MODNet portrait matting (models/modnet.onnx) → hair-accurate soft alpha, temporal median of 3 frames
         cached as PNG in <video>.v14/matte/   (fallback: bundled selfie segmenter)
  pose   MediaPipe pose every frame → CENTRED Gaussian smoothing (no lag) → anchors:
         head_top face nose chin neck chest belly shoulder_l shoulder_r hand_l hand_r hands head_c
         + head_width / shoulder_width (u units) so layers scale with the speaker.
CLI:  python3 -m mograph.v14.track media/raw.mp4      (run once; resumable)
```
</details>

| function / class | what |
|---|---|
| `cache_dir(path)` |  |
| `matte_frame(bgr, ref)` |  |
| `prep_matte(path, log)` |  |
| `prep_pose(path, log, work_w)` |  |
| `Tracks` |  |


## legacy engine v5–v13 — pure vector / infographic videos

### `mograph.__main__`  (sdk/mograph/__main__.py)
**UNDOCUMENTED — add a module docstring**

### `mograph.align`  (sdk/mograph/align.py)
mograph.align — narration audio → word-level subtitles (frame-exact karaoke + word anchors).

<details><summary>details</summary>

```
Uses faster-whisper if installed (pip install faster-whisper; CPU 'small' model is fine for Persian).
If a script text is given, recognised words are snapped to the SCRIPT's spelling (so word anchors
like "w:مستندتره؛" match exactly), keeping whisper's timing.

CLI:  python -m mograph align narration.wav --text script.txt --out subs.json
Then paste subs.json into spec["subtitles"] (word form) and set spec["audio"]["vo"].
```
</details>

| function / class | what |
|---|---|
| `transcribe_words(audio_path, language, model_size)` |  |
| `snap_to_script(words, script_text)` | Replace recognised tokens by the script's tokens (sequence alignment), interpolating timing |
| `align_cli(audio_path, script_path, out_path)` |  |

### `mograph.analyze`  (sdk/mograph/analyze.py)
mograph.analyze — v9 reference video analyzer (OpenMontage "reference-driven creation", BMTS-lite beats).

<details><summary>details</summary>

```
python3 -m mograph analyze reference.mp4 -o blueprint.json [--spec spec.json]

Extracts with ffprobe + cv2 + scipy only (offline):
  meta       duration, fps, w×h, aspect
  cuts       hard scene-cut timestamps (HSV-histogram distance + adaptive threshold) and cuts/min cadence
  shots      [{t, end, dur}] between cuts
  palette    dominant 4 colours (k-means on sampled frames) mapped to @paper (most common, darkest/lightest
             ground), @ink (max contrast vs paper), @accent (most saturated), @mute (remaining)
  motion     mean optical-flow magnitude per second (energy curve) + overall energy class
  audio      integrated LUFS (ffmpeg ebur128), onsets (spectral flux peaks), tempo (BPM, autocorrelation of the
```
</details>

| function / class | what |
|---|---|
| `scan_video(path, step_fps)` |  |
| `detect_cuts(times, hists, min_gap)` |  |
| `palette(samples, k)` |  |
| `lufs(path)` |  |
| `beats_from_audio(x, sr)` | Onsets (spectral flux) + tempo (autocorrelation 70–180 BPM) + phase-locked beat grid. |
| `analyze(path, out, spec_path, log)` |  |
| `beat_grid(bpm, duration, offset)` | Synthetic beat list for specs without a reference: spec["beats"] = {"bpm": 120, "offset": 0.1}. |

### `mograph.apex`  (sdk/mograph/apex.py)
mograph.apex — v9 Studio Apex world layer: procedural chapter backdrops, ambient depth layers, directional

<details><summary>details</summary>

```
motion blur (whip), aspect-ratio presets.

BACKDROPS  ("bg_mode" on a chapter, on meta, or from the style preset)
  solid             flat @paper (default, zero cost)
  radial_spotlight  deep edge (@edge, #070103) → glowing @glow spotlight (#5E0818) — crimson-black promo look
  studio_vignette   bright studio centre (#FAFAFA) → soft silver outer vignette (#B8B8C0)
  crimson_vivid     vibrant @accent centre (#FF1E42) → deep crimson edge (#9E0820)
  "bg_focus": [x, y] (chapter-local spotlight centre, default frame centre), "bg_radius": px (default 0.9·H)
  "bg_pulse": 0..1   spotlight breathes with the narrator (scene.vo_env)

AMBIENT LAYERS  ("ambient" on a chapter: "bokeh" | "warp_stars" | "grid" | list | {"kind": ..., options})
```
</details>

| function / class | what |
|---|---|
| `backdrop_spec(ch)` |  |
| `paint_backdrop(ctx, scene, ch, t, alpha, solid_too)` | Paint the chapter backdrop over the whole clip (world coords, chapter origin applied here). |
| `backdrop_center_color(ch)` | Representative colour (for contrast decisions) — the spotlight centre. |
| `draw_ambient(ctx, scene, ch, t, alpha)` |  |
| `whip_offset(el, s)` | Signed offset along the whip axis (scalar along x for text; generic elements use whip_vec). |
| `whip_vec(el, lt)` |  |
| `whip_word(t, wt, size, dur)` | Kinetic word offset (each word rockets in from the right, RTL reading direction). |
| `whip_velocity(el, s, scene)` | Instantaneous whip velocity in px/frame (user units) along the whip axis at the element's state. |
| `wants_whip(el, s)` |  |
| `blur_layer(ctx, draw_fn, length_px, direction, scale)` | Render draw_fn(ctx) into an offscreen layer (same device transform), apply a 1-D directional box |
| `frame_blur(surf, length_px, horizontal)` | Full-frame directional blur in place (whip_pan transition). |
| `whip_pan_blur(scene, t)` | Blur length (px at 1080 wide) of an active whip_pan transition at t, else 0. |

### `mograph.audio`  (sdk/mograph/audio.py)
mograph.audio — soundtrack builder.

<details><summary>details</summary>

```
* SFX are AUTOMATIC: every element's enter time triggers its type's default
  sound (icon→draw, stamp→thud, callout→tick, grid→ticks, counter→roll…).
  Override per element with "sfx": "stamp" | "none" | ...   (no manual cue lists).
* Procedural dark-ambient bed (drone + sparse low piano on chapter changes)
  OR a music file  ("audio": {"music": "bed.mp3"}).
* Voice-over ("audio": {"vo": "narration.wav"}) is mixed on top and music/SFX
  are side-chain DUCKED under the voice automatically.
* Loudness normalised to -14 LUFS (Instagram / YouTube target).

spec["audio"] = {"vo": "vo.wav", "music": null, "ambient": true, "sfx": true,
```
</details>

| function / class | what |
|---|---|
| `sfx(name, seed)` |  |
| `ambient_bed(duration, chapter_times, seed, key_hz)` |  |
| `spectral_carve(bed, env, cfg)` | v9 spectral voice-over carve (HyperFrames /hyperframes-audio): split the bed with zero-phase Butterworth |
| `read_audio(path, duration)` | Decode any audio/video file to mono float32 @ SR via ffmpeg. |
| `collect_cues(scene)` |  |
| `build_soundtrack(scene, out_wav, log)` |  |
| `subtitles_from_srt(srt_path)` | .srt → sentence subtitles [{"t","end","text"}] (words are auto-timed inside each block). |
| `subtitles_from_words(words, max_words, max_gap)` | Word timestamps (e.g. whisper output [{"w"/"text"/"word", "t"/"start", "end"}]) → |

### `mograph.autotrace`  (sdk/mograph/autotrace.py)
mograph.autotrace — SUB-PIXEL MARCHING SQUARES + DOUGLAS-PEUCKER + NEWTON CUBIC-BEZIER AUTO-TRACE

<details><summary>details</summary>

```
& BELONGIE SHAPE-CONTEXT + 2D PROCRUSTES SMART MORPHING
(Ported from EffectCraft `crates/path/src/{trace.rs, fit.rs, interp.rs}`).

1. `trace_scalar_field_to_beziers(field, threshold=0.5, tolerance=1.5, corner_roundness=0.65, min_area=16.0)`:
   1) Sub-pixel linear-interpolated Marching Squares contour extraction (with saddle disambiguation),
   2) Douglas-Peucker polyline simplification to `tolerance` px,
   3) Least-squares Cubic Bezier fitting (`fit_single`) with chord/Catmull-Rom tangent blending.
2. `shape_context_procrustes_morph(pts_a, pts_b, t, n_pts=64, bending_resistance=0.55)`:
   Aligns cyclic vertex order and winding between two shapes via log-polar Shape Context histograms
   (Belongie, Malik & Puzicha PAMI 2002) and interpolates their best-fit 2D Procrustes rotation + scale
   alongside vertex coordinates so turning shapes never collapse through the origin!
```
</details>

| function / class | what |
|---|---|
| `douglas_peucker(pts, tolerance)` | Simplify polyline `pts` (N x 2) using Douglas-Peucker to `tolerance` pixels. |
| `fit_cubic_bezier_span(p0, p3, pts, t0_dir, t1_dir)` | Ported from EffectCraft `crates/path/src/fit.rs` (`fit_single`): |
| `trace_scalar_field_to_beziers(field, threshold, tolerance, corner_roundness, min_area)` | Ported from EffectCraft `crates/path/src/trace.rs`: |
| `shape_context_procrustes_morph(pts_a, pts_b, t, n_pts, bending_resistance)` | Ported from EffectCraft `crates/path/src/interp.rs`: |

### `mograph.beat`  (sdk/mograph/beat.py)
mograph.beat — v12 procedural music bed (no external generator needed): trap-pop / lofi / cinematic beats that

<details><summary>details</summary>

```
follow the EDIT — every section restarts on a cut, risers land exactly on the drop, the ending hits on the last frame.

  "audio": {"beat": {"style": "trap_pop" | "lofi" | "cinematic", "bpm": 100, "key": "A", "scale": "minor",
                     "progression": [0, 5, 3, 4]   (scale degrees per bar → i–VI–iv–v),
                     "sections": [{"t": 0, "part": "intro"}, {"t": 2.4, "part": "verse"},
                                  {"t": 7.8, "part": "build"}, {"t": "abs:12.3", "part": "riser"},
                                  {"t": 13.5, "part": "drop"}, {"t": 16.8, "part": "outro"}],
                     "gain": 0.55, "seed": 3}}
  sections default: verse from 0, drop on the 2nd edit cut, outro at the edit end.
Parts: intro (filtered pluck + hats) · verse (kick, hats, 808, pluck) · build (+clap, +pad) · riser (snare roll,
noise sweep, kick cut) · drop (full, 16th hats, hard 808) · outro (pad + pluck tail, final hit).
```
</details>

| function / class | what |
|---|---|
| `kick(dur, punch)` |  |
| `clap(seed)` |  |
| `hat(open_, seed)` |  |
| `snare(seed)` |  |
| `bass808(f0, dur, glide_from)` |  |
| `pluck(freq, dur, bright)` |  |
| `pad(freqs, dur, cutoff)` |  |
| `riser(dur, seed)` |  |
| `impact(seed)` |  |
| `default_sections(scene)` |  |
| `render_beat(cfg, duration, sections, seed)` |  |
| `beat_path(scene)` |  |

### `mograph.captions`  (sdk/mograph/captions.py)
mograph.captions — v12 reel captions ("Hormozi / CapCut" style), used by hud.draw_subs when

<details><summary>details</summary>

```
hud.subs.style == "reel".

  "hud": {"subs": {"style": "reel", "y": 0.70, "size": 84, "weight": "title",
                   "color": "#ffffff", "emph": "#ffd60a", "active_bg": "#ff2d55", "active_text": "#ffffff",
                   "reveal": "word" | "group",  "stroke": "#000000", "stroke_w": 10, "shadow": 0.55,
                   "pop": 0.18, "max_width": 0.84, "upper": false, "rotate": 0}}

* group enters with a spring pop (scale 0.82 → 1, slight rise), leaves in 0.08 s
* reveal "word": each word pops in on its own timestamp; "group": whole group visible, active word highlighted
* active word: rounded slab in active_bg behind it (springs in), text in active_text
* emphasised words (**word** or edit.emph) stay in the emph colour and are 8 % larger
```
</details>

| function / class | what |
|---|---|
| `draw_reel(ctx, W, H, cfg, sub, t, …)` |  |
| `draw_novadent_jelly(ctx, W, H, cfg, sub, t, …)` | v13.9 @novadent.ir Elastic Jelly Glassmorphic Subtitle Bar (Screenshot 211936): |

### `mograph.catalog`  (sdk/mograph/catalog.py)
Render a visual catalog of every icon (name + tags) so the agent can SEE what exists.

| function / class | what |
|---|---|
| `render_catalog(path, cols, cell)` |  |

### `mograph.cli`  (sdk/mograph/cli.py)  · CLI: `python3 -m mograph.cli …`
python -m mograph <command> spec.json [options]

<details><summary>details</summary>

```
qa       spec.json                         lint + overlap/out-of-frame report (seconds, no video)
  frame    spec.json --t 12.5 [--out f.png]  render ONE frame (instant check of a moment)
  sheet    spec.json [--out sheet.jpg]       contact sheet of hero frames of every chapter
  preview  spec.json [--out p.mp4]           half-res 15fps video, all CPU cores (fast iteration)
  render   spec.json [--out final.mp4]       full quality + soundtrack + QA gate
  review   final.mp4 [--spec spec.json]       post-render check of the REAL video (black/frozen/audio/LUFS)
  align    vo.wav --text script.txt          word timestamps from your narration (needs faster-whisper)
  icons    [query]                           list/search icons (Persian or English)
  catalog  [--out catalog.png]               render a catalog of all icons
  analyze  ref.mp4 [-o blueprint.json] [--into spec.json]   v9: cuts, cadence, palette, LUFS, BPM + beats (b:N)
```
</details>

| function / class | what |
|---|---|
| `main(argv)` |  |

### `mograph.contrast`  (sdk/mograph/contrast.py)
mograph.contrast — scene-aware colour: WCAG contrast maths + background sampling.

<details><summary>details</summary>

```
Used by the HUD (subtitles, header, dots, handle) every frame: the HUD is drawn AFTER the world, so the
renderer samples the REAL pixels under each HUD zone (bg colour, plates, blocks, footage …) and every
HUD colour is passed through `adapt()` against that sample.  A light word can never land on a light
background again, whatever the style, palette or footage.

    ratio(fg, bg)                       WCAG 2.x contrast ratio (1 … 21)
    ensure(fg, bg, min_ratio, prefer)   fg if it passes, else the first preferred colour that passes,
                                        else fg pushed toward black/white until it passes
    dim(fg, bg, target)                 a "quiet" colour (upcoming karaoke words): visible but recessive
    sample(surface, rect)               median RGB of a canvas region (BGRA premultiplied cairo surface)
```
</details>

| function / class | what |
|---|---|
| `luminance(rgb)` |  |
| `ratio(fg, bg)` |  |
| `is_light(bg)` |  |
| `ensure(fg, bg, min_ratio, prefer)` | Return a colour with contrast >= min_ratio against bg (keeps fg's alpha). |
| `dim(fg, bg, target)` | Recessive colour between bg and fg whose contrast is ≈ target (never invisible). |
| `sample(surface, rect, scale)` | Median RGB (0..1) of a canvas rect (x, y, w, h in canvas px before `scale`). |
| `hud_env(surface, W, H, hud, scale)` | Background samples under each HUD zone (computed once per frame, before the HUD is drawn). |
| `mix(a, b, k)` | v9: hex mix of two colours (k = 0 → a, 1 → b). |

### `mograph.core`  (sdk/mograph/core.py)
mograph.core — time, easing, springs, keyframe tracks, camera.

<details><summary>details</summary>

```
Everything here is pure-python and deterministic: a frame at time t is a pure
function of t. That property is what lets the renderer split frames across
CPU cores and lets QA render any single frame instantly.
```
</details>

| function / class | what |
|---|---|
| `clamp(v, lo, hi)` |  |
| `lerp(a, b, p)` |  |
| `lerp_tuple(a, b, p)` |  |
| `prog(t, t0, dur)` | 0..1 progress of a window that starts at t0 and lasts dur seconds. |
| `window(t, t_in, t_out, fade_in, fade_out)` | Envelope 0→1→0 : rises after t_in, falls before t_out (t_out=None → never). |
| `linear(p)` |  |
| `ease_in(p)` |  |
| `ease_out(p)` |  |
| `ease_in_out(p)` |  |
| `ease_out_quart(p)` |  |
| `ease_out_expo(p)` |  |
| `ease_in_out_expo(p)` |  |
| `ease_out_back(p, s)` |  |
| `ease_in_out_sine(p)` |  |
| `bezier_ease(x1, y1, x2, y2)` | CSS cubic-bezier(x1,y1,x2,y2) — e.g. bezier_ease(.2,.8,.2,1) (Motion/Framer default-ish). |
| `get_ease(e)` |  |
| `spring(t, t0, stiffness, damping, mass)` | Damped harmonic spring from 0 → 1 starting at t0 (physical params, like Motion/Framer). |
| `spring_preset(t, t0, name)` |  |
| `stagger(t, t0, idx, step, dur, ease)` |  |
| `wiggle(t, freq, amp, seed)` | Smooth, deterministic idle motion (sum of incommensurate sines). |
| `Track` | Keyframe track: Track([(t, value), (t, value, "ease_name"), ...]) |
| `Camera` | World-space camera. View = (cx, cy, zoom[, rot_deg]). |
| `hex_rgba(h, a)` |  |
| `color(c, alpha_mul)` | Accept '#fff', '#ffffffcc', (r,g,b), (r,g,b,a) with 0..1 floats; multiply alpha. |
| `with_alpha(c, a)` |  |

### `mograph.craft_bridge`  (sdk/mograph/craft_bridge.py)
mograph/craft_bridge.py — FilmCraft + EffectCraft Native Studio Bridge (v14 Master)

<details><summary>details</summary>

```
===================================================================================
Ports the core mathematical, color, 3D geometry, layer-style, and broadcast audio DSP
engines from `storytold/filmcraft` and `storytold/effectcraft` into `mograph`.
```
</details>

| function / class | what |
|---|---|
| `effectcraft_iris_bokeh(bgr, radius, highlight_gain, highlight_thresh, sides)` |  |
| `filmcraft_lumetri_grade(bgr, person_alpha, bg_dim, bg_tint_bgr, bokeh_radius, dof_speaker_blur, …)` |  |
| `composite_speaker_with_auto_light_wrap(canvas_bgra, fg_bgr, person_alpha, wrap_strength, defringe)` |  |
| `filmcraft_compressor_curve_db(x_db, threshold_db, ratio, knee_db)` |  |
| `filmcraft_expander_curve_db(x_db, threshold_db, ratio, range_db)` |  |
| `synthesize_studio_music_bed(duration_s, sr, bpm)` |  |
| `filmcraft_master_reel_audio(raw_mp4_path, segments, sfx_events, out_wav_path, sr)` |  |

### `mograph.creative_reels`  (sdk/mograph/creative_reels.py)
mograph.creative_reels — v13.9 (@novadent 100/100 Master + Creative Viral Reels Engine).

<details><summary>details</summary>

```
Implements pixel-accurate, optical-light-spill, and real backdrop-blur Glassmorphic widgets
from the User's 7 forensic screenshots of @novadent.ir + the 2 companion viral reels:
  1. novadent_split_dome     (Screenshot 211857: Dashed golden semi-circle with real Glassmorphic
                              backdrop-blur dome + split trapezoidal Spartina/Ozempic panels behind speaker)
  2. novadent_cd_dial        (Screenshots 211908 & 211917: Dark glass vinyl/CD donut ring with 2-line
                              'بیشتر از / این که' white core, true curved Persian arc text, top-only red rim
                              casting red light spill onto speaker's white collar -> 180° Ease-In-Out rotation
                              to bottom red rim casting red light spill onto speaker's hands/cuff!)
  3. novadent_diamond_square (Screenshot 211936: Tilted -16° rounded-square wireframe where 'م' of 'مورد اول'
                              and 'خ/ی' of 3D golden 'خشکی' break across the border + 45° rotated Diamond ◇ badge)
```
</details>

| function / class | what |
|---|---|
| `draw_curved_arc_text(ctx, text, cx, cy, radius, size, …)` | Draw polar-warped continuous curved Persian text along a circular arc. |
| `draw_mixed_font_line(ctx, text, cx, cy, base_size, thin_font, …)` | Renders a single horizontal Persian sentence where individual words can have drastically |
| `el_novadent_split_dome(ctx, el, s)` | Screenshot 2026-10-19 211857.png ('آمپول‌های لاغری / اسپارتینا / اوزمپیک'): |
| `el_novadent_cd_dial(ctx, el, s)` | Screenshots 2026-10-19 211908.png & 211917.png ('بیشتر از / این که' Vinyl/CD Ring): |
| `el_novadent_diamond_square(ctx, el, s)` | Screenshot 2026-10-19 211936.png ('مورد اول / خشکی / دهان' + 45° Rotated Diamond Badge): |
| `el_novadent_chair_glow_text(ctx, el, s)` | Screenshot 2026-10-19 212019.png ('ماهایی که / زدیم / چیکار / کنیم'): |
| `el_novadent_edge_light_spill(ctx, el, s)` | Foreground additive/screen optical light wrap from a behind-chair glowing word onto the chair/shoulder edge! |
| `el_novadent_red_halo(ctx, el, s)` | Screenshot 2026-10-19 212032.png ('اصلاً سخت نیست' Concentric Two-Layer Red CD Halo): |
| `el_novadent_glass_hex(ctx, el, s)` | Screenshot 2026-10-19 212043.png ('از آدامس‌های / بدون قند / برای تحریک بزاق استفاده کنید'): |
| `el_orbiting_title_3d(ctx, el, s)` | 3D Head-Orbiting & Bird-Swoop Title Motion: |
| `el_wireframe_badge(ctx, el, s)` | Video 0 (@novadent.ir) Signature Element: |
| `el_circular_dial(ctx, el, s)` | Video 0 (@novadent.ir 00:04-00:08) Signature Element: |
| `el_glass_hud_frame(ctx, el, s)` | Video 1 (00:10-00:12) Signature Element: |
| `el_zigzag_path_3d(ctx, el, s)` | Video 2 (@mh3ydari 00:16-00:20) Signature Element: |
| `el_avatar_carousel_3d(ctx, el, s)` | Video 1 (00:22-00:24) Signature Element: |
| `el_silhouette_scribble(ctx, el, s)` | Video 2 (@mh3ydari 00:08-00:12) Signature Element: |
| `compose_before_after_frame(raw_bgr, after_bgr, before_label, after_label)` | Video 0 (@novadent.ir) Master Layout Composer: |
| `render_before_after_video(raw_mp4, after_mp4, out_mp4, before_label, after_label)` | Render a full @novadent.ir style BEFORE / AFTER split-card showcase video with audio from after_mp4. |

### `mograph.cut`  (sdk/mograph/cut.py)
mograph.cut — v12 "Studio Cut": edit decision list (EDL) for a RAW talking-head take.

<details><summary>details</summary>

```
"edit": {
    "src": "media/raw.mp4",          raw take (footage elements with "edit": true play the EDITED timeline)
    "words": "words.json",           word timestamps of the take  [[w, t0, t1], ...] | [{"w","t","end"}]
    "segments": [[0.45, 8.2], ...]   OR explicit source ranges; omitted → automatic silence/jump cut
    "pad_pre": 0.10, "pad_post": 0.14, "max_gap": 0.35,
    "shots": "auto" | [9.0, 15.6],   camera changes in the raw (segments never straddle one)
    "bpm": 100, "snap": 0.5, "snap_tol": 0.12,   cuts land on the music grid
    "transitions": "whip" | {"shot": "zoom_blur", "jump": "punch"},  "transition_dur": 0.3,
    "emph": ["سود", ...],            emphasised words (captions / kinetic)
    "start": 0.0, "tail": 0.6, "max_words": 4,
```
</details>

| function / class | what |
|---|---|
| `load_words(src, base_dir)` |  |
| `norm(w)` |  |
| `detect_shots(path, thresh)` |  |
| `media_duration(path)` |  |
| `plan_segments(words, src_dur, pad_pre, pad_post, max_gap, shots)` |  |
| `snap_segments(segs, grid, tol, start, last_word)` |  |
| `edit_table(segs, start)` |  |
| `to_edit_time(tab, st)` |  |
| `to_src_time(tab, t)` | Edited time → source time (before the first segment: its start; holds the last frame). |
| `edited_subtitles(words, tab, emph, max_words)` | Short caption groups (reel style: 2–4 words) in edited time; sentence punctuation always breaks. |
| `plan_cutfx(cuts, kinds, cfg)` |  |
| `enhance_video(path, opts, log)` | Cached proxy: temporal denoise → lanczos upscale → contrast-adaptive sharpen (fixes 480p phone footage). |
| `apply_edit(spec, base_dir, log)` |  |
| `report(spec)` |  |

### `mograph.edit`  (sdk/mograph/edit.py)
mograph.edit — v9 surgical voice & video disfluency editor.

<details><summary>details</summary>

```
Real recordings have dead pauses, slips of the tongue (تپق) and repeated fillers («مثلاً... مثلاً...»).
Everything here works on (pcm float32 @SR, words [{w,t,end}]) and returns an EditMap so that every
downstream time (chapters, camera keys, numeric element times, omni windows, talking-head video) moves
through the same remap(t).  Word anchors ("w:…") need nothing: they re-resolve against the new words.

spec["voice"] keys (all optional):
  "max_pause": 0.22        clamp every silence longer than this down to it (mid-sentence AND between lines)
  "min_pause": 0.06        never squeeze a gap below this
  "cut_ranges": [[t0,t1]]  excise ranges (seconds of the SOURCE clip / line audio; per clip: clips[i].cut_ranges)
  "dedupe_fillers": true   remove consecutive duplicate words and filler tokens (also CLI --dedupe-fillers)
```
</details>

| function / class | what |
|---|---|
| `EditMap` | Piecewise map source-time → edited-time built from removed intervals (sorted, non-overlapping). |
| `splice(pcm, removed, xfade_ms, sr)` | Remove [t0,t1] intervals from pcm with raised-cosine crossfades at zero-crossings (de-click). |
| `plan_cuts(words, dur, cfg)` | → removed intervals [(t0,t1)] (source time) from max_pause / cut_ranges / dedupe_fillers. |
| `edit_clip(pcm, words, cfg, sr)` | Apply the voice edit to one clip. Returns (pcm_edited, words_edited, EditMap). |
| `report(removed)` |  |

### `mograph.elements`  (sdk/mograph/elements.py)
mograph.elements — the declarative element registry.

<details><summary>details</summary>

```
Every visual thing in a scene is a dict:
    {"type": "icon", "id": "vial", "name": "vial", "x": 540, "y": 800, "size": 260,
     "t": 35.0, "out": 46.0, "enter": "draw", "exit": "fade"}

Common keys (all optional except type & t):
    id        unique name (callouts/arrows can target it, QA reports it)
    t         enter time (s)           dur      enter duration (type default)
    out       exit start time (s)      out_dur  exit duration (0.45)
    enter     draw | fade | pop | rise | wipe | words | slam | none
    exit      fade | undraw | shrink | drop | none
```
</details>

| function / class | what |
|---|---|
| `F(el, role, key)` | Font for an element: el['font'] / el['weight'] override the element's default ROLE. |
| `LF(el, role)` | Font for secondary labels (label_font). |
| `text_fx(el, s, size)` | v9 — shared typography treatment kwargs for draw_text (text / headline / kinetic / block …): |
| `element(name, dur, enter, sfx, exit)` |  |
| `State` |  |
| `val(el, key, t, default)` |  |
| `compute_state(el, t, scene)` |  |
| `xy(el, t, scene)` | Element position, resolving `target` references. |
| `resolve_point(p, t, scene)` | A point can be [x,y] or "id" or "id+dx,dy" or {"id":..., "dx":..,"dy":..}. |
| `value_at(el, s, default_key, default_from, ease)` | Numeric value of counter/ring.  Either from/to over the enter duration, or a multi-step |
| `value_punch(el, t)` | 0..1 bump right after each value keyframe (scale punch on number changes). |
| `el_icon(ctx, el, s)` |  |
| `el_text(ctx, el, s)` |  |
| `el_box(ctx, el, s)` |  |
| `el_line(ctx, el, s)` |  |
| `el_arrow(ctx, el, s)` |  |
| `el_callout(ctx, el, s)` | Leader line: anchor dot → elbow → shelf, label on the shelf. |
| `el_stamp(ctx, el, s)` |  |
| `el_strike(ctx, el, s)` | False word gets struck through live, then the true word rises above it with a check. |
| `el_counter(ctx, el, s)` | Animated number. fmt: '{:,.0f}', prefix/suffix, fa=True for Persian digits. |
| `el_dot_grid(ctx, el, s)` | Grid of dots (population/statistics). highlight: count of lit dots (from the top-right, RTL), |
| `el_icon_grid(ctx, el, s)` | Repeated icons in a grid/row (crowd, medicines...). names: str or list (cycled). |
| `el_bars(ctx, el, s)` | Bar chart. values, labels (under bars), value labels on top. RTL order by default. |
| `el_ring(ctx, el, s)` | Progress ring (percent / gauge). value 0..1, text in the middle, label below. |
| `el_timeline(ctx, el, s)` | Horizontal timeline with ticks + labels. RTL: first item on the right. |

### `mograph.extract3d`  (sdk/mograph/extract3d.py)
mograph.extract3d — Automatic 3D Studio Sprite Sheet Slicer & Alpha Matte Extractor.

<details><summary>details</summary>

```
Used by `python3 -m mograph extract3d <image_path> --names name1,name2,... [--grid 2x2]`
and dynamically by `@element("tactile_3d")` when given a raw AI-generated image.
```
</details>

| function / class | what |
|---|---|
| `key_3d_rgba(bgr, bg_mode)` | Convert a BGR image of an isolated 3D object on a solid black, green, or white |
| `extract_3d_sheet(src_path, names, grid, bg_mode, out_dir)` | Slice a grid sprite sheet (e.g. 2x2, 1x1, 2x3) and save clean RGBA PNGs into assets_3d. |

### `mograph.fetch_assets`  (sdk/mograph/fetch_assets.py)  · CLI: `python3 -m mograph.fetch_assets …`
Large binaries kept OUT of git (repo must stay < 100 MB) — downloaded once on demand, sha256-verified.

<details><summary>details</summary>

```
python3 -m mograph.fetch_assets          # fetch all (also run by tools/setup_env.sh)
    python3 -m mograph.fetch_assets --check  # show status only

Every file is byte-identical to the copy that used to be committed (sha256 below).
Archived deliverables of v14.2 (old SDK zip, final/showcase mp4) live in the GitHub Release
`v14.2-deliverables` — see `python3 -m mograph.fetch_assets --deliverables`.
```
</details>

| function / class | what |
|---|---|
| `ok(name)` |  |
| `ensure(name, log, verify)` | Return the local path of asset `name`, downloading + verifying it the first time. |
| `try_ensure(name, log)` | Like ensure() but returns None instead of raising (offline → caller uses its fallback). |
| `deliverables(dst)` |  |

### `mograph.fx`  (sdk/mograph/fx.py)
mograph.fx — v11 "Studio Pantheon": real-footage editing & compositing inside the vector engine.

<details><summary>details</summary>

```
ELEMENT  footage   — one clip = a complete edit-suite layer (everything is a pure function of t → frame-exact,
                     worker-safe, cached).  All keys optional except src.
  placement   x y w h (box, cover-fit by default)  fit: cover|contain   radius  border (#hex|true)  border_w
              shadow (true | {"blur":40,"alpha":0.45,"dy":24})   anchor_uv [u,v] (source point placed at the box centre)
  reframe     "frame": [[t, [u, v, zoom]], …]   keyframed virtual camera inside the clip (u,v = source point at centre)
              "reframe": "face" | "<tracker>"   auto-follow (keeps that point at "reframe_at" [0.5,0.4]), "reframe_zoom"
              "punch": [t, …] | "cuts" | "beats"  instant editorial punch-in toggling 1.0 ↔ 1+punch_amt (0.12)
              "zooms": [{"t", "dur", "to": 1.25, "ease": "out_expo", "back": t2}]  smooth push-ins (Hormozi zoom)
              "beat_pump": 0.02   scale bump on every music beat (spec.beats)   "shake": [{"t","dur","amp","freq"}]
  time        clip_start  speed  loop
```
</details>

| function / class | what |
|---|---|
| `to_hue(c)` | '#rrggbb' / [r,g,b] / number(OpenCV hue 0..180) → OpenCV hue. |
| `grade_params(g)` |  |
| `lerp_grade(a, b, p)` |  |
| `apply_grade(fr, g)` | BGR uint8 → graded BGR uint8 (LUT per channel + saturation mix: ~4 ms at 720p). |
| `hue_mask(fr_hsv, h0, tol, smin, vmin, feather)` |  |
| `selective_color(fr, keep_hue, tol, amount, smin)` |  |
| `hue_shift(fr, h_from, h_to, tol, p, within, …)` | Shift pixels of hue h_from toward h_to by progress p (0..1) — mask feathered, optional limit mask. |
| `halftone(fr, cell, ink, paper, amount)` | Dot-screen look: dot radius ∝ darkness (cell px). |
| `build_time_map(el, scene, span)` | Table lt → src_t (60 Hz) for the element's life span (cached on the element). |
| `src_time(el, scene, t)` | Scene time t → source clip time. |
| `normalize_footage(el, scene, ch_t, ch_end)` |  |
| `view(el, scene, t, fw, fh, src_t)` | → dict(x, y, W, H, k, cu, cv): source pixel (px, py) maps to local |
| `map_point(vw, u, v)` | normalised source point → chapter-local coordinates. |
| `auto_push(el, t)` | v12: "auto_push": 0.05 → every edit segment slowly pushes in by 5 % (resets on each cut). |
| `punch_scale(el, scene, t)` |  |
| `zoom_scale(el, t)` |  |
| `pump_scale(el, scene, t)` |  |
| `shake_offset(el, t)` |  |
| `footage_src(el)` |  |
| `footage_audio_src(el)` |  |
| `footage_tracker(el, scene, name)` |  |
| `resolve_follow(scene, ref, t)` | 'clip.hand' → (x, y, scale, rot_deg, quad/None) in the clip chapter's local coords, or None. |
| `follow_xform(scene, el, t)` | For render.draw_world: translation (+rot/scale) that glues an element to a tracker. |
| `auto_levels_params(path, src_t, shots, opts)` | v12: per-shot black/white point + neutral balance measured on 5 frames of that shot (cached). |
| `auto_grade(fr, p, opts)` |  |

### `mograph.genimage`  (sdk/mograph/genimage.py)
mograph.genimage — v11 engines that turn AI-generated images (the agent's native image generator writes .jpg/.png

<details><summary>details</summary>

```
into the workspace) into ANIMATABLE layers.  Everything is preprocessed once and cached next to the image
(<img>.trace.json / <img>.depth/ / <img>.rig/).

ELEMENT auto_trace   line-art image → vector strokes drawn live (trim-path), à la ref07 / ref09 neon line art.
  {"type":"auto_trace","src":"art/elephant.jpg","x":540,"y":900,"w":900,
   "mode":"centerline|outline", "invert":"auto", "thresh":"auto", "min_len":12, "simplify":1.2,
   "color":"#ffffff", "color_mode":"source" (sample the image colours per stroke), "width":3.0, "glow":{"color":"#2ad1ff"},
   "dur":2.5 (draw-on), "order":"top|left|size|center" (stroke order), "stagger":0.6 (0 = all strokes together,
   1 = strictly one after another), "fill_image":0.0..1 (fade the raster in under the strokes at the end)}
  mode centerline = skeleton (thin lines from strokes of any width) · outline = contours of filled shapes.
```
</details>

| function / class | what |
|---|---|
| `trace_image(path, mode, invert, thresh, min_len, simplify, …)` | → {"w","h","paths":[[[x,y],…],…], "colors":[hex…]} normalised to the source size, cached as JSON. |
| `el_auto_trace(ctx, el, s)` |  |
| `depth_layers(path, n, split, work)` | → list of (bgra uint8) layers far→near, cached as PNGs in <img>.depth_<tag>/ |
| `el_depth_plate(ctx, el, s)` |  |
| `cut_sheet(path, grid, parts, bg)` | → dict part → (bgra cropped, (w, h)), cached as PNGs. |
| `auto_joints(cut)` | Joint pivots found from the masks: limbs pivot at their top end, the head at its chin; attach points on |
| `el_sprite_rig(ctx, el, s)` |  |

### `mograph.geometry`  (sdk/mograph/geometry.py)
mograph.geometry — compiled SVG paths with EXACT trim-path (stroke-dashoffset).

<details><summary>details</summary>

```
Why this replaces v3's draw_svg_path:
  * v3 re-tokenised the SVG string on every frame and guessed the path length
    (`approx_len = 460 * scale`), so Trim-Path speed was wrong for every icon
    (short icons finished instantly, long ones never completed until p=1).
  * v3 ignored S/T/A and most relative commands (silently broke imported SVGs).
  * cairo resets dashes at every sub-path, so multi-part icons trimmed oddly.

Here every path string is parsed ONCE (lru cache) into cairo.Path objects
(one per sub-path) together with the true arc length of each sub-path
(computed from cairo's flattened polyline). Drawing is then just
```
</details>

| function / class | what |
|---|---|
| `Compiled` |  |
| `compile_path(d, fillet)` |  |
| `path_bbox(d, fillet)` |  |
| `stroke_trimmed(ctx, comp, trim, mode)` | Strokes `comp` using the CURRENT source/line width, honouring trim. |
| `draw_path(ctx, d, x, y, scale, trim, …)` | Draw an SVG path string (or Compiled) placed with its view-box centre at (x, y). |
| `rounded_rect_d(x0, y0, w, h, r)` |  |
| `circle_d(cx, cy, r, start_deg)` | Circle as 4 cubic arcs starting at start_deg (default: 12 o'clock) clockwise. |
| `polyline_d(pts, closed)` |  |
| `arrow_d(x0, y0, x1, y1, head, curve)` | Arrow from (x0,y0) to (x1,y1). curve≠0 bends it (fraction of length, sign = side). |
| `morph_plan(ds_a, ds_b, n)` | Pair sub-paths of A and B greedily by (centroid distance + size difference). |
| `draw_morph(ctx, ds_a, ds_b, p, x, y, …)` | Draw the in-between shape at progress p (0 = A, 1 = B). Optional `twist` (deg) for flair. |
| `point_on_path(d, p, fillet)` | (x, y, angle_rad) at fraction p of path d (absolute coords) — for particles along curves. |

### `mograph.glowfx`  (sdk/mograph/glowfx.py)
mograph.glowfx — v11 light & energy effects + tracked annotation.

<details><summary>details</summary>

```
GENERIC KEYS (any element, applied by render.draw_world):
  "glow":   true | "#2ad1ff" | {"color": "#2ad1ff", "radius": 18, "strength": 1.4, "pulse": 0}   neon halo of the
            element's own pixels (coloured, additive) — glowing strokes / neon text (ref06, ref09)
  "shadow": true | {"blur": 18, "alpha": 0.45, "dx": 0, "dy": 10, "color": "#000"}   soft drop shadow (photo cards,
            stickers, Hormozi boxes)
  "shiver": 4 | {"amp": 4, "freq": 18, "t": .., "dur": ..}   high-frequency stress jitter (ref09 shaking icons)
  "follow": "<footage>.<tracker>"   (see fx.py)  glue any element to a tracked point.

SCENE / CHAPTER KEYS (render_frame):
  "shake": [{"t": "w:بوم", "dur": 0.35, "amp": 18, "freq": 22}]   camera shake (impact)
```
</details>

| function / class | what |
|---|---|
| `layer_fx(ctx, draw_fn, glow, shadow, t, scale_hint)` | Draw draw_fn(ctx2) into an offscreen layer (same transform), then composite shadow → glow → content. |
| `shiver_offset(el, t)` |  |
| `scene_shake(scene, t)` |  |
| `scene_pump(scene, t)` |  |
| `scene_flash(scene, t)` |  |
| `prepare_scene_fx(scene)` | Resolve chapter/meta shake & flashes to absolute times (called once by Scene). |
| `el_saber(ctx, el, s)` |  |
| `el_flash_pulse(ctx, el, s)` |  |
| `el_glow_ring(ctx, el, s)` |  |
| `el_tracked_callout(ctx, el, s)` |  |
| `el_light_rays(ctx, el, s)` |  |
| `el_orbit_trail(ctx, el, s)` |  |
| `el_emitter(ctx, el, s)` |  |

### `mograph.hbc`  (sdk/mograph/hbc.py)
mograph.hbc — zero-wheel HarfBuzz: the system C library `libharfbuzz.so.0` through ctypes, plus fontTools

<details><summary>details</summary>

```
glyph outlines.  v9 Studio Apex.

Why: the `uharfbuzz` pip wheel is often missing in offline sandboxes, but the native shaper that every
Linux desktop stack (Pango, Chromium, ffmpeg/libass) depends on is always there.  This module exposes the
tiny subset of the uharfbuzz API that `mograph.text` uses, so text.py runs the SAME vector code path
(true glyph outlines → cairo paths, exact ink boxes, karaoke per-word geometry) without any wheel:

    Blob.from_file_path(p) · Face(blob) · face.upem · face.unicodes
    Font(face) · font.get_font_extents("ltr") · font.draw_glyph_with_pen(gid, pen)
    Buffer() · buf.add_str(s) · buf.guess_segment_properties() · buf.direction = "rtl"
    shape(font, buf, features) · buf.glyph_infos[i].codepoint/.cluster · buf.glyph_positions[i].x_advance …
```
</details>

| function / class | what |
|---|---|
| `available()` |  |
| `version()` |  |
| `Blob` |  |
| `Face` |  |
| `Font` |  |
| `Buffer` |  |
| `shape(font, buf, features)` | features are HarfBuzz defaults (kern, liga, Arabic init/medi/fina/rlig are always on). |

### `mograph.horizon`  (sdk/mograph/horizon.py)
mograph.horizon — v8 "Studio Horizon" elements.

<details><summary>details</summary>

```
SWISS DATA-JOURNALISM WIDGETS (grid-aware: "col"/"row" fill x / w / y)
  waffle_grid   10×10 percentage matrix; `value` cells pop to @accent at `fill_at`, the rest recede to @mute,
                flush Swiss callout bars (rule + swatch + label) above / below
                {"type":"waffle_grid","col":[1,6],"row":6,"value":1,"total":100,"fill_at":"w:درصد",
                 "accent_label":"۱٪ ثروتمندترین","base_label":"۹۹٪ بقیه مردم","side":620,"cell_r":0}
  swiss_bars    full-bleed slab bars with cut-out type: label + value INSIDE the slab in @paper, or outside in
                @ink when the slab is too short (decided per bar by real text width + contrast.py)
                {"type":"swiss_bars","col":[1,6],"row":7,"items":[{"label":"نفت","value":72,"t":"w:نفت"},...],
                 "max":100,"suffix":"٪","highlight":0,"bar_h":110,"gap":22,"orient":"h|v"}
  versus_table  2-column editorial comparison: heavy top / middle / bottom rules, centre divider, row-by-row
```
</details>

| function / class | what |
|---|---|
| `roll_width(v, size, weight, fmt, fa, sep, …)` |  |
| `roll_number(ctx, v, x, y, size, color, …)` | Draw value v as an odometer at (x, y) with proportional Persian digit widths so '۰' and '۱' |
| `waffle_grid(ctx, el, s)` |  |
| `swiss_bars(ctx, el, s)` |  |
| `versus_table(ctx, el, s)` |  |
| `waveform(ctx, el, s)` |  |
| `vo_ring(ctx, x, y, size, env, color, …)` | Inner ring that breathes with the voice (money_core / ai_monogram with idle:'vo_pulse'). |

### `mograph.hud`  (sdk/mograph/hud.py)
mograph.hud — persistent screen-space overlay: chapter header notch, chapter dots,

<details><summary>details</summary>

```
word-level karaoke subtitles, handle/watermark.  All configurable via spec["hud"].

spec["hud"] = {
  "header": {"prefix": "پرونده:", "y": 0.055, "show": true},
  "dots":   {"y": 0.82, "style": "dots" | "numbers", "show": true},
  "subs":   {"y": 0.885, "size": 50, "max_width": 0.86, "lines": 2,
             "spoken": "#ffffff", "upcoming": "#4d4d52", "emph": "#ffffff"},
  "handle": "@MY.PAGE"   (optional)
  "fade_out_at": 144.5   (optional, seconds — HUD fades before the end card)
}
Subtitles come from spec["subtitles"]:
```
</details>

| function / class | what |
|---|---|
| `merged_hud(spec)` |  |
| `prepare_subtitles(subs)` | Normalise to: [{"t","end","words":[{"w","t","end","emph"}], "text"}]. |
| `active_subtitle(subs, t)` |  |
| `draw_header(ctx, W, H, cfg, title, alpha, …)` |  |
| `draw_dots(ctx, W, H, cfg, n, idx, …)` |  |
| `draw_subs(ctx, W, H, cfg, sub, t, …)` |  |
| `draw_handle(ctx, W, H, handle, alpha, env)` |  |
| `draw_hud(ctx, W, H, hud, chapters, ch_idx, …)` | env = contrast.hud_env(surface…): background samples per HUD zone (scene awareness). |

### `mograph.icons`  (sdk/mograph/icons.py)
mograph.icons — data-driven multi-layer vector icon library (100×100 view-box).

<details><summary>details</summary>

```
Icons live in `icons.json` (primary + detail layers + Persian/English tags)
so the agent can ADD icons without touching code, and can SEARCH them:

    find_icons("دارو")        -> ['vial', 'pill', 'syringe', ...]
    draw_icon(ctx, "vial", x, y, size=160, p=0.6)          # trim-path reveal
    draw_icon(ctx, "vial", x, y, size=160, p=1, style="ghost")
    add_icon("rocket2", primary=[...], detail=[...], tags=["موشک"], save=True)
    import_svg_icon("/usr/share/icons/.../x.svg", "name")   # normalises to 100 box

`size` is the on-screen size in pixels of the 100-unit box. Line weight is
```
</details>

| function / class | what |
|---|---|
| `icon_names()` |  |
| `find_icons(query, limit)` | Fuzzy search by name or Persian/English tag. |
| `draw_icon(ctx, name, x, y, size, p, …)` | Draw icon `name` centred at (x, y), `size` px wide.  p = trim progress (0..1). |
| `icon_bbox(name, x, y, size)` |  |
| `add_icon(name, primary, detail, tags, save)` | Register an icon (100×100 box). save=True persists it into icons.json (evolution!). |
| `import_svg_icon(svg_path, name, tags, save, margin)` | Import an outline SVG (Lucide/Tabler/Feather/your own) as a library icon. |

### `mograph.keys`  (sdk/mograph/keys.py)
mograph.keys — API keys for the optional cloud services (voice + Omni video).

<details><summary>details</summary>

```
Lookup order: env var → <project>/.mograph_keys.json → ~/.mograph/keys.json → a dropped keys FILE
(mograph_keys.txt / .env / keys.txt / json in the project, home, AI Drive or uploads) — see HOWTO_FA.
Keys are NEVER printed, logged, or written into spec/README/git (.mograph_keys.json is git-ignored).

    python -m mograph keys                      # status (set / missing) — no values shown
    python -m mograph keys set ELEVENLABS_API_KEY sk_...
    python -m mograph keys set GEMINI_API_KEY AIza...

If a key is missing, the engine raises MissingKey whose message is a ready-to-send request
to the user (in Persian) — the agent must ASK the user, never guess or skip silently.
```
</details>

| function / class | what |
|---|---|
| `MissingKey` |  |
| `get_key(name, project_dir, required)` |  |
| `set_key(name, value, scope)` |  |
| `status()` |  |
| `parse_key_text(txt)` | Forgiving: NAME=value / NAME: value / JSON / export NAME="value" / bare keys recognised by shape. |
| `scan_key_files(project_dir)` |  |
| `import_file(path, delete)` |  |

### `mograph.matte`  (sdk/mograph/matte.py)
mograph.matte — v9 offline speaker segmentation (3-tier, automatic) + the depth-layered talking-head rig.

<details><summary>details</summary>

```
Tier 1  onnx     bundled mograph/models/portrait_seg.onnx (MediaPipe Selfie Segmentation, 462 KB, Apache-2.0)
                 via onnxruntime CPU (~25–40 ms/frame) + guided edge feathering + temporal EMA (0.8·a_t + 0.2·a_{t-1})
Tier 2  chroma   YCbCr distance keyer for green / blue screens with spill suppression (key colour auto-sampled
                 from the frame border, or "key": "#00ff00")
Tier 3  grabcut  cv2.grabCut seeded from haarcascade_frontalface_default + torso geometry (+ MOG2 motion mask)
"matte": "auto" (onnx → chroma if the border is a saturated green/blue → grabcut) | "onnx" | "chroma" | "grabcut"

Mattes are cached per (video, frame index, method) on disk next to the video (<video>.matte/NNNNN.png) so
multi-worker renders and re-renders never pay twice.
```
</details>

| function / class | what |
|---|---|
| `onnx_available()` |  |
| `matte_onnx(bgr)` |  |
| `refine(bgr, a, radius)` | Edge-aware feathering: a fast guided filter (He et al.) using the luminance as guide. |
| `is_chroma_plate(bgr)` | A real green/blue screen: the border is strongly saturated in the key hue AND uniform. |
| `matte_chroma(bgr, key, tol, spill)` |  |
| `despill(bgr, key_is_green)` | Clamp the key channel to the average of the other two (classic despill). |
| `matte_grabcut(bgr, key)` |  |
| `matte(bgr, method, key, ema, stream)` | α (float32 HxW 0..1) of the person in a BGR frame.  stream = id for temporal smoothing. |
| `cached_matte(path, idx, bgr, method, key, prev_alpha, …)` |  |
| `premultiplied_surface(bgr, alpha)` | BGR + α → cairo ARGB32 premultiplied surface (+ keep-alive buffer). |
| `rim_surface(alpha, color, width)` | Glow ring around the matte edge (outer rim light). |

### `mograph.media`  (sdk/mograph/media.py)
mograph.media — real footage inside the vector world.

<details><summary>details</summary>

```
1) element "video" — a clip placed like any element (follows camera, enters/exits, QA bbox):
   {"type":"video","src":"media/coin.mp4","x":540,"y":900,"w":720,      # or "fill":true (whole screen)
    "t":"w:سکه","dur":0.4,"out":"w:ولی","clip_start":0.0,"speed":1.0,"loop":false,
    "blend":"normal|screen|multiply","grade":"none|mono|noir","radius":24,"border":true}
   blend "screen" on a black background = only the bright parts show → footage glows into the line art.
   grade "noir" = mono + crushed blacks + soft contrast → matches the ENSAN noir style.

2) plates (used by mograph.omni) — a clip that REPLACES (or sits UNDER) the vector world between two
   absolute times; HUD/subtitles/post-FX stay vector-crisp on top.
```
</details>

| function / class | what |
|---|---|
| `reader(path)` |  |
| `grade(fr, mode)` |  |
| `to_surface(fr_bgr)` |  |
| `paint_frame(ctx, fr, x, y, w, h, …)` | Paint a BGR frame centred at (x, y) with width w (height from aspect, or cover-crop into w×h). |
| `el_video(ctx, el, s)` | Footage clip (see module doc). |
| `active_plate(scene, t)` |  |
| `plate_alpha(p, t)` |  |
| `draw_plate(ctx, scene, p, t)` |  |
| `punch_zoom(scene, el, t)` | 1.0 ↔ 1+punch alternating at every jump cut (spec._cuts) → no visible jump cut. |
| `el_talking_head(ctx, el, s)` |  |

### `mograph.obj3d`  (sdk/mograph/obj3d.py)
mograph.obj3d — v14.4 core 3D object renderer (pure NumPy + OpenCV + Cairo, CPU, no GPU).

<details><summary>details</summary>

```
Replaces the v13.9 prototypes in sam3d_bridge.py (which are kept as thin wrappers) with ONE correct,
fast renderer:

  render_extruded_sprite()   any RGBA sprite (assets_3d/*.png, image_gen output) → solid 3D object:
                             • correct perspective (front face nearer than back face — v13.9 had it inverted)
                             • front / back face visibility (past 90° you see the BACK, not a mirrored front)
                             • gap-free side wall: slice count follows the projected depth in px (no stripes)
                             • rounded profile (bevel → cushion) by distance-field inset (no Lanczos moiré)
                             • per-pixel wall normals + key/rim light + moving specular sheen on the face
                             • works at OUTPUT resolution → 5-10× faster than v13.9 (≈40-90 ms @ 300 px)
```
</details>

| function / class | what |
|---|---|
| `resolve_asset(src, base_dir)` | Find an asset file: absolute path, relative to the scene/project dir, cwd, or mograph/assets_3d. |
| `load_rgba(path)` | BGRA uint8 (straight alpha).  3-channel images are keyed (black / white / green background). |
| `find_views(src, base_dir)` | Auto-detect multi-view siblings:  laptop_3d.png → laptop_yaw_m45.png, laptop_yaw_p15.png … |
| `rot_matrix(yaw, pitch, roll)` |  |
| `paint_premul(ctx, pm, x0, y0, alpha)` | Paint a premultiplied float/uint8 BGRA array at (x0, y0) on a cairo context. |
| `contact_shadow(ctx, cx, cy, w, alpha, strength, …)` |  |
| `render_extruded_sprite_array(path, size, yaw, pitch, roll, depth, …)` | Return (premultiplied float32 BGRA D×D, D) of the sprite turned into a solid 3D object. |
| `render_extruded_sprite(ctx, src, cx, cy, size, yaw, …)` |  |
| `render_flat_sprite_array(bgra, size, yaw, pitch, roll, F)` |  |
| `view_weights(angs, yaw, dissolve, switch)` | Which real views are visible at `yaw` → [(index, weight, residual_yaw)]. |
| `render_multiview(ctx, views, cx, cy, size, yaw, …)` |  |
| `motion_angles(el, lt, yaw0, pitch0)` | Shared yaw/pitch motion presets for every 3D element. |
| `audit_sheet(src, out_png, angles, cell, pitch)` | 3D QA sheet: one row per geometry (slab / cushion / multiview) for an image, or the sculpt3d preset |

### `mograph.omni`  (sdk/mograph/omni.py)
mograph.omni — Gemini Omni Flash bridge: realistic shots where vector code is the wrong tool.

<details><summary>details</summary>

```
The engine stays the director: it renders its OWN frames (exact layout/camera/timing) and hands them
to Omni as the input; Omni returns a realistic clip that is laid back onto the timeline as a PLATE.
Vector HUD, karaoke subtitles and post-FX stay crisp on top.  If the clip is missing, the vector
version renders unchanged (safe fallback) — QA tells you.

Spec (top level, times are absolute seconds or word anchors):
  "omni": [{
     "id": "coin_real",
     "t": "w:سکه", "end": "w:سکه+5",          # 3..10 s
     "task": "edit",          # edit        : engine frames t..end → Omni restyles them (keeps motion/layout)
```
</details>

| function / class | what |
|---|---|
| `upload_file(path, key, base, mime)` | Gemini Files API resumable upload → file uri (waits until ACTIVE). |
| `interact(prompt, key, images, video, task, aspect, …)` | One Interactions API call → (mp4 bytes, interaction_id). |
| `resolve_jobs(scene)` |  |
| `plan(scene)` |  |
| `run(spec_path, yes, only, provider, log)` |  |
| `load_plates(scene)` | Spec omni jobs whose clip exists → scene.plates (used by render_frame). |

### `mograph.pbr3d`  (sdk/mograph/pbr3d.py)
mograph.pbr3d — v14.6 PHOTOREAL 3D objects: real scanned/PBR models (glTF/GLB/OBJ/FBX/STL) rendered by the

<details><summary>details</summary>

```
EffectCraft Advanced-3D software rasteriser (Cook-Torrance GGX, HDRI image-based lighting, PCF shadow maps, 2x2 SSAA).

WHY (the v14.5 problem): real3d/sculpt3d build objects from procedural primitives + procedural noise textures →
they look like "3D icons" (clean, plastic, synthetic).  Photoreal needs (a) real geometry detail, (b) real
PBR texture maps (albedo / normal / roughness-metal / AO from photogrammetry) and (c) image-based lighting from a
real HDRI.  pbr3d gets (a)+(b) from free CC0 scans (Poly Haven, 500+ models, `fetch`) or any glTF the agent
generates/imports, and (c) from EffectCraft's renderer — a single static 75 MB CLI binary, no GPU, ~270 MB RAM,
~0.15–0.4 s per 600 px frame on 2 vCPU.  (github.com/storytold/effectcraft, MIT/Apache-2.0)

Design for a low-resource agent sandbox:
  * one CLI call renders a WHOLE pose sequence (per-frame yaw/pitch/roll baked into an expression lookup table),
```
</details>

| function / class | what |
|---|---|
| `ec_bin()` | Path to effectcraft-cli (bundled in sdk/bin, $EFFECTCRAFT_CLI, PATH, or auto-download once). |
| `search(query, category, kind)` | Search the Poly Haven CC0 catalogue → [(id, name, categories)].  kind = models / hdris. |
| `fetch(asset, res, kind)` | Download a Poly Haven model (glTF + 1k PBR maps, typically 0.5–3 MB) or HDRI.  Returns the local path. |
| `resolve_model(src, base_dir, auto_fetch)` | Model reference → local file.  Accepts a path (abs / project-relative / models dir) or a Poly Haven id. |
| `prepare_model(src, base_dir)` | Normalise a model so its bounding sphere is centred on the origin with radius 1 (predictable size + pivot). |
| `hdri_path(name)` |  |
| `render_sequence(model, size, poses, hdri, light, env_rot, …)` | Render poses [(yaw, pitch[, roll]), …] → lazy list of float32 frames H×W×5 = (R, G, B, alpha, shadow): |
| `render_pose(model, size, yaw, pitch, roll)` |  |
| `to_bgra_premul(fr, shadow_strength, shadow_color, shadow_gain)` | frame (RGB, alpha[, shadow]) → premultiplied BGRA float 0..255 (object over its own cast shadow). |
| `paint(ctx, fr, cx, cy, alpha, shadow_strength)` |  |
| `comp_over(img_bgr_f, fr, cx, cy, shadow_strength)` | Composite one frame over a float BGR image (0..1) in place (numpy path, no cairo). |
| `audit_sheet(src, out, size, bg, hdri)` |  |
| `track(model, size, angles_fn, t0, t1, fps)` | Pre-render a whole on-screen interval at once: angles_fn(local_t) → (yaw, pitch, roll). |
| `paint_scaled(ctx, fr, cx, cy, scale, alpha, …)` | cairo paint of a frame centred at (cx, cy) with an extra scale (springs / pops without re-rendering). |

### `mograph.promo`  (sdk/mograph/promo.py)
mograph.promo — v9 Studio Apex: kinetic-promo, 3D and SaaS-UI mockup widgets (pure JSON, zero scripting).

<details><summary>details</summary>

```
Colours default to the chapter world's palette tokens (@accent @ink @paper @mute @panel).

  liquid_blob     organic morphing blob r(θ,t)=R(1+a1·sin(3θ+ω1t)+a2·cos(4θ−ω2t)); optional "text": base text in
                  @ink outside, the overlapping part INVERTED inside (clip to the blob)
                  {"type":"liquid_blob","x":540,"y":900,"r":300,"text":"شکل‌های سیال","size":130,"drift":[120,0]}
  iso_cube        true 3D rotating cube / prism, back-face culled, Lambert-shaded (@accent lit top)
                  {"type":"iso_cube","x":540,"y":820,"s":300,"rx":-28,"ry":38,"wy":40,"shape":"cube|prism","label":"…"}
  starburst       rotating spiky impact badge {"type":"starburst","x":540,"y":800,"r":330,"points":16,"text":"شوخیت!"}
  prompt_box      glass AI prompt bar: label chip, neon border + bloom, RTL typewriter, caret, send button pulse
                  {"type":"prompt_box","x":540,"y":900,"w":900,"h":200,"label":"Claude","prompt":"…","type_at":"w:…"}
  timeline_ui     NLE window mock: traffic lights, layer bars, keyframes ◆, scrubbing playhead, cursor,
```
</details>

| function / class | what |
|---|---|
| `blob_points(x, y, R, t, a1, a2, …)` |  |
| `liquid_blob(ctx, el, s)` |  |
| `iso_cube(ctx, el, s)` |  |
| `starburst(ctx, el, s)` |  |
| `prompt_done_time(el)` |  |
| `prompt_box(ctx, el, s)` |  |
| `timeline_ui(ctx, el, s)` |  |
| `bezier_graph(ctx, el, s)` |  |
| `viewfinder_hud(ctx, el, s)` |  |
| `filmstrip(ctx, el, s)` |  |
| `sine_wave(ctx, el, s)` |  |

### `mograph.prompts`  (sdk/mograph/prompts.py)
mograph.prompts — v11: turn a spec into the generation jobs an agent must run with its OWN platform tools

<details><summary>details</summary>

```
(native image generator, music model, TTS), and report which media files are still missing.

    python3 -m mograph prompts spec.json            → prints a plan + writes <spec>.prompts.json
    python3 -m mograph prompts spec.json --out p.json

Sources of jobs
  1. any element whose src file does not exist and that has a  "gen": {"prompt": "...", "aspect": "9:16"}
     (footage / image_card / auto_trace / depth_plate / sprite_rig / planar_insert)
     → image job (for footage: a video job; the agent may also substitute a still + depth_plate).
     Type-specific prompt suffixes are added automatically (line art on black for auto_trace, a 3×2 parts sheet on
     green for sprite_rig, clear foreground/background separation for depth_plate …).
```
</details>

| function / class | what |
|---|---|
| `plan(spec_path)` |  |
| `run(spec_path, out, log)` |  |

### `mograph.puppet_arap`  (sdk/mograph/puppet_arap.py)
mograph.puppet_arap — SIGGRAPH 2005 AS-RIGID-AS-POSSIBLE (ARAP) 2D PUPPET MESH DEFORMATION

<details><summary>details</summary>

```
(Ported from EffectCraft `crates/effects/src/puppet/{arap.rs, mesh.rs}`).

Implements Igarashi, Moscovich & Hughes, "As-Rigid-As-Possible Shape Manipulation" (SIGGRAPH 2005):
  1. Triangulates an RGBA image's alpha silhouette (or grid) into a 2D triangle mesh `(V_rest, tris)`.
  2. Step 1 (Laplacian / Similarity solve): solves the sparse symmetric positive-definite system with
     soft/hard positional pin constraints `(pin_idx, target_xy, weight)` and optional `Starch` triangle
     stiffness weights.
  3. Step 2 (2D Procrustes Scale-Adjustment step): fits each deformed triangle with a rigid rotated
     copy of its rest shape via 2D Procrustes rotation about its centroid, then solves the Poisson edge-vector
     system so triangles rotate rigidly without unnatural shearing or volume loss!
```
</details>

| function / class | what |
|---|---|
| `ArapPuppetMesh` | Pre-built 2D triangle mesh with cached rest edge vectors for fast per-frame ARAP deformation. |

### `mograph.qa`  (sdk/mograph/qa.py)
mograph.qa — automatic quality gates (run BEFORE spending time on a full render).

<details><summary>details</summary>

```
lint(scene)            static checks: unknown types, missing glyphs (▯), bad targets,
                       word anchors, timing beyond duration, HUD-zone intrusions.
overlap_report(scene)  samples the timeline and reports overlapping element bboxes
                       (text-on-text / text-on-icon collisions) + out-of-frame content
                       — computed from REAL vector geometry, not by eyeballing.
contact_sheet(scene, path, times=None, cols=5)   one JPG grid of key frames with timestamps,
                       sampled at every chapter "hero" moment (≈ end of each chapter).
probe(path)            ffprobe summary of the final mp4 (duration / fps / size / audio).

All return plain lists/dicts so the agent can read them and fix issues in the spec.
```
</details>

| function / class | what |
|---|---|
| `lint(scene)` |  |
| `overlap_report(scene, step, min_ratio)` | Sample every `step` seconds; report text collisions and out-of-frame elements. |
| `hero_times(scene)` |  |
| `contact_sheet(scene, path, times, cols, thumb_w)` |  |
| `probe(path)` |  |
| `text_audit(scene, step, min_ratio)` | Render the timeline at a coarse step with text.AUDIT on → real problems only: |
| `report(scene, overlaps, audit)` | Human-readable QA summary; returns (text, n_errors). |
| `review_video(path, expect_duration, expect_fps, sample_step, sheet_path)` | Decode the rendered mp4 and check: duration/fps, black stretches, frozen stretches |

### `mograph.real3d`  (sdk/mograph/real3d.py)
mograph.real3d — v14.5 TRUE 3D product renderer (textured meshes, deferred PBR, shadow map, SSAO, bump).

<details><summary>details</summary>

```
Why: a single image "thickened" (image_to_3d slab/cushion) is not a 3D object — it breaks as soon as it turns.
real3d rebuilds the object from REAL geometry (img2threejs idea: component tree of procedural primitives) and gives
every part a procedural texture whose palette is sampled from the reference photo.  Every angle (360° yaw, any pitch,
top, bottom, back) is a real view of a closed solid.

Pipeline per frame (CPU, numba z-buffer from zraster.py):
  meshes (pos, normal, uv, local pos) → rotate → perspective G-buffer (2× SSAA) → per-pixel materials
  (albedo / roughness / metalness / height) → screen-space bump → lights (key + fill + rim, hemisphere ambient,
  studio softbox environment for reflections) → key-light SHADOW MAP (toppings shadow the cheese, the arrow shadows
  the card) → SSAO → ACES tone map → sRGB → downsample (anti-aliased alpha).
```
</details>

| function / class | what |
|---|---|
| `noise_tex(seed, n, sigma)` |  |
| `samp(tex, u, v)` | Bilinear, wrapping sample of a 2D texture at texture-repeat coords (u, v) (any shape). |
| `fbm(u, v, oct, seed, gain)` |  |
| `smooth(e0, e1, x)` |  |
| `srgb()` | sRGB 0..255 → linear 0..1 (np array). |
| `Mesh` | v (n,3) n (n,3) uv (n,2) f (m,3).  Local coordinates (texture space = local position). |
| `grid_faces(rows, cols, wrap)` | Quads of a rows×cols vertex grid → triangles. |
| `lathe(profile, segs)` | Surface of revolution around +Y.  profile = [(r, y), ...] from bottom centre → outward → top centre. |
| `from_sculpt(v, f, n)` | Wrap a sculpt3d mesh; orient normals outward (flip if most point inward). |
| `rbox(w, h, d, bevel, subdiv)` |  |
| `prism(shape, depth)` |  |
| `cyl_z(r, depth, segs, bevel)` | Cylinder along Z (disc / joint / button), optional rounded edge. |
| `xform(m, R, t)` | Apply a rigid transform to a mesh's positions/normals but KEEP its texture space (uv + local pos). |
| `rot(rx, ry, rz)` | R = Rz · Rx · Ry (degrees). |
| `env(R, rough)` | Studio environment radiance for reflection vector R (n,3), roughness (n,). |
| `aces(x)` |  |
| `Model` | parts = [(Mesh, material_name)];  materials = {name: fn(ctx) -> dict} |
| `render(model, size, yaw, pitch, roll, ssaa, …)` | → (rgba float32 D×D×4 straight alpha 0..1, D).  `size` = on-screen diameter of the bounding sphere. |
| `to_bgra_premul(rgba)` | straight RGBA float 0..1 → premultiplied BGRA float 0..255 (for obj3d.paint_premul / cairo). |
| `paint(ctx, model_name, cx, cy, size, yaw, …)` |  |

### `mograph.real3d_models`  (sdk/mograph/real3d_models.py)
mograph.real3d_models — procedural TRUE-3D product models (geometry + procedural textures sampled from the

<details><summary>details</summary>

```
reference photo).  Every model is a closed solid: it can turn 360° / be seen from above, below and behind.

  soda_can      lathe body (dome bottom, stand ring, chime, neck, double-seam rim, recessed lid) + pull tab + rivet,
                brushed aluminium + condensation droplets (bump + gloss)
  pizza_slice   polar height-field slab (melted cheese lumps, sauce at the edges, baked dough underside and cut walls),
                puffy lumpy crust tube with pores, pepperoni (cupped, oily, fat specks), mushroom slices, olive ring,
                green pepper strip
  growth_card   two stacked rounded cards (matte plastic) with embossed UI bars / dots and an extruded, bevelled
                zig-zag growth arrow

Reference palette (sampled from work/input/three_objects/ref.jpg): crust (177,109,62) cheese (223,189,143)
```
</details>

| function / class | what |
|---|---|
| `vnormals(v, f)` |  |
| `oriented(v, f, expect)` | Mesh with geometric normals, flipped (normals AND winding) so they agree with `expect` (n,3) or (3,). |
| `merge()` |  |
| `rbox(w, h, d, b, n_in, n_b)` | Rounded box with ANALYTIC normals and dense samples inside the bevel band (smooth highlights). |
| `bevel_prism(outer, inner, depth, b)` | Extrude a 2D polygon along +Z with a 45° chamfer.  `inner` = the same polygon inset by b (same vertex |
| `stroke_arrow(pts, w, head_w, head_len)` | Polyline → simple polygon of a thick stroke with an arrow head at the end (miter joins). |
| `disc_y(r, t, bevel, dome, cup, segs)` | Flat disc on the XZ plane (axis +Y): rounded edge, optional dome (top bulge) / cup (rim raised). |
| `torus_y(R, r, su, sv)` |  |
| `cells(u, v, nu, nv, seed, density)` | Cellular dots: for each point → (distance to the cell's dot centre / cell size, dot radius frac, present). |
| `soda_can()` |  |
| `pizza_slice()` |  |
| `growth_card()` |  |
| `get_model(name)` |  |

### `mograph.reelfx`  (sdk/mograph/reelfx.py)
mograph.reelfx — v13 "Visual-DNA & Creative Director" (Zero-Box Edition).

<details><summary>details</summary>

```
Implements the Visual-DNA Motion & Procedural Asset Engine aesthetic laws:
  * A1. Engineered Calm & Full-Environment Gridded Backdrops: Procedural 0.6px technical coordinate
    grids with '+' intersection tick marks, diffused radial key-fill lighting, and f/2.8 Depth-of-Field
    speaker defocus (`dof_blur`) — completely replacing ugly rectangular boxes!
  * A2. Tactile Volume & True 3D Perspective Homography Spin (`_blit_3d_perspective`): All hero objects
    (`logo_apple_3d.png`, `logo_surface_3d.png`, `logo_hp_3d.png`, `logo_dell_3d.png`, `laptop_3d.png`,
    `grade_a_3d.png`, `grade_b_3d.png`, `grade_c_3d.png`, `plane_3d.png`, `pin_3d.png`, `pin_green_3d.png`,
    `cash_3d.png`, `warn_3d.png`, `shield_3d.png`) are isolated 3D tactile renders that rotate in 3D space
    with real perspective homography and soft SSAO contact shadows.
  * A3. One Thread of Color: Procedural parametric vector rails, coordinate arcs, and callout tracks.
```
</details>

| function / class | what |
|---|---|
| `el_title_card(ctx, el, s)` | v13 Zero-Box Visual-DNA Title Composition: |
| `el_logo_chip(ctx, el, s)` | v13 Zero-Box 3D Spinning Brand Emblems: |
| `el_route_arc(ctx, el, s)` | v13 Zero-Box Visual-DNA Flight Path (`امارات` -> `ایران`): |
| `el_price_tag(ctx, el, s)` | v13 Zero-Box 3D Tactile Profit Counter (`+۸ میلیون` + `سود خالص هر لپ‌تاپ`): |
| `el_grade_scale(ctx, el, s)` | v13 Zero-Box Tactile 3D Grade Blocks (`grade_a_3d.png`, `grade_b_3d.png`, `grade_c_3d.png`): |
| `el_profit_meter(ctx, el, s)` |  |
| `el_sticker(ctx, el, s)` | v13 Full-Screen Hero Statement over Blurred/Matte Speaker (`dof_blur` + Zero Box!): |
| `el_cta_arrow(ctx, el, s)` | v13 Zero-Box CTA: Free-floating handwritten/3D typography + glowing procedural Bezier arrow. |
| `el_light_leak(ctx, el, s)` |  |
| `el_shine_sweep(ctx, el, s)` |  |
| `el_progress_bar(ctx, el, s)` |  |
| `el_vignette_pulse(ctx, el, s)` |  |

### `mograph.render`  (sdk/mograph/render.py)
mograph.render — frame renderer + parallel segmented encoder.

<details><summary>details</summary>

```
render_frame(scene, t, scale=1.0) -> cairo.ImageSurface (BGRA premultiplied)
frame_to_rgb(surface) -> HxWx3 uint8 RGB numpy array
render_video(scene, out, preview=False, workers=None, start=0, end=None)

Speed design
------------
* every frame is a pure function of t  → frames are split into N contiguous
  chunks; each worker process renders its chunk and pipes BGRA straight into
  its OWN ffmpeg (no pickling of frames, no PNGs on disk), then segments are
  concatenated with `-c copy` and the audio is muxed once.
```
</details>

| function / class | what |
|---|---|
| `portal_radius(scene, reg)` | Iris radius that exactly covers the frame from the focal point at p = 1. |
| `draw_world(ctx, scene, t, collect, paint_bg)` | Draw all chapter elements at time t (camera already applied by caller). |
| `chapter_rest_view(scene, ch)` | Camera view the chapter was authored against (its arrival view) — parallax is measured from it. |
| `parallax(scene, el, t, ch)` | (dx, dy, scale, pivot_x, pivot_y) in chapter-local coords for an element with "z" != 0, else None. |
| `parallax_bbox(bb, pz)` |  |
| `render_frame(scene, t, scale, hud, bloom, plates)` |  |
| `frame_to_rgb(surf)` |  |
| `save_frame(scene, t, path, scale)` |  |
| `auto_workers(scale)` | CPU count, capped by RAM (each worker = cairo frame + its own x264 encoder). |
| `render_video(scene, out_path, preview, workers, start, end, …)` |  |

### `mograph.sam3d_bridge`  (sdk/mograph/sam3d_bridge.py)
mograph.sam3d_bridge — Native CPU Renderer for `facebookresearch/sam-3d-objects` Exports (`.ply` Gaussian Splats & `.obj` Meshes).

<details><summary>details</summary>

```
While the heavy PyTorch/CUDA DiT inference of `facebookresearch/sam-3d-objects` requires an external
NVIDIA GPU (32 GB VRAM) to reconstruct an object from a photo (`output["gs"].save_ply("splat.ply")`),
its exported `.ply` (3D Gaussian Splat / Point Cloud) and `.obj` (3D Mesh) files CAN be rendered,
lit, and rotated 360° in real time inside `mograph` using pure vectorized NumPy + PyCairo!

Usage in `spec.json`:
  {
    "type": "sam3d_object",
    "id": "obj1",
    "src": "assets_3d/splat.ply",
```
</details>

| function / class | what |
|---|---|
| `load_ply_or_obj(path)` | Loads either: |
| `el_sam3d_object(ctx, el, s)` | Renders a true 3D `.ply` Gaussian Splat (exported from `facebookresearch/sam-3d-objects`) |
| `reconstruct_3d_from_image(src_name, work_res, depth_amount)` | Converts ANY single 2D RGBA image (e.g. generated by `image_gen` or in `assets_3d/`) |
| `export_image_to_ply(src_name, out_ply, work_res, depth_amount)` | Exports the reconstructed 3D volumetric model of a 2D image to a standard `.ply` 3D file! |
| `render_reconstructed_3d(ctx, src_name, cx, cy, size, yaw_deg, …)` | Renders a single-image reconstructed 3D Volumetric Model at `(cx, cy)` with: |
| `el_image_to_3d(ctx, el, s)` | `image_to_3d` — ANY 2D RGBA image (assets_3d/*.png, image_gen output, project path) shown as a solid 3D |
| `render_machined_3d_slab(ctx, src_name, cx, cy, size, yaw_deg, …)` |  |
| `render_true3d_inflated_cushion(ctx, src_name, cx, cy, size, yaw_deg, …)` |  |
| `render_multiview_turntable_3d(ctx, views, cx, cy, size, yaw_deg, …)` |  |
| `render_machined_3d_slab_v139(ctx, src_name, cx, cy, size, yaw_deg, …)` | LEGACY v13.9 (kept for A/B only).  Known bugs: inverted perspective (back face projected larger than the |
| `infer_depth_anything_onnx(bgr_img, model_path)` | Cloned from `fabio-sim/Depth-Anything-ONNX` (`infer.py`) for `onnx-community/depth-anything-v2-small` (27.3 MB): |
| `bilateral_normal_integration_cpu(normal_map, mask, k, max_iter, tol)` | Adapted from `xucao-42/bilateral_normal_integration` (ECCV 2022 Official CPU Implementation): |
| `render_true3d_sphere(ctx, tex_rgba, cx, cy, radius, yaw_deg, …)` | Mode 1 (`geometry="sphere"`): |
| `render_true3d_cylinder_capsule(ctx, tex_rgba, cx, cy, radius, height, …)` | Mode 2 (`geometry="capsule"` / `"cylinder"`): |
| `render_true3d_inflated_cushion_v139(ctx, src_name, cx, cy, size, yaw_deg, …)` | LEGACY v13.9 (kept for A/B only — slow, moiré on the side wall; use render_true3d_inflated_cushion). |
| `render_multiview_turntable_3d_v139(ctx, views, cx, cy, size, yaw_deg, …)` | Mode 4 (`geometry="turntable_4view"`): |

### `mograph.scene`  (sdk/mograph/scene.py)
mograph.scene — compile a JSON/dict spec into a renderable Scene.

<details><summary>details</summary>

```
Spec skeleton (see examples/demo_spec.json and SPEC_REFERENCE.md):
{
  "meta": {"w":1080,"h":1920,"fps":30,"duration":40,"bg":"#000000","bloom":0.4,"out":"out.mp4"},
  "hud": {...}, "audio": {...}, "subtitles": [...],
  "plugins": ["my_elements.py"],
  "chapters": [
     {"t": 0, "title": "من و شما",
      "transition": {"type": "pan", "dir": "down", "dur": 0.9},   # pan | fade | cut
      "camera": [[0, [540, 960, 1.0]], [4, [540, 760, 1.35], "smooth"]],   # LOCAL coords/time
      "elements": [ {...}, ... ]}
```
</details>

| function / class | what |
|---|---|
| `Scene` |  |
| `load_spec(path_or_dict)` |  |

### `mograph.sculpt3d`  (sdk/mograph/sculpt3d.py)
mograph.sculpt3d — v14 `img2threejs`-Compatible Procedural 3D `ObjectSculptSpec` Engine &

<details><summary>details</summary>

```
Native Python Z-Buffered Triangle-Mesh PBR Renderer.

Inspired by `https://github.com/img2threejs/img2threejs` ("Rebuild the object in a reference image
as a code-only, procedural, quality-gated, animation-ready 3D model"):
  * Instead of warping a flat 2D bitmap or requiring a 32GB-VRAM GPU, objects are constructed from
    a hierarchical 3D component tree (`ObjectSculptSpec`) of true 3D triangle-mesh primitives:
      - `box` / `rounded_box` (3D beveled box with 6 faces + chamfered edges)
      - `cylinder` / `cone` (3D cylinder/frustum with top/bottom caps)
      - `sphere` / `ellipsoid` (3D UV sphere / ellipsoid)
      - `torus` (3D torus ring)
      - `lathe` (3D surface of revolution spun 360° from a 2D profile `[(r, y), ...]`)
```
</details>

| function / class | what |
|---|---|
| `mesh_lathe(profile_ry, segs_u)` | Revolves a 2D profile `[(r0, y0), (r1, y1), ...]` 360° around the Y axis to create a |
| `mesh_cylinder(r_top, r_bot, height, segs)` | 3D Cylinder / Cone / Frustum centered at origin along Y axis with top and bottom caps. |
| `mesh_capsule(radius, height, segs, cap_rings)` | 3D Capsule (cylinder + hemispherical top and bottom domes). |
| `mesh_sphere(rx, ry, rz, segs_u, segs_v)` | 3D Sphere / Ellipsoid with radii `(rx, ry, rz)`. |
| `mesh_torus(R, r, segs_u, segs_v)` | 3D Torus ring in the X-Z plane (or X-Y plane). |
| `mesh_rounded_box(w, h, d, bevel, subdiv)` | 3D Rounded/Beveled Box `(w, h, d)` with subdivided faces and smooth chamfered edges |
| `build_node_geometry(node, t, parent_R, parent_T)` | Recursively evaluates a hierarchical `ObjectSculptSpec` node and its `children` at time `t`, |
| `render_sculpt3d_array(root_node, scale_px, t, yaw_deg, pitch_deg, roll_deg, …)` | Rasterise a hierarchical ObjectSculptSpec → (BGRA uint8 premultiplied D×D, D). |
| `render_sculpt3d_scene(ctx, root_node, cx, cy, scale_px, t, …)` | Render a hierarchical ObjectSculptSpec 3D model at (cx, cy) — see render_sculpt3d_array. |
| `spec_laptop_pro_3d(lid_open_t0, lid_open_dur)` | `img2threejs`-style hierarchical `ObjectSculptSpec` + Step 2c Texture Projection for a 3D Ultrabook Laptop: |
| `spec_ozempic_pen_3d(cap_open_t0, cap_open_dur)` | `img2threejs`-style hierarchical `ObjectSculptSpec` for a 3D Medical Injection Pen (Ozempic/Spartina): |
| `spec_chalice_trophy_3d()` | `img2threejs`-style Lathed 3D Golden Trophy Goblet + Orbiting Torus Ring + Glowing Core Sphere! |
| `el_sculpt3d(ctx, el, s)` | Declarative `mograph` Element `@element("sculpt3d")`: |
| `resolve_material(mat)` |  |
| `mesh_extrude(shape, depth, bevel)` | Extrude a 2D polygon `[(x, y), ...]` (object units, CCW or CW) along Z — logos, badges, keys, plates. |
| `get_tree(tree, preset, base_dir, open_dur)` | Resolve a sculpt tree: dict / path to an ObjectSculptSpec .json / preset name. |
| `validate_tree(tree, strict)` | Deterministic gate (img2threejs `validate_sculpt_spec --strict-quality` idea): returns (errors, warnings). |
| `json_key(d)` |  |
| `silhouette_iou(render_bgra, ref_bgra)` | IoU of the two alpha silhouettes after fitting both bounding boxes to the same square (scale/position |
| `fit_pose(tree, ref_bgra, t, yaws, pitches)` | Camera match (img2threejs `solve_camera_pose` idea): coarse grid → local refine of yaw/pitch that maximises |
| `comparison_sheet(tree, ref_path, out_png, yaw, pitch, t, …)` | img2threejs `make_comparison_sheet` equivalent: reference / render at the reference angle / silhouette |

### `mograph.shatter3d`  (sdk/mograph/shatter3d.py)
mograph.shatter3d — 3D VORONOI/BRICK SHATTER PHYSICS & 3D CARD DANCE WAVE MATRIX

<details><summary>details</summary>

```
(Ported from EffectCraft `crates/effects/src/sim2.rs` — `shatter` & `card_dance`).

1. `render_shatter_3d`:
   Fractures any BGRA image/text layer into cellular 3D extruded shards that explode outward from
   `force_center` with 3D Euler angular velocity, gravity, and surface-normal shading!
2. `render_card_dance_3d`:
   Slices any BGRA image/card into a `rows x cols` grid of 3D tiles that tilt and displace in Z
   driven by a travelling sine/gradient wave with specular lighting!
```
</details>

| function / class | what |
|---|---|
| `render_shatter_3d(ctx, bgra, cx, cy, t, rows, …)` | Ported from EffectCraft `crates/effects/src/sim2.rs` (`shatter`): |

### `mograph.showcase`  (sdk/mograph/showcase.py)
mograph.showcase — v11 editor-showcase / social-reel layout widgets (refs 02–06, 08).

<details><summary>details</summary>

```
CONTAINER
  device_frame  rounded "phone" card that CONTAINS other elements:  "children": [ …any elements… ]
                Children are authored in a virtual screen vw×vh (default = scene size: author them exactly like a
                full-screen reel) and are scaled into the frame (w×h) and clipped to its rounded corners.
                keys: w h radius border border_w fill shadow(true|dict) label label_pos(top|bottom) label_size
                      label_color vw vh tilt glass  + normal enter/exit/tracks/follow.

BACKGROUND
  backdrop      full-frame procedural background (texture cached, ~1 ms/frame):
                kind: paper (teal crumpled paper, ref02-04) | tiles (dark diamond tiles, ref05) | smoke (ref06) |
```
</details>

| function / class | what |
|---|---|
| `shape_path(ctx, shape, x, y, w, h, …)` |  |
| `soft_shadow(ctx, x, y, w, h, r, …)` |  |
| `expand_children(els)` | Scene hook: flatten `children` of containers into the chapter list (child gets parent=<id>). |
| `apply_parent(ctx, xf)` | xf = (cx, cy, k, tilt_deg, x0, y0, w, h, r, vw, vh, alpha) written by the parent this frame. |
| `el_device_frame(ctx, el, s)` |  |
| `draw_frame_border(ctx, el)` | Renderer calls this AFTER the children: border + glass sheen on top of the content. |
| `make_texture(kind, W, H, c1, c2, seed)` |  |
| `el_backdrop(ctx, el, s)` |  |
| `el_badge_button(ctx, el, s)` | 3D pill button with a bottom lip ("ادیت ویدیو"). press_at = seconds after t → button presses down. |
| `draw_ig_glyph(ctx, x, y, s, color, alpha, …)` | Instagram-style glyph: rounded square + lens + dot. gradient=True → brand-gradient tile + white glyph. |
| `el_handle_tag(ctx, el, s)` |  |
| `el_social_outro(ctx, el, s)` | End card: dims everything drawn so far, gradient logo springs in, handle wipes in, logo turns white. |
| `el_nle_timeline(ctx, el, s)` | Editing timeline. mode clips (ref05): coloured clips slide under a fixed I-beam; mode segments (ref04): |
| `el_image_card(ctx, el, s)` | Photo / clip card. shape rect/rounded/circle/hex/octagon/pill, kenburns, tilt, border, shadow, focus [u,v]. |
| `el_redact_text(ctx, el, s)` | Text hidden under a censor bar that retracts at reveal_at (s after t). |
| `el_radial_formula(ctx, el, s)` | ref08 hub-and-nodes formula. nodes [{label, icon, t}] on a circle around an octagon hub; when a node's |

### `mograph.sound`  (sdk/mograph/sound.py)
mograph.sound — EVENT-LOCKED sound design (v7).

<details><summary>details</summary>

```
Old engine: one fixed sample at the element's start time → sounds felt random (a counter "rolled"
1.4 s before the digits moved; a 1.1 s drawing got a 0.5 s scribble).
New engine: every element type has a *score* that reads the SAME timing maths the renderer uses and
returns events exactly where the eye sees something happen:
  * pen strokes last as long as the stroke and follow its speed (ease derivative)
  * counters tick on every frame the SHOWN number changes, pitch rises, lock click when it settles
  * stamps hit at the CONTACT frame (spring reaches 1) with a short air pre-roll
  * grids tick per item at their stagger time; bars/rings sweep with their growth
  * transitions whoosh with the camera (peak at max velocity)
  * kinetic words hit on their word; emphasised words hit harder
```
</details>

| function / class | what |
|---|---|
| `syn(kind, dur, seed, curve, pitch, palette)` |  |
| `speed_curve(ease_name, n)` |  |
| `first_time(f, t0, t1, step)` |  |
| `Ev` |  |
| `score()` |  |
| `palette_of(scene)` |  |
| `chapter_palette(scene, ch)` | v8: sound palette of ONE chapter (its style world), meta.sound / element "sound" still override. |
| `collect(scene)` |  |
| `render_bus(scene, n)` | → (L, R) float32 stereo SFX bus of n samples. |
| `score_report(scene)` |  |
| `parse_lines(text)` |  |
| `cutfx_events(scene)` | v12: edit-point transitions get their sound (zoom/whip = whoosh into the cut, glitch = blip, flash = hit). |

### `mograph.styles`  (sdk/mograph/styles.py)
mograph.styles — three house styles as presets, so the agent writes ONE word ("meta.style") and gets a

<details><summary>details</summary>

```
coherent look + motion + typography + sound palette.  Everything stays overridable in the spec.

  lineart  (default)  noir vector line-art: black, white strokes, bloom, grain, pen sounds
  swiss               International Typographic Style: strict grid, flat colour fields (paper/red/black),
                      huge grotesque type flush to the grid, rules, numbers as graphics, hard structural
                      motion (wipe / slide / mask), no glow, crisp UI sounds
  kinetic             kinetic typography: the narration's words ARE the picture — slam / stack / scale /
                      scroll on the spoken word, bold contrast, camera punches, punchy hits

Use:  "meta": {"style": "swiss", "palette": "red"}      (palette: red | blue | yellow | mono)
Elements from every style can be mixed in one video; the preset only sets defaults.
```
</details>

| function / class | what |
|---|---|
| `palette(spec)` |  |
| `resolve_color(v, pal)` | '@accent' / '@ink' / '@paper' / '@mute' → hex; anything else unchanged. |
| `world_for(user_meta, user_hud, ch)` | v8 — the complete 'style world' of one chapter: style, fitted palette, bg, font roles, sound palette, |
| `apply(spec)` | Mutates spec in place (before Scene compiles it): fills style defaults, resolves @colour tokens |
| `fit_palette(pal, bg)` | Background awareness for the palette itself: if the real bg disagrees with the palette's paper |

### `mograph.text`  (sdk/mograph/text.py)
mograph.text — fully VECTOR Persian/English text engine.

<details><summary>details</summary>

```
HarfBuzz shaping → glyph outlines → cairo paths.  Consequences:
  * text is a real vector: stays razor-sharp under any camera zoom/rotation
    (v3 blitted raster sprites → blurry when the camera zoomed in);
  * exact per-word geometry → exact karaoke highlighting, per-word animation,
    and exact bounding boxes for automatic overlap QA;
  * mixed Persian + English + digits + symbols in ONE string, with a small
    built-in bidi resolver (digits stay LTR inside RTL, parentheses mirror,
    '@handle' does not jump to the wrong side);
  * per-character font fallback (Vazirmatn → DejaVu → Noto Symbols2), and
    `missing_glyphs()` so QA can guarantee "no ▯ tofu boxes" before rendering.
```
</details>

| function / class | what |
|---|---|
| `apply_style_roles(overrides)` | Per-video typography: meta.fonts = {"title": "peyda", "body": "medium", ...} (role → weight key / path). |
| `use_roles(fonts)` | v8: switch the active font roles WITHOUT clearing the shaping caches (caches are keyed by the |
| `concrete_font(weight)` | role / weight key / path → absolute font path (what is actually painted). |
| `resolve_font(font, weight, default)` | 'font' (role or weight key) wins over 'weight'. Unknown → default. |
| `register_weight(name, path)` | Add your own font: register_weight('title', '/path/Font.ttf'). |
| `to_fa_digits(s)` |  |
| `parse_markup(text)` | '**شما** انسان‌ها' → ('شما انسان‌ها', {0}). Emphasis is per space-separated word. |
| `fallback_chars(text, weight)` | Visible chars the chosen font lacks → they would render in a fallback font (off-brand). |
| `missing_glyphs(text, weight)` |  |
| `Line` |  |
| `Layout` |  |
| `layout_text(text, size, weight, max_width, line_height, align)` | Shape + wrap. Lines are wrapped by words in LOGICAL order (correct for RTL). |
| `measure_text(text, size, weight, max_width, line_height)` |  |
| `line_ink(ln)` | (x0, top, x1, bottom) of the real glyph outlines of a shaped line, EM units, relative to the |
| `ink_box(text, x, y, size, weight, align, …)` | Canvas bbox of the actual painted glyphs (what the eye sees). |
| `mask_rect(text, x, y, size, weight, align, …)` | Clip rectangle for a 'rise from a mask' reveal that NEVER cuts dots or descenders: |
| `text_bbox(text, x, y, size, weight, align, …)` |  |
| `word_boxes(text, x, y, size, weight, align, …)` | {word_idx: (x0,y0,x1,y1)} in canvas coords — use to point callouts at a specific word. |
| `draw_text(ctx, text, x, y, size, color, …)` | Draw vector text.  `word_style(i) -> dict(color=?, alpha=?, dy=?, scale=?)` styles words |
| `text_style(style, size)` | Normalise a text style → dict or None (solid fast path). |
| `kashida_text(text, word, after)` | Insert ONE tatweel at the first legal join of `word` (or of the first word that has one) unless the |

### `mograph.track`  (sdk/mograph/track.py)
mograph.track — v11 offline motion tracking (Motion Tracking / Point Track / Planar Track / body landmarks).

<details><summary>details</summary>

```
Every tracker runs ONCE per (video, settings), on a downscaled copy of the clip (≤ 540 px wide), and is cached as
JSON next to the video: <video>.track/<kind>_<hash>.json.  Coordinates are stored NORMALISED (u, v in 0..1 of the
source frame) so the same track works at any placement / zoom of the footage.

Kinds (all declared inside a `footage` element:  "trackers": {"name": {...}}):
  point    {"kind":"point", "at":[u,v], "frame_t": 2.0, "win": 0.06}
           Lucas–Kanade feature cloud around a point, forward + backward from frame_t, forward-backward error
           check, re-seeding — "text stuck to a hand / a product".  Output: x, y, scale, rot.
  box      {"kind":"box", "box":[u0,v0,u1,v1], "frame_t": 0}         median-flow box (centre + scale + rotation)
  planar   {"kind":"planar", "quad":[[u,v]x4 TL,TR,BR,BL], "frame_t": 0}
```
</details>

| function / class | what |
|---|---|
| `Track` | Per-frame samples (normalised).  sample(src_t) interpolates linearly; outside the analysed range the |
| `cache_path(path, spec)` |  |
| `video_info(path)` |  |
| `OneEuro` | One-Euro filter (Casiez 2012): smooth at rest, responsive when moving. |
| `track_features(path, spec, info, mode)` |  |
| `track_landmarks(path, spec, info, kind)` |  |
| `track_face(path, spec, info)` |  |
| `track_color(path, spec, info)` |  |
| `hue_mask(hsv, h0, tol, smin, vmin)` | uint8 mask of pixels whose hue is within ±tol (OpenCV 0..180 hue, wraps around red). |
| `get_track(path, spec, log)` |  |
| `debug_sheet(path, tracks, out, n, thumb_w)` | Contact sheet of the clip with every track drawn (cross + label, planar quads) — eyeball the result. |

### `mograph.transfx`  (sdk/mograph/transfx.py)
mograph.transfx — v12 edit-point transitions applied to the composited frame (graphics included, HUD excluded).

<details><summary>details</summary>

```
Events come from the EDL (spec._cutfx, see mograph.cut) or are authored:  "cut_fx": [{"t": 7.8, "type": "zoom_blur"}]
Each event is centred on its cut time t and spans dur (default 0.3 s): the first half accelerates the outgoing shot,
the second half decelerates the incoming one (frame-exact: a pure function of t).

types
  zoom_blur   radial zoom-through (scale 1 → 1.35 → 1 + radial blur)            the classic reel "zoom transition"
  whip        horizontal whip pan (offset + directional motion blur), "dir": 1|-1
  spin        roll ±18° + zoom + blur
  punch       hard cut with a 1.12 → 1 scale settle (jump-cut cover)
  flash       white exposure flash peaking at the cut
```
</details>

| function / class | what |
|---|---|
| `active(scene, t)` |  |
| `apply(scene, surf, t, scale)` |  |

### `mograph.typo`  (sdk/mograph/typo.py)
mograph.typo — Swiss (International Typographic Style) + Kinetic Typography elements (v7).

<details><summary>details</summary>

```
GRID (any element):  "col": [start, span]  "row": r        (meta.grid = {"cols":6,"rows":12,"margin":72,"gutter":24})
   → x/w come from the column grid (RTL: column 1 is on the RIGHT), y from the baseline rows.

SWISS
  block      flat colour field that wipes in from a side (mask), optional big text flush inside
             {"type":"block","col":[1,6],"row":2,"rows":3,"color":"@accent","side":"right","text":"…","text_color":"@paper"}
  headline   huge grotesque type, each LINE rises from behind its own mask (the Swiss signature reveal)
             {"type":"headline","text":"قیمتِ\nواقعیِ\nنفت","col":[1,6],"row":2,"size":180,"align":"right","stagger":0.12}
  rule       thick rule that extends from one side   {"type":"rule","col":[1,6],"row":5,"weight":10,"color":"@ink"}
  big_number giant numeral (value track supported), unit + caption on the grid
```
</details>

| function / class | what |
|---|---|
| `grid_cfg(scene)` | v9: the Swiss grid adapts to the aspect — rows span the same fraction of the height, landscape gets |
| `grid_place(el, scene)` | Fill x / w / y from col/row (called once at scene compile). |
| `pal_of(el, s)` | v8: the palette of the element's OWN chapter world (falls back to the scene palette). |
| `bg_of(el, s)` |  |
| `el_block(ctx, el, s)` |  |
| `el_headline(ctx, el, s)` | Each line rises from behind its own mask, staggered. |
| `el_rule(ctx, el, s)` |  |
| `el_big_number(ctx, el, s)` |  |
| `el_index(ctx, el, s)` |  |
| `el_disc(ctx, el, s)` |  |
| `kinetic_words(el, scene)` | [(word, t, end, emph)] — matched to the narration words (subtitles) after el.t, else staggered. |
| `el_kinetic(ctx, el, s)` |  |
| `masonry_layout(el, words, fnt)` | Bento packing: split the words into 2–4 rows (or el["rows"] = [n1, n2, …] words per row) and scale |

### `mograph.uiphysics`  (sdk/mograph/uiphysics.py)
mograph.uiphysics — v10 Studio Omni: 5th style world ("ui_physics") & Interactive Tactile UI / Physics

<details><summary>details</summary>

```
Choreography widgets (inspired by Apple / Linear / Rube-Goldberg UI motion films).

Widgets registered here:
  * ui_toggle              Frameless iOS/macOS pill toggle switch + vector cursor click & spring recoil
  * ui_slider              Horizontal progress track + tick marks + overhead rolling odometer tooltip bubble
  * ui_notification_stack  Stack of frosted dark/light notification cards with spring entrance & ball-impact compression
  * ui_keyboard            Mechanical QWERTY + stepped ISO Return keyboard with per-key spring depression & highlight
  * ui_reply_pill          Morphing Reply pill → live typing input field with caret & send button → sent message bubble
  * physics_ball           Continuous squash-and-stretch hero ball hopping across waypoints/UI widgets with bounceArc
  * ui_film_world          Complete 1-line Rube-Goldberg UI chain-reaction scene (Toggle → Slider → Cards → Keyboard → Loop)
```
</details>

| function / class | what |
|---|---|
| `smoothstep(u)` |  |
| `smootherstep(u)` |  |
| `ease_out_cubic(u)` |  |
| `ease_in_quad(u)` |  |
| `ease_out_quad(u)` |  |
| `spring_step(dt, omega0, zeta)` |  |
| `spring_impulse(dt, omega0, zeta)` |  |
| `bounce_arc(x0, y0, x1, y1, u, apex_lift)` |  |
| `rebound_drop_arc(x0, y0, x1, y1, u, rebound_h)` |  |
| `el_ui_toggle(ctx, el, s)` |  |
| `el_ui_slider(ctx, el, s)` |  |
| `el_ui_notification_stack(ctx, el, s)` |  |
| `el_ui_keyboard(ctx, el, s)` |  |
| `el_ui_reply_pill(ctx, el, s)` |  |
| `el_physics_ball(ctx, el, s)` |  |
| `el_ui_film_world(ctx, el, s)` | Renders the complete, reference-accurate 10s Rube Goldberg UI Motion Film world |

### `mograph.varstroke`  (sdk/mograph/varstroke.py)
mograph.varstroke — VARIABLE-WIDTH TAPERED & WAVE STROKES + PER-VERTEX VARIABLE MASK FEATHER

<details><summary>details</summary>

```
(Ported from EffectCraft `crates/path/src/varstroke.rs` & `crates/path/src/feather.rs`).

1. `Taper` & `Wave`:
   Builds a variable-width stroke along any polyline/curve as an exact union of positively-wound
   trapezoid segments and round-join discs, so hand-drawn underlines, strike-throughs, circles,
   and callout leaders taper smoothly to a calligraphic brush tip (`start_width -> 1.0 -> end_width`).
2. `variable_feather_mask`:
   Interpolates signed inner/outer feather radii (`FeatherPoint`) along a closed contour's arc length
   so one side of a mask can be razor-sharp (`0.5px`) while the opposite side feathers over `45px`.
```
</details>

| function / class | what |
|---|---|
| `Taper` | Ported from EffectCraft `varstroke::Taper`. |
| `Wave` | Ported from EffectCraft `varstroke::Wave`. |
| `draw_tapered_stroke(ctx, pts, base_width, color_rgba, taper, wave, …)` | Render a polyline `pts` (N x 2) with variable-width `Taper` and `Wave` profile as a union of |
| `resample_polyline(pts, step, max_pts)` | Uniform arc-length resampling of an open polyline (keeps both end points). |
| `polyline_slice(pts, s0, s1, closed, step)` | Sub-path between arc lengths s0 → s1 (wraps around a closed contour) — comets / trim-paths. |
| `variable_feather_mask(W, H, poly_pts, vertex_radii)` | Ported from EffectCraft `crates/path/src/feather.rs`: |

### `mograph.visual_dna`  (sdk/mograph/visual_dna.py)
mograph.visual_dna — v13.5 Visual-DNA & Zero-Box Procedural Asset Engine.

<details><summary>details</summary>

```
Implements the 4 signature Visual-DNA elements:
  dna_stage        Engineered Calm coordinate grid environment + key-fill lighting + telemetry
  dna_connector    Single-thread @accent vector structural device (caliper_rail, circuit_bus, etc.)
  tactile_3d       Isolated 3D studio object with 3D perspective homography spin & zero-box callouts
  hud_loading_bar  Game-style cybernetic segmented loading bar (the ONLY allowed rectangular graphic!)
```
</details>

| function / class | what |
|---|---|
| `el_dna_stage(ctx, el, s)` | Engineered Calm coordinate grid stage with + intersection ticks, diffused key-fill |
| `el_dna_connector(ctx, el, s)` | Procedural single-thread @accent vector structural connector: |
| `el_tactile_3d(ctx, el, s)` | Universal Isolated 3D Studio Object with true 3D perspective homography rotation, |
| `el_hud_loading_bar(ctx, el, s)` | Game-Style Cybernetic Segmented Loading Bar — the ONLY rectangular UI graphic |
| `generate_dna_plan_markdown(topic)` | Generate a 5-part Visual-DNA production plan + ready-to-run 3D asset generation prompts. |

### `mograph.voice`  (sdk/mograph/voice.py)
mograph.voice — natural narration (VO) from the spec's own subtitles, with frame-exact word timing.

<details><summary>details</summary>

```
Providers (REST only, no SDKs):
  elevenlabs  eleven_v3 (Persian, audio tags [whispers] [sarcastic] …) → character timestamps → exact words
  gemini      Google AI Studio TTS (gemini-3.1-flash-tts-preview …, 30 voices, style prompt) → words aligned
              by whisper if installed, otherwise by syllable weight inside the trimmed clip
  draft       offline, free: speech-shaped placeholder audio with realistic durations (timing preview)

Spec:
  "voice": {"provider": "elevenlabs", "voice": "<voice_id>", "model": "eleven_v3", "lang": "fa",
            "style": "calm, ironic documentary narrator",   # gemini: delivery prompt
            "stability": 0.5, "speed": 1.0, "pause": 0.25, "gain": 1.0}
```
</details>

| function / class | what |
|---|---|
| `tts_elevenlabs(text, cfg, prev_text, next_text, project_dir)` |  |
| `tts_gemini(text, cfg, style, project_dir)` |  |
| `tts_draft(text, cfg)` | Free offline placeholder with realistic Persian pacing (~ 2.6 words/s) — for timing previews. |
| `synth_line(line, cfg, provider, cache_dir, prev_text, next_text, …)` | One subtitle line → (pcm trimmed, words with DISPLAY spelling, times relative to clip start). |
| `file_lines(cfg, subs, base)` | → {line_index: (pcm, words_relative)} cut from the external clips at word boundaries. |
| `voice_spec(spec_path, provider, out_path, log, dedupe_fillers)` |  |
| `list_voices(provider, project_dir)` |  |

### `mograph.widgets`  (sdk/mograph/widgets.py)
mograph.widgets — v5 compound narrative widgets (@ENSAN.BESHAM signature rigs).

<details><summary>details</summary>

```
Each is ONE declarative element that would otherwise need 10-30 primitives + careful timing:

  quote_rig   forensic quote card; words get underlined / circled with numbered badges on cue
  flow_loop   closed-loop flow graph (institution ↔ people) with travelling particles + flip pill
  laser_scan  legal document tablet; a laser beam sweeps and lights up one clause
  invoice     receipt ledger: item ....... value rows appear on cue, "?" values, final stamp
  strata      ground cross-section: surface icons, underground layers, drilling pipe → "?"
  scale_rig   two-pan balance; items drop on cue, stack, and the beam tilts with spring physics

All times inside lists ("highlights", "rows", "drops") accept word anchors ("w:...") —
```
</details>

| function / class | what |
|---|---|
| `quote_rig(ctx, el, s)` | keys: text, source, w, size, font(document), highlights:[{word/index, t, style: underline/circle/box, |
| `flow_loop(ctx, el, s)` | keys: top {icon,label}, left {icon,label}, right {icon,label}, w, h, speed, particles, |
| `arrow_d_on(d, head)` | Append an arrow head to the end of a Q/L path string. |
| `laser_scan(ctx, el, s)` | keys: title, clause, w, h, lines, scan_at (time), scan_dur, clause_row (0..lines-1). |
| `invoice(ctx, el, s)` | keys: title, rows: [{item, value, t}], total {label, value, t}, stamp {text, t}, w. |
| `strata(ctx, el, s)` | keys: w, surface: [icon names], layers (int), resources (icon name or None), drill_at, drill_x, |
| `scale_rig(ctx, el, s)` | keys: left {label, item}, right {label}, drops: [{t, label}], max_tilt (deg), w. |

### `mograph.zraster`  (sdk/mograph/zraster.py)
mograph.zraster — tiny per-pixel Z-buffered triangle rasteriser (Gouraud colour, ID buffer) for the CPU 3D

<details><summary>details</summary>

```
engines (sculpt3d).  numba-JIT when available (≈ 5-15 ms for 2-5 k triangles at 600²), pure-NumPy fallback.

raster(tri_xy (N,3,2) px, tri_z (N,3) camera distance, tri_rgb (N,3,3) 0..255, tri_id (N,), W, H)
    → color (H,W,3) float32, alpha (H,W) float32 (coverage 0/1), ids (H,W) int32 (0 = empty)
Nearest surface wins (smaller z).  Pixel centres at +0.5.  Interpolation is perspective-correct for depth.
```
</details>

| function / class | what |
|---|---|
| `raster(tri_xy, tri_z, tri_rgb, tri_id, W, H)` |  |
| `raster_attr(tri_xy, tri_z, tri_attr, tri_id, W, H)` | → attr (H,W,K) float32, ids (H,W) int32 (0 = empty), zbuf (H,W) (1e30 = empty). |


## tools/ — shell & maintenance scripts

### `batch_validate_aecomp`  (sdk/tools/batch_validate_aecomp.py)
DEV TOOL: validate the whole Shape-Elements reference pack in one go (parallel, per-file timeout).

<details><summary>details</summary>

```
python3 tools/batch_validate_aecomp.py /path/to/NM_Library_Shapes_AE -o /tmp/batch.jsonl [--j 4] [--only Arcs]
    python3 tools/batch_validate_aecomp.py --report /tmp/batch.jsonl        # summary per category + worst items

For every Shape_Elements/<cat>/<n>.aep with a Previews/Shape_Elements/<cat>/<n>.mp4 it imports the comp
(tools/import_aep.py), measures IoU + F3 (tools/validate_aecomp.py) and appends one JSON line.
Nothing from the pack is written into the repo — only numbers.
```
</details>

| function / class | what |
|---|---|
| `one(aep, mp4, n)` |  |
| `report(path, worst)` |  |
| `main(argv)` |  |

### `build_aefx_catalog`  (sdk/tools/build_aefx_catalog.py)
tools/build_aefx_catalog.py — MEASURE every motion / text preset → agent-readable catalogue

<details><summary>details</summary>

```
(sdk/mograph/aefx/presets/catalog.json + CATALOG.md).  The agent picks presets from measured data, not names.

motion: dur, ends_identity, peak_scale, peak_rot, travel (frame widths), blur, energy → tags
        soft / punchy / spin / 3d / blur / distort / travel / overshoot / settles-offset / stays-visible
text:   dur, props, units, random / wiggly / 3d / blur / decode tags
    python3 tools/build_aefx_catalog.py
```
</details>

| function / class | what |
|---|---|
| `motion_stats(p)` |  |
| `text_stats(p)` |  |
| `main()` |  |

### `build_aefx_gallery`  (sdk/tools/build_aefx_gallery.py)
tools/build_aefx_gallery.py — static, searchable HTML gallery of all 703 AE-FX presets (animated WebP previews).

<details><summary>details</summary>

```
python3 tools/build_aefx_gallery.py [--out ../public/aefx] [--only motion|text]
    → public/aefx/index.html (+ media/*.webp, ~7 MB, git-ignored; regenerate any time, ~2 min)

Served by the Hono app at /aefx/ (wrangler pages dev) or open the file directly.  Each card shows the measured tags
from catalog.json and a copy-ready JSON snippet ({"motion": key} / "preset": key).
```
</details>

| function / class | what |
|---|---|
| `motion_webp(key, path)` |  |
| `aecomp_webp(name, path, lower_third, title)` |  |
| `text_webp(key, path, text)` |  |
| `main(argv)` |  |

### `build_ref_templates`  (sdk/tools/build_ref_templates.py)
Builds templates/11..19 — one starter spec per reference video (reference_videos/ref01..ref09).

<details><summary>details</summary>

```
Run:  python3 tools/build_ref_templates.py     (then: python3 -m mograph qa templates/1x_*.json)
Every template uses only shipped media (examples/media/…) so it renders out of the box; swap "src" for the
user's own footage / generated images (keep the ids: trackers and behind-layers refer to them).
```
</details>

| function / class | what |
|---|---|
| `save(name, spec)` |  |

### `build_tools_index`  (sdk/tools/build_tools_index.py)
Regenerate sdk/TOOLS_INDEX.md — the machine-built catalogue of EVERY script / module / CLI / asset in the SDK.

<details><summary>details</summary>

```
python3 sdk/tools/build_tools_index.py          (run after adding or changing any tool — it is cheap)

How it works: reads the module docstring of every .py under sdk/mograph, sdk/mograph/v14 and sdk/tools,
lists the public functions (with their first doc line) and the CLI entry points, plus models / binaries / fonts.
The hand-written part (what to use when, gaps, how to extend) lives in sdk/SKILL_TOOLBOX.md; this file is the
always-correct inventory underneath it.  If a module has no docstring, it is flagged "UNDOCUMENTED" → fix it.
```
</details>

| function / class | what |
|---|---|
| `main()` |  |

### `import_aep`  (sdk/tools/import_aep.py)
DEV TOOL (not needed to use the SDK): convert an After Effects .aep composition into mograph.aecomp JSON.

<details><summary>details</summary>

```
Used to STUDY reference packs (timing, easing, structure) and to VALIDATE our renderer against the vendor's own
preview videos (tools/validate_aecomp.py).  Imported vendor comps are NOT shipped in the repo — the SDK ships our own
generators (mograph/aecomp/generators.py) fitted to the measured numbers.

    pip install py_aep
    python3 tools/import_aep.py file.aep [-c COMP] [-o out.json]

Supported: shape layers (group, rect, ellipse, polystar, path, fill, stroke, gradient fill/stroke, trim, repeater,
zig zag, twist, round corners, pucker & bloat, offset, merge), null/solid/text layers, parenting, track mattes,
blend modes, keyframes (speed/influence, per-dimension, spatial tangents, hold/linear), and the common expressions
```
</details>

| function / class | what |
|---|---|
| `Ctx` |  |
| `keyed(q, X, dims, scale)` | property → value (static / keys / control link). |
| `translate_expr(e, base, X, dims, scale)` |  |
| `P(group, mn)` |  |
| `tr_group(g, X, layer)` |  |
| `dashes(g, X)` |  |
| `grad_stops(q, X)` |  |
| `item(q, X)` |  |
| `controls(L, X)` |  |
| `convert(comp)` |  |
| `main(argv)` |  |

### `import_motion_presets`  (sdk/tools/import_motion_presets.py)
tools/import_motion_presets.py — DEV-ONLY: Premiere "Motion Presets" .prproj templates → our compact JSON library

<details><summary>details</summary>

```
(sdk/mograph/aefx/presets/motion_presets.json).  Reads ONLY keyframe numbers (reference data); no vendor code/media.

    python3 tools/import_motion_presets.py "<pack>/templates/Motion Presets"

Each preset = ordered stack of adjustment-layer effects (Premiere track order, bottom → top), time normalised to 0:
    {"name", "cat", "dir": in|out|fx, "dur", "fx": [{"fx": transform|basic3d|fast_blur|wave_warp|twirl, "p": {...}}]}
```
</details>

| function / class | what |
|---|---|
| `index_objects(s)` | top-level objects (tab-indented once; closing tag '\n\t</Tag>') by ObjectID / ObjectUID. |
| `parse_prproj(path)` |  |
| `normalise(fx)` |  |
| `main(argv)` |  |

### `import_text_presets`  (sdk/tools/import_text_presets.py)
tools/import_text_presets.py — DEV-ONLY: AtomX "Text Animation Presets" (.atom = JSXBIN) → our JSON animator library

<details><summary>details</summary>

```
(sdk/mograph/aefx/presets/text_presets.json).  Only the animator NUMBERS are read as reference data.

    /tmp/jsxer/bin/release/jsxer "Text Animation Presets.atom" -o atom.js     # decompile once (github.com/AngeloD2022/jsxer)
    python3 tools/import_text_presets.py atom.js

Output per preset:  {"name", "cat", "sub", "dur", "animators": [{"sel": [...], "props": {...}}]}
  selector  {"type": "range", "start", "end", "offset" (percent or curve), "units": "pct"|"index",
             "based": "chars"|"chars_ns"|"words"|"lines", "mode": "add"|..., "amount" (%), "shape": "square"|
             "ramp_up"|"ramp_down"|"triangle"|"round"|"smooth", "smooth" (%), "ease_hi", "ease_lo", "random", "seed"}
            {"type": "wiggly", "mode", "min", "max", "freq" (wiggles/s), "corr" (%), "tphase", "sphase", "lock", "seed",
             "based"}
```
</details>

| function / class | what |
|---|---|
| `parse_preset(s)` |  |
| `main(argv)` |  |

### `mock_apis`  (sdk/tools/mock_apis.py)
Local mock of ElevenLabs / Gemini TTS / Gemini Omni REST APIs — test the integrations for free.

<details><summary>details</summary>

```
python3 tools/mock_apis.py &            # serves http://127.0.0.1:8765
then put  "base_url": "http://127.0.0.1:8765"  in spec.voice / spec.omni_cfg and any dummy keys.
```
</details>

| function / class | what |
|---|---|
| `wav_bytes(sec, rate)` |  |
| `mp4_bytes()` |  |
| `H` |  |

### `package`  (sdk/tools/package.py)
Build dist/motion_creator_brain_sdk_<ver>.zip (v9: ships demo voice/media + the bundled .onnx model;

<details><summary>details</summary>

```
excludes renders, caches, matte caches and keys).   python3 tools/package.py v9 [--with-demo path.mp4]
```
</details>

### `real3d_demo`  (sdk/tools/real3d_demo.py)
**UNDOCUMENTED — add a module docstring**

| function / class | what |
|---|---|
| `ease(x)` |  |
| `poses(t)` |  |
| `bg()` |  |
| `frame(i)` |  |

### `validate_aecomp`  (sdk/tools/validate_aecomp.py)
DEV TOOL: measure how faithfully mograph.aecomp reproduces a reference After Effects element.

<details><summary>details</summary>

```
python3 tools/validate_aecomp.py comp.json preview.mp4 [--sheet out.jpg] [--n 8]

Samples n instants, renders the comp (white-on-black silhouettes) and the vendor preview at the same size,
thresholds both and reports IoU per instant + mean.  The sheet shows  reference | ours | difference.
Only numbers and our own renders are kept; the vendor preview is read, never copied into the repo.
```
</details>

| function / class | what |
|---|---|
| `ref_frame(mp4, t, W, H)` |  |
| `bg_level(mp4, W, H)` |  |
| `probe(mp4)` |  |
| `fscore(a, b, tol)` | line-art F-score: share of our pixels within tol px of the reference and vice versa (chamfer tolerance). |
| `measure(comp, mp4, n, W, H, sheet, …)` | IoU of our render vs the reference.  The vendor previews frame/offset each element differently, so the |
| `main(argv)` |  |


## shell scripts

- `tools/push_backup.sh` — 
- `tools/render_chunks.sh` — Resumable chunked render:  bash render_chunks.sh <spec.json> [chunk_seconds] Each chunk → out/chunks/cNN.mp4 (skipped if it exists), then concat + audio → out/<name>.mp4
- `tools/setup_env.sh` — One-shot environment restore for mograph v14 (run after ANY sandbox reset): bash /home/user/webapp/sdk/tools/setup_env.sh

## models / binaries / assets

- `mograph/models/` (4 files, 14.1 MB): LICENSE_portrait_seg.txt, hand_landmarker.task, portrait_seg.onnx, pose_landmarker_lite.task
- `mograph/models/rnnoise/` (6 files, 1.5 MB): README_rnnoise-models.md, bd.rnnn, cb.rnnn, lq.rnnn, mp.rnnn, sh.rnnn
- `mograph/assets_3d/` (18 files, 4.0 MB): cash_3d.png, grade_a_3d.png, grade_b_3d.png, grade_c_3d.png, laptop_3d.png, laptop_yaw_m15.png, laptop_yaw_m45.png, laptop_yaw_p15.png, laptop_yaw_p45.png, logo_apple_3d.png, logo_dell_3d.png, logo_hp_3d.png, logo_surface_3d.png, pin_3d.png, pin_green_3d.png, plane_3d.png, shield_3d.png, warn_3d.png
- `fonts/` (139 files) families: AbarHigh, AbarLow, AbarMid, AriaFaNum, Capsule, Delbar, DigiAnil, DigiGhasedak, Doran, DoranFaNum, FONT, Gohar, Hasti, IRANSansWeb, IRANSansWeb.ttf, IRANYekanBlack.ttf, IRANYekanBold.ttf, IRANYekanBoldFaNum.ttf, IRANYekanExtraBlack2.ttf, IRANYekanExtraBold.ttf, IRANYekanExtraBoldFaNum.ttf, IRANYekanLight.ttf, IRANYekanMedium.ttf, IRANYekanRegular.ttf, IRANYekanThin.ttf, IRArshia.ttf, IRDastNevis.ttf, IRFarnaz.ttf, KalamehFaNum, Miremad.ttf, Morabba, PINAR, Pelak, PelakFA, Peyda, Shoko, Titr Zebr.ttf, Vazirmatn, YekanBakh
