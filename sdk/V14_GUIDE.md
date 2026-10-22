# mograph v14 "Studio Depth" — professional talking-head motion editing for ANY model

> Goal: a mid-level LLM writes ~60 lines of JSON and gets a professional, ORIGINAL edit (v21: style-bank
> synthesis, never a copy of a reference or of a previous project).
> All the craft (physics, 3D, depth, light, tracking, sound sync) lives in the ENGINE, not in the model's reasoning.

## 0. Restore + run (after any sandbox reset)
```bash
bash sdk/tools/setup_env.sh                                   # deps + MODNet matte model
cd projects/<name>
PYTHONPATH=../../sdk python3 -m mograph.v14 prep   edit.json  # once: pose + matte cache (~1 s/frame)
PYTHONPATH=../../sdk python3 -m mograph.v14 words  edit.json  # edited-time transcript (for "w:" anchors)
PYTHONPATH=../../sdk python3 -m mograph.v14 qa     edit.json  # timings + on-screen check
PYTHONPATH=../../sdk python3 -m mograph.v14 sheet  edit.json  # 12-frame contact sheet  (look at it!)
PYTHONPATH=../../sdk python3 -m mograph.v14 frame  edit.json --t 3.4 8.7
PYTHONPATH=../../sdk python3 -m mograph.v14 render edit.json  # final mp4 + voice + SFX + music, −14 LUFS
```
Render in the background for long clips: `nohup … render … > /tmp/r.log 2>&1 &` (~2 s/frame at 1080×1920).

## 1. Inspiration-Driven Director Workflow (v21 — do NOT skip steps)
> **Supreme law: no copying, no repetition.** References, templates and earlier projects are style studies.
> Every video gets its own visual plan (`python3 -m mograph.pro originality edit.json` must pass).

1. **Study, don't copy**: read `sdk/style_bank/STYLE_BANK_PLAYBOOK.md`, run `python3 -m mograph.pro style-bank
   --list`, look at `sdk/style_bank/style_bank_sheet.jpg` for the quality bar only.
2. **Transcribe** with word timestamps (elevenlabs_scribe_v2) → `words.json`. Fix misheard words by hand.
3. **Analyse tone & structure first**: hook → explanation → pivot (اما / ولی / یا …) → take-away; which beats are
   calm, which are comparisons, objects, warnings. Pick this video's identity: palette from the footage, a font
   pairing, a motion temperament, a subtitle style. It must differ from the closest previous project.
4. **Cut**: `source.segments` = the source ranges you keep (remove dead air / breaths > 0.3 s). Run `words`.
5. **Bespoke visual plan**: one idea per 2–4 s; the hero = the spoken key word (`*hero*`, `"t": "w:<word>"`).
   - **Most beats (60–70 %) = Pure Glow Typography**: `kinetic_title` with `"stage": {"grid": 0}`,
     world-stabilised (`"follow": 0.12`), placed behind a shoulder, beside the head or on the chest. Vary placement,
     glow colour and entrance.
   - **Synthesise other forms only for specific semantic moments**: a comparison → a fresh dual composition; a
     named object → one clean 3D prop; a warning → light/grade/depth shift; showing something → `broll`.
   - Never two hero layers at once. Lower-zone heroes auto-hide subtitles.
6. **Camera**: punch-in (`expo`) on cuts and hero words; slow `silk` drift between; `shakes` only on real impacts.
7. **Grade keys**: darken/blur the background when a glowing layer appears (it makes the light visible).
8. `lint` → `qa` (includes originality + subtitle collision) → `sheet` → fix → `render`.

## 2. Typography (Law A1 — research-based font systems, NEVER random pairing)
Set ONE theme per video: `"meta": {"theme": "doran"}` (doran = the @novadent reference font; also pelak ·
yekanbakh · morabba · abar · pinar).  A theme is ONE superfamily; contrast comes from weight × size × colour:

| markup | role | Doran weight | scale | colour |
|---|---|---|---|---|
| plain | thin line | Regular | 1.0 | warm white #F7F3EC |
| `^w^` | tiny / muted | Light | 0.60 | #E6E2DA |
| `_w_` | bold | Bold | 1.15 | white |
| `*w*` | HERO | ExtraBold | 1.75 | gold gradient + orange glow |
| `=w=` | hero on red/colour backgrounds | ExtraBold | 1.75 | white glow |
| `!w!` | alert | ExtraBold | 1.6 | red gradient |
| `~w~` | calligraphic accent (max once per section) | IranNastaliq / Delbar | 2.0 | gold |

- `|` = new line.  **Lockup** is automatic: a thin line before the hero tucks top-RIGHT, a line after tucks
  bottom-LEFT, pulled against the hero's real ink (column collision) — the reference's staggered composition.
- Word gaps are ink-aware (final ه / ی tails never touch the next word); Persian digits/%/Latin shape LTR.
- Rules: one hero per block · hero = the spoken key word · ≤ 2 families per video · don't override `font`
  inside a layer unless you are switching the whole theme.

## 3. Compositional Primitives & Building Blocks (World-Stabilized by Default, Tracked Only When Intentional)
Common: `t`, `out`, `anchor` (head_top · head_c · face · chin · neck · chest · belly · shoulder_l/r · hand_l/r ·
screen), `offset` [dx, dy] in HEAD WIDTHS, `size`, `text`, `styles`, `pass` back|front, `clamp_y`,
`follow` (v21 default 0.18 for kinetic_title / frame_box / halo = world-stabilised; `"track": true` = full tracking).

> **Specialized legacy/reference widgets: `tracked_tag`, `disc`, `halo`, `frame_box`.** Use at most ONE per video,
> and only if it is semantically justified. Prefer building fresh compositions from `kinetic_title`, `aecomp`
> custom shapes, `morph_shape`, `broll` and `three3d` / `object_3d`.

| type | what it is | key params |
|---|---|---|
| `kinetic_title` | 3D vector title, per-word spring entrance | `enter` jelly_pop · drop_bounce · flip_3d · swing_3d · rise_blur · stretch_in · `exit` pop_out · fall · flip_out · sink · `idle` float · breathe · jelly · `plane_in` {"rx":-70} · `extrude` {"depth":10} · `hero_hit` "w:…" · `field` {"kind":"wave"} |
| `orbit_ring` | text on a REAL 3D cylinder around the head, back half behind her | `radius` (head widths), `tilt`, `spin` deg/s, `land` |
| `disc`  *(legacy, ≤1/video)* | vinyl/CD on the chest, curved texts, half neon rim, 180° flip, red light spill | `flip_at`, `top_text`, `bottom_text`, `center_text`, `neon` |
| `glass_shape` | NESTED glass badge (reference anatomy): gold OUTLINE vector (draws on) → clean `gap` (~7 px) → inset frosted body (true backdrop blur, dark radial core, warm hairline, sheen, travelling specular). hept/hex/pent/oct/diamond/circle/rsquare/pill | `shape`, `w`, `h`, `gap`, `stroke_w`, `pulse_at`, `text` |
| `tracked_tag`  *(legacy, ≤1/video)* | MOTION-TRACKING SHOWCASE: target ring + motion trail ON a tracked point, dashed leader, lagged glass chip (clamped to safe frame) | `anchor` shoulder_l/r · hand_l/r · chest…, `offset`, `lag`, `text` |
| `halo`  *(legacy, ≤1/video)* | v14.3: translucent outer disc + bright inner disc ABOVE the head, text INSIDE the bright disc (center lockup or arc-bent), VFX (shock/flicker/sweep/bloom); lights her hair | `text_mode`, `arc_text`, `radius`, `inner_ratio`, `outer_alpha` |
| `frame_box`  *(legacy, ≤1/video)* | tilted wireframe that DRAWS ON + diamond badge; hero letters break the border | `rot`, `badge_text` |
| `strike` | hand-drawn strike / underline / circle | `kind`, `w`, `dur` |
| `counter_badge` `flash` `progress` `sweep_light` | accents | |
| `subtitles` (top-level) | karaoke pop, emphasis words; v21 `style`: `minimal_clean` (default, box-free) · `soft_scrim` · `glass_matte` · `jelly_crescent` (legacy); auto-hidden under lower thirds / chest heroes | `style`, `emph`, `hide`, `auto_hide`, `anchor`, `offset` |

## 4. What the engine does for you (never re-implement)
- **Zero-lag tracking**: offline pose, centred smoothing → layers glue to the head/chest without trailing.
- **Depth**: MODNet hair-accurate matte; `pass: back` layers sit behind her; orbit rings split by z.
- **Light**: every glowing glyph/rim is also drawn to an emission buffer → bloom + LIGHT WRAP on her edges +
  LIGHT SPILL on her face/collar (red disc rim tints the collar; flip moves the light) + ENV LIGHT: a wide
  coloured pool on the background (`post.env_light`, default 0.9; `post.spill`, `wrap_glow`).
- **Glass**: samples the real composite under the shape (true backdrop blur), not a flat fill.
- **Physics**: closed-form springs (jelly · bouncy · snappy · smooth · heavy), volume-preserving squash,
  decaying wobble, word-pivot motion so cursive joins never break.
- **Sound**: every entrance, flip, pulse, hero hit, shake and subtitle change gets its SFX on the exact frame;
  voice is cut with the EDL and levelled; music bed ducks under the voice; loudness-normalised.

## 5. Checklist before render
- [ ] each hero key word is the layer's `t` anchor   - [ ] no two hero layers overlap in time
- [ ] subtitles hidden under lower-third heroes       - [ ] `qa` clean, `sheet` looked at
- [ ] titles ≥ 70 px hero size on 1080 wide            - [ ] camera punch on each cut

## 6. Automatic "editor brains" (v14.2 — work WITHOUT a reference, protect weak agents)
| brain | file | what it decides for you | override |
|---|---|---|---|
| **Noise cancelling** | `denoise.py` | measures SNR → RNNoise (`lq` model) wet mix 0.40–0.72, presence/air EQ restore, no gate (QC-tuned by listening tests: full-wet = metallic, gate = eaten word endings) | `"audio": {"denoise": "off" \| 0.6 \| {"model": "sh"}}` |
| **Palette** | `palette.py` | samples the real background + clothes → picks an accent from curated cinematic families (teal-cyan, amber-gold, coral, crimson, azure…) by colour theory: contrast with the plate hue, away from clothing hues, WCAG ≥ 4.5:1 vs the stage. Reference colours are NOT copied — a warm park plate gets teal, a cold blue office gets amber. | `"meta": {"palette": {"accent": "#..."}}` |
| **Font pairing** | `glyphs.py` | v21: `font_lock` false by default → curated pairing allowed (`styles.hero.font`: Delbar / Miremad / FONT_2417 / DigiGhasedak / Peyda …); Doran keeps +12 % stretch | `"meta": {"font_lock": true}` restores Doran-only |
| **Ring solver** | `layers.ring_fit` | text on a circle: copies × (width + gap) ≤ circumference — shrinks text / grows ring, so the end never runs into the start. Give ONE phrase. | `max_copies`, `sep`, `max_radius` |
| **Band fit** | `layers.band_fit` | disc texts always INSIDE the band between label and rim (height ≤ 62 % of band, arc ≤ 150°) | `band_span` |
| **Layout guard** | `stage.guard` | every element is tested against a face ellipse (v21: HAIR CROWN→chin) + 9:16 safe frame each frame; it slides out (down/up/side) with temporal smoothing. v21: a title above a head that is too high is never pushed onto the forehead; it slides into the open upper-left/right space or shrinks | `"guard": false \| "down" \| "up" \| "side"` |
| **Stage light** | `stage.draw_stage` | behind each element (behind the speaker): radial scrim scaled by plate brightness + coloured light pool fading with distance (+ emission → spill on her). v21: the grid is OFF by default; enable it per layer, at most once per video | `"stage": false \| {"scrim":.6,"grid":.25,"light":.42,"radius":1.8}` |
| **QA** | `qa.py` | flags layers covering the face, touching edges, too-short layers, flashing subtitles; v21: subtitle ↔ lower-third/chest collisions + originality (copy / clone / rotation / legacy widget) | `python3 -m mograph.v14 qa edit.json` |

**Colour tokens** — use ONLY these in specs: `@accent @accent2 @alert @ink @glow @stage @grid @mute`
(`@gold`/`@red` are aliases of accent/alert).  Never hard-code hex unless it is the client's brand colour.

**Pro-editor rules the engine cannot decide for you**
1. One idea per element; the element appears ON the spoken key word (`"t": "w:word"`), leaves before the next idea.
2. Never two hero elements at the same time; subtitles hide automatically while a lower-third element owns the chest.
3. Elements live in the empty space (above the head, chest, sides) — the guard fixes mistakes, don't rely on it.
4. Ring text = one short phrase (≤ 5 words). Disc band texts ≤ 5 words each. Badge text ≤ 3 lines.
5. Cut decorative elements that don't carry meaning (a floating "1" with no list is noise).


## v14.3 additions
- **Denoise**: DeepFilterNet3 (`sdk/mograph/bin/deep-filter`, `chmod +x` after restore) is the default `audio.denoise:auto`; RNNoise fallback.
- **Orbit ring**: monotonic spin-in (no reverse jerk), facing fade, coverage ≤ 40 %.
- **Halo anatomy** (from user's reference): 2 discs (bright inner + translucent outer), above the head, text inside the
  bright disc. Off-centre/one wide line → `text_mode: arc` bends it along the edge (half-circle). VFX built in.
  Minimal spec: `{"type":"halo","t":"w:word","out":"w:x","pass":"back","text":"^small^|=HERO=","text_mode":"center"}`.


## v14.8 additions — Studio Optical & Auto-Director
- **Refractive glass** (`compositor.glass` → `_glass_optics`): every glass body (glass_shape / nested badge / disc / tags /
  subtitles) bends the frosted plate inward along the rim (`refract` ≈ 4 px), adds 1.5 px red/cyan dispersion on the
  rim (`chroma`) and a thin specular caustic facing the top-left key (`caustic` 0..1). Global:
  `"post": {"glass": {"refract": 4, "chroma": 1.5, "caustic": 0.55, "band": 16}}` (or `false`); per layer: same keys.
- **180° shutter motion blur** (`motionblur.py`): automatic on `drop_bounce` / `swing_3d` / `flip_3d` / `plane_in`
  entrances, `fall` / `flip_out` exits and every `object_3d` spin-in / drop / fall. Each glyph gets its OWN velocity
  vector (projected contour points at t and t − shutter) → 1-D line-kernel streak on a small premultiplied ROI.
  Settled frames cost nothing (threshold 2.5 px). `"motion_blur": false | true | {"shutter":180,"threshold":2.5,"max":48}`
  per layer, `"post": {"motion_blur": false}` to disable globally.
- **Bokeh dust** (`bokeh.py`): `{"type":"ambient_bokeh","t":0,"out":6,"back":12,"front":6,"alpha":1}` or
  `"post": {"bokeh_dust": true}` — 12 octagonal out-of-focus discs BEHIND the speaker (0.7× parallax, accent/warm
  tint) + 6 large soft discs IN FRONT (1.3× parallax, faded over the face zone). ≈5 ms/frame.
- **Auto-director** (`director.py`): `python3 -m mograph.v14 auto-edit words.json media/raw.mp4 -o edit_auto.json`
  → breath cuts (> 0.32 s) → Persian impact scoring → one hero every 2.5–3.5 s → **v21 semantic direction**
  (`choose_beat_treatment`): pure glow typography backbone; comparisons (pivot «اما/یا/ولی…», `compare_sides` finds
  the two concepts) → dual glow titles + morph outline; warnings → alert glow + atmosphere grade key; named objects
  → one 3D prop on the opposite side. A per-video **style seed** varies placement, glow colour, entrances, hero font
  voice and subtitle style. No fixed rotation, and no legacy widgets unless `--legacy-showcase`. Then QA fix loop.
  The output is a starting point: redesign it by hand.

## v15.1 additions — Craft-Omni (EffectCraft / FilmCraft ideas, audited) — details: `V15_REVIEW.md`
First, after any change or reset: `python3 -m mograph.v14 selftest` → must print `9/9 pass`.

**Spec keys (all optional, default off unless noted)**
```jsonc
"source": {"morph_cuts": {"dur": 0.24, "max_motion": 60, "cuts": null}},   // optical-flow morph on breath cuts (auto-edit: on)
"grade":  {"match": {"ref": "../work/input/ref_novadent.mp4", "strength": 0.8, "at": 0.5, "max_skin_deg": 8}},
"post":   {"decontaminate": 0.75,                 // un-mixes background spill from the hair/hijab edge (0..1)
           "glass": {"fresnel": 0.35}},            // + Schlick rim reflection on every glass element
"audio":  {"multiband": true, "vocal_booth_reverb": 0.04, "dehum": false,     // multiband = default ON
           "music": {"file": "media/song.mp3", "bpm": null, "downbeat": null}}  // a real track → remixed to the edit on the bar grid
```
**Text selectors** (every text layer: kinetic_title, orbit_ring, frame_box, halo, glass_shape…)
```jsonc
"selectors": [{"shape": "smooth|square|ramp_up|ramp_down|triangle|round|wiggly", "start": 0, "end": 0.35,
               "offset": [0.55, 1.25, -0.4, 1.05],   // [t_from, t_to, offset_from, offset_to] (s after layer t) = a wave travelling through the line
               "props": {"ty": -14, "rz": 8, "scale": 0.08, "alpha": -1, "glow": 0.9}}]
```
**Text on a path**: `"field": {"kind": "bezier_path", "points": [[-300,0],[0,-90],[300,0]], "align": "center"}`
(smooth Catmull-Rom curve through the points; or `"bezier": [p0, c1, c2, p3, c1, c2, p3, ...]`).

**New layers**
| type | what | key params |
|---|---|---|
| `shatter_text` | text card that explodes into 3D shards | `text, size, break_at (word anchor), rows, cols, force, gravity` |
| `puppet_sticker` | ARAP-bent sticker (waving flag / wobbling icon) | `src, size, pins: [{"uv": [0.5,1]}, {"uv": [0.5,0], "wiggle": [24,0], "freq": 1.5}]` (`"to": [dx,dy]` = one-shot bend) |
| `morph_shape` | shape → shape morph (constant area, linear rotation) | `shapes: ["circle","star","heart","hept",...], at: [anchors], dur, w, color, glow` |
| `vegas_comet` | glowing tapered comets running round a shape | `shape, w, h, comets, speed, length, width` |
| `god_rays` | rotating volumetric rays behind a hero | `radius, rays, spin, beam_deg, alpha` (pass: back) |

**Rules learnt**
- Morph cuts only suit small head moves (jump cuts in one take). If the median flow is > `max_motion` px or the
  flow agreement is < `min_agree` (0.5), the cut stays hard; the director's punch-zoom + flash covers it (on maghale:
  cut 1 morphs, cut 2 is skipped automatically). Geometry glides over `dur`; the colour switches A→B on one frame
  (`switch: 0`). A 50/50 mix of two mouth poses = a ghost face, so never use a linear cross-fade.
- `grade.match` is a LOOK transfer, not a replacement for `grade.bg/fg`. Keep strength at 0.6–0.85; the skin guard
  keeps the face natural even with an extreme reference.
- Procedural music is already composed to the edit, so it is not remixed. A `music.file` is remixed automatically.
