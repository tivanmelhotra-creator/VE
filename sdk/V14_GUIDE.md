# mograph v14 "Studio Depth" — professional talking-head motion editing for ANY model

> Goal: a mid-level LLM writes ~60 lines of JSON and gets an edit at the @novadent.ir level.
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

## 1. The workflow (do NOT skip steps)
1. **Transcribe** with word timestamps (elevenlabs_scribe_v2) → `words.json`. Fix misheard words by hand.
2. **Cut**: `source.segments` = the source ranges you keep (remove dead air / breaths > 0.3 s). Run `words`.
3. **Beat sheet** (in your head or a comment): one idea per 2–4 s, each idea = ONE hero layer. Pick its
   KEY WORD; that word gets the contrast (`*hero*`) and the layer's `t` = `"w:<key word>"`.
4. **Alternate depth**: title BEHIND the head (`pass: back`) → object on the CHEST (disc / glass) → halo behind
   → glass in front.  Never two hero layers at once.  Hide subtitles while a hero owns the lower third.
5. **Camera**: punch-in (`expo`) on every cut and hero word; slow `silk` drift between; `shakes` on impacts.
6. **Grade keys**: darken/blur the background when a glowing layer appears (it makes the light visible).
7. `sheet` → fix → `sheet` → `render`.

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

## 3. Layers (all tracked, all physics)
Common: `t`, `out`, `anchor` (head_top · head_c · face · chin · neck · chest · belly · shoulder_l/r · hand_l/r ·
screen), `offset` [dx, dy] in HEAD WIDTHS, `size`, `text`, `styles`, `pass` back|front, `clamp_y`.

| type | what it is | key params |
|---|---|---|
| `kinetic_title` | 3D vector title, per-word spring entrance | `enter` jelly_pop · drop_bounce · flip_3d · swing_3d · rise_blur · stretch_in · `exit` pop_out · fall · flip_out · sink · `idle` float · breathe · jelly · `plane_in` {"rx":-70} · `extrude` {"depth":10} · `hero_hit` "w:…" · `field` {"kind":"wave"} |
| `orbit_ring` | text on a REAL 3D cylinder around the head, back half behind her | `radius` (head widths), `tilt`, `spin` deg/s, `land` |
| `disc` | vinyl/CD on the chest, curved texts, half neon rim, 180° flip, red light spill | `flip_at`, `top_text`, `bottom_text`, `center_text`, `neon` |
| `glass_shape` | NESTED glass badge (reference anatomy): gold OUTLINE vector (draws on) → clean `gap` (~7 px) → inset frosted body (true backdrop blur, dark radial core, warm hairline, sheen, travelling specular). hept/hex/pent/oct/diamond/circle/rsquare/pill | `shape`, `w`, `h`, `gap`, `stroke_w`, `pulse_at`, `text` |
| `tracked_tag` | MOTION-TRACKING SHOWCASE: target ring + motion trail ON a tracked point, dashed leader, lagged glass chip (clamped to safe frame) | `anchor` shoulder_l/r · hand_l/r · chest…, `offset`, `lag`, `text` |
| `halo` | v14.3: translucent outer disc + bright inner disc ABOVE the head, text INSIDE the bright disc (center lockup or arc-bent), VFX (shock/flicker/sweep/bloom); lights her hair | `text_mode`, `arc_text`, `radius`, `inner_ratio`, `outer_alpha` |
| `frame_box` | tilted wireframe that DRAWS ON + diamond badge; hero letters break the border | `rot`, `badge_text` |
| `strike` | hand-drawn strike / underline / circle | `kind`, `w`, `dur` |
| `counter_badge` `flash` `progress` `sweep_light` | accents | |
| `subtitles` (top-level) | stateful jelly glass capsule, karaoke pop, emphasis words | `emph`, `hide`, `anchor`, `offset` |

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
| **Font lock** | `glyphs.py` | Doran theme = Doran ONLY (any `font` override from another family is stripped), +12 % horizontal stretch | `"meta": {"theme": "pelak"}` |
| **Ring solver** | `layers.ring_fit` | text on a circle: copies × (width + gap) ≤ circumference — shrinks text / grows ring, so the end never runs into the start. Give ONE phrase. | `max_copies`, `sep`, `max_radius` |
| **Band fit** | `layers.band_fit` | disc texts always INSIDE the band between label and rim (height ≤ 62 % of band, arc ≤ 150°) | `band_span` |
| **Layout guard** | `stage.guard` | every element is tested against a face ellipse (forehead→chin) + 9:16 safe frame each frame; it slides out (down/up/side) with temporal smoothing | `"guard": false \| "down" \| "up" \| "side"` |
| **Stage light** | `stage.draw_stage` | behind each element (behind the speaker): radial scrim scaled by plate brightness + world-locked 0.8 px grid with + ticks + coloured light pool fading with distance (+ emission → spill on her) | `"stage": false \| {"scrim":.6,"grid":.3,"light":.42,"radius":1.8}` |
| **QA** | `qa.py` | flags layers covering the face, touching edges, too-short layers, flashing subtitles | `python3 -m mograph.v14 qa edit.json` |

**Colour tokens** — use ONLY these in specs: `@accent @accent2 @alert @ink @glow @stage @grid @mute`
(`@gold`/`@red` are aliases of accent/alert).  Never hard-code hex unless it is the client's brand colour.

**Pro-editor rules the engine cannot decide for you**
1. One idea per element; the element appears ON the spoken key word (`"t": "w:word"`), leaves before the next idea.
2. Never two hero elements at the same time; subtitles hide while a lower-third element owns the chest.
3. Elements live in the empty space (above the head, chest, sides) — the guard fixes mistakes, don't rely on it.
4. Ring text = one short phrase (≤ 5 words). Disc band texts ≤ 5 words each. Badge text ≤ 3 lines.
5. Cut decorative elements that don't carry meaning (a floating "1" with no list is noise).


## v14.3 additions
- **Denoise**: DeepFilterNet3 (`sdk/mograph/bin/deep-filter`, `chmod +x` after restore) is the default `audio.denoise:auto`; RNNoise fallback.
- **Orbit ring**: monotonic spin-in (no reverse jerk), facing fade, coverage ≤ 40 %.
- **Halo anatomy** (from user's reference): 2 discs (bright inner + translucent outer), above the head, text inside the
  bright disc. Off-centre/one wide line → `text_mode: arc` bends it along the edge (half-circle). VFX built in.
  Minimal spec: `{"type":"halo","t":"w:word","out":"w:x","pass":"back","text":"^small^|=HERO=","text_mode":"center"}`.
