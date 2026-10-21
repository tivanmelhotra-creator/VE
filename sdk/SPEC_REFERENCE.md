# SPEC_REFERENCE — full spec reference for `mograph` (v6 → v9 Studio Apex)

## Top-level structure
```jsonc
{
  "meta": {"w":1080, "h":1920, "fps":30, "bg":"#000000", "bloom":0.45,
           "camera_drift": 3,          // subtle handheld camera sway (px)
           "transition": "fade",       // default transition between chapters
           "vignette": 0.35, "grain": 0.02, "slam_pulse": true,   // v5 post-FX (0/false = off)
           "duration": null},          // null = automatic (last chapter / last subtitle)
  "hud": {
    "header": {"prefix":"پرونده:", "y":0.055, "size":42, "show":true},
    "dots":   {"y":0.82, "style":"dots"|"numbers", "show":true},
    "subs":   {"y":0.885, "size":50, "max_width":0.86, "spoken":"#fff", "upcoming":"#4d4d52", "lead":0.0},
    "handle": "@PAGE", "fade_out_at": 144.5
  },
  "audio": {"vo":"vo.wav", "music":null, "ambient":true, "sfx":true,
            "music_gain":1.0, "sfx_gain":0.8, "duck":0.6, "lufs":-14},
  "subtitles": [ ... ],
  "plugins": ["plugins/my_widgets.py"],
  "voice": {"provider": "elevenlabs|gemini|draft", ...},   // v6 → INTEGRATIONS.md
  "omni": [{"id", "t", "end", "task", "prompt", "mode", "grade"}],   // v6 → INTEGRATIONS.md
  "chapters": [ ... ]
}
```

## Subtitles
```jsonc
// sentence form: words are timed automatically by their length
{"t":0.3, "end":4.0, "text":"**شما** انسان‌ها می‌گین ..."}      // **…** = emphasised word (glows)
// word form (exact, from whisper/forced alignment):
{"t":0.3, "end":4.0, "words":[{"w":"شما","t":0.3,"end":0.62,"emph":true}, ...]}
```
Helpers: `audio.subtitles_from_srt("x.srt")`, `audio.subtitles_from_words(whisper_words)`.
CLI: `python3 -m mograph srt2subs file.srt`.

## Chapters
```jsonc
{"t": 10.8,                 // absolute start (or "w:word")
 "title": "گاوصندوق",       // shown in the HUD header
 "transition": {"type":"pan","dir":"down","dur":1.0},  // pan | thread | fade | cut ; dir: down|up|left|right
 // v5 thread: {"type":"thread","from":"<id in prev chapter>","to":"<id in this chapter>","dur":1.3,
 //             "from_dy":120,"to_dy":140,"swing":140} → glowing comet line wires the chapters, camera glides along it
 "camera": [[0,[540,900,1.0]], ["w:حبس",[540,800,1.25],"smooth"]],   // [time, [cx,cy,zoom(,rot)], ease]
 "elements": [ ... ]}
```
**Element coordinates are local to the chapter** (0..1080 × 0..1920), and **times are local** (0 = chapter start).

## Time values
| Value | Meaning |
|---|---|
| `2.5` | 2.5 s after the chapter starts |
| `"w:انجام"` | the moment the word «انجام» starts (first occurrence after the chapter starts) |
| `"w:درصد#2"` | second occurrence |
| `"we:دلار"` | the moment the word ends |
| `"w:شما+0.3"` / `"w:شما-0.2"` | with an offset |
| `"end"` / `"end-0.5"` | chapter end |
| `"abs:42.0"` | absolute time in the video |

Word matching ignores diacritics, the ZWNJ half-space, and punctuation (`منِ` = `من`, `«ممکنه»` = `ممکنه`).

## Shared element keys
| Key | Description |
|---|---|
| `type` | element type (table below) |
| `id` | name; other elements can target it (`"anchor":"vial+80,-40"`) |
| `t`, `dur` | enter time and duration |
| `out`, `out_dur` | exit start time and duration |
| `enter` | `draw` `fade` `pop` `rise` `wipe` `words` `slam` `none` |
| `exit` | `fade` `undraw` `shrink` `drop` `none` |
| `ease` | `linear in out in_out out_quart out_expo in_out_expo out_back sine smooth snappy anticipate` |
| `x`,`y` | position |
| `color` | `"#fff"` / `[r,g,b,a]` (0..1) |
| `dim_at`, `undim_at`, `dim_color` | push into the background (gray) at a given time |
| `idle` | `pulse` `breathe` `float` (subtle life while on screen) |
| `tracks` | keyframes for any key: `{"x":[[0,300],[2,700,"smooth"]], "scale":..., "rot":..., "alpha":..., "color":[[0,"#fff"],[3,"#555"]]}` |
| `layer` | draw order |
| `space` | `"screen"` = ignores the camera (fixed on screen) |
| `sfx` | `"auto"` (default) / `"none"` / one of `tick draw whoosh stamp pop grid counter grow strike lock impact`; `sfx_offset`, `sfx_gain` |
| `allow_overlap` | disables the overlap warning for this element |
| `font` | **v5 font role**: `body title subtitle counter document handwritten note nastaliq ui` (or a weight key: `bold black pelak doran aria kalameh vazir …`); `label_font` for labels |
| `group` / `group_out` | **v5**: all elements sharing a `group` exit together at `group_out` (set on any of them) |
| (automatic) | **v5 auto-cascade**: callout/arrow/circle/strike targeting an `id` inherit that target's `out` if they have none — no orphan leader lines |

## Elements
| type | Keys | Default SFX |
|---|---|---|
| `icon` | `name size style(default/dim/ghost/bold) fill label label_size width dashed detail rot` | draw |
| `text` | `text size weight(regular/medium/bold/black) align max_width line_height emph_color base_color underline fa_digits` (enter `words`/`wipe` for word-by-word) | — |
| `box` | `w h r fill double tab label lines dashed width rot` | draw |
| `line` | `points:[[x,y]|"id",...] arrow closed dashed fillet width` | draw |
| `arrow` | `from to curve head label label_offset dashed` | whoosh |
| `callout` | `anchor elbow:[dx,dy] shelf(+right / −left) label size` | tick |
| `stamp` | `text size angle w h` (enter `slam`) | stamp |
| `strike` | `false true size` (struck-through false word plus the true one with a check) | strike |
| `counter` | `from to fmt prefix suffix fa size label ease` | counter |
| `dot_grid` | `rows cols gap r highlight highlight_at` | grid |
| `icon_grid` | `names(str/list) rows cols gap size highlight:[i] highlight_at cross_at stagger` | grid |
| `bars` | `values labels w h highlight fmt suffix rtl` | grow |
| `ring` | `value(0..1) r text label` | grow |
| `timeline` | `w items:[..] rtl` | draw |
| `pill` | `text size filled` (enter `pop`) | pop |
| `divider` | `h right left heading_offset` | draw |
| `circle` | `r target number dashed start` (circles something) | draw |
| `path` | `d (SVG in a 100 box) size fill fillet` | draw |
| `image` | `src(png) w radius rot` | — |
| `video` | `src w h fill clip_start speed loop blend(normal/screen/multiply) grade(none/mono/noir) radius border` (v6) | — |


## v5 additions

### Value tracks — one counter, many values (never stack duplicate counters)
```jsonc
{"type":"counter","x":540,"y":1400,"t":4.0,"suffix":" تومان",
 "tracks":{"value":[[4.0,1500],[5.4,1500],["w:۳۰۰۰",3000,"out_expo"]]}}   // auto scale-punch on each change ("punch":false = off)
```
Works for `ring` too (0..1).

### Vector morph
| type | keys | SFX |
|---|---|---|
| `morph` | `from`,`to` or `chain:["drop","money_core","question"]` + `at:[t1,t2]`; `dur ease size width draw_in twist` | whoosh |
| `icon` | `morph_to morph_at morph_dur morph_ease` on any icon | whoosh |

### Compound widgets (one element = one complete sequence)
| type | keys | internal beats (times accept w:/numbers) |
|---|---|---|
| `quote_rig` | `text source header w size line_height highlights:[{word|index, t, style: underline|circle|box, num}]` | each highlight → draw |
| `flow_loop` | `top/left/right:{icon,label} w h arrow_labels:[3] arrow_label_size particles speed flips:[[t,"center text"]]` | flips |
| `laser_scan` | `title clause w h lines clause_row scan_at scan_dur` | scan_at → whoosh |
| `invoice` | `title w row_h rows:[{item,value,t}] total:{label,value,t} stamp:{text,t}` | rows tick, stamp → stamp + exposure pulse |
| `strata` | `w surface:[icons] layers drill_at drill_x rise bend q_size label_top label_bottom` | drill_at → grow + pop |
| `scale_rig` | `w left:{label,item} right:{label} drops:[{t,label}] max_tilt item_size` | each drop → lock |

### Post-FX
`meta.vignette` (0..1), `meta.grain` (0..0.05), `meta.slam_pulse` (exposure + bloom pulse at every stamp slam).

### Stricter QA (v5)
- unknown keys → warning with the closest valid key (`lable` → `label`)
- raw-coordinate callouts pointing at an element that exits → warning
- overlap check also during cross-fades (alpha ≥ 0.2)
- pacing counts widget-internal beats (highlights, drops, rows, scans, morph steps, value changes)
- `review` checks the REAL mp4: black frames, frozen stretches, LUFS, true peak, silence + frame sheet

## Adding a new element (plugin)
```python
# plugins/my_widgets.py
from mograph.elements import element, xy
from mograph.text import draw_text
@element("speech_bubble", dur=0.6, enter="pop", sfx="pop")
def speech_bubble(ctx, el, s):          # s.p, s.trim, s.alpha, s.scale, s.color, s.t, s.lt
    x, y = xy(el, s.t)
    ...
    return (x0, y0, x1, y1)              # bbox for QA
```

## Python API (when you need it outside the spec)
```python
from mograph.scene import load_spec
from mograph.render import render_frame, render_video
from mograph.icons import draw_icon, find_icons, add_icon, import_svg_icon
from mograph.text import draw_text, word_boxes, missing_glyphs, to_fa_digits
from mograph.geometry import draw_path, rounded_rect_d, circle_d, arrow_d, draw_morph, point_on_path
from mograph.text import resolve_font, ROLES
from mograph.qa import review_video
from mograph.align import transcribe_words, snap_to_script   # needs faster-whisper
from mograph.core import Track, Camera, spring, EASINGS
```

## v7 additions

### Style, palette, fonts (`meta`)
```jsonc
"meta": {"style": "lineart" | "swiss" | "kinetic", "palette": "red|blue|yellow|mono|noir|ink|blood",
         "fonts": {"title": "peyda", "body": "bold"},   // optional role overrides (only fonts/ are ever used)
         "sound": "paper|ui|punchy",                    // optional, default follows the style
         "grid": {"cols": 6, "rows": 12, "margin": 72, "gutter": 24, "top": 200, "bottom": 1500}}
```
Any colour may be a token: `@accent` `@ink` `@paper` `@mute` `@mute_soft` (resolved from the palette).
Style presets fill bloom/grain/vignette, HUD (Swiss: no header, numbered dots; kinetic: no HUD/subs) and transitions.

### Grid placement (any element)
`"col": [start, span]` (RTL: column 1 = right edge), `"row": 3.5` (centre row, 1-based, fractional ok), `"rows": 2` (height in rows)
→ fills `x`, `w`, `y`, `h`. Explicit `x/y/w/h` still win.

### Transitions
`"transition": {"type": "wipe" | "slide" | "cut" | "pan" | "thread", "dir": "left|right|up|down", "dur": 0.55, "color": "@accent"}`
wipe = hard edge + accent band (Swiss signature); slide = the content travels with the edge.

### Swiss elements
| type | keys |
|---|---|
| `block` | colour slab that grows from `side` (right/left/top/bottom); `color`, optional `text`, `text_color`, `size`, `align`, `valign` (top/middle/bottom), `pad` |
| `headline` | multi-line (`\n`) display type, `**accent**` per line, `accent` colour, `size` (autofits to `w`), `stagger`; each line rises out of a mask |
| `rule` | thick line drawn from `side`; `weight` |
| `big_number` | huge numerals; `tracks.value` like counter; `unit`, `caption`, `fmt` |
| `index` | numbered list with rules; `items`, `gap`, `size`, `highlight` (index), `highlight_at` (time) |
| `disc` | solid circle; `r`, `color` |

### Kinetic typography
```jsonc
{"type": "kinetic", "mode": "slam|stack|scale|scroll|type", "text": "same as the subtitle line **emph**",
 "x": 540, "y": 900, "size": 180, "t": 0, "max_lines": 3}
```
Words appear exactly when they are spoken (matched to the subtitle word timings; `**emph**` words get the accent colour).
slam = one word at a time · stack = RTL lines building up · scale = the new word big, the previous ones shrink and lift ·
scroll = vertical reel · type = typewriter.

### Sound (event-locked, automatic)
Every element produces SFX from its own motion maths. Per element: `"sfx": "none"` (mute), `"sfx_gain": 0.5`,
`"sfx_offset": 0.02`, `"sound": "ui"` (palette), `"cues": [["w:صفر.", "impact", 1.0, 0.4]]` (extra hits).
Kinds: pen tick click type pop blip whoosh swipe impact hit soft rise scratch lock hum rumble drop.
`python3 -m mograph cues spec.json` prints the score. `audio.engine: "legacy"` = old engine.


## v7.1 — typography safety + scene-aware colour (automatic, nothing to configure)

### Glyph safety (never cut a dot, kasra or tail)
* `text.ink_box(text, x, y, size, weight, …)` = bbox of the **real painted outlines** (cached per shaped line).
  `text_bbox` now also includes the real ink, so layouts, overlap QA and callouts see the true size.
* `text.mask_rect(...)` = clip rect for "rise from a mask" reveals = real ink + 14 % padding.
  **Rule for new elements / plugins: never `ctx.rectangle(... size * 1.5)` around text — use `mask_rect`.**
* Headline leading is ink-aware (line i's descender can never touch line i+1's dots); `ink_gap` (default 0.05·size).
* `block` text auto-fits its slab; `index` rules sit below the deepest descender.

### Scene-aware colour (`mograph/contrast.py`)
* Before the HUD is drawn, the renderer **samples the real pixels** under each HUD zone (subtitles, header, dots,
  handle) — background, blocks, footage, Omni plates — and every HUD colour goes through WCAG checks:
  spoken/emphasis ≥ 3:1 (bold large text), body ≥ 4.5:1, upcoming words ≈ 1.9:1 (recessive but visible).
  A white karaoke word on a cream background is therefore impossible; on light grounds the spoken word
  becomes ink and emphasis keeps the brand accent (darkened only if it would fail).
* Palettes are fitted to the real `meta.bg` (`styles.fit_palette`): e.g. palette `yellow` on a cream bg →
  ink becomes near-black, accent becomes deep gold (hue kept), mute stays recessive.
* API: `contrast.ratio(fg, bg)`, `ensure(fg, bg, min_ratio, prefer=())`, `dim(fg, bg, target)`, `sample(surface, rect)`.

### QA text audit (part of `mograph qa`)
Renders the timeline at a coarse step with an audit hook inside `draw_text`:
* `glyph clipped: «word» (element) at t` — real ink outside its mask/slab **while at rest** (reveals in motion are ignored)
* `low contrast: «word» (element) at t — r:1` — fully-visible text below 3:1 against the pixels under it
  (text drawn with `quiet=True` — upcoming karaoke words, captions, index numbers — needs only 1.6:1)
Verified against the old v7 code: it flags exactly the bugs seen in review (قیمتِ / واقعیِ / پنهان / پالایش / ولی).


## v8 "Studio Horizon" additions

### 1. One spec, many style worlds (per-chapter overrides)
Any chapter may set `style`, `palette`, `bg`, `fonts`, `sound`, `hud`, `bloom`, `vignette`, `grain`, `slam_pulse`:
```jsonc
{"t": "w:قیمتِ-0.5", "title": "قیمت", "style": "swiss", "palette": "red",
 "transition": {"type": "portal", "from": "seed", "dur": 0.8}, "elements": [...]}
```
Resolution: chapter key > `meta` key (only if the chapter keeps the meta style) > the chapter style's preset.
`@paper @ink @accent @mute`, font roles, the SFX palette (`paper/ui/punchy`), the HUD (header/dots/subs) and the post-FX
are resolved **per chapter**. A chapter without `transition` uses its own style default (kinetic = cut, swiss = wipe).
Paper backgrounds never get bloom. HUD parts that the new world hides fade out instead of popping.

### Transitions between worlds
| type | keys | look |
|---|---|---|
| `portal` | `x`,`y` (focal point, chapter-local) **or** `from: "<id>"` (element in the previous chapter), `shape: circle|slab`, `angle` (slab), `dur` 0.75, `color` (ring), `edge` (ring width) | iris / diagonal slab expands from the focal point and reveals the next chapter's background + style; old world recedes, new world settles |
| `wipe` / `slide` | `dir`, `dur`, `color` | now paint each chapter's OWN background inside its region (paper ↔ black wipes work) |
QA audits contrast/clipping against each chapter's real background; frames mid-transition are not audited.

### 2. Swiss data-journalism widgets
| type | keys |
|---|---|
| `waffle_grid` | `value total n(10) side gap cell_r(0..0.5) fill_at accent_label base_label label_size align(right/center/left) build` — 10×10 matrix, `value` cells pop to @accent at `fill_at`, rest recede to @mute, flush rule+swatch callouts |
| `swiss_bars` | `items:[{label,value,t,accent}] max orient(h/v) bar_h gap stagger highlight(i or [i]) fmt suffix label_size value_size pad` — solid slabs; label+value cut out INSIDE in @paper, or outside in @ink when the slab is too short (decided on the final length + `contrast.ensure`) |
| `versus_table` | `right_title left_title rows:[{right,left,t,strike,strike_t}] size title_size row_h pad rule_weight` — heavy top/middle/bottom rules, centre divider, masked row slide-in, @accent reality column (left) |
All accept `col`/`row` grid placement. (`versus_table.rows` = data list; grid height only reads a numeric `rows`.)

### 3. Kinetic modes + odometer
| mode | behaviour | keys |
|---|---|---|
| `masonry` | words packed into 2–4 rows, every row scaled to the SAME width (`max_width`) → locked bento rectangle; `**emph**` words get their own huge row; each word rises/slams into its slot on the spoken word | `max_width max_size(420) min_size max_height rows:[n1,n2,…] line_gap` |
| `highlight_sweep` | whole phrase in @ink; on each `**emph**` word an @accent slab sweeps RTL over its real ink box, the glyphs under it invert to @paper | `sweep_dur slab_pad max_width line_height align` |
`"roll": true` on `big_number` / `counter` → mechanical odometer: each digit slot scrolls through its own ink-exact mask
(carry like a real counter; works with `tracks.value`). `big_number` with roll auto-fits its widest value to the column.

### 4. 2.5D parallax: `"z"` on any world element
`z = 0` default · background grids/rules `-0.2…-0.35` · foreground stamps/pills `+0.15…+0.25`.
When the camera moves `(dx,dy)` from the chapter's rest view the element shifts an extra `-z·(dx,dy)` and scales by
`1 + z·(zoom/zoom_rest − 1)` → depth with zero extra keyframes. QA bboxes include the parallax offset.

### 5. Audio-reactive
* `scene.vo_env(t)` → 0..1 voice envelope at 60 Hz (real `audio.vo` RMS, or a syllable cadence from subtitle timings offline).
* `"idle": "vo_pulse"` → scale × (1 + `pulse_amount`(0.08)·vo_env). On `money_core` / `ai_monogram` icons/morphs (or `"vo_ring": true`) an inner ring breathes with the voice.
* `waveform` element: `x y w h bars(31) mode(bars/bottom/line) lag floor fill color` — symmetric bar / line oscilloscope.

### Sound
Every new element is event-locked: waffle cell pops, bar slab swipes + lands, versus row hits, marker sweeps, masonry
slot hits, portal rise + land. Each chapter's events use that chapter's sound palette (`cues` shows `[palette]`).

## v9 "Studio Apex" additions

### 0. Shaper (automatic)
`mograph.text` shapes with `uharfbuzz` if installed, else **the system C `libharfbuzz.so.0` via ctypes** (`mograph/hbc.py`)
+ fontTools outlines (glyf/CFF, composites decomposed). Same 100 % vector path everywhere: `python3 -c "from mograph import text; print(text.HB_BACKEND)"`
→ `ctypes:libharfbuzz 10.x`. Pillow/raqm raster is only the last fallback (`MOGRAPH_SHAPER=pil` forces it).

### 1. Typography treatments — `text`, `headline`, `kinetic`, `liquid_blob`, `starburst`, `iso_cube.label`, `bezier_graph.label`, `filmstrip.word`
| key | values |
|---|---|
| `text_style` | `"solid"` (default) · `"outline"` · `"extrude_3d"` · `"knockout"` · or dict `{"kind", "stroke_w", "depth", "angle", "extrude": "#5e0818"\|"@mute", "shade", "pad", "radius", "fill_alpha", "face_stroke"}` |
| `stroke_reveal` | `true` / seconds — outline draws on (trim-path on every glyph contour) from `t` (or `stroke_reveal_at`) |
| `extrude_grow` | `true` — the 3D depth springs out from 0 |
| `emph_style` | (kinetic) style only the `**emphasised**` words |
| `prefix` | (text) `">_"`, `"✦"`, `"✓"` glyph on the reading side · `prefix_color`, `prefix_scale`, `prefix_font` |
| `kashida` | px or `{"px": 260, "word": "کش", "after": 1}` — live tatweel stretch of one word (letters stay vector-sharp) |
| `kashida_at` / `kashida_dur` / `kashida_ease` (`"spring"` default) / `kashida_back` | timing of the stretch (word anchors OK) |

### 2. Style `promo` + backdrops + ambient + camera roll + whip
- `meta.style: "promo"` · palettes `crimson` (radial_spotlight) · `silver` (studio_vignette) · `cyber` · `emerald`.
  Extra tokens: `@glow @edge @studio @vignette @vivid @vivid_edge @panel`. Default transition = `whip_pan`.
- **`bg_mode`** (chapter or meta): `solid` · `radial_spotlight` · `studio_vignette` · `crimson_vivid` · `bg_focus: [x,y]` · `bg_radius` · `bg_pulse` (VO-reactive).
- **`ambient`** (chapter): `"bokeh"` · `"warp_stars"` · `"grid"` · list · dict (`{"kind":"bokeh","n":22,"color":"@accent","size":[18,70],"alpha":0.22}`,
  `{"kind":"warp_stars","warp":0.2,"gain":2.2}` — streaks follow camera ZOOM velocity, `{"kind":"grid","step":90,"z":-0.25}`).
- **Camera roll:** 4th component `[x, y, zoom, angle_deg]` in any camera key, or `"cam_angle"` on the chapter → Dutch tilt.
- **`"enter": "whip_blur"`** (any element; kinetic = every word): `whip_dir` (`right` default = from the right / `left` / `up` / `down`), `whip_dur` 0.38, `whip_dist` 900,
  `blur` (gain), `blur_max` 160 px — 1-D kernel ∝ instantaneous velocity.
- **`"transition": {"type": "whip_pan", "dir": "left", "dur": 0.42, "blur": 1.4, "blur_max": 220}`** — region slide + full-frame motion blur.

### 3. Promo / 3D / UI widgets (`mograph/promo.py`)
| element | key keys |
|---|---|
| `liquid_blob` | `r`, `text`, `size`, `a1 a2 w1 w2`, `drift [dx,dy]`, `drift_speed`, `invert_color` (auto contrast) |
| `iso_cube` | `s`, `rx ry rz` (deg), `wx wy wz` (deg/s), `shape: cube\|prism`, `h`, `edges`, `face_colors`, `light`, `label` |
| `starburst` | `r`, `points` 16, `inner` 0.78, `spin` deg/s, `text`, `text_rot`, `rings` |
| `prompt_box` | `w h`, `label` ("Claude"), `prompt`, `type_at`, `cps`, `button` "↑", `placeholder` |
| `timeline_ui` | `w h`, `title`, `layers` [5], `layer_colors`, `strike_at` (giant red ✕), `scrub_speed` |
| `bezier_graph` | `w h`, `from`/`to` cubic-bezier [x1,y1,x2,y2], `flex_at`, `flex_dur`, `title`, `label` |
| `viewfinder_hud` | `w h`, `tc_start` (Persian timecode), `label`, `speed_lines`, `bracket` |
| `filmstrip` | `w h`, `angle` −8, `speed` px/s, `frames`, `word`, `size` (hero word in extrude_3d) |
| `sine_wave` | `w`, `amp`, `wavelength`, `phase_speed`, `width`, `vo: true` (amplitude follows the narrator) |

### 4. Voice disfluency surgery (`mograph/edit.py`) — in `spec.voice`
`"max_pause": 0.22` (clamps mid-line AND between-line silences) · `"min_pause"` · `"dedupe_fillers": true` (or CLI `voice --dedupe-fillers`) ·
`"fillers": [...]` · `"line_cut_ranges": {"3": [[0.8, 1.4]]}` (seconds inside line 3's audio) · `"xfade_ms": 12` (raised-cosine, zero-crossing snapped).
All chapter / camera / numeric times move through the same `remap(t)`; splices are exported to `spec._cuts` → `talking_head` punch-in.

### 5. Talking head + matte (`mograph/matte.py`, element `talking_head`)
Tiers: `onnx` (bundled `mograph/models/portrait_seg.onnx`, MediaPipe Selfie, 462 KB, Apache-2.0) → `chroma` (auto when the border is a real green/blue screen) → `grabcut` (Haar face + torso + MOG2).
`{"type":"talking_head","src":"media/host.mp4","x":540,"y":980,"w":1080,"h":1440,"bg":"keep|world","rim":"#ff1e42","behind":[ …elements drawn BEHIND the speaker… ],
 "pip_at":"w:این","pip":{"x":820,"y":1420,"r":160},"punch":0.12,"matte":"auto","matte_h":640,"feather_bottom":0.08}`
Mattes are cached in `<video>.matte/` (frame-exact, worker-safe). `spec._cut_source` [[edited_t, source_t], …] maps edited time → footage time.

### 6. Spectral VO carve — `audio.carve: "spectral"` (default) | `"duck"` (v8 behaviour)
`carve_lo` 300 · `carve_hi` 3400 · `carve_db` −11 · `keep_db` −2 (perfect reconstruction when nobody speaks).

### 7. Reference analyzer + beats
`python3 -m mograph analyze ref.mp4 -o blueprint.json [--into spec.json]` → cuts, cuts/min, shots, 4-token palette, motion energy, LUFS, BPM, beats, a suggested style.
Beat anchors: `"t": "b:12"` (12th beat, 1-based), `"bar:3"` (4/4), offsets `b:12+0.05`. Spec `"beats": [..]` or `{"bpm": 120, "offset": 0.1}`.

### 8. Aspect
`meta.aspect`: `9:16` (1080×1920, default) · `16:9` (1920×1080) · `1:1` (1080×1080) · `4:5` (1080×1350) — the Swiss grid (cols/rows/margins) and HUD subtitle size adapt.

---

## v11 Studio Pantheon additions (real-footage editing, tracking, compositing, AI-image engines)

Modules: `fx.py` (footage + time/colour/matte/depth), `track.py` (trackers), `glowfx.py` (light/energy + generic
keys), `showcase.py` (editor-reel widgets + `device_frame` container), `genimage.py` (AI-image → animatable layers),
`prompts.py` (generation jobs). Every v11 key is pure function-of-time → frame-exact & worker-safe; heavy analysis
(tracks, mattes, traces, depth layers) runs **once** and is cached next to the media file.

### 1. `footage` — one clip = a full edit-suite layer
```json
{"type":"footage","id":"clip","src":"media/host.mp4","x":540,"y":960,"w":1080,"h":1920,
 "ramps":[{"t":2.0,"dur":0.35,"speed":0.35,"hold":0.8}],          // speed ramp (smooth, ∫speed dt)
 "remap":[[0,0],[1.5,1.5],[2.5,1.5],[3.5,4.0]],                   // explicit time remap (equal src = freeze)
 "beat_ramp":{"slow":0.45,"fast":1.6}, "freeze":[[4.0,4.6]],
 "grade":"teal_orange", "grade_keys":[[0,"flat"],[3,"vivid",0.6]], // animated colour correction
 "selective":{"keep":"#e14c2c","tol":14,"t":1,"dur":2},            // everything grey except one colour
 "hue_shift":[{"t":2,"dur":0.25,"from":"#e14c2c","to":"#1e7bff","within":"person","back":4}],
 "halftone":[{"t":0,"dur":1.2,"cell":14}], "looks":[{"t":5,"dur":0.6,"look":"invert_cyan"}],
 "flash":[3.2], "matte":"auto", "bg":"keep|blur|#hex|world",
 "edge_glow":{"t":4,"dur":2,"color":"#2ad1ff","width":10,"target":"person"},
 "trackers":{"hand":{"kind":"hand","point":"index_tip"},"scr":{"kind":"planar","quad":[[.3,.3],[.7,.3],[.7,.6],[.3,.6]]}},
 "frame":[[0,[0.5,0.45,1.0]],[3,[0.5,0.35,1.3]]], "reframe":"face", "punch":"cuts", "zooms":[{"t":1,"dur":0.4,"to":1.2}],
 "beat_pump":0.02, "shake":[{"t":3,"dur":0.3,"amp":14}], "audio":true, "audio_enhance":true}
```
All `t` inside footage sub-lists are **local to the clip** (relative to the element's `t`), word anchors allowed.
Grade presets: `none flat natural vivid warm cool teal_orange mono noir bleach cyan_negative`.
Looks: `invert_cyan mono noir negative`. `hue_shift.smin` (default 110) protects skin.

### 2. Tracking (`track.py`) — `python3 -m mograph track spec.json` (cache + debug sheet)
| kind | params | output |
|---|---|---|
| `point` | `at [u,v]`, `frame_t`, `win` | LK feature cloud, forward-backward checked → x y s rot |
| `box` | `box [u0,v0,u1,v1]` | median-flow box |
| `planar` | `quad [[u,v]×4 TL TR BR BL]` | RANSAC homography → 4 corners (`quad`) |
| `hand` | `point: index_tip thumb_tip middle_tip wrist palm pinch`, `hand any/left/right` | MediaPipe (bundled) |
| `pose` | `point: nose left_shoulder … chest head_top` | MediaPipe pose lite |
| `face` | — | Haar, very fast (auto-reframe) |
| `color` | `at [u,v]` or `hsv`, `tol` | centroid of a coloured object |
Common: `step`, `smooth` (One-Euro), `start`/`end`. Single clip from CLI: `mograph track clip.mp4 --kind hand --point palm`.

**Attach anything:** `"follow": "clip.hand"` + `follow_offset [dx,dy]`, `follow_rot`, `follow_scale`, `follow_lag` (s),
`follow_axis "x|y"`. Coordinates follow the footage's placement, zoom, reframe and time remap automatically.

### 3. Compositing / depth
- `"behind": "<footage id>"` on any element of the same chapter → drawn between the plate and the cut-out person
  (text passes behind the speaker). Needs no extra key on the footage (matte turns on automatically).
- `planar_insert` `{"follow":"clip.scr","src":"ui.png|clip.mp4|#hex","fit":"cover","blend":"normal|screen","edge":0}`
  → screen replacement / box-art swap.
- `roto` on footage: `{"mode":"color","at":[u,v],"tol":16}` or `{"mode":"person"}` → object mask for glow/hue.
- `python3 -m mograph matte clip.mp4` precomputes the matte cache (`<video>.matte/`).

### 4. Generic keys (any element)
`glow` (true | #hex | {color,radius,strength,pulse}) · `shadow` (true | {blur,alpha,dx,dy,color}) ·
`shiver` (amp | {amp,freq,t,dur}) · `follow…` · `parent` (set automatically by containers) · `decor: true`
(deliberately faint decoration — excluded from the contrast audit, e.g. ghost step numbers) · `gen` (generation
prompt for a missing `src`, see §8).

### 5. Scene / chapter keys
`"shake":[{"t","dur","amp","freq"}]` · `"beat_pump":0.02` · `"flashes":[{"t","dur","color","amount"}]`.
Subtitles: `hud.subs.box: true | {"color":"#000","alpha":0.9,"pad_x","pad_y","radius","text"}` → Hormozi caption slab
(word colours re-solved against the slab).

### 6. Light & energy elements (`glowfx.py`)
| element | keys |
|---|---|
| `saber` | `points` \| `d` \| `target`+`pad`, `color`, `core`, `width`, `haze`, `dur`, `travel`, `loop`, `flicker` |
| `flash_pulse` | `x y`\|`follow`, `color`, `r`, `rings`, `rays`, `ray_rot`, `screen` (0..1 full-frame) |
| `glow_ring` | `r`, `color`, `rings`, `period`, `width`, `spread`, `follow` |
| `tracked_callout` | `anchor` ([x,y] \| id \| "clip.tracker"), `box [x,y]`, `text`, `sub`, `box_style pill/card/plain`, `elbow h/v/none`, `dot`, `lag`, `box_follow` |
| `light_rays` | `r`, `n`, `color`, `spin`, `alpha`, `width` |
| `orbit_trail` | `r`, `ry`, `period`, `start_deg`, `trail`, `ring`, `ring_alpha`, `dir`, `speed_keys` |
| `emitter` | `kind fall/rise/burst/radial`, `shape dot/heart/square/icon:<name>/text:<s>`, `n`, `rate`, `life`, `speed`, `spread`, `gravity`, `colors`, `area`, `bubble`, `seed` |

### 7. Editor-reel widgets (`showcase.py`)
| element | keys |
|---|---|
| `device_frame` | **container**: `children:[…]` authored in a virtual full screen `vw×vh` (default 1080×1920) and scaled/clipped into `w×h`; `radius border border_w fill shadow label label_pos tilt glass` |
| `backdrop` | `kind paper/tiles/smoke/streaks/studio/dots/gradient`, `color color2 seed drift alpha` |
| `badge_button` | 3D press button: `text`, `color`, `press_at`, `lip`, `edge`, `stroke` |
| `handle_tag` | IG glyph + handle: `handle`, `icon`, `gradient`, `align` |
| `social_outro` | end card (always on top): `handle`, `dim`, `dim_color`, `white_at`, `handle_size` |
| `nle_timeline` | `mode clips` (coloured clips slide under an I-beam) / `segments` (labelled bar + red playhead); `items:[{label,sub,color,t}]`, `bar_*`, `clip_w`, `rtl` |
| `image_card` | photo card: `src`, `shape rounded/circle/hex/octagon/pill/rect`, `kenburns`, `grade`, `focus`, `shadow` |
| `redact_text` | black bar that wipes away to reveal text: `reveal_at`, `reveal_dur`, `bar_color`, `dir` |
| `radial_formula` | hub + N nodes wheel that rotates node by node with a progress arc: `nodes:[{icon,label,t}]`, `hub_title hub_sub hub_r r node_r rotate rot_dur step_fmt` |

### 8. AI-image engines (`genimage.py`) + `mograph prompts`
| element | what it does | keys |
|---|---|---|
| `auto_trace` | line-art image → vector strokes drawn on live | `mode centerline/outline`, `color_mode source`, `width`, `dur`, `order`, `stagger`, `min_len`, `simplify`, `thresh`, `invert`, `fill_image` |
| `depth_plate` | one picture → far/mid/near layers (inpainted) → 2.5D parallax | `layers`, `split auto/matte/luma/y`, `push`, `pan`, `strength`, `dof`, `bg_blur` |
| `sprite_rig` | N×M parts sheet → puppet with auto joints | `grid`, `parts`, `bg`, `pose{joint:[[t,deg,ease]]}`, `idle breathe/walk/wave/talk`, `speed`, `scale`, `joints` |
`"gen": {"prompt": "...", "aspect": "1:1"}` on any element with a missing `src` → `python3 -m mograph prompts spec.json`
lists image / music / voice jobs for the agent's own generators (type-specific prompt suffixes added), then re-run QA.

### 9. Performance budget (2 CPU / 1 GB)
Footage + matte + glow ≈ 180 ms/frame at 1080p → render with `--workers 1`. Hand tracking ≈ 43 ms/frame once
(cached). Keep clips ≤ 720p source; use `matte_h: 384` for long clips.

---

## v12 Studio Cut additions — RAW take → finished reel

Modules: `cut.py` (EDL), `transfx.py` (edit-point transitions), `captions.py` (reel captions), `reelfx.py` (motion kit),
`beat.py` (procedural edit-synced music). Showcase: `projects/raw01/laptop_reel.json` (20 s raw → 17.4 s reel).

### 1. `edit` (top level) — the edit decision list
```json
"edit": {"src": "media/raw.mp4", "words": "words.json", "shots": "auto", "max_gap": 0.3, "pad_pre": 0.08,
         "pad_post": 0.12, "transitions": {"shot": "zoom_blur", "jump": "punch"} | [{"at": 1, "type": "whip"}],
         "transition_dur": 0.3, "bpm": 100, "snap": 0.5, "snap_tol": 0.1, "emph": ["سود"], "max_words": 3,
         "tail": 0.5, "enhance": {"h": 1920, "denoise": 1.2, "sharpen": 0.6}}
```
Word timestamps → speech runs → silences > max_gap removed, segments never straddle a camera change (auto scene
detection), optional snapping of cut points to the music grid. Produces subtitles in EDITED time (every `w:word`
anchor keeps working), `_cuts`, `_cutfx`, `meta.duration`. Any `footage` with `"edit": true` plays the edited
timeline; its audio is spliced with raised-cosine crossfades (`xfade_ms`, default 30). `python3 -m mograph cut spec.json`
prints the EDL. words.json = `[[word, t0, t1], …]` (from `audio_transcribe` / `mograph align`).

### 2. Footage additions
`enhance` (cached denoise + lanczos upscale + CAS sharpen proxy for 480p phone video) · `auto_grade`
`{balance, curve, vibrance, gamma, black_pct, white_pct}` (per-shot levels + neutral balance + filmic S-curve,
shots from the EDL) · `skin` `{hue 12, amount 0.6, sat 0.82, band 22}` (natural skin, removes magenta/orange cast) ·
`auto_push` 0.05 (slow push-in per segment) · `audio_enhance` now = broadcast voice chain (HPF, FFT denoise, mud cut,
presence, air, de-ess, compressor, limiter).

### 3. Edit-point transitions (`transfx.py`, applied to the composited frame, HUD stays sharp)
`zoom_blur` · `whip` (`dir`) · `spin` · `punch` (`amount`) · `flash` · `glitch` · `shake`; keys `dur`, `strength`.
Authored too: `"cut_fx": [{"t": 7.8, "type": "glitch"}]`. Each gets its SFX automatically.

### 4. Reel captions — `hud.subs.style: "reel"`
`{"y", "size", "weight", "color", "emph", "active_bg", "active_text", "reveal": "word"|"group", "stroke", "stroke_w",
"shadow", "pop", "max_width", "rotate"}` — spring pop per group, each word pops on its timestamp, active word on a
coloured slab, emphasised words coloured + 8 % larger, thick outline + shadow (readable on any footage). Per group:
`"pos": "top"|"mid"|"low"`, `"y"`, `"hide"`.

### 5. Motion kit (`reelfx.py`)
| element | keys |
|---|---|
| `title_card` | `text` (**emph**), `sub`, `size`, `slab`, `slab2`, `skew`, `rotate`, `wipe_dur`, `word_gap`, `emph_color` |
| `logo_chip` | `items [{text, t, color, bg}]`, `size`, `gap`, `max_w`, `style glass/solid/outline` |
| `route_arc` | `from`, `to`, `label_from`, `label_to`, `bend`, `dur`, `color`, `pin_color(2)`, `plane`, `plane_size` |
| `price_tag` | `value`, `from`, `prefix`, `suffix`, `label`, `w h`, `color`, `count_dur`, `shape tag/pill/burst`, `swing` |
| `grade_scale` | `grades [{g, label, color}]`, `pick`, `at`, `title`, `w h` |
| `profit_meter` | `buy`, `sell`, `unit`, `labels`, `colors` |
| `sticker` | `text`, `icon`, `color`, `text_color`, `edge`, `rotate`, `wobble` |
| `cta_arrow` | `from`, `to`, `label`, `curve`, `width`, `head` |
| `light_leak` · `shine_sweep` · `progress_bar` · `vignette_pulse` | overlays (see module doc) |
Easings: full Penner set added (`in/out/in_out_cubic`, `out_quint`, `out_circ`, `in_back`, `out_elastic`, `out_bounce`);
unknown names fall back to `in_out`.

### 6. `audio.beat` — procedural music synced to the edit
`{"bpm": 100, "key": "A", "scale": "minor", "progression": [0,5,3,6], "gain": 0.32, "swing": 0.08,
  "sections": [{"t": 0, "part": "intro"}, …, {"t": 13.5, "part": "drop"}, {"t": 16.75, "part": "outro"}]}`
Parts: intro · verse · build · riser (snare roll + HP sweep + 1/16 silence) · drop (impact) · outro. Kick sidechain,
vocal-pocket EQ, −3 dB headroom. Default sections are derived from the EDL cuts. Use `audio.music` when a real track
exists (it takes priority).
