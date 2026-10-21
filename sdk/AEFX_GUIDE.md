# AE-FX (v16) — After-Effects-grade motion toolkit for agents

`sdk/mograph/aefx/` adds four things this engine was missing. Each one was re-implemented from scratch after
studying four commercial plugins/packs as **reference data only**. No vendor binaries, scripts, UI or media are
shipped; only numeric keyframe/animator data was converted to our own JSON format.

| module | studied reference | what we built |
|---|---|---|
| `keyframes.py` | AE/Premiere temporal Bezier (speed + influence) | exact keyframe evaluator, retiming, Easy Ease |
| `textanim.py` + `presets/text_presets.json` | AtomX Text Animation Presets (351) | AE **Range & Wiggly selectors** (shapes, ease high/low, smoothness, chars/words/lines, random order, modes), all 351 presets, **Persian-aware**: joined letters move as one cursive piece |
| `motion.py` + `presets/motion_presets.json` | Premiere "Motion Presets" (352 in/out/fx) | Transform (with real shutter motion blur), Basic 3D, Fast Blur, Wave Warp, Twirl; preset player |
| `deepglow.py` | Deep Glow 2 (feature list only) | linear-light multi-octave glow, soft-knee threshold, chromatic aberration, anamorphic aspect/angle, tint, filmic shoulder |
| `pathblur.py` | PathBlurFX (feature list only) | blur along any path / radial / spin, start→end blur + taper, fall-off, one-sided trail, dynamic sampling |
| `bridge.py` | – | `"fx"` stack on any layer/element, `post.aefx`, cut transitions |
| `suggest.py` + `presets/catalog.json` | – | **measured** catalogue (energy, overshoot, spin, travel, blur, identity end) → intent search |

Size: ~0.7 MB of JSON + ~70 KB of Python. Self-test: `cd sdk && python3 -m mograph.aefx selftest` (9/9).

---

## 1. Agent workflow (pick → look → apply → check)

```bash
cd sdk
python3 -m mograph.aefx suggest punchy impact --dir in      # ranked by MEASURED data (also Persian: ضربه, نرم, تایپ…)
python3 -m mograph.aefx sheet motion --cat Glitch --dir in  # contact sheet: each preset × 6 instants → LOOK at it
python3 -m mograph.aefx sheet text atp_23 atp_100 atp_340 --text "آموزش *ویدیو*"
python3 -m mograph.aefx gif scale_10_in                     # one animated preview
python3 -m mograph.aefx info atp_23                         # raw JSON (to tweak / copy into "animators")
```
The full table with measured numbers is in `mograph/aefx/presets/CATALOG.md`. Rebuild it with `python3 tools/build_aefx_catalog.py`.
Tag **`settles-offset`** = an IN preset that ends in a pose instead of at identity (10 of them). Don't use those for clean titles.
**Browsable gallery** (for humans, and for agents that can open a URL): `public/aefx/index.html`. Build it with
`python3 tools/build_aefx_gallery.py`.

## 2. Text presets on v14 titles (talking-head engine)

```json
{"type": "kinetic_title", "text": "موقع خوندن|*مقاله*", "t": "w:مقاله-0.1", "out": 6.0,
 "preset": "atp_23", "preset_dur": 0.9,
 "preset_out": "atp_100", "preset_out_dur": 0.5}
```
- `preset` replaces the physics `enter`. Local time starts at `t`. `preset_out` plays a preset **backwards** so it ends exactly at `out`.
- `animators`: your own list in the same JSON as `text_presets.json` (range/wiggly selectors + props). Values can be curves.
- On Persian text, AE "characters" automatically become **connected pieces** (آ|مو|ز|ش), following the real joining rules.
- px values in presets are relative to a 100 px font (`preset_px`) and scale with the real glyph size.

## 3. `"fx"` stack: any v14 layer AND any v5–v13 element

```json
"fx": [
  {"motion": "scale_rotation_03_in"},                          // IN preset at the layer start
  {"motion": "blur_02_out", "at": "out"},                      // OUT preset ending at the layer "out"
  {"motion": "shake_03", "at": "w:بوم", "dur": 0.6},           // one-shot FX at a word anchor, retimed
  {"glow": {"radius": 140, "exposure": 0.7, "threshold": 0, "tint": "#1e9bff", "aberration": 0.4}},
  {"path_blur": {"path": [[120,1700],[540,1560],[960,1420]], "end_blur": 90, "bidirectional": false},
   "t": 2.4, "dur": 0.6}
]
```
- Motion presets pivot around the **layer's own centre** (`"origin": "frame"` to use the frame centre instead).
- `motion_blur` defaults to 180° (true sub-frame blur). Set it to 0 for the raw preset.
- `dur` / `speed` retime a preset while keeping its curve shape (speeds are rescaled).
- glow / path_blur items: `t`/`dur` window (default = the whole layer life), `ramp` fade (0.15 s).
- Demo: `python3 -m mograph frame tests/v16_aefx.json --t 1.2`.

## 4. Whole-frame FX and cut transitions (v14)

```json
"post": {"aefx": [{"glow": {"radius": 60, "threshold": 0.8, "exposure": 0.3}},
                  {"motion": "position_l_to_r_01_in", "at": 3.2}]},
"source": {"cut_presets": {"out": "blur_02_out", "in": "blur_02_in", "dur": 0.5}}
```
`cut_presets` can also be a list with one entry per cut (`null` = plain cut). Presets come in matching
`_out`/`_in` pairs (`bridge.paired_in`).

## 5. Python API
```python
from mograph.aefx import motion as AM, textanim as TA
from mograph.aefx.deepglow import deep_glow
from mograph.aefx.pathblur import path_blur
img = AM.apply_preset(img, "glitch_04_in", t, motion_blur=180)
st  = TA.Animator("atp_340", dur=0.8).frame(glyph_block, t);  gs = st.apply(glyph, gs)
img = deep_glow(img, radius=120, exposure=0.5, tint="#ff4fd8", aspect=4)    # anamorphic
img = path_blur(img, "radial", 0, 80, center=(540, 960))                      # zoom-burst
```

## 6. Performance (2 vCPU, 1080×1920)
motion preset ≈ 3–15 ms per effect · text animator ≈ 1 ms per block · deep_glow ≈ 0.5 s full frame (run it on a layer
crop or `source=` an emission buffer for speed) · path_blur ≈ 0.1–0.4 s depending on blur length.

## 7. Re-importing (dev only, needs the original packs; NOT needed to use AE-FX)
`tools/import_motion_presets.py "<pack>/templates/Motion Presets"` · `tools/import_text_presets.py atom.js`
(decompile the `.atom` JSXBIN once with github.com/AngeloD2022/jsxer). Validation: the prproj field order was fitted
to Premiere's own preview (2.4 px RMS), and the opacity selector curve to AtomX's preview (mean |Δα| 0.045).
