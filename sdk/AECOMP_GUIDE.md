# AE-Comp (v17) — After-Effects SHAPE-LAYER engine for agents

`sdk/mograph/aecomp/` renders After-Effects-style **shape compositions** (groups, rect/ellipse/polystar/path,
fill/stroke/gradient, Trim Paths, Repeater, Zig Zag, Twist, Round Corners, Pucker & Bloat, Offset, Merge Paths,
parenting, track mattes, blend modes, text layers, AE keyframes with speed/influence, loopOut, wiggle, time*k,
FX-Control sliders/colours, Trace-Path) from **our own JSON**, with cairo, on 2 vCPU (2–15 ms per 200 px frame).

Three commercial packs were studied as **reference data only** (Shape Elements Pack 452 .aep, Lower Thirds 50 comps,
Titles Factory 194 projects). **Nothing from them is shipped**: no shapes, scripts, UI or media. We kept only the
measured numbers (`mograph/aecomp/presets/aecomp_stats.json`) and wrote our own generators fitted to them.

| module | what |
|---|---|
| `values.py` | animatable values: AE temporal bezier (speed/influence, per-dimension, spatial tangents), loopOut, wiggle, controls |
| `geom.py` | bezier paths + AE path operators (trim simultaneous/individual, zigzag, twist, round corners, pucker, offset) |
| `render.py` | comp → BGRA: AE item order semantics, repeater, merge paths, mattes, parenting, text (v14 glyph engine, Persian) |
| `generators.py` | **12 own shape elements** (burst, confetti, ring_pop, arc_swirl, arrow_trace, slide_lines, wipe_circle, bars_wipe, star_blink, zigzag_line, spinner_loop, burst_dash) |
| `lowerthirds.py` | **4 box-free RTL lower thirds** (line_reveal, bracket, orbit_dot, split_lines), Doran, zero-box law |
| `titles.py` | **4 full-frame titles** (minimal_rule, bracket_frame, burst_hero, kinetic_stack): hero auto-fit to 86% width, AE text animators (atp_N), Doran, zero-box |
| `bridge.py` | `{"type": "aecomp", ...}` in v5–13 scenes AND v14 footage edits |
| `selftest.py` | 11 numeric tests, no vendor files needed |

Dev only (need `pip install py_aep` + the packs; **not needed to use the SDK**): `tools/import_aep.py` (AEP → our
JSON), `tools/validate_aecomp.py` (registers framing + 1-frame preview offset, then IoU + line-art F3 vs the vendor
preview video), `tools/batch_validate_aecomp.py` (whole pack, parallel, resumable JSONL + `--report` per category).

## 1. Agent workflow (pick → look → apply → check)
```bash
cd sdk
python3 -m mograph.aecomp list                      # names + category + one-line description
python3 -m mograph.aecomp sheet                     # contact sheet of every element × 6 instants → LOOK at it
python3 -m mograph.aecomp sheet --cat explosion     # one category
python3 -m mograph.aecomp lt-sheet --title "نام شما" --role "سمت"   # lower thirds with YOUR text
python3 -m mograph.aecomp title-sheet --title "عنوان" --role "زیرعنوان"  # full-frame titles
python3 -m mograph.aecomp gif burst --seed 3        # one animated preview
python3 -m mograph.aecomp mp4 line_reveal --bg alpha   # ProRes 4444 with alpha (for an NLE)
python3 -m mograph.aecomp info burst > my.json      # the comp JSON → edit → use as "comp"
python3 -m mograph.aecomp selftest                  # must say 11/11
```
Browsable gallery (humans / agents with a browser): `public/aefx/index.html`, filter **shape** / **lower_third** / **title**
(`python3 tools/build_aefx_gallery.py --only aecomp` regenerates the previews).

## 2. Spec (same keys in v5–13 scenes and v14 edits)
```json
{"type": "aecomp", "gen": "burst", "seed": 2, "t": 3.2, "x": 540, "y": 900, "size": 420,
 "c1": "#ffffff", "c2": "@accent", "c3": "#2dd4bf", "stroke": 10, "speed": 1.0, "loop": false, "rot": 0, "layer": -1}
{"type": "aecomp", "lower_third": "line_reveal", "title": "دکتر سارا احمدی", "role": "متخصص ارتودنسی",
 "t": "w:سلام", "out": 6.0, "accent": "@accent"}
{"type": "aecomp", "title_style": "burst_hero", "hero": "داستان موفقیت", "kicker": "فصل اول",
 "t": 0.5, "out": 4.0, "accent": "@accent"}
{"type": "aecomp", "comp": "my.json"}               // any comp JSON (own / edited / imported for study)
```
- `size` = longest side in px; `seed` varies the random layout; `"@accent"` uses the (auto) palette in v14.
- v14: `t`/`out` take word anchors; position can use `anchor`/`offset` like other layers (layout guard applies).
- Put bursts behind titles (`"layer": -1` in v5–13, `"pass": "back"` in v14). Combine with AE-FX: `"fx": [...]`.
- Test specs: `tests/v17_aecomp.json` (`python3 -m mograph frame tests/v17_aecomp.json --t 1.0`).

## 3. Comp JSON (write your own)
```json
{"w": 1100, "h": 1100, "dur": 1.0, "fps": 30, "controls": {"c1": "#fff", "stroke": 12},
 "layers": [{"name": "fx", "type": "shape", "in": 0, "out": 1.0, "tr": {"pos": [550, 550]},
   "shapes": [{"ty": "gr", "it": [
       {"ty": "sh", "path": {"v": [[0,-90],[0,-420]], "i": [[0,0],[0,0]], "o": [[0,0],[0,0]], "c": false}},
       {"ty": "tm", "end": {"k": [[0,0,"b",0,0.33,0,0.83],[0.55,100,"b",0,0.33,0,0.83]]},
                     "start": {"k": [[0.18,0,"b",0,0.33,0,0.83],[0.73,100,"b",0,0.33,0,0.83]]}},
       {"ty": "st", "color": {"ctl": "c1"}, "width": {"ctl": "stroke"}, "cap": "round"},
       {"ty": "rp", "copies": 8, "tr": {"rot": 45, "pos": [0, 0]}}]}]}]}
```
Keyframe row: `[t, value, "b"|"l"|"h", in_speed, in_influence, out_speed, out_influence, (out_tangent, in_tangent)]`
(influence 0..1). Item types: `gr el rc sr sh fl st gf gs tm rp zz tw rd pb op mm` (see `render.py` docstring).
AE rules implemented: an operator changes the paths ABOVE it; a fill/stroke paints the paths above it; Merge Paths
combines everything above it into ONE exact outline (union/subtract/intersect/exclude via shapely; operators below the
merge — e.g. Trim — run along that outline) and disables the paints above it; a repeater copies everything above it
and the operators below the repeater act on every copy.

## 4. Measured rules baked into the generators (from 447 elements)
- length: bursts / blow-ups / simple **1.0 s**, arrows / slides / transitions **1.27–1.33 s**, arcs **1.53 s**
- Trim draw: End leads, Start follows **0.15–0.18 s** later; each ≈ 0.6 s (bursts) … 1.1 s (arcs)
- ease: AE Easy Ease 33 % on most; bursts attack with **17 % out**; transitions use **~90 %** influence
- round caps ≈ 60 %; repeater 2–8 copies (4 most common); stroke 5–15 px at 1100 px
- lower thirds: title in 0.44–0.70 s, role +0.31 s, out 0.62 s, title:role ≈ 2.6:1, text revealed through mattes

## 5. Fidelity (dev measurement, whole Shape-Elements pack, v17.1 engine)
All 452 reference elements imported and re-rendered by our engine vs the vendor previews
(`tools/batch_validate_aecomp.py`, results in `tests/aecomp_validation.jsonl` + `tests/aecomp_validation_report.txt`).
Line-art F3 (median): Simple 0.96 · Looped 0.97 · Transitions 0.94 · Arrows 0.94 · Complicated 0.92 ·
Explosions 0.87 · Arcs 0.85 · Blow-ups 0.82 · Slides 0.68 — **ALL 0.916**, 397/452 ≥ 0.6, 0 import errors.
v17.2: layer effects Radial Wipe / Linear Wipe (exact) and Turbulent Displace (fitted approximation) on any layer
(`"fx": [{"ty": "radial_wipe", "completion": 40, "center": [550, 550], "dir": "cw"}]`); the validator also tries the
un-shifted framing (centroid registration misled it on sparse staggered elements — most "weak" Slides were a
measurement artefact). Known gaps: layer styles, 3D layers, text-layer animators on imported comps, Arcs_40
(the only Merge→Trim element: AE's start vertex on a merged outline is undocumented).
