# v19 PRO EDITOR — `mograph.pro` (agent guide)

What a professional editor does every day, as data in the same v14 `edit.json`.
**First, after any reset:** `cd sdk && python3 -m mograph.pro doctor --fix` (deps, binaries, fonts, footage, tracking
cache from the committed pack, all 5 selftests). `python3 -m mograph.pro selftest` → **12/12**.

| editor task | spec / command | notes |
|---|---|---|
| colour look | `"grade": {"look": "teal_orange", "look_strength": 0.75}` | 10 looks; `looks --spec edit.json` = contact sheet on YOUR footage → LOOK, then choose |
| client LUT | `"grade": {"lut": "luts/brand.cube", "lut_strength": 0.8}` | any 1D/3D `.cube` (Resolve, Premiere, CapCut) |
| skin tones safe | `"protect_skin": 0.6` (default when a look/lut is set) | skin pixels are pulled back to the ungraded plate |
| white balance / LGG | `"wb": {"temp": .2, "tint": -.05}`, `"lift_gamma_gain": {"lift": [0,0,.02], "gain": [.02,0,-.02]}` | `contrast`, `saturation`, `vibrance` too |
| film finish | `"grain": .2, "halation": .15, "sharpen": .15, "letterbox": 2.39` | grain is deterministic per frame → chunk-safe |
| export the look | `python3 -m mograph.pro lut-export teal_orange -o teal.cube` | same look in any NLE |
| B-roll cutaway | `{"type": "broll", "src": "media/clinic.mp4", "mode": "full", "t": "w:…", "out": "we:…", "in": "whip_left", "out_fx": "whip_right"}` | voice continues, subtitles stay ON TOP (z −50) |
| picture-in-picture | `"mode": "pip", "x": 300, "y": 520, "pip": {"w": .36, "aspect": 1.0, "radius": 40, "shadow": .6, "tilt": -4}` | or `"anchor": "shoulder_l", "offset": [...]` to follow her |
| background replace | `"mode": "behind"` | the B-roll goes BEHIND the speaker (MODNet matte) |
| split screen | `"mode": "split_top"` / `"split_bottom"` | the other half = face crop of the speaker |
| green screen | `"key": {"color": "#00ff00", "tol": .2, "soft": .1, "spill": .6}` | YCrCb distance key + despill |
| logos / stickers with alpha | `"src": "logo.png"` (PNG alpha kept) | `mode full` + `fit: contain` |
| Ken Burns | `"kenburns": true` or `{"from": [1, .5, .5], "to": [1.12, .55, .45]}` | [zoom, cx, cy] |
| transitions | `cut fade dip_black dip_white zoom whip_left whip_right slide_up push_left luma_wipe circle blur` | `in_dur` / `out_dur` |
| stock footage match | `"grade": {"look": …}` INSIDE the broll layer | its own grader |
| lightsaber in her hand | saber layer `"start_anchor": "hand_r", "angle": -80, "length": 2.0` (head widths) or `"end_anchor"` + `"end_offset"` | tracked every frame |
| time in source seconds | `"t": "src:12.4"` | also `"e:3.2"` edited, `"s:3.2"` raw screen, plain numbers = edited (retime-safe) |
| lint before render | `python3 -m mograph.pro lint edit.json` | unknown types/keys (did-you-mean), files, anchors, hero overlaps |
| timeline panel | `python3 -m mograph.pro timeline edit.json` | cuts · retime · layers · subs · sfx · camera · grade |
| final master | `python3 -m mograph.pro render edit.json -j 2` | parallel, resumable, frame-exact chunks; + .srt/.vtt + timeline |
| client preview | `python3 -m mograph.pro render edit.json --preview` | half res, ≈3–4× faster |
| captions file | `python3 -m mograph.pro captions edit.json --fmt all` | SRT/VTT/ASS in screen time, RTL-safe |
| NLE round-trip | `python3 -m mograph.pro edl edit.json` | CMX3600 of the cut + layer markers |
| YouTube chapters | `python3 -m mograph.pro chapters edit.json` | ≥ 10 s apart |
| thumbnail | `python3 -m mograph.pro thumbnail edit.json` | sharp + eyes open + hero on screen |
| other aspect ratios | `python3 -m mograph.pro reformat out/x.mp4 --to 1:1 --spec edit.json` | face-tracked crop, or `--mode blur` |
| platform rules | `python3 -m mograph.pro platform edit.json` | length / aspect / LUFS / UI safe zones |

## Depth order (v14 compositor)
`plate (graded LUT) → stage → BACK layers (incl. broll behind) → speaker (matte, light wrap/spill) → FRONT layers
(broll full/split z −50 → titles z 0 → subtitles) → bloom → AE-FX post → grain / halation / sharpen / letterbox → fade`

## Rules (pro editor)
1. B-roll carries an idea the speaker SAYS. Start it on the word (`"t": "w:…"`) and leave on a word end.
2. Cutaway ≤ 3 s, then back to her face (trust). Do not run a hero title over a full cutaway. Subtitles stay.
3. One look per video. Stock footage gets the SAME look inside the broll layer.
4. Whip / zoom transitions only on energy beats. Use fade or cut otherwise. Dip to black only at chapter breaks.
5. Always `lint` → `sheet`/`frame` (LOOK) → `render --preview` → `render`.

Test edit: `sdk/tests/legacy_maghale/edit_v19_pro.json (archived regression fixture — NOT a template)` (look + grain + cutaway + pip + background replace + hand saber).
