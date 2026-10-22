# v18 MOTION KIT — `mograph.motion` (agent guide)

Six engines, one vocabulary. **First:** `python3 -m mograph.motion selftest` → 10/10.
Every feature works in **v14 footage edits** (`"layers"`) and **v5–v13 scenes** (`"elements"`) with the same keys.

## 0. User words → engine keys (do this FIRST)
```bash
python3 -m mograph.motion resolve "یه اسلوموشن نرم با ease out و افکت نئون آبی روی متن"
# → retime/slow_motion · easing/ease_out · saber/neon + target text_layer + color #2a7dff
```
`glossary.yaml` (user brief: easing · speed · character_animation · lip_sync · vfx_glow) + Persian/English aliases.
Longest phrase wins; sequences are kept (`"ops": ["speed_ramp", "freeze_frame"]`); colours are detected.

| glossary | canonical | spec | CLI to LOOK |
|---|---|---|---|
| easing / منحنی حرکت / بزیه | `easing` | `"ease": …` on any element/layer/keyframe | `ease graph …` · `ease measure …` |
| speed / اسلوموشن / تایم‌لپس / ریمپ | `retime` | `"source": {"retime": [...]}` (v14) | `retime-plan` · `retime in out` |
| کارتونی / خرفه‌ای / rubber hose | `character_animation` | `{"type": "character"}` | `character sheet` |
| لیپ سینک / دهان با صدا | `lip_sync` | `"lip_sync": {...}` / `{"type": "mouth"}` | `lipsync chart` · `lipsync track` |
| سیبر / نئون / glow / شمشیر نوری | `saber` | `{"type": "saber"}` · `"fx": [{"saber": …}]` | `saber sheet` · `saber list` |

## 1. Easing — `motion/easing.py`
`ease(spec)` accepts ALL of: `"out_back"`, `"easeOutBack"`, `"back.out"`, `"ease_out"`, `"کاهش سرعت"`,
`"cubic-bezier(.2,.8,.2,1)"`, `[x1,y1,x2,y2]`, `"steps(6)"`,
`{"type":"elastic","dir":"out","amp":1.2,"period":0.3}`, `{"type":"back","overshoot":2.2}`,
`{"type":"bounce","bounces":3,"decay":0.35}`, `{"type":"spring","stiffness":260,"damping":11}`,
`{"type":"ae","out_speed":0,"out_influence":0.75,"in_influence":0.9}` (AE keyframe velocity dialog).
Glossary types: ease_in ease_out ease_in_out cubic quart expo back elastic bounce linear (+ sine quad quint circ).
`mograph.core.get_ease` falls back to this → **every old `"ease"` key now takes these specs**.
Penner curves are exact (err 1e-15). Choose by numbers: `ease measure out_back` → overshoot 10 %, peak speed ×4.7.
Named looks: smooth snappy anticipate punch soft_land whip swift_out material_* apple_spring ae_easy_ease ae_expo_75.

## 2. Retime — `motion/retime.py`
v14 spec (times = EDITED seconds or word anchors; `"e:3.2"` = edited time explicitly):
```json
"source": {"src": "media/raw.mp4", "words": "words.json", "segments": [...],
  "retime": [
    {"op": "speed_ramp", "t": "w:مقاله", "dur": 0.3, "from": 1, "to": 0.3, "hold": 0.3, "back": 0.3},
    {"op": "freeze_frame", "t": "we:چکیده", "dur": 0.7},
    {"op": "slow_motion", "t": 6.0, "dur": 1.0, "speed": 0.35, "ramp": 0.2},
    {"op": "fast_motion", "t": 9.0, "dur": 3.0, "speed": 6},
    {"op": "reverse", "t": 12, "dur": 1.0, "mode": "rewind|replace|boomerang"}],
  "speed": 1.0, "interpolate": "flow"},
"audio": {"retime": "auto|varispeed|stretch|mute"}
```
* the whole timeline (word anchors, subtitles, cuts, sfx, `duration`) moves to on-screen time automatically
* slow motion in-betweens = bidirectional DIS optical flow (+ matte blended) — 12 px pan error 0.002 vs blend 6.9
* fast motion = shutter blend of 2–5 source frames; whoosh / hit / scratch sfx at op starts
* ops may not overlap: the error message tells the first legal start (`"e:3.18"`)
* any file, no spec: `python3 -m mograph.motion retime in.mp4 out.mp4 --ops '[{"op":"slow_motion","t":1,"dur":1}]'`
* test edit: `sdk/tests/legacy_maghale/edit_v18_retime.json (archived regression fixture — NOT a template)`

## 3. Saber — `motion/saber.py` (Video Copilot Saber parameter set, own implementation)
51 presets = Saber's list (Arc Reactor … Zap) → `python3 -m mograph.motion saber sheet --core text --text "سلام"` → LOOK.
```json
{"type": "saber", "preset": "neon", "core": "text", "text": "مقاله", "size": 160,
 "t": "w:مقاله", "out": "we:مقاله+1.2", "anchor": "head_top", "offset": [0, -0.7], "reveal": {"dur": 0.5}}
{"type": "saber", "preset": "default", "start": [140, 760], "end": [940, 640], "core_size": 1.2, "reveal": {"dur": 0.4}}
{"type": "saber", "preset": "electric", "core": "shape", "shape": "circle|rect|hex|triangle|star|heart|line", "w": 380}
{"type": "saber", "preset": "energy", "core": "path", "paths": [[[x,y],…]], "closed": true, "evolution": {"k": [[0,0],[2,100]]}}
"fx": [{"saber": {"preset": "fire"}, "t": 2.0, "dur": 1.5}]          ← glow ANY layer from its own alpha
```
Cores: saber (blade, start/end size, roundness) · path (= masks) · shape · text (`text_mode` outline neon | fill) · alpha.
Look keys: color glow_intensity glow_spread glow_bias core_size core_softness halo_intensity flicker_intensity
flicker_speed distortion (smoke|fluid|energy|none) distortion_amount distortion_mode (displace|multiply) noise_scale
noise_speed noise_complexity noise_aspect wind_speed wind_dir core_distortion lightning{bolts,rate,jitter,branches,width}
glows [[i,s]×4] ramp (heat colour ramp) gamma brightness saturation alpha_boost composite (transparent|black).
Any scalar may be keyframed `{"k": [[t, v], …], "ease": "out_back"}`. Text reveal wipes RTL (Persian).
v14: light is ADDED over the plate; `"spill": true` (+ `"spill_amount": 0.12`) also lights the speaker via the emission buffer (keep it low: the v14 bloom re-glows it and can fill text counters).
Cost: ≈0.4–0.7 s per 1080×1920 frame (`"quality": 0.35` faster). Capabilities → presets: `saber list`.

## 4. Character — `motion/character.py`
```json
{"type": "character", "rig": "bean|blob|star", "x": 540, "y": 1700, "size": 420, "color": "#ffcc33",
 "actions": [{"do": "enter", "t": 0, "from": "left"}, {"do": "hop", "t": 1.2}, {"do": "wave", "t": 2.0},
             {"do": "talk", "t": 3.0, "dur": 2.5}, {"do": "point", "t": 5.6, "side": "r"}, {"do": "exit", "t": 8}],
 "lip_sync": {"words": "words.json", "audio": "media/vo.wav"}, "timing": "twos", "exaggeration": 1.2}
```
Actions: idle enter exit hop jump walk wave point talk shrug nod shake surprise celebrate look spin squash.
Principles are built in AND measured (`character sheet` prints them): squash & stretch (volume error < 0.05),
anticipation (crouch/wind-up), follow-through (antenna spring), overlapping/secondary (blink, breath, sway),
arcs (parabolic jumps), exaggeration (multiplier), timing (easing + on twos/threes), staging (look/lean to target).
v14: `"anchor"` works like other layers; lip sync uses the EDITED (and retimed) transcript automatically.

## 5. Lip sync — `motion/lipsync.py`
Preston Blair chart: rest MBP etc E AI O U WQ FV L (Rhubarb letters X A B C D E F G H) → `lipsync chart`.
Persian G2P infers unwritten short vowels (CV/CVC/CVCC syllables); English digraph rules. Closures ≥ 2 frames,
1-frame anticipation, rests in pauses > 0.18 s, audio RMS → jaw opening. `{"type": "mouth"}` = mouth only.
```bash
python3 -m mograph.motion lipsync g2v مقاله سلام hello
python3 -m mograph.motion lipsync track ../projects/maghale/words.json --audio ../work/input/raw.mp4 -o vis.json
```

## 6. Files
`motion/{glossary.py,glossary.yaml,glossary.json,easing.py,retime.py,saber.py,character.py,lipsync.py,bridge.py,selftest.py,__main__.py}`
Hooks: `v14/compositor.py` (retime timeline + in-betweens), `v14/audio.py` (voice remap + retime sfx),
`v14/layers.py` (DRAW saber/character/mouth), `aefx/bridge.py` (fx item `saber`), `core.get_ease`, `scene.py`.
No new binaries, no new pip deps (numpy/cv2/cairo/scipy already required) → repo size +≈0.2 MB.
