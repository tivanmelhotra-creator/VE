# V25 — `mograph.human`: production line-art characters from a real 3D human

Branch: `feat/v25-human-lineart` · Restore after a sandbox reset: `bash sdk/tools/restore_human.sh` (~40 s)

## What it is
A parametric 3D human (MakeHuman HM08, **CC0**: mesh, 163-bone skeleton, weights, 660 shape targets) dressed with
procedural clothes/hair/shoes, posed by an anatomical pose language, and drawn as **vector line art** (smooth
occluding contours + hems + folds, tapered strokes, weight hierarchy). Output: PNG/SVG stills, alpha clips
(ProRes 4444 / VP9 / PNG sequence), or a live `human_lineart` layer in the v14 video engine.

## Quick use
```bash
cd sdk
python3 -m mograph.human presets                                  # presets / poses / motions / outfit options
python3 -m mograph.human still office_woman --pose present --yaw 25 --out a.svg
python3 -m mograph.human still casual_man --outfit top=jacket,hair=side --body weight=0.8,age=50 --out b.png
python3 -m mograph.human clip business_man walk_cycle --seconds 2.2 --yaw 70 --out walk.mov   # alpha
python3 -m mograph.human sheet --out sheet.png                    # QA contact sheet of all presets
```
Python:
```python
from mograph.human.character import Character, preset
ch = Character(**preset("old_woman"), face="minimal")             # face: none | minimal | simple
rgba = ch.render(pose="think", yaw=30, W=1080, H=1920)            # premultiplied RGBA
svg  = ch.svg(pose="sit", yaw=45)
from mograph.human.clip import render_clip
render_clip(ch, {"keys": [(0, "stand"), (0.8, "wave", {"t": 0.3}), (2.0, "present")]}, "gesture.mov", seconds=2.5)
```
Edit layer (v14 spec):
```json
{"type": "human_lineart", "t": "w:قدم", "out": "we:خانه+0.4", "character": "office_man", "motion": "walk_cycle",
 "x": 760, "y": 1700, "height": 900, "yaw": 70, "color": [1,1,1], "width": 3.2, "enter": "draw", "exit": "undraw"}
```

## Body (`body.py`)
`Body(gender 0..1, age years 1..90, muscle, weight, height, proportions 0..1, fat −1..1, belly, shoulders, hips −1..1)`
→ MakeHuman macro algorithm + local fat/thin layers (a heavy body really reads heavy). Units dm, Y-up, +Z front,
+X = character's left. LBS skinning; `pose_matrices(rot)` with world-axis rotations about bone heads.

## Poses (`poses.py`)
Segment aims in body space: `(abd, flex)` degrees from straight down (abd = out to the side, flex = forward).
Minimal-rotation aiming (hinge-like elbows/knees, no candy-wrap), spine/neck spread over vertebrae, hands
(relaxed/fist/open/point/hold/flat), feet kept flat + toe-out, ground lock. Library: stand, stand_contra, walk(phase),
run(phase), sit, sit_cross, wave, point, present, think, phone, hands_on_hips, arms_crossed, typing.
Walk/run use real gait curves (hip/knee/ankle sagittal angles, counter arm swing, pelvis yaw/roll).

## Clothes, hair, shoes (`garments.py`)
* tops: tshirt shirt longsleeve tank sweater hoodie jacket coat blouse (`open`, `tuck`, `sleeve` 0..1)
* bottoms: trousers jeans shorts leggings skirt pencil_skirt (`length`) · dress: short knee midi long
* shoes: shoes sneakers boots heels flats (lofted shoe last: toe box, heel counter, collar cut — toes never show)
* hair: short buzz side bob long bun ponytail bald (cap from a measured head frame: hairline clears the ears;
  long/bob = procedural curtain that falls behind the shoulders)
Garment = exact iso-cut of the body mesh by a smooth field → offset along smoothed normals → Taubin-smoothed
(no ribs/abs/nipples through cloth) → skinned with interpolated weights. Layering: top over bottom unless tucked.
Long coats = upper shell + open-front coat skirt with a hidden joint.

## Line renderer (`lineart.py`)
1. GL (moderngl/EGL, headless llvmpipe) linear depth + part-id buffers.
2. Smooth occluding contours (zero set of n_smooth·view on each triangle, crossing points keyed by mesh edge →
   exact chaining). Per-vertex smoothing (hands keep detail) and per-vertex `min_jump` (face / ear detail dropped).
3. Visibility: grazing-angle depth test + STRICT inter-layer test (any other part > 3.5 mm in front hides).
4. Importance = depth jump behind the line → outer silhouette heavy, inner contours light, tiny jumps dropped.
5. Hems (open boundaries, smoothed along the boundary), iso lines, 3D strokes (folds, face marks).
6. RDP + Chaikin, split at sharp turns, tapered variable-width ribbons, ink "pressure" wobble, SVG export.
Covered skin is not an occluder and its contours are clipped → no see-through junk.

## Folds (`folds.py`)
Pose-driven, drawn only when the joint bends: inner-elbow fan (1-3 strokes by angle), back-of-knee + front pull,
hip crease when seated, armpit drag when the arm is raised. Built on the garment surface, skinned with it.

## Clips (`clip.py`)
Camera framed once on the union of all frames, constant line scale, fixed wobble seed → no breathing/flicker.
Motions: walk_cycle, walk_across, run_cycle, wave, idle, any pose, keyframes, or a callable.
Speed ≈ 0.3 s/frame at 540×960 (CPU llvmpipe).

## Known limits / next
* faces: `simple` marks are basic; a stylised face kit (eyes/brows/mouth curves per expression) is next
* no cloth simulation (skirt/coat follow blended weights; extreme leg poses can intersect)
* hands at small scale can be busy — `Style.detail` / hand smoothing weight is the knob
* one-line continuous mode (single stroke) is not implemented yet
