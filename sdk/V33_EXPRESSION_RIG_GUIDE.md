# V33 — Expression Rig (artist drawings at any head angle)

Pipeline for an agent: one reference image of a reaction → rig-ready drawings → video.

```
reference sheet / image ──sheetlib──▶ expression library (front decals, labels, parts)
                         ──redesign prepare──▶ jobs.json  (reference crop + exact prompt per expression)
   image model (agent)   ──────────────────▶ turnaround png (bald head: front · 3/4 · profile)
                         ──redesign ingest──▶ rig library (per-view decals, facing, parts)
                         ──decalrig──────────▶ video (tracked head, any yaw, occlusion, FaceShape, gender)
```

## 1. Build the expression library from sheets
```bash
cd sdk
python3 -m mograph.motion.toonface.kit.sheetlib "tests/fixtures/toonface/sheets/*.png" --out /tmp/lib
```

## 2. Prepare redesign jobs (no model call)
```bash
python3 -m mograph.motion.toonface.kit.redesign prepare --lib /tmp/lib --out /tmp/jobs
```
Each job in `/tmp/jobs/jobs.json` has `ref` (crop of the reaction) and `prompt`. Call the image model
(gpt-image-2 works well) with `image_urls = [ref, character_plate]` and the prompt. To add a NEW reaction that
is not in any sheet: `prompt_for(name, desc="…what the eyes / brows / mouth do…")`.

Character plates: `sdk/tests/fixtures/toonface/plates/{bf,b34,gf,g34}_blank.png` (front + 3/4 side by side).

## 3. Ingest the turnaround
```bash
python3 -m mograph.motion.toonface.kit.redesign ingest --png smug_turn.png --name smug --out /tmp/rig_lib
```
Produces `smug@front`, `smug@q34`, `smug@profile` with facing direction; back-of-head marks and ears are removed.

## 4. Render
```bash
python3 -m mograph.motion.toonface.kit.decalrig --video VIDEO.mp4 --lib /tmp/rig_lib --out out.mp4 \
    --script "0:smug,1.7:pleading,3.4:sulking" [--femininity 0..1] [--preset big_cute_eyes] [--shape char.json]
```

## How any angle works (decalrig.py)
* Keys: front (0°), 3/4 (±40°), profile (±85°) — mirrored automatically for the other side.
* Between keys every part (brow_L/R, eye_L/R, nose, mouth) is interpolated in position and size; parts that do
  not exist in one key (far eye in profile) are carried by an ellipsoid head model and hidden when they turn away.
* The drawing itself swaps at mid-interval (classic drawing substitution) with hysteresis, so no ghosting and no
  flicker when the head hovers around the switch angle.

## Character customisation (FaceShape)
`eye_scale, eye_spacing, eye_y, eye_round, brow_y, brow_len, brow_thick, nose_scale, mouth_w, mouth_y, line,
lashes, brow_layer ("under" hair = clipped to face | "over")`. Save per character with `FaceShape.save(path)`.
Presets: `as_drawn, big_cute_eyes, sharp_narrow, wide_set, brows_over_hair`.

## Male ↔ female
`gender_delta()` measures the boy and girl kits (same artist) in face units and returns the difference as a
FaceShape (girl: eyes ≈0.94×, spacing ≈0.91×, line ≈0.79×, mouth ≈0.87×, brows thinner, lashes on).
`character_shape(base, femininity=t)` blends continuously; t = 1 reproduces the girl proportions.
