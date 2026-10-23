# vecanim — code-only VECTOR line-art animation (v27)

Raster panel → vector scene (once) → rig JSON (once) → a few lines of acting → MP4 + SVG + Lottie.
No video model; nothing is warped as pixels — every frame moves Bézier control points.

| module | what | origin (Apache-2.0, ported to Python) |
|---|---|---|
| `bezier.py` | Schneider cubic fit, corners, line snap, resample/smooth | VectorCraft `trace/src/logo/{bezier,outline,geom}.rs` |
| `trace.py` | levels → skeleton graph (spur prune, junction join) → ink-centroid sub-pixel centre lines + width → strokes; two tone-layer fills with holes | VectorCraft `trace/src/centerline/*`, `fit.rs` |
| `scene.py` | Fill / Stroke / Scene: cairo render (variable width, reveal, opacity, clip), SVG, JSON | — |
| `arap.py` | ARAP puppet (Igarashi 2005), Position / Bend / Starch / Overlap pins, prefactored (splu) ≈ 3-4 ms per solve, barycentric embedding of vector points | EffectCraft `effects/src/puppet/{arap,mesh,mod}.rs` |
| `rig.py` | Character (claims items, mesh, pins, silhouette), deformer stack (ARAP + local MLS / squash / shift), Rig | — |
| `gestures.py` | `tilt_head turn_head nod sway smile blink look breathe lift_shoulder hand_on_shoulder` + vector hand asset | — |
| `anim.py` | Track keyframes + easing (`io smooth back out in hold`), expressions `wiggle lag spring`, `snap` to beats | — |
| `audio.py` | transient detector (spectral flux, triangular 24/oct bands) + procedural music box | SoundCraft `dsp/src/offline.rs` |
| `lottie.py` | Lottie 5.7 export (shape groups, path keys, trim paths, masks, camera null) + validator | EffectCraft `crates/lottie/src/export.rs` |
| `props.py` | text → vector fills, draw-on schedule, camera push, flame fx | — |

## Use
```python
rig = Rig.load('birthday_rig.json', Scene.load('scene.json'))
boy, girl = rig['boy'], rig['girl']
boy.do('tilt_head', t=2.7, dur=1.2, angle=3.5)
boy.do('hand_on_shoulder', target=girl, t=3.45, dur=1.5)
girl.do('blink', t=7.0)
img = rig.render(t, out_wh=(1920, 1080), M=camera(t))
```
Example: `PYTHONPATH=sdk python3 sdk/examples/vecanim/birthday_vector.py OUT [--preview] [--no-lottie]`

## Measured (birthday panel, 2-core sandbox)
- trace: 14 s once → 18 fills + 404 strokes, 7.7k control points; 1-px tolerant precision 0.92 / recall 0.985
- rig build 0.6 s once; pose (all deformers, both characters) 23 ms/frame; 1080p cairo draw 290 ms/frame
- audio: 13/13 notes, 0 false hits, median 14 ms error (min_gap 0.15)
- Lottie: 7 layers, 507 paths, 9.6 MB (0.45 MB gzipped); loads in lottie-web 5.12 (270 frames)

## Rigging a new character (once)
In the rig JSON: `polygon` (rough outline), `landmarks` (neck, eyes, mouth corners, shoulders),
`pins` + `pin_groups` (head / chest / shoulder_r). MediaPipe pose works on the boy crop of this
panel (nose/shoulders plausible) but gives inconsistent results on the girl → landmarks were set by hand
(~10 points per character). A click tool is the planned next step.

## Known limits / next
- strands of hair are traced as many short strokes (looks slightly "busy" vs. the source)
- the hand is a generic asset; the arm itself is not drawn (hidden behind her back by staging)
- no secondary motion on hair/cloth yet (spring chains on stroke points would do it)
- Lottie size: path keys every 3 frames; key reduction (fit per-point curves) would cut it ~5×

Restore after a sandbox reset: `bash sdk/tools/restore_vecanim.sh`
