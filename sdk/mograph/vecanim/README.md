# vecanim — code-only VECTOR line-art animation (v27-v29)

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

## v29 — one command: image -> film (no manual coordinates)

```bash
cd sdk
python3 -m mograph.vecanim templates                       # duo_affection, duo_friends, solo_portrait
python3 -m mograph.vecanim make PANEL.png --shot duo_affection -o OUT --preview   # contact sheet, ~10-25 s
python3 -m mograph.vecanim make PANEL.png --shot duo_affection -o OUT --lottie    # MP4 + SVG + Lottie
python3 -m mograph.vecanim rig  PANEL.png -o OUT            # trace + auto rig only (rig_debug.png)
```

Stages (`pipeline.py`), each cached by the panel's content hash in `OUT/cache/<hash>/`:

| stage | module | output |
|---|---|---|
| load (auto-invert dark-on-white, normalise) | `pipeline.load_panel` | `panel.npy` |
| vector trace | `trace.py` | `scene.json` |
| auto rig: faces, eyes, mouth, neck, shoulders, pins, z-order, hair springs | `autorig.py` | `rig.json`, `rig_debug.png` |
| shot: actions on `left/right/back/front/main/charN/all`, `t` in s or `beat:N`, snapped to music | `templates/*.json` | |
| auto camera (wide -> faces -> title), auto title placement in empty space, auto candle flame | `pipeline` | |
| parallel render (N processes, spring pre-roll, x264 chunks, concat + music) | `pipeline.render` | `<shot>.mp4`, `contact_sheet.png`, `final_pose.svg`, `timing.json` |
| Lottie (validated) | `lottie.py` | `<shot>.lottie.json` |

`rig.json` is never overwritten once created: fix a landmark by hand and re-run (`--rerig` regenerates).
A shot is plain JSON - copy a template and change times / gestures / title.

Measured, 2-core / 1 GB sandbox, cold cache:

| panel | items | prepare | render | lottie | total |
|---|---|---|---|---|---|
| birthday (1996x920, 12 s) | 405 | 15 s | 62 s | 14 s | ~100 s |
| unseen "friends" (1584x672, 10 s) | 87 | 5 s | 30 s | 4 s | ~47 s |

Previously the same result took ~2 h of manual rigging/keying; now ~1-2 min per new image.

Generality fixes found on the unseen panel: tracer no longer closes a skeleton "loop" whose ends
never meet (it drew a false straight line); `hand_on_shoulder(side='auto')` uses the far shoulder
when the actor stands behind and the near shoulder (mirrored hand) when in front; hand scales with
the target's face width; title placement keeps a margin from all ink.

Process note: every heavy stage runs in its own process (the orchestrator holds no scene), which is
what keeps two renderers + two x264 encoders inside 1 GB.

## v28 — professional pass (example: `examples/vecanim/birthday_v3.py`, 12 s)
- `secondary.py` — follow-through on hair / bun / collar: damped springs driven by the puppet's
  motion + optional wind wiggle, regions declared in the rig JSON (`"secondary": [...]`)
- `anim.py` — physical spring easings with the formula of Remotion's `spring()` (closed form):
  `spring`, `spring_soft`, `spring_firm`, `settle_time()`, `anticipate()`
- gestures now use anticipation + spring settles; `idle_blinks` (natural rhythm, double blinks)
- `scene.py` — ink-pen taper on open strokes, `Scene.line_scale`; vectorised sampling (≈1.8× faster)
- `props.py` — keyframed spring `Camera` with handheld wiggle; `finish()` (vignette + grain)
- parallel render: workers re-rig the scene, pre-roll 1.5 s for spring state, encode chunks, concat
- review (Gemini 3.1 Pro as a 2D director): v2 4/10 → v3 **8.5/10**; then fixed its 3 notes
  (soft staggered title, livelier flame, hand relaxes after landing)
- render: 360 frames 1080p in ~60 s on 2 cores (≈ 330 ms/frame/core)

### Remotion
Read for ideas (spring maths, frame-exact deterministic rendering, parallel chunk rendering, concat).
Not embedded: it is React + headless Chrome (heavy for this 1 GB sandbox) and its licence is
free only for individuals / companies ≤ 3 people.  A good later bridge: export each shot as
Lottie (`lottie.py`) and composite / title it in a Remotion project (`@remotion/lottie`).
