# V14.4 — 3D OBJECTS GUIDE (merged from the v13.9 3D branch + rewritten)

> Origin: the 3D work was built in a separate agent on the older v13.9 SDK (`sam3d_bridge.py`, `sculpt3d.py`,
> `laptop_yaw_*.png`, template 18). v14.4 merges it into the main SDK and **fixes / replaces the unreliable parts**.
> Design ideas borrowed from **img2threejs** (github.com/img2threejs/img2threejs): ObjectSculptSpec component tree,
> material registry, strict-quality gate before rendering, one reference-vs-render comparison sheet per review,
> camera-pose match. "Scripts enforce, the agent judges."

## 0. v14.6 — `pbr3d`: PHOTOREAL objects (the default for any real-world object)
**Why:** `real3d` / `sculpt3d` / `image_to_3d` build objects from primitives + procedural noise → they read as
"3D icons" (clean, plastic, no real texture).  Photoreal needs three things we cannot make procedurally: real
geometry detail, real PBR texture maps (albedo / normal / roughness / AO from photogrammetry) and image-based
lighting from a real HDRI.  `pbr3d` takes the first two from **scanned CC0 models** (Poly Haven, 500+; or any
`.glb/.gltf/.obj/.fbx/.stl`) and the third from the **EffectCraft Advanced-3D renderer** (Cook-Torrance GGX, HDRI
environment light, PCF shadow maps, 2×2 SSAA), bundled as one static CLI `sdk/bin/effectcraft-cli` (no GPU;
`setup_env.sh` downloads it if missing — it is NOT in git, 75 MB).

Cost in this sandbox (2 vCPU, 1 GB RAM): ~270 MB peak, ~0.3 s per frame (object pass + half-res shadow pass).
The whole on-screen interval of an object is rendered ONCE in one CLI call (poses baked into an expression table)
and cached in `~/.cache/mograph_pbr3d/<hash>/` as 8-bit PNGs (~100 KB/frame); later frames only load a PNG
(measured: 3 objects × 5 s = 136 s once, then ~2 s per full video frame).  **Never** put the cache in `/tmp`
(tmpfs = RAM, 490 MB → disk-full crash, seen).

```bash
python3 -m mograph pbr3d-search food            # catalogue;  pbr3d-search studio hdri  → lighting HDRIs
python3 -m mograph pbr3d-fetch croissant lemon  # glTF + 1k maps (0.5–3 MB each) → mograph/pbr_assets/models/
python3 -m mograph pbr3d-sheet croissant        # 10-angle audit sheet (top / bottom / back): LOOK before using
```
v5–v13 scene element:
```json
{"type": "pbr3d", "t": 0.2, "out": 4.5, "x": 540, "y": 900, "w": 520, "model": "croissant",
 "motion": "spin_360", "yaw_speed": 72, "pitch": 18, "hdri": "studio", "shadow": true, "light": {"key": 60, "env": 100}}
```
v14 footage layer — the same `object_3d`, just give `"model"` (spin-in, jelly pop, `turn_on`, anchors all work):
```json
{"type": "object_3d", "t": "w:مقاله-0.3", "out": "w:چکیده-0.1", "model": "croissant",
 "anchor": "shoulder_l", "offset": [-0.35, -0.9], "size": 1.25, "motion": "hover", "yaw": -25, "pitch": 18}
```
Python: `pbr3d.render_sequence(model, size, [(yaw, pitch, roll), …])` → lazy frames H×W×5 (RGB, alpha, shadow).

Rules / gotchas (all tested)
- Models are normalised (bounding sphere → radius 1, centred) into `<name>.norm.gltf`; `w`/`size` = sphere diameter.
- Shadow = real cast shadow on an invisible floor touching the object's lowest point (`floor: "auto"`).
  Key light is **Parallel** (a wide Spot cone silently loses its shadow); catcher must be a plane **primitive**
  (a 3D Solid does not receive shadows in Advanced 3D).  `shadow_strength` scales it in the composite.
- Object not on Poly Haven?  In order: a similar CC0 scan → any glTF the user supplies / an image-to-3D service →
  only then `real3d` / `sculpt3d` (§0b).
- Look: `hdri` = any Poly Haven HDRI id (`brown_photostudio_02`, `ferndale_studio_05` …), `env_rot` rotates it
  (moves reflections), `light.key` 0–150 (key + shadow), `light.env` (ambient / reflections).
- Tests: `tests/v146_pbr3d.json` · footage: `projects/maghale/edit_v14_pbr3d.json` · compare sheet:
  `work/renders/3d_compare_v14_6.jpg`.

## 0b. v14.5 — `real3d`: TRUE 3D textured product models (use this when the object must turn freely)
`image_to_3d` only thickens a photo → it breaks past ~60°.  `real3d` rebuilds the object as real closed geometry
(img2threejs component-tree idea) with procedural textures whose palette is sampled from the reference photo, and
renders it with a deferred PBR pipeline: numba z-buffer G-buffer (2× SSAA) → per-pixel albedo / roughness /
metalness / height → screen-space bump → key/fill/rim + hemisphere + studio-softbox reflections → key-light SHADOW MAP
→ SSAO → ACES.  360° yaw, top, bottom, back — every view is a real view.  ~0.3–0.7 s / object at 400 px.

| model | geometry | texture |
|---|---|---|
| `soda_can` | lathe: dome bottom, stand ring, neck, double-seam rim, recessed lid, tab + ring + rivet, opening | brushed aluminium, condensation droplets (bump + gloss) |
| `pizza_slice` | polar height-field (melted lumps, droop), baked underside + cut walls, lumpy crust tube | cheese/browning/blisters, sauce at edges, crust pores + flour, charred spots, cupped pepperoni with fat, mushrooms, olive ring, pepper |
| `growth_card` | 2 stacked rounded cards (analytic bevel normals), embossed UI bars + dots, bevelled extruded zig-zag arrow | matte plastic, orange arrow; arrow casts a real shadow on the card |

```json
{"type": "real3d", "t": 0.2, "x": 540, "y": 800, "w": 460, "model": "pizza_slice", "motion": "spin_360", "pitch": 30}
```
Files: `mograph/real3d.py` (renderer, `paint`, element) · `mograph/real3d_models.py` (models — add new ones here) ·
`mograph/zraster.py:raster_attr` (G-buffer rasteriser).  Demo: `work/renders/demo_real3d_green.mp4`.
New model recipe: block out with lathe / height-field / rounded box / bevel_prism / torus, sample 5–8 colours from
the reference, write one material fn per part (albedo + rough + metal + height), check top/bottom/back with a sheet.

## 1. Which tool for which object
| Source | Use | Element (v5–v13 spec) | Layer (v14 footage) |
|---|---|---|---|
| ONE RGBA image (assets_3d / image_gen) | flat-ish object turning ±60° | `image_to_3d` geometry `slab` / `badge` / `cushion` | `object_3d` `"src"` |
| 2+ real views of one object (`<stem>_yaw_m45.png …`) | most realistic turntable | `image_to_3d` (auto `multiview`) | `object_3d` (auto) |
| object you can describe as parts (pen, implant, laptop, trophy) | full 360°, moving parts | `sculpt3d` `"preset"` / `"tree"` | `object_3d` `"preset"` / `"tree"` |
| `.ply` / `.obj` (SAM-3D, Gaussian splat, img2mesh) | point cloud | `sam3d_object` | — |
| texture on a sphere / capsule | globe, pill, vial | `image_to_3d` geometry `sphere` / `capsule` | — |

**Rule:** a single image has no back. `slab`/`cushion` show a solid plate when turned past 90° (never a mirrored
front). If the back will be on screen for long → use `multiview` (generate 4 views) or `sculpt3d`.

## 2. Engines (files)
- `mograph/obj3d.py` — extruded sprite renderer (correct perspective, front/back visibility, gap-free wall whose
  slice count follows the projected depth, rounded profile, relight + moving sheen), multiview turntable, motion
  presets, `audit_sheet`.  ~70–120 ms/object at 300 px.
- `mograph/sculpt3d.py` — ObjectSculptSpec meshes (box, rounded_box, cylinder, cone, capsule, sphere, torus, lathe,
  **extrude**), hierarchy with pivots + `anim_rot`, decals (`textured_quads`), **per-pixel z-buffer + Gouraud**
  (`zraster.py`, numba), material registry `MATERIALS`, `validate_tree`, `fit_pose`, `comparison_sheet`.
- `mograph/sculpts/*.json` — reusable trees (`dental_implant.json`). `"preset": "dental_implant"` finds it.
- `mograph/sam3d_bridge.py` — elements `image_to_3d`, `sam3d_object`; sphere / capsule ray-casters; `img2mesh`;
  v13.9 functions kept as `*_v139` for A/B only.
- `mograph/v14/object3d.py` — v14 layer `object_3d` (tracked anchor, jelly pop + spin-in, `turn_on` word spin,
  layout guard "side", whoosh_pop SFX).

## 3. What was broken in v13.9 (and is fixed)
1. Slab perspective inverted (back face projected bigger than front) → depth looked inside-out.
2. Past 90° the front image was drawn mirrored instead of the back.
3. Side wall made of 52 fixed slices → stripes / gaps at steep angles; ~450 ms per object.
4. Cushion: moiré rings on the wall (Lanczos per slice), ~950 ms.
5. Multiview: snapped between views (flow code never called) → pops; auto-detection pointed at `/tmp/...`.
6. sculpt3d: models rendered at ~⅓ of the requested size; light from below (dark tops); whole-part sort when decals
   existed (lid popping through base); LINE_AA seams; flat facets.
7. `src` paths only worked relative to cwd; binary PLY parser assumed all-float properties.

## 4. Spec examples
```json
{"type": "image_to_3d", "t": 0.2, "x": 540, "y": 700, "w": 380, "src": "shield_3d.png",
 "geometry": "slab", "depth": 0.14, "round": 0.5, "motion": "turntable_osc", "yaw_amp": 45}
{"type": "sculpt3d", "t": 0.2, "x": 540, "y": 1200, "w": 420, "preset": "dental_implant", "motion": "spin_360"}
```
v14 footage (`edit_v14_3d.json` is the worked example):
```json
{"type": "object_3d", "t": "w:مقدمه-0.1", "out": 15.85, "preset": "dental_implant",
 "anchor": "shoulder_l", "offset": [-0.85, -0.45], "size": 0.8, "motion": "turntable_osc", "yaw_amp": 35}
```
motion: `turntable_osc` · `spin_360` · `spring_entry` · `hover` · `none`.  `"turn_on": "w:…"` = one 360° on a word.

## 5. Quality loop (mandatory before an object goes into an edit)
```bash
python3 -m mograph turntable3d shield_3d.png              # every angle: slab / cushion / multiview rows
python3 -m mograph sculpt-check my_tree.json               # strict gate: ids, materials, ≥3 parts → PASS/BLOCKED
python3 -m mograph sculpt-compare my_tree.json ref.png     # auto camera match + silhouette IoU (≥0.80 PASS)
python3 -m mograph img2mesh cash_3d.png --out cash.ply      # image → .ply (for sam3d_object / external tools)
```
Then render real frames (`mograph frame` / `mograph.v14 frame`) and LOOK at them (QA catches edges, not taste).

## 6. Building a new sculpt (img2threejs workflow, short)
1. Detail inventory from the reference: list parts, materials (metal / plastic / glass / emissive), bevels.
2. Blockout tree in `mograph/sculpts/<name>.json` with real parts (lathe for round bodies, extrude for logos).
3. `sculpt-check` → `sculpt-compare` (IoU) → fix proportions → `turntable3d` → use it.
4. Report what is inferred (hidden sides) — never claim exactness from one image.

## 7. Not done yet (next steps)
- textures projected onto sculpt parts from the reference image (now only procedural laptop decals)
- depth-anything ONNX depth for `image_to_3d` relief (function exists, model not bundled)
- motion blur on fast spins; object reflections in the v14 glass layers
