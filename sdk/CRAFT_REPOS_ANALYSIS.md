# storytold "craft" repos — what is useful for mograph (analysed 2026-10, v14.6)

Repos: `github.com/storytold/effectcraft` (After Effects clone) · `vectorcraft` (Illustrator clone) ·
`filmcraft` (Premiere clone).  All MIT/Apache-2.0, pure Rust, ~380–540 k lines each.  Every feature is a
**command with an id** that a headless CLI / MCP server can reach, so they are built for agents like us.

## Hard constraint check (our sandbox: 2 vCPU, ~1 GB RAM, 490 MB tmpfs /tmp, no GPU)
| | build from source | prebuilt Linux CLI | headless | RAM peak | verdict |
|---|---|---|---|---|---|
| effectcraft | ❌ (cargo on 380 k lines = hours, > 1 GB RAM) | ✅ 75 MB static binary | ✅ CPU software rasteriser | ~270 MB | **ADOPTED** (`sdk/bin/effectcraft-cli`) |
| vectorcraft | ❌ | ✅ (~120 MB tarball) | ✅ `vectorcraft-cli run … --export` | n/t | optional (below) |
| filmcraft | ❌ | ✅ (~70 MB) | ✅ | n/t | not needed (ffmpeg + our editor cover it) |

Never `cargo build` these in the sandbox; use the release binaries
(`releases/download/v0.11.1/effectcraft-0.11.1-linux-x86_64.tar.gz`).

## 1. EffectCraft fixes the "3D looks like icons" problem  (`mograph/pbr3d.py`)
What we use:
- **Advanced 3D renderer** (`crates/render/src/three_d/adv`): glTF 2.0 / OBJ / FBX / STL with PBR
  metallic-roughness + normal maps, Cook-Torrance GGX, **Environment light from an equirectangular HDRI**
  (irradiance + roughness mip chain → real reflections), PCF shadow maps, 2×2 SSAA.  The software path is the
  reference path, so it runs here without a GPU.  ~0.15–0.4 s per 520 px frame.
- `effectcraft-cli run <cmd> <json> …` builds a whole project in one call.  `prop.setExpression` bakes a per-frame
  pose table, so one `render` call produces a whole sequence (start-up is paid once).
- Shadow-catcher trick: `acceptsShadows = only` still renders the floor opaque.  So we render the floor twice
  (object shadow-only vs. nothing) and divide, which gives a clean cast-shadow matte.
- Gotchas found: a Spot light with a wide cone (140°) silently loses its shadow → use **Parallel**.  A 3D **Solid**
  does not catch shadows in Advanced 3D → use `layer.newPrimitive plane`.  There is no `layer.delete` command.

Also available but not wired in yet (each is one CLI call; good next steps):
- **Extruded 3D text** (`geometryOptions/extrusionDepth`, bevel styles), PBR-lit.  Works with our Persian fonts
  once they are in `~/.local/share/fonts` (tested with Doran: shaping and RTL are correct).
- 306 AE effects (Glow, CC Particle World, Fractal Noise, Turbulent Displace, Shatter, Card Dance, Caustics,
  Lumetri…), layer styles, Lottie import/export, JS expressions with the AE object model, motion blur, DOF bokeh.
- 3D primitives (cube/sphere/torus/cylinder…) with PBR materials: a better base for procedural objects than our
  own rasteriser.

## 2. VectorCraft (Illustrator): optional
- 3D Revolve / Extrude & Bevel / Inflate as **vector** shading (flat illustration look, not photoreal).
- Image Trace (bitmap → vector), Pathfinder booleans, Gradient Mesh, Envelope Distort, PDF/SVG export.
- Useful for the line-art / vector-noir styles (e.g. trace a generated image to clean SVG for draw-on animation).
  Not adopted yet: the binary is 120 MB, so download it only when a task really needs Image Trace.

## 3. FilmCraft (Premiere): not adopted
- NLE, colour grading (wheels, curves, HSL secondary), EBU R128 meters, DeNoise, its own H.264/HEVC/ProRes codecs,
  a Whisper speech crate.  ffmpeg + `v14/audio.py` + DeepFilterNet + our grade already cover what we need.

## Assets used by pbr3d
- Models / HDRIs: **Poly Haven** (CC0, no attribution required) through its public API, 1k textures (0.5–3 MB per
  model).  `python3 -m mograph pbr3d-search <words>` / `pbr3d-fetch <id>`.
