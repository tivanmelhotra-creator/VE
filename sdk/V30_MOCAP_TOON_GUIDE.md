# V30 — real mocap (`mograph.mocap`) + ball-head toon style (`mograph.toon`)

Restore after a sandbox reset: `bash sdk/tools/restore_human.sh` (moderngl, pycairo, EGL, MakeHuman assets).

## mograph.mocap — captured motion on our characters
* `library.py` — packs in the repo (`mocap/packs`, ~21 MB: Quaternius UAL1/2 CC0, three.js Mixamo samples MIT,
  49 CMU acting clips incl. 2-person takes, BVH) + 80 aliases (`walk`, `talk`, `sit_talk`, `cmu:22_04` …).
  Own clips: `python3 -m mograph.mocap import Talking.fbx --name talking` (FBX → GLB with FBX2glTF, fetched on demand).
* `retarget.py` — world-delta retarget onto the MakeHuman skeleton; free hips (bounce/sway), constant ground,
  root motion `file | in_place | plant` (planted foot never slides).
* `director.py` — actors = crossfaded mocap segments + additive look / lean / reach-IK / breath layers.
* `props.py`, `face.py`, `film.py`, `strip.py` (strip a GLB to animation only), `bvh.py`, `gltf.py`.
* Example: `sdk/examples/mocap/birthday_mocap.py` (two actors, props, line-art film).

## mograph.toon — white ball-head toon look
* `face.py` — 28 expressions drawn on the head ball (round screen-space eyes, far eye hidden in profile, blink, talk).
* `rig.py` `ToonRig` — procedural 3D toon model skinned to the MakeHuman skeleton (any mocap clip drives it).
* `character.py` `ToonCharacter` — MakeHuman body + ball head.
* `render.py` — flat fills + ink lines from the id buffer; `shade.py` — soft cel-shaded render (white body, warm
  shadow band, thin outlines), `render_shaded(..., shadow=False)` turns the floor cast shadow off.
* `fx.py` — sweat, Zzz, sparkles, motion lines.
* Examples: `sdk/examples/toon/turnaround.py`, `clip_strip.py`, `mood_sheet.py`.

## Not included (on purpose)
The one-day experiment of auto-rigging AI-generated GLB meshes ourselves (`glbchar.py`, bone-heat weights, delta
mush) is left out: a dedicated auto-rigger (Adobe Mixamo) gives far better skin weights. Next step = render
Mixamo-rigged FBX characters directly in this toon style.
