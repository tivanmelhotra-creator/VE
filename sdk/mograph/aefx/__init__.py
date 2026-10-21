"""
mograph.aefx — v16 "AE-FX": After-Effects-grade motion toolkit re-implemented from scratch for the agent engine.

Inspired by (data studied as reference, NO vendor code/binaries copied):
  * AtomX Text Animation Presets  → textanim.py  + presets/text_presets.json   (351 text animators)
  * Premiere Motion Presets        → motion.py    + presets/motion_presets.json (352 in/out/fx transitions)
  * Deep Glow 2                    → deepglow.py  (physically-plausible multi-scale glow)
  * PathBlurFX                     → pathblur.py  (blur along an arbitrary path with taper / falloff)

Shared:   keyframes.py  (AE temporal bezier: speed + influence)
CLI:      python3 -m mograph.aefx list|info|sheet|selftest       (see AEFX_GUIDE.md)
"""
