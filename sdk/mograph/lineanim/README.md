# lineanim — code-only line-art acting (v26)

- `puppet.py` — After Effects-style **Puppet Pin** tool:
  `warp_pins(img, src_pins, dst_pins, roi=...)` (MLS rigid, exact pins, ROI border
  auto-pinned so the warp blends seamlessly), `Track` keyframes with easing
  (`io`, `smooth`, `back` = overshoot), `Layer`, `compose`, `poly_mask`.
- `panel_anim.py` — `tonal()` (keep original tones), `soft_lines()`,
  `geodesic_order()` (one-line draw-on along the line network).
- `vectorize.py` — raster lines → centreline strokes (for SVG / re-drawing).

Example: `PYTHONPATH=sdk python3 sdk/examples/lineanim/birthday_puppet.py storyboard.png out.mp4 [--preview]`
Acting: boy tilts + smiles → his hand lands on her far shoulder → she turns, looks, smiles, blinks.
Whole scene is a pure `frame(t)` function (draw(t) pattern) → deterministic, beat-syncable.
