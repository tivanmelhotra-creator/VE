# V14 audit — why v13.9 looked "similar but not good", and what v14 changed

v13.9 copied the *shapes* from 7 screenshots (dome, CD, hex, halo…) but the soul of @novadent is in the motion,
depth and light between frames — which screenshots cannot show and the v13.9 code could not produce.

| # | v13.9 problem (root cause in code) | Visible result | v14 fix |
|---|---|---|---|
| 1 | Every animation = one scalar `p` driving alpha/scale (`elements.compute_state`) | text "slides like dry bread" | `v14/physics.py`: velocity springs, squash/stretch, decaying jelly wobble, per-word secondary motion |
| 2 | "3D" = 2D `ctx.scale + rotate` (orbiting_title_3d) | flat, fake orbit | `v14/glyphs.py`: glyph outlines → 3D points → perspective; real extrusion; cylinder ring split in front/behind the head by z |
| 3 | Curved text rasterised then `cv2.remap` warped | blurry arcs, no animation | vector arc fields; text stays sharp and animates per word |
| 4 | Live causal one-euro tracking | layers trail the speaker | offline pose, centred Gaussian smoothing = zero lag; anchors head_top/chest/hands… |
| 5 | 256 px selfie segmenter + hard threshold | halo edges, text behind looks pasted | MODNet portrait matting, temporal median, light wrap on the edge |
| 6 | "Light spill" = gradient painted on top | no real interaction with the person | emission buffer → bloom + wrap + spill computed on the speaker's own pixels |
| 7 | Glassmorphism sampled the cairo target mid-draw, often before the plate | flat translucent fill | glass reads the finished composite under its shape (true frosted glass) |
| 8 | Jelly subtitle wobbled around current width only | capsule pops between phrases | stateful width spring across phrases + area-preserving height + karaoke pop |
| 9 | Per-glyph rotation pivots on glyph centres | Persian joins break during motion | word-pivot motion (cursive-safe), glyph unit optional |
| 10 | Fonts assumed 1000 upem | Morabba/Doran/IRANSans (2048/2000/2300 upem) rendered 2× | outlines normalised by each font's real upem |
| 11 | Sound scheduled per element type only | SFX not tied to flips/pulses/subtitles | `v14/audio.py`: events from the same timing maths (flip, pulse, hero hit, shake, phrase change) |
| 12 | Work lived outside `/home/user/webapp` | lost on sandbox reset | everything in webapp + `CONTINUE_HERE.md` + `tools/setup_env.sh` |
