# V32 — FaceTrack Pro (toonface/kit)

Replaces `videotrack` (v31.3) and `WeldedSkullSandwichRig` (v31.4) for blank-face cartoon video.

```bash
cd sdk && python3 -m mograph.motion.toonface.kit.facecomp \
  --video in.mp4 --char boy|girl --out out.mp4 --debug     # + out_debug.mp4 + out_qa.json
```

## Modules
| file | role |
|---|---|
| `facetrack.py` | Mocha-style planar tracker: KLT points in 4 regions (crown/left/right/jaw) on hair/ears/outline, RANSAC + region-balanced Huber IRLS, every frame solved ref→frame (no drift), One-Euro filter, `PoseLimits` (roll dead-band 1.2°, clamps, velocity limits) |
| `expressive.py` | FACS AUs per side, 19 emotions, `blend()`, coupling (smile→squint, gaze-down→lids, jaw→brows), per-channel springs (brow overshoot), anticipation, blinks, saccades |
| `facecomp.py` | kit blank plate chamfer-fit to the video face (correct size/position — no `eye_span_ratio`), `LevelPolicy` (eye line ≤1°, mouth ≤3°, cal roll <2.5°→0), cairo render directly in frame space, partial hand occlusion, QA |

## Tuning knobs
- springiness/lag: `TrackerConfig.pos_min_cutoff / pos_beta / rot_min_cutoff`
- unwanted tilt: `PoseLimits.roll_deadband_deg`, `LevelPolicy(max_eye_tilt_deg=…)`
- feature size art-direction: `--face-scale 1.05`
- force view: `--plate front|34|34m --yaw …`

## QA (cafe clips, 120 frames)
boy: plate front, jitter median 0.12 px, cal 11.5 px · girl: plate 34m (yaw −27), jitter median 0.19 px, cal 4.9 px, 20 frames partially occluded (forehead hand → brow hidden, eyes/mouth kept).

## Next
3/4 yaw changes during a shot (dynamic plate switch), hand-over-mouth tests, auto beat sheet from audio emotion, wire into `cafe_wallet_pipeline`.
