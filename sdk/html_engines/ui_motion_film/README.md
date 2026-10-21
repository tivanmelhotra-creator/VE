# 10s 2D UI Motion Film — Complete Project Package

## Contents
- `ui_motion_film.html` — Standalone 1920x1080 @ 60fps deterministic HTML5 Canvas engine (`window.seek(t)`) with interactive transport HUD, Web Audio API synthesizer, and `?clean=1` headless capture mode.
- `ui_motion_film_1080p60.mp4` — Full 10.00-second 1920x1080 @ 60fps H.264 MP4 video (600 frames) with synchronized stereo audio.
- `stills/`
  - `still_1_toggle.png` — Toggle switch click & ball launch (`t = 0.20s`)
  - `still_2_slider.png` — Slider tracking with vertical ticks & odometer tooltip (`t = 1.60s`)
  - `still_3_notifications.png` — Ball falling off the slider axis onto the top of the notification cards (`t = 2.52s`)
  - `still_4_keyboard.png` — Ball bouncing on top of the keyboard keys while typing `retry` (`t = 5.02s`)
  - `still_5_world_return.png` — Wide pull-back showing all UI components in one continuous world (`t = 8.30s`)
  - `loop_start_0s.png` & `loop_end_10s.png` — Loop boundary verification frames (`max pixel diff = 0`)
- `scripts/`
  - `build_film_html.py` — Generates `ui_motion_film.html`.
  - `render_stills_and_verify.py` — Renders preview stills at 1920x1080 via headless Chromium and verifies `t=0.000s` vs `t=10.000s` pixel parity.
  - `render_full_video.py` — Parallel 4-worker headless Chromium renderer (600 frames @ 60fps) + FFmpeg H.264/AAC MP4 encoder.
- `audio/`
  - `soundtrack_120bpm.wav` — Procedurally synthesized 120 BPM stereo WAV soundtrack + UI contact SFX.
  - `ref_audio.m4a` / `ref_audio.wav` — Reference audio streams.

## Usage
1. **Interactive Playback**: Open `ui_motion_film.html` in any modern browser. Use the bottom HUD to play/pause, step frame-by-frame (`-1f` / `+1f`), or jump to key beats.
2. **Clean 1920x1080 Capture Mode**: Open `ui_motion_film.html?clean=1` and call `window.seek(t)` in the console or via Playwright/Selenium for deterministic frame rendering (`t` in `[0.0, 10.0]`).
