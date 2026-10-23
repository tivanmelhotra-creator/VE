#!/usr/bin/env bash
# One-shot environment restore for mograph.human (v25) after a sandbox reset (~40 s):
# pip moderngl + pycairo, libEGL (headless GL), MakeHuman CC0 assets → ~/.cache/mograph_models/human
set -e
cd "$(dirname "$0")/.."
python3 -m mograph.human.assets --setup
python3 -c "import moderngl; moderngl.create_standalone_context(backend='egl'); print('[human] EGL ok')"
