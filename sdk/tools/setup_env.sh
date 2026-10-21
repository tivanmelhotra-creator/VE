#!/usr/bin/env bash
# One-shot environment restore for mograph v14 (run after ANY sandbox reset):
#   bash /home/user/webapp/sdk/tools/setup_env.sh
set -e
cd "$(dirname "$0")/.."
pip install noisereduce numba -q pycairo uharfbuzz fonttools freetype-py mediapipe onnxruntime opencv-python numpy scipy pillow 2>&1 | tail -1 || true
(sudo apt-get install -y -q libegl1 libgles2 >/dev/null 2>&1) || true
# big binaries are NOT in git (repo < 100 MB): effectcraft-cli 75 MB, deep-filter 36 MB, modnet.onnx 25 MB
# → one-time download from their official releases, sha256-verified (byte-identical to the old committed copies)
python3 -m mograph.fetch_assets || echo "[setup] WARNING: some assets could not be downloaded (offline?) — fallbacks will be used"
python3 -c "import cairo, uharfbuzz, cv2, onnxruntime, mediapipe; print('[setup] OK  cairo', cairo.version, '| cv2', cv2.__version__)"
# v14.6 pbr3d: EffectCraft CLI (photoreal 3D, not in git: 75 MB) + fonts visible to it
chmod +x mograph/bin/deep-filter bin/effectcraft-cli 2>/dev/null || true
mkdir -p ~/.local/share/fonts && cp -n fonts/*.ttf ~/.local/share/fonts/ 2>/dev/null || true
[ -x bin/effectcraft-cli ] && bin/effectcraft-cli info --empty | head -1 || true
