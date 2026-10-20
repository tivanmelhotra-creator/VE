#!/usr/bin/env bash
# One-shot environment restore for mograph v14 (run after ANY sandbox reset):
#   bash /home/user/webapp/sdk/tools/setup_env.sh
set -e
cd "$(dirname "$0")/.."
pip install noisereduce numba -q pycairo uharfbuzz fonttools freetype-py mediapipe onnxruntime opencv-python numpy scipy pillow 2>&1 | tail -1 || true
(sudo apt-get install -y -q libegl1 libgles2 >/dev/null 2>&1) || true
M=mograph/models/modnet.onnx
if [ ! -s "$M" ]; then
  echo "[setup] downloading MODNet portrait matting model (25 MB, Apache-2.0)…"
  curl -sL -o "$M" "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx" || true
fi
python3 -c "import cairo, uharfbuzz, cv2, onnxruntime, mediapipe; print('[setup] OK  cairo', cairo.version, '| cv2', cv2.__version__)"
# v14.6 pbr3d: EffectCraft CLI (photoreal 3D, not in git: 75 MB) + fonts visible to it
chmod +x mograph/bin/deep-filter 2>/dev/null || true
[ -s bin/effectcraft-cli ] || python3 -c "import sys; sys.path.insert(0,'.'); from mograph import pbr3d; print(pbr3d.ec_bin())"
chmod +x bin/effectcraft-cli
mkdir -p ~/.local/share/fonts && cp -n fonts/*.ttf ~/.local/share/fonts/ 2>/dev/null || true
bin/effectcraft-cli info --empty | head -1
