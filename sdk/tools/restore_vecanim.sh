#!/bin/bash
# v27 vecanim: restore after a sandbox reset (~15 s). pycairo only; mediapipe is OPTIONAL
# (needs ~600 MB RAM; the 1 GB sandbox froze running it next to other work).
pip install -q pycairo scipy scikit-image opencv-python-headless shapely 2>&1 | grep -v notice
mkdir -p /home/user/refs && cd /home/user/refs && for r in vectorcraft effectcraft soundcraft; do
  [ -d $r ] || git clone -q --depth 1 --filter=blob:limit=2m https://github.com/storytold/$r.git; done
python3 -c "import cairo, scipy, skimage; print('vecanim deps ok')"
