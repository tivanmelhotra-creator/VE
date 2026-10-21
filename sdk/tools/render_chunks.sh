#!/usr/bin/env bash
# Resumable chunked render:  bash render_chunks.sh <spec.json> [chunk_seconds]
# Each chunk → out/chunks/cNN.mp4 (skipped if it exists), then concat + audio → out/<name>.mp4
set -e
SPEC=$(realpath "$1"); CH=${2:-4}
DIR=$(dirname "$SPEC"); NAME=$(basename "$SPEC" .json)
export PYTHONPATH=$(realpath "$(dirname "$0")/..")
cd "$DIR"; mkdir -p out/chunks
DUR=$(python3 -m mograph.v14 words "$SPEC" 2>/dev/null | grep duration | awk '{print $2}' | tr -d s)
N=$(python3 -c "import math;print(math.ceil($DUR/$CH))")
for i in $(seq 0 $((N-1))); do
  f=out/chunks/c$(printf %02d $i).mp4
  [ -s "$f" ] && continue
  s=$(python3 -c "print($i*$CH)"); e=$(python3 -c "print(min($DUR,($i+1)*$CH))")
  tmp=out/chunks/part_$(printf %02d $i).mp4
  python3 -m mograph.v14 render "$SPEC" --start $s --end $e --no-audio --out "$tmp" 2>&1 | grep -v W0000 | tail -1
  [ -s "$tmp" ] || { echo "chunk $i failed"; exit 1; }
  mv "$tmp" "$f"
done
ls out/chunks/c*.mp4 | sed "s/^/file '/; s/$/'/" | sed "s|file 'out/chunks/|file '|" > out/chunks/list.txt
ffmpeg -y -v error -f concat -safe 0 -i out/chunks/list.txt -c copy out/${NAME}_silent.mp4
python3 -c "
from mograph.v14.compositor import Renderer; from mograph.v14.audio import build_audio
build_audio(Renderer('$SPEC'), 'out/${NAME}.wav')" 2>&1 | grep -v W0000 | tail -1
ffmpeg -y -v error -i out/${NAME}_silent.mp4 -i out/${NAME}.wav -c:v copy -c:a aac -b:a 192k -shortest out/${NAME}.mp4
echo "DONE out/${NAME}.mp4"
