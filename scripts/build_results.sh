#!/bin/sh
# Regenerate every committed result bundle from the checked-in datasets.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY=python3
fi
cd "$ROOT"
mkdir -p results web/data

"$PY" -m minojev.cli evaluate --checkpoint runs/synth/checkpoint \
  --input data/test.jsonl --predictions results/test-predictions.jsonl \
  --device cpu > results/synth-metrics.json

"$PY" -m minojev.cli evaluate --checkpoint runs/maze/checkpoint \
  --input data/maze-test.jsonl --predictions results/maze-predictions.jsonl \
  --device cpu > results/maze-metrics.json

"$PY" -m minojev.cli demo --checkpoint runs/synth/checkpoint \
  --output web/data/demo.json --count 12 --seed 17 --device cpu

"$PY" -m minojev.cli maze-rollout --checkpoint runs/maze/checkpoint \
  --output web/data/maze.json --count 6 --seed 23 --device cpu

"$PY" scripts/maze_preview.py --input web/data/maze.json --output web/data/maze-preview.svg

echo "results and replay bundles regenerated"
