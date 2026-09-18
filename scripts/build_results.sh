#!/bin/sh
# Regenerate every committed result bundle from the checked-in datasets.
# Expects trained checkpoints under runs/synth/checkpoint and runs/maze/checkpoint.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY=python3
fi
cd "$ROOT"
mkdir -p results web/data

# 1. Fit probability temperatures on the dev splits (calibrated confidence).
"$PY" -m minojev.cli calibrate --checkpoint runs/synth/checkpoint \
  --input data/dev.jsonl --output runs/synth-calibrated --target gold --device cpu \
  > results/calibration-synth.json
"$PY" -m minojev.cli calibrate --checkpoint runs/maze/checkpoint \
  --input data/maze-dev.jsonl --output runs/maze-calibrated --target gold --device cpu \
  > results/calibration-maze.json

# 2. Evaluate on held-out test splits.
"$PY" -m minojev.cli evaluate --checkpoint runs/synth-calibrated \
  --input data/test.jsonl --predictions results/test-predictions.jsonl \
  --device cpu > results/synth-metrics.json

"$PY" -m minojev.cli evaluate --checkpoint runs/maze-calibrated \
  --input data/maze-test.jsonl --predictions results/maze-predictions.jsonl \
  --device cpu > results/maze-metrics.json

# 3. Latency and throughput for both serving modes.
"$PY" -m minojev.cli bench --checkpoint runs/synth-calibrated --input data/test.jsonl \
  --output results/bench-synth.json --device cpu
"$PY" -m minojev.cli bench --checkpoint runs/maze-calibrated --input data/maze-test.jsonl \
  --output results/bench-maze.json --device cpu

# 4. Replay bundles for the static demos.
"$PY" -m minojev.cli demo --checkpoint runs/synth-calibrated \
  --output web/data/demo.json --count 12 --seed 17 --device cpu

"$PY" -m minojev.cli maze-rollout --checkpoint runs/maze-calibrated \
  --output web/data/maze.json --count 6 --seed 23 --device cpu

"$PY" scripts/maze_preview.py --input web/data/maze.json --output web/data/maze-preview.svg

# 5. Example scores for the quickstart.
"$PY" -m minojev.cli score --checkpoint runs/synth-calibrated \
  --input examples/decisions.jsonl --output results/example-scores.jsonl --mode reuse --device cpu

echo "results and replay bundles regenerated"
