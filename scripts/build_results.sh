#!/bin/sh
# Regenerate every committed result bundle from the checked-in datasets.
#
# Checkpoints expected (train with the commands in the README):
#   runs/synth/checkpoint, runs/maze/checkpoint          - from-scratch tiny models
#   runs/qwen-lora/checkpoint or runs/qwen-head/checkpoint - post-trained Qwen3-0.6B
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY=python3
fi
cd "$ROOT"
mkdir -p results web/data

if [ -d runs/qwen-lora/checkpoint ]; then
  DOMAIN_CKPT=runs/qwen-lora/checkpoint
else
  DOMAIN_CKPT=runs/qwen-head/checkpoint
fi
GENERAL_CKPT=runs/general-head-goldcal

# 1. Fit probability temperatures on the dev splits (calibrated confidence).
"$PY" -m minojev.cli calibrate --checkpoint runs/synth/checkpoint \
  --input data/dev.jsonl --output runs/synth-calibrated --target gold --device cpu \
  > results/calibration-synth.json
"$PY" -m minojev.cli calibrate --checkpoint runs/maze/checkpoint \
  --input data/maze-dev.jsonl --output runs/maze-calibrated --target gold --device cpu \
  > results/calibration-maze.json

# 2. Evaluate held-out test splits.
"$PY" -m minojev.cli evaluate --checkpoint runs/synth-calibrated \
  --input data/test.jsonl --predictions results/test-predictions.jsonl \
  --device cpu > results/synth-metrics.json

"$PY" -m minojev.cli evaluate --checkpoint runs/maze-calibrated \
  --input data/maze-test.jsonl --predictions results/maze-predictions.jsonl \
  --device cpu > results/maze-metrics.json

# 3. Domain accuracy for the post-trained model and the toy baseline.
"$PY" -m minojev.cli evaluate --checkpoint "$DOMAIN_CKPT" \
  --input data/domains-test.jsonl --predictions results/domains-predictions.jsonl \
  --device cpu > results/domains-metrics.json

# 4. Latency and throughput.
"$PY" -m minojev.cli bench --checkpoint runs/synth-calibrated --input data/test.jsonl \
  --output results/bench-synth.json --device cpu
"$PY" -m minojev.cli bench --checkpoint "$DOMAIN_CKPT" --input data/domains-test.jsonl \
  --output results/bench-domains.json --device cpu

# 5. Replay bundles for the static demos.
"$PY" -m minojev.cli demo --checkpoint runs/synth-calibrated \
  --output web/data/demo.json --count 12 --seed 17 --device cpu

"$PY" -m minojev.cli demo --checkpoint "$DOMAIN_CKPT" \
  --input data/domains-test.jsonl --limit 25 \
  --output web/data/domains.json --device cpu

"$PY" -m minojev.cli maze-rollout --checkpoint runs/maze-calibrated \
  --output web/data/maze.json --count 6 --seed 23 --device cpu

"$PY" scripts/maze_preview.py --input web/data/maze.json --output web/data/maze-preview.svg

# 6. Example scores for the quickstart.
"$PY" -m minojev.cli score --checkpoint runs/synth-calibrated \
  --input examples/decisions.jsonl --output results/example-scores.jsonl --mode reuse --device cpu

# 7. General model: balanced evaluation, head-to-head comparison, benchmark bundle.
if [ -d "$GENERAL_CKPT" ]; then
  "$PY" -m minojev.cli evaluate --checkpoint "$GENERAL_CKPT" \
    --input data/general-test.jsonl --limit 200 --balanced --batch-requests 8 \
    --predictions results/general-predictions.jsonl --device mps > results/general-metrics.json

  "$PY" -m minojev.cli compare --checkpoint "$GENERAL_CKPT" \
    --input data/general-test.jsonl --limit 120 --chat-template \
    --output results/compare-general.json --markdown results/compare-general.md --device mps

  "$PY" -m minojev.cli compare --checkpoint "$GENERAL_CKPT" \
    --input data/domains-test.jsonl --limit 40 --chat-template \
    --output results/compare-domains.json --markdown results/compare-domains.md --device mps

  "$PY" scripts/build_benchmark_bundle.py
  "$PY" scripts/export_hf_general.py
fi

# 8. Dataset and model export bundles for the Hub.
"$PY" scripts/export_hf_dataset.py

echo "results and replay bundles regenerated"
