#!/bin/sh
# Memory-safe post-training runner.
#
# Usage:
#   MODE=head|head|lora|last-layers BACKBONE=Qwen/Qwen3-1.7B \
#   TRAIN=data/general-train.jsonl DEV=data/general-dev.jsonl \
#   OUT=runs/general-head STEPS=3000 scripts/run_posttrain.sh
#
# Safety rules learned the hard way:
#   - never disable the MPS watermark (0.0 lets unified memory eat the machine)
#   - keep a hard budget; the monitor aborts before the OS is at risk
#   - chunk every forward pass
#   - watch progress with: cat "$OUT/status.json"
set -eu
cd "$(dirname "$0")/.."

BACKBONE="${BACKBONE:-Qwen/Qwen3-1.7B}"
MODE="${MODE:-head}"
TRAIN="${TRAIN:-data/general-train.jsonl}"
DEV="${DEV:-data/general-dev.jsonl}"
OUT="${OUT:-runs/general-$MODE}"
STEPS="${STEPS:-3000}"
BATCH="${BATCH:-4}"
CACHE_BATCH="${CACHE_BATCH:-2}"
EVAL_EVERY="${EVAL_EVERY:-250}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-12}"
LOG_EVERY="${LOG_EVERY:-10}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-float32}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-1.4}"

exec .venv/bin/python -u -m minojev.cli posttrain \
  --backbone "$BACKBONE" --mode "$MODE" \
  --train "$TRAIN" --dev "$DEV" --output-dir "$OUT" \
  --steps "$STEPS" --batch-requests "$BATCH" --cache-batch-requests "$CACHE_BATCH" \
  --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
  --max-memory-gb "$MAX_MEMORY_GB" --device mps \
  --inference-dtype "$INFERENCE_DTYPE" \
  ${GRADIENT_CHECKPOINTING:+--gradient-checkpointing}
