#!/usr/bin/env bash
set -euo pipefail

# Single-run training wrapper with clear defaults and readable runtime logs.
#
# Usage:
#   ./scripts/train.sh [ticker] [timeframe] [epochs]
#
# Common overrides:
#   MODEL=snntorch_lif TASK_NAME=direction_30m_distribution SMOKE=1 timeout 900s ./scripts/train.sh AAPL 1D 2
#   SPLIT_MODE=walk_forward WALK_FORWARD_RATIO=0.8 timeout 900s ./scripts/train.sh

TICKER="${1:-AAPL}"
TIMEFRAME="${2:-1D}"
EPOCHS="${3:-5}"

MODEL="${MODEL:-mlp}"
TASK_NAME="${TASK_NAME:-direction_5m_distribution}"
OUT_DIR="${OUT_DIR:-artifacts}"
MAX_YEARS="${MAX_YEARS:-0}"
RUN_NAME="${RUN_NAME:-manual_run}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
SPLIT_MODE="${SPLIT_MODE:-random}"
WALK_FORWARD_RATIO="${WALK_FORWARD_RATIO:-0.8}"
TIMEOUT_S="${TIMEOUT_S:-900s}"

SMOKE="${SMOKE:-0}"
SMOKE_SAMPLE_SIZE="${SMOKE_SAMPLE_SIZE:-256}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"

EXTRA_FLAGS=()
if [[ "$SMOKE" == "1" ]]; then
  EXTRA_FLAGS+=(--smoke --smoke-sample-size "$SMOKE_SAMPLE_SIZE" --smoke-epochs "$SMOKE_EPOCHS")
fi

mkdir -p "$OUT_DIR"

echo "[train] ticker=$TICKER timeframe=$TIMEFRAME task=$TASK_NAME model=$MODEL"
echo "[train] epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR split_mode=$SPLIT_MODE smoke=$SMOKE"

if [[ "$SPLIT_MODE" == "walk_forward" ]]; then
  EXTRA_FLAGS+=(--walk-forward-ratio "$WALK_FORWARD_RATIO")
fi

timeout "$TIMEOUT_S" python -m snn_bench.scripts.train \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --model "$MODEL" \
  --task-name "$TASK_NAME" \
  --split-mode "$SPLIT_MODE" \
  --out-dir "$OUT_DIR" \
  --run-name "$RUN_NAME" \
  --seed "$SEED" \
  --max-years "$MAX_YEARS" \
  "${EXTRA_FLAGS[@]}"

echo "[done] run_name=$RUN_NAME out_dir=$OUT_DIR"
