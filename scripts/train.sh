#!/usr/bin/env bash
set -euo pipefail

TICKER="${1:-AAPL}"
TIMEFRAME="${2:-1D}"
EPOCHS="${3:-5}"
MAX_YEARS="${MAX_YEARS:-0}"
MODEL="${MODEL:-mlp}"
SMOKE="${SMOKE:-0}"

SMOKE_FLAGS=()
if [[ "$SMOKE" == "1" ]]; then
  SMOKE_FLAGS+=(--smoke --smoke-sample-size "${SMOKE_SAMPLE_SIZE:-256}" --smoke-epochs "${SMOKE_EPOCHS:-1}")
fi

timeout 120s python -m snn_bench.scripts.train \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME" \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --lr 0.001 \
  --model "$MODEL" \
  --out-dir artifacts \
  --max-years "$MAX_YEARS" \
  "${SMOKE_FLAGS[@]}"
