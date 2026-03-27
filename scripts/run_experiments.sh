#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-snn_bench/configs/experiments/aapl_model_sweep.yaml}"
OUT_DIR="${OUT_DIR:-artifacts/experiments}"
MAX_YEARS="${MAX_YEARS:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

EXTRA_FLAGS=()
if [[ "$STOP_ON_ERROR" == "1" ]]; then
  EXTRA_FLAGS+=(--stop-on-error)
fi

timeout 1200s python -m snn_bench.scripts.run_experiments \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" \
  --max-years "$MAX_YEARS" \
  "${EXTRA_FLAGS[@]}"
