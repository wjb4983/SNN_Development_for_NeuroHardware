#!/usr/bin/env bash
set -euo pipefail

TICKER="${1:-AAPL}"
TIMEFRAME="${2:-1D}"
EPOCHS="${3:-5}"
MAX_YEARS="${MAX_YEARS:-0}"

timeout 120s python -m snn_bench.scripts.train \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME" \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --lr 0.001 \
  --out-dir artifacts \
  --max-years "$MAX_YEARS"
