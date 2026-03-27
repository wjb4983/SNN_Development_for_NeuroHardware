#!/usr/bin/env bash
set -euo pipefail

# End-to-end helper: dependency install, data fetch, smoke run, unit tests, and training.
# Usage:
#   ./scripts/full_pipeline.sh [ticker] [timeframe] [epochs] [stock_years] [option_years]
# Example:
#   MASSIVE_API_KEY_FILE=/etc/Massive/api-key ./scripts/full_pipeline.sh AAPL 1D 10 5 2

TICKER="${1:-AAPL}"
TIMEFRAME="${2:-1D}"
EPOCHS="${3:-5}"
STOCK_YEARS="${4:-5}"
OPTION_YEARS="${5:-2}"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
OUT_DIR="${OUT_DIR:-artifacts}"

if [[ -z "${MASSIVE_API_KEY:-}" && -z "${MASSIVE_API_KEY_FILE:-}" ]]; then
  echo "[warn] MASSIVE_API_KEY or MASSIVE_API_KEY_FILE is not set."
  echo "       Data fetch and training will fail unless an API key is available in fallback locations."
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[step] Installing project dependencies"
  timeout 300s pip install -e .
fi


echo "[step] Fetching market + options cache"
timeout 900s python -m snn_bench.scripts.cache_market_data \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME" \
  --stock-years "$STOCK_YEARS" \
  --option-years "$OPTION_YEARS"


echo "[step] Running smoke pipeline"
timeout 180s python -m snn_bench.scripts.smoke_pipeline \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME"

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[step] Running unit tests"
  timeout 180s python -m unittest -v tests/test_repo_layout.py tests/test_secrets.py
fi


echo "[step] Training baseline model"
timeout 600s python -m snn_bench.scripts.train \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME" \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --lr 0.001 \
  --out-dir "$OUT_DIR"


echo "[done] Pipeline complete"
echo "       Metrics: $OUT_DIR/train_metrics.json"
