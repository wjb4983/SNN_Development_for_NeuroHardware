#!/usr/bin/env bash
set -euo pipefail

UNIVERSE="${1:-all}"
TIMEFRAME="${2:-1D}"
STOCK_YEARS="${3:-5}"
OPTION_YEARS="${4:-2}"
TICKER="${5:-}"

ARGS=(
  --universe "$UNIVERSE"
  --timeframe "$TIMEFRAME"
  --stock-years "$STOCK_YEARS"
  --option-years "$OPTION_YEARS"
)

if [[ "$UNIVERSE" == "single" ]]; then
  if [[ -z "$TICKER" ]]; then
    echo "Usage for single universe: $0 single <timeframe> <stock_years> <option_years> <ticker>" >&2
    exit 1
  fi
  ARGS+=(--ticker "$TICKER")
fi

timeout 900s python -m snn_bench.scripts.cache_market_data \
  "${ARGS[@]}"
