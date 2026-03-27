#!/usr/bin/env bash
set -euo pipefail

TICKER="${1:-AAPL}"
TIMEFRAME="${2:-1D}"
STOCK_YEARS="${3:-5}"
OPTION_YEARS="${4:-2}"

timeout 900s python -m snn_bench.scripts.cache_market_data \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME" \
  --stock-years "$STOCK_YEARS" \
  --option-years "$OPTION_YEARS"
