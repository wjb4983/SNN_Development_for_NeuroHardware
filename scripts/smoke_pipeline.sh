#!/usr/bin/env bash
set -euo pipefail

TICKER="${1:-AAPL}"
TIMEFRAME="${2:-1D}"

timeout 120s python -m snn_bench.scripts.smoke_pipeline --ticker "$TICKER" --timeframe "$TIMEFRAME"
