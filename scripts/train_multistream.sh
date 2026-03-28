#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-snn_bench/configs/runs/multistream_cross_asset_example.yaml}"
MODEL_TYPE="${2:-snn}"
ANN_MODE="${3:-lstm}"

# Generate deterministic demo data if absent
if [[ ! -f examples/multistream_data/ES_F.csv ]]; then
  timeout 120s python scripts/generate_multistream_demo_data.py --out-dir examples/multistream_data --rows 5000
fi

timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type "$MODEL_TYPE" --ann-mode "$ANN_MODE"
