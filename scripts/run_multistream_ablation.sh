#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-snn_bench/configs/runs/multistream_cross_asset_example.yaml}"

if [[ ! -f examples/multistream_data/ES_F.csv ]]; then
  timeout 120s python scripts/generate_multistream_demo_data.py --out-dir examples/multistream_data --rows 5000
fi

timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type snn --ann-mode lstm
timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type ann --ann-mode lstm
timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type ann --ann-mode tcn
