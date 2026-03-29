#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-snn_bench/configs/runs/multistream_cross_asset_example.yaml}"

DATA_DIR="examples/multistream_data"
MIN_ROWS=200

if ! timeout 30s python - "$DATA_DIR" "$MIN_ROWS" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
min_rows = int(sys.argv[2])
assets = ("ES_F", "SPY", "ZN_F", "DXY")

for asset in assets:
    path = data_dir / f"{asset}.csv"
    if not path.exists():
        raise SystemExit(1)
    with path.open("r", encoding="utf-8", newline="") as f:
        row_count = sum(1 for _ in csv.reader(f)) - 1
    if row_count < min_rows:
        raise SystemExit(1)

raise SystemExit(0)
PY
then
  echo "[multistream_ablation] generating demo data in ${DATA_DIR} (rows=5000)"
  timeout 120s python scripts/generate_multistream_demo_data.py --out-dir "$DATA_DIR" --rows 5000
fi

timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type snn --ann-mode lstm
timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type ann --ann-mode lstm
timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type ann --ann-mode tcn
