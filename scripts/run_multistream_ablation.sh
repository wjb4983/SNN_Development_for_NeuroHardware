#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-snn_bench/configs/runs/multistream_cross_asset_example.yaml}"

DATA_DIR="examples/multistream_data"
MIN_ROWS=5000

echo "[multistream_ablation] Note: this ablation consumes event CSV streams from config paths (default: ${DATA_DIR}), not cache_market_data snapshot JSON/NPZ artifacts."

if ! timeout 60s python - "$CFG" "$DATA_DIR" "$MIN_ROWS" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

cfg_path = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
min_rows = int(sys.argv[3])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
dataset = cfg["dataset"]
assets = [s["asset"] for s in dataset["streams"]]
horizons_s = tuple(int(x) for x in cfg.get("train", {}).get("horizons_s", [1, 5]))
target_asset = dataset["target_asset"]
target_path = None

for asset in assets:
    path = next((Path(s["path"]) for s in dataset["streams"] if s["asset"] == asset), data_dir / f"{asset}.csv")
    if not path.exists():
        raise SystemExit(1)
    with path.open("r", encoding="utf-8", newline="") as f:
        row_count = sum(1 for _ in csv.reader(f)) - 1
    if row_count < min_rows:
        raise SystemExit(1)
    if asset == target_asset:
        target_path = path

if target_path is None:
    raise SystemExit(1)

# Validate that target horizons can produce non-NaN labels.
target = pd.read_csv(target_path)
if "timestamp" not in target.columns:
    raise SystemExit(1)
t_ns = pd.to_datetime(target["timestamp"], utc=True, format="mixed").astype("int64").to_numpy()
if len(t_ns) < 3:
    raise SystemExit(1)
max_hz = max(horizons_s) if horizons_s else 0
future = t_ns + int(max_hz * 1_000_000_000)
idx = np.searchsorted(t_ns, future, side="left")
valid_rows = int((idx < len(t_ns)).sum())
if valid_rows < 64:
    raise SystemExit(1)

raise SystemExit(0)
PY
then
  echo "[multistream_ablation] generating demo event CSV data in ${DATA_DIR} (rows=5000)"
  timeout 120s python scripts/generate_multistream_demo_data.py --out-dir "$DATA_DIR" --rows 5000
fi

timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type snn --ann-mode lstm
timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type ann --ann-mode lstm
timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type ann --ann-mode tcn
