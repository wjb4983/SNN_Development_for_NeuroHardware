#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-}"
SNAPSHOT_DIR="${2:-src/data}"
MIN_ROWS=5000
TMP_CFG=""

if [[ -z "$CFG" ]]; then
  TMP_CFG="$(mktemp /tmp/multistream_ablation_cached.XXXXXX.yaml)"
  CFG="$TMP_CFG"
  timeout 60s python - "$CFG" "$SNAPSHOT_DIR" "$MIN_ROWS" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
snapshot_dir = Path(sys.argv[2])
min_rows = int(sys.argv[3])
assets = ("SPY", "QQQ", "DIA", "IWM")
streams = []

for asset in assets:
    path = snapshot_dir / f"{asset}.json"
    if not path.exists():
        raise SystemExit(
            f"[multistream_ablation] missing {path}; run: scripts/cache_market_data.sh all 1Min 1 1"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"[multistream_ablation] expected list payload in {path}")
    if len(payload) < min_rows:
        raise SystemExit(
            f"[multistream_ablation] {path} has only {len(payload)} rows (< {min_rows}). "
            "Re-cache with minute candles, e.g. scripts/cache_market_data.sh all 1Min 1 1"
        )
    streams.append({"asset": asset, "path": str(path), "max_staleness_ms": 120000})

cfg = {
    "dataset": {
        "target_asset": "SPY",
        "streams": streams,
        "feature": {"event_type_vocab": ["trade"]},
    },
    "model": {
        "hidden_dim": 64,
        "encoder_dim": 32,
        "fusion_dim": 64,
        "recurrent_decay": 0.9,
        "dropout": 0.1,
        "top_k_edges": 8,
    },
    "train": {
        "seed": 7,
        "batch_size": 128,
        "epochs": 5,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "horizons_s": [60, 300],
        "walk_forward_folds": 4,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "transaction_cost_bps": 1.2,
    },
    "output_dir": "artifacts/multistream",
}
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[multistream_ablation] generated cached-data config: {cfg_path}")
PY
fi

cleanup() {
  if [[ -n "$TMP_CFG" && -f "$TMP_CFG" ]]; then
    rm -f "$TMP_CFG"
  fi
}
trap cleanup EXIT

timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type snn --ann-mode lstm
timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type ann --ann-mode lstm
timeout 900s python -m snn_bench.scripts.train_multistream --config "$CFG" --model-type ann --ann-mode tcn
