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

import numpy as np
import yaml

cfg_path = Path(sys.argv[1])
snapshot_dir = Path(sys.argv[2])
min_rows = int(sys.argv[3])
assets = ("SPY", "QQQ", "DIA", "IWM")
streams = []
target_ts = None

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
    # Validate aggregate bar shape to avoid silent all-NaN targets later.
    sample = payload[0]
    if not isinstance(sample, dict) or "t" not in sample or "c" not in sample:
        raise SystemExit(
            f"[multistream_ablation] {path} does not look like aggregate bar rows "
            "(missing 't'/'c'). Re-cache this symbol with minute candles."
        )
    if asset == "SPY":
        ts = np.asarray([row.get("t") for row in payload], dtype=np.float64)
        ts = ts[np.isfinite(ts)]
        if len(ts) < min_rows:
            raise SystemExit(
                f"[multistream_ablation] {path} has insufficient finite timestamps. "
                "Re-cache this symbol with minute candles."
            )
        target_ts = np.sort(ts.astype(np.int64))
    streams.append({"asset": asset, "path": str(path), "max_staleness_ms": 120000})

if target_ts is None:
    raise SystemExit("[multistream_ablation] target timestamps unavailable for SPY.")

# Ensure requested horizons can actually produce labels for the target stream.
max_horizon_s = 300
future = target_ts + int(max_horizon_s * 1000)
idx = np.searchsorted(target_ts, future, side="left")
valid_label_rows = int((idx < len(target_ts)).sum())
if valid_label_rows < 64:
    raise SystemExit(
        "[multistream_ablation] cached SPY bars cannot produce enough forward labels "
        f"for horizon {max_horizon_s}s (valid_rows={valid_label_rows}). "
        "This usually means stale/corrupt timestamps in src/data/SPY.json. "
        "Delete SPY/QQQ/DIA/IWM JSON and re-run: scripts/cache_market_data.sh all 1Min 1 1"
    )

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
