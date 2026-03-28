# Multi-Stream SNN for Cross-Asset Lead/Lag Prediction

This module implements asynchronous multi-asset event ingestion, causal synchronization, multi-horizon direction targets, SNN/ANN baselines, explainability, and evaluation with transaction-cost-aware PnL proxies.

## What is included

- **Data ingestion** from per-asset timestamped event CSV streams (no bar forcing).
- **Causal synchronization** via `merge_asof(..., direction='backward')` with per-stream max staleness.
- **Feature engineering** with per-asset microstructure signals, event-type one-hot embeddings, and lag timing features.
- **Modeling**:
  - Per-asset spiking encoder (LIF-like cells)
  - Sparse learnable cross-asset coupling matrix (top-k edge mask)
  - Recurrent spiking fusion block
  - Multi-horizon prediction head (1s and 5s by default)
- **Training**:
  - Multi-horizon binary targets
  - Class-imbalance weighted BCE
  - Walk-forward cross-validation
- **Explainability**:
  - Coupling edge importances
  - Temporal attribution snapshots from fused state gradients
- **Baselines**: multistream ANN (`lstm` and `tcn`).
- **Evaluation**:
  - Directional metrics (balanced accuracy, F1, MCC)
  - Net PnL proxy with costs
  - Latency-adjusted inference metrics (mean/p95 and budget hit rate)

## Data format

Each asset stream CSV requires:

- `timestamp` (UTC timestamp)
- `event_type` (e.g. `trade`, `quote`, `book`)
- `price`
- `size`
- `side` (e.g. `buy`/`sell`)

## Commands

Generate synthetic demo streams:

```bash
timeout 120s python scripts/generate_multistream_demo_data.py --out-dir examples/multistream_data --rows 5000
```

Train SNN:

```bash
timeout 900s python -m snn_bench.scripts.train_multistream --config snn_bench/configs/runs/multistream_cross_asset_example.yaml --model-type snn --ann-mode lstm
```

Train ANN baseline (LSTM):

```bash
timeout 900s python -m snn_bench.scripts.train_multistream --config snn_bench/configs/runs/multistream_cross_asset_example.yaml --model-type ann --ann-mode lstm
```

Run ablation suite (SNN vs ANN):

```bash
timeout 1200s ./scripts/run_multistream_ablation.sh snn_bench/configs/runs/multistream_cross_asset_example.yaml
```

## Artifacts

Outputs are written under:

- `artifacts/multistream/snn_lstm/metrics.json`
- `artifacts/multistream/ann_lstm/metrics.json`
- `artifacts/multistream/ann_tcn/metrics.json`

Each `metrics.json` contains config snapshot, fold-level performance, explainability blocks, and latency statistics.
