# Operator Guide: Streaming SNN Risk Sentinel

## Purpose
The sentinel emits three real-time controls:
1. **Anomaly score** from reconstruction error.
2. **Regime class** (`0=normal`, `1=elevated`, `2=stress`) from a spiking classifier head.
3. **Risk gate state** (`NORMAL`, `WARNING`, `BLOCK`) for strategy enable/disable.

## Required Input Schema (CSV)
Columns:
- `spread_jumps`
- `depth_collapses`
- `realized_volatility`
- `trade_cancel_intensity`
- `feed_latency_gap`

Optional labels (for supervised calibration/evaluation):
- `regime_label`
- `stress_label`

## Train Sentinel
```bash
timeout 300s python -m snn_bench.scripts.train_sentinel \
  --input data/sentinel_stream.csv \
  --out-dir artifacts/sentinel \
  --epochs 10 --batch-size 128 --seq-len 32 --target-fpr 0.05
```

Outputs:
- `artifacts/sentinel/sentinel_checkpoint.pt`
- `artifacts/sentinel/thresholds.json`
- `artifacts/sentinel/report.md`
- charts in `artifacts/sentinel/plots/`

## Recalibrate Thresholds
```bash
timeout 60s python -m snn_bench.scripts.calibrate_thresholds \
  --scores artifacts/sentinel/anomaly_scores.npy \
  --stress-labels artifacts/sentinel/stress_labels.npy \
  --target-fpr 0.05 \
  --out artifacts/sentinel/thresholds.json
```

## Simulate Gate Impact
```bash
timeout 120s python -m snn_bench.scripts.simulate_gate_impact \
  --scores artifacts/sentinel/anomaly_scores.npy \
  --stress-labels artifacts/sentinel/stress_labels.npy \
  --pnl artifacts/sentinel/pnl.npy \
  --threshold-config artifacts/sentinel/thresholds.json \
  --out-dir artifacts/sentinel_sim
```

## Operating Notes
- Increase `min_warning_steps` / `min_block_steps` to reduce rapid toggling.
- Target false-positive rate is controlled by `--target-fpr` during calibration.
- If labels are absent, thresholds are fit unsupervised on score quantiles.
