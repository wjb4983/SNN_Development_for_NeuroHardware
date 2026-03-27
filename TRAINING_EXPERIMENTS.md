# Training Experiments (Config-Driven)

This project now supports running **multiple training runs from one YAML manifest**.

## 1) Edit one file

Start with:

- `snn_bench/configs/experiments/aapl_model_sweep.yaml`

Put shared values under `defaults`, and each experiment under `runs`.

## 2) Run all experiments with one command

```bash
timeout 1200s ./scripts/run_experiments.sh
```

Or point to another manifest:

```bash
timeout 1200s ./scripts/run_experiments.sh snn_bench/configs/experiments/aapl_model_sweep.yaml
```

## 3) Common toggles

```bash
OUT_DIR=artifacts/exp_jan MAX_YEARS=1 timeout 1200s ./scripts/run_experiments.sh
```

```bash
STOP_ON_ERROR=1 timeout 1200s ./scripts/run_experiments.sh
```

## 4) Direct Python command (same behavior)

```bash
timeout 1200s python -m snn_bench.scripts.run_experiments \
  --manifest snn_bench/configs/experiments/aapl_model_sweep.yaml \
  --out-dir artifacts/experiments \
  --max-years 0
```

## Outputs

Each run still writes a per-run folder in `OUT_DIR`. The sweep also writes:

- `OUT_DIR/experiment_summary.json`

This summary lists total/completed/failed runs and run IDs.
