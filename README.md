# SNN_Development_for_NeuroHardware

Modular benchmark framework for **spiking neural network (SNN)** quant experiments.

## Repository Layout

```text
snn_bench/
  data_connectors/      # Snapshot + backtest cache readers
  feature_pipelines/    # Transform bars to model features
  tasks/                # Dataset/task definitions
  models/               # Dummy baseline + future SNN models
  trainers/             # Training loops
  eval/                 # Evaluation metrics
  configs/              # Pydantic + YAML configs
  scripts/              # Python entrypoints
scripts/                # Shell wrappers
tests/                  # Unit tests
```

## Data Sources

The framework expects the following data locations:

1. **Snapshot cache JSON**
   - `src/data/<SAFE_TICKER>.json`
   - fallback: `../stoptions_analyzer/src/data/<SAFE_TICKER>.json`

2. **Backtest bar store**
   - `src/data/backtest_cache/<SAFE_TICKER>/<TIMEFRAME>/index.json`
   - `src/data/backtest_cache/<SAFE_TICKER>/<TIMEFRAME>/<SAFE_TICKER>_<TIMEFRAME>_<YEAR>.npz`
   - NPZ arrays: `t, o, h, l, c, v, n`

## Quickstart

```bash
cp .env.example .env
timeout 180s pip install -e .
```

Optional Lightning support:

```bash
timeout 180s pip install -e .[lightning]
```

## Smoke Pipeline (One Command)

Run a minimal end-to-end benchmark with a dummy model:

```bash
timeout 120s ./scripts/smoke_pipeline.sh AAPL 1D
```

Equivalent Python command:

```bash
timeout 120s python -m snn_bench.scripts.smoke_pipeline --ticker AAPL --timeframe 1D
```

## Experiment Flow

1. **Ingest** cached data via `SnapshotCacheConnector` and `BacktestBarStoreConnector`.
2. **Build features** using `BasicFeaturePipeline` (returns NumPy arrays `X, y`).
3. **Define task** with `BinaryDirectionDataset`.
4. **Train baseline** with `DummySNN` + `BasicTrainer`.
5. **Evaluate** with `binary_accuracy`.
6. **Swap modules** (new connector/pipeline/model/trainer/eval) without changing orchestration.

## Developer Commands

```bash
make setup
make lint
make unit-test
make smoke-run
```

All commands in the Makefile are non-interactive and use explicit `timeout` bounds.
