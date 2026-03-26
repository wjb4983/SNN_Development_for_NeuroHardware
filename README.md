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

## MASSIVE API key setup

The scripts auto-load the API key in this order:

1. `MASSIVE_API_KEY` env var
2. `MASSIVE_API_KEY_FILE` env var
3. `C:\Users\wbott\.stoptions_analyzer\api_key.txt`
4. `~/.stoptions_analyzer/api_key.txt`

Windows example:

```powershell
$env:MASSIVE_API_KEY_FILE="C:\Users\wbott\.stoptions_analyzer\api_key.txt"
```

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

```bash
timeout 120s ./scripts/smoke_pipeline.sh AAPL 1D
```

## Start Training Models

One-command training run:

```bash
timeout 120s ./scripts/train.sh AAPL 1D 5
```

Direct Python command:

```bash
timeout 120s python -m snn_bench.scripts.train --ticker AAPL --timeframe 1D --epochs 5 --batch-size 32 --lr 0.001 --out-dir artifacts
```

Artifacts written:
- `artifacts/<TICKER>_<TIMEFRAME>_dummy.pt`
- `artifacts/train_metrics.json`

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
make train-run
```

All commands are non-interactive and use explicit `timeout` bounds.
