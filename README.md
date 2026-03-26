# SNN_Development_for_NeuroHardware

Modular benchmark framework for **spiking neural network (SNN)** quant experiments.

## Repository Layout

```text
snn_bench/
  data_connectors/      # Snapshot + backtest cache readers + Massive client
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

## Data Sources and Local Cache Contract

The framework writes/reads these local cache files:

1. **Stock snapshot cache JSON**
   - `src/data/<SAFE_TICKER>.json`

2. **Backtest bar store**
   - `src/data/backtest_cache/<SAFE_TICKER>/<TIMEFRAME>/index.json`
   - `src/data/backtest_cache/<SAFE_TICKER>/<TIMEFRAME>/<SAFE_TICKER>_<TIMEFRAME>_<YEAR>.npz`
   - NPZ arrays: `t, o, h, l, c, v, n`

3. **Option cache JSON**
   - `src/data/options/<SAFE_TICKER>_options.json`

## MASSIVE API key setup

The scripts auto-load the API key in this order:

1. `MASSIVE_API_KEY` env var
2. `MASSIVE_API_KEY_FILE` env var
3. `/etc/Massive/api-key` (preferred)
4. `~/.stoptions_analyzer/api_key.txt` (legacy fallback)

```bash
export MASSIVE_API_KEY_FILE=/etc/Massive/api-key
```

## Pull and Cache Market Data (5y stock + 2y options)

One-command cache pull:

```bash
timeout 900s ./scripts/cache_market_data.sh AAPL 1D 5 2
```

Equivalent Python command:

```bash
timeout 900s python -m snn_bench.scripts.cache_market_data --ticker AAPL --timeframe 1D --stock-years 5 --option-years 2
```

What it does:
- Pulls **5 years** of daily stock bars from Massive and writes `src/data/<SAFE_TICKER>.json`.
- Splits bars by year and writes yearly NPZ files + `index.json` under `src/data/backtest_cache/...`.
- Pulls option snapshots and keeps contracts in the requested **2-year window**, saved to `src/data/options/<SAFE_TICKER>_options.json`.

## Quickstart

```bash
cp .env.example .env
timeout 180s pip install -e .
```

## Smoke Pipeline

```bash
timeout 120s ./scripts/smoke_pipeline.sh AAPL 1D
```

## Start Training Models

```bash
timeout 120s ./scripts/train.sh AAPL 1D 5
```

Direct command:

```bash
timeout 120s python -m snn_bench.scripts.train --ticker AAPL --timeframe 1D --epochs 5 --batch-size 32 --lr 0.001 --out-dir artifacts
```

## Run with Docker

Build:

```bash
timeout 1800s docker build -t snn-bench:latest .
```

Cache pull in container:

```bash
timeout 1200s docker run --rm \
  --entrypoint python \
  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
  -v "$PWD/src/data:/app/src/data" \
  -v "/etc/Massive/api-key:/etc/Massive/api-key:ro" \
  snn-bench:latest -m snn_bench.scripts.cache_market_data --ticker AAPL --timeframe 1D --stock-years 5 --option-years 2
```

Train in container:

```bash
timeout 600s docker run --rm \
  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
  -v "$PWD/src/data:/app/src/data" \
  -v "$PWD/artifacts:/app/artifacts" \
  -v "/etc/Massive/api-key:/etc/Massive/api-key:ro" \
  snn-bench:latest --ticker AAPL --timeframe 1D --epochs 5 --batch-size 32 --lr 0.001 --out-dir artifacts
```

## Developer Commands

```bash
make setup
make lint
make unit-test
make cache-data
make smoke-run
make train-run
make docker-build
make docker-cache
make docker-smoke
make docker-train
```
