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
timeout 120s python -m snn_bench.scripts.train --ticker AAPL --timeframe 1D --epochs 5 --batch-size 32 --lr 0.001 --out-dir artifacts --max-years 0
```

## End-to-End Script (fetch → smoke → tests → train)

Use the bundled shell script to run the full flow with explicit timeouts:

```bash
MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
  timeout 2400s ./scripts/full_pipeline.sh AAPL 1D 5 5 2
```

Flags are positional: `ticker timeframe epochs stock_years option_years`.

Optional environment toggles:
- `INSTALL_DEPS=1` to run `pip install -e .` before pipeline execution (local mode)
- `RUN_TESTS=0` to skip unit tests
- `OUT_DIR=artifacts_custom` to change training output path
- `MAX_YEARS=0` train on all cached years (set `1` to mimic old single-year quick train)
- `USE_DOCKER=1` to force all python steps to run in `snn-bench:latest`
- `DOCKER_IMAGE=custom:tag` to override image name in docker mode

If local Python is missing core packages (for example `numpy`), the script auto-falls back to:
1) Conda env `snnbench` (if available), then
2) Docker image `snn-bench:latest` (if available).

## Run with Docker (prebuilt Conda env inside image)

The image builds a full **Conda environment** (`snnbench`) with all required Python packages (including `numpy`), then executes commands inside that environment automatically.

Build once:

```bash
timeout 2400s docker build -t snn-bench:latest .
```

Cache pull in container:

```bash
timeout 1200s docker run --rm \
  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
  -v "$PWD/src/data:/app/src/data" \
  -v "/etc/Massive/api-key:/etc/Massive/api-key:ro" \
  snn-bench:latest python -m snn_bench.scripts.cache_market_data --ticker AAPL --timeframe 1D --stock-years 5 --option-years 2
```

Smoke run in container:

```bash
timeout 300s docker run --rm \
  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
  -v "$PWD/src/data:/app/src/data" \
  -v "/etc/Massive/api-key:/etc/Massive/api-key:ro" \
  snn-bench:latest python -m snn_bench.scripts.smoke_pipeline --ticker AAPL --timeframe 1D
```

Train in container:

```bash
timeout 600s docker run --rm \
  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
  -v "$PWD/src/data:/app/src/data" \
  -v "$PWD/artifacts:/app/artifacts" \
  -v "/etc/Massive/api-key:/etc/Massive/api-key:ro" \
  snn-bench:latest python -m snn_bench.scripts.train --ticker AAPL --timeframe 1D --epochs 5 --batch-size 32 --lr 0.001 --out-dir artifacts
```


### Docker runtime fix for `GLIBCXX_3.4.31` / `optree` errors

If you previously built an older image and hit:
`ImportError: ... libstdc++.so.6: version GLIBCXX_3.4.31 not found`,
rebuild from scratch to refresh runtime libs:

```bash
timeout 2400s docker build --no-cache -t snn-bench:latest .
```

This Dockerfile now provisions `libstdcxx-ng`/`libgcc-ng` in Conda and exports `LD_LIBRARY_PATH` to the Conda lib directory.


## Model Zoo (shared train/eval API)

Supported models via `--model`:
- `logreg` (logistic regression)
- `gbm` (`xgboost` if available, otherwise sklearn gradient boosting)
- `mlp`
- `snntorch_lif`
- `norse_lsnn`
- `spikingjelly_lif`

Config-driven run examples:

```bash
timeout 120s python -m snn_bench.scripts.train --config snn_bench/configs/runs/spy_smoke.yaml --out-dir artifacts --max-years 1
```

Fast smoke mode (small sample + few epochs):

```bash
timeout 120s python -m snn_bench.scripts.train --ticker SPY --timeframe 1D --model mlp --smoke --smoke-sample-size 256 --smoke-epochs 1 --max-years 1 --out-dir artifacts
```

Each run writes a unique run directory containing:
- model checkpoint
- prediction artifact JSON
- `train_metrics.json`


## Config-Driven Multi-Experiment Training

If you prefer editing one config and rerunning one command, use the experiment manifest flow:

1. Edit `snn_bench/configs/experiments/aapl_model_sweep.yaml`
2. Run the same command each time:

```bash
timeout 1200s ./scripts/run_experiments.sh
```

You can also point to a custom manifest:

```bash
timeout 1200s ./scripts/run_experiments.sh snn_bench/configs/experiments/aapl_model_sweep.yaml
```

Useful environment overrides:
- `OUT_DIR=artifacts/my_experiments`
- `MAX_YEARS=1`
- `STOP_ON_ERROR=1`
- `USE_DOCKER=1` (force docker mode)

If local `python` is missing dependencies like `PyYAML`, `run_experiments.sh` auto-falls back to conda env `snnbench` or Docker image `snn-bench:latest` when available.

See `TRAINING_EXPERIMENTS.md` for the full workflow.

## Developer Commands

```bash
make setup
make lint
make unit-test
make cache-data
make smoke-run
make train-run
make experiment-run
make docker-build
make docker-build-clean
make docker-cache
make docker-smoke
make docker-train
```
