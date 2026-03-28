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

Default supervised target/task:
- `task.name`: `next_bar_direction`
- `task.horizon`: `1_bar` (predict next bar)
- `task.label_type`: `binary`
- `task.classes`: `["down_or_flat", "up"]`
- Label semantics: `1 if next-bar close-to-close return > 0, else 0`

To switch tasks, pass a YAML config with a `task` section (for example `task.name`, `task.horizon`, `task.label_type`, `task.classes`, `task.label_semantics`) via `--config`.

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
- `report.md` (auto-generated post-train markdown summary)
- `plots/` PNG visualizations

The post-train report generator creates:
- `plots/confusion_matrix.png`
- `plots/roc_curve.png` (binary tasks with both classes present)
- `plots/pr_curve.png` (binary tasks with both classes present)
- `plots/calibration_plot.png`
- `plots/probability_histogram.png`
- `plots/equity_curve.png` when trading metrics are available in task evaluation config
- `plots/next_bar_prediction_vs_outcome.png` when prediction artifacts include reference close and next-close prices (auto-included for `next_bar_direction` training runs)

For next-bar tasks, `report.md` now also includes a compact efficacy block with directional hit-rate, signed confidence, and realized-move magnitude so test/eval runs visibly demonstrate prediction quality.

`train_metrics.json` now includes the full `task` metadata block (including task `evaluation` config), and prediction artifacts include `target_summary` with horizon + label semantics.  
To change this metadata, define `task` in your config file and run `python -m snn_bench.scripts.train --config ...`.


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

When Docker is used, `MASSIVE_API_KEY_FILE` is now mounted into the container automatically if the file exists on the host.

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

## LOB Alpha (Production-style SNN + ANN Baseline)

This repository now includes a dedicated event-driven limit order book alpha pipeline under `src/`:

```text
src/
  data/      # FI-2010 + generic CSV loaders, sequence dataset
  features/  # LOB feature engineering + spike encoders (rate/TTFS)
  models/    # Conv1D+recurrent SNN + LSTM ANN baseline
  train/     # Surrogate gradient trainer with TBPTT/clip/early stop
  eval/      # Purged walk-forward, metrics, cost-aware backtest
configs/
  lob_alpha.yaml
scripts/
  lob_cli.py
```

### Minimal setup

```bash
timeout 300s python -m pip install -r requirements.txt
```

### Exact commands

Train SNN (default):

```bash
timeout 1800s python scripts/lob_cli.py train --config configs/lob_alpha.yaml
```

Evaluate checkpoint:

```bash
timeout 600s python scripts/lob_cli.py evaluate \
  --config configs/lob_alpha.yaml \
  --checkpoint artifacts/lob_alpha/snn_500ms/checkpoint.pt
```

Backtest saved predictions:

```bash
timeout 300s python scripts/lob_cli.py backtest \
  --config configs/lob_alpha.yaml \
  --predictions artifacts/lob_alpha/evaluation.json
```

Switch baseline to ANN LSTM by editing `model.name: lstm` in `configs/lob_alpha.yaml`.

## Event-driven SNN execution policy approximator

This repository now includes an end-to-end pipeline for child-order execution decisions with action space:
`{join_bid, join_ask, improve, cross, cancel, hold}` and discrete size buckets.

### Data schema (event logs)
Expected columns in CSV/JSONL:
- `ts_ns` (int nanoseconds)
- `event_type` in `{market_book, market_trade, market_cancel, own_placement, own_queue, own_fill}`
- `bid_prices`, `bid_sizes`, `ask_prices`, `ask_sizes` (arrays or CSV strings)
- optional: `side`, `price`, `size`, `level`, `own_order_id`, `queue_position`, `fill_size`, `action`, `size_bucket`

### Features
The preprocessor derives causal event features including:
- top-k book levels (price + size)
- spread and depth imbalance
- short-term realized volatility from rolling mid returns
- trade intensity (trades per second in recent horizon)
- queue position
- time since last fill

### Training stages
- **Stage A (behavior cloning):** supervised imitation from historical labels.
- **Stage B (optional RL):** actor-critic fine-tuning over replay batches.
- Constraint layer enforces max participation, max order rate, and cancel throttling.

### Scripts
Preprocess raw events into sequence tensors:
```bash
timeout 120s python -m snn_bench.scripts.preprocess \
  --events data/execution_events.csv \
  --out-dir artifacts/execution_policy/preprocessed \
  --top-k 5 --lookback-events 50
```

Train behavior cloning SNN policy:
```bash
timeout 1200s python -m snn_bench.scripts.train_bc \
  --payload artifacts/execution_policy/preprocessed/sequence_payload.npz \
  --meta artifacts/execution_policy/preprocessed/meta.json \
  --out-dir artifacts/execution_policy/bc \
  --window 64 --epochs 20 --batch-size 64 --model snn
```

Optional RL fine-tuning:
```bash
timeout 1200s python -m snn_bench.scripts.train_rl \
  --payload artifacts/execution_policy/preprocessed/sequence_payload.npz \
  --meta artifacts/execution_policy/preprocessed/meta.json \
  --bc-checkpoint artifacts/execution_policy/bc/policy_bc_best.pt \
  --out-dir artifacts/execution_policy/rl \
  --epochs 5
```

Evaluate policy and emit report/plots:
```bash
timeout 300s python -m snn_bench.scripts.eval_policy \
  --payload artifacts/execution_policy/preprocessed/sequence_payload.npz \
  --meta artifacts/execution_policy/preprocessed/meta.json \
  --checkpoint artifacts/execution_policy/bc/policy_bc_best.pt \
  --out-dir artifacts/execution_policy/eval
```

### Outputs
Pipeline artifacts include:
- policy checkpoints (`policy_bc_best.pt`, optional `policy_rl_last.pt`)
- evaluation report (`eval_report.md`) + metric JSON
- plots (`action_sequence.png`)
