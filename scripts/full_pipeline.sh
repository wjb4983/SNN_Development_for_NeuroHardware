#!/usr/bin/env bash
set -euo pipefail

# End-to-end helper: dependency install, data fetch, smoke run, unit tests, and training.
# Usage:
#   ./scripts/full_pipeline.sh [ticker] [timeframe] [epochs] [stock_years] [option_years]
# Example:
#   MASSIVE_API_KEY_FILE=/etc/Massive/api-key ./scripts/full_pipeline.sh AAPL 1D 10 5 2

TICKER="${1:-AAPL}"
TIMEFRAME="${2:-1D}"
EPOCHS="${3:-5}"
STOCK_YEARS="${4:-5}"
OPTION_YEARS="${5:-2}"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
OUT_DIR="${OUT_DIR:-artifacts}"
USE_DOCKER="${USE_DOCKER:-0}"
DOCKER_IMAGE="${DOCKER_IMAGE:-snn-bench:latest}"

if [[ -z "${MASSIVE_API_KEY:-}" && -z "${MASSIVE_API_KEY_FILE:-}" ]]; then
  echo "[warn] MASSIVE_API_KEY or MASSIVE_API_KEY_FILE is not set."
  echo "       Data fetch and training will fail unless an API key is available in fallback locations."
fi

resolve_python_cmd() {
  if [[ "$USE_DOCKER" == "1" ]]; then
    PYTHON_CMD="docker"
    PYTHON_MODE="docker"
    return
  fi

  if timeout 20s python -c "import numpy" >/dev/null 2>&1; then
    PYTHON_CMD="python"
    PYTHON_MODE="local"
    return
  fi

  if command -v conda >/dev/null 2>&1 && timeout 20s conda env list | awk '{print $1}' | grep -qx "snnbench"; then
    PYTHON_CMD="conda run --no-capture-output -n snnbench python"
    PYTHON_MODE="conda"
    echo "[info] Local python is missing numpy; using conda env 'snnbench'."
    return
  fi

  if command -v docker >/dev/null 2>&1 && timeout 20s docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    PYTHON_CMD="docker"
    PYTHON_MODE="docker"
    echo "[info] Local python is missing numpy; using Docker image '$DOCKER_IMAGE'."
    return
  fi

  echo "[error] No runnable python environment found with numpy."
  echo "        Options: INSTALL_DEPS=1, activate/create conda env 'snnbench', or set USE_DOCKER=1."
  exit 2
}

run_python_module() {
  local timeout_s="$1"
  shift
  local module="$1"
  shift

  if [[ "$PYTHON_MODE" == "docker" ]]; then
    local key_mount=()
    if [[ -n "${MASSIVE_API_KEY_FILE:-}" && -f "${MASSIVE_API_KEY_FILE}" ]]; then
      key_mount=( -v "${MASSIVE_API_KEY_FILE}:/etc/Massive/api-key:ro" )
    fi

    timeout "$timeout_s" docker run --rm \
      -e MASSIVE_API_KEY="${MASSIVE_API_KEY:-}" \
      -e MASSIVE_API_KEY_FILE="${MASSIVE_API_KEY_FILE:-/etc/Massive/api-key}" \
      -v "$PWD/src/data:/app/src/data" \
      -v "$PWD/$OUT_DIR:/app/$OUT_DIR" \
      "${key_mount[@]}" \
      "$DOCKER_IMAGE" python -m "$module" "$@"
    return
  fi

  # shellcheck disable=SC2086
  timeout "$timeout_s" $PYTHON_CMD -m "$module" "$@"
}

if [[ "$INSTALL_DEPS" == "1" && "$USE_DOCKER" != "1" ]]; then
  echo "[step] Installing project dependencies"
  timeout 300s pip install -e .
fi

mkdir -p "$OUT_DIR"
resolve_python_cmd

echo "[step] Fetching market + options cache"
run_python_module 900s snn_bench.scripts.cache_market_data \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME" \
  --stock-years "$STOCK_YEARS" \
  --option-years "$OPTION_YEARS"

echo "[step] Running smoke pipeline"
run_python_module 180s snn_bench.scripts.smoke_pipeline \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME"

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[step] Running unit tests"
  if [[ "$PYTHON_MODE" == "docker" ]]; then
    timeout 240s docker run --rm \
      -v "$PWD:/app" \
      "$DOCKER_IMAGE" python -m unittest -v tests/test_repo_layout.py tests/test_secrets.py tests/test_forecast_features.py
  else
    # shellcheck disable=SC2086
    timeout 240s $PYTHON_CMD -m unittest -v tests/test_repo_layout.py tests/test_secrets.py tests/test_forecast_features.py
  fi
fi

echo "[step] Training baseline model"
run_python_module 600s snn_bench.scripts.train \
  --ticker "$TICKER" \
  --timeframe "$TIMEFRAME" \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --lr 0.001 \
  --out-dir "$OUT_DIR"

echo "[done] Pipeline complete"
echo "       Metrics: $OUT_DIR/train_metrics.json"
