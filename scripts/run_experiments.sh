#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-snn_bench/configs/experiments/aapl_model_sweep.yaml}"
OUT_DIR="${OUT_DIR:-artifacts/experiments}"
MAX_YEARS="${MAX_YEARS:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
USE_DOCKER="${USE_DOCKER:-0}"
DOCKER_IMAGE="${DOCKER_IMAGE:-snn-bench:latest}"

EXTRA_FLAGS=()
if [[ "$STOP_ON_ERROR" == "1" ]]; then
  EXTRA_FLAGS+=(--stop-on-error)
fi

resolve_python_cmd() {
  if [[ "$USE_DOCKER" == "1" ]]; then
    PYTHON_MODE="docker"
    PYTHON_CMD="docker"
    return
  fi

  if timeout 20s python -c "import yaml" >/dev/null 2>&1; then
    PYTHON_MODE="local"
    PYTHON_CMD="python"
    return
  fi

  if command -v conda >/dev/null 2>&1 && timeout 20s conda env list | awk '{print $1}' | grep -qx "snnbench"; then
    PYTHON_MODE="conda"
    PYTHON_CMD="conda run --no-capture-output -n snnbench python"
    echo "[info] Local python is missing PyYAML; using conda env 'snnbench'."
    return
  fi

  if command -v docker >/dev/null 2>&1 && timeout 20s docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    PYTHON_MODE="docker"
    PYTHON_CMD="docker"
    echo "[info] Local python is missing PyYAML; using Docker image '$DOCKER_IMAGE'."
    return
  fi

  echo "[error] No runnable python environment found with PyYAML available."
  echo "        Run 'timeout 300s pip install -e .' or activate conda env 'snnbench'."
  echo "        Alternative: set USE_DOCKER=1 with an available image (default: $DOCKER_IMAGE)."
  exit 2
}

run_experiment_module() {
  if [[ "$PYTHON_MODE" == "docker" ]]; then
    mkdir -p "$OUT_DIR"
    timeout 1200s docker run --rm \
      -e MASSIVE_API_KEY="${MASSIVE_API_KEY:-}" \
      -e MASSIVE_API_KEY_FILE="${MASSIVE_API_KEY_FILE:-/etc/Massive/api-key}" \
      -v "$PWD:/app" \
      -w /app \
      "$DOCKER_IMAGE" \
      python -m snn_bench.scripts.run_experiments \
      --manifest "$MANIFEST" \
      --out-dir "$OUT_DIR" \
      --max-years "$MAX_YEARS" \
      "${EXTRA_FLAGS[@]}"
    return
  fi

  # shellcheck disable=SC2086
  timeout 1200s $PYTHON_CMD -m snn_bench.scripts.run_experiments \
    --manifest "$MANIFEST" \
    --out-dir "$OUT_DIR" \
    --max-years "$MAX_YEARS" \
    "${EXTRA_FLAGS[@]}"
}

resolve_python_cmd
run_experiment_module
