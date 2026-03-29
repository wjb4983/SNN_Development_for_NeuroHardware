#!/usr/bin/env bash
set -euo pipefail

# Task-aware matrix sweep: each task only runs models relevant for that task.
#
# Why this exists:
# - `train_all_models_tasks.sh` sweeps broad combinations for stress coverage.
# - This script narrows each task to practical model subsets (e.g., multiclass-ready models
#   for regime classification), so runs are cleaner and more representative.
#
# Examples:
#   timeout 3600s ./scripts/train_task_relevant_models.sh
#   TASKS="direction_5m_distribution regime_classification" timeout 2400s ./scripts/train_task_relevant_models.sh
#   STOP_ON_ERROR=1 TIMEOUT_PER_RUN=1200s timeout 7200s ./scripts/train_task_relevant_models.sh AAPL 1D

TICKER="${1:-AAPL}"
TIMEFRAME="${2:-1D}"
OUT_DIR="${OUT_DIR:-artifacts/task_relevant_matrix}"
MAX_YEARS="${MAX_YEARS:-0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
SEED="${SEED:-7}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
SPLIT_MODE="${SPLIT_MODE:-random}"
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-900s}"

TASKS_DEFAULT=(
  direction_5m_distribution
  direction_30m_distribution
  realized_vol_30m
  options_iv_skew_movement
  regime_classification
)

# Binary classification candidates.
MODELS_BINARY=(
  naive_persistence
  logreg
  gbm
  mlp
  markov_discrete
  hmm_gaussian
  snntorch_lif
  snntorch_alif
  norse_lsnn
  spikingjelly_lif
  bio_plausible_alif
  spikingjelly_temporal_conv
  tcn_spike
  norse_recurrent_lsnn
  lava_lif
)

# Multiclass-ready set for regime classification (4 classes).
MODELS_MULTICLASS=(
  logreg
  gbm
  mlp
)

if [[ -n "${TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  TASK_LIST=( ${TASKS} )
else
  TASK_LIST=("${TASKS_DEFAULT[@]}")
fi

mkdir -p "$OUT_DIR"

echo "[matrix-task-aware] ticker=$TICKER timeframe=$TIMEFRAME split=$SPLIT_MODE"
echo "[matrix-task-aware] tasks=${#TASK_LIST[@]} timeout_per_run=$TIMEOUT_PER_RUN"

run_count=0
pass_count=0
skip_count=0
fail_count=0

for task in "${TASK_LIST[@]}"; do
  MODEL_LIST=()
  EXTRA_ARGS=()

  case "$task" in
    direction_5m_distribution|direction_30m_distribution|options_iv_skew_movement)
      MODEL_LIST=("${MODELS_BINARY[@]}")
      ;;
    regime_classification)
      MODEL_LIST=("${MODELS_MULTICLASS[@]}")
      EXTRA_ARGS+=(--training-strategy multiclass --num-classes 4)
      ;;
    realized_vol_30m)
      # train.py currently raises for regression tasks in validate_task_model_compatibility.
      echo "[skip-task] task=$task reason=regression task currently blocked by train.py compatibility guard"
      skip_count=$((skip_count + 1))
      continue
      ;;
    *)
      echo "[skip-task] task=$task reason=unknown task name"
      skip_count=$((skip_count + 1))
      continue
      ;;
  esac

  if [[ "${#MODEL_LIST[@]}" -eq 0 ]]; then
    echo "[skip-task] task=$task reason=no relevant models configured"
    skip_count=$((skip_count + 1))
    continue
  fi

  echo "[task] $task models=${#MODEL_LIST[@]}"

  for model in "${MODEL_LIST[@]}"; do
    run_count=$((run_count + 1))
    run_name="taskaware_${task}_${model}"

    echo "[run  $run_count] task=$task model=$model"
    if timeout "$TIMEOUT_PER_RUN" python -m snn_bench.scripts.train \
      --ticker "$TICKER" \
      --timeframe "$TIMEFRAME" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --model "$model" \
      --task-name "$task" \
      --split-mode "$SPLIT_MODE" \
      --out-dir "$OUT_DIR" \
      --max-years "$MAX_YEARS" \
      --run-name "$run_name" \
      --seed "$SEED" \
      --smoke \
      --smoke-sample-size "${SMOKE_SAMPLE_SIZE:-256}" \
      --smoke-epochs "${SMOKE_EPOCHS:-1}" \
      "${EXTRA_ARGS[@]}"; then
      pass_count=$((pass_count + 1))
      echo "[pass $run_count] task=$task model=$model"
    else
      fail_count=$((fail_count + 1))
      echo "[fail $run_count] task=$task model=$model"
      if [[ "$STOP_ON_ERROR" == "1" ]]; then
        echo "[stop] STOP_ON_ERROR=1"
        break 2
      fi
    fi
  done
done

echo "[summary] total_runs=$run_count pass=$pass_count skip=$skip_count fail=$fail_count"
if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi
