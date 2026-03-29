#!/usr/bin/env bash
set -euo pipefail

# Comprehensive smoke sweep across benchmark tasks x models.
#
# Defaults are intentionally smoke-sized so this can run as a quick system check.
# Override via env vars when you need deeper coverage.
#
# Examples:
#   timeout 3600s ./scripts/train_all_models_tasks.sh
#   MODEL_GROUP=baseline TASKS="direction_5m_distribution regime_classification" timeout 2400s ./scripts/train_all_models_tasks.sh
#   STOP_ON_ERROR=1 MAX_YEARS=1 timeout 7200s ./scripts/train_all_models_tasks.sh

TICKER="${1:-AAPL}"
TIMEFRAME="${2:-1D}"
OUT_DIR="${OUT_DIR:-artifacts/model_task_matrix}"
MAX_YEARS="${MAX_YEARS:-0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
SEED="${SEED:-7}"
MODEL_GROUP="${MODEL_GROUP:-all}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
SPLIT_MODE="${SPLIT_MODE:-random}"
TIMEOUT_PER_RUN="${TIMEOUT_PER_RUN:-900s}"

# Task configs currently available in snn_bench/configs/tasks.
TASKS_DEFAULT=(
  direction_5m_distribution
  direction_30m_distribution
  realized_vol_30m
  options_iv_skew_movement
  regime_classification
)

BASELINE_MODELS=(
  naive_persistence
  logreg
  gbm
  mlp
  markov_discrete
  hmm_gaussian
)

SNN_MODELS=(
  snntorch_lif
  snntorch_alif
  norse_lsnn
  spikingjelly_lif
  bio_plausible_alif
)

TEMPORAL_MODELS=(
  spikingjelly_temporal_conv
  tcn_spike
  norse_recurrent_lsnn
  lava_lif
)

if [[ -n "${TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  TASK_LIST=( ${TASKS} )
else
  TASK_LIST=("${TASKS_DEFAULT[@]}")
fi

case "$MODEL_GROUP" in
  baseline)
    MODEL_LIST=("${BASELINE_MODELS[@]}")
    ;;
  snn)
    MODEL_LIST=("${SNN_MODELS[@]}")
    ;;
  temporal)
    MODEL_LIST=("${TEMPORAL_MODELS[@]}")
    ;;
  all)
    MODEL_LIST=("${BASELINE_MODELS[@]}" "${SNN_MODELS[@]}" "${TEMPORAL_MODELS[@]}")
    ;;
  *)
    echo "[error] Unknown MODEL_GROUP='$MODEL_GROUP'. Use one of: all, baseline, snn, temporal." >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"

echo "[matrix] ticker=$TICKER timeframe=$TIMEFRAME model_group=$MODEL_GROUP split=$SPLIT_MODE"
echo "[matrix] models=${#MODEL_LIST[@]} tasks=${#TASK_LIST[@]} timeout_per_run=$TIMEOUT_PER_RUN"

run_count=0
pass_count=0
skip_count=0
fail_count=0

for task in "${TASK_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    run_count=$((run_count + 1))
    run_name="matrix_${task}_${model}"

    # Regression task currently not supported by train.py model-zoo path.
    if [[ "$task" == "realized_vol_30m" ]]; then
      echo "[skip $run_count] task=$task model=$model reason=train.py currently enforces classification outputs"
      skip_count=$((skip_count + 1))
      continue
    fi

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
      --smoke-epochs "${SMOKE_EPOCHS:-1}"; then
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

echo "[summary] total=$run_count pass=$pass_count skip=$skip_count fail=$fail_count"
if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi
