#!/usr/bin/env bash
# Shared launcher. Source this file from a dataset script.
set -euo pipefail
MMTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$MMTS_ROOT"
PYTHON=${PYTHON:-python3}
QWEN_DIR=${QWEN_DIR:-Qwen3-VL-2B-Instruct}
GPUS=${GPUS:-0}
MODEL_PARALLEL=${MODEL_PARALLEL:-false}
SEEDS=${SEEDS:-2026}
HORIZONS=${HORIZONS:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-runs/performance}
EXTRA_ARGS=("$@")

run_horizon() {
  local dataset=$1 horizon=$2
  shift 2
  if [[ -n "$HORIZONS" && " $HORIZONS " != *" $horizon "* ]]; then return 0; fi
  local seed item dry_run
  for seed in $SEEDS; do
    dry_run=false
    local extra=()
    for item in "${EXTRA_ARGS[@]}"; do
      case "$item" in
        --dry-run|--dry_run) dry_run=true ;;
        --seed|--seed=*|--pred_len|--pred_len=*|--output_dir|--output_dir=*)
          echo 'Use SEEDS, HORIZONS and OUTPUT_ROOT to keep result directories aligned.' >&2; return 2 ;;
        *) extra+=("$item") ;;
      esac
    done
    local cmd=("$PYTHON" -u run.py --qwen_dir "$QWEN_DIR" --gpus "$GPUS"
      --model_parallel "$MODEL_PARALLEL" "$@" "${extra[@]}"
      --seed "$seed" --output_dir "$OUTPUT_ROOT/$dataset/seed$seed/pred$horizon")
    if [[ "$dry_run" == true ]]; then
      printf '%q ' "${cmd[@]}"; printf '\n'
    else
      "${cmd[@]}"
    fi
  done
}
