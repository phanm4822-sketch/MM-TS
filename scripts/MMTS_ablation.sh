#!/usr/bin/env bash
set -euo pipefail

QWEN_DIR=${QWEN_DIR:-Qwen3-VL-2B-Instruct}
GPUS=${GPUS:-0}
MODEL_PARALLEL=${MODEL_PARALLEL:-false}
PYTHON=${PYTHON:-python3}

MMTS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMTS_ROOT="$(cd "$MMTS_SCRIPT_DIR/.." && pwd)"
cd "$MMTS_ROOT"

run_mmts() {
  local dry_run=false
  local args=()
  for item in "$@"; do
    if [[ "$item" == "--dry_run" || "$item" == "--dry-run" ]]; then
      dry_run=true
    else
      args+=("$item")
    fi
  done

  local cmd=("$PYTHON" main.py --qwen_dir "$QWEN_DIR" --gpus "$GPUS" --model_parallel "$MODEL_PARALLEL" "${args[@]}")
  if [[ "$dry_run" == true ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}


DATASETS=${DATASETS:-"etth2 weather"}
VARIANTS=${VARIANTS:-"full wo_vision wo_text wo_ts_bias"}
EXTRA_ARGS=("$@")

should_run_dataset() {
  [[ " $DATASETS " == *" $1 "* ]]
}

should_run_variant() {
  [[ " $VARIANTS " == *" $1 "* ]]
}

run_experiment() {
  local dataset=$1 data_path=$2 seq_len=$3 pred_len=$4 batch_size=$5 epochs=$6 patience=$7 lr=$8 weight_decay=$9
  shift 9
  should_run_dataset "$dataset" || return 0

  local variant
  for variant in full wo_vision wo_text wo_ts_bias; do
    should_run_variant "$variant" || continue

    local output_dir="runs/ablation/${dataset}/${variant}/pred${pred_len}"
    local variant_args=()
    case "$variant" in
      full)
        ;;
      wo_vision)
        variant_args=(--use_vision false)
        ;;
      wo_text)
        variant_args=(--use_text false)
        ;;
      wo_ts_bias)
        variant_args=(--use_ts_attn_bias false)
        ;;
      *)
        echo "Unknown ablation variant: $variant" >&2
        exit 1
        ;;
    esac

    echo "[MM-TS Ablation] dataset=${dataset} variant=${variant} pred_len=${pred_len}"
    run_mmts \
      --data_path "$data_path" \
      --output_dir "$output_dir" \
      --seq_len "$seq_len" \
      --pred_len "$pred_len" \
      --patch_len 16 \
      --stride 8 \
      --batch_size "$batch_size" \
      --epochs "$epochs" \
      --patience "$patience" \
      --lr "$lr" \
      --weight_decay "$weight_decay" \
      "$@" \
      "${EXTRA_ARGS[@]}" \
      "${variant_args[@]}"
  done
}

# dataset data_path seq_len pred_len batch_size epochs patience lr weight_decay [extra args]

# ETTH2
run_experiment etth2 datasets/ETTh2.csv 96 96 32 20 3 0.0002 0.0014 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --lora_dropout 0.1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --ts_pooling tail_concat \
  --ts_tail_k 6 \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 96 \
  --revin_subtract_last true
run_experiment etth2 datasets/ETTh2.csv 96 192 32 20 3 0.00016 0.001016468921334232 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_scale 0.06 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --ts_pooling tail_concat \
  --ts_tail_k 6 \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 96 \
  --revin_subtract_last true
run_experiment etth2 datasets/ETTh2.csv 96 336 32 20 3 0.00016 0.001016468921334232 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --lora_dropout 0.1 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --ts_pooling tail_concat \
  --ts_tail_k 6 \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 96 \
  --revin_subtract_last true
run_experiment etth2 datasets/ETTh2.csv 96 720 32 20 3 0.00016 0.001016468921334232 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --ts_pooling tail_concat \
  --ts_tail_k 6 \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 96 \
  --revin_subtract_last true

# WEATHER
run_experiment weather datasets/weather.csv 96 96 32 20 3 0.0002573176622078198 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu
run_experiment weather datasets/weather.csv 96 192 32 20 3 0.0002573176622078198 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu
run_experiment weather datasets/weather.csv 96 336 32 20 3 0.0002573176622078198 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu
run_experiment weather datasets/weather.csv 96 720 32 20 3 0.0002573176622078198 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu
