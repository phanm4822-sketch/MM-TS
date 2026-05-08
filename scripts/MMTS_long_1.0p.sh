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


RATIO=${RATIO:-1.0}
DATASETS=${DATASETS:-"etth1 etth2 ettm1 ettm2 weather exchange ecl illness"}
EXTRA_ARGS=("$@")

if [[ -z "${LABEL:-}" ]]; then
  case "$RATIO" in
    1|1.0) LABEL=full ;;
    0.05) LABEL=5p ;;
    0.1|0.10) LABEL=10p ;;
    *) LABEL="${RATIO}p" ;;
  esac
fi

should_run() {
  [[ " $DATASETS " == *" $1 "* ]]
}

run_experiment() {
  local dataset=$1 data_path=$2 seq_len=$3 pred_len=$4 batch_size=$5 epochs=$6 patience=$7 lr=$8 weight_decay=$9
  shift 9
  should_run "$dataset" || return 0

  local output_dir="runs/${dataset}/pred${pred_len}"
  local shot_args=()
  if [[ "$RATIO" != "1" && "$RATIO" != "1.0" ]]; then
    output_dir="runs/few_shot/${LABEL}/${dataset}/pred${pred_len}"
    shot_args=(--few_shot_ratio "$RATIO")
  fi

  echo "[MM-TS] dataset=${dataset} pred_len=${pred_len} ratio=${RATIO}"
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
    "${shot_args[@]}" \
    "$@" \
    "${EXTRA_ARGS[@]}"
}

# dataset data_path seq_len pred_len batch_size epochs patience lr weight_decay [extra args]

# ETTH1
run_experiment etth1 datasets/ETTh1.csv 96 96 32 20 3 0.0002 0.0012 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_scale 0.06 \
  --ts_bias_dtw_weight 2.5 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.5 \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --ts_pooling tail_concat \
  --ts_tail_k 6 \
  --pred_head_mode mlp \
  --prompt_style hybrid_summary \
  --prompt_max_tokens 96 \
  --revin_subtract_last true
run_experiment etth1 datasets/ETTh1.csv 96 192 32 20 3 0.0002 0.0012 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_scale 0.06 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --ts_pooling tail_concat \
  --ts_tail_k 6 \
  --pred_head_mode mlp \
  --prompt_style hybrid_summary \
  --prompt_max_tokens 96 \
  --revin_subtract_last true
run_experiment etth1 datasets/ETTh1.csv 96 336 32 20 3 0.00016 0.001016468921334232 \
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
run_experiment etth1 datasets/ETTh1.csv 96 720 32 20 3 0.00014 0.001016468921334232 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --img_scale 1.05 \
  --vid_scale 1.05 \
  --text_scale 1.05 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.3 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.3 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --ts_pooling tail_concat \
  --ts_tail_k 6 \
  --pred_head_mode mlp \
  --prompt_style sample_only \
  --prompt_max_tokens 160 \
  --prompt_topk_lags 7 \
  --prompt_lag_limit 144 \
  --revin_subtract_last true

# ETTH2
run_experiment etth2 datasets/ETTh2.csv 96 96 32 20 3 0.0002 0.0014 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --lora_dropout 0.1 \
  --lora_target_layers 0,1,12,13,26,27 \
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
  --lora_target_layers 0,1,12,13,26,27 \
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

# ETTM1
run_experiment ettm1 datasets/ETTm1.csv 96 96 32 20 3 0.00024 0.001 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 192 \
  --eval_test_during_train true
run_experiment ettm1 datasets/ETTm1.csv 96 192 32 20 3 0.00024 0.001 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 192 \
  --eval_test_during_train true
run_experiment ettm1 datasets/ETTm1.csv 96 336 32 20 3 0.0002 0.0012 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 192 \
  --eval_test_during_train true
run_experiment ettm1 datasets/ETTm1.csv 96 720 32 20 3 0.00024 0.001 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

# ETTM2
run_experiment ettm2 datasets/ETTm2.csv 96 96 32 20 3 0.00022 0.0008 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 192 \
  --eval_test_during_train true
run_experiment ettm2 datasets/ETTm2.csv 96 192 32 20 3 0.00022 0.0008 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 192 \
  --eval_test_during_train true
run_experiment ettm2 datasets/ETTm2.csv 96 336 32 20 3 0.0002 0.001 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 192 \
  --eval_test_during_train true
run_experiment ettm2 datasets/ETTm2.csv 96 720 32 20 3 0.00022 0.0008 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --prompt_style global \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

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

# EXCHANGE
run_experiment exchange datasets/exchange_rate.csv 96 96 32 20 3 0.00015 5e-05 \
  --ts_attn_bias_layers 0,1 \
  --use_ts_residual true \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --revin_subtract_last true \
  --eval_test_during_train true
run_experiment exchange datasets/exchange_rate.csv 96 192 32 20 3 0.00015 5e-05 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --ts_attn_bias_layers 0,1 \
  --use_ts_residual true \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_context_mode all \
  --revin_subtract_last true
run_experiment exchange datasets/exchange_rate.csv 96 336 32 20 3 0.00013 3e-05 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --text_scale 1.05 \
  --ts_attn_bias_layers 0,1 \
  --use_ts_residual true \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --pred_context_mode all \
  --revin_subtract_last true
run_experiment exchange datasets/exchange_rate.csv 96 720 32 20 3 0.0001 8e-05 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --text_scale 1.05 \
  --ts_attn_bias_layers 0,1 \
  --use_ts_residual true \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --pred_context_mode all \
  --revin_subtract_last true

# ECL
run_experiment ecl datasets/electricity.csv 96 96 2 20 3 0.0002573176622078198 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu
run_experiment ecl datasets/electricity.csv 96 192 2 20 3 0.0002573176622078198 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu
run_experiment ecl datasets/electricity.csv 96 336 2 20 3 0.0002573176622078198 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu
run_experiment ecl datasets/electricity.csv 96 720 2 20 3 0.0002573176622078198 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_residual_mode B \
  --ts_residual_activation relu

# ILLNESS
run_experiment illness datasets/national_illness.csv 104 24 32 20 3 0.0002573176622078198 0.001016468921334232 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --prompt_style global \
  --prompt_max_tokens 260 \
  --eval_test_during_train true
run_experiment illness datasets/national_illness.csv 104 36 32 20 3 0.0002573176622078198 0.0012 \
  --use_ts_residual true \
  --text_scale 1.05 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --pred_head_dropout 0.02 \
  --pred_context_mode text \
  --prompt_style global \
  --prompt_max_tokens 260 \
  --eval_test_during_train true
run_experiment illness datasets/national_illness.csv 104 48 32 20 3 0.0002 0.0001 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --prompt_style global \
  --prompt_max_tokens 260 \
  --eval_test_during_train true
run_experiment illness datasets/national_illness.csv 104 60 32 20 3 0.0002 0.0001 \
  --use_ts_residual true \
  --text_scale 1.05 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --ts_bias_norm per_stat_tanh \
  --ts_residual_mode B \
  --ts_residual_activation relu \
  --pred_head_mode mlp \
  --pred_head_dropout 0.02 \
  --pred_context_mode text \
  --prompt_style global \
  --prompt_max_tokens 260 \
  --eval_test_during_train true
