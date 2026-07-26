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


TRANSFERS=${TRANSFERS:-"etth1_to_etth2 etth1_to_ettm2 ettm2_to_ettm1 ettm2_to_etth2 etth2_to_ettm2 ettm1_to_etth2"}
TRAIN_SOURCES=${TRAIN_SOURCES:-true}
RETRAIN=${RETRAIN:-false}
EXTRA_ARGS=("$@")

should_run_transfer() {
  [[ " $TRANSFERS " == *" $1 "* ]]
}

run_transfer() {
  local transfer=$1 source_tag=$2 source_name=$3 source_data=$4 target_tag=$5 target_name=$6 target_data=$7
  local seq_len=$8 pred_len=$9 batch_size=${10} epochs=${11} patience=${12} lr=${13} weight_decay=${14}
  shift 14
  should_run_transfer "$transfer" || return 0

  local train_output_dir="runs/zero_shot/train/${source_tag}/pred${pred_len}"
  local ckpt_path="${train_output_dir}/${source_name}/checkpoints/best.latest.pt"
  local eval_output_dir="runs/zero_shot/${transfer}/pred${pred_len}"

  if [[ "$TRAIN_SOURCES" == true ]]; then
    if [[ "$RETRAIN" == true || ! -f "$ckpt_path" ]]; then
      echo "[MM-TS ZeroShot] train source=${source_tag} pred_len=${pred_len}"
      run_mmts \
        --data_path "$source_data" \
        --output_dir "$train_output_dir" \
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
        "${EXTRA_ARGS[@]}"
    else
      echo "[MM-TS ZeroShot] skip existing source checkpoint: $ckpt_path"
    fi
  fi

  echo "[MM-TS ZeroShot] eval transfer=${transfer} pred_len=${pred_len}"
  run_mmts \
    --eval_only true \
    --ckpt_path "$ckpt_path" \
    --data_path "$target_data" \
    --output_dir "$eval_output_dir" \
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
    "${EXTRA_ARGS[@]}"
}

# transfer source_tag source_name source_data target_tag target_name target_data
# seq_len pred_len batch_size epochs patience lr weight_decay [extra args]

run_etth1_transfer() {
  local transfer=$1 target_tag=$2 target_name=$3 target_data=$4

  run_transfer "$transfer" etth1 ETTh1 datasets/ETTh1.csv "$target_tag" "$target_name" "$target_data" 96 96 32 20 3 0.0002 0.0012 \
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
  run_transfer "$transfer" etth1 ETTh1 datasets/ETTh1.csv "$target_tag" "$target_name" "$target_data" 96 192 32 20 3 0.0002 0.0012 \
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
  run_transfer "$transfer" etth1 ETTh1 datasets/ETTh1.csv "$target_tag" "$target_name" "$target_data" 96 336 32 20 3 0.00016 0.001016468921334232 \
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
  run_transfer "$transfer" etth1 ETTh1 datasets/ETTh1.csv "$target_tag" "$target_name" "$target_data" 96 720 32 20 3 0.00014 0.001016468921334232 \
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
}

run_etth2_transfer() {
  local transfer=$1 target_tag=$2 target_name=$3 target_data=$4

  run_transfer "$transfer" etth2 ETTh2 datasets/ETTh2.csv "$target_tag" "$target_name" "$target_data" 96 96 32 20 3 0.0002 0.0014 \
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
  run_transfer "$transfer" etth2 ETTh2 datasets/ETTh2.csv "$target_tag" "$target_name" "$target_data" 96 192 32 20 3 0.00016 0.001016468921334232 \
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
  run_transfer "$transfer" etth2 ETTh2 datasets/ETTh2.csv "$target_tag" "$target_name" "$target_data" 96 336 32 20 3 0.00016 0.001016468921334232 \
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
  run_transfer "$transfer" etth2 ETTh2 datasets/ETTh2.csv "$target_tag" "$target_name" "$target_data" 96 720 32 20 3 0.00016 0.001016468921334232 \
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
}

run_ettm1_transfer() {
  local transfer=$1 target_tag=$2 target_name=$3 target_data=$4

  run_transfer "$transfer" ettm1 ETTm1 datasets/ETTm1.csv "$target_tag" "$target_name" "$target_data" 96 96 32 20 3 0.00024 0.001 \
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
  run_transfer "$transfer" ettm1 ETTm1 datasets/ETTm1.csv "$target_tag" "$target_name" "$target_data" 96 192 32 20 3 0.00024 0.001 \
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
  run_transfer "$transfer" ettm1 ETTm1 datasets/ETTm1.csv "$target_tag" "$target_name" "$target_data" 96 336 32 20 3 0.0002 0.0012 \
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
  run_transfer "$transfer" ettm1 ETTm1 datasets/ETTm1.csv "$target_tag" "$target_name" "$target_data" 96 720 32 20 3 0.00024 0.001 \
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
}

run_ettm2_transfer() {
  local transfer=$1 target_tag=$2 target_name=$3 target_data=$4

  run_transfer "$transfer" ettm2 ETTm2 datasets/ETTm2.csv "$target_tag" "$target_name" "$target_data" 96 96 32 20 3 0.00022 0.0008 \
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
  run_transfer "$transfer" ettm2 ETTm2 datasets/ETTm2.csv "$target_tag" "$target_name" "$target_data" 96 192 32 20 3 0.00022 0.0008 \
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
  run_transfer "$transfer" ettm2 ETTm2 datasets/ETTm2.csv "$target_tag" "$target_name" "$target_data" 96 336 32 20 3 0.0002 0.001 \
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
  run_transfer "$transfer" ettm2 ETTm2 datasets/ETTm2.csv "$target_tag" "$target_name" "$target_data" 96 720 32 20 3 0.00022 0.0008 \
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
}

run_etth1_transfer etth1_to_etth2 etth2 ETTh2 datasets/ETTh2.csv
run_etth1_transfer etth1_to_ettm2 ettm2 ETTm2 datasets/ETTm2.csv
run_ettm2_transfer ettm2_to_ettm1 ettm1 ETTm1 datasets/ETTm1.csv
run_ettm2_transfer ettm2_to_etth2 etth2 ETTh2 datasets/ETTh2.csv
run_etth2_transfer etth2_to_ettm2 ettm2 ETTm2 datasets/ETTm2.csv
run_ettm1_transfer ettm1_to_etth2 etth2 ETTh2 datasets/ETTh2.csv
