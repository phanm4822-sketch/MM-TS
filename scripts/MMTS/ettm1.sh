#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"

run_horizon ettm1 96 \
  --data_path datasets/ETTm1.csv \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00024 \
  --weight_decay 0.001 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

run_horizon ettm1 192 \
  --data_path datasets/ETTm1.csv \
  --seq_len 96 \
  --pred_len 192 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00024 \
  --weight_decay 0.001 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

run_horizon ettm1 336 \
  --data_path datasets/ETTm1.csv \
  --seq_len 96 \
  --pred_len 336 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002 \
  --weight_decay 0.0012 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

run_horizon ettm1 720 \
  --data_path datasets/ETTm1.csv \
  --seq_len 96 \
  --pred_len 720 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00024 \
  --weight_decay 0.001 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true
