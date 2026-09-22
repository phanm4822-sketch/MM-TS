#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"

run_horizon ettm2 96 \
  --data_path datasets/ETTm2.csv \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00022 \
  --weight_decay 0.0008 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

run_horizon ettm2 192 \
  --data_path datasets/ETTm2.csv \
  --seq_len 96 \
  --pred_len 192 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00022 \
  --weight_decay 0.0008 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

run_horizon ettm2 336 \
  --data_path datasets/ETTm2.csv \
  --seq_len 96 \
  --pred_len 336 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002 \
  --weight_decay 0.001 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

run_horizon ettm2 720 \
  --data_path datasets/ETTm2.csv \
  --seq_len 96 \
  --pred_len 720 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00022 \
  --weight_decay 0.0008 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true
