#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"

run_horizon ili 24 \
  --data_path datasets/national_illness.csv \
  --seq_len 104 \
  --pred_len 24 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002573176622078198 \
  --weight_decay 0.001016468921334232 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

run_horizon ili 36 \
  --data_path datasets/national_illness.csv \
  --seq_len 104 \
  --pred_len 36 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002573176622078198 \
  --weight_decay 0.0012 \
  --use_ts_residual true \
  --text_scale 1.05 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

run_horizon ili 48 \
  --data_path datasets/national_illness.csv \
  --seq_len 104 \
  --pred_len 48 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002 \
  --weight_decay 0.0001 \
  --use_ts_residual true \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true

run_horizon ili 60 \
  --data_path datasets/national_illness.csv \
  --seq_len 104 \
  --pred_len 60 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002 \
  --weight_decay 0.0001 \
  --use_ts_residual true \
  --text_scale 1.05 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.2 \
  --ts_bias_cov_weight 2.0 \
  --ts_bias_pear_weight 2.2 \
  --prompt_max_tokens 192 \
  --eval_test_during_train true
