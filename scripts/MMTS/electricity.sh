#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"

run_horizon electricity 96 \
  --data_path datasets/electricity.csv \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 2 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002573176622078198 \
  --weight_decay 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0

run_horizon electricity 192 \
  --data_path datasets/electricity.csv \
  --seq_len 96 \
  --pred_len 192 \
  --batch_size 2 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002573176622078198 \
  --weight_decay 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0

run_horizon electricity 336 \
  --data_path datasets/electricity.csv \
  --seq_len 96 \
  --pred_len 336 \
  --batch_size 2 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002573176622078198 \
  --weight_decay 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0

run_horizon electricity 720 \
  --data_path datasets/electricity.csv \
  --seq_len 96 \
  --pred_len 720 \
  --batch_size 2 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002573176622078198 \
  --weight_decay 0.001016468921334232 \
  --log_interval 1 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0
