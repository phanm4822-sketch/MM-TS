#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"

run_horizon exchange 96 \
  --data_path datasets/exchange_rate.csv \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00015 \
  --weight_decay 5e-05 \
  --ts_attn_bias_layers 0,1 \
  --use_ts_residual true \
  --revin_subtract_last true \
  --eval_test_during_train true

run_horizon exchange 192 \
  --data_path datasets/exchange_rate.csv \
  --seq_len 96 \
  --pred_len 192 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00015 \
  --weight_decay 5e-05 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --ts_attn_bias_layers 0,1 \
  --use_ts_residual true \
  --revin_subtract_last true

run_horizon exchange 336 \
  --data_path datasets/exchange_rate.csv \
  --seq_len 96 \
  --pred_len 336 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00013 \
  --weight_decay 3e-05 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --text_scale 1.05 \
  --ts_attn_bias_layers 0,1 \
  --use_ts_residual true \
  --revin_subtract_last true

run_horizon exchange 720 \
  --data_path datasets/exchange_rate.csv \
  --seq_len 96 \
  --pred_len 720 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0001 \
  --weight_decay 8e-05 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --text_scale 1.05 \
  --ts_attn_bias_layers 0,1 \
  --use_ts_residual true \
  --revin_subtract_last true
