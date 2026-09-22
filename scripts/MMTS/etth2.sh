#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" "$@"

run_horizon etth2 96 \
  --data_path datasets/ETTh2.csv \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.0002 \
  --weight_decay 0.0014 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --lora_dropout 0.1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --prompt_max_tokens 192 \
  --revin_subtract_last true

run_horizon etth2 192 \
  --data_path datasets/ETTh2.csv \
  --seq_len 96 \
  --pred_len 192 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00016 \
  --weight_decay 0.001016468921334232 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_bias_scale 0.06 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --prompt_max_tokens 192 \
  --revin_subtract_last true

run_horizon etth2 336 \
  --data_path datasets/ETTh2.csv \
  --seq_len 96 \
  --pred_len 336 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00016 \
  --weight_decay 0.001016468921334232 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --lora_dropout 0.1 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --prompt_max_tokens 192 \
  --revin_subtract_last true

run_horizon etth2 720 \
  --data_path datasets/ETTh2.csv \
  --seq_len 96 \
  --pred_len 720 \
  --batch_size 32 \
  --epochs 20 \
  --patience 3 \
  --lr 0.00016 \
  --weight_decay 0.001016468921334232 \
  --lr_schedule cosine \
  --min_lr_ratio 0.05 \
  --use_ts_residual true \
  --lora_alpha 8 \
  --ts_attn_bias_layers 0,1 \
  --ts_bias_dtw_weight 2.0 \
  --ts_bias_cov_weight 2.0 \
  --prompt_max_tokens 192 \
  --revin_subtract_last true
