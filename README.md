# MM-TS: Channel-Structured Vision-Language Modeling for Multivariate Time Series Forecasting

This repository provides the implementation of **MM-TS** for multivariate time-series forecasting.

## Introduction

MM-TS builds multimodal time-series representations with Qwen3-VL and uses them for long-term forecasting. The repository includes the model code, cache builders, and reproduction scripts for the experiments.

## Requirements

Create a conda environment and install the dependencies:

```bash
conda create -n mmts python=3.10
conda activate mmts
pip install -r requirements.txt
```

The code was tested with `transformers==4.57.3` and CUDA-enabled PyTorch. Install the PyTorch wheel that matches your CUDA driver if needed.

## Qwen3-VL Checkpoint

MM-TS uses `Qwen/Qwen3-VL-2B-Instruct` from ModelScope:

https://www.modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct

Download the checkpoint locally:

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-VL-2B-Instruct \
  --local_dir ./Qwen3-VL-2B-Instruct
```

The scripts use `./Qwen3-VL-2B-Instruct` by default. To use another location:

```bash
export QWEN_DIR=/path/to/Qwen3-VL-2B-Instruct
```

## Datasets

Datasets are not included in this repository. Please download the standard LTSF benchmark CSV files and place them under `./datasets`:

```text
datasets/ETTh1.csv
datasets/ETTh2.csv
datasets/ETTm1.csv
datasets/ETTm2.csv
datasets/exchange_rate.csv
datasets/weather.csv
datasets/national_illness.csv
datasets/electricity.csv
```

## Quick Start

Run full-shot long-term forecasting:

```bash
bash scripts/MMTS_long_1.0p.sh
```

Run few-shot long-term forecasting:

```bash
bash scripts/MMTS_long_0.05p.sh
bash scripts/MMTS_long_0.1p.sh
```

Run selected datasets:

```bash
DATASETS="etth1" bash scripts/MMTS_long_1.0p.sh
```

The long-term scripts cover ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, ECL, and ILI. For ETTh1, the command above runs all four prediction lengths: 96, 192, 336, and 720.

The first run builds the required cache automatically. Cache generation includes Qwen3-VL vision-token precomputation and requires a CUDA GPU.

Useful overrides:

```bash
GPUS=0 QWEN_DIR=/path/to/Qwen3-VL-2B-Instruct bash scripts/MMTS_long_1.0p.sh
GPUS=0,1 MODEL_PARALLEL=true bash scripts/MMTS_long_1.0p.sh
DATASETS="weather" bash scripts/MMTS_long_1.0p.sh --dry_run
```

## Usage

You can also call the main entry directly:

```bash
python main.py \
  --data_path datasets/weather.csv \
  --output_dir runs/weather/pred96 \
  --qwen_dir /path/to/Qwen3-VL-2B-Instruct \
  --pred_len 96
```

For checkpoint evaluation:

```bash
python main.py \
  --eval_only true \
  --ckpt_path /path/to/best.latest.pt \
  --data_path datasets/weather.csv \
  --output_dir runs/eval/weather96 \
  --qwen_dir /path/to/Qwen3-VL-2B-Instruct \
  --pred_len 96
```

Metrics and checkpoints are saved under the selected `--output_dir`. For example:

```text
runs/weather/pred96/weather/metrics.latest.json
runs/weather/pred96/weather/checkpoints/best.latest.pt
```

## Cache

When `--data_path` points to a raw CSV file, `main.py` builds the corresponding cache under `./cache` automatically.

- Standard datasets use `cache/<dataset>_<pred_len>.npz`.
- Electricity/ECL uses a sharded cache directory.
- Changing `batch_size` does not require rebuilding the cache.
- Rebuild the cache after changing `seq_len`, `pred_len`, `patch_len`, `stride`, or vision-cache settings.

Manual cache generation:

```bash
python -m data_provider.cache.build_cache \
  --data_path datasets/weather.csv \
  --output_path cache/weather_96.npz \
  --qwen_dir /path/to/Qwen3-VL-2B-Instruct \
  --pred_len 96
```

## Project Structure

```text
.
├── data_provider/          # dataset loaders and cache builders
├── exp/                    # training and evaluation loop
├── layers/                 # MM-TS layers
├── models/                 # Qwen3-VL loading, encoding, and LoRA helpers
├── scripts/                # reproduction scripts
├── utils/                  # rendering, statistics, and Qwen patching
├── main.py                 # main entry
└── requirements.txt
```
