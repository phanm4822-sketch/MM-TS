# MM-TS

Implementation of **MM-TS: Channel-Structured Vision-Language Modeling for Multivariate Time Series Forecasting**.

## Key Designs

- **Visual relation tokens:** DTW similarity, covariance and Pearson correlation on FFT magnitudes form RGB images and video frames for Qwen3-VL.
- **Structured attention bias:** Local relations fill diagonal patch blocks; global relations fill off-diagonal blocks in temporal-token attention.

## Getting Started

1. Install the dependencies with Python 3.10 or later and CUDA builds of PyTorch and torchvision.

   ```bash
   git clone https://github.com/phanm4822-sketch/MM-TS.git
   cd MM-TS
   pip install -r requirements.txt
   ```

2. Download [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and set its local path:

   ```bash
   export QWEN_DIR=/path/to/Qwen3-VL-2B-Instruct
   ```

3. Download the [benchmark datasets](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and place the CSV files under `datasets/`. Each CSV has a `date` column followed by the variables.

4. Run a dataset script from the repository root:

   ```bash
   bash scripts/MMTS/weather.sh
   ```

Scripts are in `scripts/MMTS/`. The default input length is 96, with prediction lengths 96, 192, 336 and 720; ILI uses input length 104 and prediction lengths 24, 36, 48 and 60. The default seed is 2026.

Standardization uses the training split. Visual features and relations are cached on the first run. Set `--rebuild_cache true` after changing the backbone or rendering settings.

## Training and Evaluation

```bash
# All datasets
bash scripts/run_all.sh

# Select datasets, prediction length and seeds
DATASETS="etth1 ettm1 weather" HORIZONS=96 SEEDS="2026 2022 2023 2024 2025" bash scripts/run_all.sh

# Evaluate a checkpoint
HORIZONS=96 SEEDS=2026 bash scripts/MMTS/etth1.sh \
  --eval_only true --ckpt_path /path/to/best.latest.pt
```

Hyperparameters are set in each dataset script. Additional arguments can be appended to a script; see `python run.py --help`. Evaluation uses the dataset, horizon and model configuration of the checkpoint.

The checkpoint with the lowest validation MSE is selected for testing. Results are saved under `runs/performance/<dataset>/seed<seed>/pred<horizon>/<source_dataset>/` as `metrics.latest.json` and `checkpoints/best.latest.pt`.
