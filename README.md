# MM-TS

Implementation of **MM-TS: Channel-Structured Vision-Language Modeling for Multivariate Time Series Forecasting**.

## Key Designs

- **Visual relation tokens:** DTW similarity, covariance and Pearson correlation
  on FFT magnitudes form RGB images and video frames for Qwen3-VL.
- **Structured attention bias:** The same relations guide temporal-token attention,
  with local relations in diagonal patch blocks and global relations in off-diagonal blocks.

## Getting Started

### 1. Installation

Use Python 3.10 or later with CUDA builds of PyTorch and torchvision.

```bash
git clone https://github.com/phanm4822-sketch/MM-TS.git
cd MM-TS
pip install -r requirements.txt
```

Download [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
and set its local directory:

```bash
export QWEN_DIR=/path/to/Qwen3-VL-2B-Instruct
```

### 2. Data Preparation

Download the [benchmark datasets](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
and place the CSV files under `datasets/`:

```text
datasets/
  ETTh1.csv
  ETTh2.csv
  ETTm1.csv
  ETTm2.csv
  weather.csv
  electricity.csv
  exchange_rate.csv
  national_illness.csv
```

Each CSV has a `date` column followed by the variables. Standardization uses
the training split. Relation matrices and frozen visual features are cached on
the first run; Electricity uses sharded caches. Use `--rebuild_cache true`
after changing the backbone or image rendering settings.

### 3. Training

Performance scripts are in `scripts/MMTS/`. Run them from the repository root:

```bash
# ETTh1, four horizons, seed 2026
GPUS=0 bash scripts/MMTS/etth1.sh

# All eight datasets
bash scripts/run_all.sh

# Select datasets, horizon and seed
DATASETS="etth1 ettm1 weather" HORIZONS=96 SEEDS=2026 bash scripts/run_all.sh

# Five seeds
SEEDS="2026 2022 2023 2024 2025" bash scripts/run_all.sh
```

The scripts use input length 96 and horizons 96, 192, 336 and 720.
ILI uses input length 104 and horizons 24, 36, 48 and 60.
Hyperparameters are set in each dataset script. Additional arguments can be
appended to a script; see `python run.py --help`.

The checkpoint with the lowest validation MSE is selected for testing.
Results and checkpoints are saved under
`runs/performance/<dataset>/seed<seed>/pred<horizon>/<source_dataset>/`
as `metrics.latest.json` and `checkpoints/best.latest.pt`.

### 4. Evaluation

Use the dataset, horizon and model configuration of the saved checkpoint:

```bash
HORIZONS=96 SEEDS=2026 bash scripts/MMTS/etth1.sh \
  --eval_only true --ckpt_path /path/to/best.latest.pt
```

## Runtime

`--optimize_runtime true` offloads frozen modules during cached training and
evaluation. Add `--cuda_graphs true` for fixed-shape CUDA graph replay with
PyTorch 2.9.1. These options use a single GPU; graph buffers require extra memory.

```bash
HORIZONS=96 SEEDS=2026 bash scripts/MMTS/etth1.sh \
  --optimize_runtime true --cuda_graphs true
```

For online inference, pass float32 windows of shape `[B, L, C]` standardized
with training-split statistics. Predictions use the same standardized scale.
The model should have its checkpoint loaded and channels configured.

```python
import torch
from runtime import optimized_runtime, predict_window

model.eval()
with torch.inference_mode(), optimized_runtime(model, cached_visual=False):
    prediction = predict_window(model, windows).clone()
```

Keep the context open across requests and use one model per worker process.
Online DTW compilation uses Numba. The `fast_vision=True` option additionally
uses a frozen BF16 visual kernel and visual CUDA graphs on Linux with
PyTorch 2.9.1+cu126, Transformers 4.57.3 and an Ada GPU.
Install `requirements-optimized.txt` for these dependencies; the visual
extension requires a C++ compiler and builds on first use.
