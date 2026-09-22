# MM-TS

Implementation of **MM-TS: Channel-Structured Vision-Language Modeling for Multivariate Time Series Forecasting**.

## Key Designs

- **Visual relation tokens:** DTW similarity, covariance and Pearson correlation
  on FFT magnitudes form RGB images and video frames for the Qwen3-VL backbone.
- **Structured attention bias:** The same relations guide temporal-token attention,
  with local relations in diagonal patch blocks and global relations in off-diagonal blocks.

## Getting Started

### 1. Installation

Use Python 3.10 or later and a CUDA-enabled PyTorch build compatible with your GPU.

```bash
git clone https://github.com/phanm4822-sketch/MM-TS.git
cd MM-TS
pip install -r requirements.txt
```

The implementation uses `transformers==4.57.3`.

Download [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
with its weights, configuration, processor and tokenizer, then set:

```bash
export QWEN_DIR=/path/to/Qwen3-VL-2B-Instruct
```

### 2. Data Preparation

Download the benchmark datasets using the
[PatchTST data links](https://github.com/yuqinie98/PatchTST#supervised-learning)
and arrange the CSV files as follows:

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

Each file should contain a `date` column followed by numeric channel columns.
Standardization is fitted on the training split. Relation matrices and frozen
visual features are cached automatically on the first run and reused across seeds.
Electricity uses sharded caches; all datasets share the same model and training loop.
If the backbone or image rendering settings change, use a separate `--cache_dir`
or rebuild with `--rebuild_cache true`.

### 3. Training

Run the scripts from the repository root with Bash:

```bash
# ETTh1, all four horizons, seed 2026
GPUS=0 bash scripts/MMTS/etth1.sh

# All eight datasets
bash scripts/run_all.sh

# Select datasets, horizon and seed
DATASETS="etth1 ettm1 weather" HORIZONS="96" SEEDS=2026 bash scripts/run_all.sh

# Run five seeds
SEEDS="2026 2022 2023 2024 2025" bash scripts/run_all.sh

# Print the commands without running them
bash scripts/MMTS/electricity.sh --dry-run
```

The scripts use input length 96 and horizons 96, 192, 336 and 720. For ILI, the
input length is 104 and the horizons are 24, 36, 48 and 60. Dataset-specific
hyperparameters are listed in [Performance configurations](docs/performance_config.md).
Use `SEEDS` and `HORIZONS` to select runs. Other options can be appended to a
script, for example `--num_workers 4`; see `python run.py --help` for the full list.

The checkpoint with the lowest validation MSE is evaluated on the test split.
Results and checkpoints are saved under:

```text
runs/performance/<dataset>/seed<seed>/pred<horizon>/<source_dataset>/
  metrics.latest.json
  checkpoints/best.latest.pt
```

Timestamped copies are retained, and each result includes the run configuration.

### 4. Evaluation

Use the same dataset, horizon and configuration as the saved checkpoint:

```bash
HORIZONS=96 SEEDS=2026 bash scripts/MMTS/etth1.sh \
  --eval_only true --ckpt_path /path/to/best.latest.pt
```

## Code Structure

```text
run.py             training and evaluation entry point
models/            MM-TS model and Qwen3-VL integration
layers/            normalization, patch projection and token fusion
exp/               training, validation and testing
data_provider/     datasets, loaders and cache builders
utils/             prompts, relation biases and attention utilities
scripts/MMTS/      performance scripts for each dataset
tests/             model and data pipeline tests
```

## Tests

Tests run on CPU with small randomly initialized models and synthetic data.
Set `MMTS_TOKENIZER_DIR` to include the tokenizer tests.

```bash
python -m unittest discover -s tests -v
MMTS_TOKENIZER_DIR="$QWEN_DIR" python -m unittest discover -s tests -v
```
