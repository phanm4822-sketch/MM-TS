# MM-TS: Channel-Structured Vision-Language Modeling for Multivariate Time Series Forecasting

This repository contains the implementation of MM-TS.

MM-TS computes channel relations from the FFT magnitude spectrum of each input
window. These relations serve as both synthetic image/video inputs and structured
attention biases in a Qwen3-VL backbone, alongside time-series patches and text
prompts. A shared prediction head maps the temporal representations to future values.

## Installation

Use Python 3.10 or later and a CUDA-enabled PyTorch build compatible with your GPU.

```bash
git clone https://github.com/phanm4822-sketch/MM-TS.git
cd MM-TS
pip install -r requirements.txt
```

The implementation requires `transformers==4.57.3`. Unit tests have been verified
with PyTorch 2.8.0 and PEFT 0.21.0 on CPU.

Download [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
with its weights, configuration, processor and tokenizer, then set:

```bash
export QWEN_DIR=/path/to/Qwen3-VL-2B-Instruct
```

## Data preparation

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

## Training

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

Checkpoints are selected by validation MSE and evaluated on the test split.
Any test scores logged during training are not used for checkpoint selection.
Results and checkpoints are saved under:

```text
runs/performance/<dataset>/seed<seed>/pred<horizon>/<source_dataset>/
  metrics.latest.json
  checkpoints/best.latest.pt
```

Timestamped copies are retained, and each result includes the run configuration.

## Evaluation

Use the same dataset, horizon and configuration as the saved checkpoint:

```bash
HORIZONS=96 SEEDS=2026 bash scripts/MMTS/etth1.sh \
  --eval_only true --ckpt_path /path/to/best.latest.pt
```

## Code structure

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

Tests use small randomly initialized models and synthetic data. Pretrained
weights are not required. Set `MMTS_TOKENIZER_DIR` to include the tokenizer tests.

```bash
python -m unittest discover -s tests -v
MMTS_TOKENIZER_DIR="$QWEN_DIR" python -m unittest discover -s tests -v
```

## Acknowledgements

The code organization follows [PatchTST](https://github.com/yuqinie98/PatchTST)
and [iTransformer](https://github.com/thuml/iTransformer). The backbone uses
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) through Transformers and PEFT.
See [Sources and dependencies](docs/sources.md) for the upstream projects.
