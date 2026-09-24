# MM-TS

Implementation of **MM-TS: Channel-Structured Vision-Language Modeling for Multivariate Time Series Forecasting**.

## Key Designs

- **Visual relation tokens:** DTW similarity, covariance and Pearson correlation
  on FFT magnitudes form RGB images and video frames for the Qwen3-VL backbone.
- **Structured attention bias:** The same relations guide temporal-token attention,
  with local relations in diagonal patch blocks and global relations in off-diagonal blocks.

## Getting Started

### 1. Installation

Use Python 3.10 or later and matching CUDA builds of PyTorch and torchvision.

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
hyperparameters are set in `scripts/MMTS/`.
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
```

## Optional Runtime Optimizations

The default entry point and checkpoint format are unchanged. Structural-bias
assembly uses indexed gathers and the decoder no longer retains unused hidden
states. These changes do not alter the forecast, attention mask or training recipe.

For cached training/evaluation, offload frozen prompt embeddings and the unused
visual encoder, and skip discarded DeepStack outputs:

```bash
HORIZONS=96 SEEDS=2026 bash scripts/MMTS/etth1.sh --optimize_runtime true

# Additionally replay fixed-shape CUDA graphs (uses extra graph-pool memory).
HORIZONS=96 SEEDS=2026 bash scripts/MMTS/etth1.sh \
  --optimize_runtime true --cuda_graphs true
```

The graph path requires PyTorch 2.9.1. Graphs preserve the caller's precision and dropout RNG, retain all windows and
channels, and use eager execution for differently shaped batches, including the
last incomplete batch. Captured inputs and parameters must stay on one GPU;
model parallelism is not supported by this optional path. Disable graphs if the
additional graph pool does not fit. No batch size or precision is changed
automatically. The runtime context restores module placement on exit.

Inactive DeepStack weights are retained on CPU for strict compatibility with
existing checkpoints. They are excluded from the active module's parameter
iterator while optimized execution is enabled; they still occupy CPU memory and
checkpoint storage. The vocabulary head shares the prompt embedding, so it does
not count as an additional set of parameters. Frozen active weights still count
toward total parameters.

For online forecasting, `runtime.predict_window` constructs the same complete
FFT relation grids and frozen visual features as the cache path. Supply float32
observed windows standardized using **training-split** means and standard
deviations. The returned predictions remain in standardized coordinates.

```python
import torch
from runtime import optimized_runtime, predict_window

# model: an initialized MM-TS Model with trained weights loaded and channels configured
# windows: float32 NumPy array [batch, seq_len, num_vars], standardized as above
model.eval()
with torch.inference_mode(), optimized_runtime(
    model, cached_visual=False, cuda_graphs=True, fast_vision=False
):
    prediction = predict_window(model, windows).clone()
```

Keep the context open across requests to reuse graphs. Clone graph outputs if
retaining them across subsequent calls. Use one model per worker process; the
relation/vision contexts are not intended for concurrent threads in one process.

```bash
pip install -r requirements-optimized.txt
```

The optional dependencies compile the unchanged DTW loop without fastmath.
Without Numba the same loop runs in Python. `fast_vision=True` also enables a
frozen BF16 patch-embedding extension and visual CUDA graphs. This specialization
is explicitly limited to the validated **PyTorch 2.9.1+cu126, Transformers 4.57.3,
Ada GPU** environment; other environments should keep it disabled. It needs a
C++ compiler and CUDA headers, builds on first use, and stores compiled artifacts
in PyTorch's extension cache (configurable with `TORCH_EXTENSIONS_DIR`). Build and
graph warm-up costs are separate from steady-state inference.

Run the CPU equivalence and strict-checkpoint checks without downloading weights:

```bash
python -m unittest discover -s tests -v
```

`runtime/` contains the optional execution code; `tests/` contains correctness
checks. Experiment logs, measurements, datasets, caches, checkpoints and compiled
binaries are not part of this repository.
