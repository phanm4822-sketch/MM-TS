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

4. Training scripts are in `scripts/MMTS/`. For Weather, run from the repository root:

   ```bash
   bash scripts/MMTS/weather.sh
   ```

Hyperparameters can be adjusted in the dataset scripts. Results are saved under `runs/performance/`.
