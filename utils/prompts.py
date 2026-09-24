import torch


UNIFIED_PROMPT_VERSION = "mmts-window-v1"


def dataset_context(dataset_name: str) -> tuple[str, str]:
    name = dataset_name.lower()
    for key, domain, frequency in (
        ("etth", "electricity transformer temperature", "hourly"),
        ("ettm", "electricity transformer temperature", "15-min"),
        ("weather", "weather", "10-min"),
        ("exchange", "exchange rate", "daily"),
        ("electricity", "electricity consumption", "hourly"),
        ("ecl", "electricity consumption", "hourly"),
        ("illness", "influenza-like illness", "weekly"),
        ("ili", "influenza-like illness", "weekly"),
        ("traffic", "traffic", "hourly"),
    ):
        if key in name:
            return domain, frequency
    return "multivariate time series", "unknown"


def build_unified_prompts(x: torch.Tensor, dataset_name: str, pred_len: int) -> list[str]:
    """Build prompts from normalized input-window statistics."""
    if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError("prompt input must be [B, L, C] with nonempty L and C")
    domain, frequency = dataset_context(dataset_name)
    seq_len, num_vars = x.shape[1:]
    prompts = []
    for sample in x.detach().float().cpu():
        mean_series = sample.mean(dim=1)
        quarter = max(1, seq_len // 4)
        delta = float(mean_series[-quarter:].mean() - mean_series[:quarter].mean())
        trend = "rising" if delta > 0.05 else "falling" if delta < -0.05 else "stable"
        prompts.append(
            f"Forecast the next {pred_len} steps from {seq_len} observed steps.\n"
            f"Dataset={dataset_name}; domain={domain}; sampling={frequency}; channels={num_vars}.\n"
            f"Normalized window: min={sample.min().item():.4f}, max={sample.max().item():.4f}, "
            f"mean={sample.mean().item():.4f}, std={sample.std(unbiased=False).item():.4f}; trend={trend}.\n"
            "RGB relations: DTW similarity, covariance, Pearson correlation of spectral magnitudes."
        )
    return prompts
