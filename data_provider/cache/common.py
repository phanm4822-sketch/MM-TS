"""Shared dataset splits, scaling and spectral relations for both cache backends."""

import argparse
import json
import os
import numpy as np
from utils.heatmap_render import mats_to_rgb_grid
from utils.ts_stats import compute_three_mats, fft_magnitude_features

NPZ_CACHE_VERSION = "mmts-fullfft-v5"
NPZ_CACHE_PROTOCOL = "global_standardize_fullfft_v5"


def str2bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


class StandardScalerPerChannel:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray) -> None:
        self.mean_ = data.mean(axis=0, keepdims=True)
        self.std_ = data.std(axis=0, keepdims=True)
        self.std_ = np.maximum(self.std_, self.eps)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.std_


def normalize_window_for_aux(window: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    reference = window.mean(axis=0, keepdims=True)
    stdev = np.sqrt(window.var(axis=0, keepdims=True) + eps)
    return ((window - reference) / stdev).astype(np.float32, copy=False)


def _resolve_root(root_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not root_path or root_path == ".":
        return os.path.dirname(os.path.dirname(script_dir))
    return os.path.abspath(root_path)


def _resolve_qwen_dir(root_path: str, qwen_dir: str) -> str:
    if os.path.isabs(qwen_dir):
        return qwen_dir
    cand1 = os.path.join(root_path, qwen_dir)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(os.path.dirname(root_path), qwen_dir)
    return cand2


def _load_vision_patch_size(qwen_dir: str) -> int:
    cfg_path = os.path.join(qwen_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg["vision_config"]["patch_size"])


def _set_visible_gpus(gpus: str) -> None:
    gpus = str(gpus).strip()
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def _detect_header(data_path: str) -> bool:
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline().strip()
    if not line:
        return True
    parts = [p.strip() for p in line.split(",")]
    # If all tokens are numeric, it's headerless.
    return not all(_is_number(p) for p in parts if p != "")


def _is_illness_name(name: str) -> bool:
    text = str(name or "").strip().lower()
    return ("ili" in text) or ("illness" in text)


def _infer_ett_freq(data_path: str) -> str | None:
    name = os.path.basename(data_path).lower()
    if "etth" in name:
        return "h"
    if "ettm" in name:
        return "m"
    return None


def _is_ili_dataset(data_path: str) -> bool:
    name = os.path.basename(data_path).lower()
    return _is_illness_name(name)


def _is_exchange_dataset(data_path: str) -> bool:
    return "exchange" in os.path.basename(data_path).lower()


def _uses_patchtst_overlap_split(data_path: str) -> bool:
    return _is_ili_dataset(data_path) or _is_exchange_dataset(data_path)


def _get_ett_borders(seq_len: int, freq: str) -> tuple[list[int], list[int]]:
    if freq == "h":
        month = 30 * 24
    else:
        month = 30 * 24 * 4
    train = 12 * month
    val = 4 * month
    test = 4 * month
    border1s = [0, train - seq_len, train + val - seq_len]
    border2s = [train, train + val, train + val + test]
    return border1s, border2s


def _get_patchtst_overlap_borders(
    total_len: int, seq_len: int, train_ratio: float, test_ratio: float
) -> tuple[list[int], list[int], int]:
    num_train = int(total_len * train_ratio)
    num_test = int(total_len * test_ratio)
    num_val = total_len - num_train - num_test
    border1s = [0, num_train - seq_len, total_len - num_test - seq_len]
    border2s = [num_train, num_train + num_val, total_len]
    return border1s, border2s, num_train


def build_split_ranges(args, data_path, total_len):
    ett_freq = _infer_ett_freq(data_path)
    is_ili = _is_ili_dataset(data_path)
    use_patchtst_overlap = _uses_patchtst_overlap_split(data_path)
    if ett_freq is not None:
        border1s, border2s = _get_ett_borders(args.seq_len, ett_freq)
        if border2s[-1] > total_len:
            raise ValueError(
                f"ETT split exceeds data length: need {border2s[-1]} but T={total_len}"
            )
        split_ranges = list(zip(border1s, border2s))
        split_sizes = [
            end - start - (args.seq_len + args.pred_len) + 1 for start, end in split_ranges
        ]
        if min(split_sizes) <= 0:
            raise ValueError("invalid ETT split: not enough windows")
        return split_ranges, split_sizes, True, is_ili, border2s[0]

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    train_T = int(total_len * args.train_ratio)
    if use_patchtst_overlap:
        border1s, border2s, train_T = _get_patchtst_overlap_borders(
            total_len, args.seq_len, float(args.train_ratio), float(args.test_ratio)
        )
        split_ranges = list(zip(border1s, border2s))
    else:
        train_end = int(total_len * args.train_ratio)
        val_end = int(total_len * (args.train_ratio + args.val_ratio))
        train_end = max(train_end, args.seq_len + args.pred_len)
        val_end = max(val_end, train_end + args.seq_len + args.pred_len)
        if val_end > total_len:
            val_end = total_len
        split_ranges = [(0, train_end), (train_end, val_end), (val_end, total_len)]
    split_sizes = [
        max(0, end - start - (args.seq_len + args.pred_len) + 1) for start, end in split_ranges
    ]
    if min(split_sizes) <= 0:
        raise ValueError("invalid ratio split: not enough windows for one of the splits")
    return split_ranges, split_sizes, False, is_ili, train_T


def build_relation_grids(
    window,
    patch_len,
    stride,
    dtw_band=8,
    dtw_eps=1e-8,
    dtw_tau=1.0,
    cov_clip_lo=1.0,
    cov_clip_hi=99.0,
):
    """Full-window FFT magnitudes, then frequency slices in FFT-bin order.

    The p-th frequency relation is later assigned to the p-th temporal block.
    This is an index correspondence, not a time-local Fourier transform.
    """
    spectrum = fft_magnitude_features(normalize_window_for_aux(window, eps=1e-5))
    count = (len(window) - patch_len) // stride + 1
    if count <= 0:
        raise ValueError("patch_len must not exceed the observed window")

    def relation(source, band):
        dtw, cov, pear = compute_three_mats(
            source, kind="relation", dtw_band=band, dtw_eps=dtw_eps, verbose=False
        )
        return mats_to_rgb_grid(
            dtw, cov, pear, cov_clip_lo=cov_clip_lo, cov_clip_hi=cov_clip_hi, dtw_tau=dtw_tau
        )

    image = relation(spectrum, dtw_band)
    frames = np.stack(
        [
            relation(spectrum[p * stride : p * stride + patch_len], min(dtw_band, patch_len))
            for p in range(count)
        ]
    )
    return image, frames
