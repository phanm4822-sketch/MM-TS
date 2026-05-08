import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import argparse
import json
import time
from typing import Optional
import numpy as np
import pandas as pd
import torch
import subprocess
from utils import time_block
from utils.heatmap_render import mats_to_rgb_grid
from utils.ts_stats import compute_three_mats, fft_magnitude_features

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

NPZ_CACHE_VERSION = "mmts-fullfft-v5"
NPZ_CACHE_PROTOCOL = "global_standardize_fullfft_v5"


def str2bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def _is_illness_name(name: str) -> bool:
    text = str(name or "").strip().lower()
    return ("ili" in text) or ("illness" in text)


def get_args():
    parser = argparse.ArgumentParser(description="Build time-series grid-layer cache (npz)")
    parser.add_argument("--root_path", type=str, default=".", help="root path of data file")
    parser.add_argument("--data_path", type=str, default="datasets/ETTh1.csv", help="raw csv path")
    parser.add_argument("--output_path", type=str, default=None, help="output npz path (optional)")
    parser.add_argument("--all", action="store_true", help="scan datasets/ for *.csv and build matching .npz files")
    parser.add_argument("--no_header", type=str2bool, default=False, help="treat CSV as headerless (true/false)")
    parser.add_argument("--auto_header", type=str2bool, default=True, help="auto-detect header row (true/false)")

    parser.add_argument("--precompute_vision", type=str2bool, default=True, help="precompute Qwen3-VL vision tokens (true/false)")
    parser.add_argument("--qwen_dir", type=str, default="Qwen3-VL-2B-Instruct", help="Qwen3-VL model directory")
    parser.add_argument("--gpus", type=str, default="0", help="visible GPU ids for precompute, e.g. '0,1'")
    parser.add_argument("--vision_mb", type=int, default=8, help="vision micro-batch size for encoding")
    parser.add_argument("--cache_batch_size", type=int, default=8, help="batch size for vision token precompute")
    parser.add_argument("--vision_render_size", type=int, default=64, help="minimum rendered image/video size before Qwen3-VL vision encoding")
    parser.add_argument("--vision_dp", type=str2bool, default=True, help="use DataParallel for vision precompute (true/false)")
    parser.add_argument("--max_vars", type=int, default=400, help="max variables allowed (guard against huge C)")
    parser.add_argument("--shard_id", type=int, default=0, help="shard index for vision precompute (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="total shards for vision precompute")
    parser.add_argument("--auto_shard", type=str2bool, default=True, help="auto-shard vision precompute across visible GPUs")
    parser.add_argument("--vision_only", type=str2bool, default=False, help="compute vision tokens only from base npz")
    parser.add_argument("--base_npz", type=str, default=None, help="base npz path for vision-only mode")
    parser.add_argument("--cleanup", type=str2bool, default=True, help="remove base/shard npz after merge (true/false)")

    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="patch stride")

    parser.add_argument("--train_ratio", type=float, default=0.70, help="train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="val split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="test split ratio")
    parser.add_argument("--window_profile",type=str, default="manual", choices=["manual","etth1", "etth2","ettm1","ettm2","ili","exchange","electricity","traffic","weather","solar",],help="manual or dataset-specific window preset",)
    parser.add_argument("--window_demean", type=str2bool, default=False, help="manual mode: apply per-window de-mean (legacy; true/false)")
    parser.add_argument("--window_norm", type=str2bool, default=False, help="manual mode: apply per-window std norm (legacy; true/false)")
    parser.add_argument("--log_every", type=int, default=100, help="progress log interval")
    return parser.parse_args()


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


def _parse_visible_gpus(gpus: str) -> list[str]:
    gpus = str(gpus).strip()
    if not gpus:
        return []
    return [p.strip() for p in gpus.split(",") if p.strip() != ""]


def _detect_visible_gpu_count(gpus: str) -> int:
    visible = _parse_visible_gpus(gpus)
    if visible:
        return len(visible)
    if torch.cuda.is_available():
        return int(torch.cuda.device_count())
    return 0


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


def _precompute_vision_tokens(
    img_grid: np.ndarray,
    vid_grids: np.ndarray,
    num_patches: int,
    args,
    qwen_dir: str,
):
    from models.qwen3_vl_utils import load_qwen3_vl
    from layers.modality_builder import build_image_modality_from_grids, build_video_modality_from_grids
    from layers.vision_encode import encode_vision_modalities

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA is required for vision token precompute; no GPU available.")

    patch_size = _load_vision_patch_size(qwen_dir)
    model, processor = load_qwen3_vl(qwen_dir, device_map=None)
    model = model.to(device)
    model.eval()

    n = int(img_grid.shape[0])
    cache_bs = max(1, int(getattr(args, "cache_batch_size", 8)))
    vision_mb = max(1, int(getattr(args, "vision_mb", 8)))

    img_tokens_np = None
    vid_tokens_np = None
    img_mask_np = None
    vid_mask_np = None
    embed_dim = None
    img_len = None
    vid_len = None

    shard_id = int(getattr(args, "shard_id", 0))
    num_shards = int(getattr(args, "num_shards", 1))
    if num_shards <= 0 or shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"invalid shard config: shard_id={shard_id}, num_shards={num_shards}")
    shard_indices = np.arange(shard_id, n, num_shards, dtype=np.int32)

    for s in range(0, len(shard_indices), cache_bs):
        batch_ids = shard_indices[s:s + cache_bs]
        start = int(batch_ids[0])
        end = int(batch_ids[-1]) + 1
        batch_img = img_grid[batch_ids]
        batch_vid = vid_grids[batch_ids]

        images, _ = build_image_modality_from_grids(batch_img, args=args, patch_size=patch_size)
        videos, _ = build_video_modality_from_grids(batch_vid, num_patches=num_patches, args=args, patch_size=patch_size)

        img_token_seqs, vid_token_seqs = encode_vision_modalities(
            model=model,
            processor=processor,
            images=images,
            videos=videos,
            vision_mb=vision_mb,
            device=device,
        )

        if not img_token_seqs or not vid_token_seqs:
            raise RuntimeError("vision token precompute returned empty sequences")

        if img_tokens_np is None:
            img_len = int(img_token_seqs[0].shape[0])
            vid_len = int(vid_token_seqs[0].shape[0])
            embed_dim = int(img_token_seqs[0].shape[1])
            img_tokens_np = np.empty((len(shard_indices), img_len, embed_dim), dtype=np.float16)
            vid_tokens_np = np.empty((len(shard_indices), vid_len, embed_dim), dtype=np.float16)
            img_mask_np = np.ones((len(shard_indices), img_len), dtype=np.uint8)
            vid_mask_np = np.ones((len(shard_indices), vid_len), dtype=np.uint8)

        for i, (img_tok, vid_tok) in enumerate(zip(img_token_seqs, vid_token_seqs)):
            dst = int(s + i)
            if int(img_tok.shape[0]) != img_len or int(vid_tok.shape[0]) != vid_len:
                raise RuntimeError("vision token length mismatch; cannot save fixed-shape cache")
            if int(img_tok.shape[1]) != embed_dim or int(vid_tok.shape[1]) != embed_dim:
                raise RuntimeError("vision token embed_dim mismatch; cannot save fixed-shape cache")
            img_tokens_np[dst] = img_tok.detach().cpu().to(torch.float16).numpy()
            vid_tokens_np[dst] = vid_tok.detach().cpu().to(torch.float16).numpy()

        if args.log_every > 0 and (len(shard_indices) > 0) and ((s + cache_bs) % max(1, args.log_every) == 0):
            done = min(len(shard_indices), s + cache_bs)
            print(f"[VisionCache] shard {shard_id}/{num_shards} {done}/{len(shard_indices)} samples encoded")

    return img_tokens_np, img_mask_np, vid_tokens_np, vid_mask_np, embed_dim, img_len, vid_len, shard_indices


def _precompute_vision_from_npz(base_npz: str, args):
    if not base_npz:
        raise ValueError("base_npz is required for vision-only mode")
    cache = np.load(base_npz, allow_pickle=True)
    img_grid = cache["img_grid"]
    vid_grids = cache["vid_grids"]
    meta = {}
    if "meta_json" in cache:
        try:
            meta = json.loads(str(cache["meta_json"]))
        except Exception:
            meta = {}
    num_patches = int(meta.get("num_patches", 0))
    if num_patches <= 0:
        raise ValueError("invalid num_patches in base npz meta")
    qwen_dir = _resolve_qwen_dir(args.root_path, args.qwen_dir)
    return _precompute_vision_tokens(
        img_grid=img_grid,
        vid_grids=vid_grids,
        num_patches=num_patches,
        args=args,
        qwen_dir=qwen_dir,
    )


def _iter_csv_paths(root_path: str, data_path: str, scan_all: bool) -> list[str]:
    if scan_all:
        data_dir = os.path.join(root_path, "datasets")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"cannot find datasets directory: {data_dir}")
        files = [
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if name.lower().endswith(".csv")
        ]
        if not files:
            raise FileNotFoundError(f"no .csv files found under: {data_dir}")
        return sorted(files)

    data_path = os.path.join(root_path, data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"cannot find data file: {data_path}")
    return [data_path]


def _default_output_path(csv_path: str, pred_len: int) -> str:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join("cache", f"{base}_{pred_len}.npz")


def _infer_ett_freq(data_path: str) -> Optional[str]:
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
    name = os.path.basename(data_path).lower()
    return "exchange" in name


def _uses_patchtst_overlap_split(data_path: str) -> bool:
    return _is_ili_dataset(data_path) or _is_exchange_dataset(data_path)


def _apply_dataset_specific_overrides(args, data_path: str):
    if _is_ili_dataset(data_path):
        args.seq_len = 104
    return args


WINDOW_PRESETS = {
    # Use train-fit global standardization only.
    # Reversible per-window normalization should live in the model, not in the cache.
    "etth1": {"window_norm": False, "window_demean": False},
    "etth2": {"window_norm": False, "window_demean": False},
    "ettm1": {"window_norm": False, "window_demean": False},
    "ettm2": {"window_norm": False, "window_demean": False},
    "ili": {"window_norm": False, "window_demean": False},
    "exchange": {"window_norm": False, "window_demean": False},
    "electricity": {"window_norm": False, "window_demean": False},
    "traffic": {"window_norm": False, "window_demean": False},
    "weather": {"window_norm": False, "window_demean": False},
    "solar": {"window_norm": False, "window_demean": False},
}


def _infer_window_profile_from_path(data_path: str) -> Optional[str]:
    name = os.path.basename(data_path).lower()
    if "etth1" in name:
        return "etth1"
    if "etth2" in name:
        return "etth2"
    if "ettm1" in name:
        return "ettm1"
    if "ettm2" in name:
        return "ettm2"
    if "ili" in name:
        return "ili"
    if "exchange" in name:
        return "exchange"
    if "electricity" in name:
        return "electricity"
    if "traffic" in name:
        return "traffic"
    if "weather" in name:
        return "weather"
    return None


def _resolve_window_flags(args, data_path: str) -> tuple[bool, bool, dict]:
    profile = str(getattr(args, "window_profile", "manual")).strip().lower()
    if profile == "manual":
        wn = bool(getattr(args, "window_norm", False))
        wd = bool(getattr(args, "window_demean", False)) or wn
        return wn, wd, {"source": "manual", "profile": "manual", "window_norm": wn, "window_demean": wd}

    if profile not in WINDOW_PRESETS:
        raise ValueError(f"unknown window_profile: {profile}")

    preset = WINDOW_PRESETS[profile]
    wn = bool(preset["window_norm"])
    wd = bool(preset["window_demean"]) or wn
    inferred = _infer_window_profile_from_path(data_path)
    return wn, wd, {
        "source": "preset",
        "profile": profile,
        "inferred_dataset": inferred,
        "window_norm": wn,
        "window_demean": wd,
    }


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


def _get_patchtst_overlap_borders(total_len: int, seq_len: int, train_ratio: float, test_ratio: float) -> tuple[list[int], list[int], int]:
    num_train = int(total_len * train_ratio)
    num_test = int(total_len * test_ratio)
    num_val = total_len - num_train - num_test
    border1s = [0, num_train - seq_len, total_len - num_test - seq_len]
    border2s = [num_train, num_train + num_val, total_len]
    return border1s, border2s, num_train


def _auto_window_flags(
    data_norm: np.ndarray,
    seq_len: int,
    pred_len: int,
    max_samples: int = 500,
) -> tuple[bool, bool, dict]:
    """
    Decide per-window normalization/centering based on standardized data.
    Returns: (window_norm, window_demean, stats)
    """
    n = int(data_norm.shape[0] - (seq_len + pred_len) + 1)
    if n <= 0:
        raise ValueError("time series too short for seq_len + pred_len")

    k = min(max_samples, n)
    if k == n:
        indices = np.arange(n, dtype=np.int32)
    else:
        indices = np.linspace(0, n - 1, k, dtype=np.int32)

    means = []
    stds = []
    for i in indices:
        win = data_norm[i:i + seq_len]
        means.append(win.mean(axis=0))
        stds.append(win.std(axis=0))

    means = np.stack(means, axis=0)
    stds = np.stack(stds, axis=0)

    mean_std = means.std(axis=0)
    std_mean = stds.mean(axis=0)
    std_cv = stds.std(axis=0) / np.maximum(std_mean, 1e-6)

    median_mean_std = float(np.median(mean_std))
    median_std_cv = float(np.median(std_cv))

    mean_std_hi = 0.80
    mean_std_lo = 0.60
    std_cv_hi = 0.30
    std_cv_lo = 0.20

    # Default decision
    window_norm = median_std_cv >= std_cv_hi
    window_demean = window_norm or (median_mean_std >= mean_std_hi)

    # Gray-zone fallback: when metrics are uncertain, only de-mean (safer than full norm).
    gray_zone = (
        (mean_std_lo <= median_mean_std < mean_std_hi)
        or (std_cv_lo <= median_std_cv < std_cv_hi)
    )
    if gray_zone and not window_norm:
        window_demean = True

    stats = {
        "median_mean_std": median_mean_std,
        "median_std_cv": median_std_cv,
        "samples": int(k),
        "threshold_mean_std_hi": mean_std_hi,
        "threshold_mean_std_lo": mean_std_lo,
        "threshold_std_cv_hi": std_cv_hi,
        "threshold_std_cv_lo": std_cv_lo,
        "gray_zone": bool(gray_zone),
    }
    return window_norm, window_demean, stats


def build_single_npz(args, data_path: str, output_path: Optional[str]) -> None:
    if bool(getattr(args, "vision_only", False)):
        (
            img_tokens,
            img_token_mask,
            vid_tokens,
            vid_token_mask,
            embed_dim,
            img_len,
            vid_len,
            shard_indices,
        ) = _precompute_vision_from_npz(getattr(args, "base_npz", None), args)
        meta = {
            "precomputed_vision": True,
            "vision_model_dir": str(_resolve_qwen_dir(args.root_path, args.qwen_dir)),
            "vision_patch_size": int(_load_vision_patch_size(_resolve_qwen_dir(args.root_path, args.qwen_dir))),
            "vision_render_size": int(getattr(args, "vision_render_size", 64)),
            "vision_embed_dim": int(embed_dim),
            "vision_img_tokens": int(img_len),
            "vision_vid_tokens": int(vid_len),
            "vision_shard_id": int(getattr(args, "shard_id", 0)),
            "vision_num_shards": int(getattr(args, "num_shards", 1)),
        }
        out_path = os.path.abspath(output_path)
        with time_block("npz.save"):
            np.savez_compressed(
                out_path,
                img_tokens=img_tokens,
                img_token_mask=img_token_mask,
                vid_tokens=vid_tokens,
                vid_token_mask=vid_token_mask,
                shard_indices=shard_indices,
                meta_json=json.dumps(meta, ensure_ascii=True),
            )
        print(f"[Done] saved vision shard -> {out_path}")
        return
    if output_path is None or str(output_path).strip() == "":
        output_path = _default_output_path(data_path, pred_len=int(args.pred_len))

    with time_block("npz.load_csv"):
        use_no_header = bool(getattr(args, "no_header", False))
        if not use_no_header and bool(getattr(args, "auto_header", True)):
            has_header = _detect_header(data_path)
            use_no_header = not has_header
        df = pd.read_csv(data_path, header=None if use_no_header else "infer")
    if "date" in df.columns:
        df_feat = df.drop(columns=["date"])
    else:
        df_feat = df

    data = df_feat.values.astype(np.float32)
    T, C = data.shape
    max_vars = int(getattr(args, "max_vars", 0))
    if max_vars > 0 and C > max_vars:
        raise ValueError(f"too many variables C={C} exceeds max_vars={max_vars}")
    ett_freq = _infer_ett_freq(data_path)
    is_ili = _is_ili_dataset(data_path)
    use_patchtst_overlap = _uses_patchtst_overlap_split(data_path)
    use_ett_split = ett_freq is not None
    if use_ett_split:
        border1s, border2s = _get_ett_borders(args.seq_len, ett_freq)
        if border2s[-1] > T:
            raise ValueError(f"ETT split exceeds data length: need {border2s[-1]} but T={T}")
        split_ranges = list(zip(border1s, border2s))
        train_T = border2s[0]
    else:
        split_ranges = [(0, T)]
        train_T = int(T * args.train_ratio)

    with time_block("npz.standardize"):
        scaler = StandardScalerPerChannel()
        scaler.fit(data[:train_T])
        data_norm = scaler.transform(data)
        mean = scaler.mean_.astype(np.float32)
        std = scaler.std_.astype(np.float32)
        data_min = data[:train_T].min(axis=0, keepdims=True).astype(np.float32)
        data_max = data[:train_T].max(axis=0, keepdims=True).astype(np.float32)

    num_patches = (args.seq_len - args.patch_len) // args.stride + 1
    if num_patches <= 0:
        raise ValueError("invalid patch params for seq_len/patch_len/stride")

    def _num_windows(start: int, end: int) -> int:
        length = end - start
        return max(0, length - (args.seq_len + args.pred_len) + 1)

    if len(split_ranges) == 1:
        train_ratio = float(args.train_ratio)
        val_ratio = float(args.val_ratio)
        test_ratio = float(args.test_ratio)
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0 for strict time split")
        if use_patchtst_overlap:
            border1s, border2s, train_T = _get_patchtst_overlap_borders(T, args.seq_len, train_ratio, test_ratio)
            split_ranges = list(zip(border1s, border2s))
        else:
            # Strict time split by ratio on the raw timeline (no overlap across splits).
            train_end = int(T * train_ratio)
            val_end = int(T * (train_ratio + val_ratio))
            train_end = max(train_end, args.seq_len + args.pred_len)
            val_end = max(val_end, train_end + args.seq_len + args.pred_len)
            if val_end > T:
                val_end = T
            split_ranges = [(0, train_end), (train_end, val_end), (val_end, T)]
        n_train = _num_windows(*split_ranges[0])
        n_val = _num_windows(*split_ranges[1])
        n_test = _num_windows(*split_ranges[2])
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError("invalid ratio split: not enough windows for one of the splits")
        split_sizes = [n_train, n_val, n_test]
        n = n_train + n_val + n_test
    else:
        n_train = _num_windows(*split_ranges[0])
        n_val = _num_windows(*split_ranges[1])
        n_test = _num_windows(*split_ranges[2])
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError("invalid ETT split: not enough windows")
        split_sizes = [n_train, n_val, n_test]
        n = n_train + n_val + n_test

    x_all = np.empty((n, args.seq_len, C), dtype=np.float32)
    y_all = np.empty((n, args.pred_len, C), dtype=np.float32)
    img_grid = np.empty((n, C, C, 3), dtype=np.float32)
    vid_grids = np.empty((n, num_patches, C, C, 3), dtype=np.float32)

    # Fixed stats config to reduce CLI surface area.
    dtw_band = 8
    dtw_eps = 1e-8
    dtw_tau = 1.0
    cov_clip_lo = 1.0
    cov_clip_hi = 99.0
    window_norm_eps = 1e-3
    with time_block("npz.window_mode"):
        window_norm, window_demean, auto_stats = _resolve_window_flags(args, data_path)
        print(
            "[WindowMode] "
            f"source={auto_stats['source']} profile={auto_stats.get('profile', 'na')} "
            f"window_norm={window_norm} window_demean={window_demean}"
        )
        if window_norm or window_demean:
            raise ValueError(
                "legacy window_norm/window_demean is no longer supported. "
                "Use the model-side normalizer instead."
            )

    t0 = time.time()
    with time_block("npz.build"):
        idx = 0
        for start, end in split_ranges:
            max_i = end - (args.seq_len + args.pred_len) + 1
            for i in range(start, max_i):
                x = data_norm[i:i + args.seq_len]
                y = data_norm[i + args.seq_len:i + args.seq_len + args.pred_len]

                x_all[idx] = x
                y_all[idx] = y
                x_aux = normalize_window_for_aux(x, eps=1e-5)
                rel_source = fft_magnitude_features(x_aux)

                dtw, cov, pear = compute_three_mats(
                    rel_source,
                    kind=f"Img sample{idx}",
                    dtw_band=dtw_band,
                    dtw_eps=dtw_eps,
                    verbose=(idx == 0),
                )
                img_grid[idx] = mats_to_rgb_grid(
                    dtw,
                    cov,
                    pear,
                    cov_clip_lo=cov_clip_lo,
                    cov_clip_hi=cov_clip_hi,
                    dtw_tau=dtw_tau,
                )

                for p in range(num_patches):
                    start_p = p * args.stride
                    patch = rel_source[start_p:start_p + args.patch_len]
                    dtw_p, cov_p, pear_p = compute_three_mats(
                        patch,
                        kind=f"Vid sample{idx} patch{p}",
                        dtw_band=min(dtw_band, args.patch_len),
                        dtw_eps=dtw_eps,
                        verbose=(idx == 0 and p == 0),
                    )
                    vid_grids[idx, p] = mats_to_rgb_grid(
                        dtw_p,
                        cov_p,
                        pear_p,
                        cov_clip_lo=cov_clip_lo,
                        cov_clip_hi=cov_clip_hi,
                        dtw_tau=dtw_tau,
                    )

                idx += 1
                if args.log_every > 0 and idx % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[Progress] {idx}/{n} windows processed ({elapsed:.1f}s)")

    meta = {
        "npz_cache_version": NPZ_CACHE_VERSION,
        "cache_protocol": NPZ_CACHE_PROTOCOL,
        "source_csv": os.path.abspath(data_path),
        "source_dataset_name": os.path.splitext(os.path.basename(data_path))[0],
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "patch_len": args.patch_len,
        "stride": args.stride,
        "num_vars": C,
        "num_patches": num_patches,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "dtw_band": dtw_band,
        "dtw_eps": dtw_eps,
        "dtw_tau": dtw_tau,
        "cov_clip_lo": cov_clip_lo,
        "cov_clip_hi": cov_clip_hi,
        "fft_transform": "full",
        "video_patch_source": "full_fft_magnitude",
        "vision_render_size": int(getattr(args, "vision_render_size", 64)),
        "columns": list(df_feat.columns),
        "window_norm": window_norm,
        "window_demean": window_demean,
        "legacy_window_transform": bool(window_norm or window_demean),
        "aux_window_standardize": True,
        "auto_window": auto_stats,
        "split_sizes": split_sizes,
        "split_method": (
            "ett_12_4_4"
            if use_ett_split
            else (
                "ili_patchtst_7_1_2_overlap"
                if is_ili
                else ("exchange_patchtst_7_1_2_overlap" if use_patchtst_overlap else "ratio")
            )
        ),
    }

    img_tokens = None
    img_token_mask = None
    vid_tokens = None
    vid_token_mask = None
    if bool(getattr(args, "precompute_vision", False)):
        qwen_dir = _resolve_qwen_dir(args.root_path, args.qwen_dir)
        _set_visible_gpus(getattr(args, "gpus", ""))
        print(f"[VisionCache] using model: {qwen_dir}")
        with time_block("npz.precompute_vision"):
            (
                img_tokens,
                img_token_mask,
                vid_tokens,
                vid_token_mask,
                embed_dim,
                img_len,
                vid_len,
                shard_indices,
            ) = _precompute_vision_tokens(
                img_grid=img_grid,
                vid_grids=vid_grids,
                num_patches=num_patches,
                args=args,
                qwen_dir=qwen_dir,
            )
        meta.update(
            {
                "precomputed_vision": True,
                "vision_model_dir": str(qwen_dir),
                "vision_patch_size": int(_load_vision_patch_size(qwen_dir)),
                "vision_embed_dim": int(embed_dim),
                "vision_img_tokens": int(img_len),
                "vision_vid_tokens": int(vid_len),
                "vision_shard_id": int(getattr(args, "shard_id", 0)),
                "vision_num_shards": int(getattr(args, "num_shards", 1)),
            }
        )

    if not os.path.isabs(output_path):
        out_path = os.path.join(args.root_path, output_path)
    else:
        out_path = output_path
    out_path = os.path.abspath(out_path)
    with time_block("npz.save"):
        if bool(getattr(args, "precompute_vision", False)):
            shard_id = int(getattr(args, "shard_id", 0))
            num_shards = int(getattr(args, "num_shards", 1))
            if num_shards > 1:
                suffix = f".shard{shard_id}_of_{num_shards}.npz"
                if out_path.lower().endswith(".npz"):
                    out_path = out_path[:-4] + suffix
                else:
                    out_path = out_path + suffix
        save_kwargs = dict(
            x=x_all,
            y=y_all,
            img_grid=img_grid,
            vid_grids=vid_grids,
            mean=mean,
            std=std,
            data_min=data_min,
            data_max=data_max,
            meta_json=json.dumps(meta, ensure_ascii=True),
        )
        if img_tokens is not None:
            save_kwargs["img_tokens"] = img_tokens
            save_kwargs["img_token_mask"] = img_token_mask
        if vid_tokens is not None:
            save_kwargs["vid_tokens"] = vid_tokens
            save_kwargs["vid_token_mask"] = vid_token_mask
        np.savez_compressed(out_path, **save_kwargs)
    print(f"[Done] saved cache -> {out_path}")
    print(f"[AutoWindow] final window_norm={window_norm} window_demean={window_demean}")


def main():
    args = get_args()
    args.root_path = _resolve_root(args.root_path)
    total_start = time.perf_counter()
    if bool(getattr(args, "vision_only", False)):
        if not args.output_path:
            raise ValueError("--output_path is required for vision_only mode")
        build_single_npz(args, data_path="", output_path=args.output_path)
        total_sec = time.perf_counter() - total_start
        print(f"[Total] standard cache build time: {total_sec:.1f}s ({total_sec/60.0:.2f} min)")
        return

    if bool(getattr(args, "precompute_vision", False)) and bool(getattr(args, "auto_shard", True)):
        if int(getattr(args, "num_shards", 1)) <= 1:
            count = _detect_visible_gpu_count(getattr(args, "gpus", ""))
            if count > 1:
                args.num_shards = count

    csv_paths = _iter_csv_paths(args.root_path, args.data_path, args.all)
    if len(csv_paths) > 1 and args.output_path:
        raise ValueError("--output_path only applies to single-file mode; omit it when using --all")

    for idx, data_path in enumerate(csv_paths, start=1):
        if len(csv_paths) > 1:
            print(f"\n[Batch] {idx}/{len(csv_paths)}: {data_path}")
        run_args = argparse.Namespace(**vars(args))
        _apply_dataset_specific_overrides(run_args, data_path)
        if bool(getattr(args, "precompute_vision", False)) and int(getattr(args, "num_shards", 1)) > 1:
            # Orchestrate multi-GPU vision precompute with shards.
            base_path = os.path.splitext(
                os.path.abspath(run_args.output_path or _default_output_path(data_path, pred_len=int(run_args.pred_len)))
            )[0]
            base_npz = base_path + ".base.npz"
            base_args = argparse.Namespace(**vars(run_args))
            base_args.precompute_vision = False
            print(f"[VisionShard] building base npz -> {base_npz}")
            build_single_npz(base_args, data_path=data_path, output_path=base_npz)

            gpus = _parse_visible_gpus(getattr(run_args, "gpus", ""))
            if not gpus:
                gpus = [str(i) for i in range(max(1, _detect_visible_gpu_count(getattr(run_args, "gpus", ""))))]
            shard_outs = []
            procs = []
            for shard_id in range(int(run_args.num_shards)):
                shard_npz = f"{base_path}.shard{shard_id}_of_{run_args.num_shards}.npz"
                shard_outs.append(shard_npz)
                env = os.environ.copy()
                if gpus:
                    env["CUDA_VISIBLE_DEVICES"] = gpus[shard_id % len(gpus)]
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--vision_only", "true",
                    "--base_npz", base_npz,
                    "--output_path", shard_npz,
                    "--qwen_dir", run_args.qwen_dir,
                    "--gpus", env.get("CUDA_VISIBLE_DEVICES", ""),
                    "--vision_mb", str(run_args.vision_mb),
                    "--cache_batch_size", str(run_args.cache_batch_size),
                    "--shard_id", str(shard_id),
                    "--num_shards", str(run_args.num_shards),
                ]
                print(f"[VisionShard] launch shard {shard_id}/{run_args.num_shards} on GPU {env.get('CUDA_VISIBLE_DEVICES','')}")
                procs.append(subprocess.Popen(cmd, env=env))

            for shard_id, proc in enumerate(procs):
                ret = proc.wait()
                if ret != 0:
                    raise RuntimeError(f"vision shard {shard_id} failed with code {ret}")

            # Merge shards into final npz
            out_path = os.path.abspath(run_args.output_path or _default_output_path(data_path, pred_len=int(run_args.pred_len)))
            print(f"[VisionShard] merging shards -> {out_path}")
            base = np.load(base_npz, allow_pickle=True)
            img_tokens = None
            vid_tokens = None
            img_mask = None
            vid_mask = None
            for shard_npz in shard_outs:
                shard = np.load(shard_npz, allow_pickle=True)
                idx = shard["shard_indices"]
                if img_tokens is None:
                    img_tokens = np.zeros(
                        (base["img_grid"].shape[0],) + shard["img_tokens"].shape[1:],
                        dtype=shard["img_tokens"].dtype,
                    )
                    vid_tokens = np.zeros(
                        (base["img_grid"].shape[0],) + shard["vid_tokens"].shape[1:],
                        dtype=shard["vid_tokens"].dtype,
                    )
                    img_mask = np.ones((base["img_grid"].shape[0], shard["img_token_mask"].shape[1]), dtype=shard["img_token_mask"].dtype)
                    vid_mask = np.ones((base["img_grid"].shape[0], shard["vid_token_mask"].shape[1]), dtype=shard["vid_token_mask"].dtype)
                img_tokens[idx] = shard["img_tokens"]
                vid_tokens[idx] = shard["vid_tokens"]

            meta = {}
            if "meta_json" in base:
                try:
                    meta = json.loads(str(base["meta_json"]))
                except Exception:
                    meta = {}
            meta.update(
                {
                    "precomputed_vision": True,
                    "vision_model_dir": str(_resolve_qwen_dir(run_args.root_path, run_args.qwen_dir)),
                    "vision_patch_size": int(_load_vision_patch_size(_resolve_qwen_dir(run_args.root_path, run_args.qwen_dir))),
                    "vision_render_size": int(getattr(run_args, "vision_render_size", 64)),
                    "vision_embed_dim": int(img_tokens.shape[-1]),
                    "vision_img_tokens": int(img_tokens.shape[1]),
                    "vision_vid_tokens": int(vid_tokens.shape[1]),
                    "vision_num_shards": int(run_args.num_shards),
                }
            )
            with time_block("npz.save"):
                np.savez_compressed(
                    out_path,
                    x=base["x"],
                    y=base["y"],
                    img_grid=base["img_grid"],
                    vid_grids=base["vid_grids"],
                    mean=base["mean"],
                    std=base["std"],
                    img_tokens=img_tokens,
                    img_token_mask=img_mask,
                    vid_tokens=vid_tokens,
                    vid_token_mask=vid_mask,
                    meta_json=json.dumps(meta, ensure_ascii=True),
                )
            print(f"[Done] saved cache -> {out_path}")
            if bool(getattr(run_args, "cleanup", True)):
                try:
                    os.remove(base_npz)
                except Exception:
                    pass
                for shard_npz in shard_outs:
                    try:
                        os.remove(shard_npz)
                    except Exception:
                        pass
        else:
            build_single_npz(run_args, data_path=data_path, output_path=run_args.output_path)

    total_sec = time.perf_counter() - total_start
    print(f"[Total] standard cache build time: {total_sec:.1f}s ({total_sec/60.0:.2f} min)")


if __name__ == "__main__":
    main()
