import argparse
import json
import multiprocessing as mp
import os
import subprocess
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from numpy.lib.stride_tricks import sliding_window_view

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from layers.modality_builder import (  # noqa: E402
    build_image_modality_from_grids,
    build_video_modality_from_grids,
)
from models.qwen3_vl_utils import load_qwen3_vl  # noqa: E402
from layers.vision_encode import encode_vision_modalities  # noqa: E402
from data_provider.cache.standard import (  # noqa: E402
    NPZ_CACHE_VERSION,
    NPZ_CACHE_PROTOCOL,
    StandardScalerPerChannel,
    _detect_header,
    _get_ett_borders,
    _get_patchtst_overlap_borders,
    _infer_ett_freq,
    _is_ili_dataset,
    _uses_patchtst_overlap_split,
    _load_vision_patch_size,
    _resolve_qwen_dir,
    _resolve_root,
    _resolve_window_flags,
    _set_visible_gpus,
    normalize_window_for_aux,
    str2bool,
)
from utils import time_block  # noqa: E402
from utils.heatmap_render import mats_to_rgb_grid  # noqa: E402
from utils.ts_stats import compute_three_mats  # noqa: E402

BASE_DONE_MARKER = "base_complete.json"
VISION_DONE_MARKER = "vision_complete.json"


def _apply_dataset_specific_overrides(args, data_path: str):
    if _is_ili_dataset(data_path):
        args.seq_len = 104
    return args


def patchify_np_fast(window, patch_len: int, stride: int):
    t, c = window.shape
    num_patches = (t - patch_len) // stride + 1
    if (t - patch_len) % stride != 0:
        print(f"[WARN] T={t}, patch_len={patch_len}, stride={stride} not aligned; last part truncated.")
    patches = sliding_window_view(window, window_shape=patch_len, axis=0)[::stride]
    patches = np.transpose(patches, (1, 0, 2))
    tokens = patches.reshape(c * num_patches, patch_len)
    return np.ascontiguousarray(tokens, dtype=np.float32)


_FAST_DATA_NORM = None
_FAST_CFG = None
_FAST_MEMMAPS = {}


def get_args():
    parser = argparse.ArgumentParser(description="Electricity-specific sharded cache builder")
    parser.add_argument("--root_path", type=str, default=".", help="root path of data file")
    parser.add_argument("--data_path", type=str, default="datasets/electricity.csv", help="raw csv path")
    parser.add_argument("--output_path", type=str, default=None, help="output directory for sharded cache")
    parser.add_argument("--no_header", type=str2bool, default=False, help="treat CSV as headerless (true/false)")
    parser.add_argument("--auto_header", type=str2bool, default=True, help="auto-detect header row (true/false)")

    parser.add_argument("--precompute_vision", type=str2bool, default=True, help="precompute Qwen3-VL vision tokens")
    parser.add_argument("--qwen_dir", type=str, default="Qwen3-VL-2B-Instruct", help="Qwen3-VL model directory")
    parser.add_argument("--gpus", type=str, default="0", help="visible GPU ids for precompute")
    parser.add_argument("--vision_only_shard", type=str2bool, default=False, help="only precompute vision tokens for one shard")
    parser.add_argument("--shard_dir", type=str, default="", help="shard directory for --vision_only_shard")
    parser.add_argument("--vision_mb", type=int, default=8, help="vision micro-batch size for encoding")
    parser.add_argument("--cache_batch_size", type=int, default=8, help="batch size for vision token precompute")
    parser.add_argument("--vision_render_size", type=int, default=64, help="minimum rendered image/video size before Qwen3-VL vision encoding")
    parser.add_argument("--max_vars", type=int, default=400, help="max variables allowed")
    parser.add_argument("--cleanup", type=str2bool, default=True, help="remove existing output directory before rebuild")
    parser.add_argument("--resume", type=str2bool, default=False, help="resume from existing shard directory")
    parser.add_argument("--shard_size", type=int, default=32, help="samples per output shard")
    parser.add_argument("--vision_jobs_per_gpu", type=int, default=1, help="max concurrent vision shard jobs per visible GPU")
    parser.add_argument("--vision_backlog", type=int, default=16, help="max queued base-complete shards waiting for vision")

    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="patch stride")

    parser.add_argument("--train_ratio", type=float, default=0.70, help="train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="val split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="test split ratio")
    parser.add_argument(
        "--window_profile",
        type=str,
        default="manual",
        choices=["manual", "etth1", "etth2", "ettm1", "ettm2", "ili", "exchange", "electricity", "traffic", "weather", "solar"],
        help="manual or dataset-specific window preset",
    )
    parser.add_argument("--window_demean", type=str2bool, default=False, help="manual mode: apply per-window de-mean (legacy)")
    parser.add_argument("--window_norm", type=str2bool, default=False, help="manual mode: apply per-window std norm (legacy)")

    parser.add_argument("--fft_dtw", type=str2bool, default=True, help="compute DTW on FFT-domain features")
    parser.add_argument("--fft_cov", type=str2bool, default=True, help="compute covariance on FFT-domain features")
    parser.add_argument("--fft_pear", type=str2bool, default=True, help="compute Pearson on FFT-domain features")
    parser.add_argument("--fft_demean", type=str2bool, default=False, help="apply per-channel de-mean before FFT")

    parser.add_argument("--build_workers", type=int, default=8, help="parallel workers for window build")
    parser.add_argument("--build_chunk_size", type=int, default=8, help="windows per worker task chunk")
    parser.add_argument("--log_every", type=int, default=100, help="progress log interval")
    return parser.parse_args()


def _default_output_dir(data_path: str, pred_len: int) -> str:
    base = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join("cache", f"{base}_{pred_len}_cache")


def _num_windows_for_ranges(split_ranges, seq_len, pred_len):
    starts = []
    for start, end in split_ranges:
        max_i = end - (seq_len + pred_len) + 1
        starts.extend(range(start, max_i))
    return starts


def _build_split_ranges(args, data_path, total_len):
    ett_freq = _infer_ett_freq(data_path)
    is_ili = _is_ili_dataset(data_path)
    use_patchtst_overlap = _uses_patchtst_overlap_split(data_path)
    if ett_freq is not None:
        border1s, border2s = _get_ett_borders(args.seq_len, ett_freq)
        if border2s[-1] > total_len:
            raise ValueError(f"ETT split exceeds data length: need {border2s[-1]} but T={total_len}")
        split_ranges = list(zip(border1s, border2s))
        split_sizes = [end - start - (args.seq_len + args.pred_len) + 1 for start, end in split_ranges]
        if min(split_sizes) <= 0:
            raise ValueError("invalid ETT split: not enough windows")
        return split_ranges, split_sizes, True, is_ili, border2s[0]

    train_T = int(total_len * args.train_ratio)
    if use_patchtst_overlap:
        border1s, border2s, train_T = _get_patchtst_overlap_borders(total_len, args.seq_len, float(args.train_ratio), float(args.test_ratio))
        split_ranges = list(zip(border1s, border2s))
    else:
        train_end = int(total_len * args.train_ratio)
        val_end = int(total_len * (args.train_ratio + args.val_ratio))
        train_end = max(train_end, args.seq_len + args.pred_len)
        val_end = max(val_end, train_end + args.seq_len + args.pred_len)
        if val_end > total_len:
            val_end = total_len
        split_ranges = [(0, train_end), (train_end, val_end), (val_end, total_len)]
    split_sizes = [max(0, end - start - (args.seq_len + args.pred_len) + 1) for start, end in split_ranges]
    if min(split_sizes) <= 0:
        raise ValueError("invalid ratio split: not enough windows for one of the splits")
    return split_ranges, split_sizes, False, is_ili, train_T


def _create_shard_arrays(shard_dir: str, n: int, seq_len: int, pred_len: int, c: int, num_patches: int, patch_len: int):
    os.makedirs(shard_dir, exist_ok=True)
    shapes = {
        "x": (n, seq_len, c),
        "y": (n, pred_len, c),
        "img_grid": (n, c, c, 3),
        "vid_grids": (n, num_patches, c, c, 3),
    }
    info = {}
    for key, shape in shapes.items():
        path = os.path.join(shard_dir, f"{key}.npy")
        open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        info[key] = (path, np.dtype(np.float32).str, shape)
    return info


def _write_json_atomic(path: str, payload: dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


def _atomic_save_npy(path: str, array):
    tmp_path = f"{path}.tmp.npy"
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


def _safe_load_npy_shape_dtype(path: str):
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return None, None
    return tuple(arr.shape), np.dtype(arr.dtype).str


def _expected_base_shapes(expected: dict):
    count = int(expected["count"])
    seq_len = int(expected["seq_len"])
    pred_len = int(expected["pred_len"])
    num_vars = int(expected["num_vars"])
    num_patches = int(expected["num_patches"])
    return {
        "x.npy": ((count, seq_len, num_vars), np.dtype(np.float32).str),
        "y.npy": ((count, pred_len, num_vars), np.dtype(np.float32).str),
        "img_grid.npy": ((count, num_vars, num_vars, 3), np.dtype(np.float32).str),
        "vid_grids.npy": ((count, num_patches, num_vars, num_vars, 3), np.dtype(np.float32).str),
    }


def _validate_npy_files(shard_dir: str, expected_files: dict) -> bool:
    for filename, (shape, dtype_str) in expected_files.items():
        path = os.path.join(shard_dir, filename)
        got_shape, got_dtype = _safe_load_npy_shape_dtype(path)
        if got_shape != tuple(shape) or got_dtype != dtype_str:
            return False
    return True


def _base_marker_payload(shard_id: int, shard_start: int, shard_count: int, args, num_vars: int, num_patches: int) -> dict:
    return {
        "version": 1,
        "id": int(shard_id),
        "start": int(shard_start),
        "count": int(shard_count),
        "seq_len": int(args.seq_len),
        "pred_len": int(args.pred_len),
        "patch_len": int(args.patch_len),
        "stride": int(args.stride),
        "num_vars": int(num_vars),
        "num_patches": int(num_patches),
    }


def _vision_marker_payload(meta: dict, shard_count: int) -> dict:
    payload = dict(meta)
    payload.update(
        {
            "version": 1,
            "count": int(shard_count),
        }
    )
    return payload


def _init_fast_worker(cfg, memmap_info):
    global _FAST_CFG, _FAST_MEMMAPS
    _FAST_CFG = cfg
    _FAST_MEMMAPS = {}
    for key, (path, _dtype, _shape) in memmap_info.items():
        _FAST_MEMMAPS[key] = np.load(path, mmap_mode="r+")


def _process_window_chunk(task):
    out_start, starts = task
    data_norm = _FAST_DATA_NORM
    cfg = _FAST_CFG

    seq_len = cfg["seq_len"]
    pred_len = cfg["pred_len"]
    patch_len = cfg["patch_len"]
    stride = cfg["stride"]
    num_patches = cfg["num_patches"]
    window_norm = cfg["window_norm"]
    window_demean = cfg["window_demean"]
    window_norm_eps = cfg["window_norm_eps"]
    dtw_band = cfg["dtw_band"]
    dtw_eps = cfg["dtw_eps"]
    dtw_tau = cfg["dtw_tau"]
    cov_clip_lo = cfg["cov_clip_lo"]
    cov_clip_hi = cfg["cov_clip_hi"]
    fft_dtw = cfg["fft_dtw"]
    fft_cov = cfg["fft_cov"]
    fft_pear = cfg["fft_pear"]
    fft_demean = cfg["fft_demean"]

    x_mm = _FAST_MEMMAPS["x"]
    y_mm = _FAST_MEMMAPS["y"]
    img_mm = _FAST_MEMMAPS["img_grid"]
    vid_mm = _FAST_MEMMAPS["vid_grids"]

    for offset, start_idx in enumerate(starts):
        idx = out_start + offset
        x = data_norm[start_idx:start_idx + seq_len].copy()
        y = data_norm[start_idx + seq_len:start_idx + seq_len + pred_len].copy()

        x_mm[idx] = x
        y_mm[idx] = y
        x_aux = normalize_window_for_aux(x, eps=1e-5)

        dtw, cov, pear = compute_three_mats(
            x_aux,
            kind=f"Img sample{idx}",
            dtw_band=dtw_band,
            dtw_eps=dtw_eps,
            fft_dtw=fft_dtw,
            fft_cov=fft_cov,
            fft_pear=fft_pear,
            fft_demean=fft_demean,
            verbose=False,
        )
        img_mm[idx] = mats_to_rgb_grid(
            dtw,
            cov,
            pear,
            cov_clip_lo=cov_clip_lo,
            cov_clip_hi=cov_clip_hi,
            dtw_tau=dtw_tau,
        )

        for p in range(num_patches):
            patch_start = p * stride
            patch = x_aux[patch_start:patch_start + patch_len]
            dtw_p, cov_p, pear_p = compute_three_mats(
                patch,
                kind=f"Vid sample{idx} patch{p}",
                dtw_band=min(dtw_band, patch_len),
                dtw_eps=dtw_eps,
                fft_dtw=fft_dtw,
                fft_cov=fft_cov,
                fft_pear=fft_pear,
                fft_demean=fft_demean,
                verbose=False,
            )
            vid_mm[idx, p] = mats_to_rgb_grid(
                dtw_p,
                cov_p,
                pear_p,
                cov_clip_lo=cov_clip_lo,
                cov_clip_hi=cov_clip_hi,
                dtw_tau=dtw_tau,
            )

    return len(starts)


def _build_shard_arrays(data_norm, shard_starts, shard_dir: str, cfg: dict, workers: int, chunk_size: int, log_every: int):
    n = len(shard_starts)
    c = int(data_norm.shape[1])
    memmap_info = _create_shard_arrays(
        shard_dir=shard_dir,
        n=n,
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        c=c,
        num_patches=cfg["num_patches"],
        patch_len=cfg["patch_len"],
    )

    global _FAST_DATA_NORM
    _FAST_DATA_NORM = data_norm

    tasks = []
    for out_start in range(0, n, chunk_size):
        tasks.append((out_start, shard_starts[out_start:out_start + chunk_size]))

    if workers <= 1:
        _init_fast_worker(cfg, memmap_info)
        done = 0
        for task in tasks:
            done += _process_window_chunk(task)
            if log_every > 0 and done % log_every == 0:
                print(f"[ShardProgress] {done}/{n} windows processed")
    else:
        ctx = mp.get_context("fork")
        done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_fast_worker,
            initargs=(cfg, memmap_info),
        ) as pool:
            pending = {pool.submit(_process_window_chunk, task) for task in tasks}
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    done += fut.result()
                    if log_every > 0 and done % log_every == 0:
                        print(f"[ShardProgress] {done}/{n} windows processed")
    return memmap_info


def _precompute_vision_for_shard(shard_dir: str, args, num_patches: int, model, processor, patch_size: int, qwen_dir: str):
    img_grids = np.load(os.path.join(shard_dir, "img_grid.npy"), mmap_mode="r")
    vid_grids = np.load(os.path.join(shard_dir, "vid_grids.npy"), mmap_mode="r")
    images, _img_stats = build_image_modality_from_grids(
        img_grids=np.asarray(img_grids),
        args=args,
        patch_size=patch_size,
    )
    videos, _vid_stats = build_video_modality_from_grids(
        vid_grids=np.asarray(vid_grids),
        num_patches=num_patches,
        args=args,
        patch_size=patch_size,
    )
    img_token_seqs, vid_token_seqs = encode_vision_modalities(
        model=model,
        processor=processor,
        images=images,
        videos=videos,
        vision_mb=max(1, int(getattr(args, "vision_mb", 8))),
        device="cuda",
    )
    img_len = int(img_token_seqs[0].shape[0])
    vid_len = int(vid_token_seqs[0].shape[0])
    embed_dim = int(img_token_seqs[0].shape[1])
    img_tokens = np.empty((len(img_token_seqs), img_len, embed_dim), dtype=np.float16)
    vid_tokens = np.empty((len(vid_token_seqs), vid_len, embed_dim), dtype=np.float16)
    img_mask = np.ones((len(img_token_seqs), img_len), dtype=np.uint8)
    vid_mask = np.ones((len(vid_token_seqs), vid_len), dtype=np.uint8)
    for i, (img_tok, vid_tok) in enumerate(zip(img_token_seqs, vid_token_seqs)):
        img_tokens[i] = img_tok.to(torch.float16).numpy()
        vid_tokens[i] = vid_tok.to(torch.float16).numpy()
    meta = {
        "vision_patch_size": patch_size,
        "vision_embed_dim": embed_dim,
        "vision_img_tokens": img_len,
        "vision_vid_tokens": vid_len,
        "vision_model_dir": str(qwen_dir),
    }
    _atomic_save_npy(os.path.join(shard_dir, "img_tokens.npy"), img_tokens)
    _atomic_save_npy(os.path.join(shard_dir, "img_token_mask.npy"), img_mask)
    _atomic_save_npy(os.path.join(shard_dir, "vid_tokens.npy"), vid_tokens)
    _atomic_save_npy(os.path.join(shard_dir, "vid_token_mask.npy"), vid_mask)
    _write_json_atomic(os.path.join(shard_dir, "vision_meta.json"), meta)
    _write_json_atomic(os.path.join(shard_dir, VISION_DONE_MARKER), _vision_marker_payload(meta, len(img_token_seqs)))
    return meta


def _run_vision_only_for_shard(args):
    shard_dir = os.path.abspath(args.shard_dir)
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(f"shard_dir not found: {shard_dir}")
    _set_visible_gpus(getattr(args, "gpus", ""))
    qwen_dir = _resolve_qwen_dir(args.root_path, args.qwen_dir)
    patch_size = int(_load_vision_patch_size(qwen_dir))
    model, processor = load_qwen3_vl(qwen_dir, device_map=None)
    model = model.to("cuda")
    model.eval()
    try:
        num_patches = (int(args.seq_len) - int(args.patch_len)) // int(args.stride) + 1
        meta = _precompute_vision_for_shard(
            shard_dir=shard_dir,
            args=args,
            num_patches=num_patches,
            model=model,
            processor=processor,
            patch_size=patch_size,
            qwen_dir=qwen_dir,
        )
    finally:
        del model
    print(f"[VisionShardDone] shard_dir={shard_dir} meta={json.dumps(meta, ensure_ascii=True)}")


def _launch_vision_job(gpu_id: str, shard_dir: str, out_dir: str, args):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--vision_only_shard",
        "true",
        "--root_path",
        str(args.root_path),
        "--data_path",
        str(args.data_path),
        "--output_path",
        str(out_dir),
        "--qwen_dir",
        str(args.qwen_dir),
        "--gpus",
        str(gpu_id),
        "--vision_mb",
        str(args.vision_mb),
        "--cache_batch_size",
        str(args.cache_batch_size),
        "--seq_len",
        str(args.seq_len),
        "--pred_len",
        str(args.pred_len),
        "--patch_len",
        str(args.patch_len),
        "--stride",
        str(args.stride),
        "--window_profile",
        str(args.window_profile),
        "--shard_dir",
        str(shard_dir),
    ]
    print(f"[VisionDispatch] gpu={gpu_id} shard={os.path.basename(shard_dir)}")
    return subprocess.Popen(cmd), gpu_id, shard_dir


def _poll_vision_jobs(active_jobs, block: bool = False):
    first_meta = None
    while active_jobs:
        progressed = False
        still_active = []
        for proc, gpu_id, shard_dir in active_jobs:
            ret = proc.poll()
            if ret is None:
                still_active.append((proc, gpu_id, shard_dir))
                continue
            progressed = True
            if ret != 0:
                raise RuntimeError(f"vision precompute failed for {shard_dir} on GPU {gpu_id} with code {ret}")
            meta_path = os.path.join(shard_dir, "vision_meta.json")
            if os.path.exists(meta_path) and first_meta is None:
                with open(meta_path, "r", encoding="utf-8") as f:
                    first_meta = json.load(f)
            print(f"[VisionDone] gpu={gpu_id} shard={os.path.basename(shard_dir)}")
        active_jobs[:] = still_active
        if first_meta is not None or (progressed and not block):
            return first_meta
        if not block:
            return None
        time.sleep(1.0)
    return first_meta


def _dispatch_vision_precompute(out_dir: str, manifest: dict, args):
    visible = [g for g in str(getattr(args, "gpus", "")).split(",") if g.strip()]
    if not visible:
        visible = ["0"]
    jobs_per_gpu = max(1, int(getattr(args, "vision_jobs_per_gpu", 1)))
    slots = visible * jobs_per_gpu
    shard_dirs = [os.path.join(out_dir, f"shard_{int(s['id']):05d}") for s in manifest["shards"]]
    active = []
    first_meta = None

    for shard_idx, shard_dir in enumerate(shard_dirs):
        while len(active) >= len(slots):
            meta = _poll_vision_jobs(active, block=True)
            if meta is not None and first_meta is None:
                first_meta = meta
        gpu_id = slots[shard_idx % len(slots)]
        active.append(_launch_vision_job(gpu_id, shard_dir, out_dir, args))

    while active:
        meta = _poll_vision_jobs(active, block=True)
        if meta is not None and first_meta is None:
            first_meta = meta

    if first_meta is None:
        raise RuntimeError("vision precompute finished but no vision_meta.json was written")
    return first_meta


def _to_rel(path: str, root: str) -> str:
    return os.path.relpath(path, root)


def _shard_base_complete(shard_dir: str, expected: dict) -> bool:
    if not os.path.isdir(shard_dir):
        return False
    marker_path = os.path.join(shard_dir, BASE_DONE_MARKER)
    if os.path.exists(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)
        except Exception:
            return False
        for key, value in expected.items():
            if int(marker.get(key, -1)) != int(value):
                return False
    return _validate_npy_files(shard_dir, _expected_base_shapes(expected))


def _shard_vision_complete(shard_dir: str, expected_count: int) -> bool:
    if not os.path.isdir(shard_dir):
        return False
    meta_path = os.path.join(shard_dir, "vision_meta.json")
    marker_path = os.path.join(shard_dir, VISION_DONE_MARKER)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    if os.path.exists(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)
        except Exception:
            return False
        if int(marker.get("count", -1)) != int(expected_count):
            return False
    img_len = int(meta["vision_img_tokens"])
    vid_len = int(meta["vision_vid_tokens"])
    embed_dim = int(meta["vision_embed_dim"])
    expected_files = {
        "img_tokens.npy": ((expected_count, img_len, embed_dim), np.dtype(np.float16).str),
        "img_token_mask.npy": ((expected_count, img_len), np.dtype(np.uint8).str),
        "vid_tokens.npy": ((expected_count, vid_len, embed_dim), np.dtype(np.float16).str),
        "vid_token_mask.npy": ((expected_count, vid_len), np.dtype(np.uint8).str),
    }
    return _validate_npy_files(shard_dir, expected_files)


def _load_existing_vision_meta(out_dir: str):
    shard_dirs = sorted(d for d in os.listdir(out_dir) if d.startswith("shard_"))
    for shard_name in shard_dirs:
        meta_path = os.path.join(out_dir, shard_name, "vision_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _load_shard_vision_meta(shard_dir: str):
    meta_path = os.path.join(shard_dir, "vision_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_single_npz_fast(args, data_path, output_path):
    if output_path is None or str(output_path).strip() == "":
        output_path = _default_output_dir(data_path, pred_len=int(args.pred_len))

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
    total_len, c = data.shape
    max_vars = int(getattr(args, "max_vars", 0))
    if max_vars > 0 and c > max_vars:
        raise ValueError(f"too many variables C={c} exceeds max_vars={max_vars}")

    split_ranges, split_sizes, use_ett_split, is_ili, train_T = _build_split_ranges(args, data_path, total_len)
    use_patchtst_overlap = _uses_patchtst_overlap_split(data_path)
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

    window_norm, window_demean, auto_stats = _resolve_window_flags(args, data_path)
    if window_norm or window_demean:
        raise ValueError(
            "legacy window_norm/window_demean is no longer supported. "
            "Use the model-side normalizer instead."
        )
    starts = _num_windows_for_ranges(split_ranges, args.seq_len, args.pred_len)
    n = len(starts)
    out_dir = os.path.abspath(output_path if os.path.isabs(output_path) else os.path.join(args.root_path, output_path))
    manifest_path = os.path.join(out_dir, "manifest.json")

    resume = bool(getattr(args, "resume", False))
    if os.path.exists(out_dir):
        if resume:
            os.makedirs(out_dir, exist_ok=True)
        elif bool(getattr(args, "cleanup", True)):
            shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
        else:
            raise FileExistsError(f"output directory already exists: {out_dir}")
    else:
        os.makedirs(out_dir, exist_ok=True)

    cfg = {
        "seq_len": int(args.seq_len),
        "pred_len": int(args.pred_len),
        "patch_len": int(args.patch_len),
        "stride": int(args.stride),
        "num_patches": int(num_patches),
        "window_norm": bool(window_norm),
        "window_demean": bool(window_demean),
        "window_norm_eps": 1e-3,
        "dtw_band": 8,
        "dtw_eps": 1e-8,
        "dtw_tau": 1.0,
        "cov_clip_lo": 1.0,
        "cov_clip_hi": 99.0,
        "fft_dtw": bool(getattr(args, "fft_dtw", True)),
        "fft_cov": bool(getattr(args, "fft_cov", True)),
        "fft_pear": bool(getattr(args, "fft_pear", True)),
        "fft_demean": bool(getattr(args, "fft_demean", False)),
    }

    shard_size = max(1, int(getattr(args, "shard_size", 32)))
    workers = max(1, int(getattr(args, "build_workers", 1)))
    chunk_size = max(1, int(getattr(args, "build_chunk_size", 8)))
    log_every = int(getattr(args, "log_every", 100))

    manifest = {
        "format": "electricity_sharded_v1",
        "npz_cache_version": NPZ_CACHE_VERSION,
        "cache_protocol": NPZ_CACHE_PROTOCOL,
        "source_csv": os.path.abspath(data_path),
        "source_dataset_name": os.path.splitext(os.path.basename(data_path))[0],
        "seq_len": int(args.seq_len),
        "pred_len": int(args.pred_len),
        "patch_len": int(args.patch_len),
        "stride": int(args.stride),
        "num_vars": int(c),
        "num_patches": int(num_patches),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "split_sizes": [int(x) for x in split_sizes],
        "split_method": (
            "ett_12_4_4"
            if use_ett_split
            else (
                "ili_patchtst_7_1_2_overlap"
                if is_ili
                else ("exchange_patchtst_7_1_2_overlap" if use_patchtst_overlap else "ratio")
            )
        ),
        "window_norm": bool(window_norm),
        "window_demean": bool(window_demean),
        "legacy_window_transform": bool(window_norm or window_demean),
        "aux_window_standardize": True,
        "auto_window": auto_stats,
        "columns": list(df_feat.columns),
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
        "data_min": data_min.reshape(-1).tolist(),
        "data_max": data_max.reshape(-1).tolist(),
        "num_samples": int(n),
        "shard_size": int(shard_size),
        "vision_jobs_per_gpu": int(getattr(args, "vision_jobs_per_gpu", 1)),
        "vision_backlog": int(getattr(args, "vision_backlog", 16)),
        "precomputed_vision": bool(getattr(args, "precompute_vision", False)),
        "shards": [],
    }

    vision_meta = _load_existing_vision_meta(out_dir) if resume else None
    vision_visible = [g for g in str(getattr(args, "gpus", "")).split(",") if g.strip()]
    if not vision_visible:
        vision_visible = ["0"]
    jobs_per_gpu = max(1, int(getattr(args, "vision_jobs_per_gpu", 1)))
    vision_slots = vision_visible * jobs_per_gpu
    vision_backlog = max(1, int(getattr(args, "vision_backlog", 16)))
    active_vision_jobs = []
    ready_vision_queue = []
    dispatch_idx = 0
    t0 = time.time()

    def _service_vision(block: bool = False):
        nonlocal vision_meta, dispatch_idx
        while True:
            should_block = block and len(active_vision_jobs) >= len(vision_slots)
            meta = _poll_vision_jobs(active_vision_jobs, block=should_block)
            if meta is not None and vision_meta is None:
                vision_meta = meta

            dispatched = False
            while ready_vision_queue and len(active_vision_jobs) < len(vision_slots):
                queued_shard_id, queued_shard_dir, queued_count = ready_vision_queue.pop(0)
                if _shard_vision_complete(queued_shard_dir, expected_count=queued_count):
                    print(f"[VisionResume] reuse vision shard={queued_shard_id:05d}")
                    if vision_meta is None:
                        meta = _load_shard_vision_meta(queued_shard_dir)
                        if meta is not None:
                            vision_meta = meta
                    continue
                gpu_id = vision_slots[dispatch_idx % len(vision_slots)]
                dispatch_idx += 1
                active_vision_jobs.append(_launch_vision_job(gpu_id, queued_shard_dir, out_dir, args))
                dispatched = True

            if dispatched:
                block = False
                continue
            if not block:
                return
            if not active_vision_jobs:
                return

    for shard_id, shard_start in enumerate(range(0, n, shard_size)):
        if bool(getattr(args, "precompute_vision", False)):
            while len(ready_vision_queue) >= vision_backlog:
                _service_vision(block=True)
        shard_starts = starts[shard_start:shard_start + shard_size]
        shard_count = len(shard_starts)
        shard_dir = os.path.join(out_dir, f"shard_{shard_id:05d}")
        print(f"[Shard] {shard_id} start={shard_start} count={shard_count}")
        base_expected = _base_marker_payload(
            shard_id=shard_id,
            shard_start=shard_start,
            shard_count=shard_count,
            args=args,
            num_vars=c,
            num_patches=num_patches,
        )
        base_complete = _shard_base_complete(shard_dir, expected=base_expected)
        vision_complete = _shard_vision_complete(shard_dir, expected_count=shard_count)

        if base_complete:
            print(f"[ShardResume] reuse base shard={shard_id:05d}")
        else:
            if os.path.isdir(shard_dir):
                shutil.rmtree(shard_dir)
            _build_shard_arrays(
                data_norm=data_norm,
                shard_starts=shard_starts,
                shard_dir=shard_dir,
                cfg=cfg,
                workers=workers,
                chunk_size=chunk_size,
                log_every=log_every,
            )
            _write_json_atomic(os.path.join(shard_dir, BASE_DONE_MARKER), base_expected)

        shard_files = {
            "x": _to_rel(os.path.join(shard_dir, "x.npy"), out_dir),
            "y": _to_rel(os.path.join(shard_dir, "y.npy"), out_dir),
            "img_grid": _to_rel(os.path.join(shard_dir, "img_grid.npy"), out_dir),
            "vid_grids": _to_rel(os.path.join(shard_dir, "vid_grids.npy"), out_dir),
        }
        if bool(getattr(args, "precompute_vision", False)):
            shard_files.update(
                {
                    "img_tokens": _to_rel(os.path.join(shard_dir, "img_tokens.npy"), out_dir),
                    "img_token_mask": _to_rel(os.path.join(shard_dir, "img_token_mask.npy"), out_dir),
                    "vid_tokens": _to_rel(os.path.join(shard_dir, "vid_tokens.npy"), out_dir),
                    "vid_token_mask": _to_rel(os.path.join(shard_dir, "vid_token_mask.npy"), out_dir),
                }
            )
        manifest["shards"].append(
            {
                "id": int(shard_id),
                "start": int(shard_start),
                "count": int(shard_count),
                "files": shard_files,
            }
        )

        if bool(getattr(args, "precompute_vision", False)):
            if vision_complete:
                print(f"[VisionResume] reuse vision shard={shard_id:05d}")
                if vision_meta is None:
                    meta = _load_shard_vision_meta(shard_dir)
                    if meta is not None:
                        vision_meta = meta
                continue
            ready_vision_queue.append((shard_id, shard_dir, shard_count))
            _service_vision(block=False)

    if vision_meta is not None:
        manifest.update(vision_meta)

    if bool(getattr(args, "precompute_vision", False)):
        while ready_vision_queue or active_vision_jobs:
            _service_vision(block=True)
        manifest.update(vision_meta)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
    elapsed = time.time() - t0
    print(f"[Done] saved electricity shard manifest -> {manifest_path}")
    print(f"[Total] electricity shard build time: {elapsed:.1f}s ({elapsed/60.0:.2f} min)")
    return manifest_path


def main():
    args = get_args()
    args.root_path = _resolve_root(args.root_path)
    if bool(getattr(args, "vision_only_shard", False)):
        total_start = time.perf_counter()
        _run_vision_only_for_shard(args)
        total_sec = time.perf_counter() - total_start
        print(f"[Total] electricity vision-only shard time: {total_sec:.1f}s ({total_sec/60.0:.2f} min)")
        return
    data_path = os.path.join(args.root_path, args.data_path)
    _apply_dataset_specific_overrides(args, data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"cannot find data file: {data_path}")

    total_start = time.perf_counter()
    build_single_npz_fast(args, data_path=data_path, output_path=args.output_path)
    total_sec = time.perf_counter() - total_start
    print(f"[Total] electricity cache build time: {total_sec:.1f}s ({total_sec/60.0:.2f} min)")


if __name__ == "__main__":
    main()
