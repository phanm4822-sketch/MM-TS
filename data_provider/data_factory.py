import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_loader import ElectricityShardDataset, NPZWindowDataset
from utils import seed_worker

EXPECTED_CACHE_PROTOCOL = "global_standardize_only_v4"


def _load_npz_meta(cache) -> dict:
    meta = {}
    if "meta_json" in cache:
        try:
            meta = json.loads(str(cache["meta_json"]))
        except Exception:
            meta = {}
    return meta


def _resolve_split_indices(n: int, args, flag: str, meta: dict):
    train_ratio = float(getattr(args, "train_ratio", 0.7))
    val_ratio = float(getattr(args, "val_ratio", 0.1))
    few_shot_ratio = float(getattr(args, "few_shot_ratio", 1.0))
    if not (0.0 < few_shot_ratio <= 1.0):
        raise ValueError(f"few_shot_ratio must be in (0, 1], got: {few_shot_ratio}")
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = np.arange(0, n_train, dtype=np.int32)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int32)
    test_idx = np.arange(n_train + n_val, n, dtype=np.int32)

    if "split_sizes" in meta and isinstance(meta.get("split_sizes"), (list, tuple)):
        split_sizes = tuple(int(x) for x in meta["split_sizes"])
        n_train, n_val, n_test = split_sizes
        train_idx = np.arange(0, n_train, dtype=np.int32)
        val_idx = np.arange(n_train, n_train + n_val, dtype=np.int32)
        test_idx = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int32)
        if test_idx.shape[0] != n_test:
            raise ValueError("cache meta split_sizes mismatch with computed indices")

    if few_shot_ratio < 1.0 and train_idx.shape[0] > 0:
        keep = max(1, int(np.ceil(train_idx.shape[0] * few_shot_ratio)))
        if keep < train_idx.shape[0]:
            seed = int(getattr(args, "seed", 42))
            rng = np.random.default_rng(seed)
            # Match TimeVLM-style few-shot runs: sample a random subset from the
            # original train split while keeping val/test intact.
            perm = rng.permutation(train_idx.shape[0])[:keep]
            train_idx = train_idx[perm]

    if flag == "train":
        indices = train_idx
    elif flag == "val":
        indices = val_idx
    elif flag == "test":
        indices = test_idx
    else:
        raise ValueError("split must be train/val/test")
    return indices, (int(train_idx.shape[0]), int(val_idx.shape[0]), int(test_idx.shape[0]))


def _validate_meta(meta: dict, args, num_vars: int):
    if not meta:
        return
    cache_protocol = str(meta.get("cache_protocol", "") or "").strip()
    if cache_protocol != EXPECTED_CACHE_PROTOCOL:
        raise ValueError(
            f"unsupported cache protocol: {cache_protocol or 'missing'}. "
            f"Regenerate the cache with protocol={EXPECTED_CACHE_PROTOCOL}."
        )
    legacy_window_transform = bool(meta.get("window_norm", False) or meta.get("window_demean", False))
    if legacy_window_transform:
        raise ValueError(
            "cache meta indicates legacy per-window normalization is baked into x/y. "
            "Regenerate the cache with the updated npz builder."
        )
    mismatches = []
    if "seq_len" in meta and int(meta["seq_len"]) != int(args.seq_len):
        mismatches.append(f"seq_len cache={meta['seq_len']} args={args.seq_len}")
    if "pred_len" in meta and int(meta["pred_len"]) != int(args.pred_len):
        mismatches.append(f"pred_len cache={meta['pred_len']} args={args.pred_len}")
    if "patch_len" in meta and int(meta["patch_len"]) != int(args.patch_len):
        mismatches.append(f"patch_len cache={meta['patch_len']} args={args.patch_len}")
    if "stride" in meta and int(meta["stride"]) != int(args.stride):
        mismatches.append(f"stride cache={meta['stride']} args={args.stride}")
    if "num_vars" in meta and int(meta["num_vars"]) != int(num_vars):
        mismatches.append(f"num_vars cache={meta['num_vars']} data={num_vars}")
    if mismatches:
        raise ValueError(f"cache meta mismatch: {'; '.join(mismatches)}")


def is_electricity_manifest(data_path: str) -> bool:
    if not str(data_path).lower().endswith(".json"):
        return False
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    return str(meta.get("format", "")).strip() == "electricity_sharded_v1"


def load_cache_metadata(data_path: str) -> dict:
    if is_electricity_manifest(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "meta": meta,
            "mean": np.asarray(meta.get("mean", []), dtype=np.float32) if "mean" in meta else None,
            "std": np.asarray(meta.get("std", []), dtype=np.float32) if "std" in meta else None,
            "data_min": np.asarray(meta.get("data_min", []), dtype=np.float32) if "data_min" in meta else None,
            "data_max": np.asarray(meta.get("data_max", []), dtype=np.float32) if "data_max" in meta else None,
        }

    if not str(data_path).lower().endswith(".npz"):
        raise ValueError(f"unsupported cache path: {data_path}")
    cache = np.load(data_path, allow_pickle=True)
    meta = _load_npz_meta(cache)
    return {
        "meta": meta,
        "mean": cache["mean"] if "mean" in cache else None,
        "std": cache["std"] if "std" in cache else None,
        "data_min": cache["data_min"] if "data_min" in cache else None,
        "data_max": cache["data_max"] if "data_max" in cache else None,
    }


def data_provider(args, flag: str):
    data_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"cannot find data file: {data_path}")

    if is_electricity_manifest(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        n = int(meta["num_samples"])
        args.num_vars = int(meta["num_vars"])
        _validate_meta(meta, args, num_vars=args.num_vars)
        indices, split_sizes = _resolve_split_indices(n=n, args=args, flag=flag, meta=meta)
        data_set = ElectricityShardDataset(
            manifest_path=data_path,
            indices=indices,
            split_sizes=split_sizes,
        )
        data_set.columns = meta.get("columns", None)
    else:
        if not data_path.endswith(".npz"):
            raise ValueError(f"only .npz or electricity manifest are supported, got: {data_path}")

        cache = np.load(data_path, allow_pickle=True)
        x_all = cache["x"]
        y_all = cache["y"]
        img_grid = cache["img_grid"]
        vid_grids = cache["vid_grids"]
        img_tokens = cache["img_tokens"] if "img_tokens" in cache else None
        img_token_mask = cache["img_token_mask"] if "img_token_mask" in cache else None
        vid_tokens = cache["vid_tokens"] if "vid_tokens" in cache else None
        vid_token_mask = cache["vid_token_mask"] if "vid_token_mask" in cache else None
        n = int(x_all.shape[0])
        args.num_vars = int(x_all.shape[2])
        meta = _load_npz_meta(cache)
        _validate_meta(meta, args, num_vars=args.num_vars)
        indices, split_sizes = _resolve_split_indices(n=n, args=args, flag=flag, meta=meta)
        data_set = NPZWindowDataset(
            x=x_all,
            y=y_all,
            img_grid=img_grid,
            vid_grids=vid_grids,
            img_tokens=img_tokens,
            img_token_mask=img_token_mask,
            vid_tokens=vid_tokens,
            vid_token_mask=vid_token_mask,
            indices=indices,
            split_sizes=split_sizes,
        )
        data_set.columns = meta.get("columns", None)

    shuffle_flag = flag == "train"
    drop_last = flag == "train"
    generator = None
    seed = getattr(args, "seed", None)
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        num_workers=getattr(args, "num_workers", 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return data_set, data_loader
