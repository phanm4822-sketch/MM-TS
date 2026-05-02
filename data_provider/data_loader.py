import bisect
import json
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZWindowDataset(Dataset):
    """
    Returns:
      x: [seq_len, C], y: [pred_len, C],
      img_grid: [C, C, 3], vid_grids: [num_patches, C, C, 3]
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        img_grid: np.ndarray,
        vid_grids: np.ndarray,
        indices: np.ndarray,
        split_sizes: Tuple[int, int, int],
        img_tokens: np.ndarray | None = None,
        img_token_mask: np.ndarray | None = None,
        vid_tokens: np.ndarray | None = None,
        vid_token_mask: np.ndarray | None = None,
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.img_grid = img_grid
        self.vid_grids = vid_grids
        self.img_tokens = img_tokens
        self.img_token_mask = img_token_mask
        self.vid_tokens = vid_tokens
        self.vid_token_mask = vid_token_mask
        self.indices = indices
        self.split_sizes = split_sizes
        self.num_vars = int(x.shape[2])

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        x = torch.from_numpy(self.x[i]).float()
        y = torch.from_numpy(self.y[i]).float()
        img_grid = torch.from_numpy(self.img_grid[i]).float()
        vid_grids = torch.from_numpy(self.vid_grids[i]).float()
        if self.img_tokens is not None and self.vid_tokens is not None:
            img_tokens = torch.from_numpy(self.img_tokens[i]).float()
            img_mask = torch.from_numpy(self.img_token_mask[i]).long()
            vid_tokens = torch.from_numpy(self.vid_tokens[i]).float()
            vid_mask = torch.from_numpy(self.vid_token_mask[i]).long()
            return (x, y, img_grid, vid_grids, img_tokens, img_mask, vid_tokens, vid_mask)
        return (x, y, img_grid, vid_grids)


class ElectricityShardDataset(Dataset):
    """
    Electricity-specific dataset backed by sharded .npy files described by a
    manifest. Each worker keeps a small shard-local mmap cache instead of
    materializing the full dataset.
    """

    def __init__(
        self,
        manifest_path: str,
        indices: np.ndarray,
        split_sizes: Tuple[int, int, int],
    ):
        super().__init__()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if str(manifest.get("format", "")).strip() != "electricity_sharded_v1":
            raise ValueError(f"unsupported electricity manifest format: {manifest.get('format')}")

        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        self.indices = indices
        self.split_sizes = split_sizes
        self.num_vars = int(manifest["num_vars"])
        self.columns = manifest.get("columns", None)
        self.meta = manifest
        self.shards = list(manifest["shards"])
        self._shard_starts = [int(s["start"]) for s in self.shards]
        self._cache = {}
        self._cached_shard_idx = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _resolve_file(self, rel_path: str) -> str:
        return rel_path if os.path.isabs(rel_path) else os.path.join(self.manifest_dir, rel_path)

    def _load_shard(self, shard_idx: int):
        if self._cached_shard_idx == shard_idx and self._cache:
            return self._cache

        shard = self.shards[shard_idx]
        files = shard["files"]
        cache = {
            "x": np.load(self._resolve_file(files["x"]), mmap_mode="r"),
            "y": np.load(self._resolve_file(files["y"]), mmap_mode="r"),
            "img_grid": np.load(self._resolve_file(files["img_grid"]), mmap_mode="r"),
            "vid_grids": np.load(self._resolve_file(files["vid_grids"]), mmap_mode="r"),
        }
        if "img_tokens" in files and "vid_tokens" in files:
            cache["img_tokens"] = np.load(self._resolve_file(files["img_tokens"]), mmap_mode="r")
            cache["img_token_mask"] = np.load(self._resolve_file(files["img_token_mask"]), mmap_mode="r")
            cache["vid_tokens"] = np.load(self._resolve_file(files["vid_tokens"]), mmap_mode="r")
            cache["vid_token_mask"] = np.load(self._resolve_file(files["vid_token_mask"]), mmap_mode="r")
        self._cached_shard_idx = shard_idx
        self._cache = cache
        return cache

    def __getitem__(self, idx: int):
        global_i = int(self.indices[idx])
        shard_idx = bisect.bisect_right(self._shard_starts, global_i) - 1
        if shard_idx < 0 or shard_idx >= len(self.shards):
            raise IndexError(f"global index out of shard range: {global_i}")
        shard = self.shards[shard_idx]
        local_i = global_i - int(shard["start"])
        if local_i < 0 or local_i >= int(shard["count"]):
            raise IndexError(f"local shard index out of range: global={global_i} shard={shard_idx} local={local_i}")

        cache = self._load_shard(shard_idx)
        x = torch.from_numpy(np.array(cache["x"][local_i], copy=True)).float()
        y = torch.from_numpy(np.array(cache["y"][local_i], copy=True)).float()
        img_grid = torch.from_numpy(np.array(cache["img_grid"][local_i], copy=True)).float()
        vid_grids = torch.from_numpy(np.array(cache["vid_grids"][local_i], copy=True)).float()
        if "img_tokens" in cache and "vid_tokens" in cache:
            img_tokens = torch.from_numpy(np.array(cache["img_tokens"][local_i], copy=True)).float()
            img_mask = torch.from_numpy(np.array(cache["img_token_mask"][local_i], copy=True)).long()
            vid_tokens = torch.from_numpy(np.array(cache["vid_tokens"][local_i], copy=True)).float()
            vid_mask = torch.from_numpy(np.array(cache["vid_token_mask"][local_i], copy=True)).long()
            return (x, y, img_grid, vid_grids, img_tokens, img_mask, vid_tokens, vid_mask)
        return (x, y, img_grid, vid_grids)
