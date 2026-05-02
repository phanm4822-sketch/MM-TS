import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    # Set CUDA math workspace before kernels initialize so repeated runs are
    # less sensitive to backend-specific reduction ordering.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def print_box(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


@contextmanager
def time_block(name: str):
    start = time.perf_counter()
    yield
