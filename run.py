"""Command-line entry point for MM-TS forecasting."""

import os
from contextlib import nullcontext

from utils.config import get_args


def main():
    args = get_args()
    if args.cuda_graphs and not args.optimize_runtime:
        raise ValueError("--cuda_graphs true requires --optimize_runtime true")
    if str(args.gpus).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # Configure CUDA visibility before importing the tensor/model stack.
    import torch
    from data_provider.cache_manager import (
        _resolve_paths,
        _apply_dataset_specific_overrides,
        _ensure_dataset_cache,
    )
    from exp.exp_main import Exp_Main
    from utils.tools import seed_everything

    args = _apply_dataset_specific_overrides(_resolve_paths(args))
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.eval_only and not args.ckpt_path:
        raise ValueError("--ckpt_path is required with --eval_only true")
    args = _ensure_dataset_cache(args)
    seed_everything(args.seed)
    exp = Exp_Main(args)
    runtime = nullcontext()
    if args.optimize_runtime:
        from runtime import optimized_runtime

        runtime = optimized_runtime(
            exp.model, cached_visual=True, cuda_graphs=args.cuda_graphs
        )
    with runtime:
        if args.eval_only:
            exp.load_checkpoint(args.ckpt_path)
            exp.save_eval_metrics(exp.test(), ckpt_path=args.ckpt_path)
        else:
            exp.train()


if __name__ == "__main__":
    main()
