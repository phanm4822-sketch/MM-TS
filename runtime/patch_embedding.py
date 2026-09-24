"""BF16 patch projection for frozen Qwen3-VL visual weights."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

_EXTENSION = None


def load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import CUDA_HOME, load

        # CUDA headers from installed packages or CUDA_HOME.
        roots = {Path(p) for p in sys.path if p and Path(p).is_dir()}
        includes = {
            str(p) for root in roots for p in (root / "nvidia").glob("*/include")
        }
        includes.update(
            str(root / "triton/backends/nvidia/include")
            for root in roots
            if (root / "triton/backends/nvidia/include").is_dir()
        )
        if CUDA_HOME:
            includes.add(str(Path(CUDA_HOME) / "include"))
        libraries = sorted(
            p
            for root in roots
            for p in (root / "nvidia/cublas/lib").glob("libcublas.so.12")
        )
        flags = [
            "-L" + str(Path(torch.__file__).parent / "lib"),
            "-ltorch_cuda",
            "-lc10_cuda",
        ]
        flags += [str(libraries[0])] if libraries else ["-lcublas"]
        os.environ.setdefault("MAX_JOBS", "1")
        _EXTENSION = load(
            name="mmts_exact_patch",
            sources=[str(Path(__file__).with_name("exact_patch.cpp"))],
            extra_include_paths=sorted(includes),
            extra_cflags=["-O2"],
            extra_ldflags=flags,
            with_cuda=False,
            verbose=False,
        )
    return _EXTENSION


@contextmanager
def exact_patch_embedding(visual):
    """Replace frozen BF16 patch projection with per-patch cuBLAS GEMM."""
    import transformers

    if (
        torch.__version__ != "2.9.1+cu126"
        or transformers.__version__ != "4.57.3"
        or torch.cuda.get_device_capability() != (8, 9)
    ):
        raise RuntimeError(
            "fast_vision requires torch 2.9.1+cu126, transformers 4.57.3 and an Ada GPU"
        )
    patch = visual.patch_embed
    proj = patch.proj
    if (
        tuple(proj.kernel_size) != tuple(proj.stride)
        or tuple(proj.padding) != (0, 0, 0)
        or tuple(proj.dilation) != (1, 1, 1)
        or proj.groups != 1
        or proj.bias is None
        or proj.weight.dtype != torch.bfloat16
        or any(p.requires_grad for p in visual.parameters())
    ):
        raise ValueError(
            "fast_vision requires the frozen BF16 Qwen3-VL patch embedding"
        )
    ext = load_extension()
    original = patch.forward

    def call(x):
        if torch.is_grad_enabled():
            return original(x)
        return ext.exact_patch(
            x.to(dtype=proj.weight.dtype).contiguous().view(-1, proj.weight[0].numel()),
            proj.weight,
            proj.bias,
        )

    patch.forward = call
    try:
        yield
    finally:
        patch.forward = original
