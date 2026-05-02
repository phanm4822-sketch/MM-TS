import numpy as np
import torch


def patchify_ts(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    x: [B, L, C] -> tokens: [B, (num_patches*C), patch_len]
    """
    B, L, C = x.shape
    num_patches = (L - patch_len) // stride + 1
    if (L - patch_len) % stride != 0:
        print(f"[WARN] L={L}, patch_len={patch_len}, stride={stride} not aligned; last part truncated by formula.")

    patches = []
    for p in range(num_patches):
        start = p * stride
        patches.append(x[:, start:start + patch_len, :])

    patches = torch.stack(patches, dim=0).permute(1, 0, 2, 3).contiguous()
    patches = patches.permute(0, 3, 1, 2).contiguous()
    tokens = patches.view(B, C * num_patches, patch_len).contiguous()
    return tokens


def patchify_np(window, patch_len: int, stride: int):
    """
    window: [T, C] -> tokens: [C*num_patches, patch_len] with channel-major order.
    """
    T, C = window.shape
    num_patches = (T - patch_len) // stride + 1
    if (T - patch_len) % stride != 0:
        print(f"[WARN] T={T}, patch_len={patch_len}, stride={stride} not aligned; last part truncated.")

    tokens = np.empty((C * num_patches, patch_len), dtype=np.float32)
    idx = 0
    for c in range(C):
        for p in range(num_patches):
            start = p * stride
            tokens[idx] = window[start:start + patch_len, c]
            idx += 1
    return tokens
