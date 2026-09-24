import torch

from utils.ts_attention import TS_TOKEN_LAYOUTS


def _combine_stats(
    dtw,
    cov,
    pear,
    weights,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    w_dtw, w_cov, w_pear = weights
    dtw_t = torch.as_tensor(dtw, device=device, dtype=dtype)
    cov_t = torch.as_tensor(cov, device=device, dtype=dtype)
    pear_t = torch.as_tensor(pear, device=device, dtype=dtype)
    return dtw_t * w_dtw + cov_t * w_cov + pear_t * w_pear


def build_ts_attention_bias(
    img_stats,
    vid_stats,
    num_patches: int,
    num_vars: int,
    device: torch.device,
    dtype: torch.dtype,
    weights: tuple[float, float, float],
    token_layout: str = "channel_major",
) -> torch.Tensor:
    if token_layout not in TS_TOKEN_LAYOUTS:
        raise ValueError(f"unknown TS token layout: {token_layout}")
    img_dtw, img_cov, img_pear = img_stats
    vid_dtw, vid_cov, vid_pear = vid_stats
    img = _combine_stats(img_dtw, img_cov, img_pear, weights, device=device, dtype=dtype)
    vid = _combine_stats(vid_dtw, vid_cov, vid_pear, weights, device=device, dtype=dtype)

    if img.ndim != 3 or img.shape[-2] != num_vars or img.shape[-1] != num_vars:
        raise ValueError(f"img_stats must be [B, {num_vars}, {num_vars}], got {tuple(img.shape)}")
    if vid.ndim != 4 or vid.shape[-2] != num_vars or vid.shape[-1] != num_vars:
        raise ValueError(
            f"vid_stats must be [B, P, {num_vars}, {num_vars}], got {tuple(vid.shape)}"
        )
    if vid.shape[1] != num_patches:
        raise ValueError(f"vid_stats patches mismatch: got {vid.shape[1]}, expected {num_patches}")
    if vid.shape[0] != img.shape[0]:
        raise ValueError("img_stats and vid_stats must have the same batch size")

    # Local relations fill same-patch blocks; global relations fill the rest.
    ids = torch.arange(num_patches * num_vars, device=device)
    if token_layout == "patch_major":
        patch, channel = ids // num_vars, ids % num_vars
    else:
        patch, channel = ids % num_patches, ids // num_patches
    global_bias = img[:, channel[:, None], channel[None, :]]
    local_bias = vid[:, patch[:, None], channel[:, None], channel[None, :]]
    return torch.where((patch[:, None] == patch[None, :])[None], local_bias, global_bias)


def build_full_ts_attention_bias(
    img_stats,
    vid_stats,
    num_patches: int,
    num_vars: int,
    ts_range: tuple[int, int],
    total_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    weights: tuple[float, float, float],
    token_layout: str = "channel_major",
) -> torch.Tensor:
    img_dtw, img_cov, img_pear = img_stats
    vid_dtw, vid_cov, vid_pear = vid_stats
    ts_bias = build_ts_attention_bias(
        img_stats=(img_dtw, img_cov, img_pear),
        vid_stats=(vid_dtw, vid_cov, vid_pear),
        num_patches=num_patches,
        num_vars=num_vars,
        device=device,
        dtype=dtype,
        weights=weights,
        token_layout=token_layout,
    )
    ts_start, ts_end = ts_range
    full_bias = torch.zeros(
        (ts_bias.shape[0], total_tokens, total_tokens), device=device, dtype=dtype
    )
    full_bias[:, ts_start : ts_end + 1, ts_start : ts_end + 1] = ts_bias

    return full_bias
