import math

import torch


def _normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    centered = matrix - matrix.mean(dim=(-2, -1), keepdim=True)
    scale = centered.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return torch.tanh(centered / scale)


def _combine_stats(
    dtw,
    cov,
    pear,
    weights,
    device: torch.device,
    dtype: torch.dtype,
    norm_mode: str = "none",
) -> torch.Tensor:
    w_dtw, w_cov, w_pear = weights
    dtw_t = torch.as_tensor(dtw, device=device, dtype=dtype)
    cov_t = torch.as_tensor(cov, device=device, dtype=dtype)
    pear_t = torch.as_tensor(pear, device=device, dtype=dtype)
    if norm_mode == "per_stat_tanh":
        dtw_t = _normalize_matrix(dtw_t)
        cov_t = _normalize_matrix(cov_t)
        pear_t = _normalize_matrix(pear_t)
    combined = dtw_t * w_dtw + cov_t * w_cov + pear_t * w_pear
    if norm_mode == "final_tanh":
        combined = _normalize_matrix(combined)
    return combined


def _normalize_cross_rows(row_scores: torch.Tensor) -> torch.Tensor:
    centered = row_scores - row_scores.mean(dim=1, keepdim=True)
    scale = centered.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.tanh(centered / scale)


def build_ts_attention_bias(
    img_stats,
    vid_stats,
    num_patches: int,
    num_vars: int,
    device: torch.device,
    dtype: torch.dtype,
    weights: tuple[float, float, float],
    norm_mode: str = "none",
) -> torch.Tensor:
    img_dtw, img_cov, img_pear = img_stats
    vid_dtw, vid_cov, vid_pear = vid_stats
    img = _combine_stats(img_dtw, img_cov, img_pear, weights, device=device, dtype=dtype, norm_mode=norm_mode)
    vid = _combine_stats(vid_dtw, vid_cov, vid_pear, weights, device=device, dtype=dtype, norm_mode=norm_mode)

    if img.ndim != 3 or img.shape[-2] != num_vars or img.shape[-1] != num_vars:
        raise ValueError(f"img_stats must be [B, {num_vars}, {num_vars}], got {tuple(img.shape)}")
    if vid.ndim != 4 or vid.shape[-2] != num_vars or vid.shape[-1] != num_vars:
        raise ValueError(f"vid_stats must be [B, P, {num_vars}, {num_vars}], got {tuple(vid.shape)}")
    if vid.shape[1] != num_patches:
        raise ValueError(f"vid_stats patches mismatch: got {vid.shape[1]}, expected {num_patches}")
    if vid.shape[0] != img.shape[0]:
        raise ValueError("img_stats and vid_stats must have the same batch size")

    bsz = int(img.shape[0])
    ts_len = num_patches * num_vars
    bias = torch.empty((bsz, ts_len, ts_len), device=device, dtype=dtype)
    # Temporal tokens are flattened as channel-major: index(c, p) = c * P + p.
    var_offsets = torch.arange(num_vars, device=device) * num_patches
    decay_base = max(float(num_patches) / 3.0, 1.0)
    for i in range(num_patches):
        rows = var_offsets + i
        for j in range(num_patches):
            cols = var_offsets + j
            if i == j:
                block = vid[:, i]
            else:
                local_mix = 0.5 * (vid[:, i] + vid[:, j])
                alpha = math.exp(-abs(i - j) / decay_base)
                block = local_mix * alpha + img * (1.0 - alpha)
            bias[:, rows[:, None], cols[None, :]] = block
    return bias


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
    token_counts: dict | None = None,
    cross_vision_scale: float = 0.0,
    norm_mode: str = "none",
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
        norm_mode=norm_mode,
    )
    ts_start, ts_end = ts_range
    full_bias = torch.zeros((ts_bias.shape[0], total_tokens, total_tokens), device=device, dtype=dtype)
    full_bias[:, ts_start:ts_end + 1, ts_start:ts_end + 1] = ts_bias

    if token_counts and float(cross_vision_scale) > 0.0:
        text_len = int(token_counts.get("text", 0))
        img_len = int(token_counts.get("img", 0))
        vid_len = int(token_counts.get("vid", 0))
        if img_len > 0 or vid_len > 0:
            img = _combine_stats(img_dtw, img_cov, img_pear, weights, device=device, dtype=dtype, norm_mode=norm_mode)
            vid = _combine_stats(vid_dtw, vid_cov, vid_pear, weights, device=device, dtype=dtype, norm_mode=norm_mode)
            # Collapse variable-to-variable structure into a per-query relevance score
            # so TS queries can also attend more strongly to useful visual prefixes.
            img_row = img.mean(dim=-1).unsqueeze(-1).expand(-1, -1, num_patches).reshape(img.shape[0], -1)
            vid_row = vid.mean(dim=-1).permute(0, 2, 1).contiguous().reshape(vid.shape[0], -1)
            img_row = _normalize_cross_rows(img_row)
            vid_row = _normalize_cross_rows(vid_row)

            img_start = text_len
            img_end = img_start + img_len
            vid_start = img_end
            vid_end = vid_start + vid_len

            if img_len > 0:
                full_bias[:, ts_start:ts_end + 1, img_start:img_end] = img_row.unsqueeze(-1) * float(cross_vision_scale)
            if vid_len > 0:
                full_bias[:, ts_start:ts_end + 1, vid_start:vid_end] = vid_row.unsqueeze(-1) * float(cross_vision_scale)
    return full_bias
