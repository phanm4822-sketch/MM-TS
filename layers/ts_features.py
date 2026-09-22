"""Temporal patch embedding and a shared flatten/linear forecasting head."""

import torch
from utils.ts_utils import patchify_ts
from utils.ts_attention import reorder_ts_tokens


def encode_ts_embeddings(
    x: torch.Tensor,
    ts_mlp,
    patch_len: int,
    stride: int,
    device: torch.device,
    use_grad: bool,
):
    x_dev = x.to(device)
    ts_tokens = patchify_ts(x_dev, patch_len=patch_len, stride=stride)

    with torch.set_grad_enabled(use_grad):
        ts_embeds = ts_mlp(ts_tokens)

    return ts_embeds


def decode_ts_predictions(
    hidden_states: torch.Tensor,
    ts_range: tuple[int, int],
    num_vars: int,
    num_patches: int,
    pred_head,
    use_grad: bool,
    token_layout: str = "channel_major",
) -> torch.Tensor:
    ts_start, ts_end = ts_range
    ts_hidden = hidden_states[:, ts_start : ts_end + 1, :]
    bsz, ts_tokens, hdim = ts_hidden.shape
    expected = int(num_vars * num_patches)
    if ts_tokens != expected:
        raise RuntimeError(f"ts token count mismatch: got {ts_tokens}, expected {expected}")

    # Group tokens by channel before flattening the patch dimension.
    ts_hidden = reorder_ts_tokens(ts_hidden, num_vars, num_patches, token_layout, "channel_major")
    ts_view = ts_hidden.reshape(bsz, num_vars, num_patches, hdim)
    ts_pool = ts_view.reshape(bsz, num_vars, num_patches * hdim)
    with torch.set_grad_enabled(use_grad):
        pred = pred_head(ts_pool)
    return pred.permute(0, 2, 1).contiguous()
