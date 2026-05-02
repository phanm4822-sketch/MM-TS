import torch
import torch.nn as nn

from utils.ts_utils import patchify_ts
from utils.tools import time_block


def _resolved_tail_k(num_patches: int, tail_k: int | None = None) -> int:
    requested = 3 if tail_k is None else int(tail_k)
    return max(1, min(int(num_patches), requested))


class PerVariablePredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        num_vars: int,
        mode: str = "linear",
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        mode = str(mode).strip().lower()
        self.num_vars = int(num_vars)
        heads = []
        for _ in range(self.num_vars):
            if mode == "mlp":
                if hidden_dim is None:
                    raise ValueError("hidden_dim is required for mode='mlp'")
                heads.append(
                    nn.Sequential(
                        nn.Linear(input_dim, int(hidden_dim)),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                        nn.Linear(int(hidden_dim), int(pred_len)),
                    )
                )
            elif mode == "linear":
                heads.append(nn.Linear(input_dim, int(pred_len)))
            else:
                raise ValueError(f"unknown per-variable prediction head mode: {mode}")
        self.heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B, V, D] input for per-variable head, got {tuple(x.shape)}")
        if x.shape[1] != self.num_vars:
            raise ValueError(f"input variable count mismatch: got {x.shape[1]}, expected {self.num_vars}")
        outputs = [head(x[:, i, :]) for i, head in enumerate(self.heads)]
        return torch.stack(outputs, dim=1)


def pooled_ts_dim(embed_dim: int, num_patches: int, pooling: str = "mean", tail_k: int | None = None) -> int:
    mode = str(pooling).strip().lower()
    if mode in {"mean", "attention", "last", "tail_mean"}:
        return int(embed_dim)
    if mode == "tail_concat":
        return int(embed_dim) * _resolved_tail_k(num_patches=num_patches, tail_k=tail_k)
    raise ValueError(f"unknown ts pooling mode: {pooling}")


def context_summary_dim(
    embed_dim: int,
    use_text: bool,
    use_vision: bool,
    mode: str = "none",
) -> int:
    mode = str(mode).strip().lower()
    dim = 0
    if mode in {"text", "all"} and use_text:
        dim += int(embed_dim)
    if mode in {"vision", "all"} and use_vision:
        dim += int(embed_dim) * 2
    return dim


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

    with time_block("ts.mlp"):
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
    pooling: str = "mean",
    tail_k: int | None = None,
    context_summary: torch.Tensor | None = None,
) -> torch.Tensor:
    ts_start, ts_end = ts_range
    ts_hidden = hidden_states[:, ts_start:ts_end + 1, :]
    bsz, ts_tokens, hdim = ts_hidden.shape
    expected = int(num_vars * num_patches)
    if ts_tokens != expected:
        raise RuntimeError(f"ts token count mismatch: got {ts_tokens}, expected {expected}")

    ts_view = ts_hidden.view(bsz, num_vars, num_patches, hdim)
    mode = str(pooling).strip().lower()
    if mode == "mean":
        ts_pool = ts_view.mean(dim=2)
    elif mode == "attention":
        # Parameter-free attention pooling: score each patch token then softmax over patches.
        patch_scores = ts_view.mean(dim=-1)
        patch_weights = torch.softmax(patch_scores, dim=2).unsqueeze(-1)
        ts_pool = torch.sum(ts_view * patch_weights, dim=2)
    elif mode == "last":
        # Preserve the most recent patch token for each variable.
        ts_pool = ts_view[:, :, -1, :]
    elif mode == "tail_mean":
        # Keep a short recency window instead of averaging the entire history.
        tail_k_resolved = _resolved_tail_k(num_patches=num_patches, tail_k=tail_k)
        ts_pool = ts_view[:, :, -tail_k_resolved:, :].mean(dim=2)
    elif mode == "tail_concat":
        # Preserve a short recency window without averaging away patch-order information.
        tail_k_resolved = _resolved_tail_k(num_patches=num_patches, tail_k=tail_k)
        ts_pool = ts_view[:, :, -tail_k_resolved:, :].reshape(bsz, num_vars, tail_k_resolved * hdim)
    else:
        raise ValueError(f"unknown ts pooling mode: {pooling}")
    if context_summary is not None:
        if context_summary.ndim != 2 or context_summary.shape[0] != bsz:
            raise ValueError(
                f"context_summary must be [B, C] matching batch size, got {tuple(context_summary.shape)}"
            )
        context_summary = context_summary.unsqueeze(1).expand(-1, num_vars, -1)
        ts_pool = torch.cat([ts_pool, context_summary], dim=-1)
    with time_block("pred.decode"):
        with torch.set_grad_enabled(use_grad):
            pred = pred_head(ts_pool)
    return pred.permute(0, 2, 1).contiguous()
