"""Token layout and visibility for the observed forecasting window."""

import torch


DEFAULT_TS_TOKEN_LAYOUT = "patch_major"
DEFAULT_TS_ATTENTION_MODE = "history_bidirectional"
TS_TOKEN_LAYOUTS = ("channel_major", "patch_major")


def reorder_ts_tokens(
    tokens: torch.Tensor,
    num_vars: int,
    num_patches: int,
    source_layout: str,
    target_layout: str,
) -> torch.Tensor:
    """Reorder [B, C*P, ...] without changing channel/patch identities."""
    if source_layout not in TS_TOKEN_LAYOUTS or target_layout not in TS_TOKEN_LAYOUTS:
        raise ValueError(f"unknown TS token layout: {source_layout!r} -> {target_layout!r}")
    if num_vars <= 0 or num_patches <= 0:
        raise ValueError("num_vars and num_patches must be positive")
    if tokens.ndim < 2 or tokens.shape[1] != num_vars * num_patches:
        raise ValueError(f"expected [B, {num_vars * num_patches}, ...], got {tuple(tokens.shape)}")
    if source_layout == target_layout:
        return tokens
    first, second = (
        (num_vars, num_patches) if source_layout == "channel_major" else (num_patches, num_vars)
    )
    shaped = tokens.reshape(tokens.shape[0], first, second, *tokens.shape[2:])
    return shaped.transpose(1, 2).reshape(tokens.shape).contiguous()


def build_history_attention_mask(
    padding_mask: torch.Tensor,
    ts_range: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Keep the prefix causal; let all observed TS queries see all valid keys.

    TS occupies the final segment. Targets are not part of this sequence.
    Invalid query rows may attend only to themselves to avoid empty softmax
    rows; padded keys remain invisible to every valid query.
    """
    if padding_mask.ndim != 2:
        raise ValueError("padding_mask must have shape [B, S]")
    if not dtype.is_floating_point:
        raise ValueError("the additive attention mask requires a floating dtype")
    seq_len = padding_mask.shape[1]
    ts_start, ts_end = ts_range
    if not (0 <= ts_start <= ts_end == seq_len - 1):
        raise ValueError("TS tokens must form a nonempty suffix of the input sequence")
    valid = padding_mask.to(dtype=torch.bool)
    positions = torch.arange(seq_len, device=padding_mask.device)
    query = positions[:, None]
    key = positions[None, :]
    allowed = (key <= query) | (query >= ts_start)
    allowed = allowed.unsqueeze(0) & valid[:, None, :] & valid[:, :, None]
    diagonal = torch.eye(seq_len, dtype=torch.bool, device=padding_mask.device)
    allowed = allowed | (~valid[:, :, None] & diagonal.unsqueeze(0))
    mask = torch.full(
        allowed.shape, torch.finfo(dtype).min, device=padding_mask.device, dtype=dtype
    )
    return mask.masked_fill_(allowed, 0.0).unsqueeze(1)


def positions_from_padding_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """Match Qwen3-VL's inputs_embeds-only RoPE positions from the 2D mask.

    Pass these explicitly with the custom 4D mask: its safe padded-query
    diagonals must not be mistaken for real tokens by Qwen's mask conversion.
    """
    if padding_mask.ndim != 2:
        raise ValueError("padding_mask must have shape [B, S]")
    positions = padding_mask.long().cumsum(-1) - 1
    positions = positions.masked_fill(padding_mask == 0, 1)
    return positions.unsqueeze(0).expand(3, -1, -1)


def validate_checkpoint_attention(args, checkpoint_args: dict) -> None:
    """Prevent loading checkpoints with different token semantics."""
    settings = (
        ("ts_token_layout", DEFAULT_TS_TOKEN_LAYOUT, "channel_major"),
        ("ts_attention_mode", DEFAULT_TS_ATTENTION_MODE, "causal"),
    )
    mismatches = []
    for name, default, legacy in settings:
        requested = getattr(args, name, default)
        saved = checkpoint_args.get(name, legacy)
        if requested != saved:
            mismatches.append(f"{name}: checkpoint={saved}, requested={requested}")
    if mismatches:
        raise ValueError(
            "Checkpoint attention configuration mismatch (" + "; ".join(mismatches) + "). "
            "Use the source version matching that checkpoint, or train the current architecture. "
            "Checkpoints without these settings are treated as channel_major + causal."
        )


def validate_checkpoint_forecasting(args, checkpoint_args: dict) -> None:
    """Validate the output head, bias and prompt configuration against a checkpoint."""
    from utils.prompts import UNIFIED_PROMPT_VERSION

    settings = (
        ("use_ts_residual", True, False),
        ("ts_residual_mode", "add", "study"),
        ("ts_pooling", "flatten", "mean"),
        ("pred_head_mode", "linear", "linear"),
        ("pred_context_mode", "none", "none"),
        ("ts_bias_norm", "none", "none"),
        ("prompt_style", "unified", "sample"),
        ("prompt_max_tokens", 192, 128),
    )
    mismatches = []
    for name, default, legacy in settings:
        if name == "ts_residual_mode" and not getattr(args, "use_ts_residual", True):
            continue
        if name.startswith("prompt_") and not getattr(args, "use_text", True):
            continue
        saved, requested = checkpoint_args.get(name, legacy), getattr(args, name, default)
        if saved != requested:
            mismatches.append(f"{name}: checkpoint={saved}, requested={requested}")
    if getattr(args, "use_text", True) and getattr(args, "prompt_style", "unified") == "unified":
        if checkpoint_args.get("prompt_template_version") != UNIFIED_PROMPT_VERSION:
            mismatches.append("prompt_template_version differs or is missing")
    if mismatches:
        raise ValueError(
            "Checkpoint forecasting configuration mismatch (" + "; ".join(mismatches) + "). "
            "Load a checkpoint with a matching model configuration."
        )
