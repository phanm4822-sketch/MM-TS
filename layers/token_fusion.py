import torch
import torch.nn.functional as F


def _gate_scale(gate: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    # Keep zero init neutral: sigmoid(0)=0.5, so rescale to make 0 -> 1.0.
    return (2.0 * torch.sigmoid(gate)).to(device=device, dtype=dtype)


def stack_token_seqs(token_seqs, device: str, embed_dim: int, dtype: torch.dtype):
    if not token_seqs:
        out = torch.zeros((1, 0, embed_dim), dtype=dtype, device=device)
        mask = torch.zeros((1, 0), dtype=torch.long, device=device)
        return out, mask

    bsz = len(token_seqs)
    max_len = max(int(t.shape[0]) for t in token_seqs)
    out = torch.zeros((bsz, max_len, embed_dim), dtype=dtype, device=device)
    mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    for i, t in enumerate(token_seqs):
        cur = t.to(device)
        cur_len = int(cur.shape[0])
        out[i, :cur_len, :] = cur
        mask[i, :cur_len] = 1
    return out, mask


def _as_stacked_tokens(token_seqs, device: str, embed_dim: int, dtype: torch.dtype):
    if isinstance(token_seqs, tuple) and len(token_seqs) == 2:
        tokens, mask = token_seqs
        if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError("precomputed tokens must be torch tensors")
        tokens = tokens.to(device=device, dtype=dtype)
        mask = mask.to(device=device)
        return tokens, mask
    return stack_token_seqs(token_seqs, device=device, embed_dim=embed_dim, dtype=dtype)


def _apply_norm(tokens: torch.Tensor, norm_module):
    if norm_module is None or int(tokens.shape[1]) == 0:
        return tokens
    weight = getattr(norm_module, "weight", None)
    bias = getattr(norm_module, "bias", None)
    eps = float(getattr(norm_module, "eps", 1e-5))
    normalized_shape = getattr(norm_module, "normalized_shape", tokens.shape[-1:])
    if not isinstance(normalized_shape, tuple):
        normalized_shape = (normalized_shape,)
    weight = weight.to(device=tokens.device, dtype=tokens.dtype) if weight is not None else None
    bias = bias.to(device=tokens.device, dtype=tokens.dtype) if bias is not None else None
    return F.layer_norm(tokens, normalized_shape, weight=weight, bias=bias, eps=eps)


def fuse_modalities(
    model,
    ts_embeds,
    img_token_seqs,
    vid_token_seqs,
    txt_token_embeds,
    device: str = None,
    img_norm=None,
    vid_norm=None,
    text_norm=None,
    img_gate=None,
    vid_gate=None,
    text_gate=None,
    img_scale: float = 1.0,
    vid_scale: float = 1.0,
    text_scale: float = 1.0,
):
    if device is None:
        device = str(next(model.parameters()).device)
    model_dtype = next(model.parameters()).dtype
    bsz = int(ts_embeds.shape[0])
    embed_dim = int(ts_embeds.shape[-1])

    ts_embeds = ts_embeds.to(device=device, dtype=model_dtype)
    ts_len = int(ts_embeds.shape[1])

    txt_embeds, txt_mask = _as_stacked_tokens(
        txt_token_embeds, device=device, embed_dim=embed_dim, dtype=model_dtype
    )
    if txt_embeds.shape[0] == 1 and bsz > 1:
        txt_embeds = txt_embeds.repeat(bsz, 1, 1)
        txt_mask = txt_mask.repeat(bsz, 1)
    if text_norm is not None and txt_embeds.shape[1] > 0:
        txt_embeds = _apply_norm(txt_embeds, text_norm)
    if text_gate is not None and txt_embeds.shape[1] > 0:
        txt_embeds = txt_embeds * _gate_scale(text_gate, device=device, dtype=model_dtype)
    if txt_embeds.shape[1] > 0 and float(text_scale) != 1.0:
        txt_embeds = txt_embeds * torch.as_tensor(float(text_scale), device=device, dtype=model_dtype)
    txt_len = int(txt_embeds.shape[1])

    img_tokens, img_mask = _as_stacked_tokens(
        img_token_seqs, device=device, embed_dim=embed_dim, dtype=model_dtype
    )
    vid_tokens, vid_mask = _as_stacked_tokens(
        vid_token_seqs, device=device, embed_dim=embed_dim, dtype=model_dtype
    )
    img_tokens = img_tokens.to(dtype=model_dtype)
    vid_tokens = vid_tokens.to(dtype=model_dtype)
    if img_tokens.shape[0] == 1 and bsz > 1:
        img_tokens = img_tokens.repeat(bsz, 1, 1)
        img_mask = img_mask.repeat(bsz, 1)
    if vid_tokens.shape[0] == 1 and bsz > 1:
        vid_tokens = vid_tokens.repeat(bsz, 1, 1)
        vid_mask = vid_mask.repeat(bsz, 1)
    if img_norm is not None and img_tokens.shape[1] > 0:
        img_tokens = _apply_norm(img_tokens, img_norm)
    if vid_norm is not None and vid_tokens.shape[1] > 0:
        vid_tokens = _apply_norm(vid_tokens, vid_norm)
    if img_gate is not None and img_tokens.shape[1] > 0:
        img_tokens = img_tokens * _gate_scale(img_gate, device=device, dtype=model_dtype)
    if vid_gate is not None and vid_tokens.shape[1] > 0:
        vid_tokens = vid_tokens * _gate_scale(vid_gate, device=device, dtype=model_dtype)
    if img_tokens.shape[1] > 0 and float(img_scale) != 1.0:
        img_tokens = img_tokens * torch.as_tensor(float(img_scale), device=device, dtype=model_dtype)
    if vid_tokens.shape[1] > 0 and float(vid_scale) != 1.0:
        vid_tokens = vid_tokens * torch.as_tensor(float(vid_scale), device=device, dtype=model_dtype)

    ts_mask = torch.ones((bsz, ts_len), dtype=torch.long, device=device)
    blocks = [
        ("text", txt_embeds, txt_mask),
        ("img", img_tokens, img_mask),
        ("vid", vid_tokens, vid_mask),
        ("ts", ts_embeds, ts_mask),
    ]

    fused = torch.cat([b[1] for b in blocks], dim=1)
    attn_mask = torch.cat([b[2] for b in blocks], dim=1)

    # compute ts range based on actual order
    ts_start = 0
    for name, tokens, _mask in blocks:
        if name == "ts":
            break
        ts_start += int(tokens.shape[1])
    ts_end = ts_start + ts_len - 1

    token_counts = {
        "img": int(img_tokens.shape[1]),
        "vid": int(vid_tokens.shape[1]),
        "ts": ts_len,
        "text": txt_len,
        "total": int(fused.shape[1]),
    }
    return fused, attn_mask, (ts_start, ts_end), token_counts
