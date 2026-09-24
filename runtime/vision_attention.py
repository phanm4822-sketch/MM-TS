"""Batch independent equal-sized visual attention segments using original eager kernels.
Used only by the opt-in, frozen visual CUDA graph.
"""

from contextlib import contextmanager
import types
import torch


def batched_attention(
    self,
    hidden_states,
    cu_seqlens,
    rotary_pos_emb=None,
    position_embeddings=None,
    **kwargs
):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )

    lengths = getattr(self, "_cache_segment_lengths", None)
    if not lengths or len(set(lengths)) != 1:
        return self._unbatched_forward(
            hidden_states,
            cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs
        )
    seq_length = hidden_states.shape[0]
    length = lengths[0]
    count = len(lengths)
    assert count * length == seq_length
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q = q.reshape(count, length, self.num_heads, -1).transpose(1, 2)
    k = k.reshape(count, length, self.num_heads, -1).transpose(1, 2)
    v = v.reshape(count, length, self.num_heads, -1).transpose(1, 2)
    out, _ = eager_attention_forward(
        self,
        q,
        k,
        v,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0,
        is_causal=False,
    )
    return self.proj(out.reshape(seq_length, -1).contiguous())


@contextmanager
def batched_visual_attention(visual, enabled=True):
    if not enabled:
        yield
        return
    originals = []
    for block in visual.blocks:
        attention = block.attn
        originals.append((attention, attention.forward))
        attention._unbatched_forward = attention.forward
        attention.forward = types.MethodType(batched_attention, attention)

    def prepare(module, args, kwargs):
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        rows = grid.detach().cpu().tolist()
        lengths = [h * w for t, h, w in rows for _ in range(t)]
        for attention, _ in originals:
            attention._cache_segment_lengths = lengths

    hook = visual.register_forward_pre_hook(prepare, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()
        for attention, forward in originals:
            attention.forward = forward
            for key in ["_unbatched_forward", "_cache_segment_lengths"]:
                if hasattr(attention, key):
                    delattr(attention, key)
