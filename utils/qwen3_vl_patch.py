from contextlib import contextmanager
import importlib
from typing import Optional

import torch
import torch.nn.functional as F


_QWEN3_VL_MODULES = (
    "transformers.models.qwen3_vl.modeling_qwen3_vl",
    "transformers.models.qwen3_vl.modular_qwen3_vl",
)


def _normalize_ts_bias(ts_attn_bias: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
    if ts_attn_bias.ndim == 3:
        ts_attn_bias = ts_attn_bias.unsqueeze(1)
    if ts_attn_bias.ndim != 4:
        raise ValueError(
            f"ts_attn_bias must be [B, 1, S, S] or [B, S, S], got {tuple(ts_attn_bias.shape)}"
        )
    if (
        ts_attn_bias.shape[0] != attn_weights.shape[0]
        or ts_attn_bias.shape[-2] != attn_weights.shape[-2]
        or ts_attn_bias.shape[-1] != attn_weights.shape[-1]
    ):
        raise ValueError(
            "ts_attn_bias shape must match attention scores on batch and seq dims: "
            f"bias={tuple(ts_attn_bias.shape)} scores={tuple(attn_weights.shape)}"
        )
    return ts_attn_bias


def _layer_matches(layer_spec, layer_idx: int) -> bool:
    if layer_spec is None:
        return True
    if isinstance(layer_spec, (set, tuple, list)):
        return int(layer_idx) in {int(x) for x in layer_spec}
    return int(layer_spec) == int(layer_idx)


def _eager_attention_with_ts_bias(
    mod,
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    ts_attn_bias: torch.Tensor,
    ts_bias_scale: torch.Tensor,
    scaling: float,
    dropout: float,
):
    key_states = mod.repeat_kv(key, module.num_key_value_groups)
    value_states = mod.repeat_kv(value, module.num_key_value_groups)

    raw_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = raw_scores
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    ts_attn_bias = _normalize_ts_bias(ts_attn_bias, attn_weights)
    bias_mode = str(getattr(mod, "_ts_attn_bias_mode", "add")).lower()
    if bias_mode == "softmax":
        ts_attn_bias = F.softmax(ts_attn_bias, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
    elif bias_mode == "sigmoid":
        ts_attn_bias = torch.sigmoid(ts_attn_bias)
    if ts_bias_scale is None:
        scale = raw_scores.abs() * 0.05
    else:
        scale = F.softplus(ts_bias_scale).to(attn_weights.device, attn_weights.dtype)
    attn_weights = attn_weights + ts_attn_bias.to(attn_weights.device, attn_weights.dtype) * scale

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _patch_module(mod) -> bool:
    if getattr(mod, "_ts_attn_bias_patched", False):
        return True

    if not hasattr(mod, "Qwen3VLTextAttention"):
        return False

    if not hasattr(mod, "_ts_attn_bias"):
        mod._ts_attn_bias = None
    if not hasattr(mod, "_ts_attn_bias_layer"):
        mod._ts_attn_bias_layer = None
    if not hasattr(mod, "_ts_attn_bias_mode"):
        mod._ts_attn_bias_mode = "add"
    if not hasattr(mod, "_ts_attn_bias_scale"):
        mod._ts_attn_bias_scale = None
    orig_forward = mod.Qwen3VLTextAttention.forward

    def _forward_with_ts_bias(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        ts_attn_bias = getattr(mod, "_ts_attn_bias", None)
        ts_layer = getattr(mod, "_ts_attn_bias_layer", None)
        use_ts_bias = ts_attn_bias is not None and _layer_matches(ts_layer, self.layer_idx)
        ts_bias_scale = getattr(mod, "_ts_attn_bias_scale", None)

        attention_interface = mod.eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = mod.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if use_ts_bias:
            attn_output, attn_weights = _eager_attention_with_ts_bias(
                mod=mod,
                module=self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                ts_attn_bias=ts_attn_bias,
                ts_bias_scale=ts_bias_scale,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    mod.Qwen3VLTextAttention.forward = _forward_with_ts_bias
    mod._ts_attn_bias_patched = True
    mod._ts_attn_bias_orig_forward = orig_forward
    return True


def patch_qwen3_vl_ts_attn_bias():
    patched_any = False
    for mod_name in _QWEN3_VL_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        patched_any = _patch_module(mod) or patched_any

    if not patched_any:
        raise RuntimeError("transformers Qwen3-VL module not available for patching")


@contextmanager
def ts_attn_bias_context(
    ts_attn_bias,
    layer_idx: Optional[int] = None,
    bias_mode: str = "add",
    bias_scale: torch.Tensor = None,
):
    patch_qwen3_vl_ts_attn_bias()
    mods = []
    for mod_name in _QWEN3_VL_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        mods.append(mod)

    old = [
        (
            getattr(mod, "_ts_attn_bias", None),
            getattr(mod, "_ts_attn_bias_layer", None),
            getattr(mod, "_ts_attn_bias_mode", "add"),
            getattr(mod, "_ts_attn_bias_scale", None),
        )
        for mod in mods
    ]
    for mod in mods:
        mod._ts_attn_bias = ts_attn_bias
        mod._ts_attn_bias_layer = layer_idx
        mod._ts_attn_bias_mode = str(bias_mode).lower()
        mod._ts_attn_bias_scale = bias_scale
    try:
        yield
    finally:
        for mod, (
            prev_bias,
            prev_layer,
            prev_mode,
            prev_scale,
        ) in zip(mods, old):
            mod._ts_attn_bias = prev_bias
            mod._ts_attn_bias_layer = prev_layer
            mod._ts_attn_bias_mode = prev_mode
            mod._ts_attn_bias_scale = prev_scale
