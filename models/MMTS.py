from contextlib import nullcontext
import json
import math
import os
from typing import Optional

import numpy as np
import torch
from torch import nn

from layers.normalization import build_forecast_normalizer
from layers.ts_mlp import build_ts_mlp
from layers.ts_features import encode_ts_embeddings, decode_ts_predictions
from layers.token_fusion import fuse_modalities
from models.qwen3_vl_utils import (
    load_qwen3_vl,
    freeze_qwen3_vl,
    apply_qwen3_vl_lora,
    encode_text_batch,
)
from utils.tools import ensure_dir, seed_everything, to_numpy
from utils.prompts import UNIFIED_PROMPT_VERSION, build_unified_prompts
from utils.qwen3_vl_patch import ts_attn_bias_context, patch_qwen3_vl_ts_attn_bias
from utils.ts_attention import (
    build_history_attention_mask,
    positions_from_padding_mask,
    reorder_ts_tokens,
)
from utils.ts_bias import build_full_ts_attention_bias


def _infer_mp_devices(model):
    input_device = None
    output_device = None
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return input_device, output_device

    for key, dev in device_map.items():
        if any(token in key for token in ("embed_tokens", "input_embeddings", "wte")):
            input_device = torch.device(dev)
            break

    max_layer_idx = None
    max_layer_dev = None
    for key, dev in device_map.items():
        if ".layers." in key:
            try:
                idx = int(key.split(".layers.")[1].split(".")[0])
            except Exception:
                continue
            if max_layer_idx is None or idx > max_layer_idx:
                max_layer_idx = idx
                max_layer_dev = dev
    if max_layer_dev is not None:
        output_device = torch.device(max_layer_dev)
    else:
        devices = list({v for v in device_map.values()})
        if devices:
            output_device = torch.device(devices[-1])

    if input_device is None and device_map:
        input_device = torch.device(next(iter(device_map.values())))
    return input_device, output_device


class Model(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.args = args
        self.device = torch.device(
            device or ("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
        )
        self.dataset_name = str(getattr(args, "dataset_name", "") or "")
        self.device_in = self.device_out = None
        self.ts_mlp = self.pred_head = None
        self.ts_time_embed = self.ts_var_embed = self.ts_normalizer = None
        self._ts_embed_cache = {}
        if args.use_ts_attn_bias:
            patch_qwen3_vl_ts_attn_bias()
        device_map = "auto" if args.model_parallel else None
        vlm, self.processor = load_qwen3_vl(args.qwen_dir, device_map=device_map)
        if device_map is None:
            vlm = vlm.to(self.device)
        freeze_qwen3_vl(vlm, freeze_lm=args.freeze_lm, freeze_vit=True)
        self.vlm = apply_qwen3_vl_lora(vlm, args)
        if args.model_parallel:
            self.device_in, self.device_out = _infer_mp_devices(self.vlm)
            if self.device_in is not None:
                self.device = self.device_in
        self._init_modality_params()
        self._model_dtype = next(self.vlm.parameters()).dtype
        seed_everything(args.seed)
        self.configure_channels()

    @property
    def backbone(self):
        # Access the decoder through the optional PEFT wrapper.
        return self.vlm.model.model if hasattr(self.vlm, "peft_config") else self.vlm.model

    def _init_modality_params(self) -> None:
        model = self.vlm
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        embed_dim = int(getattr(self.args, "embed_dim", 2048))
        for name in ("img_norm", "vid_norm", "text_norm"):
            if not hasattr(model, name):
                model.add_module(
                    name,
                    nn.LayerNorm(embed_dim, elementwise_affine=True).to(device=device, dtype=dtype),
                )
        for name in ("img_gate", "vid_gate", "text_gate"):
            if not hasattr(model, name):
                model.register_parameter(
                    name, nn.Parameter(torch.zeros((), device=device, dtype=dtype))
                )
        if bool(getattr(self.args, "ts_bias_learnable", True)):
            if not hasattr(model, "ts_bias_scale"):
                init = float(getattr(self.args, "ts_bias_scale", 0.05))
                init = max(init, 1e-6)
                init = math.log(math.exp(init) - 1.0)
                model.register_parameter(
                    "ts_bias_scale",
                    nn.Parameter(torch.tensor(init, device=device, dtype=dtype)),
                )

    def _resolve_dataset_name(self) -> str:
        dataset_name = str(
            getattr(self, "dataset_name", "") or getattr(self.args, "dataset_name", "") or ""
        ).strip()
        if dataset_name:
            return dataset_name
        data_path = str(getattr(self.args, "data_path", "") or "")
        if not data_path:
            return "dataset"
        return os.path.splitext(os.path.basename(data_path))[0] or "dataset"

    def _build_ts_modules(self, device, model_dtype=None):
        ts_mlp = build_ts_mlp(self.args.patch_len, self.args.embed_dim).to(device)
        dtype = model_dtype if model_dtype is not None else next(ts_mlp.parameters()).dtype
        num_patches = (self.args.seq_len - self.args.patch_len) // self.args.stride + 1
        pred_head = nn.Linear(num_patches * self.args.embed_dim, self.args.pred_len).to(
            device, dtype=dtype
        )
        return ts_mlp, pred_head

    def configure_channels(self) -> None:
        if self.ts_mlp is not None and self.pred_head is not None:
            self._maybe_build_time_embeds()
            self._maybe_build_ts_normalizer()
            return
        num_vars = int(getattr(self.args, "num_vars", -1))
        if num_vars <= 0:
            return
        model_dtype = self._model_dtype
        self.ts_mlp, self.pred_head = self._build_ts_modules(self.device, model_dtype=model_dtype)
        self._maybe_build_time_embeds()
        self._maybe_build_ts_normalizer()
        if bool(getattr(self.args, "model_parallel", False)) and self.device_out is not None:
            if next(self.pred_head.parameters()).device != self.device_out:
                self.pred_head = self.pred_head.to(self.device_out)

    def _maybe_build_ts_normalizer(self) -> None:
        if getattr(self, "ts_normalizer", None) is not None:
            return
        num_vars = int(getattr(self.args, "num_vars", -1))
        if num_vars <= 0:
            return
        dtype = self._model_dtype or torch.float32
        self.ts_normalizer = build_forecast_normalizer(
            num_features=num_vars,
            eps=float(getattr(self.args, "revin_eps", 1e-5)),
            subtract_last=bool(getattr(self.args, "revin_subtract_last", False)),
        ).to(device=self.device, dtype=dtype)

    def _maybe_build_time_embeds(self) -> None:
        if not bool(getattr(self.args, "use_time_features", False)):
            return
        if self.ts_time_embed is not None and self.ts_var_embed is not None:
            return
        num_vars = int(getattr(self.args, "num_vars", -1))
        if num_vars <= 0:
            return
        num_patches = (self.args.seq_len - self.args.patch_len) // self.args.stride + 1
        if num_patches <= 0:
            return
        device = self.device
        dtype = self._model_dtype or torch.float32
        self.ts_time_embed = nn.Embedding(num_patches, self.args.embed_dim).to(
            device=device, dtype=dtype
        )
        self.ts_var_embed = nn.Embedding(num_vars, self.args.embed_dim).to(
            device=device, dtype=dtype
        )

    def _add_time_features(self, ts_embeds: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self.args, "use_time_features", False)):
            return ts_embeds
        if not hasattr(self, "ts_time_embed") or self.ts_time_embed is None:
            return ts_embeds
        if not hasattr(self, "ts_var_embed") or self.ts_var_embed is None:
            return ts_embeds
        bsz, total_tokens, _ = ts_embeds.shape
        num_vars = int(getattr(self.args, "num_vars", -1))
        num_patches = (self.args.seq_len - self.args.patch_len) // self.args.stride + 1
        if num_vars <= 0 or num_patches <= 0:
            return ts_embeds
        if total_tokens != num_vars * num_patches:
            return ts_embeds
        key = (num_vars, num_patches, ts_embeds.device)
        cache = getattr(self, "_ts_embed_cache", {})
        if key in cache:
            var_idx, patch_idx = cache[key]
        else:
            var_idx = torch.arange(num_vars, device=ts_embeds.device).repeat_interleave(num_patches)
            patch_idx = torch.arange(num_patches, device=ts_embeds.device).repeat(num_vars)
            cache[key] = (var_idx, patch_idx)
            self._ts_embed_cache = cache
        var_embed = self.ts_var_embed(var_idx)
        time_embed = self.ts_time_embed(patch_idx)
        scale = float(getattr(self.args, "time_embed_scale", 1.0))
        add = (var_embed + time_embed) * scale
        return ts_embeds + add.unsqueeze(0)

    def _encode_unified_text(self, x: torch.Tensor):
        budget = int(getattr(self.args, "prompt_max_tokens", 192))
        if budget <= 0:
            raise ValueError("unified prompts require a positive total token budget")
        prompts = build_unified_prompts(x, self._resolve_dataset_name(), self.args.pred_len)
        first_batch = not getattr(self, "_prompt_metadata_written", False)
        result = encode_text_batch(
            self.vlm,
            self.processor,
            prompts=prompts,
            device=str(self.device),
            max_length=budget,
            pad=bool(getattr(self.args, "prompt_pad", True)),
            truncate=True,
            return_metadata=first_batch,
        )
        tokens, mask = result[:2]
        lengths = mask.sum(dim=1)
        if tokens.shape[1] > budget or bool((lengths > budget).any()):
            raise RuntimeError("the complete text segment exceeded its token budget")
        if first_batch:
            ensure_dir(self.args.output_dir)
            record = {
                "template_version": UNIFIED_PROMPT_VERSION,
                "style": "unified",
                "total_token_budget": budget,
                "allocated_text_tokens": int(tokens.shape[1]),
                "example_scope": "first batch, at most eight examples",
                "first_batch_min_tokens": int(lengths.min()),
                "first_batch_max_tokens": int(lengths.max()),
                "examples": result[2][:8],
            }
            with open(
                os.path.join(self.args.output_dir, "prompt_metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            self._prompt_metadata_written = True
            print(
                f"[Prompt] style=unified tokens={int(lengths.min())}..{int(lengths.max())} total_budget={budget}"
            )
        return tokens, mask

    def _run_backbone(
        self,
        fused_embeds,
        attn_mask,
        use_grad: bool,
        ts_attn_bias: torch.Tensor = None,
        ts_attn_layer: Optional[int] = None,
        ts_attn_bias_scale: torch.Tensor = None,
        ts_range: tuple[int, int] = None,
    ):
        if ts_range is None:
            raise ValueError("ts_range is required for observed-history attention")
        position_ids = positions_from_padding_mask(attn_mask)
        attn_mask = build_history_attention_mask(attn_mask, ts_range, fused_embeds.dtype)
        bias_context = (
            ts_attn_bias_context(ts_attn_bias, layer_idx=ts_attn_layer, bias_scale=ts_attn_bias_scale)
            if ts_attn_bias is not None
            else nullcontext()
        )
        with torch.set_grad_enabled(use_grad), bias_context:
            out = self.backbone(
                inputs_embeds=fused_embeds,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        return out.last_hidden_state

    def forward(
        self,
        x: torch.Tensor,
        img_grids: torch.Tensor = None,
        vid_grids: torch.Tensor = None,
        img_tokens: torch.Tensor = None,
        img_token_mask: torch.Tensor = None,
        vid_tokens: torch.Tensor = None,
        vid_token_mask: torch.Tensor = None,
    ):
        """Map x [B, L, C] and cached multimodal inputs to forecasts [B, H, C]."""
        num_patches = (self.args.seq_len - self.args.patch_len) // self.args.stride + 1
        train = self.training and torch.is_grad_enabled()
        x_model = x.to(self.device)
        x_aux, ts_norm_ctx = self.ts_normalizer.normalize(x_model)
        ts_embeds = self._build_ts_embeddings(x_aux, train)
        use_vision = bool(getattr(self.args, "use_vision", True))
        use_text = bool(getattr(self.args, "use_text", True))
        use_bias = bool(getattr(self.args, "use_ts_attn_bias", False))

        img_stats = None
        vid_stats = None
        if use_bias:
            if img_grids is not None and vid_grids is not None:
                img_stats = self._relation_stats(img_grids)
                vid_stats = self._relation_stats(vid_grids)
            else:
                raise ValueError("Structural bias requires cached image/video relation grids.")

        if use_vision:
            if img_tokens is not None and vid_tokens is not None:
                img_token_seqs = (img_tokens, img_token_mask)
                vid_token_seqs = (vid_tokens, vid_token_mask)
            else:
                raise RuntimeError(
                    "Cached visual tokens are missing. "
                    "Run from the source CSV with --rebuild_cache true."
                )
        else:
            img_token_seqs, vid_token_seqs = [], []

        fused_embeds, attn_mask, ts_range = fuse_modalities(
            model=self.vlm,
            ts_embeds=ts_embeds,
            img_token_seqs=img_token_seqs,
            vid_token_seqs=vid_token_seqs,
            txt_token_embeds=self._encode_unified_text(x_aux) if use_text else None,
            device=str(self.device),
            img_norm=getattr(self.vlm, "img_norm", None) if use_vision else None,
            vid_norm=getattr(self.vlm, "vid_norm", None) if use_vision else None,
            text_norm=getattr(self.vlm, "text_norm", None) if use_text else None,
            img_gate=getattr(self.vlm, "img_gate", None) if use_vision else None,
            vid_gate=getattr(self.vlm, "vid_gate", None) if use_vision else None,
            text_gate=getattr(self.vlm, "text_gate", None) if use_text else None,
            img_scale=float(getattr(self.args, "img_scale", 1.0)),
            vid_scale=float(getattr(self.args, "vid_scale", 1.0)),
            text_scale=float(getattr(self.args, "text_scale", 1.0)),
        )
        ts_attn_bias = None
        if use_bias and img_stats is not None and vid_stats is not None:
            weights = (
                float(getattr(self.args, "ts_bias_dtw_weight", 1.0)),
                float(getattr(self.args, "ts_bias_cov_weight", 1.0)),
                float(getattr(self.args, "ts_bias_pear_weight", 1.0)),
            )
            ts_attn_bias = build_full_ts_attention_bias(
                img_stats=img_stats,
                vid_stats=vid_stats,
                num_patches=num_patches,
                ts_range=ts_range,
                total_tokens=int(fused_embeds.shape[1]),
                num_vars=self.args.num_vars,
                device=self.device,
                dtype=fused_embeds.dtype,
                weights=weights,
                token_layout="patch_major",
            )
        backbone_grad = bool(train)
        ts_attn_layer = self._resolve_ts_attn_layers()
        bias_scale = None
        if use_bias:
            if bool(getattr(self.args, "ts_bias_learnable", False)):
                bias_scale = getattr(self.vlm, "ts_bias_scale", None)
            else:
                # qwen3_vl_patch applies softplus to the raw scale parameter.
                fixed_scale = float(getattr(self.args, "ts_bias_scale", 0.05))
                fixed_scale = max(fixed_scale, 1e-6)
                fixed_scale = math.log(math.exp(fixed_scale) - 1.0)
                bias_scale = torch.tensor(
                    fixed_scale,
                    device=fused_embeds.device,
                    dtype=fused_embeds.dtype,
                )
        hidden_states = self._run_backbone(
            fused_embeds=fused_embeds,
            attn_mask=attn_mask,
            use_grad=backbone_grad,
            ts_attn_bias=ts_attn_bias,
            ts_attn_layer=ts_attn_layer,
            ts_attn_bias_scale=bias_scale,
            ts_range=ts_range,
        )
        if bool(getattr(self.args, "use_ts_residual", False)) and ts_range is not None:
            ts_start, ts_end = ts_range
            if ts_start <= ts_end:
                ts_add = ts_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
                hidden_states = hidden_states.clone()
                ts_hidden = hidden_states[:, ts_start : ts_end + 1, :]

                hidden_states[:, ts_start : ts_end + 1, :] = ts_hidden + ts_add
        pred_device = next(self.pred_head.parameters()).device
        if hidden_states.device != pred_device:
            hidden_states = hidden_states.to(pred_device)

        preds = decode_ts_predictions(
            hidden_states=hidden_states,
            ts_range=ts_range,
            num_vars=self.args.num_vars,
            num_patches=num_patches,
            pred_head=self.pred_head,
            use_grad=bool(train),
            token_layout="patch_major",
        )
        if ts_norm_ctx is not None:
            preds = self.ts_normalizer.denormalize(preds, ts_norm_ctx)
        return preds

    def _build_ts_embeddings(self, x_input: torch.Tensor, train: bool):
        ts_embeds = encode_ts_embeddings(
            x=x_input,
            ts_mlp=self.ts_mlp,
            patch_len=self.args.patch_len,
            stride=self.args.stride,
            device=self.device,
            use_grad=bool(train),
        )
        ts_embeds = self._add_time_features(ts_embeds)
        # Reorder embedded patches from c*P+p to p*C+c for the backbone.
        num_patches = (self.args.seq_len - self.args.patch_len) // self.args.stride + 1
        return reorder_ts_tokens(
            ts_embeds,
            self.args.num_vars,
            num_patches,
            "channel_major",
            "patch_major",
        )

    @staticmethod
    def _relation_stats(relation_grids: torch.Tensor):
        grids = to_numpy(relation_grids)
        dtw = (grids[..., 0] * 2.0 - 1.0).astype(np.float32)
        cov = (grids[..., 1] * 2.0 - 1.0).astype(np.float32)
        pear = (grids[..., 2] * 2.0 - 1.0).astype(np.float32)
        return dtw, cov, pear

    def _resolve_ts_attn_layers(self):
        layers_text = str(getattr(self.args, "ts_attn_bias_layers", "") or "").strip().lower()
        if layers_text:
            if layers_text == "all":
                return None
            layers = []
            for part in layers_text.split(","):
                part = part.strip()
                if not part:
                    continue
                idx = int(part)
                if idx < 0:
                    return None
                layers.append(idx)
            if not layers:
                return None
            return tuple(sorted(set(layers)))

        raise ValueError("ts_attn_bias_layers must contain indices or all")
