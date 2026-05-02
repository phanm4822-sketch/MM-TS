import json
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from layers.ts_mlp import build_ts_mlp
from models import encode_text_batch
from layers.modality_builder import (
    build_image_modality,
    build_image_modality_from_grids,
    build_video_modality,
    build_video_modality_from_grids,
)
from layers.token_fusion import fuse_modalities
from layers.ts_features import (
    PerVariablePredictionHead,
    context_summary_dim,
    decode_ts_predictions,
    encode_ts_embeddings,
    pooled_ts_dim,
)
from utils import ensure_dir, print_box, time_block, to_numpy
from utils.qwen3_vl_patch import ts_attn_bias_context
from utils.ts_bias import build_full_ts_attention_bias


class ExpMainSteps:
    def _prepare_output(self):
        with time_block("io.prepare_output"):
            ensure_dir(self.args.output_dir)

    def _resolve_dataset_name(self) -> str:
        dataset_name = str(getattr(self, "dataset_name", "") or getattr(self.args, "dataset_name", "") or "").strip()
        if dataset_name:
            return dataset_name
        data_path = str(getattr(self.args, "data_path", "") or "")
        if not data_path:
            return "dataset"
        return os.path.splitext(os.path.basename(data_path))[0] or "dataset"

    @staticmethod
    def _load_vision_patch_size(qwen_dir: str) -> int:
        config_path = os.path.join(qwen_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg["vision_config"]["patch_size"])

    def _build_ts_modules(self, device, model_dtype=None):
        print_box("3) Build TS MLP")
        ts_mlp = build_ts_mlp(
            self.args.patch_len,
            embed_dim=self.args.embed_dim,
            mode=getattr(self.args, "ts_mlp_mode", "shared"),
            num_vars=getattr(self.args, "num_vars", None),
        ).to(device)
        dtype = model_dtype if model_dtype is not None else next(ts_mlp.parameters()).dtype
        num_patches = (self.args.seq_len - self.args.patch_len) // self.args.stride + 1
        pred_in_dim = pooled_ts_dim(
            embed_dim=self.args.embed_dim,
            num_patches=num_patches,
            pooling=getattr(self.args, "ts_pooling", "mean"),
            tail_k=int(getattr(self.args, "ts_tail_k", 3)),
        )
        pred_in_dim += context_summary_dim(
            embed_dim=self.args.embed_dim,
            use_text=bool(getattr(self.args, "use_text", True)),
            use_vision=bool(getattr(self.args, "use_vision", True)),
            mode=getattr(self.args, "pred_context_mode", "none"),
        )
        pred_head_mode = str(getattr(self.args, "pred_head_mode", "linear")).strip().lower()
        hidden_mult = max(1.0, float(getattr(self.args, "pred_head_hidden_mult", 1.0)))
        hidden_dim = int(round(self.args.embed_dim * hidden_mult))
        head_dropout = float(getattr(self.args, "pred_head_dropout", 0.0))
        if pred_head_mode == "mlp":
            pred_head = nn.Sequential(
                nn.Linear(pred_in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, self.args.pred_len),
            ).to(device, dtype=dtype)
        elif pred_head_mode == "mlp_per_var":
            pred_head = PerVariablePredictionHead(
                input_dim=pred_in_dim,
                pred_len=self.args.pred_len,
                num_vars=self.args.num_vars,
                mode="mlp",
                hidden_dim=hidden_dim,
                dropout=head_dropout,
            ).to(device, dtype=dtype)
        elif pred_head_mode == "linear_per_var":
            pred_head = PerVariablePredictionHead(
                input_dim=pred_in_dim,
                pred_len=self.args.pred_len,
                num_vars=self.args.num_vars,
                mode="linear",
            ).to(device, dtype=dtype)
        else:
            pred_head = nn.Linear(pred_in_dim, self.args.pred_len).to(device, dtype=dtype)
        return ts_mlp, pred_head

    def _build_prediction_context(self, hidden_states: torch.Tensor, token_counts: dict | None):
        mode = str(getattr(self.args, "pred_context_mode", "none")).strip().lower()
        if mode == "none" or not token_counts:
            return None

        parts = []
        text_len = int(token_counts.get("text", 0))
        img_len = int(token_counts.get("img", 0))
        vid_len = int(token_counts.get("vid", 0))
        offset = 0

        if mode in {"text", "all"} and bool(getattr(self.args, "use_text", True)):
            if text_len > 0:
                parts.append(hidden_states[:, offset:offset + text_len, :].mean(dim=1))
            else:
                parts.append(torch.zeros((hidden_states.shape[0], hidden_states.shape[-1]), device=hidden_states.device, dtype=hidden_states.dtype))
        offset += text_len

        if mode in {"vision", "all"} and bool(getattr(self.args, "use_vision", True)):
            if img_len > 0:
                parts.append(hidden_states[:, offset:offset + img_len, :].mean(dim=1))
            else:
                parts.append(torch.zeros((hidden_states.shape[0], hidden_states.shape[-1]), device=hidden_states.device, dtype=hidden_states.dtype))
            offset += img_len
            if vid_len > 0:
                parts.append(hidden_states[:, offset:offset + vid_len, :].mean(dim=1))
            else:
                parts.append(torch.zeros((hidden_states.shape[0], hidden_states.shape[-1]), device=hidden_states.device, dtype=hidden_states.dtype))

        if not parts:
            return None
        return torch.cat(parts, dim=-1)

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

    def _build_global_prompt(self) -> str:
        num_patches = (self.args.seq_len - self.args.patch_len) // self.args.stride + 1
        dataset_name = self._resolve_dataset_name()
        dataset_lower = dataset_name.lower()
        scene = "unknown"
        if "electricity" in dataset_lower or "etth" in dataset_lower or "ettm" in dataset_lower:
            scene = "electricity load"
        elif "weather" in dataset_lower:
            scene = "weather"
        elif "traffic" in dataset_lower:
            scene = "traffic"
        elif "exchange" in dataset_lower:
            scene = "exchange rate"
        elif "ili" in dataset_lower:
            scene = "influenza-like illness"

        freq = "unknown"
        if "etth" in dataset_lower:
            freq = "hourly"
        elif "ettm" in dataset_lower:
            freq = "15-min"
        elif "ili" in dataset_lower:
            freq = "weekly"
        elif "exchange" in dataset_lower:
            freq = "daily"
        elif "electricity" in dataset_lower:
            freq = "hourly"
        elif "traffic" in dataset_lower:
            freq = "hourly"
        elif "weather" in dataset_lower:
            freq = "10-min"

        mean_val = "unknown"
        var_val = "unknown"
        min_val = "unknown"
        max_val = "unknown"
        if getattr(self, "data_mean", None) is not None and getattr(self, "data_std", None) is not None:
            try:
                mean_val = f"{float(self.data_mean.mean().item()):.6f}"
                var_val = f"{float((self.data_std ** 2).mean().item()):.6f}"
            except Exception:
                mean_val = "unknown"
                var_val = "unknown"
        if getattr(self, "data_min", None) is not None and getattr(self, "data_max", None) is not None:
            try:
                min_val = f"{float(torch.as_tensor(self.data_min).min().item()):.6f}"
                max_val = f"{float(torch.as_tensor(self.data_max).max().item()):.6f}"
            except Exception:
                min_val = "unknown"
                max_val = "unknown"

        return (
            "Task: multivariate time-series forecasting.\n"
            f"[1] Scenario: {scene} dataset={dataset_name}.\n"
            f"[2] Sampling frequency: {freq}.\n"
            f"[3] Stats (train windows): min={min_val}, max={max_val}, mean={mean_val}, var={var_val}.\n"
            "[4] Image channels: DTW distance heatmap, covariance heatmap, Pearson correlation heatmap.\n"
            f"[5] Sequence lengths: input={self.args.seq_len}, output={self.args.pred_len}, "
            f"variables={self.args.num_vars}, patches={num_patches}.\n"
            "[6] The prompt tokens are placed immediately before TS tokens so the causal backbone can use them.\n"
            "Use text, visual priors, and TS attention bias to forecast the next horizon."
        )

    def _encode_global_text(self):
        prompt = self._build_global_prompt()
        max_tokens = int(getattr(self.args, "prompt_max_tokens", 0) or 0)
        if max_tokens > 0:
            max_tokens = min(max_tokens, 64)
        txt_token_embeds, txt_attn_mask = encode_text_batch(
            self.model_base,
            self.processor,
            prompts=[prompt],
            device=str(self.device),
            max_length=max_tokens,
            pad=bool(getattr(self.args, "prompt_pad", False)),
            truncate=bool(getattr(self.args, "prompt_truncate", False)),
        )
        token_len = int(txt_attn_mask.sum().item())
        print(f"[Prompt] style=global tokens={token_len} max_tokens={getattr(self.args, 'prompt_max_tokens', None)}")
        return txt_token_embeds, txt_attn_mask

    @staticmethod
    def _top_lags(mean_series: torch.Tensor, max_lag: int, top_k: int) -> str:
        max_lag = max(1, int(max_lag))
        top_k = max(1, int(top_k))
        series = mean_series.detach().float().cpu()
        series = series - series.mean()
        denom = float(torch.sum(series ** 2).item())
        if denom <= 1e-8:
            return "none"

        scores = []
        for lag in range(1, min(max_lag, int(series.shape[0]) - 1) + 1):
            lhs = series[:-lag]
            rhs = series[lag:]
            if lhs.numel() == 0:
                continue
            corr = float(torch.sum(lhs * rhs).item()) / denom
            scores.append((lag, corr))
        if not scores:
            return "none"
        scores.sort(key=lambda item: abs(item[1]), reverse=True)
        chosen = scores[:top_k]
        return ", ".join(f"{lag}:{corr:.2f}" for lag, corr in chosen)

    def _build_sample_prompts(self, x: torch.Tensor) -> list[str]:
        dataset_name = self._resolve_dataset_name()
        dataset_lower = dataset_name.lower()
        if "etth" in dataset_lower:
            freq = "hourly"
        elif "ettm" in dataset_lower:
            freq = "15-min"
        elif "weather" in dataset_lower:
            freq = "10-min"
        elif "exchange" in dataset_lower:
            freq = "daily"
        elif "ili" in dataset_lower:
            freq = "weekly"
        else:
            freq = "unknown"

        x_cpu = x.detach().float().cpu()
        top_k = int(getattr(self.args, "prompt_topk_lags", 3))
        lag_limit = int(getattr(self.args, "prompt_lag_limit", 12))
        prompts = []
        for sample in x_cpu:
            mean_series = sample.mean(dim=1)
            flat = sample.reshape(-1)
            win_mean = float(flat.mean().item())
            win_std = float(flat.std(unbiased=False).item())
            win_min = float(flat.min().item())
            win_max = float(flat.max().item())
            win_median = float(flat.median().item())
            start_mean = float(mean_series[: max(1, mean_series.shape[0] // 4)].mean().item())
            end_mean = float(mean_series[-max(1, mean_series.shape[0] // 4):].mean().item())
            delta = end_mean - start_mean
            if delta > 0.05:
                trend = "rising"
            elif delta < -0.05:
                trend = "falling"
            else:
                trend = "stable"
            lag_text = self._top_lags(mean_series, max_lag=lag_limit, top_k=top_k)
            prompts.append(
                "Task: multivariate time-series forecasting.\n"
                f"Dataset={dataset_name}, freq={freq}, vars={self.args.num_vars}, "
                f"input={self.args.seq_len}, output={self.args.pred_len}.\n"
                f"Window stats: mean={win_mean:.4f}, std={win_std:.4f}, min={win_min:.4f}, "
                f"max={win_max:.4f}, median={win_median:.4f}.\n"
                f"Window trend={trend}, start_mean={start_mean:.4f}, end_mean={end_mean:.4f}, "
                f"dominant_lags={lag_text}.\n"
                "Use the text summary together with visual priors and TS attention bias to forecast the next horizon."
            )
        return prompts

    def _encode_sample_text(self, x: torch.Tensor):
        prompts = self._build_sample_prompts(x)
        return encode_text_batch(
            self.model_base,
            self.processor,
            prompts=prompts,
            device=str(self.device),
            max_length=int(getattr(self.args, "prompt_max_tokens", 0) or 0),
            pad=bool(getattr(self.args, "prompt_pad", True)),
            truncate=bool(getattr(self.args, "prompt_truncate", True)),
        )

    def _ensure_text_embeddings(self):
        if self.global_txt_token_embeds is None:
            self.global_txt_token_embeds = self._encode_global_text()

    @staticmethod
    def _concat_text_embeddings(base, extra):
        base_tokens, base_mask = base
        extra_tokens, extra_mask = extra
        if base_tokens.shape[0] == 1 and extra_tokens.shape[0] > 1:
            base_tokens = base_tokens.repeat(extra_tokens.shape[0], 1, 1)
            base_mask = base_mask.repeat(extra_mask.shape[0], 1)
        elif extra_tokens.shape[0] == 1 and base_tokens.shape[0] > 1:
            extra_tokens = extra_tokens.repeat(base_tokens.shape[0], 1, 1)
            extra_mask = extra_mask.repeat(base_mask.shape[0], 1)
        return (
            torch.cat([base_tokens, extra_tokens], dim=1),
            torch.cat([base_mask, extra_mask], dim=1),
        )

    def _collect_train_params(self):
        params = []
        if bool(getattr(self.args, "train_ts_mlp", True)):
            params.extend(list(self.ts_mlp.parameters()))
            params.extend(list(self.pred_head.parameters()))
        if getattr(self, "ts_time_embed", None) is not None:
            params.extend(list(self.ts_time_embed.parameters()))
        if getattr(self, "ts_var_embed", None) is not None:
            params.extend(list(self.ts_var_embed.parameters()))
        if getattr(self, "ts_normalizer", None) is not None:
            params.extend(list(self.ts_normalizer.parameters()))
        if self.model is not None:
            params.extend([p for p in self.model.parameters() if p.requires_grad])
        seen = set()
        uniq = []
        for p in params:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append(p)
        return uniq

    def _backbone_requires_grad(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def _run_backbone(
        self,
        fused_embeds,
        attn_mask,
        use_grad: bool,
        ts_attn_bias: torch.Tensor = None,
        ts_attn_layer: Optional[int] = None,
        ts_attn_bias_mode: str = "add",
        ts_attn_bias_scale: torch.Tensor = None,
    ):
        with time_block("backbone.forward"):
            with torch.set_grad_enabled(use_grad):
                if ts_attn_bias is None:
                    out = self.backbone(
                        inputs_embeds=fused_embeds,
                        attention_mask=attn_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    with ts_attn_bias_context(
                        ts_attn_bias,
                        layer_idx=ts_attn_layer,
                        bias_mode=ts_attn_bias_mode,
                        bias_scale=ts_attn_bias_scale,
                    ):
                        out = self.backbone(
                            inputs_embeds=fused_embeds,
                            attention_mask=attn_mask,
                            use_cache=False,
                            output_hidden_states=True,
                            return_dict=True,
                        )
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        if hasattr(out, "hidden_states") and out.hidden_states:
            return out.hidden_states[-1]
        # Fallback to tuple-like output
        return out[0]

    def _forward_batch(
        self,
        x: torch.Tensor,
        num_patches: int,
        train: bool,
        img_grids: torch.Tensor = None,
        vid_grids: torch.Tensor = None,
        img_tokens: torch.Tensor = None,
        img_token_mask: torch.Tensor = None,
        vid_tokens: torch.Tensor = None,
        vid_token_mask: torch.Tensor = None,
    ):
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
                img_stats = self._stats_from_img_grids(img_grids)
                vid_stats = self._stats_from_vid_grids(vid_grids)
            else:
                _images, img_stats = self._build_image_modality(x_aux, img_grids)
                _videos, vid_stats = self._build_video_modality(x_aux, num_patches, vid_grids)

        if use_vision:
            if img_tokens is not None and vid_tokens is not None:
                img_token_seqs = (img_tokens, img_token_mask)
                vid_token_seqs = (vid_tokens, vid_token_mask)
            else:
                raise RuntimeError(
                    "use_vision=True but no precomputed vision tokens found in batch. "
                    "Rebuild .npz with --precompute_vision true."
                )
        else:
            img_token_seqs, vid_token_seqs = [], []

        with time_block("fuse.modalities"):
            fused_embeds, attn_mask, ts_range, token_counts = fuse_modalities(
                model=self.model,
                ts_embeds=ts_embeds,
                img_token_seqs=img_token_seqs,
                vid_token_seqs=vid_token_seqs,
                txt_token_embeds=self._get_text_embeddings(x_aux) if use_text else None,
                device=str(self.device),
                img_norm=getattr(self.model, "img_norm", None) if use_vision else None,
                vid_norm=getattr(self.model, "vid_norm", None) if use_vision else None,
                text_norm=getattr(self.model, "text_norm", None) if use_text else None,
                img_gate=getattr(self.model, "img_gate", None) if use_vision else None,
                vid_gate=getattr(self.model, "vid_gate", None) if use_vision else None,
                text_gate=getattr(self.model, "text_gate", None) if use_text else None,
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
                token_counts=token_counts,
                cross_vision_scale=float(getattr(self.args, "ts_bias_cross_vision_scale", 0.0)),
                norm_mode=str(getattr(self.args, "ts_bias_norm", "none")).lower(),
            )
        backbone_grad = bool(train) and self._backbone_requires_grad()
        ts_attn_layer = self._resolve_ts_attn_layers()
        bias_mode = getattr(self.args, "ts_attn_bias_mode", "add")
        bias_scale = None
        if use_bias:
            if bool(getattr(self.args, "ts_bias_learnable", False)):
                bias_scale = getattr(self.model, "ts_bias_scale", None)
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
            ts_attn_bias_mode=bias_mode,
            ts_attn_bias_scale=bias_scale,
        )
        if bool(getattr(self.args, "use_ts_residual", False)) and ts_range is not None:
            ts_start, ts_end = ts_range
            if ts_start <= ts_end:
                ts_add = ts_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
                hidden_states = hidden_states.clone()
                ts_hidden = hidden_states[:, ts_start:ts_end + 1, :]

                mode = str(getattr(self.args, "ts_residual_mode", "study"))
                if mode == "study":
                    gate = getattr(self.model, "ts_residual_gate", None)
                    if gate is not None:
                        scale = torch.sigmoid(gate).to(device=hidden_states.device, dtype=hidden_states.dtype)
                    else:
                        scale = torch.tensor(1.0, device=hidden_states.device, dtype=hidden_states.dtype)
                    ts_new = ts_hidden + ts_add * scale
                else:
                    alpha = float(getattr(self.args, "ts_residual_alpha", 0.1))
                    act_name = str(getattr(self.args, "ts_residual_activation", "sigmoid")).lower()
                    if act_name == "sigmoid":
                        act = torch.sigmoid
                    elif act_name == "tanh":
                        act = torch.tanh
                    elif act_name == "relu":
                        act = torch.relu
                    else:
                        raise ValueError(f"unknown ts_residual_activation: {act_name}")

                    if mode == "A":
                        ts_new = ts_hidden + alpha * act(ts_add)
                    elif mode == "B":
                        ts_new = act(ts_hidden + alpha * ts_add)
                    else:
                        raise ValueError(f"unknown ts_residual_mode: {mode}")

                hidden_states[:, ts_start:ts_end + 1, :] = ts_new
        pred_device = next(self.pred_head.parameters()).device
        if hidden_states.device != pred_device:
            hidden_states = hidden_states.to(pred_device)
        context_summary = self._build_prediction_context(hidden_states, token_counts)

        preds = decode_ts_predictions(
            hidden_states=hidden_states,
            ts_range=ts_range,
            num_vars=self.args.num_vars,
            num_patches=num_patches,
            pred_head=self.pred_head,
            use_grad=bool(train),
            pooling=getattr(self.args, "ts_pooling", "mean"),
            tail_k=int(getattr(self.args, "ts_tail_k", 3)),
            context_summary=context_summary,
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
        return ts_embeds

    @staticmethod
    def _freeze_model(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    def _build_image_modality(self, x: torch.Tensor, img_grids: torch.Tensor):
        batch_size = x.shape[0]
        if img_grids is not None:
            img_grids_np = to_numpy(img_grids)
            images, img_stats = build_image_modality_from_grids(
                img_grids=img_grids_np,
                args=self.args,
                patch_size=self.vision_patch_size,
            )
        else:
            x_np = to_numpy(x)
            images, img_stats = build_image_modality(
                x_np=x_np,
                batch_size=batch_size,
                args=self.args,
                patch_size=self.vision_patch_size,
            )
        return images, img_stats

    def _build_video_modality(self, x: torch.Tensor, num_patches: int, vid_grids: torch.Tensor):
        batch_size = x.shape[0]
        if vid_grids is not None:
            vid_grids_np = to_numpy(vid_grids)
            videos, vid_stats = build_video_modality_from_grids(
                vid_grids=vid_grids_np,
                num_patches=num_patches,
                args=self.args,
                patch_size=self.vision_patch_size,
            )
        else:
            x_np = to_numpy(x)
            videos, vid_stats = build_video_modality(
                x_np=x_np,
                batch_size=batch_size,
                num_patches=num_patches,
                args=self.args,
                patch_size=self.vision_patch_size,
            )
        return videos, vid_stats
 
    @staticmethod
    def _stats_from_img_grids(img_grids: torch.Tensor):
        grids = to_numpy(img_grids)
        dtw = (grids[..., 0] * 2.0 - 1.0).astype(np.float32)
        cov = (grids[..., 1] * 2.0 - 1.0).astype(np.float32)
        pear = (grids[..., 2] * 2.0 - 1.0).astype(np.float32)
        return dtw, cov, pear

    @staticmethod
    def _stats_from_vid_grids(vid_grids: torch.Tensor):
        grids = to_numpy(vid_grids)
        dtw = (grids[..., 0] * 2.0 - 1.0).astype(np.float32)
        cov = (grids[..., 1] * 2.0 - 1.0).astype(np.float32)
        pear = (grids[..., 2] * 2.0 - 1.0).astype(np.float32)
        return dtw, cov, pear

    def _get_text_embeddings(self, x: torch.Tensor = None):
        prompt_style = str(getattr(self.args, "prompt_style", "sample")).strip().lower()
        self._ensure_text_embeddings()
        if prompt_style == "global":
            return self.global_txt_token_embeds
        if x is None:
            raise ValueError("sample prompt mode requires the current input window")
        sample_text = self._encode_sample_text(x)
        if prompt_style == "sample_only":
            return sample_text
        if prompt_style == "hybrid_summary":
            summary_tokens = int(getattr(self.args, "prompt_summary_tokens", 4))
            sample_text = self._compress_text_embeddings(sample_text, summary_tokens=summary_tokens)
        return self._concat_text_embeddings(self.global_txt_token_embeds, sample_text)

    @staticmethod
    def _compress_text_embeddings(text_pair, summary_tokens: int):
        tokens, mask = text_pair
        if summary_tokens <= 0 or tokens.shape[1] <= summary_tokens:
            return text_pair

        bsz, _, dim = tokens.shape
        device = tokens.device
        dtype = tokens.dtype
        out_tokens = torch.zeros((bsz, summary_tokens, dim), device=device, dtype=dtype)
        out_mask = torch.zeros((bsz, summary_tokens), device=device, dtype=mask.dtype)

        for b in range(bsz):
            valid = int(mask[b].sum().item())
            if valid <= 0:
                continue
            cur = tokens[b, :valid]
            splits = torch.tensor_split(cur, summary_tokens, dim=0)
            filled = 0
            for chunk in splits:
                if chunk.shape[0] == 0:
                    continue
                out_tokens[b, filled] = chunk.mean(dim=0)
                out_mask[b, filled] = 1
                filled += 1
                if filled >= summary_tokens:
                    break
        return out_tokens, out_mask

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

        ts_attn_layer = getattr(self.args, "ts_attn_bias_layer", None)
        if ts_attn_layer is not None and ts_attn_layer < 0:
            return None
        return ts_attn_layer
