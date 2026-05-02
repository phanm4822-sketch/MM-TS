from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from utils import time_block


def load_qwen3_vl(model_dir: str, device_map: str = "auto"):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype="auto",
        attn_implementation="eager",
        device_map=device_map,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    return model, processor


def freeze_qwen3_vl(model, freeze_lm: bool = True, freeze_vit: bool = True) -> None:
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("model has no .model attribute; expected Qwen3VLForConditionalGeneration")
    if freeze_lm:
        for p in base.language_model.parameters():
            p.requires_grad = False
    if freeze_vit:
        for p in base.visual.parameters():
            p.requires_grad = False


def _parse_layer_list(value: Optional[str], num_layers: Optional[int]) -> Optional[list[int]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    indices = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if num_layers is not None and idx < 0:
            idx = num_layers + idx
        if num_layers is not None and (idx < 0 or idx >= num_layers):
            raise ValueError(f"layer index out of range: {idx} (num_layers={num_layers})")
        indices.append(idx)
    if not indices:
        return None
    return sorted(set(indices))


def apply_qwen3_vl_lora(model, args):
    if not getattr(args, "use_lora", False):
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise RuntimeError("PEFT is required for LoRA. Install peft in your environment.") from exc

    target_modules = [
        item.strip()
        for item in str(getattr(args, "lora_target_modules", "")).split(",")
        if item.strip()
    ]
    if not target_modules:
        raise ValueError("lora_target_modules must be non-empty when use_lora is True")

    num_layers = getattr(model.config.text_config, "num_hidden_layers", None)
    layers = _parse_layer_list(getattr(args, "lora_target_layers", None), num_layers=num_layers)

    layers_pattern = None
    if layers is not None and any(".language_model.layers." in name for name, _ in model.named_modules()):
        layers_pattern = "language_model.layers"

    lora_cfg = LoraConfig(
        r=int(getattr(args, "lora_r", 8)),
        lora_alpha=int(getattr(args, "lora_alpha", 8)),
        lora_dropout=float(getattr(args, "lora_dropout", 0.0)),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        layers_to_transform=layers,
        layers_pattern=layers_pattern,
    )
    model = get_peft_model(model, lora_cfg)
    lora_param_tensors = sum(
        1 for name, _ in model.named_parameters() if "lora_A" in name or "lora_B" in name
    )
    print(
        "[LoRA] enabled=True "
        f"r={lora_cfg.r} alpha={lora_cfg.lora_alpha} dropout={lora_cfg.lora_dropout} "
        f"targets={target_modules} layers={layers} pattern={layers_pattern} "
        f"lora_param_tensors={lora_param_tensors}"
    )
    return model

def encode_images_qwen3vl(
    model,
    processor,
    images: List[Image.Image],
    device: str,
    micro_bs: int = 8,
) -> List[torch.Tensor]:
    model.eval()
    base = model.model
    if isinstance(base, torch.nn.DataParallel):
        base = base.module

    all_token_seqs = []

    with time_block("vision.image.total"):
        with torch.inference_mode():
            for start in range(0, len(images), micro_bs):
                end = min(len(images), start + micro_bs)
                chunk = images[start:end]
                with time_block(f"vision.image.preprocess[{start}:{end}]"):
                    try:
                        enc = processor.image_processor(
                            images=chunk,
                            return_tensors="pt",
                            do_resize=False,
                            do_center_crop=False,
                        )
                    except TypeError:
                        enc = processor.image_processor(images=chunk, return_tensors="pt")

                keys = list(enc.keys())
                pixel_values = enc.get("pixel_values", None)
                image_grid_thw = enc.get("image_grid_thw", None)

                if pixel_values is None or image_grid_thw is None:
                    raise RuntimeError(f"processor did not return pixel_values/image_grid_thw. got keys={keys}")

                pixel_values = pixel_values.to(device)
                image_grid_thw = image_grid_thw.to(device)

                with time_block(f"vision.image.encode[{start}:{end}]"):
                    if hasattr(base, "get_image_features"):
                        img_feats_list, _deep = base.get_image_features(pixel_values, image_grid_thw)
                    else:
                        raise RuntimeError("model.model has no get_image_features; check transformers/model class")

                for feats in img_feats_list:
                    all_token_seqs.append(feats.detach().cpu())
    return all_token_seqs


def encode_videos_qwen3vl(
    model,
    processor,
    videos: List[List[Image.Image]],
    device: str,
    micro_bs: int = 2,
) -> List[torch.Tensor]:
    model.eval()
    base = model.model
    if isinstance(base, torch.nn.DataParallel):
        base = base.module

    all_token_seqs = []

    with time_block("vision.video.total"):
        with torch.inference_mode():
            for start in range(0, len(videos), micro_bs):
                end = min(len(videos), start + micro_bs)
                chunk = videos[start:end]
                with time_block(f"vision.video.preprocess[{start}:{end}]"):
                    try:
                        enc = processor(videos=chunk, return_tensors="pt")
                    except Exception:
                        enc = processor.video_processor(
                            videos=chunk,
                            return_tensors="pt",
                            do_sample_frames=False,
                            do_resize=False,
                        )

                keys = list(enc.keys())
                pixel_values_videos = enc.get("pixel_values_videos", None)
                video_grid_thw = enc.get("video_grid_thw", None)

                if pixel_values_videos is None or video_grid_thw is None:
                    raise RuntimeError(f"processor did not return pixel_values_videos/video_grid_thw. got keys={keys}")

                pixel_values_videos = pixel_values_videos.to(device)
                video_grid_thw = video_grid_thw.to(device)

                with time_block(f"vision.video.encode[{start}:{end}]"):
                    if hasattr(base, "get_video_features"):
                        vid_feats_list, _deep = base.get_video_features(pixel_values_videos, video_grid_thw)
                    else:
                        raise RuntimeError("model.model has no get_video_features; check transformers/model class")

                for feats in vid_feats_list:
                    all_token_seqs.append(feats.detach().cpu())
    return all_token_seqs


def encode_text_global(
    model,
    processor,
    prompt: str,
    device: str,
    max_length: int = 0,
    pad: bool = False,
    truncate: bool = False,
    return_length: bool = False,
):
    kwargs = {"return_tensors": "pt"}
    if max_length and max_length > 0:
        if pad:
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = int(max_length)
        if truncate:
            kwargs["truncation"] = True
            kwargs["max_length"] = int(max_length)
    enc = processor.tokenizer(prompt, **kwargs)
    input_ids = enc.get("input_ids", None)
    attn_mask = enc.get("attention_mask", None)
    if input_ids is None:
        raise RuntimeError("tokenizer did not return input_ids")

    input_ids = input_ids.to(device)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    with time_block("text.embed"):
        with torch.inference_mode():
            base = model.model
            if isinstance(base, torch.nn.DataParallel):
                base = base.module
            emb_layer = base.get_input_embeddings()
            token_embeds = emb_layer(input_ids)
    token_embeds = token_embeds.detach().cpu()
    if return_length:
        if attn_mask is not None:
            content_len = int(attn_mask.sum().item())
        else:
            content_len = int(input_ids.shape[1])
        return token_embeds, content_len
    return token_embeds


def encode_text_batch(
    model,
    processor,
    prompts: List[str],
    device: str,
    max_length: int = 0,
    pad: bool = True,
    truncate: bool = True,
):
    if not prompts:
        raise ValueError("prompts must be non-empty")

    kwargs = {"return_tensors": "pt"}
    if max_length and max_length > 0:
        kwargs["max_length"] = int(max_length)
        if pad:
            kwargs["padding"] = "max_length"
        else:
            kwargs["padding"] = True
        if truncate:
            kwargs["truncation"] = True
    else:
        kwargs["padding"] = True
        if truncate:
            kwargs["truncation"] = True

    enc = processor.tokenizer(prompts, **kwargs)
    input_ids = enc.get("input_ids", None)
    attn_mask = enc.get("attention_mask", None)
    if input_ids is None:
        raise RuntimeError("tokenizer did not return input_ids")

    input_ids = input_ids.to(device)
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids, dtype=torch.long)
    else:
        attn_mask = attn_mask.to(device)

    with time_block("text.embed.batch"):
        with torch.inference_mode():
            base = model.model
            if isinstance(base, torch.nn.DataParallel):
                base = base.module
            emb_layer = base.get_input_embeddings()
            token_embeds = emb_layer(input_ids)

    return token_embeds.detach().cpu(), attn_mask.detach().cpu()
