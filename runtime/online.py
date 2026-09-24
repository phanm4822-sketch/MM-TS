"""Online forecasting with the same relation grids and visual-token precision as cache construction."""

import numpy as np
import torch


@torch.no_grad()
def predict_window(model, standardized_windows):
    """Predict [B, H, C] from float32 [B, L, C] standardized observed windows.

    Fit channel means/stds on training data only and apply them before calling.
    This function does not fit a scaler or change model precision. Call eval()
    first. Graph outputs share storage: clone if retaining across later calls.
    """
    from data_provider.cache.common import build_relation_grids
    from layers.modality_builder import (
        build_image_modality_from_grids,
        build_video_modality_from_grids,
    )

    if model.training:
        raise ValueError("call model.eval() before online prediction")
    windows = np.asarray(standardized_windows)
    args = model.args
    if (
        windows.dtype != np.float32
        or windows.ndim != 3
        or windows.shape[1:] != (args.seq_len, args.num_vars)
    ):
        raise ValueError("expected float32 windows with shape [B, seq_len, num_vars]")
    features = {}
    if args.use_vision or args.use_ts_attn_bias:
        pairs = [build_relation_grids(x, args.patch_len, args.stride) for x in windows]
        image_grids = np.stack([p[0] for p in pairs]).astype(np.float32)
        video_grids = np.stack([p[1] for p in pairs]).astype(np.float32)
        features.update(
            img_grids=torch.from_numpy(image_grids),
            vid_grids=torch.from_numpy(video_grids),
        )
        if args.use_vision:
            backbone = model.backbone
            if next(backbone.visual.parameters()).device.type != model.device.type:
                raise ValueError("online prediction requires cached_visual=False")
            patch_size = backbone.config.vision_config.patch_size
            images, _ = build_image_modality_from_grids(image_grids, args, patch_size)
            videos, _ = build_video_modality_from_grids(
                video_grids, video_grids.shape[1], args, patch_size
            )
            processor = model.processor
            # Frozen visual features use native backbone precision, then the
            # same float16 storage round-trip as the on-disk cache.
            with torch.autocast(model.device.type, enabled=False):
                enc = processor.image_processor(
                    images=images, return_tensors="pt", do_resize=False
                )
                image, _ = backbone.get_image_features(
                    enc["pixel_values"].to(model.device), enc["image_grid_thw"]
                )
                image = torch.stack(
                    [x.detach().to(torch.float16).to(torch.float32) for x in image]
                )
                enc = processor.video_processor(
                    videos=videos,
                    return_tensors="pt",
                    do_sample_frames=False,
                    do_resize=False,
                )
                video, _ = backbone.get_video_features(
                    enc["pixel_values_videos"].to(model.device), enc["video_grid_thw"]
                )
                video = torch.stack(
                    [x.detach().to(torch.float16).to(torch.float32) for x in video]
                )
            features.update(
                img_tokens=image,
                vid_tokens=video,
                img_token_mask=torch.ones(
                    image.shape[:2], dtype=torch.long, device=model.device
                ),
                vid_token_mask=torch.ones(
                    video.shape[:2], dtype=torch.long, device=model.device
                ),
            )
    return model(x=torch.from_numpy(windows).to(model.device), **features)
