from models.qwen3_vl_utils import encode_images_qwen3vl, encode_videos_qwen3vl


def encode_vision_modalities(
    model,
    processor,
    images,
    videos,
    vision_mb: int,
    device: str,
):
    img_token_seqs = encode_images_qwen3vl(
        model=model,
        processor=processor,
        images=images,
        device=device,
        micro_bs=vision_mb,
    )
    vid_token_seqs = encode_videos_qwen3vl(
        model=model,
        processor=processor,
        videos=videos,
        device=device,
        micro_bs=max(1, vision_mb // 4),
    )

    if not img_token_seqs or not vid_token_seqs:
        raise RuntimeError("empty vision token sequences; check input images/videos")

    return img_token_seqs, vid_token_seqs
