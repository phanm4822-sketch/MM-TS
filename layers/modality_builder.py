import math
from typing import Tuple

import numpy as np
from PIL import Image

from utils.heatmap_render import render_rgb_grid_to_image


def _compute_render_params(
    num_vars: int,
    patch_size: int,
    merge_size: int = 2,
    min_render_size: int | None = None,
) -> Tuple[int, int]:
    align = int(patch_size) * max(1, int(merge_size))
    base = align if min_render_size is None else max(int(min_render_size), align)
    target_size = int(math.ceil(base / align) * align)
    if num_vars <= target_size:
        cell_pix = max(1, target_size // num_vars)
    else:
        target_size = int(math.ceil(num_vars / align) * align)
        cell_pix = 1
    return cell_pix, target_size


def _pad_image_to_square(img: Image.Image, target_size: int) -> Image.Image:
    if img.size[0] == target_size and img.size[1] == target_size:
        return img
    if img.size[0] > target_size or img.size[1] > target_size:
        raise ValueError(f"image size {img.size} exceeds target {target_size}")
    out = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    out.paste(img, (0, 0))
    return out


def build_image_modality_from_grids(
    img_grids: np.ndarray,
    args,
    patch_size: int,
) -> Tuple[list, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    images = []
    dtw_corrs = []
    cov_corrs = []
    pear_corrs = []
    cell_pix, target_size = _compute_render_params(
        int(img_grids.shape[1]),
        patch_size,
        min_render_size=int(getattr(args, "vision_render_size", patch_size * 2)),
    )
    for i in range(img_grids.shape[0]):
        grid = img_grids[i]
        img = render_rgb_grid_to_image(grid, cell_pix=cell_pix)
        img = _pad_image_to_square(img, target_size)
        images.append(img)
        dtw_corrs.append((grid[..., 0] * 2.0 - 1.0).astype(np.float32))
        cov_corrs.append((grid[..., 1] * 2.0 - 1.0).astype(np.float32))
        pear_corrs.append((grid[..., 2] * 2.0 - 1.0).astype(np.float32))
    return images, (
        np.stack(dtw_corrs, axis=0),
        np.stack(cov_corrs, axis=0),
        np.stack(pear_corrs, axis=0),
    )


def build_video_modality_from_grids(
    vid_grids: np.ndarray,
    num_patches: int,
    args,
    patch_size: int,
) -> Tuple[list, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    videos = []
    dtw_corrs = []
    cov_corrs = []
    pear_corrs = []
    cell_pix, target_size = _compute_render_params(
        int(vid_grids.shape[2]),
        patch_size,
        min_render_size=int(getattr(args, "vision_render_size", patch_size * 2)),
    )
    for i in range(vid_grids.shape[0]):
        frames = []
        frame_dtw = []
        frame_cov = []
        frame_pear = []
        for p in range(num_patches):
            grid = vid_grids[i, p]
            frame = render_rgb_grid_to_image(grid, cell_pix=cell_pix)
            frame = _pad_image_to_square(frame, target_size)
            frames.append(frame)
            frame_dtw.append((grid[..., 0] * 2.0 - 1.0).astype(np.float32))
            frame_cov.append((grid[..., 1] * 2.0 - 1.0).astype(np.float32))
            frame_pear.append((grid[..., 2] * 2.0 - 1.0).astype(np.float32))

        videos.append(frames)
        dtw_corrs.append(np.stack(frame_dtw, axis=0))
        cov_corrs.append(np.stack(frame_cov, axis=0))
        pear_corrs.append(np.stack(frame_pear, axis=0))

    return videos, (
        np.stack(dtw_corrs, axis=0),
        np.stack(cov_corrs, axis=0),
        np.stack(pear_corrs, axis=0),
    )
