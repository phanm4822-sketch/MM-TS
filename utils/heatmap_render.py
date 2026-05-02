import numpy as np
from PIL import Image

from .ts_stats import normalize_dtw_to_01, normalize_cov_to_01, normalize_pearson_to_01


def mats_to_rgb_grid(
    dtw: np.ndarray,
    cov: np.ndarray,
    pearson: np.ndarray,
    cov_clip_lo: float,
    cov_clip_hi: float,
    dtw_tau: float,
) -> np.ndarray:
    if dtw.ndim != 2 or cov.ndim != 2 or pearson.ndim != 2:
        raise ValueError("dtw/cov/pearson must be 2D square matrices")
    C = dtw.shape[0]
    if dtw.shape != (C, C) or cov.shape != (C, C) or pearson.shape != (C, C):
        raise ValueError("dtw/cov/pearson must have matching square shapes")

    dtw01 = normalize_dtw_to_01(dtw, tau=dtw_tau)
    cov01 = normalize_cov_to_01(cov, lo=cov_clip_lo, hi=cov_clip_hi)
    pear01 = normalize_pearson_to_01(pearson)

    rgb = np.stack([dtw01, cov01, pear01], axis=-1)
    assert rgb.shape == (C, C, 3)
    return rgb.astype(np.float32)


def render_rgb_grid_to_image(rgb: np.ndarray, cell_pix: int) -> Image.Image:
    """
    rgb: [C,C,3] values in [0,1]
    returns PIL RGB image size (C*cell_pix, C*cell_pix)
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] != rgb.shape[1]:
        raise ValueError("rgb must be [C, C, 3]")
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    img_arr = np.repeat(np.repeat(arr, cell_pix, axis=0), cell_pix, axis=1)
    return Image.fromarray(img_arr, mode="RGB")
