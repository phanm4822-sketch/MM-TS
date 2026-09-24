import math
from contextlib import contextmanager

import numpy as np

try:
    from numba import njit
except ImportError:

    def njit(**kwargs):
        return lambda function: function


@njit(cache=True, fastmath=False)
def dtw_distance_banded(a, b, band=8):
    length = len(a)
    dp = np.full((length + 1, length + 1), 1e18, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, length + 1):
        for j in range(max(1, i - band), min(length, i + band) + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(math.sqrt(dp[length, length]))


def compute_three_mats(window, kind, dtw_band=8, dtw_eps=1e-8, verbose=False):
    if verbose:
        from utils.ts_stats import compute_three_mats as original

        return original(window, kind, dtw_band, dtw_eps, verbose=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pear = np.corrcoef(window, rowvar=False)
    pear = np.nan_to_num(pear, nan=0.0, posinf=0.0, neginf=0.0)
    centered = window - window.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(window.shape[0] - 1, 1)
    # Normalize contiguous float64 channel vectors.
    z = np.ascontiguousarray(window.T, dtype=np.float64)
    z = (z - z.mean(axis=1, keepdims=True)) / np.maximum(
        z.std(axis=1, keepdims=True), dtw_eps
    )
    channels = len(z)
    dtw = np.zeros((channels, channels), dtype=np.float64)
    band = min(int(dtw_band), max(1, window.shape[0]))
    for i in range(channels):
        for j in range(i + 1, channels):
            distance = dtw_distance_banded(z[i], z[j], band)
            dtw[i, j] = dtw[j, i] = distance
    return dtw, cov, pear


def normalize_cov_to_01(cov, lo, hi):
    lower, upper = np.percentile(cov.reshape(-1), [lo, hi])
    if upper <= lower:
        upper = lower + 1e-6
    return np.clip((np.clip(cov, lower, upper) - lower) / (upper - lower), 0.0, 1.0)


@contextmanager
def accelerated_relations():
    import data_provider.cache.common as common
    import utils.heatmap_render as heat

    old_stats, old_cov = common.compute_three_mats, heat.normalize_cov_to_01
    common.compute_three_mats = compute_three_mats
    heat.normalize_cov_to_01 = normalize_cov_to_01
    try:
        yield
    finally:
        common.compute_three_mats, heat.normalize_cov_to_01 = old_stats, old_cov
