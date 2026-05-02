import math
from typing import Tuple

import numpy as np


def z_norm_1d(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = a.mean()
    s = a.std()
    s = max(s, eps)
    return (a - m) / s


def dtw_distance_banded(a: np.ndarray, b: np.ndarray, band: int = 8) -> float:
    T = len(a)
    INF = 1e18
    dp = np.full((T + 1, T + 1), INF, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, T + 1):
        j_start = max(1, i - band)
        j_end = min(T, i + band)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = (ai - b[j - 1]) ** 2
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(math.sqrt(dp[T, T]))


def _fft_window_features(window: np.ndarray, fft_demean: bool) -> np.ndarray:
    # Map [T, C] time-domain window to non-negative FFT magnitude features [F, C].
    x = window.astype(np.float64, copy=False)
    if fft_demean:
        x = x - x.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(x, axis=0)
    return np.abs(spec)


def compute_three_mats(
    window: np.ndarray,
    kind: str,
    dtw_band: int = 8,
    dtw_eps: float = 1e-8,
    fft_dtw: bool = False,
    fft_cov: bool = False,
    fft_pear: bool = False,
    fft_demean: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    window: [T, C] where C is the channel count
    Returns:
      dtw_mat, cov_mat, pearson_mat, each [C, C]
    """
    T, C = window.shape

    fft_feats = None

    def _pick_view(use_fft: bool) -> np.ndarray:
        nonlocal fft_feats
        if not use_fft:
            return window
        if fft_feats is None:
            fft_feats = _fft_window_features(window, fft_demean=fft_demean)
        return fft_feats

    pear_view = _pick_view(fft_pear)
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = np.corrcoef(pear_view, rowvar=False)
    pearson = np.nan_to_num(pearson, nan=0.0, posinf=0.0, neginf=0.0)

    cov_view = _pick_view(fft_cov)
    T_cov = cov_view.shape[0]
    w0 = cov_view - cov_view.mean(axis=0, keepdims=True)
    cov = (w0.T @ w0) / max(T_cov - 1, 1)

    dtw_view = _pick_view(fft_dtw)
    dtw = np.zeros((C, C), dtype=np.float64)
    zch = [z_norm_1d(dtw_view[:, i].astype(np.float64), eps=dtw_eps) for i in range(C)]
    dtw_band_eff = min(int(dtw_band), max(1, dtw_view.shape[0]))
    for i in range(C):
        dtw[i, i] = 0.0
        for j in range(i + 1, C):
            d = dtw_distance_banded(zch[i], zch[j], band=dtw_band_eff)
            dtw[i, j] = d
            dtw[j, i] = d

    if verbose:
        print(
            f"[{kind}] three mats computed: pearson range=({pearson.min():.3f},{pearson.max():.3f}), "
            f"cov range=({cov.min():.3g},{cov.max():.3g}), dtw range=({dtw.min():.3g},{dtw.max():.3g}), "
            f"fft(dtw/cov/pear)=({fft_dtw}/{fft_cov}/{fft_pear}), fft_demean={fft_demean}"
        )
    return dtw, cov, pearson


def normalize_pearson_to_01(pearson: np.ndarray) -> np.ndarray:
    x = (pearson + 1.0) * 0.5
    return np.clip(x, 0.0, 1.0)


def normalize_cov_to_01(cov: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = cov.reshape(-1)
    a = np.percentile(v, lo)
    b = np.percentile(v, hi)
    if b <= a:
        b = a + 1e-6
    x = np.clip(cov, a, b)
    x = (x - a) / (b - a)
    return np.clip(x, 0.0, 1.0)


def normalize_dtw_to_01(dtw: np.ndarray, tau: float) -> np.ndarray:
    sim = np.exp(-dtw / max(tau, 1e-6))
    v = sim.reshape(-1)
    a, b = float(v.min()), float(v.max())
    if b <= a:
        return np.zeros_like(sim, dtype=np.float32)
    x = (sim - a) / (b - a)
    return np.clip(x, 0.0, 1.0).astype(np.float32)
