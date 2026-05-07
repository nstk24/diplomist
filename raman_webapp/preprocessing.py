from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


DEFAULT_SAVGOL_WINDOW = 11
DEFAULT_SAVGOL_POLYORDER = 3


def als_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    length = y.size
    d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2), format="csc")
    w = np.ones(length)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, length, length, format="csc")
        z = spsolve(W + lam * (d @ d.T), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def snv(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-12)


def smooth_savgol(
    x: np.ndarray,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Savitzky-Golay smoothing expects a one-dimensional spectrum.")
    if x.size < 5:
        return x.copy()

    max_window = x.size if x.size % 2 == 1 else x.size - 1
    window = min(int(window_length), max_window)
    if window < 5:
        return x.copy()
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return x.copy()

    return savgol_filter(x, window_length=window, polyorder=polyorder)


def batch_smooth_savgol(
    X: np.ndarray,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.vstack(
        [
            smooth_savgol(row, window_length=window_length, polyorder=polyorder)
            for row in X
        ]
    )


def snr_raman(
    x: np.ndarray,
    wn: np.ndarray,
    signal_region: tuple[float, float] = (400, 1700),
    noise_region: tuple[float, float] = (1800, 1950),
    k: float = 6.0,
    baseline_func=None,
) -> float:
    x = np.asarray(x, dtype=float)

    if baseline_func is not None:
        b = baseline_func(x)
        x = x - b

    sig_mask = (wn >= signal_region[0]) & (wn <= signal_region[1])
    noise_mask = (wn >= noise_region[0]) & (wn <= noise_region[1])

    signal = np.max(x[sig_mask])
    sigma = np.std(x[noise_mask], ddof=1)

    return float(signal / (k * (sigma + 1e-12)))


def batch_snr_raman(
    X: np.ndarray,
    wn: np.ndarray,
    signal_region: tuple[float, float] = (400, 1700),
    noise_region: tuple[float, float] = (1800, 1950),
    k: float = 6.0,
    baseline_func=None,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.array(
        [
            snr_raman(
                row,
                wn,
                signal_region=signal_region,
                noise_region=noise_region,
                k=k,
                baseline_func=baseline_func,
            )
            for row in X
        ],
        dtype=float,
    )
