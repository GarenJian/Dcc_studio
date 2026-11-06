"""Dynamic functional connectivity: sliding-window correlation and binarisation.

For a window of length ``SWL`` starting every ``SWS`` time points, this module
computes the Pearson correlation between each ROI signal and each network-mean
signal. The result has shape ``(n_roi, n_net, n_windows)`` with
``n_windows = 1 + (T - SWL) // SWS``.

Implementation notes
--------------------
* Correlations are computed in chunks of windows. Materialising all windows at
  once for a long time series would allocate ``W * R * T`` doubles, so the
  ``chunk`` argument bounds peak memory without changing the arithmetic.
* Each window is z-scored with ``ddof=1`` before the dot product; this matches
  the reference implementation exactly and matters for the parity tests.
* ``apply_sign`` implements ``abs`` (magnitude only), ``raw`` (keep signed
  correlations) and ``pos`` (clip negatives to zero).
* ``binarize_dfc`` thresholds at ``mean + k * std`` (``ddof=1``). ``global``
  uses one threshold for the whole tensor; ``roi`` computes one per ROI.

All functions are pure NumPy: no file IO, no global state and no randomness,
which makes them straightforward to test on synthetic arrays.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def sliding_window_fc(ts_roi: np.ndarray, ts_net: np.ndarray,
                      swl: int, sws: int, chunk: int = 128) -> np.ndarray:
    """Sliding-window correlation between ROI signals and network signals.

    ts_roi : (n_roi, T)
    ts_net : (n_net, T)
    return : (n_roi, n_net, n_win), with ``n_win = 1 + (T - swl) // sws``

    The windowed correlation is vectorised with stride views and einsum.
    """
    x = np.ascontiguousarray(np.asarray(ts_roi, dtype=float).T)
    y = np.ascontiguousarray(np.asarray(ts_net, dtype=float).T)
    timepoints, n_roi = x.shape
    n_net = y.shape[1]
    if y.shape[0] != timepoints:
        raise ValueError("ts_roi and ts_net have different time lengths")
    if swl < 2 or swl > timepoints:
        raise ValueError(f"SWL={swl} exceeds the available {timepoints} time points")
    n_win = 1 + (timepoints - swl) // sws

    xw = sliding_window_view(x, swl, axis=0)[::sws]
    yw = sliding_window_view(y, swl, axis=0)[::sws]
    out = np.empty((n_roi, n_net, n_win), dtype=float)

    for start in range(0, n_win, chunk):
        stop = min(start + chunk, n_win)
        xc = xw[start:stop] - xw[start:stop].mean(axis=2, keepdims=True)
        yc = yw[start:stop] - yw[start:stop].mean(axis=2, keepdims=True)
        num = np.einsum("wrl,wgl->wrg", xc, yc)
        den = (np.sqrt(np.einsum("wrl,wrl->wr", xc, xc))[:, :, None]
               * np.sqrt(np.einsum("wgl,wgl->wg", yc, yc))[:, None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = num / den
        out[:, :, start:stop] = np.nan_to_num(
            corr, nan=0.0, posinf=0.0, neginf=0.0).transpose(1, 2, 0)
    return out


def apply_sign(dfc: np.ndarray, sign_type: str) -> np.ndarray:
    """Apply abs / raw / pos sign handling without changing the dFC shape."""
    if sign_type == "abs":
        return np.abs(dfc)
    if sign_type == "raw":
        return dfc
    if sign_type == "pos":
        return np.maximum(dfc, 0.0)
    raise ValueError("sign_type must be abs / raw / pos")


def binarize_dfc(dfc: np.ndarray, k: float, mode: str = "global") -> tuple:
    """Threshold correlations at ``mean + k * std`` (std with ddof=1).

    mode="global": one threshold for the whole array.
    mode="roi"   : a separate threshold per ROI.
    Returns ``(binary, threshold, mean, std)``.
    """
    if dfc.ndim != 3:
        raise ValueError("dFC must be a (n_roi, n_net, n_win) array")
    data = np.asarray(dfc, dtype=float)
    if mode == "global":
        mu = float(np.nanmean(data))
        sigma = float(np.nanstd(data, ddof=1))
        thr = mu + k * sigma
        binary = data > thr
    elif mode == "roi":
        flat = data.reshape(data.shape[0], -1)
        mu = np.nanmean(flat, axis=1)
        sigma = np.nanstd(flat, axis=1, ddof=1)
        thr = mu + k * sigma
        binary = data > thr[:, None, None]
    else:
        raise ValueError("mode must be global / roi")
    return binary, thr, mu, sigma
