"""Network-level aggregation and connectivity-state analysis.

The modules above produce one value (and one state distribution) per ROI. This
module lifts those ROI-wise quantities to networks and to the whole cohort:

* ``net_analysis``         -- sum/mean of a per-ROI vector per network;
* ``group_state_prob``     -- mean state probability across samples;
* ``network_state_prob``   -- average ROI state probabilities within a network;
* ``high_frequency_states`` / ``network_high_frequency_states`` -- states whose
  probability exceeds a configurable threshold, decoded back into the binary
  ``G``-vector of participating networks;
* ``top_state_coverage``   -- fraction of state occurrences explained by the
  most frequent states (``top_state_fraction``).

State decoding relies on :func:`dcc.complexity.decode_state`, so the bit order
is defined in exactly one place: the first network is the most significant bit.
Every function here is aggregation only; no statistical inference happens in
this module (that belongs to ``dcc.stats``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .atlas import Atlas
from .complexity import decode_state
from .timeseries import get_ts_roi


def net_analysis(values: np.ndarray, roi_idx: np.ndarray) -> tuple:
    """Return (sum per network, mean per network)."""
    values = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(roi_idx).ravel()
    nets = np.unique(labels[labels != 0])
    sums = np.array([values[labels == g].sum() for g in nets])
    means = np.array([values[labels == g].mean() for g in nets])
    return sums, means


def group_state_prob(results: list) -> np.ndarray:
    """Mean state probability across samples -> (n_roi, n_states)."""
    return np.mean(np.stack([r.states_prob for r in results], axis=0), axis=0)


def stack_state_prob(results: list) -> np.ndarray:
    """State probability of every sample -> (n_roi, n_states, n_samples)."""
    return np.stack([r.states_prob for r in results], axis=2)


def network_state_prob(group_prob: np.ndarray, atlas: Atlas) -> np.ndarray:
    """Average node state probabilities within each network -> (n_net, n_states)."""
    return get_ts_roi(group_prob, atlas.roi_idx)


def keep_above(values: np.ndarray, thr: float) -> np.ndarray:
    """Set values below a threshold to zero."""
    values = np.asarray(values, dtype=float)
    return np.where(values >= thr, values, 0.0)


def state_table(mean_prob: np.ndarray, thr: float, n_net: int = 7) -> pd.DataFrame:
    """Table of high-frequency states: code, probability, binary pattern."""
    mean_prob = np.asarray(mean_prob, dtype=float).ravel()
    order = np.argsort(-mean_prob)
    rows = []
    for idx in order:
        if mean_prob[idx] < thr:
            continue
        code = int(idx) + 1  # index 0 corresponds to state code 1 (all-zero removed)
        pattern = decode_state(code, n_net)
        rows.append({
            "state_index": int(idx) + 1,
            "state_code": code,
            "mean_probability": float(mean_prob[idx]),
            "binary_pattern": pattern,
            "active_networks": ",".join(
                f"N{j + 1}" for j, ch in enumerate(pattern) if ch == "1"),
        })
    return pd.DataFrame(rows, columns=[
        "state_index", "state_code", "mean_probability",
        "binary_pattern", "active_networks"])


def high_frequency_states(group_prob: np.ndarray, thr: float,
                          n_net: int = 7) -> pd.DataFrame:
    """Whole-brain high-frequency states, ranked by mean probability."""
    return state_table(group_prob.mean(axis=0), thr, n_net)


def network_high_frequency_states(net_prob: np.ndarray, thr: float,
                                  atlas: Atlas) -> dict:
    """High-frequency states computed separately for each network."""
    return {atlas.name(i): state_table(net_prob[i], thr, atlas.n_net)
            for i in range(atlas.n_net)}


def top_state_coverage(group_prob: np.ndarray, fraction: float = 0.1) -> dict:
    """Fraction of all state occurrences covered by the top states."""
    prob = np.asarray(group_prob, dtype=float).mean(axis=0)
    order = np.argsort(-prob)
    k = max(1, int(round(len(prob) * fraction)))
    total = prob.sum()
    return {
        "n_states": int(len(prob)),
        "top_k": int(k),
        "top_coverage": float(prob[order[:k]].sum() / total) if total > 0 else 0.0,
        "top_states": [int(i + 1) for i in order[:k]],
        "top_probabilities": [float(prob[i]) for i in order[:k]],
    }
