"""Core DCC computation: connectivity states -> probability -> Shannon entropy.

This is the numerical heart of the built-in method. :func:`connectome_complex`
turns one sample's ``(n_roi, n_timepoints)`` matrix into one value per ROI:

1. Average ROI signals inside each atlas network (``Atlas.aggregate``).
2. Compute sliding-window Pearson correlations between every ROI and every
   network signal (``dcc.dfc.sliding_window_fc``), giving ``(R, G, W)``.
3. Apply sign handling (``abs`` / ``raw`` / ``pos``), then binarise with
   ``value > mean + k * std`` where ``std`` uses ``ddof=1``. In ``global``
   mode the mean/std are taken over the whole tensor; in ``roi`` mode they are
   taken per ROI.
4. Interpret each window's binary ``G``-vector as a bit pattern
   (``2^(G-1) ... 2^0``), discard the all-zero pattern, and count how often
   each remaining state occurs at each ROI.
5. Report ``-sum(p * log2(p))`` over non-zero state probabilities, in bits.

A value of 0 means the ROI always occupies the same connectivity state; a
high value means it visits many states with comparable frequency.

``SubjectResult`` is deliberately generic despite its field names: any
registered method can reuse it as the per-sample result container that the
workflow, cache and figures understand.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .atlas import Atlas
from .config import DCCConfig
from .dfc import apply_sign, binarize_dfc, sliding_window_fc
from .timeseries import get_ts_roi


def state_weights(n_net: int) -> np.ndarray:
    """Bit weights ``[2^(n-1), ..., 2, 1]``; the first network is the high bit."""
    return 2 ** np.arange(n_net - 1, -1, -1)


def decode_state(code: int, n_net: int = 7) -> str:
    """Integer state code -> binary pattern of network participation."""
    return format(int(code), f"0{n_net}b")


@dataclass
class SubjectResult:
    """DCC result of one sample (one time-series matrix)."""

    entropy: np.ndarray                        # (n_roi,)
    states_prob: np.ndarray                    # (n_roi, 2**n_net - 1)
    states_counts: np.ndarray                  # (n_roi, 2**n_net - 1)
    dfc_binary: np.ndarray | None = None       # (n_roi, n_net, n_win)
    threshold: float = np.nan
    mu: float = np.nan
    sigma: float = np.nan
    n_windows: int = 0
    subject_id: str = ""

    @property
    def n_roi(self) -> int:
        """Number of ROI values stored in this per-sample result."""
        return self.entropy.shape[0]

    def to_dict(self, save_binary: bool = False) -> dict:
        """Serialise the result into the flat dictionary used by the NPZ cache."""
        data = {
            "entropy": self.entropy,
            "states_prob": self.states_prob,
            "states_counts": self.states_counts.astype(np.uint16),
            "threshold": np.array([self.threshold]),
            "mu": np.array([self.mu]),
            "sigma": np.array([self.sigma]),
            "n_windows": np.array([self.n_windows]),
            "subject_id": np.array([self.subject_id]),
        }
        if save_binary and self.dfc_binary is not None:
            data["dfc_binary"] = self.dfc_binary.astype(np.uint8)
        return data

    @classmethod
    def from_dict(cls, d: dict) -> "SubjectResult":
        """Rebuild a SubjectResult from a cached dictionary with its original dtypes."""
        get = lambda key, default: np.asarray(d[key]).item() if key in d else default
        return cls(
            entropy=np.asarray(d["entropy"], dtype=float).ravel(),
            states_prob=np.asarray(d["states_prob"], dtype=float),
            states_counts=np.asarray(d["states_counts"]),
            dfc_binary=(np.asarray(d["dfc_binary"]) if "dfc_binary" in d else None),
            threshold=float(get("threshold", np.nan)),
            mu=float(get("mu", np.nan)),
            sigma=float(get("sigma", np.nan)),
            n_windows=int(get("n_windows", 0)),
            subject_id=str(get("subject_id", "")),
        )


def compute_states(dfc_binary: np.ndarray, n_net: int) -> tuple:
    """Encode binary dFC as states and count their occurrence.

    ``dfc_binary`` has shape ``(n_roi, n_net, n_win)``. The all-zero state is
    discarded, leaving ``2**n_net - 1`` states. Returns
    ``(codes, counts, prob, entropy)``.
    """
    binary = np.asarray(dfc_binary, dtype=np.uint8)
    if binary.ndim != 3 or binary.shape[1] != n_net:
        raise ValueError(f"binary dFC must have shape (n_roi, {n_net}, n_win), "
                         f"got {binary.shape}")
    n_roi, _, n_win = binary.shape
    weights = state_weights(n_net)
    codes = np.tensordot(binary.transpose(0, 2, 1).astype(np.int32),
                         weights.astype(np.int32), axes=([2], [0]))
    n_all = 2 ** n_net
    counts = np.zeros((n_roi, n_all), dtype=np.int32)
    np.add.at(counts, (np.repeat(np.arange(n_roi), n_win), codes.ravel()), 1)
    counts = counts[:, 1:]
    total = counts.sum(axis=1, keepdims=True)
    prob = np.divide(counts, np.maximum(total, 1), dtype=float)
    positive = np.where(prob > 0, prob, 1.0)
    entropy = -np.sum(np.where(prob > 0, prob * np.log2(positive), 0.0), axis=1)
    return codes, counts.astype(np.uint16), prob, entropy


def connectome_complex(ts: np.ndarray, atlas: Atlas, cfg: DCCConfig,
                       swl: int | None = None, subject_id: str = "") -> SubjectResult:
    """Compute the dynamic connectivity complexity of one sample.

    1. Average ROI signals within each atlas network.
    2. Sliding-window Pearson correlation between ROIs and networks.
    3. Sign handling and sparsification with ``mean + k * std``.
    4. Encode the resulting binary pattern per window and take the Shannon
       entropy of its frequency distribution.
    """
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 2 or ts.shape[0] != atlas.n_roi:
        raise ValueError(f"ts must have shape ({atlas.n_roi}, T), got {ts.shape}")
    window = swl if swl is not None else cfg.swl

    ts_net = get_ts_roi(ts, atlas.roi_idx)
    dfc = sliding_window_fc(ts, ts_net, window, cfg.sws)
    dfc = apply_sign(dfc, cfg.sign_type)
    binary, thr, mu, sigma = binarize_dfc(dfc, cfg.k, cfg.binarize_mode)
    _, counts, prob, entropy = compute_states(binary, atlas.n_net)

    thr_val = float(np.ravel(thr)[0]) if np.ndim(thr) else float(thr)
    mu_val = float(np.ravel(mu)[0]) if np.ndim(mu) else float(mu)
    sigma_val = float(np.ravel(sigma)[0]) if np.ndim(sigma) else float(sigma)
    return SubjectResult(
        entropy=entropy, states_prob=prob, states_counts=counts,
        dfc_binary=binary.astype(np.uint8),
        threshold=thr_val, mu=mu_val, sigma=sigma_val,
        n_windows=int(dfc.shape[2]), subject_id=subject_id,
    )
