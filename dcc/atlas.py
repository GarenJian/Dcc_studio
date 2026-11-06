"""ROI-to-network atlas of arbitrary size.

An atlas is simply one integer label per ROI. Nothing in this module assumes a
fixed number of ROIs, a fixed number of networks, or a fixed naming scheme: the
state space of the DCC method is derived from ``G``, the number of *distinct
non-zero* labels, as ``2**G - 1`` states. ``G`` is capped at 16 so the state
distribution remains a manageable ``n_roi x (2**G - 1)`` matrix.

Conventions
-----------
* Label ``0`` means "unassigned". Such ROIs keep their own signal but are
  excluded from network averages and from network-level summaries.
* Labels need not be contiguous (``3, 7, 9`` is valid). ``Atlas.name(i)``
  addresses networks by their zero-based index in the sorted distinct labels,
  *not* by the raw label value.
* ``Atlas.load`` accepts either a one-column integer text/CSV file or an
  optional ``net_idx,net_name`` name table. Both are validated eagerly so a
  mismatch is reported before any computation starts.

The workflow uses :meth:`Atlas.aggregate` to form network-mean signals and
:meth:`Atlas.boundaries` only for plotting; the numerical core never depends
on how networks are named.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .timeseries import get_ts_roi

MAX_NETWORKS = 16  # the state space grows as 2 ** n_net - 1


@dataclass
class Atlas:
    roi_idx: np.ndarray
    net_names: tuple = ()

    def __post_init__(self) -> None:
        """Normalise labels to a 1-D integer array and reject more than 16 networks."""
        self.roi_idx = np.asarray(self.roi_idx, dtype=int).ravel()
        if self.roi_idx.size == 0:
            raise ValueError("Atlas labels are empty")
        if self.n_net < 1:
            raise ValueError("Atlas must contain at least one non-zero network label")
        if self.n_net > MAX_NETWORKS:
            raise ValueError(f"Too many networks ({self.n_net}); maximum is {MAX_NETWORKS}")
        self.net_names = tuple(self.net_names)
        if len(self.net_names) < self.n_net:
            self.net_names += tuple(f"Network {i + 1}" for i in
                                    range(len(self.net_names), self.n_net))

    @property
    def n_roi(self) -> int:
        """Number of ROIs, i.e. the length of the label vector."""
        return int(self.roi_idx.shape[0])

    @property
    def n_net(self) -> int:
        """Number of distinct non-zero network labels; it sets the state-space size."""
        return int(np.unique(self.roi_idx[self.roi_idx != 0]).size)

    @property
    def n_states(self) -> int:
        """Number of non-zero connectivity states: ``2 ** n_net - 1``."""
        return 2 ** self.n_net - 1

    @property
    def sizes(self) -> np.ndarray:
        """Number of ROIs assigned to each network, in network-index order."""
        values = np.unique(self.roi_idx[self.roi_idx != 0])
        return np.array([(self.roi_idx == v).sum() for v in values])

    def name(self, i: int) -> str:
        """Zero-based network name."""
        return self.net_names[i] if i < len(self.net_names) else f"Network {i + 1}"

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        """Mean over the first axis for each network."""
        return get_ts_roi(values, self.roi_idx)

    def boundaries(self) -> list:
        """Network boundaries used when drawing ROI profiles."""
        return list(np.cumsum(self.sizes)[:-1])

    @classmethod
    def from_labels(cls, labels, names=None) -> "Atlas":
        """Build an atlas directly from labels, optionally supplying network names."""
        return cls(roi_idx=np.asarray(labels, dtype=int),
                   net_names=tuple(names or ()))

    @classmethod
    def load(cls, labels_path: str | Path,
             names_path: str | Path | None = None) -> "Atlas":
        """Read label and optional name files, validate them, and return an Atlas."""
        labels = _read_labels(labels_path)
        names = (_read_names(names_path)
                 if names_path and Path(names_path).exists() else ())
        return cls.from_labels(labels, names)


def _read_labels(path: str | Path) -> np.ndarray:
    """Read one integer label per ROI from a plain text / CSV file."""
    path = Path(path)
    try:
        values = np.loadtxt(path, dtype=float, ndmin=1).ravel()
    except ValueError:
        import pandas as pd
        frame = pd.read_csv(path, header=None)
        values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"No atlas labels found in {path}")
    if not np.allclose(values, np.rint(values)):
        raise ValueError(f"Atlas labels in {path} must be integers")
    return np.rint(values).astype(int).ravel()


def _read_names(path: str | Path) -> tuple:
    """Read network names: either a net_idx/net_name CSV or one name per line."""
    path = Path(path)
    if path.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f, delimiter=sep))
        if rows:
            header = [c.strip().lower() for c in rows[0]]
            if "net_idx" in header and "net_name" in header:
                i, j = header.index("net_idx"), header.index("net_name")
                pairs = [(int(float(r[i])), r[j].strip())
                         for r in rows[1:] if len(r) > max(i, j) and r[i].strip()]
                return tuple(name for _, name in sorted(pairs))
        return tuple(r[0].strip() for r in rows if r and r[0].strip())
    return tuple(line.strip() for line in
                 path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
