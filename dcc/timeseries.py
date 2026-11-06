"""Generic time-series input discovery and reading.

A *sample* is one 2-D matrix ``(n_roi, n_timepoints)``; it may represent a
subject, a session, a run or a condition. Nothing here assumes a study design.
Samples can arrive in two ways:

* a **single container file** holding a 3-D array ``(n_roi, T, n_sample)``,
  a cell/object array of matrices, or a named variable inside a ``.mat`` file;
* a **directory** in which every supported file is one sample and the file
  stem becomes the ``sample_id``.

Supported extensions are listed once in ``dcc.config.SUPPORTED_SUFFIXES``:
``.mat`` (classic or HDF5-based), ``.npy``, ``.npz``, ``.csv``, ``.tsv`` and
``.txt``. ``load_timeseries`` normalises every one of them to a float array,
including the quirks of object arrays whose elements are 0-D scalars.

``discover_samples`` optionally merges a metadata table (``sample_id``,
``group``, ``subject``, ``session``) so later stages can decide which optional
analyses are valid. ``get_ts_roi`` is the small shared helper used by the core
and network modules to coerce and validate an ROI matrix.

This layer is pure IO: no method-specific logic, no plotting and no GUI code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import SUPPORTED_SUFFIXES


def get_ts_roi(data: np.ndarray, labels: np.ndarray, ignore_zero: bool = True) -> np.ndarray:
    """Average ``data`` along axis 0 within each label group.

    data   : (N, ...)  e.g. (n_roi, T) signals or (n_roi, n_states) probabilities
    labels : (N,)      network / parcel index of each ROI
    return : (n_label, ...)
    """
    data = np.asarray(data)
    labels = np.asarray(labels).ravel()
    if labels.shape[0] != data.shape[0]:
        raise ValueError(f"Label count {labels.shape[0]} does not match "
                         f"data axis 0 ({data.shape[0]})")
    values = np.unique(labels)
    if ignore_zero:
        values = values[values != 0]
    if len(values) == 0:
        raise ValueError("Label vector has no valid group")
    return np.stack([data[labels == v].mean(axis=0) for v in values], axis=0)


@dataclass
class Sample:
    """One time-series matrix plus optional study metadata."""

    sample_id: str
    data: np.ndarray
    group: str = ""
    subject: str = ""
    session: str = ""
    source: str = ""

    @property
    def shape(self) -> tuple:
        """Return the ``(n_roi, n_timepoints)`` shape of this sample."""
        return tuple(self.data.shape)


def _is_v73(path: Path) -> bool:
    """Detect HDF5-based containers by their file header."""
    with open(path, "rb") as f:
        head = f.read(128)
    return b"MAT-file" in head and b"7.3" in head


def _load_mat_v5(path: Path, variable: str | None) -> object:
    """Read a classic (non-HDF5) .mat container and extract the requested variable."""
    from scipy.io import loadmat
    data = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    if variable:
        if variable not in data:
            raise KeyError(f"Variable '{variable}' not found in {path}")
        return data[variable]
    for key, value in data.items():
        if not key.startswith("__"):
            return value
    raise ValueError(f"No data variable in {path}")


def _load_mat_v73(path: Path, variable: str | None) -> object:
    """HDF5-based .mat: numeric arrays only."""
    import h5py
    with h5py.File(str(path), "r") as f:
        keys = [k for k in f.keys() if not k.startswith("#")]
        key = variable or (keys[0] if keys else None)
        if key is None or key not in f:
            raise KeyError(f"Variable '{key}' not found in {path}")
        return np.array(f[key])


def _as_samples(obj, prefix: str, path: Path) -> list[tuple[str, np.ndarray]]:
    """Turn a decoded array / cell array into (id, matrix) pairs."""
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size:
        # Some writers store an ordinary numeric array with dtype=object.
        if np.asarray(obj.ravel()[0]).ndim == 0:
            obj = np.asarray(obj, dtype=float)
    out: list[tuple[str, np.ndarray]] = []
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for i, cell in enumerate(obj.ravel()):
            arr = np.asarray(cell, dtype=float)
            if arr.ndim != 2:
                raise ValueError(f"{path}: cell {i} is not a 2-D matrix ({arr.shape})")
            out.append((f"{prefix}_{i + 1:03d}", arr))
    elif isinstance(obj, np.ndarray) and obj.ndim == 3:
        for i in range(obj.shape[2]):
            out.append((f"{prefix}_{i + 1:03d}",
                        np.asarray(obj[:, :, i], dtype=float)))
    elif isinstance(obj, np.ndarray) and obj.ndim == 2:
        out.append((f"{prefix}_001", np.asarray(obj, dtype=float)))
    else:
        raise ValueError(f"{path}: unsupported data shape "
                         f"{getattr(obj, 'shape', None)}")
    return out


def load_timeseries(path: str | Path, variable: str | None = None,
                    prefix: str = "sample") -> list[tuple[str, np.ndarray]]:
    """Load all samples stored in one file.

    Supported containers:
      * .mat  numeric 3-D (n_roi, T, n_sample), object cell, or 2-D matrix
      * .npy  object / 3-D / 2-D arrays
      * .npz  first numeric array in the archive
      * .csv / .tsv / .txt  numeric table with time points in rows and ROIs
        in columns (transposed to (n_roi, T))
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix == ".mat":
        obj = (_load_mat_v73(path, variable) if _is_v73(path)
               else _load_mat_v5(path, variable))
        return _as_samples(obj, prefix, path)

    if suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        return _as_samples(obj, prefix, path)

    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            keys = list(archive.files)
            if not keys:
                raise ValueError(f"{path} is empty")
            return _as_samples(archive[keys[0]], prefix, path)

    if suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix in (".tsv", ".txt") else ","
        try:
            arr = np.loadtxt(path, delimiter=sep, ndmin=2)
        except ValueError as exc:
            raise ValueError(f"{path}: expected a numeric table "
                             f"(rows = time points, columns = ROIs)") from exc
        return [(f"{prefix}_001", arr.T)]

    raise ValueError(f"Unsupported time-series format: {suffix}")


def _apply_metadata(samples: list[Sample], metadata_path: Path) -> list[Sample]:
    """Attach group / subject / session columns from a CSV table.

    The table may carry a ``sample_id`` column for explicit matching; without
    it, rows are matched to samples in discovery order.
    """
    import pandas as pd
    frame = pd.read_csv(metadata_path, dtype=str).fillna("")
    cols = {c.strip().lower(): c for c in frame.columns}
    fields = ("group", "subject", "session")

    if "sample_id" in cols:
        lookup = {str(row[cols["sample_id"]]).strip(): row
                  for _, row in frame.iterrows()}
        for sample in samples:
            row = lookup.get(sample.sample_id)
            if row is None:
                continue
            for field in fields:
                if field in cols:
                    setattr(sample, field, str(row[cols[field]]).strip())
    else:
        for i, sample in enumerate(samples):
            if i >= len(frame):
                break
            row = frame.iloc[i]
            for field in fields:
                if field in cols:
                    setattr(sample, field, str(row[cols[field]]).strip())
    return samples


def discover_samples(source: str | Path, variable: str | None = None,
                     metadata_path: str | Path | None = None) -> list[Sample]:
    """Collect all samples from a file or a directory of files.

    Every sample must have the same number of ROIs so that a single atlas can
    be applied. Sample IDs come from file names (with an index suffix when a
    file contains several samples).
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        files = sorted(p for p in source.rglob("*")
                       if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    else:
        files = [source]
    if not files:
        raise ValueError(f"No supported time-series files in {source}")

    samples: list[Sample] = []
    for file in files:
        pairs = load_timeseries(file, variable=variable, prefix=file.stem)
        for i, (_, matrix) in enumerate(pairs):
            sid = file.stem if len(pairs) == 1 else f"{file.stem}_{i + 1:03d}"
            samples.append(Sample(sample_id=sid, data=np.asarray(matrix, dtype=float),
                                  source=str(file)))
    if not samples:
        raise ValueError(f"No samples found in {source}")

    n_roi = samples[0].data.shape[0]
    for sample in samples:
        if sample.data.ndim != 2:
            raise ValueError(f"{sample.sample_id}: expected a 2-D matrix, got "
                             f"{sample.data.shape}")
        if sample.data.shape[0] != n_roi:
            raise ValueError(f"{sample.sample_id}: ROI count {sample.data.shape[0]} "
                             f"differs from {n_roi}")
        if sample.data.shape[1] < 2:
            raise ValueError(f"{sample.sample_id}: not enough time points")

    ids = [s.sample_id for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs; use distinct file names")
    if metadata_path and Path(metadata_path).exists():
        samples = _apply_metadata(samples, Path(metadata_path))
    return samples
