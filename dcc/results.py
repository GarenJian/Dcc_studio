"""Result files for one analysis run: NPZ caches, CSV, JSON and PNG.

:class:`ResultStore` owns one directory (``results/<run_tag>/``) and provides
a single, consistent writer for every output type:

=====================  =====================================================
Kind                   Content
=====================  =====================================================
``samples/*.npz``      Per-sample cache (entropy, states, optional binary dFC)
``*.csv``              Tables, always UTF-8 with a header row
``*.json``             Run manifest, state statistics, final summary
``*.png``              Figures saved at publication-friendly resolution
=====================  =====================================================

Caching is what makes re-runs cheap: a sample whose NPZ file exists is loaded
instead of recomputed. Because ``run_tag()`` encodes the parameters *and* the
selected method, two different configurations never share a directory and
therefore never reuse each other's cache.

The GUI's Results page lists files through ``list_files()`` and groups them by
workflow category; it never hard-codes a file name. ``save_binary_dfc`` is off
by default because the binary dFC tensor is typically the largest object.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .complexity import SubjectResult


class ResultStore:
    def __init__(self, root: str | Path):
        """Create the run directory and its per-sample cache subdirectory if needed."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "samples").mkdir(exist_ok=True)
        (self.root / "figures").mkdir(exist_ok=True)

    def file(self, name: str) -> Path:
        """Return a path inside the run directory for the given file name."""
        return self.root / name

    # ---------- per-sample cache ----------
    def sample_path(self, sid: str) -> Path:
        """Return the filesystem-safe NPZ cache path for one sample id."""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(sid))
        return self.root / "samples" / f"{safe}.npz"

    def has_sample(self, sid: str) -> bool:
        """True when a cached result already exists for this sample."""
        return self.sample_path(sid).exists()

    def save_sample(self, sid: str, res: SubjectResult,
                    save_binary: bool = False) -> Path:
        """Write one SubjectResult to the cache, optionally including the binary tensor."""
        path = self.sample_path(sid)
        np.savez_compressed(path, **res.to_dict(save_binary=save_binary))
        return path

    def load_sample(self, sid: str) -> SubjectResult:
        """Read a cached SubjectResult back from its NPZ file."""
        with np.load(self.sample_path(sid)) as data:
            return SubjectResult.from_dict({k: data[k] for k in data.files})

    # ---------- tabular / summary output ----------
    def save_csv(self, name: str, data, columns=None, index: bool = False) -> Path:
        """Write a DataFrame as UTF-8 CSV and return the created path."""
        path = self.file(name if str(name).endswith(".csv") else f"{name}.csv")
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=index, encoding="utf-8")
        else:
            pd.DataFrame(np.asarray(data), columns=columns).to_csv(
                path, index=index, encoding="utf-8")
        return path

    def save_json(self, name: str, obj) -> Path:
        """Write a JSON file with NumPy-aware conversion and return the created path."""
        path = self.file(name if str(name).endswith(".json") else f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        return path

    def save_npy(self, name: str, arr: np.ndarray) -> Path:
        """Write a raw NumPy array and return the created path."""
        path = self.file(name if str(name).endswith(".npy") else f"{name}.npy")
        np.save(path, np.asarray(arr))
        return path

    def save_figure(self, name: str, fig, dpi: int = 150) -> Path:
        """Save a matplotlib Figure as PNG at fixed DPI and return the created path."""
        path = self.root / "figures" / (name if str(name).endswith(".png")
                                        else f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        return path

    def list_files(self, extensions: tuple = (".png", ".csv", ".json",
                                              ".npy", ".txt", ".md")) -> list:
        """Return all result files sorted by name, excluding the internal sample cache."""
        return sorted(p for p in self.root.rglob("*")
                      if p.is_file() and p.suffix.lower() in extensions)


def _json_default(obj):
    """Convert NumPy scalars and arrays to plain Python values for JSON output."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
