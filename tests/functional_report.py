"""Collect structured details of functional comparisons for the test report.

The regression suite compares values computed by the current implementation
with values stored in the copied reference result files. A green dot in
``unittest`` cannot show *which dataset* was used, *what* the stored result
was, or *how close* the recomputation came. This module records one compact
entry per numerical comparison so ``run_tests.py`` can render a detailed
functional section in ``reports/test-report.md``.

Programmatic unit tests deliberately do not use this collector: their part of
the report is a short summary, not a per-case walkthrough.

Lifecycle
---------
1. ``run_tests.py`` sets ``DCC_FUNCTIONAL_REPORT`` to the report directory and
   calls ``reset()`` (also called by the regression suite's ``setUpClass``).
2. Each comparison calls :func:`record` immediately after its assertion, with
   the dataset description, the reference file, the quantity name, both arrays
   and the tolerance that was actually used.
3. ``flush()`` (called in ``tearDownClass``) writes
   ``reports/functional-details.json``. ``run_tests.py`` renders it.

Entries are intentionally small: full arrays are never written. Instead each
entry stores shapes, descriptive statistics, the largest deviation (with the
values at that position) and a few representative element-by-element values.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

DEFAULT_PATH = (Path(__file__).resolve().parents[1] / "reports"
                / "functional-details.json")
SAMPLE_COUNT = 3

_records: list[dict] = []


def output_path() -> Path:
    """Return the JSON destination (overridable for custom report folders)."""
    return Path(os.environ.get("DCC_FUNCTIONAL_REPORT", str(DEFAULT_PATH)))


def reset() -> None:
    """Discard records from a previous run."""
    _records.clear()


def records() -> list[dict]:
    """Return a shallow copy of the collected entries."""
    return list(_records)


def _number(value) -> float | None:
    """Convert a NumPy scalar to a JSON-safe float, mapping non-finite to None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _statistics(array: np.ndarray) -> dict:
    """Compute min / max / mean of a numeric array, guarding empty input."""
    if array.size == 0:
        return {"min": None, "max": None, "mean": None}
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": _number(finite.min()), "max": _number(finite.max()),
            "mean": _number(finite.mean())}


def record(case: str, dataset: str, source: str, quantity: str,
           expected, actual, tolerance: float, note: str = "") -> dict:
    """Record one completed numerical comparison.

    Parameters
    ----------
    case:
        Name of the functional test method, used as a report subsection heading.
    dataset:
        Human-readable description of the tested slice (file, sample, shape).
    source:
        Relative path of the reference result file holding ``expected``.
    quantity:
        What was compared, e.g. ``"entropy (200 ROIs)"``.
    expected / actual:
        Arrays or scalars. ``expected`` is the *stored reference result*;
        ``actual`` is what the current implementation computed.
    tolerance:
        Absolute tolerance used by the assertion (``0`` means exact equality).
    note:
        Optional remark such as the window length or ``ddof`` convention.
    """
    exp = np.asarray(expected, dtype=float).reshape(-1)
    act = np.asarray(actual, dtype=float).reshape(-1)
    if exp.shape != act.shape:
        raise ValueError(f"Shape mismatch for {case}/{quantity}: "
                         f"{exp.shape} vs {act.shape}")

    diff = np.abs(exp - act) if exp.size else np.zeros(0)
    worst = int(np.argmax(diff)) if diff.size else 0
    max_diff = _number(diff[worst]) if diff.size else 0.0

    # Representative values: the first few elements plus the position with the
    # largest deviation, so a reader sees concrete numbers even for huge arrays.
    indexes = list(range(min(SAMPLE_COUNT, exp.size)))
    if exp.size and worst not in indexes:
        indexes.append(worst)
    samples = [{"index": int(i), "expected": _number(exp[i]),
                "actual": _number(act[i]), "diff": _number(diff[i])}
               for i in indexes]

    entry = {
        "case": case,
        "dataset": dataset,
        "source": source,
        "quantity": quantity,
        "shape": list(np.asarray(expected).shape),
        "dtype": str(np.asarray(expected).dtype),
        "elements": int(exp.size),
        "expected_stats": _statistics(exp),
        "actual_stats": _statistics(act),
        "max_abs_diff": max_diff,
        "worst_index": worst,
        "tolerance": float(tolerance),
        "passed": bool(max_diff is not None and max_diff <= tolerance),
        "mismatches": int(np.count_nonzero(diff > tolerance)) if diff.size else 0,
        "samples": samples,
        "note": note,
    }
    _records.append(entry)
    return entry


def flush() -> Path:
    """Write all collected entries as UTF-8 JSON and return the file path."""
    path = output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": _records, "count": len(_records)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path
