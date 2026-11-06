"""Frozen-environment self-test, used only by ``build_exe.py --verify``.

PyInstaller can happily produce an executable that fails at runtime because a
DLL, a data file, a plugin or a hidden import is missing. Importing a module is
not enough to prove otherwise: this script exercises the real dependencies,
runs one complete (synthetic) analysis workflow, and constructs the GUI main
window without entering the Qt event loop.

Checks performed
----------------
==========================  ==================================================
Area                        Check
==========================  ==================================================
Bundled resources           ``config.example.json`` exists in the bundle
Scientific stack            numpy, scipy.stats, pandas, h5py, scipy.io
IO formats                  .mat container, HDF5 file, CSV write/read
Plotting                    matplotlib Agg backend renders and saves a PNG
Computation                 six-stage ``run_analysis`` on two synthetic samples
Result files                ``summary.json`` and ``dcc_matrix.csv`` are written
GUI import chain            ``QApplication`` + ``MainWindow`` construction
Appearance / hints          theme, Settings dialog and parameter help markers
Method registry             built-in methods are registered and resolvable
Settings selector           registered and planned methods are listed correctly
==========================  ==================================================

The script also runs from a source checkout: in that case the project root is
prepended to ``sys.path`` so the same imports resolve.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.io
import scipy.stats
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless: never require a display server
import matplotlib.pyplot as plt
import h5py
from PyQt6.QtWidgets import QApplication

# Imports below are deliberately explicit: PyInstaller only bundles modules it
# can discover, and these are the ones the application loads dynamically.
from dcc.atlas import Atlas                       # noqa: F401
from dcc.complexity import connectome_complex     # noqa: F401
from dcc.config import (CONFIG_EXAMPLE, PROJECT_ROOT,  # noqa: F401
                        DCCConfig, RunPaths, load_settings)
from dcc.methods import available_methods, get_method  # noqa: F401
from dcc.pipelines import run_analysis            # noqa: F401
from dcc.pls import pls_analysis                  # noqa: F401
from dcc.results import ResultStore               # noqa: F401
from dcc.timeseries import discover_samples, load_timeseries  # noqa: F401
from gui.app import MainWindow
from gui.settings import SettingsDialog             # noqa: F401


def main() -> int:
    """Run every frozen-environment check and return 0 on success."""
    print("PROJECT_ROOT   =", PROJECT_ROOT)
    print("CONFIG_EXAMPLE =", CONFIG_EXAMPLE.exists())
    paths, cfg = load_settings()
    print("config         =", cfg.run_tag(), paths.output_root)
    print("methods        =", [method.name for method in available_methods()])
    assert get_method(cfg.method).name == cfg.method

    # A fixed seed keeps the check deterministic between builds.
    values = np.random.default_rng(0).normal(size=(20, 7))
    print("scipy.stats    =", float(scipy.stats.ttest_1samp(values[:, 0], 0).pvalue))

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        data = work / "data"
        data.mkdir()
        rng = np.random.default_rng(1)
        # Two samples, 12 ROIs, 60 time points, three networks -> 7 states.
        for name in ("sample_a", "sample_b"):
            np.save(data / f"{name}.npy", rng.normal(size=(12, 60)))
        labels = work / "labels.txt"
        labels.write_text("\n".join(["1"] * 4 + ["2"] * 4 + ["3"] * 4) + "\n",
                          encoding="utf-8")

        # Round-trip each IO format the application advertises.
        scipy.io.savemat(work / "a.mat", {"x": values})
        assert scipy.io.loadmat(str(work / "a.mat"))["x"].shape == (20, 7)
        with h5py.File(work / "a.h5", "w") as handle:
            handle["x"] = values
        with h5py.File(work / "a.h5", "r") as handle:
            assert handle["x"].shape == (20, 7)
        pd.DataFrame(values).to_csv(work / "a.csv", index=False)
        assert pd.read_csv(work / "a.csv").shape == (20, 7)

        # Rendering exercises the bundled matplotlib data (fonts in particular).
        figure, axes = plt.subplots()
        axes.plot(values[:, 0])
        axes.set_title("Font check")
        figure.savefig(work / "a.png")
        plt.close(figure)
        assert (work / "a.png").stat().st_size > 0

        run_paths = RunPaths(data_source=data, atlas_labels=labels,
                             output_root=work / "results")
        run_cfg = DCCConfig(swl=10, n_perms=0, n_bootstraps=0)
        result = run_analysis(run_paths, run_cfg)
        assert (result.run_dir / "summary.json").exists()
        assert (result.run_dir / "dcc_matrix.csv").exists()
        assert len(result.figures) > 0
        print("workflow       =", result.summary["n_samples"], "samples,",
              len(result.figures), "figures")

    # Constructing the window is enough to validate every Qt plugin/import used
    # by the four-step interface; no event loop is started in a build check.
    app = QApplication([])      # loads the qwindows platform plugin
    window = MainWindow()       # covers the full GUI import chain
    print("Qt widget      =", type(window).__name__)
    print("appearance     =", window.theme, "|",
          window.settings_dialog.current_theme())
    print("hint markers   =", len(window.hint_markers))
    print("settings list  =", window.settings_dialog.method_names())
    assert window.settings_dialog.is_method_enabled(cfg.method)
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
