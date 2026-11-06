"""DCC Studio: generic dynamic connectivity complexity analysis.

This package is the computation core. It has no dependency on PyQt6, so it can
be used from scripts, notebooks, tests or a future service layer, and it is
importable in a headless CI environment.

Layer map
---------
=========================  ===================================================
``config``                 Run parameters, paths and settings persistence
``timeseries``             Sample discovery and format-agnostic input IO
``atlas``                  ROI-to-network mapping of arbitrary size
``dfc``                    Sliding-window correlation and thresholding
``complexity``             State encoding and Shannon entropy for one sample
``methods``                Pluggable method contract, registry, built-in DCC
``pipelines``              Six-stage, method-agnostic analysis workflow
``network`` / ``stats``     Aggregation, group tests, ICC, FDR
``pls``                    Optional PLS-C association with behaviour data
``results`` / ``plotting``  Output files (NPZ/CSV/JSON) and PNG figures
=========================  ===================================================

The names re-exported below form the stable public API used by the GUI and by
the regression tests. Internal modules may be reorganised without affecting
callers as long as these symbols keep their behaviour.
"""

__version__ = "2.0.0"

from .config import DCCConfig, RunPaths
from .atlas import Atlas
from .timeseries import Sample, discover_samples, load_timeseries
from .complexity import SubjectResult, connectome_complex

__all__ = [
    "DCCConfig", "RunPaths", "Atlas", "Sample", "discover_samples",
    "load_timeseries", "SubjectResult", "connectome_complex",
]
