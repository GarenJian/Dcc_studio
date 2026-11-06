"""Pluggable dynamic connectivity analysis methods.

Importing this package registers every built-in method. The order of the
imports matters: ``registry`` must exist before ``dcc_method`` is imported,
because the latter is decorated with ``@register_method`` at class-definition
time.

Example of adding a new method (full guide in ``docs/METHOD.md``)::

    from dcc.methods import AnalysisMethod, register_method

    @register_method
    class FlexibilityMethod(AnalysisMethod):
        name = "flexibility"
        title = "Network flexibility"
        description = "Fraction of windows in which an ROI changes module."
        metric = "FLEX"

        def compute(self, ts, atlas, cfg, swl=None, subject_id=""):
            ...  # return a SubjectResult with one value per ROI

No other module needs to change: the GUI selector, ``run_tag`` cache
separation, output naming and validation all read from the registry.
"""
from __future__ import annotations

from .base import AnalysisMethod
from .registry import (available_methods, clear_registry, get_method,
                       method_choices, method_descriptions,
                       registered_names, register_method)
from .dcc_method import DCCMethod         # noqa: F401 - registers "dcc"

__all__ = [
    "AnalysisMethod", "DCCMethod", "register_method", "get_method",
    "available_methods", "method_choices", "method_descriptions",
    "registered_names", "clear_registry",
]
