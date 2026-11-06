"""Built-in DCC method: entropy of binary connectivity states.

This is a thin adapter around :func:`dcc.complexity.connectome_complex`. All
numerical work lives in the core modules; this class only supplies the
metadata the method registry and the workflow need. Keeping the adapter thin
is important: the exact same function is used by the parity tests, so the
registered method cannot drift away from the tested implementation.

Algorithm recap (see ``docs/METHOD.md`` for the full derivation)
---------------------------------------------------------------
1. Average ROI signals within each atlas network.
2. Correlate every ROI with every network signal inside a sliding window.
3. Apply sign handling and threshold at ``mean + k * std``.
4. Interpret the resulting binary pattern of each window as an integer state
   code; discard the all-zero state.
5. Report the Shannon entropy (bits) of the state-frequency distribution.

The result is one non-negative value per ROI: zero means the ROI always uses
a single connectivity state, while a high value means it visits many states
with comparable frequency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..complexity import SubjectResult, connectome_complex
from .base import AnalysisMethod
from .registry import register_method

if TYPE_CHECKING:                       # pragma: no cover - hints only
    from ..atlas import Atlas
    from ..config import DCCConfig


@register_method
class DCCMethod(AnalysisMethod):
    """State-entropy DCC, the default and currently only built-in method."""

    name = "dcc"
    title = "DCC (state entropy)"
    description = (
        "Shannon entropy of the ROI-wise distribution of binary connectivity "
        "states obtained from sliding-window correlations with atlas networks."
    )
    metric = "DCC"
    result_is_entropy = True

    def compute(self, ts: np.ndarray, atlas: "Atlas", cfg: "DCCConfig",
                swl: int | None = None,
                subject_id: str = "") -> SubjectResult:
        """Delegate to the tested core implementation.

        ``swl`` is forwarded so parity tests can reproduce a historical window
        length without mutating the shared configuration object. All other
        parameters (window step, threshold coefficient, sign handling and
        binarisation scope) come from ``cfg``.
        """
        return connectome_complex(ts, atlas, cfg, swl=swl,
                                  subject_id=subject_id)

    def validate(self, cfg: "DCCConfig") -> None:
        """DCC needs no method-specific checks beyond ``cfg.validate()``.

        The sliding-window length is checked against the shortest input series
        by the workflow, which knows the sample shapes; the configuration
        object only knows that ``swl >= 2``.
        """
        return None

    def extra_parameters(self) -> dict[str, str]:
        """No extra UI parameters: the generic parameter page covers DCC."""
        return {}
