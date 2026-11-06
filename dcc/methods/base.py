"""Extension point for dynamic functional connectivity (dFC) methods.

DCC (the Shannon entropy of connectivity-state occurrences) is only *one* way
of characterising time-varying connectivity. DCC Studio is designed so that a
second method -- for example flexibility, integration, or a future
state-transition metric -- can be added without touching the workflow, the
GUI, the result exporter, or any of the statistics modules.

The contract is deliberately small. A method only has to answer one question:

    given one sample's ROI time series and an atlas, what is the per-ROI value?

Everything else in the application is method-agnostic:

* ``dcc.pipelines.run_analysis`` discovers inputs, iterates samples, caches
  per-sample results, aggregates them and exports CSV/JSON/PNG files;
* ``dcc.stats``, ``dcc.network`` and ``dcc.pls`` consume the common
  ``SubjectResult`` structure;
* ``gui.app`` builds its method selector from the registry, so a newly
  registered method appears in the interface automatically.

Adding a method therefore means: subclass :class:`AnalysisMethod`, implement
:meth:`AnalysisMethod.compute`, decorate the class with
:func:`dcc.methods.registry.register_method`, and import it once (normally
from ``dcc/methods/__init__.py``). See ``docs/METHOD.md`` for a walkthrough.

Design notes
------------
* A method instance is stateless and cheap: the registry holds one instance
  per registered name and reuses it across runs and GUI sessions.
* ``name`` is a stable, lowercase identifier used in configuration files,
  result directory names and output file names. It must never change once
  published, otherwise existing caches become unreachable.
* ``title`` / ``description`` are presentation-only and may be reworded.
* ``compute`` must be deterministic. Randomness belongs in the optional
  permutation/bootstrap analyses, which are driven by ``cfg.random_seed``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

# Type-only imports keep this module importable at package initialisation time:
# importing the concrete config/atlas/result classes here would create a cycle,
# because those modules (indirectly) import the registry.
if TYPE_CHECKING:                       # pragma: no cover - hints only
    from ..atlas import Atlas
    from ..complexity import SubjectResult
    from ..config import DCCConfig


class AnalysisMethod(ABC):
    """Base class for a per-sample dynamic connectivity measure.

    Subclasses declare a small set of class attributes and implement
    :meth:`compute`. The workflow reads these attributes to label its output,
    so a method never needs to know about files, plots, threads or widgets.

    Attributes
    ----------
    name:
        Stable machine identifier, e.g. ``"dcc"``. Lowercase, alphanumeric and
        underscores only: it becomes part of a directory and file name.
    title:
        Short human-readable label shown in the GUI method selector.
    description:
        One or two sentences explaining the measure. Used for the parameter
        tooltip; keep it readable for non-specialist researchers.
    metric:
        Abbreviation of the per-ROI quantity (``"DCC"`` for the built-in
        method). Appears in progress messages and figure titles.
    result_is_entropy:
        Whether the per-ROI value is an entropy in bits. The workflow uses
        this only for axis labels; it does not affect the arithmetic.
    """

    name: str = ""
    title: str = ""
    description: str = ""
    metric: str = "value"
    result_is_entropy: bool = False

    @abstractmethod
    def compute(self, ts: np.ndarray, atlas: "Atlas", cfg: "DCCConfig",
                swl: int | None = None,
                subject_id: str = "") -> "SubjectResult":
        """Compute the per-ROI measure for one sample.

        Parameters
        ----------
        ts:
            ROI signals shaped ``(n_roi, n_timepoints)``. Columns are time
            points in acquisition order; rows follow the atlas order.
        atlas:
            ROI-to-network mapping. ``atlas.n_net`` also determines the size
            of the state space for state-based methods.
        cfg:
            Validated run parameters. A method may read any field it needs;
            unrelated fields must be ignored rather than rejected.
        swl:
            Optional override of the sliding-window length. The workflow
            passes ``None`` so that ``cfg.swl`` is used; the override exists
            for parity tests and for methods with their own window semantics.
        subject_id:
            Identifier copied into the returned result and into cache names.

        Returns
        -------
        SubjectResult
            ``entropy`` holds the per-ROI measure (the field name is kept for
            backwards compatibility with the result cache), ``states_prob``
            and ``states_counts`` hold any state distribution the method
            produces, and ``dfc_binary`` may hold an optional debug tensor.

        Notes
        -----
        Implementations must raise ``ValueError`` for malformed input rather
        than silently returning zeros: the workflow turns exceptions into a
        clear error message in the GUI.
        """
        raise NotImplementedError

    def validate(self, cfg: "DCCConfig") -> None:
        """Optional method-specific parameter validation.

        The workflow calls ``cfg.validate()`` first (generic checks), then
        this hook. A method can, for example, require a minimum window length.
        The default implementation accepts any already-valid configuration.
        """
        return None

    def extra_parameters(self) -> dict[str, str]:
        """Optional ``{tooltip_key: help_text}`` entries for method-specific UI.

        The GUI merges these into its tooltip table so a new method can
        document its own parameters without editing ``gui/app.py``.
        """
        return {}

    def output_prefix(self) -> str:
        """Prefix used for result file names, e.g. ``dcc_matrix.csv``."""
        return self.name

    def __str__(self) -> str:
        """Readable representation used in logs and error messages."""
        return f"{self.name} ({self.title})"
