"""Configuration and on-disk settings for one analysis run.

A run is fully described by two objects:

* :class:`RunPaths` -- where the inputs live and where results are written;
* :class:`DCCConfig` -- the numerical parameters of the selected method.

Neither object refers to a particular study, dataset or acquisition protocol.
The selected computation method is stored as a *name* only; the registry in
``dcc.methods`` resolves it at run time. This keeps configuration files small,
human-readable and independent of Python classes.

Settings are persisted as a flat JSON object in ``config.json`` next to the
application (or next to the executable in a frozen build). ``config.json`` is
local state and is not committed; ``config.example.json`` is the documented
template used when no local file exists.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# File extensions understood by ``dcc.timeseries.load_timeseries``. Kept here
# so the IO layer, the GUI file dialogs and the documentation share one list.
SUPPORTED_SUFFIXES = (".mat", ".npy", ".npz", ".csv", ".tsv", ".txt")


def _runtime_roots() -> tuple[Path, Path]:
    """Return (writable application root, read-only bundled resource root).

    When frozen by PyInstaller, ``config.json`` and ``results/`` must live
    next to the executable instead of the temporary ``_MEIPASS`` extraction
    directory, which is deleted on exit. In a source checkout both roots are
    simply the project directory.
    """
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
        return app_dir, Path(getattr(sys, "_MEIPASS", str(app_dir)))
    root = Path(__file__).resolve().parents[1]
    return root, root


PROJECT_ROOT, RESOURCE_ROOT = _runtime_roots()
CONFIG_FILE = PROJECT_ROOT / "config.json"
CONFIG_EXAMPLE = RESOURCE_ROOT / "config.example.json"


@dataclass
class DCCConfig:
    """Numerical parameters of an analysis run; dataset independent.

    The defaults reproduce the reference parameter set (window length 30,
    step 1, threshold coefficient 0.5, absolute correlations, global
    threshold). Every field is written to ``config.json`` and to
    ``summary.json`` so a run can be reproduced exactly.

    ``method`` selects an implementation from ``dcc.methods``. Because the
    built-in DCC method is the default, ``run_tag()`` deliberately omits its
    name: existing result directories and caches keep their original names.
    Any other method prefixes the tag, so two methods never share a cache.
    """

    swl: int = 30                     # sliding window length (time points)
    sws: int = 1                      # step between consecutive windows
    k: float = 0.5                    # threshold = mean + k * std (ddof=1)
    sign_type: str = "abs"            # abs | raw | pos
    binarize_mode: str = "global"     # global threshold | one per ROI
    state_prob_threshold: float = 0.1  # min probability of a "high" state
    top_state_fraction: float = 0.1   # fraction of states counted as "top"
    cohen_threshold: float = 0.3      # |d| cutoff for effect-size counts
    n_perms: int = 1000               # PLS permutation count (0 disables)
    n_bootstraps: int = 500           # PLS bootstrap count (0 disables)
    norm_img: int = 1                 # PLS normalisation of the ROI matrix
    norm_behav: int = 1               # PLS normalisation of behaviour data
    pls_n_lc: int = 2                 # latent components shown if FDR finds none
    fdr_q: float = 0.05               # false discovery rate level
    random_seed: int = 42             # seed for permutation / bootstrap
    save_binary_dfc: bool = False     # store the (large) binary dFC tensor
    variable: str = ""                # optional variable name in a container
    method: str = "dcc"               # registered analysis method name

    def validate(self) -> None:
        """Reject parameter combinations that would produce meaningless output.

        Method-specific checks belong to ``AnalysisMethod.validate``; this
        method only validates the generic values, which is why it can run
        without importing the registry (and thus without circular imports).
        """
        if self.sws < 1:
            raise ValueError("SWS must be >= 1")
        if self.swl < 2:
            raise ValueError("SWL must be >= 2")
        if self.sign_type not in ("abs", "raw", "pos"):
            raise ValueError("sign_type must be abs / raw / pos")
        if self.binarize_mode not in ("global", "roi"):
            raise ValueError("binarize_mode must be global / roi")
        if self.n_perms < 0 or self.n_bootstraps < 0:
            raise ValueError("permutation / bootstrap counts cannot be negative")
        if not 0 < self.top_state_fraction <= 1:
            raise ValueError("top_state_fraction must be in (0, 1]")
        if not 0 < self.fdr_q < 1:
            raise ValueError("fdr_q must be in (0, 1)")
        if not self.method or not isinstance(self.method, str):
            raise ValueError("method must be a non-empty registered name")

    def run_tag(self) -> str:
        """Short, filesystem-safe name describing this parameter set.

        The tag is also the result directory name, which makes it the cache
        key: identical parameters reuse cached samples, different parameters
        (or a different method) never overwrite each other.
        """
        base = (f"swl{self.swl}_sws{self.sws}_k{self.k:g}_"
                f"{self.sign_type}_{self.binarize_mode}")
        # Keep the historical DCC directory names unchanged for compatibility.
        return base if self.method == "dcc" else f"{self.method}_{base}"

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy of all parameters."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DCCConfig":
        """Build a validated config from a possibly partial dictionary.

        Unknown keys are ignored, which lets old configuration files be read
        after a parameter was removed. Missing keys fall back to defaults.
        """
        keys = {f.name for f in fields(cls)}
        obj = cls(**{k: v for k, v in d.items() if k in keys})
        obj.validate()
        return obj


@dataclass
class RunPaths:
    """Input and output locations for one analysis run.

    ``data_source`` may be a single container file or a directory of sample
    files. ``atlas_labels`` is mandatory; the other inputs are optional and
    their presence controls which optional analyses are executed.
    """

    data_source: Path = PROJECT_ROOT / "data"          # file or directory
    atlas_labels: Path = PROJECT_ROOT / "atlas_labels.txt"
    atlas_names: Path | None = None
    metadata: Path | None = None                       # sample table
    behaviour: Path | None = None                      # behaviour table
    output_root: Path = PROJECT_ROOT / "results"

    def __post_init__(self) -> None:
        """Normalise path types and resolve a relative output directory."""
        self.data_source = Path(self.data_source).expanduser()
        self.atlas_labels = Path(self.atlas_labels).expanduser()
        self.atlas_names = self._opt(self.atlas_names)
        self.metadata = self._opt(self.metadata)
        self.behaviour = self._opt(self.behaviour)
        self.output_root = Path(self.output_root).expanduser()
        if not self.output_root.is_absolute():
            self.output_root = (PROJECT_ROOT / self.output_root).resolve()

    @staticmethod
    def _opt(value) -> Path | None:
        """Convert an optional (possibly empty) path to ``Path`` or ``None``."""
        if value is None or str(value).strip() == "":
            return None
        return Path(value).expanduser()

    def check_paths(self) -> list[str]:
        """Return human-readable problems with the configured input paths.

        The GUI calls this before starting a worker thread so that missing
        files are reported immediately rather than halfway through a run.
        """
        problems: list[str] = []
        if not self.data_source.exists():
            problems.append(f"Time-series source not found: {self.data_source}")
        if not self.atlas_labels.exists():
            problems.append(f"Atlas labels not found: {self.atlas_labels}")
        for label, value in (("Atlas names", self.atlas_names),
                             ("Metadata", self.metadata),
                             ("Behaviour table", self.behaviour)):
            if value is not None and not value.exists():
                problems.append(f"{label} file not found: {value}")
        return problems

    def to_dict(self) -> dict:
        """Return a JSON-serialisable copy of all paths (as strings)."""
        return {
            "data_source": str(self.data_source),
            "atlas_labels": str(self.atlas_labels),
            "atlas_names": str(self.atlas_names or ""),
            "metadata": str(self.metadata or ""),
            "behaviour": str(self.behaviour or ""),
            "output_root": str(self.output_root),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunPaths":
        """Build paths from a settings dictionary, ignoring unknown keys."""
        allowed = {"data_source", "atlas_labels", "atlas_names",
                   "metadata", "behaviour", "output_root"}
        return cls(**{k: v for k, v in d.items() if k in allowed})


def load_settings(path: Path | None = None) -> tuple[RunPaths, DCCConfig]:
    """Load ``config.json``, falling back to the bundled example, then defaults.

    The fallback order means a fresh installation always has sensible values
    even before the user presses "Save configuration". A malformed local file
    raises ``JSONDecodeError``; callers show it as an actionable error.
    """
    cfg_path = Path(path) if path else CONFIG_FILE
    if not cfg_path.exists():
        cfg_path = CONFIG_EXAMPLE
    if not cfg_path.exists():
        return RunPaths(), DCCConfig()
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return RunPaths.from_dict(raw), DCCConfig.from_dict(raw)


def save_settings(paths: RunPaths, cfg: DCCConfig,
                  path: Path | None = None) -> None:
    """Persist the current run configuration next to the application.

    Path entries and numerical parameters are flattened into one JSON object,
    matching the layout of ``config.example.json`` so the file can be edited
    by hand and inspected in a text editor.
    """
    cfg_path = Path(path) if path else CONFIG_FILE
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({**paths.to_dict(), **cfg.to_dict()}, f,
                  ensure_ascii=False, indent=2)
