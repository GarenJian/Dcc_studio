"""The single, method-agnostic analysis workflow.

Every sample goes through the same six stages, regardless of which dynamic
connectivity method is selected:

=====  ==========  ==========================================================
Index  Stage       Responsibility
=====  ==========  ==========================================================
0      Input       Discover samples, load the atlas, validate shapes
1      Compute     Evaluate the selected method once per sample (with cache)
2      Summary     Aggregate per-ROI values, state distributions, CSV/JSON
3      Optional    Group comparisons, ICC(1,1), PLS-C when metadata allows
4      Figures     Publication-style PNG plots of whatever was computed
5      Export      Write ``summary.json`` and report the finished run
=====  ==========  ==========================================================

Two design rules keep this module extensible:

1. **No dataset names.** Optional analyses are enabled by the metadata the
   user supplies (groups, subjects/sessions, behaviour table), never by a
   hard-coded study identifier.
2. **No hard-coded method.** Stage 1 asks ``dcc.methods.get_method`` for an
   implementation. Output file names, metric labels and figure titles are
   derived from ``method.name`` / ``method.metric``, so registering a second
   method requires no change here.

The workflow is synchronous and side-effect free apart from the result
directory. The GUI wraps it in a ``QThread`` and injects ``Callbacks`` so log
lines, progress and stage changes can cross the thread boundary as Qt
signals; the one-click test runner calls the very same function directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .atlas import Atlas
from .config import DCCConfig, RunPaths
from .methods import get_method
from .network import (group_state_prob, high_frequency_states,
                      network_high_frequency_states, network_state_prob,
                      top_state_coverage)
from .plotting import (loadings_bar, plot_cohen_d, plot_explained_covariance,
                       plot_flexibility_profile, plot_group_profiles, plot_icc,
                       plot_matrix, plot_network_bars, plot_network_grouped_bars,
                       plot_state_heatmap, plot_state_patterns, plot_top_coverage)
from .pls import (bootstrap_stats, pls_analysis, pls_bootstrap,
                  pls_permutation)
from .results import ResultStore
from .stats import (cohen_network_counts, fdr_bh, group_summary, node_icc,
                    two_sample_ttest)
from .timeseries import discover_samples

# Stage names are shared with the GUI (``gui.app.STAGE_LABELS``). "Compute" is
# intentionally method-neutral so a future method does not need to rename it.
STAGES = ("Input", "Compute", "Summary", "Optional analyses", "Figures",
          "Export")


class AnalysisCancelled(Exception):
    """Raised when the user interrupts a running analysis.

    The GUI worker maps this to its ``sig_cancelled`` signal. Already computed
    samples remain in the cache, so restarting the run is cheap.
    """


def _log(msg: str) -> None:
    """Default log sink: unbuffered stdout, useful for scripts and tests."""
    print(msg, flush=True)


def _noop(*args, **kwargs) -> None:
    """Default no-op callback used when a caller does not need progress."""
    return None


def _not_cancelled() -> bool:
    """Default cancellation probe: a non-interactive run is never cancelled."""
    return False


@dataclass
class Callbacks:
    """Optional hooks for progress reporting.

    Passing plain callables (rather than Qt signals) keeps the computation
    core importable without PyQt6. ``gui.workers.AnalysisWorker`` simply wires
    these callables to ``pyqtSignal.emit``.

    log(str)                 -- one informational line
    progress(stage, done, total) -- fractional progress inside a stage
    stage(name, index, total)    -- a six-stage transition, 1-based index
    cancelled() -> bool          -- polled between samples
    """

    log: Callable[[str], None] = _log
    progress: Callable[[str, int, int], None] = _noop
    stage: Callable[[str, int, int], None] = _noop
    cancelled: Callable[[], bool] = _not_cancelled


@dataclass
class RunResult:
    """Everything a caller needs after a successful run.

    ``summary`` is the same dictionary written to ``summary.json``; ``files``
    and ``figures`` list the paths that were produced; ``stages`` carries the
    timing of each stage in seconds.
    """

    run_dir: Path
    tag: str
    summary: dict = field(default_factory=dict)
    figures: list = field(default_factory=list)
    files: list = field(default_factory=list)
    stages: list = field(default_factory=list)


def _safe(name: str) -> str:
    """Turn an arbitrary metadata value into a safe file-name fragment."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def _network_names(atlas: Atlas) -> np.ndarray:
    """Return one network name per ROI, including ``0`` ROIs as unassigned.

    ``Atlas.name`` uses zero-based indices over *distinct non-zero* labels, so
    a lookup table is built first; this avoids assuming labels are 1..G.
    """
    labels = np.unique(atlas.roi_idx[atlas.roi_idx != 0])
    lookup = {int(v): atlas.name(i) for i, v in enumerate(labels)}
    return np.array([lookup.get(int(v), "unassigned") for v in atlas.roi_idx])


def _roi_frame(atlas: Atlas) -> pd.DataFrame:
    """ROI number / network name columns shared by several output tables."""
    return pd.DataFrame({"roi": np.arange(1, atlas.n_roi + 1),
                         "network": _network_names(atlas)})


def _save_fig(store: ResultStore, name: str, fig, figures: list) -> None:
    """Save a figure, record its path, and release matplotlib memory.

    Closing the figure is essential: a full run creates a dozen or more, and
    keeping them alive would steadily grow memory across runs.
    """
    figures.append(store.save_figure(name, fig))
    plt.close(fig)


def _load_behaviour(path: Path, samples: list) -> tuple[np.ndarray, list]:
    """Read a numeric behaviour table aligned to the sample order.

    A ``sample_id`` column enables explicit matching; rows are reindexed to
    the discovered sample order so PLS never silently pairs a behaviour row
    with the wrong time series. Missing IDs are a hard error (the caller turns
    it into a warning and skips PLS), because a partially matched table would
    produce statistically meaningless loadings.

    Non-numeric columns are dropped, and columns that are entirely empty are
    removed as well. Returns ``(matrix, column_names)``.
    """
    frame = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in frame.columns}
    if "sample_id" in cols:
        ids = [s.sample_id for s in samples]
        frame = frame.set_index(cols["sample_id"])
        missing = [sid for sid in ids if sid not in frame.index]
        if missing:
            raise ValueError(f"behaviour table is missing {len(missing)} "
                             f"sample(s): {', '.join(map(str, missing[:3]))}")
        frame = frame.reindex(ids)
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric behaviour columns in {path}")
    return numeric.to_numpy(dtype=float), [str(c) for c in numeric.columns]


def run_analysis(paths: RunPaths, cfg: DCCConfig,
                 cb: Callbacks | None = None) -> RunResult:
    """Run the complete workflow and return its outputs.

    Parameters
    ----------
    paths:
        Validated input/output locations. Missing files raise
        ``FileNotFoundError`` before any result directory is created.
    cfg:
        Run parameters *and* the selected method name. ``cfg.validate()`` and
        the method's own ``validate()`` hook run first.
    cb:
        Optional progress/log callbacks; defaults to printing to stdout.

    Raises
    ------
    ValueError
        Unknown method, atlas/sample shape mismatch or window longer than the
        shortest time series.
    FileNotFoundError
        One or more configured input paths do not exist.
    AnalysisCancelled
        ``cb.cancelled()`` returned ``True`` between two samples.

    Notes
    -----
    Per-sample results are cached as ``samples/<id>.npz`` inside the run
    directory. Because ``cfg.run_tag()`` includes the method name for
    non-default methods, caches from different methods never collide.
    """
    cb = cb or Callbacks()
    cfg.validate()
    method = get_method(cfg.method)
    method.validate(cfg)
    prefix = method.output_prefix()
    problems = paths.check_paths()
    if problems:
        raise FileNotFoundError("\n".join(problems))

    # Timings are collected for summary.json, so a slow stage is visible
    # without having to profile the application.
    timings: list[dict] = []

    def begin(index: int) -> float:
        """Announce a stage and return its high-resolution start time."""
        cb.stage(STAGES[index], index + 1, len(STAGES))
        return time.perf_counter()

    def end(index: int, started: float) -> None:
        """Record the elapsed time of the stage that just finished."""
        timings.append({"stage": STAGES[index],
                        "seconds": round(time.perf_counter() - started, 3)})

    # ---------------------------------------------------------------- Input
    # Discovery is intentionally eager: shape errors are reported before any
    # expensive computation or disk writes take place.
    started = begin(0)
    atlas = Atlas.load(paths.atlas_labels, paths.atlas_names)
    samples = discover_samples(paths.data_source, cfg.variable or None,
                               paths.metadata)
    if samples[0].data.shape[0] != atlas.n_roi:
        raise ValueError(f"Atlas has {atlas.n_roi} ROIs but the data have "
                         f"{samples[0].data.shape[0]}")
    shortest = min(s.data.shape[1] for s in samples)
    if cfg.swl > shortest:
        raise ValueError(f"SWL={cfg.swl} exceeds the shortest time series "
                         f"({shortest} time points)")
    store = ResultStore(paths.output_root / cfg.run_tag())
    cb.log(f"Input: {len(samples)} samples | {atlas.n_roi} ROIs | "
           f"{atlas.n_net} networks | {atlas.n_states} states | SWL={cfg.swl}")
    cb.log(f"Output: {store.root}")
    end(0, started)

    # -------------------------------------------------------------- Compute
    # This is the only stage that knows about a concrete method. Everything
    # downstream consumes the common SubjectResult structure.
    started = begin(1)
    results, computed, cached = [], 0, 0
    step = max(1, len(samples) // 20)   # ~20 progress updates for any cohort
    for i, sample in enumerate(samples):
        if cb.cancelled():
            raise AnalysisCancelled()
        if store.has_sample(sample.sample_id):
            res = store.load_sample(sample.sample_id)
            cached += 1
            flag = "cache"
        else:
            res = method.compute(sample.data, atlas, cfg,
                                 subject_id=sample.sample_id)
            store.save_sample(sample.sample_id, res, cfg.save_binary_dfc)
            computed += 1
            flag = "computed"
        results.append(res)
        if (i + 1) % step == 0 or (i + 1) == len(samples):
            cb.progress("Compute", i + 1, len(samples))
            cb.log(f"[{i + 1}/{len(samples)}] {sample.sample_id}: {flag}, "
                   f"windows={res.n_windows}, threshold={res.threshold:.4f}, "
                   f"mean {method.metric}={res.entropy.mean():.3f}")
    cb.log(f"{method.metric} finished: {computed} computed, "
           f"{cached} loaded from cache")
    end(1, started)

    # -------------------------------------------------------------- Summary
    # All CSV/JSON outputs are written through ResultStore so the file naming
    # and encoding stay consistent and the GUI can list them generically.
    started = begin(2)
    summary = group_summary(results, atlas)
    roi_frame = _roi_frame(atlas)
    sample_ids = [s.sample_id for s in samples]
    files: list[Path] = []

    matrix = pd.DataFrame(summary["node_temporal"], columns=sample_ids)
    files.append(store.save_csv(f"{prefix}_matrix.csv",
                                pd.concat([roi_frame, matrix], axis=1)))
    long_rows = []
    for sample, res in zip(samples, results):
        for roi, value in enumerate(res.entropy, start=1):
            long_rows.append({"sample_id": sample.sample_id,
                              "group": sample.group,
                              "subject": sample.subject,
                              "session": sample.session,
                              "roi": roi,
                              "network": _network_names(atlas)[roi - 1],
                              prefix: float(value)})
    files.append(store.save_csv(f"{prefix}_per_sample.csv",
                                pd.DataFrame(long_rows)))
    sd = (summary["node_temporal"].std(axis=1, ddof=1)
          if len(samples) > 1 else np.zeros(atlas.n_roi))
    files.append(store.save_csv("group_summary.csv", pd.DataFrame({
        "roi": roi_frame["roi"], "network": roi_frame["network"],
        f"mean_{prefix}": summary["group_flexibility"],
        f"sd_{prefix}": sd,
        f"mean_{prefix}_z": summary["group_flexibility_z"]})))
    files.append(store.save_csv("network_summary.csv", pd.DataFrame({
        "network": [atlas.name(i) for i in range(atlas.n_net)],
        "n_roi": atlas.sizes,
        f"mean_{prefix}": summary["net_flexibility"],
        f"mean_{prefix}_z": summary["net_flexibility_z"]})))

    group_prob = group_state_prob(results)
    net_prob = network_state_prob(group_prob, atlas)
    state_cols = [f"state_{i + 1}" for i in range(atlas.n_states)]
    files.append(store.save_csv("state_probability.csv",
                                pd.concat([roi_frame,
                                           pd.DataFrame(group_prob,
                                                        columns=state_cols)],
                                          axis=1)))
    files.append(store.save_csv("network_state_probability.csv",
                                pd.DataFrame(net_prob, columns=state_cols)
                                .assign(network=[atlas.name(i)
                                                 for i in range(atlas.n_net)])
                                .set_index("network").reset_index()))
    high = high_frequency_states(group_prob, cfg.state_prob_threshold, atlas.n_net)
    files.append(store.save_csv("high_frequency_states.csv", high))
    for name, table in network_high_frequency_states(
            net_prob, cfg.state_prob_threshold, atlas).items():
        files.append(store.save_csv(f"high_states_{_safe(name)}.csv", table))
    coverage = top_state_coverage(group_prob, cfg.top_state_fraction)
    files.append(store.save_json("top_state_coverage.json", coverage))
    if len(samples) > 1:
        files.append(store.save_csv("sample_similarity.csv",
                                    pd.DataFrame(summary["sample_correlation"],
                                                 index=sample_ids,
                                                 columns=sample_ids),
                                    index=True))
    files.append(store.save_json("run_manifest.json", {
        "method": method.name,
        "data_source": str(paths.data_source),
        "atlas_labels": str(paths.atlas_labels),
        "atlas_names": str(paths.atlas_names or ""),
        "metadata": str(paths.metadata or ""),
        "behaviour": str(paths.behaviour or ""),
        "samples": [{"sample_id": s.sample_id, "shape": list(s.shape),
                     "group": s.group, "subject": s.subject,
                     "session": s.session, "source": s.source}
                    for s in samples],
    }))
    end(2, started)

    # ------------------------------------------------------- Optional analyses
    # Each block is guarded purely by the available metadata. A failure in an
    # optional block (for example a malformed behaviour table) is recorded as
    # a warning; the core results above are always produced.
    started = begin(3)
    analyses: list[str] = []
    warnings: list[str] = []
    group_means: dict[str, np.ndarray] = {}
    group_network: dict[str, np.ndarray] = {}
    comparisons: list[tuple] = []
    icc_values = None
    pls_outputs = None

    groups: dict[str, list[int]] = {}
    for i, sample in enumerate(samples):
        if sample.group:
            groups.setdefault(sample.group, []).append(i)
    usable = {name: idx for name, idx in groups.items() if len(idx) >= 2}
    if len(usable) >= 2:
        analyses.append("group comparison")
        for name, idx in usable.items():
            sub = group_summary([results[i] for i in idx], atlas)
            group_means[name] = sub["group_flexibility"]
            group_network[name] = sub["net_flexibility"]
            files.append(store.save_csv(f"group_{_safe(name)}_{prefix}.csv",
                                        pd.concat([roi_frame, pd.DataFrame({
                                            f"mean_{prefix}":
                                                sub["group_flexibility"]})],
                                                  axis=1)))
        names = list(usable)
        pairs = [(a, b) for i, a in enumerate(names)
                 for b in names[i + 1:]][:12]
        for a, b in pairs:
            stat = two_sample_ttest(
                np.column_stack([results[i].entropy for i in usable[a]]),
                np.column_stack([results[i].entropy for i in usable[b]]))
            files.append(store.save_csv(
                f"comparison_{_safe(a)}_vs_{_safe(b)}.csv", stat))
            counts = cohen_network_counts(stat["cohens_d"].to_numpy(), atlas,
                                          cfg.cohen_threshold)
            files.append(store.save_csv(
                f"comparison_{_safe(a)}_vs_{_safe(b)}_networks.csv", counts))
            comparisons.append((a, b, stat, counts))
        files.append(store.save_csv(f"group_network_{prefix}.csv",
                                    pd.DataFrame({"network": [atlas.name(i)
                                                             for i in range(atlas.n_net)],
                                                  **group_network})))
    elif len(groups) >= 2:
        warnings.append("Group comparison skipped: each group needs >= 2 samples")

    # ICC requires a balanced repeated-measures layout: every subject must
    # have the same number of sessions, with at least two of each.
    if all(s.subject and s.session for s in samples):
        by_subject: dict[str, list] = {}
        for i, sample in enumerate(samples):
            by_subject.setdefault(sample.subject, []).append((sample.session, i))
        session_counts = {len(v) for v in by_subject.values()}
        if len(by_subject) >= 2 and len(session_counts) == 1 and min(session_counts) >= 2:
            order = [i for subject in sorted(by_subject)
                     for _, i in sorted(by_subject[subject], key=lambda x: x[0])]
            flex = np.column_stack([results[i].entropy for i in order]).T
            icc_values = node_icc(flex, len(by_subject), min(session_counts))
            files.append(store.save_csv("icc.csv", pd.DataFrame({
                "roi": roi_frame["roi"], "network": roi_frame["network"],
                "icc_1_1": icc_values})))
            analyses.append("ICC(1,1)")

    # PLS-C: SVD of the ROI-by-behaviour covariance, then within-group
    # permutation and bootstrap resampling for significance/stability.
    if paths.behaviour and paths.behaviour.exists():
        try:
            behaviour, behaviour_names = _load_behaviour(paths.behaviour, samples)
            x = np.column_stack([r.entropy for r in results]).T
            if x.shape[0] != behaviour.shape[0]:
                warnings.append("PLS skipped: behaviour rows do not match samples")
            else:
                labels = sorted({s.group for s in samples if s.group})
                grouping = np.array([labels.index(s.group) if s.group in labels else 0
                                     for s in samples])
                pls = pls_analysis(x, behaviour, cfg.norm_img, cfg.norm_behav,
                                   grouping)
                cb.progress("PLS permutation", 0, max(1, cfg.n_perms))
                pvals = pls_permutation(
                    x, behaviour, pls["singular_values"], cfg.n_perms, grouping,
                    cfg.norm_img, cfg.norm_behav, cfg.random_seed,
                    progress=lambda d, t: cb.progress("PLS permutation", d, t))
                signif = list(fdr_bh(pvals[:3], cfg.fdr_q))
                if not signif:
                    signif = list(range(min(cfg.pls_n_lc, len(pvals))))
                cb.progress("PLS bootstrap", 0, max(1, cfg.n_bootstraps))
                boot = pls_bootstrap(
                    x, behaviour, pls["U"], signif, cfg.n_bootstraps, grouping,
                    cfg.norm_img, cfg.norm_behav, cfg.random_seed,
                    progress=lambda d, t: cb.progress("PLS bootstrap", d, t))
                rs = bootstrap_stats(pls["rsfc_loadings"][:, signif],
                                     boot["rsfc_boot"])
                bs = bootstrap_stats(pls["behav_loadings"][:, signif],
                                     boot["behav_boot"])
                files.append(store.save_csv("pls_explained_covariance.csv",
                                            pd.DataFrame({
                                                "lc": np.arange(1, len(pvals) + 1),
                                                "explained_covariance": pls["expl_cov"],
                                                "p_value": pvals})))
                files.append(store.save_csv("pls_behaviour_loadings.csv",
                                            pd.DataFrame(bs["z"],
                                                         columns=[f"LC{i + 1}"
                                                                  for i in signif])))
                files.append(store.save_csv(f"pls_{prefix}_loadings.csv",
                                            pd.DataFrame(rs["z"],
                                                         columns=[f"LC{i + 1}"
                                                                  for i in signif])))
                pls_outputs = {"expl": pls["expl_cov"], "behaviour_names": behaviour_names,
                               "behav_z": bs["z"], "dcc_z": rs["z"],
                               "signif": signif}
                analyses.append("PLS-C")
        except Exception as exc:         # optional module must not break the core
            warnings.append(f"PLS skipped: {type(exc).__name__}: {exc}")
            cb.log(f"Warning: PLS skipped ({exc})")
    for warning in warnings:
        cb.log(f"Warning: {warning}")
    end(3, started)

    # --------------------------------------------------------------- Figures
    # Each helper returns a Figure; _save_fig writes and closes it. Figures
    # for optional analyses are only produced when those analyses ran.
    started = begin(4)
    figures: list[Path] = []
    _save_fig(store, f"01_{prefix}_profile.png",
              plot_flexibility_profile(summary["group_flexibility"], atlas,
                                       f"Group mean {method.metric} profile"), figures)
    _save_fig(store, f"02_network_{prefix}.png",
              plot_network_bars(summary["net_flexibility"], atlas), figures)
    _save_fig(store, "03_state_probability.png",
              plot_state_heatmap(group_prob, "ROI state probability (group mean)"),
              figures)
    _save_fig(store, "04_network_state_probability.png",
              plot_state_heatmap(net_prob, "Network state probability",
                                 ylabel="Network"), figures)
    if len(samples) > 1:
        _save_fig(store, "05_sample_similarity.png",
                  plot_matrix(summary["sample_correlation"],
                              f"Between-sample {method.metric} similarity",
                              xlabel="Sample", ylabel="Sample"), figures)
    _save_fig(store, "06_high_state_patterns.png",
              plot_state_patterns(high.head(20), atlas), figures)
    _save_fig(store, "07_top_state_coverage.png",
              plot_top_coverage(coverage), figures)
    if group_means:
        _save_fig(store, "08_group_profiles.png",
                  plot_group_profiles(group_means, atlas), figures)
    for a, b, stat, _ in comparisons:
        _save_fig(store, f"09_{_safe(a)}_vs_{_safe(b)}_cohen_d.png",
                  plot_cohen_d(stat["cohens_d"].to_numpy(), atlas,
                               cfg.cohen_threshold, f"{a} vs {b}: Cohen's d"),
                  figures)
    if icc_values is not None:
        _save_fig(store, "10_icc.png", plot_icc(icc_values, atlas), figures)
    if pls_outputs is not None:
        _save_fig(store, "11_pls_explained_covariance.png",
                  plot_explained_covariance(pls_outputs["expl"]), figures)
        for j, lc in enumerate(pls_outputs["signif"][:3]):
            _save_fig(store, f"12_pls_behaviour_loadings_LC{lc + 1}.png",
                      loadings_bar(pls_outputs["behav_z"][:, j],
                                   pls_outputs["behaviour_names"], lc,
                                   f"LC{lc + 1} behaviour loadings (bootstrap z)",
                                   max_items=25), figures)
            _save_fig(store, f"13_pls_{prefix}_loadings_LC{lc + 1}.png",
                      plot_flexibility_profile(pls_outputs["dcc_z"][:, j], atlas,
                                               f"LC{lc + 1} ROI loadings (bootstrap z)",
                                               ylabel="z"), figures)
    end(4, started)

    # ---------------------------------------------------------------- Export
    # summary.json is written last so it always describes files that really
    # exist. The GUI reads it to populate the Results page.
    started = begin(5)
    cb.progress("Export", 1, 1)
    end(5, started)
    info = {
        "method": method.name,
        "method_title": method.title,
        "run_tag": cfg.run_tag(),
        "n_samples": len(samples),
        "n_roi": atlas.n_roi,
        "n_networks": atlas.n_net,
        "n_states": atlas.n_states,
        "n_windows": results[0].n_windows,
        "computed": computed,
        "cached": cached,
        "mean_dcc": float(summary["group_flexibility"].mean()),
        "network_mean_dcc": {atlas.name(i): float(v) for i, v in
                             enumerate(summary["net_flexibility"])},
        "n_high_states": int(len(high)),
        "top_state_coverage": coverage["top_coverage"],
        "analyses": analyses,
        "warnings": warnings,
        "parameters": cfg.to_dict(),
        "stages": timings,
    }
    files.append(store.save_json("summary.json", info))
    cb.log(f"Done: {len(figures)} figures, {len(files)} files -> {store.root}")
    return RunResult(store.root, cfg.run_tag(), info, figures, files, timings)
