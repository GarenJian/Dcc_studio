# Method: Dynamic Connectivity Complexity (DCC)

DCC Studio computes a single, dataset-independent quantity: for every region of
interest (ROI), the Shannon entropy of the distribution of its time-varying
connectivity states with respect to a set of networks.

## Inputs

| Input | Description |
|---|---|
| Time series | One matrix per sample, shaped `(n_roi, n_timepoints)`. A single file may contain several samples (3-D array or cell array), or a folder may contain one file per sample. Supported: `.mat`, `.npy`, `.npz`, `.csv`, `.tsv`, `.txt`. |
| Atlas labels | One integer network index per ROI, in ROI order. Any number of ROIs and networks is supported (up to 16 networks). |
| Network names | Optional file mapping network indices to readable names, or one name per line. |
| Sample metadata | Optional CSV with `sample_id`, `group`, `subject`, `session`. Enables group comparisons and ICC. |
| Behaviour table | Optional CSV with numeric columns (and optionally `sample_id`). Enables PLS-C. |

A sample is any independent time-series observation: a subject, a session, a
run, or a condition. The software makes no assumption about study design.

## Algorithm

Let `X` be the `(R, T)` ROI signal matrix of one sample and `L` the `(R,)`
atlas label vector with `G` distinct non-zero networks.

1. **Network signals.** `N_g(t) = mean_{i: L_i = g} X_i(t)` for each network `g`.
2. **Dynamic functional connectivity.** For window `w` of length `W = SWL`
   starting every `SWS` time points, compute the Pearson correlation between
   each ROI signal and each network signal:
   `D[i, g, w] = corr(X_i[w:w+W], N_g[w:w+W])`.
   The number of windows is `1 + floor((T - W) / SWS)`.
3. **Sign handling and sparsification.** Apply `abs`, `raw` or `pos`, then
   binarise with `B = (D > mean(D) + k * std(D, ddof=1))` (global mode), or
   with a per-ROI threshold (roi mode).
4. **Connectivity states.** Each window yields a `G`-bit pattern for every ROI.
   The pattern is decoded as an integer using weights `[2^(G-1), ..., 1]`. The
   all-zero state is discarded, leaving `2^G - 1` states. Count the occurrences
   of each state across windows for each ROI.
5. **DCC.** Convert counts to probabilities `p` and compute
   `DCC_i = -sum_s p_{i,s} log2(p_{i,s})`, in bits. Higher values indicate a
   more diverse repertoire of connectivity states.

Identical parameters and input produce identical results; the random seed only
affects optional permutation / bootstrap analyses.

## Group-level results

Across samples the software reports the mean, standard deviation and z-scored
DCC per ROI, network means, mean state probabilities (per ROI and per network),
high-frequency states, top-state coverage and between-sample DCC similarity.

## Optional analyses (driven by metadata, never by dataset names)

| Analysis | Enabled when | Output |
|---|---|---|
| Group comparison | >= 2 groups with >= 2 samples each | Per-group DCC, pairwise t tests, Cohen's d, per-network counts |
| ICC(1,1) | Every sample has `subject` and `session`, with a balanced repeated-measures layout | Per-ROI test-retest reliability |
| PLS-C | A numeric behaviour table aligned to the samples | Explained covariance, permutation p values, bootstrapped loadings |

Any optional analysis that cannot be run is reported as a warning in
`summary.json`; the core DCC results are always produced.

## Outputs

`<output_root>/<run_tag>/` where `run_tag` encodes the parameters, e.g.
`swl30_sws1_k0.5_abs_global`. Different parameter sets never overwrite each
other.

| File | Content |
|---|---|
| `dcc_matrix.csv` | ROI x sample DCC matrix with network labels |
| `dcc_per_sample.csv` | Long-format DCC with all available metadata |
| `group_summary.csv`, `network_summary.csv` | Group and network aggregates |
| `state_probability.csv`, `network_state_probability.csv` | State distributions |
| `high_frequency_states.csv`, `top_state_coverage.json` | State statistics |
| `sample_similarity.csv` | Between-sample correlation of DCC profiles |
| `comparison_*.csv`, `icc.csv`, `pls_*.csv` | Optional analyses |
| `figures/*.png` | Publication-style figures |
| `run_manifest.json`, `summary.json` | Inputs, parameters, statistics, stage timings |
| `samples/*.npz` | Per-sample cache for fast re-runs |

## Numerical validation

Results are checked against the reference implementation on a separately
distributed dataset. `tests/fixtures/` holds verbatim copies of the reference
project's inputs and published results (~146 MB, not committed): time-series
containers, atlas labels and names, per-sample DCC results, group statistics,
t tests / Cohen's d and PLS outputs. Download it into `tests/fixtures/` before
running the parity suite; see [`test.md`](test.md) for the link (to be filled in).
`tests/test_reference_parity.py` recomputes every quantity element-by-element
and compares it with the stored result. The latest run performed 35 numerical
comparisons with 0 tolerance failures (DCC 1e-5, state probabilities 1e-6,
statistics and PLS 1e-8; exact comparisons use 0). No external folder, no
generation script and no user-configured environment variable is needed.

Chapter 1 of `reports/test-report.md` records the full comparison -- dataset,
reference file, stored and recomputed min / max / mean, largest deviation,
tolerance and verdict per case -- and the structured records are also written
to `reports/functional-details.json`. For the one-click reproduction procedure
and the report structure, see [`test.md`](test.md).

## Extending the software with another method

DCC is one way of characterising time-varying connectivity. The workflow,
statistics, figures, result exporter and GUI are method-agnostic; only the
per-sample computation is pluggable.

The contract lives in `dcc/methods/base.py`. A method declares a stable `name`
(used in configuration, result directories and file names), a `title` and
`description` for the GUI, a `metric` abbreviation for labels, and implements:

```
compute(ts, atlas, cfg, swl=None, subject_id="") -> SubjectResult
```

`SubjectResult.entropy` carries the per-ROI measure, `states_prob` and
`states_counts` carry any state distribution, and `dfc_binary` is an optional
debug tensor. All downstream stages consume this common structure.

To add a method:

1. Subclass `AnalysisMethod` in a new module under `dcc/methods/`, implement
   `compute()` and optionally `validate()` / `extra_parameters()`.
2. Decorate the class with `@register_method` and import it once from
   `dcc/methods/__init__.py` (registration happens at import time, which also
   keeps PyInstaller's dependency graph explicit).
3. Nothing else changes. The Settings drop-down on page 1 is filled from
   `method_choices()`, the selected name is stored in `config.json`,
   `run_tag()` prefixes the result directory so caches never mix methods,
   output files use the method name as prefix, and `summary.json` records
   which method produced the run.

Methods that are planned but not implemented are listed in `gui/settings.py`
as `PLANNED_METHODS`. They appear in the drop-down as disabled, greyed-out
rows (currently "Standard dynamic functional connectivity", i.e. classic
sliding-window dFC without state coding or entropy). Once a method with the
same stable `name` is registered, its placeholder is skipped and the real
entry becomes selectable, so the roadmap and the implementation can never
show a contradictory duplicate.

Step-by-step Word guides for operating the software are available in
`docs/DCC_Studio_使用教程_中文.docx` (Chinese) and
`docs/DCC_Studio_User_Guide_EN.docx` (English).

Keep `name` stable once published: renaming it would make existing caches
unreachable. Methods must be deterministic; permutation/bootstrap randomness
is controlled by `cfg.random_seed` in the optional-analysis stage.
