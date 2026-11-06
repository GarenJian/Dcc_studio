"""Group-level statistics, test-retest reliability and group comparisons.

These helpers are all *optional* stages of the workflow: they are invoked only
when the supplied metadata makes them meaningful, never because of a study
name. They consume the common per-sample result structure, so they work for
any registered analysis method.

* ``group_summary``  -- per-ROI mean/z-score, network means, and the
  between-sample correlation matrix of per-ROI profiles;
* ``two_sample_ttest`` -- Welch-style two-sample t statistic, p value and
  Cohen's d, returned as a ``(n_roi, 3)`` DataFrame;
* ``node_icc``       -- ICC(1,1) from a balanced subjects x sessions layout,
  derived from a one-way random-effects ANOVA;
* ``cohen_network_counts`` -- how many ROIs per network exceed an effect-size
  threshold, separately for positive and negative effects;
* ``fdr_bh``         -- Benjamini-Hochberg correction for PLS p-values.

Numerical formulas here are covered by the committed reference fixtures: the
t/Cohen's d outputs are compared element-by-element against published results,
so a change in ``ddof`` or pooled variance would fail the suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .atlas import Atlas
from .network import net_analysis


def group_summary(results: list, atlas: Atlas) -> dict:
    """Summarise DCC across samples."""
    node_temporal = np.column_stack([r.entropy for r in results])
    node_temporal = np.nan_to_num(node_temporal, nan=0.0)
    node_prob = np.stack([r.states_prob for r in results], axis=2)
    group_flex = node_temporal.mean(axis=1)
    z = sps.zscore(group_flex, ddof=1)
    corr = (np.corrcoef(node_temporal.T)
            if node_temporal.shape[1] > 1 else np.ones((1, 1)))
    _, net_raw = net_analysis(group_flex, atlas.roi_idx)
    _, net_z = net_analysis(z, atlas.roi_idx)
    return {
        "node_temporal": node_temporal,
        "node_prob": node_prob,
        "sample_correlation": corr,
        "group_flexibility": group_flex,
        "group_flexibility_z": z,
        "net_flexibility": net_raw,
        "net_flexibility_z": net_z,
        "n_samples": len(results),
    }


def icc_1_1(data: np.ndarray) -> float:
    """ICC(1,1), one-way random effects, single measurement.

    data : (n_targets, n_measurements), e.g. subjects x sessions.
    """
    d = np.asarray(data, dtype=float)
    n, k = d.shape
    if k < 2 or n < 2:
        return np.nan
    grand = d.mean()
    msr = k * np.sum((d.mean(axis=1) - grand) ** 2) / (n - 1)
    msw = np.sum((d - d.mean(axis=1, keepdims=True)) ** 2) / (n * (k - 1))
    den = msr + (k - 1) * msw
    return float((msr - msw) / den) if den > 0 else np.nan


def node_icc(flex_mat: np.ndarray, n_sub: int, n_ses: int) -> np.ndarray:
    """ICC(1,1) per ROI.

    flex_mat : (n_sub * n_ses, n_roi), rows ordered subject by subject.
    """
    flex_mat = np.nan_to_num(np.asarray(flex_mat, dtype=float), nan=0.0)
    return np.array([icc_1_1(flex_mat[:, i].reshape(n_sub, n_ses))
                     for i in range(flex_mat.shape[1])])


def two_sample_ttest(group_a: np.ndarray, group_b: np.ndarray) -> pd.DataFrame:
    """Independent two-sample t-test and Cohen's d, one row per ROI.

    group_a / group_b : (n_roi, n_samples)
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    t, p = sps.ttest_ind(group_a.T, group_b.T, axis=0, equal_var=True)
    n1, n2 = group_a.shape[1], group_b.shape[1]
    pooled = np.sqrt(((n1 - 1) * group_a.var(axis=1, ddof=1)
                      + (n2 - 1) * group_b.var(axis=1, ddof=1))
                     / (n1 + n2 - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (group_a.mean(axis=1) - group_b.mean(axis=1)) / pooled
    return pd.DataFrame({"t": np.asarray(t), "p": np.asarray(p), "cohens_d": d})


def cohen_network_counts(d: np.ndarray, atlas: Atlas,
                         thr: float = 0.3) -> pd.DataFrame:
    """Count ROIs with |d| above a threshold, per network."""
    d = np.asarray(d, dtype=float).ravel()
    rows = []
    for i in range(atlas.n_net):
        values = d[atlas.roi_idx == i + 1]
        rows.append({"network": atlas.name(i), "n_roi": len(values),
                     "d_gt_thr": int((values > thr).sum()),
                     "d_lt_thr": int((values < -thr).sum()),
                     "mean_d": float(values.mean())})
    return pd.DataFrame(rows)


def fdr_bh(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR; returns significant zero-based indices."""
    p = np.asarray(pvals, dtype=float).ravel()
    n = len(p)
    if n == 0:
        return np.array([], dtype=int)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, n + 1) / n)
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    return np.sort(order[:k])
