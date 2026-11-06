"""PLS-C association between per-ROI values and optional behaviour variables.

Partial Least Squares Correlation (PLS-C) finds latent components that maximally
covary an imaging matrix ``X`` (samples x ROIs) with a behaviour matrix ``Y``
(samples x behaviours). The workflow runs this analysis only when a behaviour
table is supplied and its rows align with the discovered samples.

Pipeline
--------
1. Optionally normalise X/Y, substitute column means for missing values, then
   compute the cross-covariance ``M = X^T Y`` (with group centring when group
   labels are available).
2. Singular-value decomposition of ``M`` yields ``U`` (behaviour loadings),
   ``V`` (ROI loadings), singular values, and the latent-variable scores
   ``Lx = X V`` / ``Ly = Y U``. Explained covariance is
   ``singular_value^2 / sum(singular_value^2)``.
3. ``pls_permutation`` shuffles rows within groups to produce a p-value per
   component; ``pls_bootstrap`` resamples subjects to assess the stability of
   loadings; ``bootstrap_stats`` converts the bootstrap distribution to z-like
   ratios. ``fdr_bh`` in ``dcc.stats`` selects significant components.

Signs of singular vectors are arbitrary, so parity tests compare absolute
values. All randomness flows from ``cfg.random_seed`` and separate NumPy
generators are used for permutations and bootstraps so one does not perturb the
other.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def _fill_nan(a: np.ndarray) -> np.ndarray:
    """Replace missing values with column means so SVD can run on incomplete tables."""
    a = np.asarray(a, dtype=float).copy()
    if np.isnan(a).any():
        col_mean = np.nanmean(a, axis=0)
        idx = np.where(np.isnan(a))
        a[idx] = np.take(col_mean, idx[1])
    return a


def normalize(a: np.ndarray, mode: int = 1,
              grouping: np.ndarray | None = None) -> np.ndarray:
    """Normalise imaging / behaviour matrices.

    mode 0: none; 1: z-score over all samples; 2: z-score within group;
            3: divide by global SD; 4: divide by within-group SD.
    """
    a = _fill_nan(a)
    if mode == 0:
        return a
    if mode in (1, 3):
        groups = [np.arange(a.shape[0])]
    else:
        g = (np.asarray(grouping).ravel() if grouping is not None
             else np.zeros(a.shape[0]))
        groups = [np.flatnonzero(g == v) for v in np.unique(g)]
    out = a.copy()
    for idx in groups:
        mu = out[idx].mean(axis=0) if mode in (1, 2) else 0.0
        sd = out[idx].std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        out[idx] = (out[idx] - mu) / sd
    return out


def colcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of a with every column of b."""
    a = _fill_nan(a) - _fill_nan(a).mean(axis=0)
    b = _fill_nan(b) - _fill_nan(b).mean(axis=0)
    den = (np.sqrt((a ** 2).sum(axis=0))[:, None]
           * np.sqrt((b ** 2).sum(axis=0))[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a.T @ b) / den
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _svd_core(x: np.ndarray, y: np.ndarray) -> tuple:
    """Shared SVD step: build cross-covariance, decompose it and return all PLS quantities."""
    n = x.shape[0]
    r = y.T @ x / (n - 1)
    u, s, vt = np.linalg.svd(r, full_matrices=False)
    return u, s, vt.T


def pls_analysis(x: np.ndarray, y: np.ndarray, norm_img: int = 1,
                 norm_behav: int = 1, grouping: np.ndarray | None = None) -> dict:
    """Main PLS analysis: SVD -> latent components and loadings."""
    x = _fill_nan(x)
    y = _fill_nan(y)
    xn = normalize(x, norm_img, grouping)
    yn = normalize(y, norm_behav, grouping)
    u, s, v = _svd_core(xn, yn)
    lx, ly = xn @ v, yn @ u
    return {
        "X": x, "Y": y,
        "U": u, "S": np.diag(s), "singular_values": s,
        "V": v, "Lx": lx, "Ly": ly,
        "expl_cov": s ** 2 / np.sum(s ** 2),
        "rsfc_loadings": colcorr(x, lx),
        "behav_loadings": colcorr(y, ly),
        "R": yn.T @ xn / (x.shape[0] - 1),
    }


def pls_permutation(x: np.ndarray, y: np.ndarray, singular_values: np.ndarray,
                    n_perm: int, grouping: np.ndarray | None = None,
                    norm_img: int = 1, norm_behav: int = 1,
                    seed: int = 42, progress=None) -> np.ndarray:
    """Within-group permutation test on the singular values."""
    rng = np.random.default_rng(seed)
    xn = normalize(x, norm_img, grouping)
    yn = normalize(y, norm_behav, grouping)
    g = (np.asarray(grouping).ravel() if grouping is not None
         else np.zeros(x.shape[0]))
    blocks = [np.flatnonzero(g == v) for v in np.unique(g)]
    k = len(singular_values)
    exceed = np.zeros(k)
    for it in range(int(n_perm)):
        order = np.arange(x.shape[0])
        for blk in blocks:
            order[blk] = rng.permutation(blk)
        _, s, _ = _svd_core(xn[order], yn)
        exceed += (s[:k] >= singular_values[:k]).astype(float)
        if progress and (it + 1) % max(1, n_perm // 10) == 0:
            progress(it + 1, n_perm)
    if n_perm > 0:
        return (1.0 + exceed) / (1.0 + n_perm)
    return np.ones(k)


def _procrustes(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Orthogonal rotation aligning source columns to target columns."""
    p, _, q = np.linalg.svd(source.T @ target, full_matrices=False)
    return p @ q


def pls_bootstrap(x: np.ndarray, y: np.ndarray, u: np.ndarray,
                  signif_lc: list, n_boot: int, grouping: np.ndarray | None = None,
                  norm_img: int = 1, norm_behav: int = 1,
                  seed: int = 42, progress=None) -> dict:
    """Within-group bootstrap distribution of the loadings."""
    rng = np.random.default_rng(seed)
    x, y = _fill_nan(x), _fill_nan(y)
    g = (np.asarray(grouping).ravel() if grouping is not None
         else np.zeros(x.shape[0]))
    blocks = [np.flatnonzero(g == v) for v in np.unique(g)]
    lcs = [int(i) for i in signif_lc]
    rsfc = np.zeros((x.shape[1], len(lcs), int(n_boot)))
    behav = np.zeros((y.shape[1], len(lcs), int(n_boot)))
    for b in range(int(n_boot)):
        order = np.concatenate([rng.choice(blk, size=len(blk), replace=True)
                                for blk in blocks])
        xb, yb = normalize(x[order], norm_img), normalize(y[order], norm_behav)
        ub, _, vb = _svd_core(xb, yb)
        rot = _procrustes(u, ub)
        ub, vb = ub @ rot, vb @ rot
        rl, bl = colcorr(xb, xb @ vb), colcorr(yb, yb @ ub)
        for j, lc in enumerate(lcs):
            rsfc[:, j, b] = rl[:, lc]
            behav[:, j, b] = bl[:, lc]
        if progress and (b + 1) % max(1, n_boot // 10) == 0:
            progress(b + 1, n_boot)
    return {"rsfc_boot": rsfc, "behav_boot": behav, "signif_lc": lcs}


def bootstrap_stats(loadings: np.ndarray, boot: np.ndarray) -> dict:
    """Standard error, z score and one-sided p from a bootstrap distribution."""
    l = np.asarray(loadings, dtype=float)
    std = np.asarray(boot, dtype=float).std(axis=2, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 0, l / np.where(std > 0, std, 1.0), 0.0)
    return {"std": std, "z": z, "p": 1.0 - sps.norm.cdf(np.abs(z))}
