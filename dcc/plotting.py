"""Figures for DCC results; every function returns a matplotlib ``Figure``.

The module is deliberately headless: it selects the ``Agg`` backend at import
time, so it works on a server, inside a test runner and inside a frozen GUI
build. No function saves or shows a figure; the workflow owns file naming and
calls ``ResultStore.save_figure``, then closes the figure to release memory.

What each helper draws
----------------------
* ``plot_flexibility_profile`` -- per-ROI measure with vertical network
  boundaries (the shared x-axis layout for most one-dimensional results);
* ``plot_network_bars`` / ``plot_network_grouped_bars`` -- network summaries;
* ``plot_state_heatmap`` -- ROI (or network) x state probability;
* ``plot_matrix`` -- symmetric matrices such as sample similarity;
* ``plot_state_patterns`` / ``plot_top_coverage`` -- state statistics;
* ``plot_group_profiles`` / ``plot_cohen_d`` / ``plot_icc`` -- optional analyses;
* ``plot_explained_covariance`` / ``loadings_bar`` -- PLS-C components.

All helpers accept plain NumPy arrays and an optional ``Atlas``; none of them
import the configuration, workflow or GUI layers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .atlas import Atlas

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.size"] = 10

PALETTE = ("#7b3fb2", "#3f7fc1", "#2f9e8f", "#8fbf3f", "#e2b23c",
           "#e07a3f", "#c04463", "#556b2f", "#4682b4", "#8e44ad")
CMAP = LinearSegmentedColormap.from_list("dcc", ["#f7fbff", "#4a90c2", "#12365c"])


def _color(i: int) -> str:
    """Return a stable palette colour for a network index."""
    return PALETTE[i % len(PALETTE)]


def _net_bands(ax, atlas: Atlas) -> None:
    """Compute the x positions of network boundaries on a per-ROI axis."""
    pos = 0
    for i, size in enumerate(atlas.sizes):
        if i % 2 == 1:
            ax.axvspan(pos, pos + size, color="#888888", alpha=0.07, zorder=0)
        pos += size
    for boundary in atlas.boundaries():
        ax.axvline(boundary - 0.5, color="#999999", lw=0.6, ls="--",
                   alpha=0.7, zorder=0)
    ticks = np.cumsum(atlas.sizes) - np.array(atlas.sizes) / 2
    ax.set_xticks(ticks)
    ax.set_xticklabels([atlas.name(i) for i in range(atlas.n_net)], fontsize=9)
    ax.set_xlim(0, atlas.n_roi - 1)


def plot_flexibility_profile(values, atlas: Atlas,
                             title: str = "Whole-brain DCC profile",
                             label: str = "DCC",
                             ylabel: str = "Entropy (bits)") -> plt.Figure:
    """Plot one value per ROI with vertical network boundaries."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    v = np.asarray(values, dtype=float).ravel()
    _net_bands(ax, atlas)
    for i, size in enumerate(atlas.sizes):
        start = int(np.sum(atlas.sizes[:i]))
        ax.plot(np.arange(start, start + size), v[start:start + size],
                color=_color(i), lw=1.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("ROI (atlas order)")
    fig.tight_layout()
    return fig


def plot_network_bars(net_means, atlas: Atlas,
                      title: str = "Mean DCC per network",
                      ylabel: str = "Mean DCC") -> plt.Figure:
    """Bar chart of one scalar value per network."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    values = np.asarray(net_means, dtype=float).ravel()
    names = [atlas.name(i) for i in range(atlas.n_net)]
    bars = ax.bar(names, values, color=[_color(i) for i in range(len(values))],
                  edgecolor="white", width=0.65)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="#555555", lw=0.8)
    fig.tight_layout()
    return fig


def plot_state_heatmap(prob, title: str = "State probability",
                       xlabel: str = "State code",
                       ylabel: str = "ROI / network",
                       aspect="auto") -> plt.Figure:
    """Heat map of state probabilities across ROIs (or networks) and states."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    image = ax.imshow(np.asarray(prob, dtype=float), cmap=CMAP,
                      aspect=aspect, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label="Probability")
    fig.tight_layout()
    return fig


def plot_matrix(mat, title: str = "Matrix heatmap", cmap: str = "RdBu_r",
                xlabel: str = "", ylabel: str = "") -> plt.Figure:
    """Render a square numeric matrix such as between-sample similarity."""
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(np.asarray(mat, dtype=float), cmap=cmap, aspect="auto",
                      vmin=-1 if cmap == "RdBu_r" else None,
                      vmax=1 if cmap == "RdBu_r" else None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_icc(icc, atlas: Atlas,
             title: str = "DCC test-retest reliability ICC(1,1)") -> plt.Figure:
    """Plot ICC(1,1) per ROI using the shared network layout."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    values = np.nan_to_num(np.asarray(icc, dtype=float))
    means = [np.mean(values[atlas.roi_idx == i + 1])
             for i in range(atlas.n_net)]
    ax.bar([atlas.name(i) for i in range(atlas.n_net)], means,
           color=[_color(i) for i in range(atlas.n_net)], edgecolor="white")
    ax.axhline(np.mean(values), color="#444444", ls="--", lw=1,
               label=f"Whole-brain mean {np.mean(values):.3f}")
    ax.set_ylim(min(0, values.min() - 0.05), max(0.2, values.max() + 0.05))
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cohen_d(d, atlas: Atlas, thr: float = 0.3,
                 title: str = "Group difference: Cohen's d") -> plt.Figure:
    """Plot Cohen's d per ROI and mark the effect-size threshold."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    values = np.asarray(d, dtype=float).ravel()
    _net_bands(ax, atlas)
    colors = np.where(values > thr, "#c0392b",
                      np.where(values < -thr, "#2c6fbb", "#b9c2cc"))
    ax.bar(np.arange(atlas.n_roi), values, color=colors, width=1.0)
    ax.axhline(thr, color="#c0392b", ls="--", lw=0.9)
    ax.axhline(-thr, color="#2c6fbb", ls="--", lw=0.9)
    ax.set_title(f"{title} (dashed: |d| = {thr})")
    ax.set_ylabel("Cohen's d")
    ax.set_xlabel("ROI (atlas order)")
    fig.tight_layout()
    return fig


def plot_group_profiles(groups: dict, atlas: Atlas,
                        title: str = "DCC by group",
                        ylabel: str = "Entropy (bits)") -> plt.Figure:
    """Overlay one per-ROI profile for each experimental group."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    _net_bands(ax, atlas)
    for i, (name, values) in enumerate(groups.items()):
        ax.plot(np.arange(atlas.n_roi), np.asarray(values).ravel(),
                lw=1.1, color=_color(i), label=name, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("ROI (atlas order)")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def plot_network_grouped_bars(table: pd.DataFrame, value_col: str,
                              title: str, ylabel: str = "") -> plt.Figure:
    """Grouped bar chart comparing several values within each network."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    cols = [c for c in table.columns if c not in ("network", "n_roi")]
    width = 0.8 / max(1, len(cols))
    x = np.arange(len(table))
    for i, col in enumerate(cols):
        ax.bar(x + (i - len(cols) / 2 + 0.5) * width, table[col].values,
               width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(table["network"])
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel or value_col)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_explained_covariance(expl, n_show: int = 10,
                              title: str = "PLS explained covariance") -> plt.Figure:
    """Bar chart of covariance explained by each PLS component."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    values = np.asarray(expl, dtype=float).ravel()[:n_show]
    ax.bar([f"LC{i + 1}" for i in range(len(values))], values * 100,
           color="#3f7fc1", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Explained covariance (%)")
    fig.tight_layout()
    return fig


def loadings_bar(loadings, names, lc: int, title: str,
                 max_items: int = 30, horizontal: bool = True) -> plt.Figure:
    """Horizontal bar chart of behaviour loadings, optionally keeping the largest items."""
    values = np.asarray(loadings, dtype=float).ravel()
    names = list(names)
    if len(values) > max_items:
        idx = np.argsort(-np.abs(values))[:max_items]
        values, names = values[idx], [names[i] for i in idx]
    colors = ["#c0392b" if x > 0 else "#2c6fbb" for x in values]
    if horizontal:
        fig, ax = plt.subplots(figsize=(8, max(3, len(values) * 0.24)))
        ax.barh(names, values, color=colors, edgecolor="white", height=0.75)
        ax.axvline(0, color="#444444", lw=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Loading (correlation with composite score)")
    else:
        fig, ax = plt.subplots(figsize=(max(7, len(values) * 0.2), 3.4))
        ax.bar(names, values, color=colors, edgecolor="white")
        ax.axhline(0, color="#444444", lw=0.8)
        ax.set_ylabel("Loading")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_state_patterns(table: pd.DataFrame, atlas: Atlas,
                        title: str = "High-frequency state patterns") -> plt.Figure:
    """Visualise the binary network pattern of the most frequent states."""
    fig, ax = plt.subplots(figsize=(8.5, max(2.6, len(table) * 0.32 + 1)))
    if len(table) == 0:
        ax.text(0.5, 0.5, "No states above the threshold",
                ha="center", va="center")
        ax.axis("off")
        return fig
    mat = np.array([[int(ch) for ch in p] for p in table["binary_pattern"]])
    ax.imshow(mat,
              cmap=LinearSegmentedColormap.from_list("bw", ["#f4f6f8", "#2c3e50"]),
              aspect="auto", interpolation="nearest")
    ax.set_xticks(range(atlas.n_net))
    ax.set_xticklabels([atlas.name(i) for i in range(atlas.n_net)])
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([f"{int(c)}  ({p})" for c, p in
                        zip(table["state_code"], table["binary_pattern"])],
                       fontsize=8)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1],
                 label="Disconnected / connected")
    fig.tight_layout()
    return fig


def plot_top_coverage(coverage: dict,
                      title: str = "Top-state coverage") -> plt.Figure:
    """Plot the fraction of state occurrences covered by the top states."""
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    values = np.asarray(coverage["top_probabilities"], dtype=float)
    ax.bar(range(1, len(values) + 1), values * 100,
           color="#27836b", edgecolor="white")
    ax.set_xlabel("State (descending mean probability)")
    ax.set_ylabel("Mean probability (%)")
    ax.set_title(f"{title}: {coverage['top_coverage'] * 100:.1f}%")
    fig.tight_layout()
    return fig
