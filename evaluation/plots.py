"""Static figures for the report (matplotlib, light theme, written as PNG).

Colour is assigned by job: categorical series colours in a fixed order, one blue ramp for
magnitudes (heatmaps, confusion matrices), grey for unclustered points.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _short(text: str, n: int = 46) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def plot_confusion(matrix: Sequence[Sequence[int]], labels: Sequence[str], title: str,
                   path: Path, xlabel: str = "Predicted", ylabel: str = "Reference") -> Path:
    """Row-normalised confusion matrix with counts in the cells."""
    m = np.asarray(matrix, dtype=float)
    norm = m / np.clip(m.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{int(m[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color="#ffffff" if norm[i, j] > 0.55 else INK)
    return _save(fig, path)


def plot_cluster_scatter(emb: np.ndarray, labels: np.ndarray, title: str, path: Path,
                         cluster_names: dict[int, str] | None = None, seed: int = 42,
                         flagged: set[int] | None = None) -> Path:
    """2-D projection (UMAP, t-SNE fallback) of claim embeddings.

    The 8 largest clusters get the categorical colours and a legend entry; all other
    clustered claims are one neutral "other clusters" colour, unclustered claims are light
    grey. Flagged clusters are ringed.
    """
    n = emb.shape[0]
    if n < 5:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "too few claims to project", ha="center")
        return _save(fig, path)
    try:
        import warnings

        import umap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xy = umap.UMAP(n_components=2, n_neighbors=min(15, n - 1), min_dist=0.15,
                           metric="cosine", random_state=seed).fit_transform(emb)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE

        xy = TSNE(n_components=2, metric="cosine", init="random", random_state=seed,
                  perplexity=min(30, max(2, n // 4))).fit_transform(emb)
        method = "t-SNE"
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    sizes = {c: int((labels == c).sum()) for c in set(labels) if c >= 0}
    top = [c for c, _ in sorted(sizes.items(), key=lambda kv: -kv[1])[: len(SERIES)]]
    noise = labels < 0
    other = ~noise & ~np.isin(labels, top)
    ax.scatter(xy[noise, 0], xy[noise, 1], s=10, c="#d8d7d0", linewidths=0,
               label=f"unclustered ({int(noise.sum())})")
    ax.scatter(xy[other, 0], xy[other, 1], s=14, c="#a9a8a1", linewidths=0,
               label=f"other clusters ({len(sizes) - len(top)})")
    for colour, c in zip(SERIES, top):
        mask = labels == c
        name = _short((cluster_names or {}).get(c, f"cluster {c}"), 38)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=22, c=colour, edgecolors=SURFACE, linewidths=0.5,
                   label=f"{name} ({sizes[c]})")
    if flagged:
        mask = np.isin(labels, list(flagged))
        ax.scatter(xy[mask, 0], xy[mask, 1], s=70, facecolors="none", edgecolors=INK,
                   linewidths=0.7, label="in a flagged cluster")
    ax.set_title(title)
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    return _save(fig, path)


def plot_contradiction_heatmap(texts: Sequence[str], matrix: np.ndarray, title: str, path: Path) -> Path:
    """Claim x claim heatmap of P(contradiction) inside one cluster (NaN = not scored)."""
    n = len(texts)
    fig, ax = plt.subplots(figsize=(min(11, 4.2 + 0.32 * n), min(9, 2.4 + 0.32 * n)))
    cmap = BLUES.copy()
    cmap.set_bad("#eeeeea")
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), [str(i + 1) for i in range(n)], fontsize=7)
    ax.set_yticks(range(n), [f"{i + 1}. {_short(t, 58)}" for i, t in enumerate(texts)], fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("P(contradiction)")
    cbar.outline.set_visible(False)
    return _save(fig, path)


def plot_contradiction_graph(texts: Sequence[str], accounts: Sequence[str],
                             edges: Sequence[tuple[int, int, float]], title: str, path: Path,
                             seed: int = 42) -> Path:
    """Graph of one cluster: nodes = claims, edges = pairs at/above the contradiction threshold."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(len(texts)))
    graph.add_weighted_edges_from(edges)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(False)
    ax.set_axis_off()
    pos = nx.spring_layout(graph, seed=seed, k=1.6 / max(1, np.sqrt(len(texts))))
    involved = {i for e in edges for i in e[:2]}
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[i for i in graph if i not in involved],
                           node_color="#d8d7d0", node_size=60, linewidths=0)
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=sorted(involved), node_color=SERIES[0],
                           node_size=170, edgecolors=SURFACE, linewidths=1.5)
    widths = [0.8 + 3.0 * (w - 0.5) for _, _, w in edges]
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[(a, b) for a, b, _ in edges],
                           width=widths, edge_color=SERIES[7], alpha=0.65)
    degree = dict(graph.degree())
    for i in sorted(involved, key=lambda i: -degree[i])[:8]:  # label the most contested claims
        ax.annotate(f"@{accounts[i]}: {_short(texts[i], 60)}", pos[i], xytext=(6, 6),
                    textcoords="offset points", fontsize=7, color=INK2)
    ax.set_title(title)
    ax.plot([], [], color=SERIES[7], lw=2, label="pair scored as contradicting")
    ax.scatter([], [], c=SERIES[0], s=60, label="claim in a contradicting pair")
    ax.scatter([], [], c="#d8d7d0", s=40, label="other claim in the cluster")
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def plot_threshold_curves(thresholds: Sequence[float], curves: dict[str, dict[str, Sequence[float]]],
                          title: str, path: Path) -> Path:
    """Small multiples (precision / recall / F1) of cluster-level scores vs threshold."""
    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, metric in zip(axes, metrics):
        for colour, (name, values) in zip(SERIES, curves.items()):
            ax.plot(thresholds, values[metric], color=colour, lw=2, marker="o", ms=4, label=name)
        ax.set_title(metric.capitalize() if metric != "f1" else "F1")
        ax.set_xlabel("contradiction threshold")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("cluster-level score vs proxy")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle(title, x=0.01, y=1.06, ha="left", fontweight="bold")
    return _save(fig, path)


def plot_bars(categories: Sequence[str], series: dict[str, Sequence[float]], title: str,
              ylabel: str, path: Path, log: bool = False, fmt: str = "{:.2f}") -> Path:
    """Grouped bar chart with value labels (one colour per series, fixed order)."""
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(categories) + 2), 4))
    for k, (colour, (name, values)) in enumerate(zip(SERIES, series.items())):
        pos = x + (k - (n_series - 1) / 2) * width
        ax.bar(pos, values, width * 0.9, color=colour, label=name)
        for px, v in zip(pos, values):
            if v is not None and np.isfinite(v):
                ax.text(px, v, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
    if n_series > 1:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    return _save(fig, path)
