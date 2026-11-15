"""Unsupervised sub-event clustering of claim embeddings.

Four interchangeable algorithms behind one function, ``cluster_embeddings``:

* ``hdbscan``        density-based, finds the number of clusters itself, emits noise (-1)
* ``kmeans``         k chosen by silhouette over a grid
* ``agglomerative``  average-linkage on cosine distance, cut at a distance threshold
* ``graph``          kNN similarity graph + Louvain communities (networkx)

All return an int label per claim; ``-1`` means "unclustered" (noise). Unclustered claims
are never paired for NLI, so they cannot create false contradictions.
"""
from __future__ import annotations

import warnings
from collections import Counter

import numpy as np

ALGORITHMS = ("hdbscan", "kmeans", "agglomerative", "graph")


def _relabel(labels: np.ndarray, min_size: int = 2) -> np.ndarray:
    """Make labels contiguous (0..k-1, by decreasing size); clusters < ``min_size`` -> -1."""
    labels = np.asarray(labels).copy()
    counts = Counter(int(l) for l in labels if l >= 0)
    keep = [l for l, n in counts.most_common() if n >= min_size]
    mapping = {old: new for new, old in enumerate(keep)}
    return np.array([mapping.get(int(l), -1) for l in labels], dtype=int)


def _hdbscan(emb: np.ndarray, params: dict, seed: int) -> np.ndarray:
    x = emb
    umap_dim = params.get("umap_dim")
    if umap_dim and emb.shape[0] > umap_dim + 2:
        import umap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = umap.UMAP(n_components=umap_dim, n_neighbors=min(15, emb.shape[0] - 1),
                          min_dist=0.0, metric="cosine", random_state=seed).fit_transform(emb)
    kwargs = dict(min_cluster_size=params.get("min_cluster_size", 3),
                  min_samples=params.get("min_samples", 2))
    try:
        import hdbscan

        return hdbscan.HDBSCAN(metric="euclidean", **kwargs).fit_predict(x)
    except ImportError:  # scikit-learn >= 1.3 ships its own implementation
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(metric="euclidean", **kwargs).fit_predict(x)


def _kmeans(emb: np.ndarray, params: dict, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = emb.shape[0]
    if params.get("k"):
        grid = [int(params["k"])]
    else:
        grid = list(range(params.get("k_min", 5), params.get("k_max", 60) + 1, params.get("k_step", 5)))
    grid = [k for k in grid if 2 <= k < n] or [max(2, min(n - 1, 2))]
    best_labels, best_score = None, -2.0
    for k in grid:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)
        score = silhouette_score(emb, labels, metric="cosine") if len(set(labels)) > 1 else -1.0
        if score > best_score:
            best_labels, best_score = labels, score
    return best_labels


def _agglomerative(emb: np.ndarray, params: dict, seed: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage=params.get("linkage", "average"),
        distance_threshold=params.get("distance_threshold", 0.45)).fit_predict(emb)


def _graph(emb: np.ndarray, params: dict, seed: int) -> np.ndarray:
    import networkx as nx

    n = emb.shape[0]
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    k = min(params.get("knn", 10), n - 1)
    min_sim = params.get("min_similarity", 0.55)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    for i in range(n):
        for j in top[i]:
            if sim[i, j] >= min_sim:
                graph.add_edge(i, int(j), weight=float(sim[i, j]))
    labels = np.full(n, -1, dtype=int)
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=params.get("resolution", 1.0), seed=seed)
    for cid, members in enumerate(communities):
        if len(members) > 1:
            labels[list(members)] = cid
    return labels


def assign_noise(emb: np.ndarray, labels: np.ndarray, min_sim: float) -> np.ndarray:
    """Attach each noise point to its nearest cluster centroid if cosine sim >= ``min_sim``."""
    labels = labels.copy()
    ids = sorted(set(labels[labels >= 0]))
    noise = np.where(labels < 0)[0]
    if not ids or noise.size == 0:
        return labels
    centroids = np.stack([emb[labels == c].mean(axis=0) for c in ids])
    centroids /= np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
    sims = emb[noise] @ centroids.T
    best = sims.argmax(axis=1)
    for row, point in enumerate(noise):
        if sims[row, best[row]] >= min_sim:
            labels[point] = ids[best[row]]
    return labels


def cluster_embeddings(emb: np.ndarray, algorithm: str = "agglomerative",
                       params: dict | None = None, noise_strategy: str = "nearest",
                       noise_assign_min_sim: float = 0.6, seed: int = 42) -> np.ndarray:
    """Cluster unit-norm embeddings. Returns labels in 0..k-1, ``-1`` = unclustered.

    Args:
        emb: array [n, dim] of L2-normalised embeddings.
        algorithm: one of ``ALGORITHMS``.
        params: algorithm-specific parameters (see ``configs/default.yaml``).
        noise_strategy: ``nearest`` re-attaches noise points that are close enough to a
            cluster centroid; ``keep`` leaves them unclustered.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
    n = emb.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n < 3:
        return np.full(n, -1, dtype=int) if n == 1 else np.zeros(n, dtype=int)
    func = {"hdbscan": _hdbscan, "kmeans": _kmeans,
            "agglomerative": _agglomerative, "graph": _graph}[algorithm]
    labels = _relabel(func(emb, params or {}, seed))
    if noise_strategy == "nearest":
        labels = assign_noise(emb, labels, noise_assign_min_sim)
    elif noise_strategy != "keep":
        raise ValueError(f"noise_strategy must be 'nearest' or 'keep', got {noise_strategy!r}")
    return labels


def cluster_from_config(emb: np.ndarray, cfg: dict, algorithm: str | None = None) -> np.ndarray:
    """``cluster_embeddings`` with parameters taken from ``cfg['clustering']``."""
    c = cfg["clustering"]
    algo = algorithm or c["algorithm"]
    return cluster_embeddings(emb, algo, c.get(algo, {}), c["noise_strategy"],
                              c["noise_assign_min_sim"], cfg["seed"])
