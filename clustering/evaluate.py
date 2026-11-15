"""Clustering quality metrics: intrinsic, plus a weak extrinsic check against PHEME stories."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def intrinsic_metrics(emb: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    """Silhouette (cosine) and Davies-Bouldin over clustered points; noise is excluded."""
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    labels = np.asarray(labels)
    mask = labels >= 0
    n_clusters = len(set(labels[mask]))
    sizes = np.bincount(labels[mask]) if mask.any() else np.array([])
    out: dict[str, float | int | None] = {
        "n_claims": int(labels.size),
        "n_clusters": int(n_clusters),
        "noise_fraction": float(1.0 - mask.mean()) if labels.size else 0.0,
        "median_cluster_size": float(np.median(sizes)) if sizes.size else 0.0,
        "max_cluster_size": int(sizes.max()) if sizes.size else 0,
        "silhouette": None,
        "davies_bouldin": None,
    }
    if 2 <= n_clusters < mask.sum():
        out["silhouette"] = float(silhouette_score(emb[mask], labels[mask], metric="cosine"))
        out["davies_bouldin"] = float(davies_bouldin_score(emb[mask], labels[mask]))
    return out


def extrinsic_metrics(labels: np.ndarray, reference: Sequence[str | None]) -> dict[str, float | int | None]:
    """ARI / NMI / homogeneity / completeness against a reference grouping.

    Only claims with a reference (rumour story) are scored. Noise points count as
    singleton clusters, so leaving everything unclustered is not rewarded.
    """
    from sklearn.metrics import (adjusted_rand_score, completeness_score,
                                 homogeneity_score, normalized_mutual_info_score)

    labels = np.asarray(labels).copy()
    idx = [i for i, r in enumerate(reference) if r is not None]
    if len(idx) < 2:
        return {"n_scored": len(idx), "ari": None, "nmi": None,
                "homogeneity": None, "completeness": None}
    next_id = labels.max() + 1 if labels.size else 0
    for i in np.where(labels < 0)[0]:
        labels[i] = next_id
        next_id += 1
    pred = labels[idx]
    ref = [reference[i] for i in idx]
    return {
        "n_scored": len(idx),
        "ari": float(adjusted_rand_score(ref, pred)),
        "nmi": float(normalized_mutual_info_score(ref, pred)),
        "homogeneity": float(homogeneity_score(ref, pred)),
        "completeness": float(completeness_score(ref, pred)),
    }
