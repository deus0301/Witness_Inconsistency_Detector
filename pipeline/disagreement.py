"""In-cluster pair generation and the cluster-level disagreement score."""
from __future__ import annotations

import random
from itertools import combinations
from typing import Sequence

import numpy as np

from data.schema import Claim

C = 2  # contradiction column


def cluster_pairs(members: Sequence[int], claims: Sequence[Claim], max_pairs: int | None,
                  cross_account_only: bool = True, seed: int = 42) -> list[tuple[int, int]]:
    """All (i, j) index pairs inside one cluster, optionally cross-account only.

    If there are more than ``max_pairs`` candidates a seeded random sample is returned,
    which bounds NLI cost for very large clusters.
    """
    pairs = [
        (i, j) for i, j in combinations(sorted(members), 2)
        if not cross_account_only or claims[i].account_id != claims[j].account_id
    ]
    if max_pairs and len(pairs) > max_pairs:
        pairs = sorted(random.Random(seed).sample(pairs, max_pairs))
    return pairs


def disagreement_stats(probs: np.ndarray, threshold: float, weight_fraction: float = 0.5,
                       min_pairs: int = 1) -> dict[str, float | int | bool]:
    """Summarise the NLI scores of one cluster's pairs.

    ``disagreement_score = w * fraction_contradicting + (1 - w) * max_contradiction`` where
    a pair "contradicts" when its contradiction probability is >= ``threshold``. The
    cluster is flagged when at least ``min_pairs`` pairs contradict.
    """
    n = int(probs.shape[0])
    if n == 0:
        return {"n_pairs_scored": 0, "n_contradicting_pairs": 0, "contradiction_fraction": 0.0,
                "max_contradiction": 0.0, "mean_contradiction": 0.0,
                "disagreement_score": 0.0, "flagged": False}
    contra = probs[:, C]
    n_contra = int((contra >= threshold).sum())
    fraction = n_contra / n
    max_c = float(contra.max())
    return {
        "n_pairs_scored": n,
        "n_contradicting_pairs": n_contra,
        "contradiction_fraction": float(fraction),
        "max_contradiction": max_c,
        "mean_contradiction": float(contra.mean()),
        "disagreement_score": float(weight_fraction * fraction + (1 - weight_fraction) * max_c),
        "flagged": bool(n_contra >= min_pairs),
    }
