"""Interpretable cluster labels: top TF-IDF terms and the most central claim.

Labels are extractive only (words and sentences that accounts actually wrote). Nothing is
generated, so a label can never introduce a narrative of its own.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def top_terms_per_cluster(texts: Sequence[str], labels: np.ndarray, top_n: int = 5) -> dict[int, list[str]]:
    """Class-based TF-IDF: each cluster is one document; return its highest-weighted terms."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids = sorted(int(c) for c in set(labels) if c >= 0)
    if not ids:
        return {}
    docs = [" ".join(t for t, l in zip(texts, labels) if l == c) for c in ids]
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                              token_pattern=r"(?u)\b[A-Za-z0-9][A-Za-z0-9'\-]+\b")
        matrix = vec.fit_transform(docs)
    except ValueError:  # empty vocabulary (e.g. only stop words)
        return {c: [] for c in ids}
    vocab = np.array(vec.get_feature_names_out())
    result: dict[int, list[str]] = {}
    for row, c in enumerate(ids):
        weights = matrix[row].toarray().ravel()
        chosen: list[str] = []
        for idx in np.argsort(-weights):
            if weights[idx] <= 0 or len(chosen) >= top_n:
                break
            term = vocab[idx]
            # skip a unigram already covered by a chosen bigram and vice versa
            if any(term in t.split() or t in term.split() for t in chosen):
                continue
            chosen.append(term)
        result[c] = chosen
    return result


def central_claim_per_cluster(emb: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """Index of the claim closest to each cluster centroid (the cluster's medoid-like claim)."""
    result: dict[int, int] = {}
    for c in sorted(int(c) for c in set(labels) if c >= 0):
        members = np.where(labels == c)[0]
        centroid = emb[members].mean(axis=0)
        result[c] = int(members[np.argmax(emb[members] @ centroid)])
    return result
