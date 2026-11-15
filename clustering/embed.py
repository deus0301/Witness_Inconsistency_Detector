"""Sentence embeddings for claims (sentence-transformers) with an on-disk cache."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from pipeline.config import pick_device, resolve_model, resolve_path, slug, text_hash


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class Embedder:
    """L2-normalised sentence embeddings from a sentence-transformers model."""

    def __init__(self, model_name: str, cache_dir: str | Path | None = None,
                 batch_size: int = 64, display_name: str | None = None,
                 device: str | None = None, verbose: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = display_name or model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = SentenceTransformer(model_name, device=device or pick_device())
        if verbose:
            print(f"[embed] {self.name} on {self.model.device}", flush=True)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(
            list(texts), batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` -> array [n, dim], unit norm. Cached per (model, exact text list)."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        cache_file = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"emb_{slug(self.name)}_{text_hash(*texts)[:16]}.npy"
            if cache_file.exists():
                return np.load(cache_file)
        emb = self._encode(texts)
        if cache_file is not None:
            np.save(cache_file, emb)
        return emb


class MockEmbedder:
    """Hashed bag-of-words embeddings (no model download). For tests and ``--mock`` runs."""

    name = "mock-embedder"

    def __init__(self, dim: int = 256) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(n_features=dim, alternate_sign=False,
                                            norm=None, stop_words="english")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        dense = self.vectorizer.transform(texts).toarray().astype(np.float32)
        dense += 1e-6  # avoid all-zero rows for texts made only of stop words
        return _normalise(dense)


def build_embedder(cfg: dict, model_key: str = "embedding"):
    """Create the embedder named by ``cfg['models'][model_key]`` (or the mock)."""
    if cfg["models"].get("mock"):
        print("[embed] using MockEmbedder (hashed bag of words, not a real model)", flush=True)
        return MockEmbedder()
    name = cfg["models"][model_key]
    return Embedder(resolve_model(name, cfg), cache_dir=resolve_path(cfg, "cache_dir"),
                    display_name=name, device=pick_device(cfg))
