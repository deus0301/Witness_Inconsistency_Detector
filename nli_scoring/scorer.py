"""Pairwise NLI scoring: entailment / neutral / contradiction probabilities.

``NLIScorer`` wraps any HuggingFace sequence-classification NLI model. The label order is
read from the model's ``config.id2label`` (it differs between models), scores are batched,
run on GPU when available (CPU fallback) and cached on disk in SQLite.

``MockNLIScorer`` is a deterministic lexical stand-in with the same interface, used by the
unit tests and by ``--mock`` runs when the real model cannot be downloaded.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data.schema import NLI_LABELS
from pipeline.config import (FINETUNE_META_FILE, pick_device, resolve_model, resolve_path, slug,
                             text_hash)

E, N, C = 0, 1, 2  # column order of every probability array in this project


@dataclass(frozen=True)
class PairScore:
    """Symmetric score of an unordered pair of statements."""

    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        return NLI_LABELS[int(np.argmax([self.entailment, self.neutral, self.contradiction]))]

    @property
    def confidence(self) -> float:
        return max(self.entailment, self.neutral, self.contradiction)

    def to_dict(self) -> dict[str, float | str]:
        return {"entailment": self.entailment, "neutral": self.neutral,
                "contradiction": self.contradiction, "label": self.label,
                "confidence": self.confidence}


def label_permutation(id2label: dict) -> list[int]:
    """Indices that reorder a model's logits into (entailment, neutral, contradiction).

    Raises ``ValueError`` if the model config does not name the three NLI classes.
    """
    names = {int(i): str(name).lower() for i, name in id2label.items()}
    order: list[int] = []
    for wanted in NLI_LABELS:
        matches = [i for i, name in names.items() if name.startswith(wanted[:6])]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot find a unique '{wanted}' class in model id2label={id2label}. "
                "The model config must name its NLI labels."
            )
        order.append(matches[0])
    return order


def aggregate_directions(ab: np.ndarray, ba: np.ndarray, how: str = "max_contradiction") -> np.ndarray:
    """Combine premise->hypothesis scores of both directions into one row per pair.

    ``max_contradiction`` keeps, for each pair, the direction with the higher contradiction
    probability (so the result is still a valid distribution and order-independent).
    ``min_contradiction`` keeps the direction with the LOWER contradiction probability, i.e.
    a pair only counts as contradicting when both directions say so (contradiction is a
    symmetric relation, so this is the stricter, higher-precision reading).
    ``mean`` averages the two directions.
    """
    if how == "mean":
        return (ab + ba) / 2.0
    if how == "min_contradiction":
        take_ba = ba[:, C] < ab[:, C]
        return np.where(take_ba[:, None], ba, ab)
    if how == "max_contradiction":
        take_ba = ba[:, C] > ab[:, C]
        return np.where(take_ba[:, None], ba, ab)
    raise ValueError(f"Unknown aggregation '{how}'")


class PairCache:
    """Disk cache of directional NLI probabilities (SQLite, one file per model).

    The connection may be used from several threads (Streamlit reruns the script in a new
    thread each time while keeping the scorer alive), so access is serialised with a lock.
    """

    def __init__(self, path: str | Path | None) -> None:
        self.conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path), check_same_thread=False)
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, e REAL, n REAL, c REAL)")
            self.conn.commit()

    def get_many(self, keys: Sequence[str]) -> dict[str, tuple[float, float, float]]:
        if self.conn is None or not keys:
            return {}
        found: dict[str, tuple[float, float, float]] = {}
        unique = list(set(keys))
        with self._lock:
            for i in range(0, len(unique), 500):
                chunk = unique[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for key, e, n, c in self.conn.execute(
                        f"SELECT key, e, n, c FROM scores WHERE key IN ({marks})", chunk).fetchall():
                    found[key] = (e, n, c)
        return found

    def put_many(self, rows: Sequence[tuple[str, float, float, float]]) -> None:
        if self.conn is None or not rows:
            return
        with self._lock:
            self.conn.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)", rows)
            self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class BaseScorer:
    """Shared symmetric-scoring and caching logic; subclasses implement ``_predict``."""

    name = "base"

    def __init__(self, cache_path: str | Path | None = None,
                 aggregation: str = "max_contradiction") -> None:
        self.cache = PairCache(cache_path)
        self.aggregation = aggregation
        self.n_model_calls = 0
        self.n_cache_hits = 0

    def _predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def score_directional(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """(premise, hypothesis) pairs -> array [n, 3] in (entailment, neutral, contradiction) order."""
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        keys = [text_hash(a, b) for a, b in pairs]
        cached = self.cache.get_many(keys)
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        todo: dict[str, tuple[str, str]] = {}
        for key, pair in zip(keys, pairs):
            if key not in cached:
                todo.setdefault(key, pair)
        self.n_cache_hits += len(pairs) - len(todo)
        if todo:
            todo_keys = list(todo)
            probs = self._predict([todo[k] for k in todo_keys])
            self.n_model_calls += len(todo_keys)
            self.cache.put_many([(k, float(p[E]), float(p[N]), float(p[C]))
                                 for k, p in zip(todo_keys, probs)])
            cached.update({k: tuple(map(float, p)) for k, p in zip(todo_keys, probs)})
        for i, key in enumerate(keys):
            out[i] = cached[key]
        return out

    def score_pairs(self, pairs: Sequence[tuple[str, str]], symmetric: bool = True) -> np.ndarray:
        """Score pairs; with ``symmetric`` both directions are scored and aggregated."""
        pairs = list(pairs)
        if not symmetric:
            return self.score_directional(pairs)
        both = self.score_directional(pairs + [(b, a) for a, b in pairs])
        n = len(pairs)
        return aggregate_directions(both[:n], both[n:], self.aggregation)

    def score_pair(self, a: str, b: str) -> dict[str, float | str]:
        """Symmetric score of one pair as a dict with probabilities, label and confidence."""
        e, n, c = self.score_pairs([(a, b)])[0]
        return PairScore(float(e), float(n), float(c)).to_dict()


class NLIScorer(BaseScorer):
    """HuggingFace NLI cross-encoder (default ``cross-encoder/nli-deberta-v3-base``)."""

    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 128,
                 device: str | None = None, fp16: bool = True,
                 cache_path: str | Path | None = None,
                 aggregation: str = "max_contradiction", verbose: bool = True,
                 model=None, tokenizer=None) -> None:
        super().__init__(cache_path, aggregation)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or pick_device()
        self.fp16 = fp16 and self.device == "cuda"
        if model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
        else:  # an already loaded model (used during fine-tuning)
            self.tokenizer, self.model = tokenizer, model
        self.model.eval()
        self.id2label = dict(self.model.config.id2label)
        self.permutation = label_permutation(self.id2label)
        if verbose:
            where = (f"GPU ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else "CPU")
            print(f"[nli] {model_name} on {where}; model label order = "
                  f"{[self.id2label[i] for i in sorted(self.id2label)]}", flush=True)

    def _predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        torch = self.torch
        order = np.argsort([len(a) + len(b) for a, b in pairs])  # length-bucketed batches
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            idx = order[start:start + self.batch_size]
            batch = self.tokenizer(
                [pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                if self.fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = self.model(**batch).logits
                else:
                    logits = self.model(**batch).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            out[idx] = probs[:, self.permutation]
        return out


_NEGATIONS = {"not", "no", "never", "n't", "false", "fake", "hoax", "denies", "deny",
              "denied", "untrue", "incorrect", "wrong", "isn't", "wasn't", "didn't", "aren't"}
_TOKEN = re.compile(r"[a-z0-9']+")


class MockNLIScorer(BaseScorer):
    """Deterministic lexical heuristic with the scorer interface (NOT a real NLI model).

    High token overlap -> entailment; overlap plus a negation or number mismatch ->
    contradiction; little overlap -> neutral.
    """

    name = "mock-nli"

    def _predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        for i, (a, b) in enumerate(pairs):
            ta, tb = set(_TOKEN.findall(a.lower())), set(_TOKEN.findall(b.lower()))
            overlap = len(ta & tb) / max(1, len(ta | tb))
            neg_mismatch = bool(ta & _NEGATIONS) != bool(tb & _NEGATIONS)
            na, nb = {t for t in ta if t.isdigit()}, {t for t in tb if t.isdigit()}
            num_mismatch = bool(na and nb and na != nb)
            if overlap >= 0.2 and (neg_mismatch or num_mismatch):
                out[i] = (0.05, 0.10, 0.85)
            elif overlap >= 0.5:
                out[i] = (0.85, 0.10, 0.05)
            else:
                out[i] = (0.10, 0.80, 0.10)
        return out


def build_scorer(cfg: dict, model_key: str = "nli", use_cache: bool = True) -> BaseScorer:
    """Create the scorer named by ``cfg['models'][model_key]`` (or the mock)."""
    cache_dir = resolve_path(cfg, "cache_dir")
    agg = cfg["nli"]["aggregation"]
    if cfg["models"].get("mock"):
        print("[nli] using MockNLIScorer (lexical heuristic, not a real NLI model)", flush=True)
        return MockNLIScorer(cache_dir / "nli_mock.sqlite" if use_cache else None, agg)
    name = cfg["models"][model_key]
    return NLIScorer(
        resolve_model(name, cfg), batch_size=cfg["nli"]["batch_size"],
        max_length=cfg["nli"]["max_length"], fp16=cfg["nli"]["fp16"], device=pick_device(cfg),
        cache_path=cache_dir / cache_file_name(name, cfg) if use_cache else None,
        aggregation=agg,
    )


def cache_file_name(name: str, cfg: dict) -> str:
    """SQLite cache file of a model. A fine-tuned checkpoint also carries its training run id,
    so retraining into the same folder never reuses old scores."""
    run_id = ""
    meta = Path(resolve_model(name, cfg)) / FINETUNE_META_FILE
    if meta.is_file():
        import json

        run_id = "_" + str(json.loads(meta.read_text(encoding="utf-8")).get("run_id", ""))
    return f"nli_{slug(name)}{run_id}.sqlite"
