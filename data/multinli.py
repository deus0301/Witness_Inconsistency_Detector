"""MultiNLI loader (local ``multinli_1.0`` jsonl files). The only NLI dataset in the project."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from data.schema import NLI_LABELS

SPLIT_FILES = {
    "train": "multinli_1.0_train.jsonl",
    "dev_matched": "multinli_1.0_dev_matched.jsonl",
    "dev_mismatched": "multinli_1.0_dev_mismatched.jsonl",
}


@dataclass(frozen=True)
class NLIExample:
    """One premise/hypothesis pair with its gold label."""

    premise: str
    hypothesis: str
    label: str
    genre: str
    pair_id: str


def load_multinli(root: str | Path, split: str = "dev_matched",
                  max_examples: int | None = None, seed: int = 42) -> list[NLIExample]:
    """Read one MultiNLI split. Examples without annotator consensus (``-``) are dropped.

    ``max_examples`` takes a seeded random subsample (used by ``--sample`` mode).
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}, got {split!r}")
    path = Path(root) / SPLIT_FILES[split]
    examples: list[NLIExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("gold_label") not in NLI_LABELS:
                continue
            examples.append(NLIExample(
                premise=row["sentence1"], hypothesis=row["sentence2"], label=row["gold_label"],
                genre=row.get("genre", ""), pair_id=str(row.get("pairID", "")),
            ))
    if max_examples and len(examples) > max_examples:
        examples = random.Random(seed).sample(examples, max_examples)
    return examples
