"""Shared data schema. Every downstream module consumes ONLY ``Claim`` records.

The fields are documented on the ``Claim`` dataclass below.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: Allowed values of ``Claim.proxy_label`` (PHEME rumour veracity) besides ``None``.
PROXY_LABELS = ("true", "false", "unverified")

#: Canonical NLI label order used across the whole project.
NLI_LABELS = ("entailment", "neutral", "contradiction")


@dataclass
class Claim:
    """One statement made by one account about one event."""

    claim_id: str
    event: str
    thread_id: str
    account_id: str
    text: str
    timestamp: str | None = None
    proxy_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if the record violates the schema."""
        for name in ("claim_id", "event", "thread_id", "account_id", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Claim.{name} must be a non-empty string, got {value!r}")
        if self.timestamp is not None and not isinstance(self.timestamp, str):
            raise ValueError("Claim.timestamp must be an ISO-8601 string or None")
        if self.proxy_label is not None and self.proxy_label not in PROXY_LABELS:
            raise ValueError(
                f"Claim.proxy_label must be one of {PROXY_LABELS} or None, got {self.proxy_label!r}"
            )
        if not isinstance(self.metadata, dict):
            raise ValueError("Claim.metadata must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "Claim":
        known = {k: record.get(k) for k in
                 ("claim_id", "event", "thread_id", "account_id", "text", "timestamp", "proxy_label")}
        return cls(**known, metadata=dict(record.get("metadata") or {}))


def save_claims(claims: Iterable[Claim], path: str | Path) -> None:
    """Write claims as JSON lines (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for claim in claims:
            fh.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")


def load_claims(path: str | Path) -> list[Claim]:
    """Read claims written by :func:`save_claims`."""
    with open(path, "r", encoding="utf-8") as fh:
        return [Claim.from_dict(json.loads(line)) for line in fh if line.strip()]
