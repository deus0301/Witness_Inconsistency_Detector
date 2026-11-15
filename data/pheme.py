"""PHEME loader: local thread folders -> list of ``Claim`` records.

Real on-disk layout (verified, both v1 and v2)::

    <root>/<event>[-all-rnr-threads]/{rumours,non-rumours}/<thread_id>/
        source-tweets/<thread_id>.json      one tweet (Twitter API v1.1 JSON)
        reactions/<tweet_id>.json           replies
        structure.json                      reply tree
        annotation.json                     v2 only

``annotation.json`` for rumour threads holds ``is_rumour``, ``category`` (the rumour
story the thread belongs to), ``misinformation``, ``true``, ``links`` and
``is_turnaround``. Non-rumour threads only hold ``is_rumour``. There are NO per-reply
stance labels in these releases. macOS ``._*`` resource-fork files are skipped.
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from data.cleaning import clean_text, deduplicate, n_tokens
from data.schema import Claim

V2_SUFFIX = "-all-rnr-threads"
_TWITTER_TIME = "%a %b %d %H:%M:%S %z %Y"


def convert_veracity(annotation: dict[str, Any]) -> str | None:
    """Map PHEME's ``misinformation`` / ``true`` flags to true / false / unverified.

    Same rule as the ``convert_veracity_annotations.py`` script shipped with PHEME.
    """
    has_m, has_t = "misinformation" in annotation, "true" in annotation
    if not has_m:
        return None
    m = int(annotation["misinformation"])
    if has_t:
        t = int(annotation["true"])
        if m == 0 and t == 0:
            return "unverified"
        if m == 0 and t == 1:
            return "true"
        if m == 1 and t == 0:
            return "false"
        return None  # both flags set: inconsistent annotation
    return "false" if m == 1 else "unverified"


def parse_timestamp(created_at: str | None) -> str | None:
    """Twitter ``created_at`` -> ISO-8601 (UTC)."""
    if not created_at:
        return None
    try:
        return datetime.strptime(created_at, _TWITTER_TIME).isoformat()
    except ValueError:
        return None


def event_dir(root: Path, event: str) -> Path:
    """Locate the folder of ``event`` under a v2 or v1 root."""
    for candidate in (root / f"{event}{V2_SUFFIX}", root / event):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"PHEME event '{event}' not found under {root}")


def list_events(root: Path) -> list[str]:
    """Event names available under ``root`` (suffix stripped)."""
    return sorted(
        p.name[: -len(V2_SUFFIX)] if p.name.endswith(V2_SUFFIX) else p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix == ".json" and not p.name.startswith("."))


def _tweet_to_claim(tweet: dict, event: str, thread_id: str, is_source: bool,
                    is_rumour: bool, annotation: dict) -> Claim | None:
    raw = tweet.get("text") or ""
    text = clean_text(raw)
    if not text:
        return None
    veracity = convert_veracity(annotation) if is_rumour else None
    user = tweet.get("user") or {}
    return Claim(
        claim_id=f"{event}:{tweet.get('id_str') or tweet.get('id')}",
        event=event,
        thread_id=str(thread_id),
        account_id=str(user.get("screen_name") or user.get("id_str") or user.get("id") or "unknown"),
        text=text,
        timestamp=parse_timestamp(tweet.get("created_at")),
        # veracity belongs to the rumour story the *source* tweet reports; replies get none
        proxy_label=veracity if is_source else None,
        metadata={
            "is_source": is_source,
            "is_rumour": is_rumour,
            "veracity": veracity,
            "rumour_category": annotation.get("category") if is_rumour else None,
            "is_turnaround": bool(annotation.get("is_turnaround")) if is_rumour and is_source else False,
            "lang": tweet.get("lang"),
            "parent_id": str(tweet.get("in_reply_to_status_id_str") or "") or None,
            "raw_text": raw,
        },
    )


def load_pheme_event(
    root: str | Path,
    event: str,
    include_replies: bool = False,
    languages: list[str] | None = ("en",),
    min_tokens: int = 4,
    dedup: bool = True,
    max_claims: int | None = None,
    rumours_only: bool = False,
    seed: int = 42,
) -> list[Claim]:
    """Load, clean, filter and (optionally) deduplicate one PHEME event.

    Args:
        root: PHEME root (v2 ``all-rnr-annotated-threads`` or v1 ``pheme-rnr-dataset``).
        event: event name, e.g. ``ottawashooting`` or ``germanwings-crash``.
        include_replies: also return reaction tweets as claims (default: source tweets only).
        languages: keep tweets whose Twitter ``lang`` is in this list; ``None``/empty keeps all.
        min_tokens: drop cleaned texts shorter than this many whitespace tokens.
        dedup: drop exact duplicates of the normalised text.
        max_claims: seeded random subsample to at most this many claims.
        rumours_only: keep only threads annotated as rumours (these carry story + veracity).
    """
    folder = event_dir(Path(root), event)
    claims: list[Claim] = []
    kinds = ("rumours",) if rumours_only else ("rumours", "non-rumours")
    for kind in kinds:
        kind_dir = folder / kind
        if not kind_dir.is_dir():
            continue
        for thread in sorted(kind_dir.iterdir()):
            if thread.name.startswith(".") or not thread.is_dir():
                continue
            ann_path = thread / "annotation.json"
            annotation = _read_json(ann_path) if ann_path.exists() else {}
            is_rumour = kind == "rumours"
            tweets = [(p, True) for p in _json_files(thread / "source-tweets")]
            if include_replies:
                tweets += [(p, False) for p in _json_files(thread / "reactions")]
            for path, is_source in tweets:
                tweet = _read_json(path)
                if languages and tweet.get("lang") not in languages:
                    continue
                claim = _tweet_to_claim(tweet, event, thread.name, is_source, is_rumour, annotation)
                if claim is not None and n_tokens(claim.text) >= min_tokens:
                    claims.append(claim)
    if dedup:
        claims = deduplicate(claims)
    claims.sort(key=lambda c: (c.timestamp or "9999", c.claim_id))
    if max_claims and len(claims) > max_claims:
        rng = random.Random(seed)
        claims = sorted(rng.sample(claims, max_claims), key=lambda c: (c.timestamp or "9999", c.claim_id))
    return claims


def load_event_from_config(cfg: dict, event: str, rumours_only: bool = False,
                           max_claims: int | None = None) -> list[Claim]:
    """Load ``event`` using the paths / data options in the config (v2 first, then v1)."""
    from data.schema import load_claims, save_claims
    from pipeline.config import resolve_path, text_hash

    d = cfg["data"]
    cap = max_claims if max_claims is not None else d.get("max_claims_per_event")
    # PHEME is tens of thousands of small files; cache the cleaned claims per option set
    key = text_hash(event, str(d["include_replies"]), str(d["languages"]), str(d["min_tokens"]),
                    str(d["dedup"]), str(rumours_only), str(cfg["paths"]["pheme_v2"]))[:12]
    cache = resolve_path(cfg, "cache_dir") / f"claims_{event}_{key}.jsonl"
    claims: list[Claim] | None = load_claims(cache) if cache.exists() else None
    last_error: Exception | None = None
    if claims is None:
        for root_key in ("pheme_v2", "pheme_v1"):
            try:
                claims = load_pheme_event(
                    resolve_path(cfg, root_key), event,
                    include_replies=d["include_replies"], languages=d["languages"],
                    min_tokens=d["min_tokens"], dedup=d["dedup"], rumours_only=rumours_only)
                save_claims(claims, cache)
                break
            except FileNotFoundError as exc:
                last_error = exc
    if claims is None:
        raise FileNotFoundError(f"Event '{event}' not found in PHEME v2 or v1: {last_error}")
    if cap and len(claims) > cap:
        rng = random.Random(cfg["seed"])
        claims = sorted(rng.sample(claims, cap), key=lambda c: (c.timestamp or "9999", c.claim_id))
    return claims
