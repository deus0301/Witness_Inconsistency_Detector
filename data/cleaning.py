"""Text cleaning for tweets and free-text accounts."""
from __future__ import annotations

import html
import re
from typing import Iterable

from data.schema import Claim

_URL = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+)", re.IGNORECASE)
_RT = re.compile(r"^\s*(?:(?:RT|MT)\s+@\w+\s*:?\s*)+", re.IGNORECASE)
_RT_INLINE = re.compile(r"\b(?:RT|MT)\s+@\w+\s*:?", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_LONE_AT = re.compile(r"(?<!\S)@(?!\w)")
_HASHTAG = re.compile(r"#(\w+)")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F\U0000200D\U000020E3]+"
)
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "…": "..."})
_SPACE = re.compile(r"\s+")
_LEAD_PUNCT = re.compile(r"^[\s:;,.\-|>]+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _split_hashtag(match: re.Match) -> str:
    """``#OttawaShooting`` -> ``Ottawa Shooting`` (keeps the words, drops the marker)."""
    return _CAMEL.sub(" ", match.group(1))


def clean_text(text: str) -> str:
    """Normalise one tweet / statement.

    Removes URLs, retweet markers and @mentions, unescapes HTML entities, strips emoji,
    keeps hashtag words (without ``#``, CamelCase split) and collapses whitespace.
    """
    if not text:
        return ""
    text = html.unescape(text).translate(_QUOTES)
    text = _URL.sub(" ", text)
    text = _RT.sub(" ", text)
    text = _RT_INLINE.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _LONE_AT.sub("at", text)  # "look @ Gaza" -> "look at Gaza"
    text = _HASHTAG.sub(_split_hashtag, text)
    text = _EMOJI.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    text = _LEAD_PUNCT.sub("", text)
    return text.strip()


def normalise_for_dedup(text: str) -> str:
    """Key used for duplicate detection: lower-cased alphanumerics only."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def n_tokens(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Light sentence splitter used to turn a free-text account into claims."""
    parts = [p.strip() for chunk in text.splitlines() for p in _SENT_SPLIT.split(chunk)]
    return [p for p in parts if p]


def deduplicate(claims: Iterable[Claim]) -> list[Claim]:
    """Drop claims whose normalised text was already seen (earliest timestamp wins).

    The surviving claim records how many copies were folded into it in
    ``metadata['n_duplicates']`` and which accounts posted them in
    ``metadata['duplicate_accounts']``.
    """
    ordered = sorted(claims, key=lambda c: (c.timestamp or "9999", c.claim_id))
    kept: dict[str, Claim] = {}
    for claim in ordered:
        key = normalise_for_dedup(claim.text)
        if key in kept:
            first = kept[key]
            first.metadata["n_duplicates"] = first.metadata.get("n_duplicates", 0) + 1
            first.metadata.setdefault("duplicate_accounts", []).append(claim.account_id)
        else:
            claim.metadata.setdefault("n_duplicates", 0)
            kept[key] = claim
    return list(kept.values())
