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


# A full stop after these never ends a sentence when a capitalised word follows
# ("Dr. Patel", "St. James Street", "approx. 10 p.m."). a.m. / p.m. are not listed: they often
# DO end a sentence ("It was 10 p.m. The van left."), so they keep the default behaviour.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "rd", "ave", "blvd", "sgt",
    "cpl", "lt", "col", "capt", "gen", "insp", "det", "supt", "pc", "approx", "ca", "c", "no",
    "nos", "vs", "v", "fig", "dept", "est", "e.g", "i.e", "cf", "inc", "ltd", "co", "corp",
}
_DOT = "․"  # placeholder for a protected full stop
_ABBR_DOT = re.compile(r"\b([A-Za-z][A-Za-z.]{0,5})\.(?=\s)")
_DOTTED_ACRONYM = re.compile(r"\b((?:[A-Za-z]\.){2,})(?=\s+[a-z])")  # "U.S. embassy", not "p.m. The"


def _protect(match: re.Match) -> str:
    word = match.group(1)
    if word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.isupper()):  # titles, initials
        return word + _DOT
    return match.group(0)


def split_sentences(text: str, mode: str = "abbrev") -> list[str]:
    """Sentence splitter used to turn a free-text account into claims.

    ``mode="abbrev"`` (default) does not split after common abbreviations, titles, initials
    and dotted acronyms, so "It was approx. 10 p.m." stays one sentence. ``mode="simple"``
    is the original rule (split after . ! ? before a capital or digit), kept for comparison.
    """
    if mode not in ("simple", "abbrev"):
        raise ValueError(f"split mode must be simple | abbrev, got {mode!r}")
    parts = []
    for chunk in text.splitlines():
        if mode == "abbrev":
            chunk = _DOTTED_ACRONYM.sub(lambda m: m.group(1).replace(".", _DOT), chunk)
            chunk = _ABBR_DOT.sub(_protect, chunk)
        parts += [p.replace(_DOT, ".").strip() for p in _SENT_SPLIT.split(chunk)]
    return [p for p in parts if p]


# Clause boundaries tried in order when a sentence is too long for the NLI input window.
_CLAUSE_BOUNDARIES = [re.compile(p) for p in (
    r";\s+", r",\s+and then\s+", r"\s+and then\s+", r",\s+(?:and|but|so|while|when)\s+",
    r",\s+", r"\s+(?:and|but)\s+")]


def split_long_claim(text: str, max_words: int | None) -> list[str]:
    """Split a run-on sentence into clause-sized claims of at most ``max_words`` words.

    The NLI model sees premise + hypothesis together, truncated at ``nli.max_length`` tokens,
    so a 90-word sentence loses its end. Pieces are cut at the first boundary type that
    applies (``;``, ``, and then``, ``, and``/``, but``, ``,``, `` and ``), and neighbouring
    pieces are joined again while they still fit, to keep as much context as possible.
    """
    if not max_words or n_tokens(text) <= max_words:
        return [text]
    for boundary in _CLAUSE_BOUNDARIES:
        atoms = [a.strip() for a in boundary.split(text) if a.strip()]
        if len(atoms) < 2:
            continue
        out: list[str] = []
        for atom in atoms:
            if out and n_tokens(out[-1]) + n_tokens(atom) <= max_words:
                out[-1] = f"{out[-1]}, {atom}"
            else:
                out.append(atom)
        # recurse on pieces that are still too long (with the finer boundaries)
        return [piece for o in out for piece in split_long_claim(o, max_words)]
    words = text.split()  # no boundary at all: hard cut
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


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
