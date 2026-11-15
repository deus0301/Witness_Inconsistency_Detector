"""Proxy labels derived from PHEME annotations.

PHEME has NO entailment/contradiction labels and (in the releases used here) NO reply
stance labels. What it does have, for rumour threads only, is:

* ``rumour_category`` - the rumour *story* a source tweet reports (e.g. "There were 148
  people onboard the Germanwings plane"),
* ``veracity`` of that story (true / false / unverified), constant within a story,
* ``is_turnaround`` - the tweet resolved the story's status (for a false story: a debunk).

Everything below is a PROXY built from those fields. It is never NLI gold.

Pair-level proxy (3 classes), used to sanity-check the NLI scorer on tweets:

* ``contradiction`` - a turnaround (debunking) tweet of a FALSE story paired with a
  non-turnaround tweet of the same story.
* ``entailment``    - two non-turnaround tweets of the same TRUE story.
* ``neutral``       - two tweets from different stories of the same event.

Cluster-level proxy, used to evaluate the full pipeline:

* a cluster is disagreement-positive if it mixes tweets from a TRUE story and a FALSE
  story, or holds a false-story turnaround tweet together with a non-turnaround tweet of
  that same story.
"""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations
from typing import Sequence

from data.schema import Claim


def _story(claim: Claim) -> str | None:
    return claim.metadata.get("rumour_category")


def is_debunk_pair(a: Claim, b: Claim) -> bool:
    """True when exactly one of two same-story FALSE-rumour tweets is a turnaround."""
    return (
        _story(a) is not None and _story(a) == _story(b)
        and a.proxy_label == "false" and b.proxy_label == "false"
        and bool(a.metadata.get("is_turnaround")) != bool(b.metadata.get("is_turnaround"))
    )


def pair_proxy_label(a: Claim, b: Claim) -> str | None:
    """Proxy NLI-like label for a pair of rumour source tweets, or ``None`` if undefined."""
    sa, sb = _story(a), _story(b)
    if sa is None or sb is None:
        return None
    if sa != sb:
        return "neutral"
    if is_debunk_pair(a, b):
        return "contradiction"
    if (a.proxy_label == "true" and b.proxy_label == "true"
            and not a.metadata.get("is_turnaround") and not b.metadata.get("is_turnaround")):
        return "entailment"
    return None


def build_proxy_pairs(claims: Sequence[Claim], per_class: int = 400,
                      seed: int = 42) -> list[tuple[Claim, Claim, str]]:
    """Sample up to ``per_class`` proxy-labelled pairs per class from one event's claims."""
    rng = random.Random(seed)
    rumour = [c for c in claims if _story(c) is not None and c.metadata.get("is_source", True)]
    by_story: dict[str, list[Claim]] = defaultdict(list)
    for c in rumour:
        by_story[_story(c)].append(c)

    buckets: dict[str, list[tuple[Claim, Claim, str]]] = {"contradiction": [], "entailment": [], "neutral": []}
    for members in by_story.values():
        for a, b in combinations(members, 2):
            if a.account_id == b.account_id:
                continue
            label = pair_proxy_label(a, b)
            if label in ("contradiction", "entailment"):
                buckets[label].append((a, b, label))
    # neutral: sample cross-story pairs without enumerating all of them
    stories = [s for s, m in by_story.items() if m]
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(stories) > 1 and len(buckets["neutral"]) < per_class and attempts < per_class * 20:
        attempts += 1
        s1, s2 = rng.sample(stories, 2)
        a, b = rng.choice(by_story[s1]), rng.choice(by_story[s2])
        key = tuple(sorted((a.claim_id, b.claim_id)))
        if key in seen or a.account_id == b.account_id:
            continue
        seen.add(key)
        buckets["neutral"].append((a, b, "neutral"))

    pairs: list[tuple[Claim, Claim, str]] = []
    for label in ("entailment", "neutral", "contradiction"):
        items = buckets[label]
        if len(items) > per_class:
            items = rng.sample(items, per_class)
        pairs.extend(items)
    return pairs


def cluster_proxy_positive(members: Sequence[Claim]) -> bool:
    """Primary cluster-level proxy: mixed true/false stories, or a debunk next to its rumour."""
    labels = {c.proxy_label for c in members}
    if "true" in labels and "false" in labels:
        return True
    return any(is_debunk_pair(a, b) for a, b in combinations(members, 2))


def cluster_proxy_contested(members: Sequence[Claim]) -> bool:
    """Looser secondary proxy: a false-story tweet shares the cluster with any other-story
    or non-false tweet (true rumour or non-rumour)."""
    false = [c for c in members if c.proxy_label == "false"]
    if not false:
        return False
    return any(c.proxy_label != "false" for c in members) or cluster_proxy_positive(members)
