"""PHEME RTE loader (``pheme_rte_datasets_2016``): tweet pairs labelled for entailment.

Verified layout of the local copy::

    pheme_rte_datasets_2016/pheme_rte_datasets_2016/per_event_splits/pheme_rte_<event>_en.xml
        <pair entailment="ENTAILMENT|CONTRADICTION|UNKNOWN" id="<tweetA>.json_<tweetB>.json"
              rte_type="..." rumours="Story 1: ...; Story 2: ..."> <t>text</t> <h>text</h> </pair>

What the files show about the labels: every ENTAILMENT pair names ONE rumour story, every
CONTRADICTION and UNKNOWN pair names TWO. So the label is assigned through the pair's
rumour stories (same story -> entailment; two stories judged to conflict -> contradiction;
two unrelated stories -> unknown), not by reading each tweet pair on its own. It is a
human-defined, pair-level reference and much closer to the task than veracity proxies,
but it is still not sentence-level NLI gold. ``UNKNOWN`` is mapped to ``neutral``.

URLs are already replaced by the token ``URL`` in these files; it is removed on load.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from data.cleaning import clean_text

#: file-name key -> PHEME event name used everywhere else in the project
EVENT_FILES = {
    "ottawashooting": "pheme_rte_ottawa_en.xml",
    "sydneysiege": "pheme_rte_ssiege_en.xml",
    "charliehebdo": "pheme_rte_chebdo_en.xml",
    "germanwings-crash": "pheme_rte_gw_en.xml",
}
LABEL_MAP = {"ENTAILMENT": "entailment", "CONTRADICTION": "contradiction", "UNKNOWN": "neutral"}
_URL_TOKEN = re.compile(r"\bURL\b")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class RTEPair:
    """One labelled tweet pair from the PHEME RTE release."""

    event: str
    pair_id: str
    tweet_a: str          # tweet id of the premise
    tweet_b: str          # tweet id of the hypothesis
    premise: str
    hypothesis: str
    label: str            # entailment | neutral | contradiction
    stories: str          # the ``rumours`` attribute, verbatim


def _clean(text: str | None) -> str:
    text = _URL_TOKEN.sub(" ", clean_text(text or ""))
    return _SPACE.sub(" ", text).strip()


def load_pheme_rte(root: str | Path, event: str, drop_identical: bool = True) -> list[RTEPair]:
    """Load the labelled pairs of one event. Duplicate pair ids keep their first occurrence.

    ``drop_identical`` removes pairs whose two cleaned texts are the same string.
    """
    if event not in EVENT_FILES:
        raise ValueError(f"No PHEME RTE file for event '{event}' (have {sorted(EVENT_FILES)})")
    path = Path(root) / EVENT_FILES[event]
    if not path.exists():
        raise FileNotFoundError(path)
    pairs: list[RTEPair] = []
    seen: set[str] = set()
    for node in ET.parse(path).getroot().iter("pair"):
        pair_id = node.get("id", "")
        label = LABEL_MAP.get(node.get("entailment", ""))
        if label is None or pair_id in seen:
            continue
        seen.add(pair_id)
        premise, hypothesis = _clean(node.findtext("t")), _clean(node.findtext("h"))
        if not premise or not hypothesis or (drop_identical and premise == hypothesis):
            continue
        ids = [part.replace(".json", "") for part in pair_id.split("_")]
        pairs.append(RTEPair(event, pair_id, ids[0], ids[-1], premise, hypothesis, label,
                             node.get("rumours", "")))
    return pairs
