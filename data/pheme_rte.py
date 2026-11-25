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

The release also ships leave-one-event-out splits in the same ``<pair>`` format::

    .../test_training_splits/pheme_{train,test}_leave_{chebdo,gw,ott,syd}.xml

``load_rte_file`` parses any of these files; ``load_rte_split`` picks the train or test file of
one held-out event. A pair's event is read from its ``rte_type`` prefix (``gw_all_cd`` ->
``germanwings-crash``), so pairs in the mixed train files keep their own event.
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
#: PHEME event -> key used in the leave-one-event-out split file names
SPLIT_KEYS = {
    "charliehebdo": "chebdo",
    "germanwings-crash": "gw",
    "ottawashooting": "ott",
    "sydneysiege": "syd",
}
#: prefix of the ``rte_type`` attribute -> PHEME event
RTE_TYPE_EVENTS = {"chebdo": "charliehebdo", "gw": "germanwings-crash",
                   "ottawa": "ottawashooting", "syd": "sydneysiege"}
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


def load_rte_file(path: str | Path, event: str | None = None,
                  drop_identical: bool = True) -> list[RTEPair]:
    """Parse one PHEME RTE XML file (per-event or leave-one-event-out split).

    Duplicate pair ids keep their first occurrence; ``drop_identical`` removes pairs whose
    two cleaned texts are the same string. ``event`` is used only when a pair's ``rte_type``
    has no known event prefix.
    """
    path = Path(path)
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
        pair_event = RTE_TYPE_EVENTS.get(node.get("rte_type", "").split("_")[0], event)
        if pair_event is None:
            raise ValueError(f"{path.name}: pair {pair_id} has unknown rte_type "
                             f"{node.get('rte_type')!r} and no event was given")
        ids = [part.replace(".json", "") for part in pair_id.split("_")]
        pairs.append(RTEPair(pair_event, pair_id, ids[0], ids[-1], premise, hypothesis, label,
                             node.get("rumours", "")))
    return pairs


def load_pheme_rte(root: str | Path, event: str, drop_identical: bool = True) -> list[RTEPair]:
    """Load the labelled pairs of one event from ``per_event_splits``."""
    if event not in EVENT_FILES:
        raise ValueError(f"No PHEME RTE file for event '{event}' (have {sorted(EVENT_FILES)})")
    return load_rte_file(Path(root) / EVENT_FILES[event], event, drop_identical)


def split_file(splits_dir: str | Path, held_out: str, part: str) -> Path:
    """Path of ``pheme_<part>_leave_<key>.xml`` for held-out event ``held_out``."""
    if held_out not in SPLIT_KEYS:
        raise ValueError(f"No leave-one-out split for event '{held_out}' (have {sorted(SPLIT_KEYS)})")
    if part not in ("train", "test"):
        raise ValueError(f"part must be 'train' or 'test', got {part!r}")
    return Path(splits_dir) / f"pheme_{part}_leave_{SPLIT_KEYS[held_out]}.xml"


def load_rte_split(splits_dir: str | Path, held_out: str, part: str,
                   drop_identical: bool = True) -> list[RTEPair]:
    """Load the train or test half of the leave-``held_out``-out split."""
    return load_rte_file(split_file(splits_dir, held_out, part),
                         held_out if part == "test" else None, drop_identical)


def text_key(pair: RTEPair) -> frozenset[str]:
    """Order-free identity of a pair's two texts (catches (A,B) vs (B,A) duplicates)."""
    return frozenset((pair.premise, pair.hypothesis))


def remove_leakage(train: list[RTEPair], test: list[RTEPair]) -> tuple[list[RTEPair], dict[str, int]]:
    """Drop train pairs whose id or (unordered) text pair also appears in ``test``."""
    test_ids = {p.pair_id for p in test}
    test_texts = {text_key(p) for p in test}
    kept, by_id, by_text = [], 0, 0
    for p in train:
        if p.pair_id in test_ids:
            by_id += 1
        elif text_key(p) in test_texts:
            by_text += 1
        else:
            kept.append(p)
    return kept, {"removed_same_id": by_id, "removed_same_text": by_text,
                  "removed_total": by_id + by_text}
