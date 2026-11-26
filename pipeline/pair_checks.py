"""Checks after the NLI stage: set aside flagged pairs that are not real contradictions.

The NLI model flags some pairs of compatible statements as contradictions with probability
close to 1.0, in patterns that are easy to recognise from the text. Each check below looks at
a pair the model flagged and, if the pattern applies, gives a reason to set it aside:

* ``same_quantity``: the same quantity in different units or with an approximate number
  ("about 30 mph" / "about 50 km/h", "about 20 people" / "around 25 people").
* ``non_observation``: one account says it did not see or hear something ("I didn't see a
  gun"); not having seen something is not evidence that it did not happen.
* ``reported_speech``: one account reports what someone said ("the driver told me he had
  stopped"), not what happened.
* ``different_time``: both statements are tied to different moments ("alive when I got there"
  / "dead when the ambulance arrived"); the situation may have changed in between.
* ``different_referent``: the statements are about different people or things ("the cyclist"
  / "the pedestrian", or each witness's own "my car").

A set-aside pair is not counted as a conflict, but it stays in the report with its reason and
its NLI score, so a reader can disagree with the check. The checks never add a conflict.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

CHECKS = ("same_quantity", "non_observation", "reported_speech", "different_time", "different_referent")

REASONS = {
    "same_quantity": "the same quantity, in different units or as an approximate number",
    "non_observation": "one account says it did not see or hear something, which does not mean it did not happen",
    "reported_speech": "one account reports what someone said, not what happened",
    "different_time": "the statements describe different moments; the situation may have changed in between",
    "different_referent": "the statements are about different people or things",
}

# --------------------------------------------------------------------------- quantities
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "a dozen": 12, "dozen": 12,
}
# unit -> (dimension, factor to the dimension's base unit)
_UNITS = {
    "mph": ("speed", 1.609344), "miles per hour": ("speed", 1.609344), "mile per hour": ("speed", 1.609344),
    "km/h": ("speed", 1.0), "kmh": ("speed", 1.0), "kph": ("speed", 1.0), "km per hour": ("speed", 1.0),
    "kilometres per hour": ("speed", 1.0), "kilometers per hour": ("speed", 1.0),
    "km": ("distance", 1000.0), "kilometres": ("distance", 1000.0), "kilometers": ("distance", 1000.0),
    "miles": ("distance", 1609.344), "mile": ("distance", 1609.344),
    "metres": ("distance", 1.0), "meters": ("distance", 1.0), "metre": ("distance", 1.0), "meter": ("distance", 1.0),
    "m": ("distance", 1.0), "feet": ("distance", 0.3048), "foot": ("distance", 0.3048), "ft": ("distance", 0.3048),
    "yards": ("distance", 0.9144), "yard": ("distance", 0.9144), "cm": ("distance", 0.01),
    "seconds": ("duration", 1.0), "second": ("duration", 1.0), "secs": ("duration", 1.0),
    "minutes": ("duration", 60.0), "minute": ("duration", 60.0), "mins": ("duration", 60.0),
    "hours": ("duration", 3600.0), "hour": ("duration", 3600.0),
    "kg": ("mass", 1.0), "kilos": ("mass", 1.0), "pounds": ("mass", 0.4536), "lbs": ("mass", 0.4536),
}
_HEDGES = r"(?:about|around|roughly|approximately|approx\.?|nearly|almost|some|maybe|perhaps|~)"
_NUM = r"(\d+(?:[.,]\d+)?|" + "|".join(sorted(map(re.escape, _WORD_NUMBERS), key=len, reverse=True)) + r")"
_UNIT_RE = "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True))
_QUANTITY = re.compile(rf"(?:(?P<hedge>{_HEDGES})\s+)?{_NUM}\s*(?:(?P<unit>{_UNIT_RE})\b|(?P<noun>[a-z]+))",
                       re.IGNORECASE)
_NOT_COUNT_NOUNS = {"a", "an", "the", "of", "or", "and", "to", "in", "at", "on", "p", "pm", "am", "o"}


def _value(token: str) -> float | None:
    token = token.lower().replace(",", "")
    if token in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[token])
    try:
        return float(token)
    except ValueError:
        return None


def quantities(text: str) -> list[tuple[str, float, bool, str]]:
    """(dimension, value in the base unit, hedged, unit as written) for each quantity in ``text``.

    Counts of things use the dimension ``count:<noun>`` ("20 people" -> count:people). Clock
    times are skipped (a number right after "at" followed by a.m./p.m. or a colon).
    """
    out = []
    for m in _QUANTITY.finditer(text):
        value = _value(m.group(2))
        if value is None:
            continue
        after = text[m.end(2):m.end(2) + 3]
        if after.startswith(":") or re.match(r"\s*[ap]\.?m\b", after, re.IGNORECASE):
            continue  # a clock time, not a quantity
        hedged = bool(m.group("hedge")) or bool(re.search(r"\bor so\b", text[m.end():m.end() + 8], re.I))
        if m.group("unit"):
            dim, factor = _UNITS[m.group("unit").lower()]
            out.append((dim, value * factor, hedged, m.group("unit").lower()))
        elif m.group("noun") and m.group("noun").lower() not in _NOT_COUNT_NOUNS:
            noun = m.group("noun").lower()
            noun = {"men": "people", "women": "people", "persons": "people", "person": "people"}.get(noun, noun)
            out.append((f"count:{noun.rstrip('s')}", value, hedged, noun))
    return out


def _non_numeric_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _WORD_NUMBERS and w not in _UNITS}


def same_quantity(a: str, b: str) -> bool:
    """The statements differ in how a quantity is written, but the values agree.

    Both must give one quantity of the same kind, written differently (another number or
    unit), with the rest of the two sentences largely the same (word overlap >= 0.5). If the
    quantities are written identically ("one shooter" twice), the contradiction is about
    something else and the check does not apply. Exact numbers in the same unit must then be
    equal; in different units they may differ by 3% (rounding in the conversion); if either
    side is approximate ("about", "around", "roughly"...) they may differ by up to 25%.
    """
    qa, qb = quantities(a), quantities(b)
    if not qa or not qb:
        return False
    shared = {q[0] for q in qa} & {q[0] for q in qb}
    if not shared:
        return False
    wa, wb = _non_numeric_words(a), _non_numeric_words(b)
    if not wa or not wb or len(wa & wb) / len(wa | wb) < 0.5:
        return False
    if all([q[1:] for q in qa if q[0] == d] == [q[1:] for q in qb if q[0] == d] for d in shared):
        return False  # same quantities as written: the conflict lies elsewhere
    for dim in shared:
        va = [q[1:] for q in qa if q[0] == dim]
        vb = [q[1:] for q in qb if q[0] == dim]
        if len(va) != 1 or len(vb) != 1:
            return False
        (x, hx, ux), (y, hy, uy) = va[0], vb[0]
        tolerance = 0.25 if (hx or hy) else (0.03 if ux != uy else 0.0)
        if abs(x - y) > tolerance * max(abs(x), abs(y), 1e-9):
            return False
    return True


# --------------------------------------------------------------------------- non-observation
_NON_OBSERVATION = re.compile(
    r"\b(?:didn't|did not|never|couldn't|could not|can't|cannot|don't|do not|wasn't able to|"
    r"was not able to|have no|had no)\s+(?:actually\s+|really\s+|clearly\s+)?"
    r"(?:see|notice|hear|spot|remember|recall|witness|memory of)\b", re.IGNORECASE)


def non_observation(a: str, b: str) -> bool:
    return bool(_NON_OBSERVATION.search(a)) != bool(_NON_OBSERVATION.search(b))


# --------------------------------------------------------------------------- reported speech
_REPORTED = re.compile(
    r"\b(?:told (?:me|us|him|her|them|the police|police|officers)|said (?:to me|to us)|"
    r"(?:he|she|they|the driver|the man|the woman) (?:said|says|claimed|claims|insisted|admitted)(?! nothing)|"
    r"kept saying|according to (?:him|her|them|the driver))\b", re.IGNORECASE)


def reported_speech(a: str, b: str) -> bool:
    return bool(_REPORTED.search(a)) != bool(_REPORTED.search(b))


# --------------------------------------------------------------------------- different time
_ANCHOR = re.compile(r"\b(?:when|by the time|once|as soon as)\s+([^,.;]+)", re.IGNORECASE)
_STOP = {"the", "a", "an", "i", "we", "he", "she", "they", "it", "was", "were", "is", "are", "had",
         "has", "to", "of", "and", "my", "our", "his", "her", "their"}


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP}


def different_time(a: str, b: str) -> bool:
    """Both statements are anchored to a moment ("when ...", "by the time ...") and the
    two moments share few content words, so each may describe a different point in time."""
    ma, mb = _ANCHOR.search(a), _ANCHOR.search(b)
    if not ma or not mb:
        return False
    # only the first few words name the moment ("when it crashed", not what follows)
    ca = _content(" ".join(ma.group(1).split()[:4]))
    cb = _content(" ".join(mb.group(1).split()[:4]))
    if not ca or not cb:
        return False
    # overlap relative to the shorter phrase: "it crashed" is the same moment as "it crashed in"
    return len(ca & cb) / min(len(ca), len(cb)) < 0.5


# --------------------------------------------------------------------------- different referent
# specific roles; members of one group may be the same person, different groups may not
_ROLE_GROUPS = {
    "cyclist": "cyclist", "rider": "cyclist", "biker": "cyclist",
    "pedestrian": "pedestrian",
    "driver": "driver", "motorist": "driver",
    "passenger": "passenger",
    "gunman": "attacker", "shooter": "attacker", "attacker": "attacker", "assailant": "attacker",
    "suspect": "attacker", "killer": "attacker", "perpetrator": "attacker",
    "officer": "police", "policeman": "police", "policewoman": "police", "cop": "police",
    "victim": "victim", "bystander": "bystander", "paramedic": "medic", "nurse": "medic", "doctor": "medic",
    "guard": "guard", "shopkeeper": "shopkeeper", "cashier": "shopkeeper",
}
_ROLE_SUBJECT = re.compile(r"^\s*(?:the|a|an|that|this)\s+(?:[a-z\-]+\s+){0,2}?(" +
                           "|".join(_ROLE_GROUPS) + r")s?\b", re.IGNORECASE)
_OWN = re.compile(r"^\s*(?:my|our)\s+([a-z]+)", re.IGNORECASE)


def different_referent(a: str, b: str, account_a: str, account_b: str) -> bool:
    """Subjects are different specific roles ("the cyclist" / "the pedestrian"), or each
    witness talks about their own thing ("my car" in two different accounts)."""
    oa, ob = _OWN.match(a), _OWN.match(b)
    if oa and ob and account_a != account_b:
        return True
    ra, rb = _ROLE_SUBJECT.match(a), _ROLE_SUBJECT.match(b)
    if ra and rb:
        return _ROLE_GROUPS[ra.group(1).lower()] != _ROLE_GROUPS[rb.group(1).lower()]
    return False


# --------------------------------------------------------------------------- driver
def check_pair(a: str, b: str, account_a: str = "a", account_b: str = "b",
               enabled: Sequence[str] = CHECKS) -> str | None:
    """Name of the first enabled check that applies to the statement pair, or None."""
    tests = {
        "same_quantity": lambda: same_quantity(a, b),
        "non_observation": lambda: non_observation(a, b),
        "reported_speech": lambda: reported_speech(a, b),
        "different_time": lambda: different_time(a, b),
        "different_referent": lambda: different_referent(a, b, account_a, account_b),
    }
    for name in CHECKS:
        if name in enabled and tests[name]():
            return name
    return None


def enabled_checks(cfg: dict) -> list[str]:
    c = cfg.get("checks", {})
    if not c.get("enabled", False):
        return []
    return [name for name in CHECKS if c.get(name, True)]


def apply_checks(texts_a: Sequence[str], texts_b: Sequence[str], accounts_a: Sequence[str],
                 accounts_b: Sequence[str], probs: np.ndarray, threshold: float,
                 enabled: Sequence[str]) -> tuple[np.ndarray, dict[int, str]]:
    """Run the checks on pairs at or above ``threshold``.

    Returns ``(effective_probs, set_aside)``: a copy of ``probs`` in which every set-aside
    pair's contradiction mass is moved to neutral, and ``{pair index: check name}``.
    """
    effective = np.array(probs, dtype=np.float32, copy=True)
    set_aside: dict[int, str] = {}
    if not enabled or not len(probs):
        return effective, set_aside
    for k in np.where(effective[:, 2] >= threshold)[0]:
        name = check_pair(texts_a[k], texts_b[k], accounts_a[k], accounts_b[k], enabled)
        if name:
            set_aside[int(k)] = name
            effective[k, 1] += effective[k, 2]
            effective[k, 2] = 0.0
    return effective, set_aside


def describe(name: str) -> dict[str, Any]:
    return {"check": name, "reason": REASONS[name]}
