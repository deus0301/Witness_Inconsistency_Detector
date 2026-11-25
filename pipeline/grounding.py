"""Speaker grounding: rewrite first-person words to the account's name before NLI.

NLI models read "I" in the premise and "I" in the hypothesis as the same person. Across
accounts they never are: "I was inside the shop" (Witness A) and "I was outside on the
pavement" (Witness B) score contradiction 1.000, though both can be true. Rewriting to
"Witness A was inside the shop" / "Witness B was outside on the pavement" tells the model
who is speaking. The rewritten text is used for NLI only; reports quote the original.

Rule-based on purpose (no extra model): first-person singular and plural pronouns plus the
few verb forms that must agree ("I am" -> "Witness A is"). Third-person pronouns are left
alone; resolving them needs coreference, which this does not attempt.
"""
from __future__ import annotations

import re

_ACCOUNT_N = re.compile(r"^account_(\d+)$", re.IGNORECASE)

# (phrase, replacement template) in order; {n} = speaker name, {g} = "<name>'s group".
# Matched as whole words, case-insensitively, except "us" ("the US embassy" is left alone).
_RULE_SPECS = [
    ("I'm", "{n} is"), ("I am", "{n} is"), ("I've", "{n} has"), ("I have", "{n} has"),
    ("I'll", "{n} will"), ("I'd", "{n} would"), ("I don't", "{n} doesn't"),
    ("I do not", "{n} does not"), ("I do", "{n} does"), ("I", "{n}"),
    ("myself", "{n}"), ("mine", "{n}'s"), ("my", "{n}'s"), ("me", "{n}"),
    ("we're", "{g} is"), ("we've", "{g} has"), ("we are", "{g} is"), ("we were", "{g} was"),
    ("we have", "{g} has"), ("we", "{g}"), ("ourselves", "{g}"), ("ours", "{g}'s"), ("our", "{g}'s"),
]
_WORD = r"(?<![\w']){}(?![\w'])"
_RULES = ([(re.compile(_WORD.format(re.escape(p)), re.IGNORECASE), r) for p, r in _RULE_SPECS]
          + [(re.compile(_WORD.format("us")), "{g}")])


def speaker_name(account_id: str) -> str:
    """Readable speaker name: ``account_3`` -> ``Witness 3``; other ids are used as given."""
    m = _ACCOUNT_N.match(account_id.strip())
    return f"Witness {m.group(1)}" if m else account_id.strip()


# "I heard two shots" is a claim about the event (two shots), not about the speaker: naming
# the hearer ("Witness A heard two shots" vs "Witness B heard three shots") makes the model
# call two different hearers compatible. An affirmative perception verb at the start of the
# sentence is therefore dropped instead of grounded. Negated perception ("I didn't see a
# gun") is a statement about the speaker and is grounded as usual.
_PERCEPTION = re.compile(r"^\s*(?:I|we)\s+(?:also\s+)?(?:clearly\s+)?(?:heard|saw|noticed|counted|watched|"
                         r"spotted|witnessed)\s+(?:that\s+)?", re.IGNORECASE)


def ground_speaker(text: str, account_id: str) -> str:
    """Replace first-person references in ``text`` with the speaker's name."""
    name = speaker_name(account_id)
    group = f"{name}'s group"
    out = _PERCEPTION.sub("", text, count=1)
    for pattern, template in _RULES:
        out = pattern.sub(template.format(n=name, g=group).replace("\\", "\\\\"), out)
    return out[0].upper() + out[1:] if out else out
