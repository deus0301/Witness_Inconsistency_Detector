"""Witness-style regression probes for the text layer around the NLI stage.

    python -m evaluation.witness_probes --tag before     # writes reports/results/witness_probes_before.json

The PHEME tweets never pass through the sentence splitter or the speaker rewrite, so the
labelled-pair metrics cannot show what those steps do. These probes can. They come from the
adversarial review (``critic/adversary.md``): each case has two or more short accounts, run
through the REAL pipeline (``accounts_to_claims`` -> clustering -> NLI), and expectations:

* ``claims_include``: a substring that must appear in some claim (nothing lost by splitting).
* ``no_fragment_claims``: no claim may be a stub such as "It was approx." (< 4 words).
* ``flag``: a pair of substrings whose claims must be flagged as contradicting.
* ``no_flag``: a pair of substrings whose claims must NOT be flagged.

A flag expectation passes only if the two claims were compared and scored at or above the
threshold, so a conflict lost to clustering also fails. This is a small, hand-written suite
(about 30 checks): it shows failure modes and regressions, not error rates.
"""
from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Sequence

from pipeline.config import fix_console_encoding, load_config

_LONG_BODY = ("I was walking along the canal path at the time and I heard a loud bang which I first thought "
              "was a firework, and I turned around towards the junction near the bakery and the pharmacy "
              "and I saw that the blue car had hit the cyclist, and the cyclist was lying on the ground "
              "near the kerb with his bike under him, and a woman in a yellow coat ran over to help him and "
              "somebody else was shouting for an ambulance, and the driver of the blue car got out and "
              "looked at the bike for a few seconds, ")

CASES: list[dict[str, Any]] = [
    {"name": "first_person_location", "type": "first_person",
     "accounts": [{"account_id": "Witness A", "text": "I was inside the shop when it happened."},
                  {"account_id": "Witness B", "text": "I was outside on the pavement when it happened."}],
     "no_flag": [("inside the shop", "outside on the pavement")]},
    {"name": "first_person_car", "type": "first_person",
     "accounts": [{"account_id": "Witness A", "text": "My car was parked outside the bank."},
                  {"account_id": "Witness B", "text": "My car was in the garage all day."}],
     "no_flag": [("parked outside the bank", "in the garage")]},
    {"name": "first_person_position", "type": "first_person",
     "accounts": [{"account_id": "Witness A", "text": "I was standing at the bus stop when the van came past."},
                  {"account_id": "Witness B", "text": "I was sitting in my car at the lights when the van came past."}],
     "no_flag": [("standing at the bus stop", "sitting in my car")]},
    {"name": "first_person_shots_disagree", "type": "first_person_control",
     "accounts": [{"account_id": "Witness A", "text": "I heard two shots from the alley."},
                  {"account_id": "Witness B", "text": "I heard three shots from the alley."}],
     "flag": [("two shots", "three shots")]},
    {"name": "first_person_saw_vs_denial", "type": "first_person_control",
     "accounts": [{"account_id": "Witness A", "text": "I saw the driver stop at the crossing."},
                  {"account_id": "Witness B", "text": "The driver did not stop at the crossing."}],
     "flag": [("saw the driver stop", "did not stop")]},
    {"name": "plain_conflicts", "type": "control",
     "accounts": [{"account_id": "Witness A", "text": "The light was red when the blue car went through. "
                                                      "The van was white."},
                  {"account_id": "Witness B", "text": "The light was green when the blue car went through. "
                                                      "The van was black."}],
     "flag": [("light was red", "light was green"), ("van was white", "van was black")]},
    {"name": "abbreviations_and_times", "type": "splitter",
     "accounts": [{"account_id": "Witness C", "text": "It was approx. 10 p.m. when it started. "
                                                      "The van was parked on St. James Street. "
                                                      "Dr. Patel arrived first."},
                  {"account_id": "Witness D", "text": "It was around 6 p.m. when it started. "
                                                      "The van was parked on Kings Road. "
                                                      "Dr. Patel arrived last."}],
     "claims_include": ["10 p.m.", "St. James Street", "Dr. Patel arrived first"],
     "no_fragment_claims": True,
     "flag": [("10 p.m.", "6 p.m."), ("arrived first", "arrived last")]},
    {"name": "abbreviations_misc", "type": "splitter",
     "accounts": [{"account_id": "Witness E", "text": "Mr. Hale left at 9 a.m. on Monday. "
                                                      "He drove approx. 3.5 miles to the depot."},
                  {"account_id": "Witness F", "text": "Mr. Hale was at home all of Monday morning."}],
     "claims_include": ["Mr. Hale left at 9 a.m.", "3.5 miles"],
     "no_fragment_claims": True,
     "flag": [("Hale left at 9 a.m.", "at home all of Monday morning")]},
    {"name": "long_run_on_statement", "type": "truncation",
     "accounts": [{"account_id": "Witness A", "text":
                   "I was walking my dog along the canal path at the time and I heard a loud bang which I "
                   "first thought was a firework, and I turned around towards the junction near the bakery "
                   "and the pharmacy and I saw that the blue car had hit the cyclist, and the cyclist was "
                   "lying on the ground near the kerb with his bike under him, and a woman in a yellow coat "
                   "ran over to help him and somebody else was shouting for an ambulance, and the driver of "
                   "the blue car got out and looked at the bike for a few seconds, and then he got back into "
                   "the car and drove off towards the bridge."},
                  {"account_id": "Witness B", "text": "The driver stayed with the cyclist until the police arrived."}],
     "claims_include": ["drove off towards the bridge"],
     "flag": [("drove off towards the bridge", "stayed with the cyclist")]},
    {"name": "two_long_accounts_differ_at_end", "type": "truncation",
     "accounts": [{"account_id": "Witness A", "text": _LONG_BODY + "and then he got back into the car and drove "
                                                                  "off towards the bridge without stopping."},
                  {"account_id": "Witness B", "text": _LONG_BODY + "and then he stayed next to the cyclist "
                                                                  "until the police arrived."}],
     "flag": [("drove off towards the bridge", "stayed next to the cyclist")]},
    {"name": "long_run_on_agreement", "type": "truncation",
     "accounts": [{"account_id": "Witness A", "text":
                   "We had just come out of the cinema on the high street and were standing by the cash "
                   "machine talking about where to eat, and my friend was checking her phone, and a bus went "
                   "past quite slowly, and then there was shouting from across the road near the taxi rank, "
                   "and a man in a grey hoodie ran past us towards the station holding a black rucksack."},
                  {"account_id": "Witness B", "text": "A man in a grey hoodie ran towards the station."}],
     "claims_include": ["grey hoodie ran past us towards the station"],
     "no_flag": [("grey hoodie ran past us", "grey hoodie ran towards the station")]},
    {"name": "example_file", "type": "example",
     "accounts_file": "examples/witness_accounts.txt",
     "no_flag": [("stayed at the scene", "called an ambulance")]},
]


def _two(a: str, b: str) -> list[dict[str, str]]:
    return [{"account_id": "Witness A", "text": a}, {"account_id": "Witness B", "text": b}]


# Wider coverage (``--suite extended``). Kept apart from CASES so the core numbers quoted in the
# report stay reproducible. Several of these are EXPECTED to fail today: they mark what the
# post-NLI checks (numbers/units, attribution, epistemic, same-referent, update) would have to fix.
EXTENDED_CASES: list[dict[str, Any]] = [
    # --- real disagreements that must be flagged
    {"name": "direction_left_right", "type": "conflict",
     "accounts": _two("The car turned left onto Mill Road.", "The car turned right onto Mill Road."),
     "flag": [("turned left", "turned right")]},
    {"name": "clothing_colour", "type": "conflict",
     "accounts": _two("The man who ran off wore a red jacket.", "The man who ran off wore a blue jacket."),
     "flag": [("red jacket", "blue jacket")]},
    {"name": "event_order", "type": "conflict",
     "accounts": _two("The car hit the cyclist before the bus arrived.",
                      "The bus arrived before the car hit the cyclist."),
     "flag": [("hit the cyclist before", "arrived before")]},
    {"name": "weather", "type": "conflict",
     "accounts": _two("It was raining heavily at the time.", "The road was completely dry at the time."),
     "flag": [("raining heavily", "completely dry")]},
    {"name": "head_count", "type": "conflict",
     "accounts": _two("Two men got out of the car.", "Only one person was in the car."),
     "flag": [("Two men got out", "Only one person")]},
    {"name": "time_of_day", "type": "conflict",
     "accounts": _two("It happened at about 9:15 in the morning.", "It happened late in the evening."),
     "flag": [("9:15 in the morning", "late in the evening")]},
    {"name": "hedged_but_conflicting", "type": "conflict",
     "accounts": _two("I think the car was red.", "The car was blue."),
     "flag": [("car was red", "car was blue")]},
    {"name": "coreference_across_sentences", "type": "conflict",
     "accounts": _two("A man in a grey hoodie ran towards the station. He was carrying a black bag.",
                      "The man in the grey hoodie had nothing in his hands."),
     "flag": [("carrying a black bag", "nothing in his hands")]},
    {"name": "conflict_clustered_apart", "type": "conflict",
     "accounts": _two("The driver stayed at the scene.",
                      "He was gone before anyone could get his number plate."),
     "flag": [("stayed at the scene", "was gone before")]},
    # --- compatible statements that must NOT be flagged
    {"name": "paraphrase", "type": "compatible",
     "accounts": _two("The van was white.", "It was a white van."),
     "no_flag": [("van was white", "white van")]},
    {"name": "approximate_times", "type": "compatible",
     "accounts": _two("It happened at about 9:15 a.m.", "It happened just after 9 a.m."),
     "no_flag": [("9:15", "just after 9")]},
    {"name": "approximate_counts", "type": "compatible",
     "accounts": _two("About 20 people were waiting at the stop.", "Around 25 people were waiting at the stop."),
     "no_flag": [("About 20", "Around 25")]},
    {"name": "unit_conversion", "type": "compatible",
     "accounts": _two("The car was doing about 30 mph.", "The car was doing about 50 km/h."),
     "no_flag": [("30 mph", "50 km/h")]},
    {"name": "non_observation", "type": "compatible",
     "accounts": _two("The man had a gun in his right hand.", "I didn't see a gun."),
     "no_flag": [("had a gun", "didn't see a gun")]},
    {"name": "reported_speech", "type": "compatible",
     "accounts": _two("The driver told me he had stopped at the crossing.",
                      "The driver did not stop at the crossing."),
     "no_flag": [("told me he had stopped", "did not stop")]},
    {"name": "update_over_time", "type": "compatible",
     "accounts": _two("The man was alive when I got there.", "The man was dead when the ambulance arrived."),
     "no_flag": [("alive when", "dead when")]},
    {"name": "different_people", "type": "compatible",
     "accounts": _two("The cyclist was badly injured.", "The pedestrian was not hurt at all."),
     "no_flag": [("cyclist was badly injured", "pedestrian was not hurt")]},
    {"name": "different_groups_we", "type": "compatible",
     "accounts": _two("We were sitting at the window table.", "We were standing outside the cafe."),
     "no_flag": [("sitting at the window", "standing outside")]},
    {"name": "more_detail_not_conflict", "type": "compatible",
     "accounts": _two("A car hit the cyclist.", "A blue hatchback hit the cyclist at the junction."),
     "no_flag": [("A car hit the cyclist", "blue hatchback")]},
    # --- known structural gap: inconsistency INSIDE one account is never checked
    {"name": "self_contradiction_one_account", "type": "within_account",
     "accounts": [{"account_id": "Witness A", "text": "The light was red when the car went through. "
                                                      "Later on, the light was definitely green when the car went through."},
                  {"account_id": "Witness B", "text": "The car went through the junction quickly."}],
     "flag": [("light was red", "definitely green")]},
]
SUITES = {"core": CASES, "extended": EXTENDED_CASES, "all": CASES + EXTENDED_CASES}


def _find(claims, needle: str) -> list[int]:
    return [i for i, c in enumerate(claims) if needle.lower() in c.text.lower()
            or needle.lower() in str(c.metadata.get("raw_text", "")).lower()]


def run_case(case: dict[str, Any], cfg: dict, embedder, scorer) -> dict[str, Any]:
    from pipeline.run import (accounts_to_claims, read_accounts_file, run_pipeline,
                              small_input_overrides)
    from pipeline.config import REPO_ROOT

    accounts = case.get("accounts") or read_accounts_file(REPO_ROOT / case["accounts_file"])
    c = copy.deepcopy(cfg)
    claims = accounts_to_claims(accounts, event=case["name"], cfg=c)
    small_input_overrides(c, len(claims))
    res = run_pipeline(claims, c, embedder=embedder, scorer=scorer, name=case["name"], verbose=False)
    thr = c["disagreement"]["contradiction_threshold"]
    scored = {tuple(sorted(p)): float(res.probs[k, 2]) for k, p in enumerate(res.pairs)}
    checks = []
    for needle in case.get("claims_include", []):
        checks.append({"check": f"claim contains {needle!r}", "pass": bool(_find(res.claims, needle))})
    if case.get("no_fragment_claims"):
        stubs = [cl.text for cl in res.claims if len(cl.text.split()) < 4]
        checks.append({"check": "no fragment claims", "pass": not stubs, "fragments": stubs})
    for kind in ("flag", "no_flag"):
        for a, b in case.get(kind, []):
            best = None
            for i in _find(res.claims, a):
                for j in _find(res.claims, b):
                    if i != j and tuple(sorted((i, j))) in scored:
                        s = scored[tuple(sorted((i, j)))]
                        best = s if best is None else max(best, s)
            flagged = best is not None and best >= thr
            checks.append({"check": f"{kind}: {a!r} vs {b!r}", "p_contradiction": best,
                           "compared": best is not None,
                           "pass": flagged if kind == "flag" else not flagged})
    return {"name": case["name"], "type": case["type"],
            "claims": [{"account": cl.account_id, "text": cl.text, "nli_text": cl.metadata.get("nli_text"),
                        "cluster": int(res.labels[i])} for i, cl in enumerate(res.claims)],
            "checks": checks, "n_pass": sum(ch["pass"] for ch in checks), "n_checks": len(checks)}


def run_probes(cfg: dict, suite: str = "core") -> dict[str, Any]:
    from clustering.embed import build_embedder
    from nli_scoring.scorer import build_scorer

    embedder, scorer = build_embedder(cfg), build_scorer(cfg)
    cases = [run_case(case, cfg, embedder, scorer) for case in SUITES[suite]]
    by_type: dict[str, list[int]] = {}
    for case in cases:
        t = by_type.setdefault(case["type"], [0, 0])
        t[0] += case["n_pass"]
        t[1] += case["n_checks"]
    total = [sum(v[0] for v in by_type.values()), sum(v[1] for v in by_type.values())]
    return {"nli_model": getattr(scorer, "name", "?"),
            "settings": {"nli": cfg["nli"], "threshold": cfg["disagreement"]["contradiction_threshold"],
                         "data": {k: cfg["data"].get(k) for k in ("sentence_splitter", "max_claim_words")}},
            "n_truncated_inputs": getattr(scorer, "n_truncated", None),
            "summary": {"passed": total[0], "checks": total[1],
                        "by_type": {k: {"passed": v[0], "checks": v[1]} for k, v in by_type.items()}},
            "cases": cases}


def main(argv: Sequence[str] | None = None) -> None:
    from evaluation.experiments import results_dir, write_json

    fix_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", default="current", help="suffix of the output file")
    parser.add_argument("--suite", choices=list(SUITES), default="core",
                        help="core: the 23 checks quoted in the report; extended: wider coverage")
    parser.add_argument("--config")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="config override, e.g. --set nli.ground_speaker=true (YAML value)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    import yaml

    for item in args.set:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        cfg[section][name] = yaml.safe_load(value)
    if args.mock:
        cfg["models"]["mock"] = True
    if args.finetuned:
        from pipeline.config import apply_finetuned

        apply_finetuned(cfg)
    out = run_probes(cfg, args.suite)
    out["suite"] = args.suite
    prefix = "witness_probes" if args.suite == "core" else f"witness_probes_{args.suite}"
    path = results_dir(cfg) / f"{prefix}_{args.tag}.json"
    write_json(out, path)
    s = out["summary"]
    print(f"[probes] {s['passed']}/{s['checks']} checks pass; by type: "
          + ", ".join(f"{k} {v['passed']}/{v['checks']}" for k, v in s["by_type"].items()))
    for case in out["cases"]:
        for ch in case["checks"]:
            if not ch["pass"]:
                print(f"  FAIL {case['name']}: {ch['check']} (p={ch.get('p_contradiction')})")
    print(f"[probes] wrote {path}")


if __name__ == "__main__":
    main()
