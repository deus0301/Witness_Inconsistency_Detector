"""Findings summary: what the accounts agree on, where they conflict, what only one says.

Built only from the statements themselves and the NLI scores. It is extractive (every line
is a quote plus counts) and it describes the ACCOUNTS, not the world:

* ``corroborated``  - a statement that is not part of any flagged conflict and that at least
  one other account also asserts (that account's claim entails it). This means "the
  accounts agree", not "this is true". It can sit inside a topic that is otherwise disputed.
* ``disputed``      - a topic cluster that was flagged: at least one pair of accounts conflicts.
* ``single_source`` - a statement that no other account restates and none contradicts.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from data.schema import Claim
from nli_scoring.scorer import BaseScorer

CAVEATS = [
    "Agreement between accounts is not proof: accounts can repeat the same mistake or the same source.",
    "A flagged conflict is a prompt to check, not a finding. In our audit of tweets about 6 in 10 "
    "flagged pairs were false alarms.",
    "Two accounts can differ because the situation changed between them; time is not modelled.",
    "A claim made by a single account is neither supported nor contradicted by the others.",
    "Only claims grouped into the same sub-topic are compared, and very large groups are sampled, "
    "so some conflicts can be missed.",
    "Nothing here says which account is correct or assigns fault.",
]


def _view(claim: Claim) -> dict[str, str]:
    return {"account_id": claim.account_id, "text": claim.text}


def build_findings(claims: Sequence[Claim], labels: np.ndarray, clusters: list[dict[str, Any]],
                   pairs: Sequence[tuple[int, int]], probs: np.ndarray, scorer: BaseScorer,
                   cfg: dict) -> dict[str, Any]:
    """Summarise the pipeline output into corroborated / disputed / single-source statements.

    Works on the pairs the pipeline already scored. A claim is *contested* if it is in a pair
    at or above the contradiction threshold. An uncontested claim is *corroborated* when
    another account has a claim that entails it (probability >= ``findings.entailment_threshold``).
    Overlapping corroborated statements are merged greedily, largest support first.
    """
    ent_thr = cfg.get("findings", {}).get("entailment_threshold", 0.8)
    con_thr = cfg["disagreement"]["contradiction_threshold"]
    topic = {c["cluster_id"]: ", ".join(c["label_terms"]) or f"cluster {c['cluster_id']}" for c in clusters}
    flagged = {c["cluster_id"] for c in clusters if c["flagged"]}

    contested: set[int] = set()
    for (i, j), p in zip(pairs, probs):
        if p[2] >= con_thr:
            contested.update((i, j))

    # supporters[s] = claims of OTHER accounts that entail statement s (directional, cached)
    supporters: dict[int, set[int]] = defaultdict(set)
    if len(pairs):
        texts = [c.metadata.get("nli_text") or c.text for c in claims]  # same text NLI scored
        fwd = scorer.score_directional([(texts[i], texts[j]) for i, j in pairs])   # i entails j ?
        back = scorer.score_directional([(texts[j], texts[i]) for i, j in pairs])  # j entails i ?
        for (i, j), a, b in zip(pairs, fwd, back):
            if claims[i].account_id == claims[j].account_id:
                continue
            if a[0] >= ent_thr:
                supporters[j].add(i)
            if b[0] >= ent_thr:
                supporters[i].add(j)

    corroborated: list[dict[str, Any]] = []
    covered: set[int] = set()
    candidates = sorted((s for s in supporters if s not in contested),
                        key=lambda s: (-len({claims[m].account_id for m in supporters[s]}), s))
    for s in candidates:
        if s in covered:
            continue
        backers = sorted({claims[m].account_id for m in supporters[s]} - {claims[s].account_id})
        if not backers:
            continue
        covered.add(s)
        covered.update(supporters[s])
        cid = int(labels[s])
        corroborated.append({
            "statement": claims[s].text, "stated_by": claims[s].account_id,
            "restated_by": backers, "n_accounts_stating": 1 + len(backers),
            "cluster_id": cid, "topic": topic.get(cid, ""),
            "topic_also_disputed": cid in flagged,
        })

    disputed: list[dict[str, Any]] = []
    for cluster in clusters:
        if not cluster["flagged"]:
            continue
        top = cluster["contradicting_pairs"][0] if cluster["contradicting_pairs"] else None
        disputed.append({
            "cluster_id": cluster["cluster_id"], "topic": topic[cluster["cluster_id"]],
            "n_accounts": cluster["n_accounts"],
            "n_contradicting_pairs": cluster["n_contradicting_pairs"],
            "n_pairs_scored": cluster["n_pairs_scored"],
            "disagreement_score": cluster["disagreement_score"],
            "example": None if top is None else {
                "side_a": {"account_id": top["claim_a"]["account_id"], "text": top["claim_a"]["text"]},
                "side_b": {"account_id": top["claim_b"]["account_id"], "text": top["claim_b"]["text"]},
                "p_contradiction": top["p_contradiction"]},
        })

    single = [_view(claims[i]) for i in range(len(claims)) if i not in contested and i not in covered]

    corroborated.sort(key=lambda e: -e["n_accounts_stating"])
    disputed.sort(key=lambda e: -e["disagreement_score"])
    return {
        "counts": {"corroborated": len(corroborated), "disputed": len(disputed),
                   "single_source": len(single), "contested_claims": len(contested)},
        "corroborated": corroborated, "disputed": disputed, "single_source": single,
        "caveats": CAVEATS,
        "settings": {"entailment_threshold": ent_thr, "contradiction_threshold": con_thr},
    }


def render_findings_markdown(findings: dict[str, Any], max_items: int = 15) -> list[str]:
    """Markdown lines for the findings section of a report."""
    c = findings["counts"]
    lines = [
        "## Findings summary",
        "",
        "Built only from the statements. \"Agree\" means the accounts say the same thing, "
        "not that it is true.",
        "",
        f"- Statements that more than one account makes, and none disputes: **{c['corroborated']}**",
        f"- Topics where accounts conflict: **{c['disputed']}** "
        f"({c['contested_claims']} statements are part of a conflict)",
        f"- Statements only one account makes (not confirmed, not contradicted): **{c['single_source']}**",
        "",
        "### What the accounts agree on",
        "",
    ]
    if not findings["corroborated"]:
        lines.append("_No undisputed statement was restated by a second account._")
    for e in findings["corroborated"][:max_items]:
        who = ", ".join([e["stated_by"]] + e["restated_by"][:4])
        more = f" and {len(e['restated_by']) - 4} more" if len(e["restated_by"]) > 4 else ""
        note = "; other parts of this topic are disputed" if e["topic_also_disputed"] else ""
        lines.append(f"- \"{e['statement']}\" ({e['n_accounts_stating']} accounts: {who}{more}{note})")
    if len(findings["corroborated"]) > max_items:
        lines.append(f"- _{len(findings['corroborated']) - max_items} more in the JSON report._")
    lines += ["", "### Where the accounts conflict", ""]
    if not findings["disputed"]:
        lines.append("_No conflict reached the threshold._")
    for e in findings["disputed"][:max_items]:
        lines.append(f"- **{e['topic']}** ({e['n_contradicting_pairs']} conflicting pairs among "
                     f"{e['n_accounts']} accounts)")
        if e["example"]:
            a, b = e["example"]["side_a"], e["example"]["side_b"]
            lines.append(f"  - {a['account_id']}: \"{a['text']}\"")
            lines.append(f"  - {b['account_id']}: \"{b['text']}\"")
    if len(findings["disputed"]) > max_items:
        lines.append(f"- _{len(findings['disputed']) - max_items} more below and in the JSON report._")
    lines += ["", "### Said by one account only (not confirmed, not contradicted)", ""]
    if not findings["single_source"]:
        lines.append("_None._")
    for e in findings["single_source"][:max_items]:
        lines.append(f"- {e['account_id']}: \"{e['text']}\"")
    if len(findings["single_source"]) > max_items:
        lines.append(f"- _{len(findings['single_source']) - max_items} more in the JSON report._")
    lines += ["", "### What this summary cannot tell you", ""]
    lines += [f"- {caveat}" for caveat in findings["caveats"]]
    lines.append("")
    return lines
