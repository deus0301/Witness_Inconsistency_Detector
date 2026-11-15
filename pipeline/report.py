"""Human-readable (Markdown) rendering of a disagreement report."""
from __future__ import annotations

from typing import Any

from pipeline.findings import render_findings_markdown


def _quote(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


def render_markdown(report: dict[str, Any], max_clusters: int = 50) -> str:
    """Render the structured report as Markdown. Purely extractive: quotes, counts, scores."""
    s, cfg = report["summary"], report["config"]
    lines = [
        f"# Disagreement report: {report['name']}",
        "",
        f"> {report['disclaimer']}",
        "",
        "## Summary",
        "",
        f"- Claims: **{s['n_claims']}** from **{s['n_accounts']}** accounts",
        f"- Sub-event clusters: **{s['n_clusters']}** ({s['n_unclustered_claims']} claims left unclustered)",
        f"- Claim pairs scored with NLI: **{s['n_pairs_scored']}** "
        f"(comparing every pair would need {s['n_pairs_all_pairs']})",
        f"- Clusters flagged for internal disagreement: **{s['n_flagged_clusters']}**",
        "",
        f"Settings: embeddings `{cfg['embedding_model']}`, clustering `{cfg['clustering_algorithm']}`, "
        f"NLI `{cfg['nli_model']}`, contradiction threshold {cfg['contradiction_threshold']}.",
        "",
    ]
    if report.get("findings"):
        lines += render_findings_markdown(report["findings"])
    lines += [
        "## Flagged clusters",
        "",
    ]
    flagged = [c for c in report["clusters"] if c["flagged"]]
    if not flagged:
        lines.append("_No cluster reached the contradiction threshold._")
    for cluster in flagged[:max_clusters]:
        label = ", ".join(cluster["label_terms"]) or "(no label terms)"
        lines += [
            f"### Cluster {cluster['cluster_id']}: {label}",
            "",
            f"- Most central claim: \"{_quote(cluster['central_claim'] or '')}\"",
            f"- {cluster['size']} claims from {cluster['n_accounts']} accounts; "
            f"{cluster['n_contradicting_pairs']} of {cluster['n_pairs_scored']} scored pairs contradict "
            f"({cluster['contradiction_fraction']:.1%})",
            f"- Disagreement score: **{cluster['disagreement_score']:.3f}** "
            f"(max contradiction confidence {cluster['max_contradiction']:.3f})",
            "",
            "| # | Account A | Claim A | Account B | Claim B | P(contradiction) |",
            "|---|---|---|---|---|---|",
        ]
        for k, pair in enumerate(cluster["contradicting_pairs"], 1):
            a, b = pair["claim_a"], pair["claim_b"]
            lines.append(f"| {k} | {_quote(a['account_id'])} | {_quote(a['text'])} | "
                         f"{_quote(b['account_id'])} | {_quote(b['text'])} | {pair['p_contradiction']:.3f} |")
        lines.append("")
    if len(flagged) > max_clusters:
        lines.append(f"_{len(flagged) - max_clusters} more flagged clusters are in the JSON report._")
    lines += ["", "## All clusters", "",
              "| Cluster | Label | Claims | Accounts | Pairs scored | Contradicting | Score | Flagged |",
              "|---|---|---|---|---|---|---|---|"]
    for cluster in report["clusters"]:
        lines.append(
            f"| {cluster['cluster_id']} | {_quote(', '.join(cluster['label_terms']))} | {cluster['size']} | "
            f"{cluster['n_accounts']} | {cluster['n_pairs_scored']} | {cluster['n_contradicting_pairs']} | "
            f"{cluster['disagreement_score']:.3f} | {'yes' if cluster['flagged'] else 'no'} |")
    return "\n".join(lines) + "\n"
