"""End-to-end pipeline: accounts -> claims -> clusters -> in-cluster NLI -> disagreement report.

CLI examples (work the same in PowerShell, cmd and bash)::

    python -m pipeline.run --event ottawashooting
    python -m pipeline.run --event germanwings-crash --sample
    python -m pipeline.run --input my_accounts.json --name my_case

``--input`` takes a JSON file ``[{"account_id": "...", "text": "..."}, ...]`` or a plain
text file with one account per paragraph (blank-line separated).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from clustering.cluster import cluster_from_config
from clustering.embed import build_embedder
from clustering.labels import central_claim_per_cluster, top_terms_per_cluster
from data.cleaning import clean_text, n_tokens, split_long_claim, split_sentences
from data.schema import Claim
from nli_scoring.scorer import BaseScorer, build_scorer
from pipeline.config import (apply_finetuned, fix_console_encoding, load_config, resolve_path,
                             set_seed)
from pipeline.disagreement import cluster_pairs, disagreement_stats
from pipeline.findings import build_findings
from pipeline.grounding import ground_speaker
from pipeline.pair_checks import apply_checks, describe, enabled_checks
from pipeline.report import render_markdown


@dataclass
class PipelineResult:
    """Everything the pipeline computed, kept for evaluation and plotting."""

    claims: list[Claim]
    embeddings: np.ndarray
    labels: np.ndarray
    pairs: list[tuple[int, int]]            # scored index pairs (into ``claims``)
    probs: np.ndarray                       # [n_pairs, 3] (entailment, neutral, contradiction),
                                            # after the post-NLI checks (set-aside pairs -> neutral)
    report: dict[str, Any] = field(default_factory=dict)
    raw_probs: np.ndarray | None = None     # the NLI model's own probabilities, before the checks
    set_aside: dict[int, str] = field(default_factory=dict)  # pair index -> check name


def accounts_to_claims(accounts: Sequence[dict[str, str]], event: str = "custom",
                       min_tokens: int = 3, cfg: dict | None = None) -> list[Claim]:
    """Split free-text accounts into sentence-level claims.

    ``accounts`` is a list of ``{"account_id": ..., "text": ...}``; every sentence of an
    account becomes one ``Claim`` attributed to that account. ``cfg`` (optional) supplies
    the text-layer settings in ``data`` and ``nli``.
    """
    data_cfg = (cfg or {}).get("data", {})
    nli_cfg = (cfg or {}).get("nli", {})
    split_mode = data_cfg.get("sentence_splitter", "abbrev")
    max_words = data_cfg.get("max_claim_words")
    ground = bool(nli_cfg.get("ground_speaker", False))
    claims: list[Claim] = []
    for a_idx, account in enumerate(accounts):
        account_id = str(account.get("account_id") or f"account_{a_idx + 1}")
        pieces = [(s_idx, sentence, clause)
                  for s_idx, sentence in enumerate(split_sentences(account.get("text", ""), split_mode))
                  for clause in split_long_claim(sentence, max_words)]
        for c_idx, (s_idx, sentence, clause) in enumerate(pieces):
            text = clean_text(clause)
            if n_tokens(text) < min_tokens:
                continue
            metadata = {"is_source": True, "sentence_index": s_idx, "raw_text": sentence}
            if clause != sentence:
                metadata["clause_of_sentence"] = True
            if ground:
                metadata["nli_text"] = ground_speaker(text, account_id)
            claims.append(Claim(
                claim_id=f"{event}:{account_id}:{c_idx}", event=event, thread_id=account_id,
                account_id=account_id, text=text, metadata=metadata,
            ))
    return claims


def nli_text(claim: Claim) -> str:
    """The text the NLI model sees: speaker-grounded when available, else the claim text."""
    return claim.metadata.get("nli_text") or claim.text


def read_accounts_file(path: str | Path) -> list[dict[str, str]]:
    """Read accounts from ``.json`` (list of dicts) or text (one account per paragraph)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    return parse_accounts_text(raw)


def parse_accounts_text(raw: str) -> list[dict[str, str]]:
    """One account per paragraph (blank-line separated).

    A paragraph may start with a short name and a colon (``Witness A: ...``); that name
    becomes the ``account_id``. Otherwise accounts are named ``account_1``, ``account_2``...
    """
    accounts: list[dict[str, str]] = []
    blocks = [b.strip() for b in raw.replace("\r\n", "\n").split("\n\n") if b.strip()]
    for i, block in enumerate(blocks):
        # a name is short, starts with a letter and its colon is followed by whitespace
        # (so "It was 5:30 when..." is not read as an account called "It was 5")
        match = re.match(r"^([A-Za-z][\w '\-]{0,39}):\s+(\S.*)$", block, flags=re.DOTALL)
        if match:
            accounts.append({"account_id": match.group(1).strip(), "text": match.group(2).strip()})
        else:
            accounts.append({"account_id": f"account_{i + 1}", "text": block})
    return accounts


def small_input_overrides(cfg: dict, n_claims: int, distance_threshold: float = 0.6) -> dict:
    """Loosen clustering for a handful of pasted sentences (in place; returns ``cfg``).

    The defaults are set for hundreds of tweets per event. With ten sentences, density
    clustering finds little, so pasted input uses a looser agglomerative cut and small
    minimum cluster sizes.
    """
    c = cfg["clustering"]
    c["algorithm"] = "agglomerative"
    c["agglomerative"] = dict(c.get("agglomerative", {}), distance_threshold=distance_threshold)
    c["kmeans"] = {"k_min": 2, "k_max": max(2, n_claims // 2), "k_step": 1}
    c["hdbscan"] = {"min_cluster_size": 2, "min_samples": 1, "umap_dim": None}
    c["noise_assign_min_sim"] = 0.4
    return cfg


def _claim_view(claim: Claim) -> dict[str, Any]:
    return {"claim_id": claim.claim_id, "account_id": claim.account_id,
            "thread_id": claim.thread_id, "timestamp": claim.timestamp, "text": claim.text}


def run_pipeline(claims: Sequence[Claim], cfg: dict, embedder=None,
                 scorer: BaseScorer | None = None, algorithm: str | None = None,
                 all_pairs: bool = False, name: str | None = None,
                 verbose: bool = True) -> PipelineResult:
    """Cluster ``claims`` by sub-event, score in-cluster pairs with NLI, build the report.

    Args:
        claims: claims about ONE event.
        cfg: loaded configuration.
        embedder / scorer: reuse already-loaded models (otherwise built from the config).
        algorithm: override ``cfg['clustering']['algorithm']``.
        all_pairs: baseline mode - skip clustering and treat the event as one cluster.
    """
    claims = list(claims)
    set_seed(cfg["seed"])
    embedder = embedder or build_embedder(cfg)
    scorer = scorer or build_scorer(cfg)
    dis = cfg["disagreement"]
    algo = "none (all pairs)" if all_pairs else (algorithm or cfg["clustering"]["algorithm"])
    texts = [c.text for c in claims]

    emb = embedder.encode(texts) if claims else np.zeros((0, 1), dtype=np.float32)
    if all_pairs:
        labels = np.zeros(len(claims), dtype=int)
    else:
        labels = cluster_from_config(emb, cfg, algorithm)
    cluster_ids = sorted(int(c) for c in set(labels) if c >= 0)

    # 1) collect the pairs of every cluster, 2) score them all in one batched call
    per_cluster: dict[int, list[tuple[int, int]]] = {}
    for cid in cluster_ids:
        members = np.where(labels == cid)[0].tolist()
        if len(members) < cfg["clustering"]["min_cluster_size_for_pairs"]:
            per_cluster[cid] = []
            continue
        per_cluster[cid] = cluster_pairs(
            members, claims, None if all_pairs else dis["max_pairs_per_cluster"],
            dis["cross_account_only"], cfg["seed"] + cid)
    pairs = [p for cid in cluster_ids for p in per_cluster[cid]]
    if verbose:
        n = len(claims)
        print(f"[pipeline] {name or 'event'}: {n} claims, {len(cluster_ids)} clusters "
              f"({int((labels < 0).sum())} unclustered), {len(pairs)} pairs to score "
              f"(all-pairs would be {n * (n - 1) // 2})", flush=True)
    nli_texts = [nli_text(c) for c in claims]
    raw_probs = scorer.score_pairs([(nli_texts[i], nli_texts[j]) for i, j in pairs])
    # post-NLI checks: set aside flagged pairs that match a known non-contradiction pattern
    checks = enabled_checks(cfg)
    probs, set_aside = apply_checks(
        [claims[i].text for i, _ in pairs], [claims[j].text for _, j in pairs],
        [claims[i].account_id for i, _ in pairs], [claims[j].account_id for _, j in pairs],
        raw_probs, dis["contradiction_threshold"], checks)

    terms = top_terms_per_cluster(texts, labels, cfg["clustering"]["label_top_terms"])
    central = central_claim_per_cluster(emb, labels) if claims else {}
    clusters_out: list[dict[str, Any]] = []
    offset = 0
    for cid in cluster_ids:
        cpairs = per_cluster[cid]
        cprobs = probs[offset:offset + len(cpairs)]
        set_aside_here = [
            {"claim_a": _claim_view(claims[cpairs[k - offset][0]]),
             "claim_b": _claim_view(claims[cpairs[k - offset][1]]),
             "p_contradiction": float(raw_probs[k, 2]), **describe(set_aside[k])}
            for k in range(offset, offset + len(cpairs)) if k in set_aside]
        offset += len(cpairs)
        members = np.where(labels == cid)[0].tolist()
        stats = disagreement_stats(cprobs, dis["contradiction_threshold"],
                                   dis["score_weight_fraction"], dis["min_contradicting_pairs"])
        order = np.argsort(-cprobs[:, 2]) if len(cpairs) else []
        contradicting = [
            {"claim_a": _claim_view(claims[cpairs[k][0]]),
             "claim_b": _claim_view(claims[cpairs[k][1]]),
             "p_contradiction": float(cprobs[k, 2]), "p_entailment": float(cprobs[k, 0]),
             "p_neutral": float(cprobs[k, 1])}
            for k in order[: dis["top_pairs_in_report"]]
            if cprobs[k, 2] >= dis["contradiction_threshold"]
        ]
        clusters_out.append({
            "cluster_id": cid,
            "label_terms": terms.get(cid, []),
            "central_claim": claims[central[cid]].text if cid in central else None,
            "size": len(members),
            "n_accounts": len({claims[i].account_id for i in members}),
            "claim_ids": [claims[i].claim_id for i in members],
            **stats,
            "contradicting_pairs": contradicting,
            "set_aside_pairs": set_aside_here,
        })
    clusters_out.sort(key=lambda c: (-int(c["flagged"]), -c["disagreement_score"], -c["size"]))

    report = {
        "name": name or (claims[0].event if claims else "empty"),
        "disclaimer": ("This report surfaces where accounts disagree. It does not decide "
                       "which account is correct and assigns no fault."),
        "config": {
            "embedding_model": getattr(embedder, "name", "?"),
            "nli_model": getattr(scorer, "name", "?"),
            "clustering_algorithm": algo,
            "contradiction_threshold": dis["contradiction_threshold"],
            "min_contradicting_pairs": dis["min_contradicting_pairs"],
            "max_pairs_per_cluster": dis["max_pairs_per_cluster"],
            "nli_aggregation": cfg["nli"]["aggregation"],
            "nli_max_length": cfg["nli"].get("max_length"),
            "speaker_grounding": any("nli_text" in c.metadata for c in claims),
            "post_nli_checks": checks,
            "seed": cfg["seed"],
        },
        "summary": {
            "n_claims": len(claims),
            "n_accounts": len({c.account_id for c in claims}),
            "n_clusters": len(cluster_ids),
            "n_unclustered_claims": int((labels < 0).sum()),
            "n_pairs_scored": len(pairs),
            "n_pairs_all_pairs": len(claims) * (len(claims) - 1) // 2,
            "n_flagged_clusters": sum(c["flagged"] for c in clusters_out),
            "n_pairs_set_aside": len(set_aside),
            "set_aside_by_check": {name: sum(v == name for v in set_aside.values())
                                   for name in sorted(set(set_aside.values()))},
        },
        "findings": build_findings(claims, labels, clusters_out, pairs, probs, scorer, cfg),
        "clusters": clusters_out,
    }
    return PipelineResult(claims, emb, labels, pairs, probs, report, raw_probs, set_aside)


def save_report(result: PipelineResult, out_dir: str | Path, name: str) -> tuple[Path, Path]:
    """Write ``<name>_report.json`` and ``<name>_report.md``; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{name}_report.json", out_dir / f"{name}_report.md"
    json_path.write_text(json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(result.report), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> None:
    fix_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--event", help="PHEME event name, e.g. ottawashooting")
    src.add_argument("--input", help="JSON or text file with free-text accounts")
    parser.add_argument("--name", help="report name (default: event name / input file stem)")
    parser.add_argument("--config", help="path to a YAML config (default: configs/default.yaml)")
    parser.add_argument("--sample", action="store_true", help="small-subset mode (minutes on CPU)")
    parser.add_argument("--mock", action="store_true", help="use mock models (no downloads)")
    parser.add_argument("--finetuned", action="store_true",
                        help="use the fine-tuned NLI checkpoint (finetune.checkpoint)")
    parser.add_argument("--algorithm", help="override the clustering algorithm")
    parser.add_argument("--threshold", type=float, help="override the contradiction threshold")
    parser.add_argument("--rumours-only", action="store_true",
                        help="PHEME: only threads annotated as rumours")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, sample=args.sample)
    if args.mock:
        cfg["models"]["mock"] = True
    if args.finetuned and not cfg["finetune"].get("active"):
        apply_finetuned(cfg)
    if args.threshold is not None:
        cfg["disagreement"]["contradiction_threshold"] = args.threshold

    if args.event:
        from data.pheme import load_event_from_config

        claims = load_event_from_config(cfg, args.event, rumours_only=args.rumours_only)
        name = args.name or args.event
    else:
        claims = accounts_to_claims(read_accounts_file(args.input),
                                    event=args.name or Path(args.input).stem, cfg=cfg)
        name = args.name or Path(args.input).stem
        small_input_overrides(cfg, len(claims))

    result = run_pipeline(claims, cfg, algorithm=args.algorithm, name=name)
    json_path, md_path = save_report(result, resolve_path(cfg, "output_dir") / "reports", name)
    s = result.report["summary"]
    print(f"[pipeline] flagged {s['n_flagged_clusters']} of {s['n_clusters']} clusters")
    print(f"[pipeline] wrote {json_path}\n[pipeline] wrote {md_path}")


if __name__ == "__main__":
    main()
