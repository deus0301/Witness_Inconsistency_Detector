"""All experiments: NLI sanity checks, clustering comparison, full-pipeline evaluation,
baselines, threshold sweeps, error-analysis candidates and figures.

Run via ``python run_all.py`` or ``python -m evaluation.experiments``. Every number is
written to ``<results_dir>/*.json``; nothing in the report is typed by hand.

All PHEME-based scores use PROXY labels (see ``data/proxy_labels.py``), never NLI gold.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from clustering.cluster import ALGORITHMS, cluster_from_config
from clustering.embed import build_embedder
from clustering.evaluate import extrinsic_metrics, intrinsic_metrics
from data.multinli import load_multinli
from data.pheme import load_event_from_config
from data.pheme_rte import EVENT_FILES as RTE_EVENTS, load_pheme_rte
from data.proxy_labels import (build_proxy_pairs, cluster_proxy_contested,
                               cluster_proxy_positive, is_debunk_pair, pair_proxy_label)
from data.schema import NLI_LABELS, Claim
from nli_scoring.scorer import BaseScorer, aggregate_directions, build_scorer
from pipeline.config import (REPO_ROOT, fix_console_encoding, load_config, resolve_path,
                             set_seed)
from pipeline.disagreement import disagreement_stats
from pipeline.run import PipelineResult, run_pipeline, save_report

STAGES = ("nli", "rte", "clustering", "pipeline", "reports")


# --------------------------------------------------------------------------- helpers
def results_dir(cfg: dict) -> Path:
    """Full runs write to ``reports/results`` (versioned); sample runs to ``outputs/sample``."""
    path = (resolve_path(cfg, "output_dir") / "results") if cfg.get("sample_mode") \
        else REPO_ROOT / "reports" / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                           labels: Sequence[str] = NLI_LABELS) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class precision/recall/F1 and the confusion matrix."""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(support[i])}
                      for i, lab in enumerate(labels)},
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
    }


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def load_eval_claims(cfg: dict) -> dict[str, list[Claim]]:
    """Evaluation set: rumour source tweets of the main events (they carry story + veracity)."""
    out = {}
    for event in cfg["events"]["main"]:
        out[event] = load_event_from_config(cfg, event, rumours_only=True)
        print(f"[data] {event}: {len(out[event])} rumour source claims", flush=True)
    return out


# --------------------------------------------------------------------------- NLI
def eval_nli_mnli(cfg: dict, scorer: BaseScorer) -> dict[str, Any]:
    """Sanity check of the scorer on MultiNLI validation matched + mismatched (directional)."""
    out: dict[str, Any] = {}
    for split in ("dev_matched", "dev_mismatched"):
        examples = load_multinli(resolve_path(cfg, "multinli"), split,
                                 cfg["evaluation"]["mnli_max_examples"], cfg["seed"])
        t0 = time.time()
        probs = scorer.score_pairs([(e.premise, e.hypothesis) for e in examples], symmetric=False)
        pred = [NLI_LABELS[i] for i in probs.argmax(axis=1)]
        out[split] = classification_metrics([e.label for e in examples], pred)
        out[split]["seconds"] = round(time.time() - t0, 1)
        print(f"[nli] MultiNLI {split}: n={len(examples)} acc={out[split]['accuracy']:.4f} "
              f"macro-F1={out[split]['macro_f1']:.4f}", flush=True)
    return out


def eval_nli_pheme_proxy(cfg: dict, scorer: BaseScorer,
                         event_claims: dict[str, list[Claim]]) -> dict[str, Any]:
    """Symmetric scorer vs the pair-level PHEME proxy labels (pooled and per event)."""
    y_true: list[str] = []
    y_pred: list[str] = []
    contra_probs: list[float] = []
    per_event: dict[str, Any] = {}
    for event, claims in event_claims.items():
        pairs = build_proxy_pairs(claims, cfg["evaluation"]["proxy_pairs_per_class"], cfg["seed"])
        if not pairs:
            continue
        probs = scorer.score_pairs([(a.text, b.text) for a, b, _ in pairs])
        truth = [label for _, _, label in pairs]
        pred = [NLI_LABELS[i] for i in probs.argmax(axis=1)]
        per_event[event] = classification_metrics(truth, pred)
        y_true += truth
        y_pred += pred
        contra_probs += probs[:, 2].tolist()
    pooled = classification_metrics(y_true, y_pred) if y_true else {}
    # binary view: is "contradiction-like vs rest" separable at the configured threshold?
    thr = cfg["disagreement"]["contradiction_threshold"]
    if y_true:
        truth_bin = np.array([t == "contradiction" for t in y_true])
        pred_bin = np.array(contra_probs) >= thr
        pooled["binary_contradiction_at_threshold"] = {
            "threshold": thr,
            **prf(int((truth_bin & pred_bin).sum()), int((~truth_bin & pred_bin).sum()),
                  int((truth_bin & ~pred_bin).sum()))}
        print(f"[nli] PHEME proxy pairs: n={pooled['n']} acc={pooled['accuracy']:.4f} "
              f"macro-F1={pooled['macro_f1']:.4f}", flush=True)
    return {"pooled": pooled, "per_event": per_event}


# --------------------------------------------------------------------------- PHEME RTE
def summarise_labelled(truth: Sequence[str], probs: np.ndarray, thr: float,
                       mask: np.ndarray | None = None) -> dict[str, Any]:
    """3-class metrics, contradiction P/R/F1 at ``thr``, average precision and flag rate per
    gold label for symmetric probabilities ``probs`` [n, 3] of labelled pairs."""
    from sklearn.metrics import average_precision_score

    idx = np.arange(len(truth)) if mask is None else np.where(mask)[0]
    t = [truth[i] for i in idx]
    out = classification_metrics(t, [NLI_LABELS[k] for k in probs[idx].argmax(axis=1)])
    flagged = probs[idx, 2] >= thr
    c = np.array([x == "contradiction" for x in t], dtype=bool)
    out["contradiction_at_threshold"] = {
        "threshold": thr, "n_flagged": int(flagged.sum()),
        **prf(int((c & flagged).sum()), int((~c & flagged).sum()), int((c & ~flagged).sum()))}
    out["contradiction_average_precision"] = (
        float(average_precision_score(c, probs[idx, 2])) if 0 < c.sum() < len(idx) else None)
    out["flag_rate_by_label"] = {lab: float(flagged[[x == lab for x in t]].mean())
                                 if lab in t else None for lab in NLI_LABELS}
    return out


def pipeline_on_labelled_pairs(cfg: dict, pairs_by_event: dict[str, list], probs: np.ndarray,
                               thr: float, embedder: Any, algo: str) -> dict[str, Any]:
    """What clustering does to labelled pairs: a pair is flagged by the pipeline only if both
    tweets share a cluster and its NLI contradiction probability reaches ``thr``.

    ``probs`` rows follow the pairs of ``pairs_by_event`` in dict order. Each event's rumour
    source tweets are clustered without deduplication.
    """
    nodedup = dict(cfg, data=dict(cfg["data"], dedup=False, max_claims_per_event=None))
    events = list(pairs_by_event)
    all_pairs = [p for e in events for p in pairs_by_event[e]]
    truth = np.array([p.label for p in all_pairs])
    is_contra = truth == "contradiction"
    same_cluster = np.zeros(len(all_pairs), dtype=bool)
    mapped = np.zeros(len(all_pairs), dtype=bool)
    clustering: dict[str, Any] = {}
    offset = 0
    for e in events:
        claims = load_event_from_config(nodedup, e, rumours_only=True)
        index = {c.claim_id.split(":", 1)[1]: i for i, c in enumerate(claims)}
        emb = embedder.encode([c.text for c in claims])
        labels = cluster_from_config(emb, cfg, algo)
        stories = [c.metadata.get("rumour_category") for c in claims]
        clustering[e] = {"n_reference_stories": len({s for s in stories if s is not None}),
                         **intrinsic_metrics(emb, labels), **extrinsic_metrics(labels, stories)}
        for k, pair in enumerate(pairs_by_event[e]):
            i, j = index.get(pair.tweet_a), index.get(pair.tweet_b)
            if i is None or j is None:
                continue
            mapped[offset + k] = True
            same_cluster[offset + k] = labels[i] >= 0 and labels[i] == labels[j]
        offset += len(pairs_by_event[e])
    m = mapped
    c = is_contra[m]
    nli_flag = probs[m, 2] >= thr
    pipe_flag = nli_flag & same_cluster[m]
    lab = truth[m]
    return {
        "n_pairs_mapped": int(m.sum()),
        "clustering_metrics": clustering,
        "same_cluster_rate_by_label": {l: float(same_cluster[m][lab == l].mean()) for l in NLI_LABELS},
        "flag_rate_by_label_nli_only": {l: float(nli_flag[lab == l].mean()) for l in NLI_LABELS},
        "flag_rate_by_label_pipeline": {l: float(pipe_flag[lab == l].mean()) for l in NLI_LABELS},
        "nli_only": {"n_flagged": int(nli_flag.sum()),
                     **prf(int((c & nli_flag).sum()), int((~c & nli_flag).sum()), int((c & ~nli_flag).sum()))},
        "pipeline": {"n_flagged": int(pipe_flag.sum()),
                     **prf(int((c & pipe_flag).sum()), int((~c & pipe_flag).sum()), int((c & ~pipe_flag).sum()))},
    }


def eval_rte(cfg: dict, scorer: BaseScorer, embedder: Any) -> dict[str, Any]:
    """Evaluate against the labelled tweet pairs of the PHEME RTE release.

    Part A scores every labelled pair with the NLI model (no clustering) under each way of
    combining the two directions. Part B asks what the pipeline does with the same pairs:
    are the two tweets put in the same cluster, and is the pair then flagged?
    """
    root = resolve_path(cfg, "pheme_rte")
    thr = cfg["disagreement"]["contradiction_threshold"]
    default_agg = cfg["nli"]["aggregation"]
    events = [e for e in cfg["events"]["main"] if e in RTE_EVENTS]
    pairs_by_event = {e: load_pheme_rte(root, e) for e in events}
    if cfg.get("sample_mode"):
        rng = np.random.default_rng(cfg["seed"])
        cap = 4 * cfg["evaluation"]["proxy_pairs_per_class"]
        pairs_by_event = {e: [p[i] for i in sorted(rng.choice(len(p), min(len(p), cap), replace=False))]
                          for e, p in pairs_by_event.items()}
    all_pairs = [p for e in events for p in pairs_by_event[e]]
    truth = [p.label for p in all_pairs]
    fwd = scorer.score_directional([(p.premise, p.hypothesis) for p in all_pairs])
    back = scorer.score_directional([(p.hypothesis, p.premise) for p in all_pairs])
    views = {"premise_to_hypothesis": fwd,
             **{a: aggregate_directions(fwd, back, a)
                for a in ("max_contradiction", "mean", "min_contradiction")}}
    is_contra = np.array([t == "contradiction" for t in truth])

    def summarise(probs: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
        return summarise_labelled(truth, probs, thr, mask)

    counts = {lab: truth.count(lab) for lab in NLI_LABELS}
    nli_only = {name: summarise(probs) for name, probs in views.items()}
    event_of = np.array([p.event for p in all_pairs])
    per_event = {e: summarise(views[default_agg], event_of == e) for e in events}
    d = nli_only[default_agg]
    print(f"[rte] {len(all_pairs)} labelled pairs {counts}; NLI ({default_agg}): acc={d['accuracy']:.3f} "
          f"macro-F1={d['macro_f1']:.3f}; contradiction at {thr}: "
          f"P={d['contradiction_at_threshold']['precision']:.3f} "
          f"R={d['contradiction_at_threshold']['recall']:.3f}", flush=True)

    # ---- Part B: the pipeline on the same pairs (cluster the event's rumour tweets, no dedup)
    probs = views[default_agg]
    pipeline: dict[str, Any] = {}
    for algo in ALGORITHMS:
        pipeline[algo] = pipeline_on_labelled_pairs(cfg, pairs_by_event, probs, thr, embedder, algo)
        r = pipeline[algo]
        print(f"[rte] {algo:>13}: same-cluster by label {({k: round(v, 2) for k, v in r['same_cluster_rate_by_label'].items()})}; "
              f"NLI only P={r['nli_only']['precision']:.3f} R={r['nli_only']['recall']:.3f} | "
              f"pipeline P={r['pipeline']['precision']:.3f} R={r['pipeline']['recall']:.3f}", flush=True)

    majority = max(counts.values()) / max(1, len(truth))
    return {"n_pairs": len(all_pairs), "label_counts": counts, "threshold": thr,
            "default_aggregation": default_agg, "majority_class_accuracy": majority,
            "contradiction_prevalence": float(is_contra.mean()),
            "nli_only": nli_only, "per_event": per_event,
            "pipeline_on_labelled_pairs": pipeline,
            "embedding": getattr(embedder, "name", "?")}


# --------------------------------------------------------------------------- clustering
def eval_clustering(cfg: dict, event_claims: dict[str, list[Claim]],
                    embedders: dict[str, Any]) -> list[dict[str, Any]]:
    """Intrinsic + weak extrinsic metrics for every embedder x algorithm x event."""
    rows: list[dict[str, Any]] = []
    for emb_name, embedder in embedders.items():
        for event, claims in event_claims.items():
            emb = embedder.encode([c.text for c in claims])
            stories = [c.metadata.get("rumour_category") for c in claims]
            for algo in ALGORITHMS:
                t0 = time.time()
                labels = cluster_from_config(emb, cfg, algo)
                row = {"embedding": emb_name, "algorithm": algo, "event": event,
                       "n_reference_stories": len(set(stories)),
                       **intrinsic_metrics(emb, labels), **extrinsic_metrics(labels, stories),
                       "seconds": round(time.time() - t0, 2)}
                rows.append(row)
                print(f"[cluster] {emb_name.split('/')[-1]:>18} {algo:>13} {event:>18}: "
                      f"k={row['n_clusters']:>3} noise={row['noise_fraction']:.2f} "
                      f"sil={row['silhouette'] if row['silhouette'] is None else round(row['silhouette'], 3)} "
                      f"ARI={row['ari']:.3f} NMI={row['nmi']:.3f}", flush=True)
    return rows


def summarise_clustering(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean over events for each embedder x algorithm."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["embedding"], row["algorithm"])].append(row)
    out = []
    for (emb, algo), items in groups.items():
        def mean(key: str) -> float | None:
            vals = [r[key] for r in items if r[key] is not None]
            return float(np.mean(vals)) if vals else None
        out.append({"embedding": emb, "algorithm": algo, "n_events": len(items),
                    **{k: mean(k) for k in ("n_clusters", "noise_fraction", "silhouette",
                                            "davies_bouldin", "ari", "nmi", "homogeneity",
                                            "completeness")}})
    return out


# --------------------------------------------------------------------------- pipeline
def cluster_records(result: PipelineResult, event: str) -> list[dict[str, Any]]:
    """One record per cluster with >= 2 claims: its members, scored-pair probs and proxies."""
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for k, (i, _) in enumerate(result.pairs):
        by_cluster[int(result.labels[i])].append(k)
    records = []
    for cid in sorted(int(c) for c in set(result.labels) if c >= 0):
        members = [result.claims[i] for i in np.where(result.labels == cid)[0]]
        if len(members) < 2:
            continue
        idx = by_cluster.get(cid, [])
        records.append({
            "event": event, "cluster_id": cid, "size": len(members),
            "probs": result.probs[idx] if idx else np.zeros((0, 3), dtype=np.float32),
            "positive": cluster_proxy_positive(members),
            "contested": cluster_proxy_contested(members),
        })
    return records


def cluster_level_metrics(records: list[dict], threshold: float, cfg: dict,
                          target: str = "positive") -> dict[str, Any]:
    """Precision / recall / F1 of "flagged" against a cluster-level proxy."""
    from sklearn.metrics import average_precision_score

    dis = cfg["disagreement"]
    y_true = np.array([r[target] for r in records], dtype=bool)
    stats = [disagreement_stats(r["probs"], threshold, dis["score_weight_fraction"],
                                dis["min_contradicting_pairs"]) for r in records]
    y_pred = np.array([s["flagged"] for s in stats], dtype=bool)
    rank = np.array([s["max_contradiction"] for s in stats])
    out = {"threshold": threshold, "n_clusters": int(len(records)),
           "n_proxy_positive": int(y_true.sum()), "n_flagged": int(y_pred.sum()),
           "prevalence": float(y_true.mean()) if len(records) else 0.0,
           **prf(int((y_true & y_pred).sum()), int((~y_true & y_pred).sum()),
                 int((y_true & ~y_pred).sum()))}
    out["average_precision"] = (float(average_precision_score(y_true, rank))
                                if 0 < y_true.sum() < len(records) else None)
    return out


def pair_level_metrics(result: PipelineResult, threshold: float) -> dict[str, Any]:
    """How many pairs were compared / flagged, and what the flagged pairs look like by proxy."""
    claims = result.claims
    n = len(claims)
    flagged = [k for k in range(len(result.pairs)) if result.probs[k, 2] >= threshold]
    same_story = conflict = 0
    for k in flagged:
        a, b = claims[result.pairs[k][0]], claims[result.pairs[k][1]]
        sa, sb = a.metadata.get("rumour_category"), b.metadata.get("rumour_category")
        same_story += int(sa is not None and sa == sb)
        conflict += int(is_debunk_pair(a, b) or {a.proxy_label, b.proxy_label} == {"true", "false"})
    scored = {tuple(sorted(p)): k for k, p in enumerate(result.pairs)}
    debunk = [(i, j) for i, j in combinations(range(n), 2)
              if claims[i].account_id != claims[j].account_id and is_debunk_pair(claims[i], claims[j])]
    co_clustered = sum(result.labels[i] >= 0 and result.labels[i] == result.labels[j] for i, j in debunk)
    debunk_scored = [scored[p] for p in debunk if p in scored]
    debunk_flagged = sum(result.probs[k, 2] >= threshold for k in debunk_scored)
    return {
        "n_claims": n,
        "n_pairs_all": n * (n - 1) // 2,
        "n_pairs_scored": len(result.pairs),
        "pair_reduction": 1 - len(result.pairs) / max(1, n * (n - 1) // 2),
        "n_pairs_flagged": len(flagged),
        "flagged_same_story_rate": same_story / len(flagged) if flagged else None,
        "flagged_conflict_proxy_rate": conflict / len(flagged) if flagged else None,
        "n_flagged_same_story": same_story,
        "n_flagged_conflict_proxy": conflict,
        "n_debunk_pairs": len(debunk),
        "debunk_co_clustered": int(co_clustered),
        "debunk_scored": len(debunk_scored),
        "debunk_flagged": int(debunk_flagged),
    }


def _pool_pair_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum counts over events and recompute the rates."""
    keys = ("n_claims", "n_pairs_all", "n_pairs_scored", "n_pairs_flagged", "n_flagged_same_story",
            "n_flagged_conflict_proxy", "n_debunk_pairs", "debunk_co_clustered", "debunk_scored",
            "debunk_flagged")
    total = {k: int(sum(i[k] for i in items)) for k in keys}
    f = total["n_pairs_flagged"]
    total["pair_reduction"] = 1 - total["n_pairs_scored"] / max(1, total["n_pairs_all"])
    total["flagged_same_story_rate"] = total["n_flagged_same_story"] / f if f else None
    total["flagged_conflict_proxy_rate"] = total["n_flagged_conflict_proxy"] / f if f else None
    d = total["n_debunk_pairs"]
    total["debunk_recall"] = total["debunk_flagged"] / d if d else None
    total["debunk_coverage"] = total["debunk_co_clustered"] / d if d else None
    return total


def random_baseline(records: list[dict], flag_rate: float, seed: int, reps: int = 2000) -> dict[str, float]:
    """Expected cluster-level P/R/F1 when flagging clusters at random with ``flag_rate``."""
    rng = np.random.default_rng(seed)
    y_true = np.array([r["positive"] for r in records], dtype=bool)
    scores = []
    for _ in range(reps):
        y_pred = rng.random(len(records)) < flag_rate
        scores.append(prf(int((y_true & y_pred).sum()), int((~y_true & y_pred).sum()),
                          int((y_true & ~y_pred).sum())))
    return {k: float(np.mean([s[k] for s in scores])) for k in ("precision", "recall", "f1")}


def eval_pipeline(cfg: dict, event_claims: dict[str, list[Claim]], embedders: dict[str, Any],
                  scorer: BaseScorer) -> tuple[dict[str, Any], dict[str, PipelineResult]]:
    """Ablation over embedder x algorithm, threshold sweeps, baselines."""
    thr = cfg["disagreement"]["contradiction_threshold"]
    grid = cfg["evaluation"]["threshold_grid"]
    default_key = f"{cfg['models']['embedding']}|{cfg['clustering']['algorithm']}"
    configs: dict[str, Any] = {}
    default_results: dict[str, PipelineResult] = {}
    default_records: list[dict] = []

    for emb_name, embedder in embedders.items():
        for algo in ALGORITHMS:
            key = f"{emb_name}|{algo}"
            records: list[dict] = []
            pair_items: list[dict] = []
            per_event: dict[str, Any] = {}
            t0 = time.time()
            for event, claims in event_claims.items():
                result = run_pipeline(claims, cfg, embedder, scorer, algo,
                                      name=f"{event} [{emb_name.split('/')[-1]}, {algo}]")
                recs = cluster_records(result, event)
                pm = pair_level_metrics(result, thr)
                per_event[event] = {"cluster_level": cluster_level_metrics(recs, thr, cfg),
                                    "pair_level": pm}
                records += recs
                pair_items.append(pm)
                if key == default_key:
                    default_results[event] = result
            configs[key] = {
                "embedding": emb_name, "algorithm": algo,
                "cluster_level": cluster_level_metrics(records, thr, cfg),
                "cluster_level_contested_proxy": cluster_level_metrics(records, thr, cfg, "contested"),
                "pair_level": _pool_pair_metrics(pair_items),
                "threshold_sweep": [cluster_level_metrics(records, t, cfg) for t in grid],
                "per_event": per_event,
                "seconds": round(time.time() - t0, 1),
            }
            cl = configs[key]["cluster_level"]
            print(f"[eval] {key}: clusters={cl['n_clusters']} proxy+={cl['n_proxy_positive']} "
                  f"flagged={cl['n_flagged']} P={cl['precision']:.3f} R={cl['recall']:.3f} "
                  f"F1={cl['f1']:.3f}", flush=True)
            if key == default_key:
                default_records = records

    # ---- trivial baselines on the default configuration's clusters
    baselines: dict[str, Any] = {}
    if default_records:
        d = configs[default_key]["cluster_level"]
        prevalence = d["prevalence"]
        flag_rate = d["n_flagged"] / max(1, d["n_clusters"])
        baselines = {
            "random_at_system_flag_rate": random_baseline(default_records, flag_rate, cfg["seed"]),
            "flag_all_clusters": {"precision": prevalence, "recall": 1.0,
                                  "f1": 2 * prevalence / (1 + prevalence) if prevalence else 0.0},
            "flag_no_cluster (majority class)": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }

    # ---- no-clustering baseline: all-pairs NLI on a capped subset, vs clustering on the same subset
    cap = cfg["evaluation"]["allpairs_max_claims"]
    embedder = embedders[cfg["models"]["embedding"]]
    all_items, clu_items, per_event_ap = [], [], {}
    t0 = time.time()
    for event, claims in event_claims.items():
        subset = claims
        if len(claims) > cap:
            rng = np.random.default_rng(cfg["seed"])
            subset = [claims[i] for i in sorted(rng.choice(len(claims), cap, replace=False))]
        uncapped = dict(cfg, disagreement=dict(cfg["disagreement"], max_pairs_per_cluster=None))
        clustered = run_pipeline(subset, uncapped, embedder, scorer, name=f"{event} [subset, clustered]")
        everything = run_pipeline(subset, cfg, embedder, scorer, all_pairs=True,
                                  name=f"{event} [subset, all pairs]")
        a, c = pair_level_metrics(everything, thr), pair_level_metrics(clustered, thr)
        per_event_ap[event] = {"all_pairs": a, "clustered": c}
        all_items.append(a)
        clu_items.append(c)
    no_clustering = {
        "max_claims_per_event": cap,
        "all_pairs": _pool_pair_metrics(all_items),
        "clustered_same_subset": _pool_pair_metrics(clu_items),
        "per_event": per_event_ap,
        "seconds": round(time.time() - t0, 1),
    }
    a, c = no_clustering["all_pairs"], no_clustering["clustered_same_subset"]
    print(f"[eval] all-pairs: scored={a['n_pairs_scored']} flagged={a['n_pairs_flagged']} "
          f"same-story={a['flagged_same_story_rate']} | clustered: scored={c['n_pairs_scored']} "
          f"flagged={c['n_pairs_flagged']} same-story={c['flagged_same_story_rate']}", flush=True)

    # ---- how the two NLI directions are combined (directional scores are cached, so this is cheap)
    aggregation_ablation: dict[str, Any] = {}
    original = scorer.aggregation
    for agg in cfg["evaluation"].get("aggregations", []):
        scorer.aggregation = agg
        records, items = [], []
        for event, claims in event_claims.items():
            result = run_pipeline(claims, cfg, embedder, scorer, verbose=False)
            records += cluster_records(result, event)
            items.append(pair_level_metrics(result, thr))
        aggregation_ablation[agg] = {
            "cluster_level": cluster_level_metrics(records, thr, cfg),
            "pair_level": _pool_pair_metrics(items),
            "threshold_sweep": [cluster_level_metrics(records, t, cfg) for t in grid],
        }
        cl = aggregation_ablation[agg]["cluster_level"]
        print(f"[eval] aggregation={agg}: flagged={cl['n_flagged']}/{cl['n_clusters']} "
              f"P={cl['precision']:.3f} R={cl['recall']:.3f} F1={cl['f1']:.3f}", flush=True)
    scorer.aggregation = original

    return ({"default_config": default_key, "threshold": thr, "threshold_grid": grid,
             "nli_aggregation": original, "aggregation_ablation": aggregation_ablation,
             "configs": configs, "baselines": baselines, "no_clustering_baseline": no_clustering},
            default_results)


# --------------------------------------------------------------------------- error analysis
def error_candidates(cfg: dict, results: dict[str, PipelineResult], scorer: BaseScorer) -> dict[str, Any]:
    """Representative false positives / negatives w.r.t. the PROXY, for manual commentary.

    * false positives: confidently flagged pairs of two non-turnaround tweets of the same
      TRUE story (the proxy says they should agree);
    * false negatives: debunk pairs (false-story turnaround vs an assertion of that story)
      that scored below the threshold, with whether clustering kept them together.
    """
    thr = cfg["disagreement"]["contradiction_threshold"]
    n_each = cfg["evaluation"]["error_examples"]
    fps: list[dict] = []
    fns: list[dict] = []
    flagged_all: list[dict] = []
    for event, result in results.items():
        claims = result.claims
        for k, (i, j) in enumerate(result.pairs):
            if result.probs[k, 2] >= thr:
                flagged_all.append({
                    "event": event, "same_story": claims[i].metadata.get("rumour_category")
                    == claims[j].metadata.get("rumour_category"),
                    "story_a": claims[i].metadata.get("rumour_category"),
                    "story_b": claims[j].metadata.get("rumour_category"),
                    "veracity_a": claims[i].proxy_label, "veracity_b": claims[j].proxy_label,
                    "text_a": claims[i].text, "text_b": claims[j].text,
                    "cosine_similarity": float(result.embeddings[i] @ result.embeddings[j]),
                    "p_contradiction": float(result.probs[k, 2])})
            if result.probs[k, 2] >= thr and pair_proxy_label(claims[i], claims[j]) == "entailment":
                fps.append({"event": event, "story": claims[i].metadata.get("rumour_category"),
                            "story_veracity": claims[i].proxy_label,
                            "account_a": claims[i].account_id, "text_a": claims[i].text,
                            "account_b": claims[j].account_id, "text_b": claims[j].text,
                            "p_contradiction": float(result.probs[k, 2])})
        debunk = [(i, j) for i, j in combinations(range(len(claims)), 2)
                  if claims[i].account_id != claims[j].account_id and is_debunk_pair(claims[i], claims[j])]
        if debunk:
            probs = scorer.score_pairs([(claims[i].text, claims[j].text) for i, j in debunk])
            for (i, j), p in zip(debunk, probs):
                if p[2] < thr:
                    turn, other = (i, j) if claims[i].metadata.get("is_turnaround") else (j, i)
                    fns.append({"event": event, "story": claims[i].metadata.get("rumour_category"),
                                "story_veracity": claims[i].proxy_label,
                                "debunk_account": claims[turn].account_id, "debunk_text": claims[turn].text,
                                "assertion_account": claims[other].account_id,
                                "assertion_text": claims[other].text,
                                "p_contradiction": float(p[2]), "p_entailment": float(p[0]),
                                "p_neutral": float(p[1]),
                                "same_cluster": bool(result.labels[i] >= 0
                                                     and result.labels[i] == result.labels[j])})
    rng = np.random.default_rng(cfg["seed"])

    def spread(items: list[dict], key: str, reverse: bool) -> list[dict]:
        """Half the most extreme cases, half a random draw, so examples are representative."""
        items = sorted(items, key=lambda d: d[key], reverse=reverse)
        head = items[: n_each // 2]
        rest = items[n_each // 2:]
        extra = [rest[i] for i in sorted(rng.choice(len(rest), min(len(rest), n_each - len(head)),
                                                    replace=False))] if rest else []
        return head + extra

    audit_n = min(len(flagged_all), 2 * n_each)
    audit = [flagged_all[i] for i in sorted(rng.choice(len(flagged_all), audit_n, replace=False))]         if flagged_all else []
    return {"threshold": thr, "n_flagged_pairs": len(flagged_all),
            "random_flagged_sample_for_manual_audit": audit,
            "n_false_positive_candidates": len(fps),
            "n_false_negative_candidates": len(fns),
            "false_positives": spread(fps, "p_contradiction", True),
            "false_negatives": spread(fns, "p_contradiction", False)}


# --------------------------------------------------------------------------- figures
def make_figures(cfg: dict, res_dir: Path, results: dict[str, PipelineResult] | None = None,
                 prefix: str = "") -> list[str]:
    """Create every figure whose inputs exist. Returns the written file names."""
    from evaluation import plots

    fig_dir = resolve_path(cfg, "figures_dir")
    written: list[Path] = []
    thr = cfg["disagreement"]["contradiction_threshold"]

    def read(name: str) -> Any | None:
        path = res_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    nli = read("nli_results.json")
    if nli:
        for model, block in nli.items():
            tag = model.split("/")[-1]
            for split, title in (("dev_matched", "MultiNLI validation matched"),
                                 ("dev_mismatched", "MultiNLI validation mismatched")):
                if split in block.get("multinli", {}):
                    m = block["multinli"][split]
                    written.append(plots.plot_confusion(
                        m["confusion_matrix"], m["labels"], f"{title}\n{tag}, acc {m['accuracy']:.3f}",
                        fig_dir / f"confusion_mnli_{split}_{tag}.png", ylabel="Gold label"))
            pooled = block.get("pheme_proxy", {}).get("pooled")
            if pooled:
                written.append(plots.plot_confusion(
                    pooled["confusion_matrix"], pooled["labels"],
                    f"PHEME proxy pairs (not NLI gold)\n{tag}, acc {pooled['accuracy']:.3f}",
                    fig_dir / f"confusion_pheme_proxy_{tag}.png", ylabel="Proxy label"))

    rte = read("rte_results.json")
    if rte:
        d = rte["nli_only"][rte["default_aggregation"]]
        written.append(plots.plot_confusion(
            d["confusion_matrix"], d["labels"],
            f"PHEME RTE labelled pairs, NLI only\nacc {d['accuracy']:.3f}, macro-F1 {d['macro_f1']:.3f}",
            fig_dir / "confusion_pheme_rte.png", ylabel="RTE label"))
        algo = cfg["clustering"]["algorithm"]
        if algo in rte["pipeline_on_labelled_pairs"]:
            r = rte["pipeline_on_labelled_pairs"][algo]
            names = {"entailment": "labelled\nentailment", "neutral": "labelled\nunknown",
                     "contradiction": "labelled\ncontradiction"}
            written.append(plots.plot_bars(
                [names[l] for l in NLI_LABELS],
                {"NLI on every pair": [r["flag_rate_by_label_nli_only"][l] for l in NLI_LABELS],
                 f"pipeline ({algo} clusters, then NLI)": [r["flag_rate_by_label_pipeline"][l] for l in NLI_LABELS]},
                "Share of labelled pairs flagged as contradicting", "share flagged",
                fig_dir / "rte_flag_rates.png"))

    clustering = read("clustering_results.json")
    if clustering:
        summary = clustering["summary"]
        embs = sorted({r["embedding"] for r in summary})
        for metric, label in (("ari", "ARI vs rumour story"), ("nmi", "NMI vs rumour story"),
                              ("silhouette", "Silhouette (cosine)")):
            series = {e.split("/")[-1]: [next((r[metric] or 0) for r in summary
                                              if r["embedding"] == e and r["algorithm"] == a)
                                         for a in ALGORITHMS] for e in embs}
            written.append(plots.plot_bars(list(ALGORITHMS), series,
                                           f"Clustering comparison: {label} (mean over events)",
                                           label, fig_dir / f"clustering_{metric}.png"))

    pipe = read("pipeline_results.json")
    if pipe:
        default_emb = pipe["default_config"].split("|")[0]
        curves = {}
        for key, block in pipe["configs"].items():
            if block["embedding"] == default_emb:
                sweep = block["threshold_sweep"]
                curves[block["algorithm"]] = {m: [s[m] for s in sweep] for m in ("precision", "recall", "f1")}
        written.append(plots.plot_threshold_curves(
            pipe["threshold_grid"], curves,
            f"Cluster-level detection vs contradiction threshold ({default_emb.split('/')[-1]})",
            fig_dir / "threshold_curves.png"))
        nc = pipe["no_clustering_baseline"]
        events = list(nc["per_event"])
        written.append(plots.plot_bars(
            events,
            {"all pairs (no clustering)": [nc["per_event"][e]["all_pairs"]["n_pairs_scored"] for e in events],
             "in-cluster pairs": [nc["per_event"][e]["clustered"]["n_pairs_scored"] for e in events]},
            "Claim pairs sent to the NLI model", "pairs scored (log scale)",
            fig_dir / "pairs_compared.png", log=True, fmt="{:,.0f}"))
        written.append(plots.plot_bars(
            ["flagged pairs about\nthe same story", "flagged pairs matching\nthe conflict proxy"],
            {"all pairs (no clustering)": [nc["all_pairs"]["flagged_same_story_rate"] or 0,
                                           nc["all_pairs"]["flagged_conflict_proxy_rate"] or 0],
             "in-cluster pairs": [nc["clustered_same_subset"]["flagged_same_story_rate"] or 0,
                                  nc["clustered_same_subset"]["flagged_conflict_proxy_rate"] or 0]},
            "What the flagged pairs look like (proxy precision)", "share of flagged pairs",
            fig_dir / "pair_precision.png"))

    for event, result in (results or {}).items():
        names = {c["cluster_id"]: ", ".join(c["label_terms"][:3]) for c in result.report["clusters"]}
        flagged = [c for c in result.report["clusters"] if c["flagged"]]
        written.append(plots.plot_cluster_scatter(
            result.embeddings, result.labels,
            f"{event}: sub-event clusters of {len(result.claims)} claims",
            fig_dir / f"{prefix}clusters_{event}.png", names, cfg["seed"],
            {c["cluster_id"] for c in flagged}))
        pair_index = {tuple(sorted(p)): k for k, p in enumerate(result.pairs)}
        heat = next((c for c in flagged if 6 <= c["size"] <= 24), None)
        if heat:
            members = np.where(result.labels == heat["cluster_id"])[0].tolist()
            matrix = np.full((len(members), len(members)), np.nan)
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    k = pair_index.get((members[a], members[b]))
                    if k is not None:
                        matrix[a, b] = matrix[b, a] = result.probs[k, 2]
            written.append(plots.plot_contradiction_heatmap(
                [result.claims[i].text for i in members], matrix,
                f"{event}: P(contradiction) inside cluster \"{names[heat['cluster_id']]}\"",
                fig_dir / f"{prefix}heatmap_{event}.png"))
        graph = max((c for c in flagged if c["size"] <= 80),
                    key=lambda c: c["n_contradicting_pairs"], default=None)
        if graph:
            members = np.where(result.labels == graph["cluster_id"])[0].tolist()
            local = {g: l for l, g in enumerate(members)}
            edges = [(local[i], local[j], float(result.probs[k, 2]))
                     for (i, j), k in pair_index.items()
                     if i in local and j in local and result.probs[k, 2] >= thr]
            written.append(plots.plot_contradiction_graph(
                [result.claims[i].text for i in members],
                [result.claims[i].account_id for i in members], edges,
                f"{event}: contradiction graph of cluster \"{names[graph['cluster_id']]}\" "
                f"({len(edges)} pairs at P >= {thr})",
                fig_dir / f"{prefix}graph_{event}.png", cfg["seed"]))
    return [p.name for p in written]


# --------------------------------------------------------------------------- orchestration
def run_evaluation(cfg: dict, stages: Sequence[str] = STAGES, compare_nli: bool = False) -> dict[str, Any]:
    """Run the requested stages and write JSON results + figures."""
    set_seed(cfg["seed"])
    res_dir = results_dir(cfg)
    timings: dict[str, float] = {}
    event_claims = load_eval_claims(cfg)
    scorer = build_scorer(cfg)
    default_results: dict[str, PipelineResult] = {}

    if "nli" in stages:
        t0 = time.time()
        nli_out = {scorer.name: {"multinli": eval_nli_mnli(cfg, scorer),
                                 "pheme_proxy": eval_nli_pheme_proxy(cfg, scorer, event_claims)}}
        if compare_nli and not cfg["models"].get("mock"):
            try:
                other = build_scorer(cfg, "nli_comparison")
                nli_out[other.name] = {"multinli": eval_nli_mnli(cfg, other),
                                       "pheme_proxy": eval_nli_pheme_proxy(cfg, other, event_claims)}
                other.cache.close()
                del other
                gc.collect()
            except Exception as exc:  # download/size problems must not kill the run
                print(f"[nli] comparison model skipped: {exc!r}", flush=True)
                nli_out["comparison_model_error"] = repr(exc)
        timings["nli"] = round(time.time() - t0, 1)
        write_json(nli_out, res_dir / "nli_results.json")

    embedders: dict[str, Any] = {}
    if "clustering" in stages or "pipeline" in stages or "rte" in stages:
        if cfg["models"].get("mock"):
            embedders = {cfg["models"]["embedding"]: build_embedder(cfg)}
        else:
            embedders = {cfg["models"][k]: build_embedder(cfg, k)
                         for k in ("embedding", "embedding_comparison")}

    if "rte" in stages:
        if resolve_path(cfg, "pheme_rte").is_dir():
            t0 = time.time()
            write_json(eval_rte(cfg, scorer, embedders[cfg["models"]["embedding"]]),
                       res_dir / "rte_results.json")
            timings["rte"] = round(time.time() - t0, 1)
        else:
            print(f"[rte] skipped: {resolve_path(cfg, 'pheme_rte')} not found", flush=True)

    if "clustering" in stages:
        t0 = time.time()
        rows = eval_clustering(cfg, event_claims, embedders)
        timings["clustering"] = round(time.time() - t0, 1)
        write_json({"rows": rows, "summary": summarise_clustering(rows)},
                   res_dir / "clustering_results.json")

    if "pipeline" in stages:
        t0 = time.time()
        pipe_out, default_results = eval_pipeline(cfg, event_claims, embedders, scorer)
        timings["pipeline"] = round(time.time() - t0, 1)
        write_json(pipe_out, res_dir / "pipeline_results.json")
        write_json(error_candidates(cfg, default_results, scorer), res_dir / "error_candidates.json")

    figures = make_figures(cfg, res_dir, default_results, prefix="eval_")

    if "reports" in stages:
        # the user-facing run: ALL source tweets (rumour + non-rumour) of each main event
        t0 = time.time()
        embedder = embedders.get(cfg["models"]["embedding"]) or build_embedder(cfg)
        full: dict[str, PipelineResult] = {}
        summary = {}
        for event in cfg["events"]["main"]:
            claims = load_event_from_config(cfg, event)
            full[event] = run_pipeline(claims, cfg, embedder, scorer, name=event)
            save_report(full[event], resolve_path(cfg, "output_dir") / "reports", event)
            summary[event] = full[event].report["summary"]
        timings["reports"] = round(time.time() - t0, 1)
        write_json(summary, res_dir / "event_report_summaries.json")
        figures += make_figures(cfg, Path("__none__"), full)

    meta = {"sample_mode": cfg.get("sample_mode", False), "mock_models": bool(cfg["models"].get("mock")),
            "stages": list(stages), "timings_seconds": timings, "figures": sorted(set(figures)),
            "models": cfg["models"], "events": cfg["events"]["main"],
            "n_eval_claims": {e: len(c) for e, c in event_claims.items()},
            "nli_cache": {"model_calls": scorer.n_model_calls, "cache_hits": scorer.n_cache_hits}}
    from pipeline.config import pick_device

    meta["device"] = "cpu"
    if pick_device(cfg) == "cuda":
        import torch

        meta["device"] = torch.cuda.get_device_name(0)
    write_json(meta, res_dir / "run_meta.json")
    print(f"[eval] results in {res_dir}; figures in {resolve_path(cfg, 'figures_dir')}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    fix_console_encoding()
    parser = argparse.ArgumentParser(description="Run the evaluation suite.")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--compare-nli", action="store_true",
                        help="also evaluate models.nli_comparison (large download)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, sample=args.sample)
    if args.mock:
        cfg["models"]["mock"] = True
    run_evaluation(cfg, args.stages, args.compare_nli)


if __name__ == "__main__":
    main()
