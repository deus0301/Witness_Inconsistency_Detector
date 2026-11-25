"""Evaluation validity: per-pair predictions, story-group bootstrap CIs and pair-level baselines.

Why this exists. A PHEME RTE label is a deterministic function of the pair's rumour stories:
grouping the 5,413 pairs by (event, stories) gives 85 groups and none of them mixes labels,
and the 1,378 contradiction pairs come from only 21 story pairs (one in germanwings-crash).
Pairs inside a group are therefore not independent observations, and a CI that resamples
pairs is far too narrow. Every interval here resamples whole story groups, within each event.

What it adds (``python -m evaluation.validity``, or the ``validity`` stage of ``run_all``):

* ``pair_predictions_<run>.csv.gz``: one row per labelled pair and model, with both
  directional probability triples, so any metric can be recomputed without the models.
* Story-group bootstrap CIs for contradiction P / R / F1 / AP, for the zero-shot scorer on
  every labelled pair, and for every fine-tuning setting on each held-out event and pooled.
* Paired deltas with CIs. The primary comparison is fine-tuned vs zero-shot with BOTH
  operating points chosen the same way on the validation event.
* Pair-level baselines: flag-all, random at the system's flag rate, an untuned negation /
  number / antonym cue lexicon and its union, TF-IDF cosine with a validation-tuned threshold,
  and a "story oracle" (two different stories AND cosine >= t) that reads the label structure.
  The oracle is a diagnostic of what the labels reward, never a deployable system.

Scores come from the NLI caches when they are warm (no GPU work); missing pairs are scored.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data.pheme_rte import RTEPair, load_pheme_rte, load_rte_split, remove_leakage
from nli_scoring.scorer import C, aggregate_directions, build_scorer
from pipeline.config import fix_console_encoding, load_config, resolve_path, set_seed

RESULTS_FILE = "validity_results.json"
AGGREGATIONS = ("max_contradiction", "mean", "min_contradiction")


# --------------------------------------------------------------------------- story groups
def story_key(pair: RTEPair) -> str:
    """Event plus the unordered set of rumour stories named by the pair."""
    stories = sorted(s.strip() for s in pair.stories.split(";") if s.strip())
    return pair.event + "||" + " ## ".join(stories)


def n_stories(pair: RTEPair) -> int:
    return len([s for s in pair.stories.split(";") if s.strip()])


def group_summary(pairs: Sequence[RTEPair]) -> dict[str, Any]:
    """How many independent units the labelled pairs really contain, per event."""
    out: dict[str, Any] = {}
    by_event: dict[str, list[RTEPair]] = defaultdict(list)
    for p in pairs:
        by_event[p.event].append(p)
    for event, ps in [*sorted(by_event.items()), ("all", list(pairs))]:
        labels_of: dict[str, set[str]] = defaultdict(set)
        contra_sizes: Counter = Counter()
        for p in ps:
            labels_of[story_key(p)].add(p.label)
            if p.label == "contradiction":
                contra_sizes[story_key(p)] += 1
        n_contra = sum(contra_sizes.values())
        out[event] = {
            "n_pairs": len(ps), "n_story_groups": len(labels_of),
            "n_mixed_label_groups": sum(len(v) > 1 for v in labels_of.values()),
            "n_contradiction_groups": len(contra_sizes),
            "n_contradiction_pairs": n_contra,
            "largest_contradiction_group_share": (max(contra_sizes.values()) / n_contra
                                                  if n_contra else None),
        }
    return out


# --------------------------------------------------------------------------- metrics
def _prf(y: np.ndarray, flag: np.ndarray) -> tuple[float, float, float]:
    tp = int((y & flag).sum())
    fp = int((~y & flag).sum())
    fn = int((y & ~flag).sum())
    if tp + fn == 0:  # no positives in this (re)sample: recall and F1 are undefined
        return (tp / (tp + fp) if tp + fp else 0.0), float("nan"), float("nan")
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _ap(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, score))


def point_metrics(y: np.ndarray, flag: np.ndarray | None, score: np.ndarray | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if flag is not None:
        p, r, f = _prf(y, flag)
        out.update(precision=p, recall=r, f1=f, flag_rate=float(flag.mean()))
    if score is not None:
        out["ap"] = _ap(y, score)
    return out


def group_bootstrap(y: np.ndarray, groups: Sequence[str], strata: Sequence[str],
                    systems: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
                    deltas: Sequence[tuple[str, str]] = (), n_boot: int = 2000,
                    seed: int = 42, alpha: float = 0.05) -> dict[str, Any]:
    """Point estimates and percentile CIs that resample whole story groups within each stratum.

    ``systems`` maps a name to ``(flags or None, scores or None)`` over the same rows as ``y``.
    ``deltas`` lists ``(a, b)`` pairs; their metric differences ``a - b`` are computed on the
    same resample, so the interval is a paired one. Resamples in which a metric is undefined
    (no positives drawn) are dropped for that metric, and the share kept is reported.
    """
    groups = np.asarray(groups)
    strata = np.asarray(strata)
    members: dict[str, dict[str, np.ndarray]] = {}
    for s in np.unique(strata):
        gs = np.unique(groups[strata == s])
        members[s] = {g: np.where((groups == g) & (strata == s))[0] for g in gs}
    rng = np.random.default_rng(seed)
    names = list(systems)
    keys = ("precision", "recall", "f1", "ap")
    draws: dict[str, dict[str, list[float]]] = {n: defaultdict(list) for n in names}
    ddraws: dict[str, dict[str, list[float]]] = {f"{a} - {b}": defaultdict(list) for a, b in deltas}
    for _ in range(n_boot):
        idx_parts = []
        for s, gmap in members.items():
            gnames = list(gmap)
            pick = rng.integers(0, len(gnames), len(gnames))
            idx_parts += [gmap[gnames[k]] for k in pick]
        idx = np.concatenate(idx_parts)
        yb = y[idx]
        cur: dict[str, dict[str, float]] = {}
        for n in names:
            flag, score = systems[n]
            cur[n] = point_metrics(yb, None if flag is None else flag[idx],
                                   None if score is None else score[idx])
            for k in keys:
                if k in cur[n]:
                    draws[n][k].append(cur[n][k])
        for a, b in deltas:
            for k in keys:
                if k in cur[a] and k in cur[b]:
                    ddraws[f"{a} - {b}"][k].append(cur[a][k] - cur[b][k])

    def interval(values: list[float]) -> dict[str, float | None]:
        v = np.array(values, dtype=float)
        v = v[~np.isnan(v)]
        if not len(v):
            return {"lo": None, "hi": None, "kept": 0.0}
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"lo": float(lo), "hi": float(hi), "kept": len(v) / n_boot}

    out: dict[str, Any] = {"n_boot": n_boot, "unit": "story group within event",
                           "n_groups": int(sum(len(g) for g in members.values())),
                           "systems": {}, "deltas": {}}
    for n in names:
        flag, score = systems[n]
        point = point_metrics(y, flag, score)
        out["systems"][n] = {k: {"value": point[k], **interval(draws[n][k])} for k in keys if k in point}
        if "flag_rate" in point:
            out["systems"][n]["flag_rate"] = point["flag_rate"]
    for a, b in deltas:
        pa = point_metrics(y, *systems[a])
        pb = point_metrics(y, *systems[b])
        out["deltas"][f"{a} - {b}"] = {k: {"value": pa[k] - pb[k], **interval(ddraws[f"{a} - {b}"][k])}
                                       for k in keys if k in pa and k in pb}
    return out


# --------------------------------------------------------------------------- baselines
_NEG = re.compile(r"\b(not|no|never|none|nobody|deny|denies|denied|false|fake|hoax|untrue|"
                  r"debunk\w*|rumou?r\w*|unconfirmed)\b|n't", re.IGNORECASE)
_NUM = re.compile(r"\b\d+\b")
# written once, never tuned on the labelled pairs
_ANTONYMS = [("dead", "alive"), ("killed", "injured"), ("killed", "wounded"), ("arrested", "escaped"),
             ("arrested", "at large"), ("confirmed", "denied"), ("confirmed", "unconfirmed"),
             ("one", "two"), ("released", "held"), ("true", "false"), ("shot dead", "on the loose"),
             ("captured", "on the loose"), ("inside", "outside")]


def _antonym(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return any((x in a and y in b and y not in a) or (y in a and x in b and x not in a)
               for x, y in _ANTONYMS)


def cue_flags(pairs: Sequence[RTEPair]) -> dict[str, np.ndarray]:
    """Untuned lexical contradiction cues (a trivial baseline, no learning)."""
    neg = np.array([bool(_NEG.search(p.premise)) != bool(_NEG.search(p.hypothesis)) for p in pairs])
    num = np.array([bool(na) and bool(nb) and na != nb
                    for na, nb in ((set(_NUM.findall(p.premise)), set(_NUM.findall(p.hypothesis)))
                                   for p in pairs)])
    ant = np.array([_antonym(p.premise, p.hypothesis) for p in pairs])
    return {"cue_negation": neg, "cue_number": num, "cue_antonym": ant, "cue_union": neg | num | ant}


def tfidf_cosine(pairs: Sequence[RTEPair], fit_texts: Sequence[str] | None = None) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = list(fit_texts) if fit_texts is not None else [t for p in pairs for t in (p.premise, p.hypothesis)]
    vec = TfidfVectorizer().fit(texts)
    a = vec.transform([p.premise for p in pairs])
    b = vec.transform([p.hypothesis for p in pairs])
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold with the best F1 on (validation) scores; quantile grid, ties -> higher."""
    best = (-1.0, 0.0)
    for t in np.unique(np.quantile(score, np.linspace(0.02, 0.98, 49))):
        f = _prf(y, score >= t)[2]
        f = -1.0 if np.isnan(f) else f
        if (f, t) > best:
            best = (f, float(t))
    return best[1]


def baseline_systems(test: Sequence[RTEPair], val: Sequence[RTEPair] | None,
                     flag_rate: float, seed: int) -> tuple[dict[str, tuple], dict[str, Any]]:
    """Baseline (flags, scores) on ``test``; thresholds that need tuning use ``val`` only."""
    y_val = np.array([p.label == "contradiction" for p in val]) if val else None
    texts = [t for p in [*test, *(val or [])] for t in (p.premise, p.hypothesis)]
    cos_t = tfidf_cosine(test, texts)
    two_t = np.array([n_stories(p) >= 2 for p in test])
    systems: dict[str, tuple] = {
        "flag_all": (np.ones(len(test), dtype=bool), None),
        "random_at_system_flag_rate": (np.random.default_rng(seed).random(len(test)) < flag_rate, None),
        **{k: (v, None) for k, v in cue_flags(test).items()},
    }
    info: dict[str, Any] = {}
    if val:
        cos_v = tfidf_cosine(val, texts)
        t_cos = best_threshold(y_val, cos_v)
        two_v = np.array([n_stories(p) >= 2 for p in val])
        t_orc = best_threshold(y_val, np.where(two_v, cos_v, -1.0))
        systems["tfidf_cosine_val_tuned"] = (cos_t >= t_cos, cos_t)
        systems["story_oracle_diagnostic"] = (two_t & (cos_t >= t_orc), np.where(two_t, cos_t, -1.0))
        info = {"tfidf_cosine_threshold": t_cos, "story_oracle_threshold": t_orc}
    else:
        systems["tfidf_cosine"] = (None, cos_t)
        systems["story_oracle_diagnostic"] = (None, np.where(two_t, cos_t, -1.0))
    return systems, info


# --------------------------------------------------------------------------- predictions
def write_predictions(path: Path, pairs: Sequence[RTEPair], scores: dict[str, tuple[np.ndarray, np.ndarray]],
                      extra: dict[str, Sequence[Any]] | None = None) -> None:
    """One row per pair; ``scores`` maps a model tag to (fwd, back) [n, 3] arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    cols = ["event", "pair_id", "tweet_a", "tweet_b", "label", "story_group", "premise", "hypothesis",
            *extra]
    for tag in scores:
        cols += [f"{tag}_{d}_{k}" for d in ("fwd", "back") for k in ("e", "n", "c")]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, p in enumerate(pairs):
            row = [p.event, p.pair_id, p.tweet_a, p.tweet_b, p.label, story_key(p), p.premise, p.hypothesis,
                   *(extra[k][i] for k in extra)]
            for fwd, back in scores.values():
                row += [f"{x:.6f}" for x in (*fwd[i], *back[i])]
            w.writerow(row)


def _score_both(scorer, pairs: Sequence[RTEPair]) -> tuple[np.ndarray, np.ndarray]:
    fwd = scorer.score_directional([(p.premise, p.hypothesis) for p in pairs])
    back = scorer.score_directional([(p.hypothesis, p.premise) for p in pairs])
    return fwd, back


def _release(scorer) -> None:
    """Free a scorer's model and GPU memory (callers may still hold the scorer object)."""
    import gc

    scorer.cache.close()
    if getattr(scorer, "model", None) is not None:
        scorer.model = None
        scorer.tokenizer = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- zero-shot, all pairs
def zero_shot_all_pairs(cfg: dict, scorer, res_dir: Path, n_boot: int) -> dict[str, Any]:
    """Every labelled pair with the configured (zero-shot) scorer, at the configured threshold."""
    from clustering.embed import build_embedder
    from evaluation.experiments import RTE_EVENTS, pipeline_on_labelled_pairs

    root = resolve_path(cfg, "pheme_rte")
    events = [e for e in cfg["events"]["main"] if e in RTE_EVENTS]
    by_event = {e: load_pheme_rte(root, e) for e in events}
    if cfg.get("sample_mode"):  # same subsample as eval_rte
        rng = np.random.default_rng(cfg["seed"])
        cap = 4 * cfg["evaluation"]["proxy_pairs_per_class"]
        by_event = {e: [p[i] for i in sorted(rng.choice(len(p), min(len(p), cap), replace=False))]
                    for e, p in by_event.items()}
    pairs = [p for e in events for p in by_event[e]]
    y = np.array([p.label == "contradiction" for p in pairs])
    groups = [story_key(p) for p in pairs]
    strata = [p.event for p in pairs]
    thr = cfg["disagreement"]["contradiction_threshold"]
    fwd, back = _score_both(scorer, pairs)

    systems: dict[str, tuple] = {}
    for agg in AGGREGATIONS:
        c = aggregate_directions(fwd, back, agg)[:, C]
        systems[f"nli_{agg}@{thr}"] = (c >= thr, c)
    default = f"nli_{cfg['nli']['aggregation']}@{thr}"
    # the pipeline view: flagged only if both tweets share a cluster (default clustering)
    embedder = build_embedder(cfg)
    probs = aggregate_directions(fwd, back, cfg["nli"]["aggregation"])
    view = pipeline_on_labelled_pairs(cfg, by_event, probs, thr, embedder,
                                      cfg["clustering"]["algorithm"], return_masks=True)
    mapped, same = view.pop("_mapped"), view.pop("_same_cluster")
    nli_flag = probs[:, C] >= thr
    systems["pipeline_" + cfg["clustering"]["algorithm"]] = (nli_flag & same & mapped,
                                                            np.where(same & mapped, probs[:, C], -1.0))
    flag_rate = float(systems[default][0].mean())
    base, _ = baseline_systems(pairs, None, flag_rate, cfg["seed"])
    systems.update(base)
    deltas = [(default, "flag_all"), (default, "cue_union"),
              ("pipeline_" + cfg["clustering"]["algorithm"], default)]
    boot = group_bootstrap(y, groups, strata, systems, deltas, n_boot, cfg["seed"])
    # pair-bootstrap width for the default system, to show how much too narrow it is
    pair_boot = group_bootstrap(y, [str(i) for i in range(len(pairs))], strata,
                                {default: systems[default]}, (), min(n_boot, 500), cfg["seed"])
    write_predictions(res_dir / "pair_predictions_zero_shot.csv.gz", pairs,
                      {"zero_shot": (fwd, back)},
                      {"same_cluster_default": same.astype(int), "mapped_to_claims": mapped.astype(int)})
    per_event = {}
    for e in events:
        m = np.array(strata) == e
        per_event[e] = group_bootstrap(
            y[m], np.array(groups)[m], np.array(strata)[m],
            {k: (None if f is None else f[m], None if s is None else s[m])
             for k, (f, s) in systems.items() if k in (default, "flag_all", "cue_union", "cue_number")},
            (), n_boot, cfg["seed"])
    return {"model": getattr(scorer, "name", "?"), "threshold": thr, "default_system": default,
            "groups": group_summary(pairs), "bootstrap": boot,
            "pair_bootstrap_default": pair_boot["systems"][default], "per_event": per_event}


# --------------------------------------------------------------------------- fine-tuning folds
def finetune_folds(cfg: dict, res_dir: Path, n_boot: int) -> dict[str, Any] | None:
    """Recompute every fold's held-out scores (from caches), then CIs, deltas and baselines."""
    from training.finetune_nli import (checkpoint_name, validation_split, zero_shot_cfg)
    from pipeline.config import finetuned_model_name

    path = res_dir / "finetune_results.json"
    if not path.is_file():
        print(f"[validity] {path} not found; fine-tuning section skipped", flush=True)
        return None
    ft_results = json.loads(path.read_text(encoding="utf-8"))
    splits = resolve_path(cfg, "pheme_rte_splits")
    pooled_pairs: list[RTEPair] = []
    pooled_sys: dict[str, list[tuple]] = defaultdict(list)
    fold_out: dict[str, Any] = {}
    zcfg = zero_shot_cfg(cfg)
    zs = build_scorer(zcfg)
    fold_scores: dict[str, dict[str, Any]] = {}
    for held_out, fold in ft_results["folds"].items():
        train_all = load_rte_split(splits, held_out, "train")
        test = load_rte_split(splits, held_out, "test")
        train_all, _ = remove_leakage(train_all, test)
        _, val, val_event = validation_split(cfg, held_out, train_all)
        fold_scores[held_out] = {"test": test, "val": val, "val_event": val_event,
                                 "zero_shot": _score_both(zs, test)}
    _release(zs)
    for held_out, fold in ft_results["folds"].items():
        fcfg = copy.deepcopy(zcfg)
        fcfg["models"]["nli"] = finetuned_model_name(cfg, checkpoint_name(cfg, held_out))
        fs = build_scorer(fcfg)
        fold_scores[held_out]["finetuned"] = _score_both(fs, fold_scores[held_out]["test"])
        _release(fs)

    for held_out, fold in ft_results["folds"].items():
        d = fold_scores[held_out]
        test, val = d["test"], d["val"]
        y = np.array([p.label == "contradiction" for p in test])
        systems: dict[str, tuple] = {}
        for name, s in fold["settings"].items():
            fwd, back = d[s["model"]]
            c = aggregate_directions(fwd, back, s["aggregation"])[:, C]
            systems[name] = (c >= s["threshold"], c)
        flag_rate = float(systems["finetuned_best_f1"][0].mean())
        base, info = baseline_systems(test, val, flag_rate, cfg["seed"])
        systems.update(base)
        deltas = [("finetuned_best_f1", "zero_shot_val_best_f1"),
                  ("finetuned_best_f1", "zero_shot_default"),
                  ("finetuned_best_f1", "cue_union"),
                  ("finetuned_best_f1", "story_oracle_diagnostic")]
        fold_out[held_out] = {
            "validation_event": d["val_event"], "baseline_thresholds": info,
            "groups": group_summary(test)[held_out],
            "bootstrap": group_bootstrap(y, [story_key(p) for p in test], [p.event for p in test],
                                         systems, deltas, n_boot, cfg["seed"]),
        }
        pooled_pairs += test
        for k, v in systems.items():
            pooled_sys[k].append(v)
        write_predictions(res_dir / f"pair_predictions_fold_{held_out}.csv.gz", test,
                          {"zero_shot": d["zero_shot"], "finetuned": d["finetuned"]})
        b = fold_out[held_out]["bootstrap"]["deltas"]["finetuned_best_f1 - zero_shot_val_best_f1"]["f1"]
        print(f"[validity] {held_out}: dF1 (fine-tuned - zero-shot val-tuned) = {b['value']:+.3f} "
              f"[{b['lo']:+.3f}, {b['hi']:+.3f}]", flush=True)

    # pooled over folds (micro): every pair scored by its own fold's model and operating point
    y = np.array([p.label == "contradiction" for p in pooled_pairs])

    def cat(parts: list[tuple]) -> tuple:
        flags = None if parts[0][0] is None else np.concatenate([p[0] for p in parts])
        scores = None if parts[0][1] is None else np.concatenate([p[1] for p in parts])
        return flags, scores

    systems = {k: cat(v) for k, v in pooled_sys.items() if len(v) == len(fold_out)}
    deltas = [("finetuned_best_f1", "zero_shot_val_best_f1"), ("finetuned_best_f1", "zero_shot_default"),
              ("finetuned_best_f1", "cue_union"), ("finetuned_best_f1", "story_oracle_diagnostic"),
              ("zero_shot_default", "flag_all")]
    pooled = group_bootstrap(y, [story_key(p) for p in pooled_pairs], [p.event for p in pooled_pairs],
                             systems, deltas, n_boot, cfg["seed"])
    # leave-one-event-out sensitivity of the pooled primary delta
    sensitivity = {}
    for e in fold_out:
        keep = np.array([p.event != e for p in pooled_pairs])
        a, b = systems["finetuned_best_f1"], systems["zero_shot_val_best_f1"]
        fa = _prf(y[keep], a[0][keep])[2]
        fb = _prf(y[keep], b[0][keep])[2]
        sensitivity[f"without {e}"] = {"finetuned_f1": fa, "zero_shot_val_f1": fb, "delta_f1": fa - fb}
    return {"folds": fold_out, "pooled": pooled, "pooled_without_one_event": sensitivity,
            "note": ("Each pair is scored by its own fold's model at that fold's validation-chosen "
                     "operating point; CIs resample story groups within each event.")}


# --------------------------------------------------------------------------- entry point
def run_validity(cfg: dict, scorer=None, n_boot: int | None = None,
                 folds: bool = True) -> dict[str, Any]:
    from evaluation.experiments import results_dir, write_json

    set_seed(cfg["seed"])
    n_boot = n_boot or int(cfg["evaluation"].get("bootstrap_samples", 2000))
    res_dir = results_dir(cfg)
    own = scorer is None
    scorer = scorer or build_scorer(cfg)
    out: dict[str, Any] = {"n_boot": n_boot,
                           "zero_shot_all_pairs": zero_shot_all_pairs(cfg, scorer, res_dir, n_boot)}
    if own:
        _release(scorer)
    if folds and not cfg.get("sample_mode") and not cfg["models"].get("mock"):
        out["finetune"] = finetune_folds(cfg, res_dir, n_boot)
    write_json(out, res_dir / RESULTS_FILE)
    zs = out["zero_shot_all_pairs"]
    d = zs["bootstrap"]["systems"][zs["default_system"]]["f1"]
    print(f"[validity] zero-shot {zs['default_system']}: F1 {d['value']:.3f} "
          f"[{d['lo']:.3f}, {d['hi']:.3f}] (story-group bootstrap); wrote {res_dir / RESULTS_FILE}",
          flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    fix_console_encoding()
    parser = argparse.ArgumentParser(description="Story-group CIs, pair baselines, per-pair predictions.")
    parser.add_argument("--config")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--n-boot", type=int)
    parser.add_argument("--no-folds", action="store_true", help="skip the fine-tuning folds")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, sample=args.sample)
    if args.mock:
        cfg["models"]["mock"] = True
    run_validity(cfg, n_boot=args.n_boot, folds=not args.no_folds)


if __name__ == "__main__":
    main()
