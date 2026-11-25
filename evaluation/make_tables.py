"""Render every results table as Markdown, straight from ``outputs/results/*.json``.

    python -m evaluation.make_tables            # writes outputs/RESULTS_TABLES.md
    python -m evaluation.make_tables --sample   # reads and writes under outputs/sample instead
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.experiments import results_dir
from pipeline.config import REPO_ROOT, load_config

LABELS = ("entailment", "neutral", "contradiction")


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_f(c) if not isinstance(c, str) else c for c in row) + " |" for row in rows]
    return out + [""]


def _short(model: str) -> str:
    return model.split("/")[-1].replace("all-", "").replace("-L6-v2", "").replace("-base-v2", "")


FT_SETTINGS = (
    ("zero_shot_default", "zero-shot, default (min, 0.9)"),
    ("zero_shot_val_best_f1", "zero-shot, val-tuned best F1"),
    ("zero_shot_val_recall_at_precision", "zero-shot, val-tuned recall at P>=min"),
    ("finetuned_best_f1", "fine-tuned, val-tuned best F1"),
    ("finetuned_recall_at_precision", "fine-tuned, val-tuned recall at P>=min"),
)


def finetune_tables(ft: dict[str, Any]) -> list[str]:
    """Leave-one-event-out fine-tuning: per fold, mean over folds, pipeline view."""
    lines = ["## NLI fine-tuned on PHEME RTE (leave one event out)", "",
             "Each row is scored on the held-out event only. Direction rule and threshold were "
             "chosen on a validation split of the training events; the held-out event was not used "
             "for training, early stopping or choosing the operating point.", ""]
    header = ["Setting", "Rule", "Threshold", "3-class acc.", "Macro-F1", "Contr. P", "Contr. R",
              "Contr. F1", "Avg. precision", "Flag rate: entailment / unknown / contradiction"]
    for event, fold in ft["folds"].items():
        counts = fold["test_label_counts"]
        lines += [f"### Held out: {event}", "",
                  f"Test pairs {fold['n_test_pairs']:,} (entailment / unknown / contradiction "
                  f"{counts.get('entailment', 0)} / {counts.get('neutral', 0)} / {counts.get('contradiction', 0)}); "
                  f"train {fold['n_train_pairs']:,}, validation {fold['n_validation_pairs']:,} pairs "
                  f"({'event ' + fold['validation_event'] if fold.get('validation_event') else 'pair split'}); "
                  f"best epoch {fold['training']['best_epoch']}; leakage removed from train: "
                  f"{fold['leakage_removed']['removed_total']}.", ""]
        rows = []
        for key, label in FT_SETTINGS:
            r = fold["test"][key]
            c = r["contradiction_at_threshold"]
            fr = r["flag_rate_by_label"]
            rows.append([label, r["aggregation"], f"{c['threshold']:.2f}", r["accuracy"], r["macro_f1"],
                         c["precision"], c["recall"], c["f1"], r["contradiction_average_precision"],
                         f"{_pct(fr['entailment'])} / {_pct(fr['neutral'])} / {_pct(fr['contradiction'])}"])
        lines += _table(header, rows)
    mean = ft.get("mean_over_folds") or {}
    if mean.get("test"):
        lines += [f"### Mean over {mean['n_folds']} folds (unweighted)", ""]
        rows = []
        for key, label in FT_SETTINGS:
            m = mean["test"][key]
            rows.append([label, m["accuracy"], m["macro_f1"], m["contradiction_at_threshold.precision"],
                         m["contradiction_at_threshold.recall"], m["contradiction_at_threshold.f1"],
                         m["contradiction_average_precision"],
                         f"{_pct(m['flag_rate_by_label.entailment'])} / {_pct(m['flag_rate_by_label.neutral'])} / "
                         f"{_pct(m['flag_rate_by_label.contradiction'])}"])
        lines += _table(["Setting", "3-class acc.", "Macro-F1", "Contr. P", "Contr. R", "Contr. F1",
                         "Avg. precision", "Flag rate: entailment / unknown / contradiction"], rows)
    if all(f.get("pipeline_on_labelled_pairs") for f in ft["folds"].values()):
        algo = next(iter(ft["folds"].values()))["pipeline_on_labelled_pairs"]["algorithm"]
        lines += [f"### Pipeline on the held-out labelled pairs ({algo} clusters, then NLI)", "",
                  "A pair is flagged only if both tweets share a cluster and the NLI score reaches the "
                  "setting's threshold (as in the labelled-pair pipeline table above).", ""]
        rows = []
        for event, fold in ft["folds"].items():
            pv = fold["pipeline_on_labelled_pairs"]
            for key, label in FT_SETTINGS:
                r = pv[key]
                rows.append([event, label, r["n_pairs_mapped"],
                             _pct(r["same_cluster_rate_by_label"]["contradiction"]),
                             r["pipeline"]["n_flagged"], r["pipeline"]["precision"], r["pipeline"]["recall"],
                             r["pipeline"]["f1"], r["nli_only"]["recall"]])
        lines += _table(["Held out", "Setting", "Mapped pairs", "Contradictions in same cluster", "Flagged",
                         "Pipeline P", "Pipeline R", "Pipeline F1", "NLI-only R"], rows)
        cm = {e: f["pipeline_on_labelled_pairs"].get("clustering_metrics") for e, f in ft["folds"].items()}
        if all(cm.values()):
            lines += [f"Clustering quality on each held-out event ({algo}, rumour source tweets, reference = "
                      "annotated rumour story). These do not depend on the NLI model.", ""]
            lines += _table(["Held out", "Claims", "Clusters", "Reference stories", "Silhouette (cosine)",
                             "Davies-Bouldin", "ARI", "NMI", "Homogeneity", "Completeness", "V-measure",
                             "Fowlkes-Mallows"],
                            [[e, m["n_claims"], m["n_clusters"], m["n_reference_stories"], m["silhouette"],
                              m["davies_bouldin"], m["ari"], m["nmi"], m["homogeneity"], m["completeness"],
                              m["v_measure"], m["fowlkes_mallows"]] for e, m in cm.items()])
        if mean.get("pipeline"):
            lines += _table(["Mean over folds", "Pipeline P", "Pipeline R", "Pipeline F1"],
                            [[label, mean["pipeline"][k]["precision"], mean["pipeline"][k]["recall"],
                              mean["pipeline"][k]["f1"]] for k, label in FT_SETTINGS])
    rules = ft.get("direction_rules_on_validation")
    if rules:
        lines += [f"### Direction rule for the demo (validation events only, threshold {rules['threshold']:.2f})", "",
                  "Each fold's fine-tuned model on its validation event; the test events are not used.", ""]
        rows = []
        for agg, m in rules["mean"].items():
            fr = m["flag_rate_by_label"]
            rows.append([agg, m["precision"], m["recall"], m["f1"], m["average_precision"],
                         f"{_pct(fr['entailment'])} / {_pct(fr['neutral'])} / {_pct(fr['contradiction'])}"])
        lines += _table(["Rule (mean of 4 validation events)", "Contr. P", "Contr. R", "Contr. F1",
                         "Avg. precision", "Flag rate: entailment / unknown / contradiction"], rows)
        rows = []
        for held, f in rules["per_fold"].items():
            for agg, r in f["rules"].items():
                rows.append([f"{held} (validation: {f['validation_event']})", agg, r["precision"], r["recall"],
                             r["f1"], _pct(r["flag_rate_by_label"]["entailment"])])
        lines += _table(["Fold model", "Rule", "Contr. P", "Contr. R", "Contr. F1", "Entailment flag rate"], rows)

    allm = ft.get("all_events_model")
    if allm:
        modes = allm["operating_point"]["modes"]
        lines += ["### All-events demo model", "",
                  f"`{allm['checkpoint']}`: {allm['n_train_pairs']:,} pairs, {allm['epochs']} epochs (median "
                  f"fold best epoch). Operating points (median of fold choices): best F1 "
                  f"`{modes['best_f1']['aggregation']}` at {modes['best_f1']['threshold']:.2f}; recall at "
                  f"precision `{modes['recall_at_precision']['aggregation']}` at "
                  f"{modes['recall_at_precision']['threshold']:.2f}. It has no held-out test set.", ""]
    return lines


def _ci(m: dict[str, Any] | None, nd: int = 3, signed: bool = False) -> str:
    """``value [lo, hi]`` from a bootstrap entry."""
    if not m or m.get("value") is None:
        return "n/a"
    fmt = f"{{:{'+' if signed else ''}.{nd}f}}"
    v = fmt.format(m["value"])
    if m.get("lo") is None:
        return v
    return f"{v} [{fmt.format(m['lo'])}, {fmt.format(m['hi'])}]"


VALIDITY_SYSTEMS = (
    ("zero_shot_default", "zero-shot, default (min, 0.9)"),
    ("zero_shot_val_best_f1", "zero-shot, val-tuned best F1"),
    ("finetuned_best_f1", "fine-tuned, val-tuned best F1"),
    ("flag_all", "flag every pair"),
    ("random_at_system_flag_rate", "random at the fine-tuned flag rate"),
    ("cue_negation", "cue: negation mismatch"),
    ("cue_number", "cue: number mismatch"),
    ("cue_antonym", "cue: antonym pair"),
    ("cue_union", "cue: any of the three"),
    ("tfidf_cosine_val_tuned", "TF-IDF cosine, val-tuned"),
    ("story_oracle_diagnostic", "story oracle (diagnostic, reads the labels' structure)"),
)


def validity_tables(v: dict[str, Any], pipe: dict[str, Any] | None) -> list[str]:
    """Story-group bootstrap CIs, pair-level baselines and the cluster-size check."""
    lines = ["## Evaluation validity (story-group bootstrap)", "",
             "Every PHEME RTE label follows from the pair's rumour stories, so pairs inside one "
             "(event, stories) group are not independent. Intervals below are 95% percentile "
             f"intervals from {v['n_boot']:,} resamples of whole story groups within each event. "
             "Generated by `python -m evaluation.validity`; per-pair predictions are in "
             "`outputs/results/pair_predictions_*.csv.gz`.", ""]
    zs = v.get("zero_shot_all_pairs")
    if zs:
        g = zs["groups"]
        lines += ["### How many independent units the labelled pairs contain", ""]
        lines += _table(["Event", "Pairs", "Story groups", "Groups with mixed labels", "Contradiction groups",
                         "Contradiction pairs", "Largest contradiction group"],
                        [[e, d["n_pairs"], d["n_story_groups"], d["n_mixed_label_groups"],
                          d["n_contradiction_groups"], d["n_contradiction_pairs"],
                          _pct(d["largest_contradiction_group_share"])] for e, d in g.items()])
        b = zs["bootstrap"]["systems"]
        default = zs["default_system"]
        pb = zs["pair_bootstrap_default"]["f1"]
        sg = b[default]["f1"]
        lines += [f"### Every labelled pair, zero-shot `{zs['model']}` (threshold {zs['threshold']})", "",
                  f"Width of the F1 interval for `{default}`: resampling pairs "
                  f"{pb['hi'] - pb['lo']:.3f}, resampling story groups {sg['hi'] - sg['lo']:.3f}.", ""]
        order = [k for k in b if k.startswith("nli_") or k.startswith("pipeline_")] + \
                [k for k in b if not (k.startswith("nli_") or k.startswith("pipeline_"))]
        lines += _table(["System", "Flag rate", "Contr. P", "Contr. R", "Contr. F1", "AP"],
                        [[k, _pct(b[k].get("flag_rate")), _ci(b[k].get("precision")), _ci(b[k].get("recall")),
                          _ci(b[k].get("f1")), _ci(b[k].get("ap"))] for k in order])
        d = zs["bootstrap"]["deltas"]
        lines += _table(["Paired difference", "Δ precision", "Δ recall", "Δ F1"],
                        [[k, _ci(x.get("precision"), signed=True), _ci(x.get("recall"), signed=True),
                          _ci(x.get("f1"), signed=True)] for k, x in d.items()])
    ft = v.get("finetune")
    if ft:
        names = dict(VALIDITY_SYSTEMS)
        lines += ["### Fine-tuning, held-out events, with baselines", "",
                  "Each pair is scored by its own fold's model at that fold's validation-chosen "
                  "operating point. Baseline thresholds (cosine, oracle) are also chosen on the "
                  "validation event. Pooled = all four held-out events together.", ""]
        p = ft["pooled"]["systems"]
        lines += _table(["System (pooled)", "Flag rate", "Contr. P", "Contr. R", "Contr. F1", "AP"],
                        [[names[k], _pct(p[k].get("flag_rate")), _ci(p[k].get("precision")),
                          _ci(p[k].get("recall")), _ci(p[k].get("f1")), _ci(p[k].get("ap"))]
                         for k, _ in VALIDITY_SYSTEMS if k in p])
        rows = []
        for e, fold in ft["folds"].items():
            s = fold["bootstrap"]["systems"]
            rows.append([e, fold["groups"]["n_contradiction_groups"],
                         *[_ci(s[k]["f1"]) if k in s and "f1" in s[k] else "n/a"
                           for k in ("zero_shot_default", "zero_shot_val_best_f1", "finetuned_best_f1",
                                     "flag_all", "cue_union", "story_oracle_diagnostic")]])
        lines += ["Contradiction F1 per held-out event:", ""]
        lines += _table(["Held out", "Contradiction groups", "Zero-shot default", "Zero-shot val-tuned",
                         "Fine-tuned", "Flag all", "Cue union", "Story oracle"], rows)
        rows = []
        for scope, deltas in [("pooled", ft["pooled"]["deltas"]),
                              *[(e, f["bootstrap"]["deltas"]) for e, f in ft["folds"].items()]]:
            for k, x in deltas.items():
                rows.append([scope, k, _ci(x.get("f1"), signed=True), _ci(x.get("ap"), signed=True),
                             _ci(x.get("recall"), signed=True)])
        lines += ["Paired differences (same resamples for both systems):", ""]
        lines += _table(["Scope", "Comparison", "Δ F1", "Δ AP", "Δ recall"], rows)
        lines += ["Pooled fine-tuned minus zero-shot val-tuned F1, leaving one event out:", ""]
        lines += _table(["Subset", "Fine-tuned F1", "Zero-shot val-tuned F1", "Δ F1"],
                        [[k, x["finetuned_f1"], x["zero_shot_val_f1"], x["delta_f1"]]
                         for k, x in ft["pooled_without_one_event"].items()])
    if pipe and pipe.get("cluster_level_bootstrap"):
        cb = pipe["cluster_level_bootstrap"]
        lines += ["### Cluster level: is it more than cluster size?", "",
                  "The size baselines flag the k largest clusters, with k = the detector's own flag "
                  "count. Clusters are resampled within each event.", ""]
        lines += _table(["System", "Precision", "Recall", "F1", "AP"],
                        [[k, _ci(x.get("precision")), _ci(x.get("recall")), _ci(x.get("f1")), _ci(x.get("ap"))]
                         for k, x in cb["systems"].items()])
        lines += _table(["Paired difference", "Δ F1"],
                        [[k, _ci(x.get("f1"), signed=True)] for k, x in cb["deltas"].items()])
        ap = pipe["cluster_level_ranking_ap"]
        lines += [f"Cluster ranking AP against the proxy: disagreement score {_f(ap['disagreement_score'])}, "
                  f"max contradiction {_f(ap['max_contradiction'])}, size {_f(ap['size'])}, "
                  f"pairs scored {_f(ap['pairs_scored'])}.", ""]
    return lines


def build(res: Path) -> str:
    def read(name: str) -> Any | None:
        path = res / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    lines = ["# Results tables (generated)", "",
             "Generated by `python -m evaluation.make_tables` from the JSON files in "
             f"`{res.relative_to(REPO_ROOT).as_posix()}`. Do not edit by hand.", ""]

    nli = read("nli_results.json")
    if nli:
        for model, block in nli.items():
            if not isinstance(block, dict) or "multinli" not in block:
                continue
            lines += [f"## NLI scorer: `{model}`", "", "### MultiNLI validation", ""]
            rows = []
            for split, name in (("dev_matched", "Matched"), ("dev_mismatched", "Mismatched")):
                m = block["multinli"][split]
                pc = m["per_class"]
                rows.append([name, m["n"], _f(m["accuracy"], 4), _f(m["macro_f1"], 4)]
                            + [f"{_f(pc[l]['precision'])} / {_f(pc[l]['recall'])}" for l in LABELS])
            lines += _table(["Set", "Pairs", "Accuracy", "Macro-F1", "Entailment P / R",
                             "Neutral P / R", "Contradiction P / R"], rows)
            p = block["pheme_proxy"]["pooled"]
            if p:
                b = p["binary_contradiction_at_threshold"]
                lines += ["### PHEME veracity-proxy pairs (weak proxy, not gold)", ""]
                lines += _table(["Pairs", "Accuracy", "Macro-F1", "Contradiction-like P / R at threshold"],
                                [[p["n"], p["accuracy"], p["macro_f1"],
                                  f"{_f(b['precision'])} / {_f(b['recall'])}"]])

    rte = read("rte_results.json")
    if rte:
        c = rte["label_counts"]
        lines += ["## PHEME RTE labelled pairs", "",
                  f"{rte['n_pairs']:,} pairs: {c['entailment']:,} entailment, {c['neutral']:,} unknown, "
                  f"{c['contradiction']:,} contradiction. Majority-class accuracy "
                  f"{_f(rte['majority_class_accuracy'])}; contradiction prevalence "
                  f"{_pct(rte['contradiction_prevalence'])}. Threshold {rte['threshold']}.", "",
                  "### NLI on every labelled pair (no clustering)", ""]
        rows = []
        for name, v in rte["nli_only"].items():
            t = v["contradiction_at_threshold"]
            fr = v["flag_rate_by_label"]
            rows.append([name + (" (default)" if name == rte["default_aggregation"] else ""),
                         v["accuracy"], v["macro_f1"], t["n_flagged"], t["precision"], t["recall"], t["f1"],
                         v["contradiction_average_precision"],
                         f"{_pct(fr['entailment'])} / {_pct(fr['neutral'])} / {_pct(fr['contradiction'])}"])
        lines += _table(["Direction rule", "3-class accuracy", "Macro-F1", "Flagged", "Contradiction precision",
                         "Recall", "F1", "Avg. precision", "Flag rate: entailment / unknown / contradiction"], rows)
        lines += ["### Per event (default direction rule)", ""]
        rows = []
        for event, v in rte["per_event"].items():
            t = v["contradiction_at_threshold"]
            s = {l: v["per_class"][l]["support"] for l in LABELS}
            rows.append([event, v["n"], f"{s['entailment']} / {s['neutral']} / {s['contradiction']}",
                         v["accuracy"], t["precision"], t["recall"]])
        lines += _table(["Event", "Pairs", "Entailment / unknown / contradiction", "3-class accuracy",
                         "Contradiction precision", "Recall"], rows)
        lines += ["### What the pipeline does with the labelled pairs", "",
                  "A pair is flagged by the pipeline only if both tweets are in the same cluster and the "
                  "NLI score reaches the threshold.", ""]
        rows = []
        for algo, v in rte["pipeline_on_labelled_pairs"].items():
            sc = v["same_cluster_rate_by_label"]
            rows.append([algo, f"{_pct(sc['entailment'])} / {_pct(sc['neutral'])} / {_pct(sc['contradiction'])}",
                         v["pipeline"]["n_flagged"], v["pipeline"]["precision"], v["pipeline"]["recall"],
                         v["pipeline"]["f1"]])
        any_algo = next(iter(rte["pipeline_on_labelled_pairs"].values()))
        rows.append(["no clustering (NLI on every pair)", "100% / 100% / 100%", any_algo["nli_only"]["n_flagged"],
                     any_algo["nli_only"]["precision"], any_algo["nli_only"]["recall"], any_algo["nli_only"]["f1"]])
        lines += _table(["Clustering", "Same cluster: entailment / unknown / contradiction", "Flagged",
                         "Contradiction precision", "Recall", "F1"], rows)

    clu = read("clustering_results.json")
    if clu:
        lines += ["## Clustering (mean over events)", ""]
        lines += _table(["Embedding", "Algorithm", "Clusters", "Unclustered", "Silhouette", "Davies-Bouldin",
                         "ARI", "NMI"],
                        [[_short(r["embedding"]), r["algorithm"], _f(r["n_clusters"], 1), _pct(r["noise_fraction"]),
                          r["silhouette"], _f(r["davies_bouldin"], 2), r["ari"], r["nmi"]] for r in clu["summary"]])

    pipe = read("pipeline_results.json")
    if pipe:
        lines += ["## Full pipeline vs the cluster-level veracity proxy", "",
                  f"Default configuration: `{pipe['default_config']}`, threshold {pipe['threshold']}, "
                  f"direction rule `{pipe.get('nli_aggregation')}`.", ""]
        rows = []
        for key, c in pipe["configs"].items():
            cl, pl = c["cluster_level"], c["pair_level"]
            rows.append([_short(c["embedding"]), c["algorithm"] + (" (default)" if key == pipe["default_config"] else ""),
                         cl["n_clusters"], cl["n_proxy_positive"], cl["n_flagged"], cl["precision"], cl["recall"],
                         cl["f1"], cl["average_precision"], pl["n_pairs_scored"], pl["n_pairs_flagged"]])
        lines += _table(["Embedding", "Algorithm", "Clusters", "Proxy-positive", "Flagged", "Precision", "Recall",
                         "F1", "Avg. precision", "Pairs scored", "Pairs flagged"], rows)
        d = pipe["configs"][pipe["default_config"]]
        lines += ["### Baselines on the default configuration's clusters", ""]
        rows = [["Detector (default)", d["cluster_level"]["precision"], d["cluster_level"]["recall"],
                 d["cluster_level"]["f1"]]]
        rows += [[name, b["precision"], b["recall"], b["f1"]] for name, b in pipe["baselines"].items()]
        lines += _table(["Method", "Precision", "Recall", "F1"], rows)
        lines += ["### Per event (default configuration)", ""]
        lines += _table(["Event", "Clusters", "Proxy-positive", "Flagged", "Precision", "Recall", "F1"],
                        [[e, v["cluster_level"]["n_clusters"], v["cluster_level"]["n_proxy_positive"],
                          v["cluster_level"]["n_flagged"], v["cluster_level"]["precision"],
                          v["cluster_level"]["recall"], v["cluster_level"]["f1"]]
                         for e, v in d["per_event"].items()])
        lines += ["### Direction rule ablation (default embedding and clustering)", ""]
        lines += _table(["Direction rule", "Flagged pairs", "Flagged clusters", "Precision", "Recall", "F1"],
                        [[a, b["pair_level"]["n_pairs_flagged"],
                          f"{b['cluster_level']['n_flagged']} of {b['cluster_level']['n_clusters']}",
                          b["cluster_level"]["precision"], b["cluster_level"]["recall"], b["cluster_level"]["f1"]]
                         for a, b in pipe["aggregation_ablation"].items()])
        sweep = d["threshold_sweep"]
        lines += ["### Threshold sweep (default configuration)", ""]
        lines += _table(["Threshold"] + [str(s["threshold"]) for s in sweep],
                        [["Flagged clusters"] + [s["n_flagged"] for s in sweep],
                         ["Precision"] + [s["precision"] for s in sweep],
                         ["Recall"] + [s["recall"] for s in sweep],
                         ["F1"] + [s["f1"] for s in sweep]])
        nc = pipe["no_clustering_baseline"]
        a, c = nc["all_pairs"], nc["clustered_same_subset"]
        lines += [f"### All pairs vs in-cluster pairs (at most {nc['max_claims_per_event']} claims per event, "
                  f"{a['n_claims']:,} claims)", ""]
        lines += _table(["", "All pairs, no clustering", "In-cluster pairs"], [
            ["Pairs scored", a["n_pairs_scored"], f"{c['n_pairs_scored']:,} ({_pct(c['pair_reduction'])} fewer)"],
            ["Pairs flagged", a["n_pairs_flagged"], c["n_pairs_flagged"]],
            ["Flagged pairs about the same rumour story", _pct(a["flagged_same_story_rate"]),
             _pct(c["flagged_same_story_rate"])],
            ["Debunk pairs kept together", f"{a['debunk_co_clustered']} of {a['n_debunk_pairs']}",
             f"{c['debunk_co_clustered']} of {c['n_debunk_pairs']}"],
            ["Debunk pairs flagged", a["debunk_flagged"], c["debunk_flagged"]],
        ])

    ft = read("finetune_results.json")
    if ft and ft.get("folds"):
        lines += finetune_tables(ft)

    validity = read("validity_results.json")
    if validity:
        lines += validity_tables(validity, pipe)

    probe_files = sorted(f for f in res.glob("witness_probes_*.json")
                         if not f.stem.startswith(("witness_probes_extended", "witness_probes_all")))
    if probe_files:
        types = ["first_person", "first_person_control", "control", "splitter", "truncation", "example"]
        lines += ["## Witness-style probes (text layer around NLI)", "",
                  "Hand-written checks from the adversarial review, run through the real pipeline "
                  "(`python -m evaluation.witness_probes --tag <name>`). They show failure modes and "
                  "regressions, not error rates. Rows are in the order the changes were applied.", ""]
        rows = []
        for f in probe_files:
            d = json.loads(f.read_text(encoding="utf-8"))
            bt = d["summary"]["by_type"]
            s = d["settings"]
            setting = (f"split={s.get('data', {}).get('sentence_splitter', '?')}, "
                       f"max words={s.get('data', {}).get('max_claim_words')}, "
                       f"ground={s['nli'].get('ground_speaker')}")
            rows.append([f.stem.replace("witness_probes_", ""),
                         d["nli_model"].replace("\\", "/").split("/")[-1], setting,
                         f"{d['summary']['passed']}/{d['summary']['checks']}",
                         *[f"{bt[t]['passed']}/{bt[t]['checks']}" if t in bt else "n/a" for t in types]])
        lines += _table(["Run", "NLI model", "Text layer", "Passed", *types], rows)

    summaries = read("event_report_summaries.json")
    if summaries:
        lines += ["## Per-event reports on all source tweets", ""]
        lines += _table(["Event", "Claims", "Clusters", "Unclustered", "Pairs scored", "All pairs would be",
                         "Flagged clusters"],
                        [[e, s["n_claims"], s["n_clusters"], s["n_unclustered_claims"], s["n_pairs_scored"],
                          s["n_pairs_all_pairs"], s["n_flagged_clusters"]] for e, s in summaries.items()])

    meta = read("run_meta.json")
    if meta:
        lines += ["## Run", "", f"Device: {meta['device']}. Stage timings in seconds (warm caches make "
                  f"these much shorter than a first run): {meta['timings_seconds']}.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    cfg = load_config(sample=args.sample)
    res = results_dir(cfg)
    out = res.parent / "RESULTS_TABLES.md"
    out.write_text(build(res), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
