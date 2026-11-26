"""Effect of the post-NLI checks on the labelled PHEME RTE pairs (data the checks were not written on).

    python -m evaluation.checks_eval          # needs outputs/results/pair_predictions_*.csv.gz
                                              # (written by python -m evaluation.validity)

For each set of saved predictions (zero-shot on every pair; each fine-tuning fold on its
held-out event), the pairs flagged at the saved operating point are passed through the checks,
and contradiction precision / recall / F1 are compared with and without them. Intervals resample
whole story groups within each event (see evaluation/validity.py). Also reports how many flagged
pairs each check set aside, and how many of those were labelled contradictions (wrongly set aside).
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluation.experiments import results_dir, write_json
from nli_scoring.scorer import C, aggregate_directions
from pipeline.config import fix_console_encoding, load_config
from pipeline.pair_checks import CHECKS, check_pair
from evaluation.validity import group_bootstrap

RESULTS_FILE = "checks_results.json"


def read_predictions(path: Path, tag: str) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fwd = np.array([[float(r[f"{tag}_fwd_{k}"]) for k in "enc"] for r in rows], dtype=np.float32)
    back = np.array([[float(r[f"{tag}_back_{k}"]) for k in "enc"] for r in rows], dtype=np.float32)
    return rows, fwd, back


def evaluate(rows: Sequence[dict[str, str]], score: np.ndarray, threshold: float,
             n_boot: int, seed: int) -> dict[str, Any]:
    y = np.array([r["label"] == "contradiction" for r in rows])
    flag = score >= threshold
    reason = [check_pair(r["premise"], r["hypothesis"], "a", "b") if f else None for r, f in zip(rows, flag)]
    kept = flag & np.array([x is None for x in reason])
    per_check = {}
    for name in CHECKS:
        hit = np.array([x == name for x in reason])
        per_check[name] = {"set_aside": int(hit.sum()), "of_which_labelled_contradiction": int((hit & y).sum())}
    boot = group_bootstrap(y, [r["story_group"] for r in rows], [r["event"] for r in rows],
                           {"without_checks": (flag, None), "with_checks": (kept, None)},
                           [("with_checks", "without_checks")], n_boot, seed)
    return {"n_pairs": len(rows), "threshold": threshold, "n_flagged": int(flag.sum()),
            "n_set_aside": int((flag & ~kept).sum()),
            "n_set_aside_labelled_contradiction": int((flag & ~kept & y).sum()),
            "per_check": per_check, "bootstrap": boot,
            "examples_set_aside": [
                {"premise": r["premise"], "hypothesis": r["hypothesis"], "label": r["label"], "check": x}
                for r, x in zip(rows, reason) if x][:40]}


def run(cfg: dict, n_boot: int = 2000) -> dict[str, Any]:
    res = results_dir(cfg)
    out: dict[str, Any] = {}
    zs = res / "pair_predictions_zero_shot.csv.gz"
    if zs.is_file():
        rows, fwd, back = read_predictions(zs, "zero_shot")
        score = aggregate_directions(fwd, back, cfg["nli"]["aggregation"])[:, C]
        out["zero_shot_all_pairs"] = evaluate(rows, score, cfg["disagreement"]["contradiction_threshold"],
                                              n_boot, cfg["seed"])
    ft_path = res / "finetune_results.json"
    if ft_path.is_file():
        ft = json.loads(ft_path.read_text(encoding="utf-8"))
        folds = {}
        all_rows, all_flag_scores = [], []
        for held_out, fold in ft["folds"].items():
            path = res / f"pair_predictions_fold_{held_out}.csv.gz"
            if not path.is_file():
                continue
            s = fold["settings"]["finetuned_best_f1"]
            rows, fwd, back = read_predictions(path, "finetuned")
            score = aggregate_directions(fwd, back, s["aggregation"])[:, C]
            folds[held_out] = evaluate(rows, score, s["threshold"], n_boot, cfg["seed"])
            all_rows += rows
            all_flag_scores.append((score >= s["threshold"]).astype(np.float32))  # 1 = flagged
        if folds:
            out["finetuned_folds"] = folds
            out["finetuned_pooled"] = evaluate(all_rows, np.concatenate(all_flag_scores), 0.5, n_boot, cfg["seed"])
    write_json(out, res / RESULTS_FILE)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    fix_console_encoding()
    cfg = load_config()
    out = run(cfg)
    for name, r in [("zero-shot, all pairs", out.get("zero_shot_all_pairs")),
                    ("fine-tuned, held-out events pooled", out.get("finetuned_pooled"))]:
        if not r:
            continue
        d = r["bootstrap"]["deltas"]["with_checks - without_checks"]
        s = r["bootstrap"]["systems"]
        fmt = lambda m: f"{m['value']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"
        print(f"\n{name}: flagged {r['n_flagged']}, set aside {r['n_set_aside']} "
              f"({r['n_set_aside_labelled_contradiction']} of them labelled contradictions)")
        for k in ("precision", "recall", "f1"):
            print(f"  {k:9s} without {fmt(s['without_checks'][k])}  with {fmt(s['with_checks'][k])}  "
                  f"delta {d[k]['value']:+.3f} [{d[k]['lo']:+.3f}, {d[k]['hi']:+.3f}]")
        print("  per check:", {k: (v["set_aside"], v["of_which_labelled_contradiction"])
                               for k, v in r["per_check"].items()})
    print(f"\nwrote {results_dir(cfg) / RESULTS_FILE}")


if __name__ == "__main__":
    main()
