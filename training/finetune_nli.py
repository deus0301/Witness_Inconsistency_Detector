"""Fine-tune the NLI cross-encoder on the PHEME RTE pairs, leave one event out.

    python -m training.finetune_nli --folds germanwings-crash     # one fold
    python -m training.finetune_nli                               # every fold in the config
    python -m training.finetune_nli --all-events --skip-folds     # demo model from fold results

For each held-out event:

1. Load the train and test split files, drop train pairs that also appear in the test file.
2. Choose a validation set from the TRAIN events only. Default (``finetune.validation:
   held_out_event``): one whole training event. Alternative (``pairs``): ``val_fraction`` of
   the train pairs, stratified by label and grouped by the unordered text pair. The pair split
   leaks, because each tweet appears in about 18 pairs and labels follow rumour stories.
3. Train on both directions of every remaining pair (same label), plus an optional MultiNLI
   replay sample, with class weighting or a weighted sampler. Early stopping on validation.
4. Choose the direction rule and threshold on validation only, in two modes (best F1;
   highest recall with precision >= ``min_precision``), and save them with the checkpoint.
5. Score the held-out event with the zero-shot model (default point and its own val-tuned
   point) and the fine-tuned model, and run the "what clustering does" view.

The test event is never used for training, early stopping or choosing an operating point.
Every hyperparameter is read from ``configs/default.yaml`` (section ``finetune``).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data.multinli import load_multinli
from data.pheme_rte import (RTEPair, SPLIT_KEYS, load_pheme_rte, load_rte_split,
                            remove_leakage, text_key)
from data.schema import NLI_LABELS
from nli_scoring.scorer import C, aggregate_directions, build_scorer, label_permutation
from pipeline.config import (FINETUNE_META_FILE, OPERATING_POINT_FILE, finetuned_model_name,
                             fix_console_encoding, load_config, pick_device, resolve_model,
                             resolve_path, set_seed)

RESULTS_FILE = "finetune_results.json"


def log(msg: str) -> None:
    print(f"[finetune] {msg}", flush=True)


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Example:
    """One directional training example."""

    premise: str
    hypothesis: str
    label: str          # entailment | neutral | contradiction
    source: str         # pheme | mnli


def both_directions(pairs: Sequence[RTEPair], enabled: bool = True) -> list[Example]:
    """(A,B) for every pair, plus (B,A) with the same label when ``enabled``."""
    out: list[Example] = []
    for p in pairs:
        out.append(Example(p.premise, p.hypothesis, p.label, "pheme"))
        if enabled:
            out.append(Example(p.hypothesis, p.premise, p.label, "pheme"))
    return out


def grouped_stratified_split(pairs: Sequence[RTEPair], val_fraction: float,
                             seed: int) -> tuple[list[RTEPair], list[RTEPair]]:
    """Split pairs into (train, validation), stratified by label and grouped by text pair.

    A group is every pair with the same unordered pair of texts, so a pair and its reverse
    (or a duplicate under another id) always land on the same side. A group's label for
    stratification is its most common label.
    """
    groups: dict[frozenset, list[RTEPair]] = {}
    for p in pairs:
        groups.setdefault(text_key(p), []).append(p)
    by_label: dict[str, list[frozenset]] = {}
    for key, members in groups.items():
        label = Counter(m.label for m in members).most_common(1)[0][0]
        by_label.setdefault(label, []).append(key)
    rng = random.Random(seed)
    val_keys: set[frozenset] = set()
    for label in sorted(by_label):
        keys = sorted(by_label[label], key=lambda k: sorted(k))
        rng.shuffle(keys)
        val_keys.update(keys[:int(round(val_fraction * len(keys)))])
    train = [p for p in pairs if text_key(p) not in val_keys]
    val = [p for p in pairs if text_key(p) in val_keys]
    return train, val


def mnli_replay(cfg: dict) -> list[Example]:
    """``finetune.mnli_replay`` random MultiNLI examples (one direction, as annotated)."""
    ft = cfg["finetune"]
    n = int(ft.get("mnli_replay") or 0)
    if n <= 0:
        return []
    try:
        examples = load_multinli(resolve_path(cfg, "multinli"), ft["mnli_split"], n, cfg["seed"])
    except FileNotFoundError as exc:
        log(f"WARNING: MultiNLI replay skipped, file not found ({exc.filename})")
        return []
    return [Example(e.premise, e.hypothesis, e.label, "mnli") for e in examples]


def label_to_model_id(id2label: dict) -> dict[str, int]:
    """Project label name -> the model's class index, read from ``config.id2label``."""
    perm = label_permutation(id2label)
    return {lab: perm[k] for k, lab in enumerate(NLI_LABELS)}


def class_weights(labels: Sequence[str], label2id: dict[str, int]) -> list[float]:
    """Inverse-frequency weights in model-index order, normalised to mean 1."""
    counts = Counter(labels)
    raw = {lab: len(labels) / (len(counts) * counts[lab]) for lab in counts}
    out = [1.0] * len(label2id)
    for lab, idx in label2id.items():
        out[idx] = raw.get(lab, 1.0)
    return out


def cap(items: list, n: int | None, seed: int) -> list:
    if not n or len(items) <= n:
        return items
    idx = sorted(random.Random(seed).sample(range(len(items)), n))
    return [items[i] for i in idx]


# --------------------------------------------------------------------------- operating point
def threshold_grid(cfg: dict) -> list[float]:
    fine = cfg["finetune"]["operating_point"]["fine_grid"]
    steps = np.arange(fine["start"], fine["stop"] + 1e-9, fine["step"])
    return sorted({round(float(t), 4) for t in [*cfg["evaluation"]["threshold_grid"], *steps]})


def _prf_at(is_contra: np.ndarray, score: np.ndarray, thr: float) -> dict[str, float]:
    flagged = score >= thr
    tp = int((is_contra & flagged).sum())
    fp = int((~is_contra & flagged).sum())
    fn = int((is_contra & ~flagged).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "n_flagged": int(flagged.sum())}


def choose_operating_point(truth: Sequence[str], fwd: np.ndarray, back: np.ndarray,
                           cfg: dict) -> dict[str, Any]:
    """Pick (direction rule, threshold) on VALIDATION scores, in two modes.

    ``best_f1``: highest contradiction F1 (ties: higher precision, then higher threshold).
    ``recall_at_precision``: highest recall with precision >= ``min_precision`` (ties:
    higher precision). Falls back to ``best_f1`` if no point reaches that precision.
    """
    op = cfg["finetune"]["operating_point"]
    is_contra = np.array([t == "contradiction" for t in truth])
    grid = threshold_grid(cfg)
    points = []
    for agg in op["aggregations"]:
        score = aggregate_directions(fwd, back, agg)[:, C]
        for thr in grid:
            points.append({"aggregation": agg, "threshold": thr, **_prf_at(is_contra, score, thr)})
    best_f1 = max(points, key=lambda d: (d["f1"], d["precision"], d["threshold"]))
    feasible = [d for d in points if d["precision"] >= op["min_precision"] and d["n_flagged"] > 0]
    if feasible:
        rap = dict(max(feasible, key=lambda d: (d["recall"], d["precision"], d["threshold"])),
                   feasible=True)
    else:
        rap = dict(best_f1, feasible=False)
    return {"modes": {"best_f1": best_f1, "recall_at_precision": rap},
            "min_precision": op["min_precision"], "n_validation_pairs": len(truth),
            "chosen_on": "validation"}


def evaluate_at(truth: Sequence[str], fwd: np.ndarray, back: np.ndarray,
                aggregation: str, threshold: float) -> dict[str, Any]:
    from evaluation.experiments import summarise_labelled

    probs = aggregate_directions(fwd, back, aggregation)
    out = summarise_labelled(list(truth), probs, threshold)
    out["aggregation"] = aggregation
    return out


def score_both(scorer, pairs: Sequence[RTEPair]) -> tuple[np.ndarray, np.ndarray]:
    fwd = scorer.score_directional([(p.premise, p.hypothesis) for p in pairs])
    back = scorer.score_directional([(p.hypothesis, p.premise) for p in pairs])
    return fwd, back


# --------------------------------------------------------------------------- training
def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def train_model(cfg: dict, examples: Sequence[Example], val_pairs: Sequence[RTEPair],
                out_dir: Path, epochs: int | None = None, early_stopping: bool = True,
                base_model: str | None = None) -> dict[str, Any]:
    """Fine-tune ``models.nli`` (or ``base_model``) on ``examples`` and save the best epoch.

    With ``early_stopping`` the model is scored on ``val_pairs`` after every epoch and the
    best epoch (``finetune.early_stopping_metric``) is kept. Without it, the last epoch is
    saved. Returns the training history.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    from nli_scoring.scorer import NLIScorer

    ft = cfg["finetune"]
    seed = cfg["seed"]
    set_seed(seed)
    device = pick_device(cfg)
    fp16 = bool(ft["fp16"]) and device == "cuda"
    base = base_model or resolve_model(cfg["models"]["nli"], cfg)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base).to(device)
    label2id = label_to_model_id(model.config.id2label)
    labels = [e.label for e in examples]
    targets = torch.tensor([label2id[lab] for lab in labels], dtype=torch.long)
    log(f"training on {len(examples)} examples {dict(Counter(labels))} "
        f"(sources {dict(Counter(e.source for e in examples))}); device={device} fp16={fp16}; "
        f"model label order {[model.config.id2label[i] for i in sorted(model.config.id2label)]}")

    imbalance = ft["imbalance"]
    weights = class_weights(labels, label2id)
    loss_weight = torch.tensor(weights, dtype=torch.float, device=device) \
        if imbalance == "weighted_loss" else None
    generator = torch.Generator().manual_seed(seed)
    if imbalance == "sampler":
        per_example = [weights[label2id[lab]] for lab in labels]
        sampler = WeightedRandomSampler(per_example, num_samples=len(examples), replacement=True,
                                        generator=generator)
        loader_kwargs = {"sampler": sampler}
    elif imbalance in ("weighted_loss", "none"):
        loader_kwargs = {"shuffle": True, "generator": generator}
    else:
        raise ValueError(f"finetune.imbalance must be weighted_loss | sampler | none, got {imbalance!r}")

    def collate(idx: list[int]):
        batch = tokenizer([examples[i].premise for i in idx], [examples[i].hypothesis for i in idx],
                          padding=True, truncation=True, max_length=ft["max_length"],
                          return_tensors="pt")
        batch["labels"] = targets[idx]
        return batch

    loader = DataLoader(list(range(len(examples))), batch_size=ft["batch_size"],
                        collate_fn=collate, num_workers=0, **loader_kwargs)
    n_epochs = int(epochs or ft["epochs"])
    total_steps = n_epochs * len(loader)
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": ft["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=float(ft["lr"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(round(ft["warmup_ratio"] * total_steps)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)

    val_truth = [p.label for p in val_pairs]
    metric_name = ft["early_stopping_metric"]
    history: list[dict[str, Any]] = []
    best_value, best_epoch, best_state, bad_epochs = -math.inf, 0, None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0, running, steps = time.time(), 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), ft["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss)
            steps += 1
        row: dict[str, Any] = {"epoch": epoch, "train_loss": running / max(1, steps),
                               "seconds": round(time.time() - t0, 1)}
        if early_stopping and val_pairs:
            scorer = NLIScorer(str(out_dir), batch_size=ft["eval_batch_size"],
                               max_length=ft["max_length"], device=device, fp16=fp16,
                               cache_path=None, verbose=False, model=model, tokenizer=tokenizer)
            row.update(validation_metrics(val_truth, *score_both(scorer, val_pairs), cfg))
            value = row[metric_name]
            if value > best_value:
                best_value, best_epoch, bad_epochs = value, epoch, 0
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
        history.append(row)
        log(f"epoch {epoch}/{n_epochs}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k != "epoch"))
        if early_stopping and val_pairs and bad_epochs >= ft["early_stopping_patience"]:
            log(f"early stopping after epoch {epoch} (best epoch {best_epoch})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(history)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    saved_id2label = dict(model.config.id2label)
    label_permutation(saved_id2label)  # raises if the saved config lost its label names
    info = {"history": history, "best_epoch": best_epoch, "early_stopping_metric": metric_name,
            "best_value": None if best_value == -math.inf else best_value,
            "label_to_model_id": label2id, "id2label": {str(k): v for k, v in saved_id2label.items()},
            "class_weights_model_order": weights if imbalance != "none" else None}
    del model, optimizer, best_state
    free_gpu()
    return info


def validation_metrics(truth: Sequence[str], fwd: np.ndarray, back: np.ndarray,
                       cfg: dict) -> dict[str, float]:
    """Early-stopping metrics: best over the configured direction rules of contradiction
    average precision and of best-threshold contradiction F1."""
    from sklearn.metrics import average_precision_score

    is_contra = np.array([t == "contradiction" for t in truth])
    aps, f1s = [], []
    grid = threshold_grid(cfg)
    for agg in cfg["finetune"]["operating_point"]["aggregations"]:
        score = aggregate_directions(fwd, back, agg)[:, C]
        aps.append(float(average_precision_score(is_contra, score)) if 0 < is_contra.sum() < len(truth) else 0.0)
        f1s.append(max(_prf_at(is_contra, score, t)["f1"] for t in grid))
    return {"contradiction_ap": max(aps), "contradiction_f1": max(f1s)}


# --------------------------------------------------------------------------- folds
def validation_event(cfg: dict, held_out: str) -> str:
    """The event after ``held_out`` in ``finetune.folds`` (cyclic), so each event validates once."""
    folds = list(cfg["finetune"]["folds"])
    order = folds if held_out in folds else [held_out, *folds]
    return order[(order.index(held_out) + 1) % len(order)]


def validation_split(cfg: dict, held_out: str,
                     train_all: list[RTEPair]) -> tuple[list[RTEPair], list[RTEPair], str | None]:
    """(train, validation, validation event or None) from the TRAIN events of a fold."""
    ft = cfg["finetune"]
    if ft["validation"] == "held_out_event":
        event = validation_event(cfg, held_out)
        train = [p for p in train_all if p.event != event]
        val = [p for p in train_all if p.event == event]
        if not train or not val:
            raise ValueError(f"validation event {event} leaves an empty train or validation set")
        return train, val, event
    if ft["validation"] == "pairs":
        train, val = grouped_stratified_split(train_all, ft["val_fraction"], cfg["seed"])
        return train, val, None
    raise ValueError(f"finetune.validation must be held_out_event | pairs, got {ft['validation']!r}")


def checkpoint_name(cfg: dict, held_out: str | None) -> str:
    base = cfg["models"]["nli"].split("/")[-1]
    return f"{base}-pheme-leave-{SPLIT_KEYS[held_out]}" if held_out else cfg["finetune"]["checkpoint"]


def zero_shot_cfg(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    out["finetune"]["use_finetuned"] = False
    return out


def write_checkpoint_json(model_dir: Path, point: dict[str, Any], meta: dict[str, Any], cfg: dict) -> None:
    point = dict(point, default_mode=cfg["finetune"]["operating_mode"])
    (model_dir / OPERATING_POINT_FILE).write_text(json.dumps(point, indent=2), encoding="utf-8")
    (model_dir / FINETUNE_META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_fold(cfg: dict, held_out: str, pipeline_eval: bool = True) -> dict[str, Any]:
    """Train, choose the operating point on validation, evaluate on the held-out event."""
    ft = cfg["finetune"]
    seed = cfg["seed"]
    t_start = time.time()
    splits = resolve_path(cfg, "pheme_rte_splits")
    train_all = load_rte_split(splits, held_out, "train")
    test = load_rte_split(splits, held_out, "test")
    train_all, leakage = remove_leakage(train_all, test)
    log(f"fold leave-{SPLIT_KEYS[held_out]} ({held_out}): train {len(train_all)} pairs, test "
        f"{len(test)} pairs; leakage removed from train: {leakage}")
    train_all = cap(train_all, ft.get("max_train_pairs"), seed)
    test = cap(test, ft.get("max_test_pairs"), seed)
    train_pairs, val_pairs, val_event = validation_split(cfg, held_out, train_all)
    assert not ({text_key(p) for p in train_pairs} & {text_key(p) for p in val_pairs})
    assert not ({text_key(p) for p in train_pairs + val_pairs} & {text_key(p) for p in test})
    val_truth, test_truth = [p.label for p in val_pairs], [p.label for p in test]
    log(f"train {len(train_pairs)} / validation {len(val_pairs)} pairs "
        f"(validation: {val_event or 'pair split'}; val labels {dict(Counter(val_truth))})")

    # zero-shot: default point and its own val-tuned point
    zcfg = zero_shot_cfg(cfg)
    zs = build_scorer(zcfg)
    z_val, z_test = score_both(zs, val_pairs), score_both(zs, test)
    del zs
    free_gpu()
    z_point = choose_operating_point(val_truth, *z_val, cfg)
    default_agg, default_thr = zcfg["nli"]["aggregation"], zcfg["disagreement"]["contradiction_threshold"]

    # fine-tune
    name = finetuned_model_name(cfg, checkpoint_name(cfg, held_out))
    model_dir = resolve_path(cfg, "finetuned_dir") / checkpoint_name(cfg, held_out)
    examples = both_directions(train_pairs, ft["both_directions"]) + mnli_replay(cfg)
    random.Random(seed).shuffle(examples)
    train_info = train_model(cfg, examples, val_pairs, model_dir)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    meta = {"run_id": run_id, "base_model": cfg["models"]["nli"], "held_out_event": held_out,
            "train_events": sorted({p.event for p in train_all}),
            "n_train_pairs": len(train_pairs), "n_validation_pairs": len(val_pairs),
            "validation": ft["validation"], "validation_event": val_event,
            "n_test_pairs": len(test), "n_training_examples": len(examples),
            "leakage_removed": leakage, "hyperparameters": {k: v for k, v in ft.items() if k not in (
                "use_finetuned", "checkpoint", "operating_mode", "folds", "active")},
            "seed": seed, **train_info}
    write_checkpoint_json(model_dir, {"modes": {}}, meta, cfg)  # run_id first: names the cache file

    # load the saved checkpoint the way the pipeline does (resolve_model + own cache file)
    fcfg = copy.deepcopy(zcfg)
    fcfg["models"]["nli"] = name
    fs = build_scorer(fcfg)
    f_val, f_test = score_both(fs, val_pairs), score_both(fs, test)
    del fs
    free_gpu()
    f_point = choose_operating_point(val_truth, *f_val, cfg)
    write_checkpoint_json(model_dir, f_point, meta, cfg)

    settings = {
        "zero_shot_default": ("zero_shot", default_agg, default_thr),
        "zero_shot_val_best_f1": ("zero_shot", z_point["modes"]["best_f1"]["aggregation"],
                                  z_point["modes"]["best_f1"]["threshold"]),
        "zero_shot_val_recall_at_precision": (
            "zero_shot", z_point["modes"]["recall_at_precision"]["aggregation"],
            z_point["modes"]["recall_at_precision"]["threshold"]),
        "finetuned_best_f1": ("finetuned", f_point["modes"]["best_f1"]["aggregation"],
                              f_point["modes"]["best_f1"]["threshold"]),
        "finetuned_recall_at_precision": (
            "finetuned", f_point["modes"]["recall_at_precision"]["aggregation"],
            f_point["modes"]["recall_at_precision"]["threshold"]),
    }
    scores = {"zero_shot": (z_val, z_test), "finetuned": (f_val, f_test)}
    test_results, val_results = {}, {}
    for key, (model, agg, thr) in settings.items():
        (v_fwd, v_back), (t_fwd, t_back) = scores[model]
        val_results[key] = evaluate_at(val_truth, v_fwd, v_back, agg, thr)
        test_results[key] = evaluate_at(test_truth, t_fwd, t_back, agg, thr)
        r = test_results[key]["contradiction_at_threshold"]
        log(f"test {key:>34} ({agg}, {thr:.2f}): acc={test_results[key]['accuracy']:.3f} "
            f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
            f"AP={test_results[key]['contradiction_average_precision']:.3f}")

    pipeline = {}
    if pipeline_eval:
        pipeline = pipeline_view(cfg, held_out, test, settings, scores)
    figure = confusion_figure(cfg, held_out, test_results["finetuned_best_f1"])
    return {"held_out_event": held_out, "split_key": SPLIT_KEYS[held_out],
            "checkpoint": name, "run_id": run_id,
            "n_train_pairs": len(train_pairs), "n_validation_pairs": len(val_pairs),
            "validation_mode": ft["validation"], "validation_event": val_event,
            "train_events": sorted({p.event for p in train_pairs}),
            "n_test_pairs": len(test), "n_training_examples": len(examples),
            "test_label_counts": dict(Counter(test_truth)),
            "validation_label_counts": dict(Counter(val_truth)),
            "leakage_removed": leakage, "training": train_info,
            "operating_points": {"zero_shot": z_point, "finetuned": f_point},
            "settings": {k: {"model": m, "aggregation": a, "threshold": t} for k, (m, a, t) in settings.items()},
            "validation": val_results, "test": test_results, "pipeline_on_labelled_pairs": pipeline,
            "confusion_figure": figure, "seconds": round(time.time() - t_start, 1)}


def pipeline_view(cfg: dict, held_out: str, test: list[RTEPair],
                  settings: dict[str, tuple[str, str, float]],
                  scores: dict[str, Any]) -> dict[str, Any]:
    """REPORT 6.4 on the held-out event: cluster its rumour tweets, flag only same-cluster pairs."""
    from clustering.embed import build_embedder
    from evaluation.experiments import pipeline_on_labelled_pairs

    algo = cfg["finetune"]["pipeline_eval_algorithm"]
    embedder = build_embedder(zero_shot_cfg(cfg))
    out = {"algorithm": algo}
    for key, (model, agg, thr) in settings.items():
        _, (t_fwd, t_back) = scores[model]
        probs = aggregate_directions(t_fwd, t_back, agg)
        out[key] = pipeline_on_labelled_pairs(cfg, {held_out: test}, probs, thr, embedder, algo)
        r = out[key]["pipeline"]
        log(f"pipeline ({algo}) {key:>34}: flagged={r['n_flagged']} P={r['precision']:.3f} "
            f"R={r['recall']:.3f} F1={r['f1']:.3f}")
    # clustering does not depend on the NLI model: one set of clustering metrics per fold
    out["clustering_metrics"] = out[next(iter(settings))]["clustering_metrics"][held_out]
    m = out["clustering_metrics"]
    log(f"clustering ({algo}) on {held_out}: k={m['n_clusters']} silhouette={m['silhouette']} "
        f"DB={m['davies_bouldin']} ARI={m['ari']:.3f} NMI={m['nmi']:.3f} "
        f"V={m['v_measure']:.3f} FM={m['fowlkes_mallows']:.3f}")
    del embedder
    free_gpu()
    return out


def confusion_figure(cfg: dict, held_out: str, result: dict[str, Any]) -> str:
    from evaluation.plots import plot_confusion

    path = resolve_path(cfg, "figures_dir") / f"confusion_finetune_leave_{SPLIT_KEYS[held_out]}.png"
    plot_confusion(result["confusion_matrix"], result["labels"],
                   f"Fine-tuned, held-out {held_out}\n{result['aggregation']}: acc "
                   f"{result['accuracy']:.3f}, macro-F1 {result['macro_f1']:.3f}",
                   path, ylabel="RTE label")
    return path.name


# --------------------------------------------------------------------------- summary
SUMMARY_KEYS = (("accuracy",), ("macro_f1",), ("contradiction_at_threshold", "precision"),
                ("contradiction_at_threshold", "recall"), ("contradiction_at_threshold", "f1"),
                ("contradiction_average_precision",), ("flag_rate_by_label", "entailment"),
                ("flag_rate_by_label", "neutral"), ("flag_rate_by_label", "contradiction"))


def _get(d: dict, path: Sequence[str]) -> Any:
    for k in path:
        d = d[k]
    return d


def mean_over_folds(folds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Unweighted mean over folds of every test metric and of pipeline recall/precision."""
    if not folds:
        return {}
    any_fold = next(iter(folds.values()))
    out: dict[str, Any] = {"n_folds": len(folds), "folds": sorted(folds), "test": {}, "pipeline": {}}
    for setting in any_fold["test"]:
        out["test"][setting] = {".".join(path): float(np.mean([_get(f["test"][setting], path)
                                                              for f in folds.values()]))
                                for path in SUMMARY_KEYS}
        if all(f.get("pipeline_on_labelled_pairs") for f in folds.values()):
            out["pipeline"][setting] = {m: float(np.mean([f["pipeline_on_labelled_pairs"][setting]["pipeline"][m]
                                                          for f in folds.values()]))
                                        for m in ("precision", "recall", "f1")}
    cms = [f["pipeline_on_labelled_pairs"].get("clustering_metrics") for f in folds.values()
           if f.get("pipeline_on_labelled_pairs")]
    if cms and all(cms):
        keys = ("silhouette", "davies_bouldin", "ari", "nmi", "homogeneity", "completeness",
                "v_measure", "fowlkes_mallows")
        out["clustering_metrics"] = {k: float(np.mean([m[k] for m in cms])) for k in keys
                                     if all(m.get(k) is not None for m in cms)}
    return out


def results_path(cfg: dict) -> Path:
    from evaluation.experiments import results_dir

    return results_dir(cfg) / RESULTS_FILE


def load_results(cfg: dict) -> dict[str, Any]:
    path = results_path(cfg)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"folds": {}}


def save_results(cfg: dict, results: dict[str, Any]) -> Path:
    from evaluation.experiments import write_json

    results["mean_over_folds"] = mean_over_folds(results["folds"])
    path = results_path(cfg)
    write_json(results, path)
    return path


# --------------------------------------------------------------------------- all-events model
def train_all_events(cfg: dict, folds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Demo model on every labelled pair, with the median fold-chosen epoch count, direction
    rule (most common) and threshold (median among folds that chose that rule)."""
    if not folds:
        raise RuntimeError("train the folds first: the all-events model uses their chosen settings")
    ft = cfg["finetune"]
    epochs = int(math.floor(np.median([f["training"]["best_epoch"] for f in folds.values()]) + 0.5))
    order = ft["operating_point"]["aggregations"]
    point: dict[str, Any] = {"modes": {}, "chosen_on": "median of leave-one-event-out folds",
                             "folds": sorted(folds), "min_precision": ft["operating_point"]["min_precision"]}
    for mode in ("best_f1", "recall_at_precision"):
        chosen = [f["operating_points"]["finetuned"]["modes"][mode] for f in folds.values()]
        counts = Counter(c["aggregation"] for c in chosen)
        agg = max(counts, key=lambda a: (counts[a], -order.index(a)))
        thr = float(np.median([c["threshold"] for c in chosen if c["aggregation"] == agg]))
        point["modes"][mode] = {"aggregation": agg, "threshold": round(thr, 4),
                                "fold_choices": [(c["aggregation"], c["threshold"]) for c in chosen]}
    pairs = [p for event in SPLIT_KEYS for p in load_pheme_rte(resolve_path(cfg, "pheme_rte"), event)]
    pairs = cap(pairs, ft.get("max_train_pairs"), cfg["seed"])
    examples = both_directions(pairs, ft["both_directions"]) + mnli_replay(cfg)
    random.Random(cfg["seed"]).shuffle(examples)
    log(f"all-events model: {len(pairs)} pairs, {epochs} epochs, operating points {point['modes']}")
    model_dir = resolve_path(cfg, "finetuned_dir") / ft["checkpoint"]
    info = train_model(cfg, examples, [], model_dir, epochs=epochs, early_stopping=False)
    meta = {"run_id": time.strftime("%Y%m%d-%H%M%S"), "base_model": cfg["models"]["nli"],
            "held_out_event": None, "train_events": sorted(SPLIT_KEYS), "n_train_pairs": len(pairs),
            "n_training_examples": len(examples), "epochs": epochs, "seed": cfg["seed"], **info}
    write_checkpoint_json(model_dir, point, meta, cfg)
    return {"checkpoint": finetuned_model_name(cfg, ft["checkpoint"]), "epochs": epochs,
            "operating_point": point, "n_train_pairs": len(pairs), "run_id": meta["run_id"]}


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> None:
    fix_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", help="path to a YAML config (default: configs/default.yaml)")
    parser.add_argument("--sample", action="store_true", help="tiny subset (smoke run)")
    parser.add_argument("--folds", nargs="+", help="held-out events (default: finetune.folds)")
    parser.add_argument("--skip-folds", action="store_true", help="do not train any fold")
    parser.add_argument("--all-events", action="store_true",
                        help="also train the all-events demo model from the fold results")
    parser.add_argument("--no-pipeline-eval", action="store_true",
                        help="skip the clustering view (needs the raw PHEME threads)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, sample=args.sample)
    if cfg["models"].get("mock"):
        raise SystemExit("fine-tuning needs a real model; models.mock is set")
    cfg["finetune"]["use_finetuned"] = False  # training always starts from the zero-shot base
    folds = [] if args.skip_folds else (args.folds or cfg["finetune"]["folds"])
    unknown = [f for f in folds if f not in SPLIT_KEYS]
    if unknown:
        raise SystemExit(f"unknown events {unknown}; choose from {sorted(SPLIT_KEYS)}")
    results = load_results(cfg)
    for event in folds:
        results["folds"][event] = run_fold(cfg, event, pipeline_eval=not args.no_pipeline_eval)
        log(f"wrote {save_results(cfg, results)}")
    if args.all_events:
        results["all_events_model"] = train_all_events(cfg, results["folds"])
        log(f"wrote {save_results(cfg, results)}")


if __name__ == "__main__":
    main()
