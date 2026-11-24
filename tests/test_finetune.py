"""Fine-tuning data handling, label mapping, operating point and checkpoint integration.

The smoke-train test builds a tiny random DeBERTa model and needs the base tokenizer in the
local HuggingFace cache. It is marked ``smoke`` and runs only with a GPU or with
``FINETUNE_SMOKE=1`` (deselect with ``-m "not smoke"``).
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pytest

from data.pheme_rte import (SPLIT_KEYS, RTEPair, load_pheme_rte, load_rte_split,
                            remove_leakage, split_file, text_key)
from pipeline.config import (OPERATING_POINT_FILE, apply_finetuned, finetuned_model_name,
                             resolve_model, resolve_path, slug)
from training.finetune_nli import (Example, both_directions, choose_operating_point,
                                   class_weights, grouped_stratified_split, label_to_model_id,
                                   mean_over_folds)


def _pair(i: int, label: str, a: str | None = None, b: str | None = None, pid: str | None = None) -> RTEPair:
    return RTEPair("test", pid or f"{i}.json_{i + 1000}.json", str(i), str(i + 1000),
                   a or f"premise number {i}", b or f"hypothesis number {i}", label, "")


# --------------------------------------------------------------------------- loader
@pytest.fixture(scope="module")
def real_cfg():
    from pipeline.config import load_config

    return load_config()


def _splits(cfg):
    path = resolve_path(cfg, "pheme_rte_splits")
    if not path.is_dir():
        pytest.skip("leave-one-event-out split files not present")
    return path


@pytest.mark.parametrize("event", sorted(SPLIT_KEYS))
def test_split_loader_and_no_leakage(real_cfg, event):
    splits = _splits(real_cfg)
    train = load_rte_split(splits, event, "train")
    test = load_rte_split(splits, event, "test")
    assert train and test
    assert {p.event for p in test} == {event}
    assert event not in {p.event for p in train}
    assert {p.label for p in train} == {"entailment", "neutral", "contradiction"}
    assert len({p.pair_id for p in train}) == len(train)
    # the test file of a fold is the per-event file of that event
    per_event = load_pheme_rte(resolve_path(real_cfg, "pheme_rte"), event)
    assert {p.pair_id for p in test} == {p.pair_id for p in per_event}
    kept, info = remove_leakage(train, test)
    assert not ({p.pair_id for p in kept} & {p.pair_id for p in test})
    assert not ({text_key(p) for p in kept} & {text_key(p) for p in test})
    assert info["removed_total"] == len(train) - len(kept)


def test_split_file_names(real_cfg):
    splits = _splits(real_cfg)
    assert split_file(splits, "germanwings-crash", "test").name == "pheme_test_leave_gw.xml"
    with pytest.raises(ValueError):
        split_file(splits, "ferguson", "test")
    with pytest.raises(ValueError):
        split_file(splits, "germanwings-crash", "dev")


def test_remove_leakage_by_id_and_reversed_text():
    test = [_pair(1, "contradiction", "a b", "c d"), _pair(2, "entailment")]
    train = [_pair(1, "contradiction"),                               # same id
             _pair(7, "neutral", "c d", "a b"),                       # same texts, reversed
             _pair(8, "neutral")]
    kept, info = remove_leakage(train, test)
    assert [p.pair_id for p in kept] == [train[2].pair_id]
    assert info == {"removed_same_id": 1, "removed_same_text": 1, "removed_total": 2}


# --------------------------------------------------------------------------- training data
def test_both_directions_augmentation():
    pairs = [_pair(1, "contradiction", "x one", "y two"), _pair(2, "entailment", "p", "q")]
    ex = both_directions(pairs)
    assert len(ex) == 4
    assert Example("x one", "y two", "contradiction", "pheme") in ex
    assert Example("y two", "x one", "contradiction", "pheme") in ex
    assert Example("q", "p", "entailment", "pheme") in ex
    assert len(both_directions(pairs, enabled=False)) == 2


def test_grouped_stratified_split_keeps_groups_together():
    labels = ["entailment"] * 40 + ["neutral"] * 30 + ["contradiction"] * 30
    pairs = [_pair(i, lab) for i, lab in enumerate(labels)]
    # reversed duplicates under other ids must follow their twin
    pairs += [_pair(500 + i, p.label, p.hypothesis, p.premise, pid=f"dup{i}") for i, p in enumerate(pairs[:20])]
    train, val = grouped_stratified_split(pairs, 0.1, seed=42)
    assert len(train) + len(val) == len(pairs)
    assert not ({text_key(p) for p in train} & {text_key(p) for p in val})
    val_labels = Counter(p.label for p in val if not p.pair_id.startswith("dup"))
    assert val_labels == {"entailment": 4, "neutral": 3, "contradiction": 3}  # stratified, 10% each
    assert grouped_stratified_split(pairs, 0.1, seed=42) == (train, val)  # deterministic


def test_label_mapping_uses_id2label():
    id2label = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}  # cross-encoder order
    m = label_to_model_id(id2label)
    assert m == {"entailment": 1, "neutral": 2, "contradiction": 0}
    m2 = label_to_model_id({0: "entailment", 1: "neutral", 2: "contradiction"})
    assert m2 == {"entailment": 0, "neutral": 1, "contradiction": 2}
    with pytest.raises(ValueError):
        label_to_model_id({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    w = class_weights(["entailment"] * 6 + ["neutral"] * 3 + ["contradiction"] * 1, m)
    assert w[m["contradiction"]] > w[m["neutral"]] > w[m["entailment"]]


# --------------------------------------------------------------------------- operating point
def test_operating_point_modes(cfg):
    rng = np.random.default_rng(0)
    truth = ["contradiction"] * 50 + ["neutral"] * 50 + ["entailment"] * 50
    c = np.r_[rng.uniform(0.4, 1.0, 50), rng.uniform(0.0, 0.7, 100)]
    probs = np.stack([(1 - c) / 2, (1 - c) / 2, c], axis=1)
    point = choose_operating_point(truth, probs, probs, cfg)
    best, rap = point["modes"]["best_f1"], point["modes"]["recall_at_precision"]
    assert best["aggregation"] in cfg["finetune"]["operating_point"]["aggregations"]
    assert rap["precision"] >= cfg["finetune"]["operating_point"]["min_precision"]
    assert rap["recall"] >= best["recall"] or rap["precision"] >= best["precision"]


def test_mean_over_folds():
    def fold(r):
        res = {"accuracy": r, "macro_f1": r, "contradiction_at_threshold": {"precision": r, "recall": r, "f1": r},
               "contradiction_average_precision": r,
               "flag_rate_by_label": {"entailment": r, "neutral": r, "contradiction": r}}
        return {"test": {"s": res}, "pipeline_on_labelled_pairs": {}}
    m = mean_over_folds({"a": fold(0.2), "b": fold(0.4)})
    assert m["test"]["s"]["contradiction_at_threshold.recall"] == pytest.approx(0.3)


# --------------------------------------------------------------------------- integration
def test_checkpoint_names_resolve_and_cache_names_differ(cfg, tmp_path):
    from nli_scoring.scorer import cache_file_name

    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["paths"]["finetuned_dir"] = str(tmp_path / "models" / "finetuned")
    names = [finetuned_model_name(cfg, f"nli-deberta-v3-base-pheme-leave-{k}") for k in SPLIT_KEYS.values()]
    names.append(finetuned_model_name(cfg, cfg["finetune"]["checkpoint"]))
    assert all(n.startswith("finetuned/") for n in names)
    slugs = {slug(n) for n in names} | {slug(cfg["models"]["nli"])}
    assert len(slugs) == len(names) + 1
    d = tmp_path / "models" / "finetuned" / "nli-deberta-v3-base-pheme-leave-gw"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_model("finetuned/nli-deberta-v3-base-pheme-leave-gw", cfg) == str(d)
    before = cache_file_name("finetuned/nli-deberta-v3-base-pheme-leave-gw", cfg)
    (d / "finetune_meta.json").write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
    after = cache_file_name("finetuned/nli-deberta-v3-base-pheme-leave-gw", cfg)
    assert before != after and "r1" in after
    assert cache_file_name(cfg["models"]["nli"], cfg) == f"nli_{slug(cfg['models']['nli'])}.sqlite"


def test_apply_finetuned_switch(cfg, tmp_path):
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["paths"]["finetuned_dir"] = str(tmp_path / "models" / "finetuned")
    zero_shot = cfg["models"]["nli"]
    apply_finetuned(cfg)  # missing checkpoint: warn and keep zero-shot settings
    assert cfg["models"]["nli"] == zero_shot and cfg["finetune"]["active"] is False
    d = tmp_path / "models" / "finetuned" / cfg["finetune"]["checkpoint"]
    d.mkdir(parents=True)
    (d / OPERATING_POINT_FILE).write_text(json.dumps({"modes": {
        "best_f1": {"aggregation": "mean", "threshold": 0.6},
        "recall_at_precision": {"aggregation": "max_contradiction", "threshold": 0.4}}}), encoding="utf-8")
    cfg["finetune"]["operating_mode"] = "recall_at_precision"
    apply_finetuned(cfg)
    assert cfg["finetune"]["active"] is True
    assert cfg["models"]["nli"] == f"finetuned/{cfg['finetune']['checkpoint']}"
    assert cfg["nli"]["aggregation"] == "max_contradiction"
    assert cfg["disagreement"]["contradiction_threshold"] == 0.4


def test_mock_mode_unaffected_by_finetune_switch(cfg, toy_claims):
    from nli_scoring.scorer import MockNLIScorer, build_scorer

    cfg["finetune"]["use_finetuned"] = True
    apply_finetuned(cfg)
    assert isinstance(build_scorer(cfg), MockNLIScorer)


# --------------------------------------------------------------------------- smoke train
def _smoke_enabled() -> bool:
    if os.environ.get("FINETUNE_SMOKE") == "1":
        return True
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.smoke
@pytest.mark.skipif(not _smoke_enabled(), reason="no GPU (set FINETUNE_SMOKE=1 to run on CPU)")
def test_smoke_train_tiny_model(cfg, tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForSequenceClassification

    from nli_scoring.scorer import NLIScorer, label_permutation
    from training.finetune_nli import score_both, train_model

    try:
        tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("base tokenizer not in the local HuggingFace cache")
    id2label = {0: "neutral", 1: "contradiction", 2: "entailment"}  # deliberately unusual order
    config = DebertaV2Config(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                             num_attention_heads=2, intermediate_size=64, max_position_embeddings=128,
                             num_labels=3, id2label=id2label, label2id={v: k for k, v in id2label.items()})
    torch.manual_seed(0)
    base = tmp_path / "tiny"
    DebertaV2ForSequenceClassification(config).save_pretrained(base)
    tok.save_pretrained(base)

    pairs = [_pair(i, lab, f"the plane had {i} people on board", f"there were {i + 1} people")
             for i, lab in enumerate(["contradiction", "entailment", "neutral"] * 8)]
    train, val = grouped_stratified_split(pairs, 0.25, seed=1)
    cfg["finetune"].update(epochs=1, batch_size=8, eval_batch_size=8, fp16=False)
    out = tmp_path / "ckpt"
    info = train_model(cfg, both_directions(train), val, out, base_model=str(base))
    assert info["best_epoch"] == 1 and len(info["history"]) == 1
    assert info["label_to_model_id"] == {"entailment": 2, "neutral": 0, "contradiction": 1}
    scorer = NLIScorer(str(out), batch_size=8, device="cpu", fp16=False, verbose=False)
    assert scorer.id2label == id2label
    assert label_permutation(scorer.id2label) == [2, 0, 1]
    fwd, back = score_both(scorer, val)
    assert fwd.shape == (len(val), 3) and np.allclose(fwd.sum(axis=1), 1, atol=1e-4)
    point = choose_operating_point([p.label for p in val], fwd, back, cfg)
    assert set(point["modes"]) == {"best_f1", "recall_at_precision"}
