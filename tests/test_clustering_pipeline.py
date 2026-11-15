"""Clustering interface, disagreement scoring and end-to-end pipeline tests (mock models)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from clustering.cluster import ALGORITHMS, assign_noise, cluster_embeddings
from clustering.embed import MockEmbedder
from clustering.evaluate import extrinsic_metrics, intrinsic_metrics
from clustering.labels import central_claim_per_cluster, top_terms_per_cluster
from nli_scoring.scorer import MockNLIScorer
from pipeline.config import load_config, resolve_model
from pipeline.disagreement import cluster_pairs, disagreement_stats
from pipeline.report import render_markdown
from pipeline.run import (accounts_to_claims, parse_accounts_text, read_accounts_file,
                          run_pipeline, save_report, small_input_overrides)
from tests.conftest import make_claim


def _blobs(seed: int = 0, per: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated groups of unit vectors."""
    rng = np.random.default_rng(seed)
    centres = np.eye(8)[:3] * 5
    x = np.vstack([c + rng.normal(scale=0.15, size=(per, 8)) for c in centres])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32), np.repeat(np.arange(3), per)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithms_recover_separated_groups(algorithm):
    if algorithm == "hdbscan":
        pytest.importorskip("umap")
    emb, truth = _blobs()
    params = {"kmeans": {"k_min": 2, "k_max": 6, "k_step": 1},
              "hdbscan": {"min_cluster_size": 4, "min_samples": 2, "umap_dim": None}}.get(algorithm, {})
    labels = cluster_embeddings(emb, algorithm, params, seed=0)
    assert labels.shape == truth.shape and labels.dtype.kind == "i"
    assert extrinsic_metrics(labels, [str(t) for t in truth])["ari"] > 0.95
    assert sorted(set(labels[labels >= 0])) == list(range(len(set(labels[labels >= 0]))))


def test_cluster_edge_cases():
    assert cluster_embeddings(np.zeros((0, 4)), "kmeans").shape == (0,)
    assert cluster_embeddings(np.ones((1, 4)), "kmeans").tolist() == [-1]
    with pytest.raises(ValueError):
        cluster_embeddings(np.ones((5, 4)), "spectral")
    with pytest.raises(ValueError):
        cluster_embeddings(_blobs()[0], "kmeans", {"k": 3}, noise_strategy="drop")


def test_assign_noise_respects_similarity():
    emb = np.array([[1, 0], [1, 0.05], [0, 1], [0.05, 1], [0.99, 0.1], [-1, 0]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([0, 0, 1, 1, -1, -1])
    assert assign_noise(emb, labels, min_sim=0.9).tolist() == [0, 0, 1, 1, 0, -1]


def test_metrics_and_labels():
    emb, truth = _blobs()
    intrinsic = intrinsic_metrics(emb, truth)
    assert intrinsic["n_clusters"] == 3 and intrinsic["silhouette"] > 0.8
    assert intrinsic["noise_fraction"] == 0.0
    assert intrinsic_metrics(emb, np.full(len(truth), -1))["silhouette"] is None
    texts = (["soldier shot memorial"] * 12 + ["plane crash alps"] * 12 + ["hockey game postponed"] * 12)
    terms = top_terms_per_cluster(texts, truth, top_n=3)
    assert "soldier" in " ".join(terms[0]) and "plane" in " ".join(terms[1])
    central = central_claim_per_cluster(emb, truth)
    assert set(central) == {0, 1, 2} and all(truth[i] == c for c, i in central.items())
    # noise points are singletons for extrinsic scoring
    assert extrinsic_metrics(np.full(len(truth), -1), [str(t) for t in truth])["ari"] == pytest.approx(0.0)


def test_cluster_pairs_cross_account_and_cap():
    claims = [make_claim(i, f"claim number {i}", account="same" if i < 2 else None) for i in range(6)]
    pairs = cluster_pairs(range(6), claims, max_pairs=None)
    assert (0, 1) not in pairs and len(pairs) == 14
    assert len(cluster_pairs(range(6), claims, max_pairs=None, cross_account_only=False)) == 15
    capped = cluster_pairs(range(6), claims, max_pairs=5, seed=3)
    assert len(capped) == 5 and capped == cluster_pairs(range(6), claims, max_pairs=5, seed=3)


def test_disagreement_stats():
    probs = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.3, 0.6], [0.2, 0.7, 0.1]], dtype=np.float32)
    stats = disagreement_stats(probs, threshold=0.7, weight_fraction=0.5)
    assert stats["n_contradicting_pairs"] == 1 and stats["flagged"]
    assert stats["contradiction_fraction"] == pytest.approx(0.25)
    assert stats["disagreement_score"] == pytest.approx(0.5 * 0.25 + 0.5 * 0.8)
    assert not disagreement_stats(probs, threshold=0.9)["flagged"]
    assert not disagreement_stats(probs, threshold=0.5, min_pairs=3)["flagged"]
    assert disagreement_stats(np.zeros((0, 3)), 0.5)["disagreement_score"] == 0.0


def test_pipeline_end_to_end_with_mocks(cfg, toy_claims, tmp_path):
    cfg["clustering"]["algorithm"] = "agglomerative"
    cfg["clustering"]["agglomerative"]["distance_threshold"] = 0.7
    cfg["disagreement"]["contradiction_threshold"] = 0.8
    result = run_pipeline(toy_claims, cfg, embedder=MockEmbedder(), scorer=MockNLIScorer(), verbose=False)
    report = result.report
    s = report["summary"]
    assert s["n_claims"] == 11 and s["n_clusters"] >= 2
    assert 0 < s["n_pairs_scored"] < s["n_pairs_all_pairs"]  # clustering prunes comparisons
    assert s["n_flagged_clusters"] >= 1
    flagged_pairs = [p for c in report["clusters"] if c["flagged"] for p in c["contradicting_pairs"]]
    texts = {(p["claim_a"]["text"], p["claim_b"]["text"]) for p in flagged_pairs}
    assert any("has died" in a and "not died" in b or "not died" in a and "has died" in b for a, b in texts)
    for pair in flagged_pairs:
        assert pair["claim_a"]["account_id"] != pair["claim_b"]["account_id"]
        assert pair["p_contradiction"] >= 0.8
    # no verdicts: the report never names a "correct" account
    dumped = json.dumps(report).lower()
    assert "disclaimer" in dumped and "winner" not in dumped and "guilty" not in dumped
    json_path, md_path = save_report(result, tmp_path, "toy")
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"] == s
    assert "Flagged clusters" in md_path.read_text(encoding="utf-8")
    assert render_markdown(report).startswith("# Disagreement report")


def test_pipeline_all_pairs_baseline_and_empty(cfg, toy_claims):
    result = run_pipeline(toy_claims, cfg, embedder=MockEmbedder(), scorer=MockNLIScorer(),
                          all_pairs=True, verbose=False)
    assert result.report["summary"]["n_pairs_scored"] == 55
    assert result.report["summary"]["n_clusters"] == 1
    empty = run_pipeline([], cfg, embedder=MockEmbedder(), scorer=MockNLIScorer(), verbose=False)
    assert empty.report["summary"]["n_claims"] == 0 and empty.report["clusters"] == []


def test_accounts_to_claims(tmp_path):
    accounts = [{"account_id": "witness_1", "text": "The car was red. It drove north at speed."},
                {"text": "The car was blue.\nOk."}]
    claims = accounts_to_claims(accounts, event="case")
    assert [c.account_id for c in claims] == ["witness_1", "witness_1", "account_2"]
    assert claims[0].text == "The car was red."
    txt = tmp_path / "a.txt"
    txt.write_text("First account here.\n\nSecond account here.", encoding="utf-8")
    assert len(read_accounts_file(txt)) == 2
    named = parse_accounts_text("Witness A: The car was red.\n\nIt was 5:30 when I saw it. The car was blue.")
    assert named[0] == {"account_id": "Witness A", "text": "The car was red."}
    assert named[1]["account_id"] == "account_2" and named[1]["text"].startswith("It was 5:30")
    small = small_input_overrides(load_config(), 8)
    assert small["clustering"]["algorithm"] == "agglomerative" and small["clustering"]["kmeans"]["k_max"] == 4
    js = tmp_path / "a.json"
    js.write_text(json.dumps(accounts), encoding="utf-8")
    assert read_accounts_file(js) == accounts


def test_config_loading_and_model_resolution(tmp_path):
    full, sample = load_config(), load_config(sample=True)
    assert full["models"]["nli"] == "cross-encoder/nli-deberta-v3-base"
    assert "snli" not in json.dumps(full).lower()
    assert sample["data"]["max_claims_per_event"] == 60 and full["data"]["max_claims_per_event"] is None
    assert sample["clustering"]["algorithm"] == full["clustering"]["algorithm"]
    cfg = load_config(overrides={"paths": {"models_dir": str(tmp_path)}})
    assert resolve_model("org/some-model", cfg) == "org/some-model"
    local = tmp_path / "org__some-model"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_model("org/some-model", cfg) == str(local)
