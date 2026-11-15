"""NLI scorer interface tests (mock model + a fake HF-style model; no downloads)."""
from __future__ import annotations

import numpy as np
import pytest

from nli_scoring.scorer import (BaseScorer, MockNLIScorer, PairScore, aggregate_directions,
                                build_scorer, label_permutation)


def test_label_permutation_reads_model_config():
    # cross-encoder/nli-deberta-v3-base order
    assert label_permutation({0: "contradiction", 1: "entailment", 2: "neutral"}) == [1, 2, 0]
    # roberta-large-mnli order (upper case)
    assert label_permutation({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}) == [2, 1, 0]
    assert label_permutation({"0": "entailment", "1": "neutral", "2": "contradiction"}) == [0, 1, 2]
    with pytest.raises(ValueError):
        label_permutation({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_aggregate_directions():
    ab = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)
    ba = np.array([[0.1, 0.1, 0.8], [0.6, 0.3, 0.1]], dtype=np.float32)
    agg = aggregate_directions(ab, ba, "max_contradiction")
    np.testing.assert_allclose(agg, [[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
    np.testing.assert_allclose(aggregate_directions(ab, ba, "mean")[0], [0.4, 0.15, 0.45])
    np.testing.assert_allclose(aggregate_directions(ab, ba, "min_contradiction"),
                               [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    with pytest.raises(ValueError):
        aggregate_directions(ab, ba, "nope")


def test_mock_scorer_interface_and_symmetry():
    scorer = MockNLIScorer()
    a, b = "The soldier has died", "The soldier has not died"
    out = scorer.score_pair(a, b)
    assert set(out) == {"entailment", "neutral", "contradiction", "label", "confidence"}
    assert out["label"] == "contradiction"
    assert abs(out["entailment"] + out["neutral"] + out["contradiction"] - 1.0) < 1e-5
    assert scorer.score_pair(b, a) == out  # order independent
    assert scorer.score_pair("The soldier has died", "The soldier has died today")["label"] == "entailment"
    assert scorer.score_pair("The soldier has died", "Bananas are yellow fruit")["label"] == "neutral"
    probs = scorer.score_pairs([(a, b), (a, a)])
    assert probs.shape == (2, 3) and probs.dtype == np.float32
    assert scorer.score_pairs([]).shape == (0, 3)


class _AsymmetricScorer(BaseScorer):
    """Contradiction only when the premise is longer: exercises direction aggregation."""

    def _predict(self, pairs):
        return np.array([(0.1, 0.1, 0.8) if len(a) > len(b) else (0.2, 0.7, 0.1)
                         for a, b in pairs], dtype=np.float32)


def test_symmetric_scoring_takes_max_contradiction():
    scorer = _AsymmetricScorer()
    short, long = "short", "a much longer statement"
    assert scorer.score_pairs([(short, long)], symmetric=False)[0, 2] == pytest.approx(0.1)
    assert scorer.score_pairs([(short, long)])[0, 2] == pytest.approx(0.8)
    assert scorer.score_pairs([(long, short)])[0, 2] == pytest.approx(0.8)


def test_cache_avoids_recomputation(tmp_path):
    path = tmp_path / "cache.sqlite"
    pairs = [("a b c", "a b d"), ("x y", "z w"), ("a b c", "a b d")]
    first = MockNLIScorer(path)
    r1 = first.score_pairs(pairs)
    assert first.n_model_calls == 4  # 2 unique pairs x 2 directions
    first.cache.close()
    second = MockNLIScorer(path)
    r2 = second.score_pairs(pairs)
    assert second.n_model_calls == 0 and second.n_cache_hits == 6
    np.testing.assert_allclose(r1, r2)
    second.cache.close()


def test_pair_score_label():
    assert PairScore(0.1, 0.2, 0.7).label == "contradiction"
    assert PairScore(0.6, 0.3, 0.1).confidence == pytest.approx(0.6)


def test_build_scorer_mock(cfg):
    scorer = build_scorer(cfg)
    assert isinstance(scorer, MockNLIScorer)
    scorer.cache.close()


def test_hf_scorer_with_fake_model(monkeypatch):
    """NLIScorer must reorder logits using id2label, whatever order the model uses."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from nli_scoring.scorer import NLIScorer

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        def __call__(self, a, b, **kwargs):
            # one feature per pair: 1.0 when the premise contains "not"
            return FakeBatch(flag=torch.tensor([[1.0 if "not" in x else 0.0] for x in a]))

    class FakeModel:
        class config:
            id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, flag):
            logits = torch.where(flag > 0, torch.tensor([[9.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 9.0]]))
            return type("Out", (), {"logits": logits})

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda name: FakeTokenizer())
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained",
                        lambda name: FakeModel())
    scorer = NLIScorer("fake", device="cpu", batch_size=2, verbose=False)
    probs = scorer.score_directional([("it is not so", "x"), ("it is so", "x"), ("so", "x")])
    assert probs[0].argmax() == 2  # contradiction column
    assert probs[1].argmax() == 0 and probs[2].argmax() == 0  # entailment column


def test_cache_usable_from_another_thread(tmp_path):
    """Streamlit keeps the scorer alive and calls it from a new thread on every rerun."""
    import threading

    scorer = MockNLIScorer(tmp_path / "cache.sqlite")
    scorer.score_pairs([("a b c", "a b d")])
    errors: list[Exception] = []

    def work():
        try:
            scorer.score_pairs([("a b c", "a b d"), ("new pair here", "another new pair")])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    scorer.cache.close()
    assert errors == []
