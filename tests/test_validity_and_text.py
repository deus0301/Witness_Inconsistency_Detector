"""Evaluation-validity helpers and the free-text layer around NLI (splitter, clauses, grounding)."""
from __future__ import annotations

import numpy as np

from data.cleaning import split_long_claim, split_sentences
from data.pheme_rte import RTEPair
from evaluation.validity import cue_flags, group_bootstrap, group_summary, story_key
from nli_scoring.scorer import cache_file_name
from pipeline.config import load_config
from pipeline.grounding import ground_speaker, speaker_name
from pipeline.run import accounts_to_claims, nli_text


def _pair(event, stories, label, a="a", b="b"):
    return RTEPair(event, f"{a}_{b}", a, b, a, b, label, stories)


# ---------------------------------------------------------------- story groups and bootstrap
def test_story_key_is_order_free_and_per_event():
    p1 = _pair("e1", "Story 1: x; Story 2: y", "contradiction")
    p2 = _pair("e1", "Story 2: y;Story 1: x", "contradiction")
    p3 = _pair("e2", "Story 1: x; Story 2: y", "contradiction")
    assert story_key(p1) == story_key(p2) != story_key(p3)


def test_group_summary_counts_groups_and_mixed_labels():
    pairs = [_pair("e", "A; B", "contradiction"), _pair("e", "A; B", "contradiction"),
             _pair("e", "A", "entailment"), _pair("e", "C; D", "neutral"), _pair("e", "C; D", "entailment")]
    s = group_summary(pairs)["e"]
    assert s["n_story_groups"] == 3 and s["n_mixed_label_groups"] == 1
    assert s["n_contradiction_groups"] == 1 and s["largest_contradiction_group_share"] == 1.0


def test_group_bootstrap_is_wider_than_pair_bootstrap():
    rng = np.random.default_rng(0)
    groups, y, flag = [], [], []
    for g in range(20):  # 20 groups of 50 identical-label pairs; the system is right per group
        label = g % 2 == 0
        hit = rng.random() < 0.7
        for _ in range(50):
            groups.append(f"g{g}")
            y.append(label)
            flag.append(label if hit else not label)
    y, flag = np.array(y), np.array(flag)
    strata = ["s"] * len(y)
    by_group = group_bootstrap(y, groups, strata, {"sys": (flag, None)}, n_boot=300, seed=1)
    by_pair = group_bootstrap(y, [str(i) for i in range(len(y))], strata, {"sys": (flag, None)},
                              n_boot=300, seed=1)
    w = lambda r: r["systems"]["sys"]["f1"]["hi"] - r["systems"]["sys"]["f1"]["lo"]
    assert w(by_group) > 2 * w(by_pair)
    assert by_group["systems"]["sys"]["f1"]["value"] == by_pair["systems"]["sys"]["f1"]["value"]


def test_group_bootstrap_paired_delta_of_identical_systems_is_zero():
    y = np.array([True, False] * 30)
    flag = np.array([True, True, False] * 20)
    r = group_bootstrap(y, [str(i // 3) for i in range(60)], ["s"] * 60,
                        {"a": (flag, None), "b": (flag.copy(), None)}, [("a", "b")], n_boot=100)
    d = r["deltas"]["a - b"]["f1"]
    assert d["value"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_cue_flags():
    pairs = [RTEPair("e", "1", "1", "2", "148 people on board", "150 people on board", "contradiction", ""),
             RTEPair("e", "2", "1", "2", "The gunman is dead", "The gunman is not dead", "contradiction", ""),
             RTEPair("e", "3", "1", "2", "Police are at the scene", "Police at the scene", "entailment", "")]
    f = cue_flags(pairs)
    assert f["cue_number"].tolist() == [True, False, False]
    assert f["cue_negation"].tolist() == [False, True, False]
    assert f["cue_union"].tolist() == [True, True, False]


# ---------------------------------------------------------------- sentence splitter
def test_split_sentences_keeps_abbreviations_together():
    text = ("It was approx. 10 p.m. when the car hit Dr. Patel on St. James Street. "
            "It was 10 p.m. The van left. The U.S. embassy closed.")
    assert split_sentences(text) == [
        "It was approx. 10 p.m. when the car hit Dr. Patel on St. James Street.",
        "It was 10 p.m.", "The van left.", "The U.S. embassy closed."]


def test_split_sentences_simple_mode_is_the_old_rule():
    assert split_sentences("It was approx. 10 p.m. then.", mode="simple") == ["It was approx.", "10 p.m. then."]


def test_split_long_claim():
    short = "The driver stayed at the scene."
    assert split_long_claim(short, 40) == [short]
    assert split_long_claim(short, None) == [short]
    long = ("I was walking my dog along the canal path at the time and I heard a loud bang, and I turned "
            "around towards the junction near the bakery and the pharmacy and I saw that the car had hit "
            "the cyclist, and then he got back into the car and drove off towards the bridge.")
    pieces = split_long_claim(long, 20)
    assert len(pieces) > 1 and all(len(p.split()) <= 20 for p in pieces)
    assert pieces[-1].endswith("drove off towards the bridge.")


# ---------------------------------------------------------------- speaker grounding
def test_ground_speaker():
    assert ground_speaker("I was inside the shop.", "Witness A") == "Witness A was inside the shop."
    assert ground_speaker("My car was in the garage.", "account_2") == "Witness 2's car was in the garage."
    assert ground_speaker("I'm sure we saw it.", "Ann") == "Ann is sure Ann's group saw it."
    assert ground_speaker("They went to the US embassy.", "Ann") == "They went to the US embassy."
    assert speaker_name("account_7") == "Witness 7"
    # affirmative perception reports the event; negated perception stays about the speaker
    assert ground_speaker("I heard two shots.", "Witness A") == "Two shots."
    assert ground_speaker("I didn't see a gun.", "Witness A") == "Witness A didn't see a gun."


def test_accounts_to_claims_uses_text_layer_settings():
    cfg = load_config()
    cfg["data"].update(sentence_splitter="abbrev", max_claim_words=12)
    cfg["nli"]["ground_speaker"] = True
    accounts = [{"account_id": "Witness A",
                 "text": "I met Dr. Patel at approx. 10 p.m. near the bank, and then I walked home along the river."}]
    claims = accounts_to_claims(accounts, event="t", cfg=cfg)
    assert len(claims) == 2 and all(c.metadata.get("clause_of_sentence") for c in claims)
    assert "Dr. Patel" in claims[0].text and claims[0].text.startswith("I met")
    assert nli_text(claims[0]).startswith("Witness A met Dr. Patel")
    plain = accounts_to_claims(accounts, event="t")  # no cfg: no grounding, no clause split
    assert len(plain) == 1 and nli_text(plain[0]) == plain[0].text


def test_cache_file_name_includes_non_default_length():
    cfg = load_config()
    base = cache_file_name("cross-encoder/nli-deberta-v3-base", cfg)
    cfg["nli"]["max_length"] = 256
    assert cache_file_name("cross-encoder/nli-deberta-v3-base", cfg) == base.replace(".sqlite", "_len256.sqlite")
