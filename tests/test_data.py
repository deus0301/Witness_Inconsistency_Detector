"""Schema, cleaning, PHEME loader, MultiNLI loader and proxy-label tests."""
from __future__ import annotations

import json

import pytest

from data.cleaning import clean_text, deduplicate, normalise_for_dedup, split_sentences
from data.multinli import load_multinli
from data.pheme import convert_veracity, list_events, load_pheme_event, parse_timestamp
from data.proxy_labels import (build_proxy_pairs, cluster_proxy_contested,
                               cluster_proxy_positive, pair_proxy_label)
from data.schema import Claim, load_claims, save_claims
from tests.conftest import make_claim


# ---------------------------------------------------------------- schema
def test_claim_roundtrip(tmp_path):
    claim = make_claim(1, "Shots fired on Parliament Hill", proxy_label="true", is_source=True)
    path = tmp_path / "claims.jsonl"
    save_claims([claim], path)
    assert load_claims(path) == [claim]
    assert Claim.from_dict(claim.to_dict()) == claim


@pytest.mark.parametrize("field", ["claim_id", "event", "thread_id", "account_id", "text"])
def test_claim_requires_non_empty_strings(field):
    kwargs = dict(claim_id="a", event="e", thread_id="t", account_id="u", text="hello")
    kwargs[field] = "  "
    with pytest.raises(ValueError):
        Claim(**kwargs)


def test_claim_rejects_unknown_proxy_label():
    with pytest.raises(ValueError):
        Claim(claim_id="a", event="e", thread_id="t", account_id="u", text="x", proxy_label="contradiction")


# ---------------------------------------------------------------- cleaning
def test_clean_text_removes_twitter_noise():
    raw = "RT @cnn: BREAKING: Shots fired &amp; one hurt http://t.co/abc #OttawaShooting \U0001F622 @user"
    cleaned = clean_text(raw)
    assert cleaned == "BREAKING: Shots fired & one hurt Ottawa Shooting"
    for junk in ("http", "@", "#", "RT", "\U0001F622"):
        assert junk not in cleaned


def test_clean_text_handles_empty_and_quotes():
    assert clean_text("") == ""
    assert clean_text("@a @b http://t.co/x") == ""
    assert clean_text("look @ Gaza and the shooting @ the cafe") == "look at Gaza and the shooting at the cafe"
    assert clean_text("“He is dead” …") == '"He is dead" ...'


def test_dedup_keeps_earliest_and_counts():
    a = make_claim(2, "Shots fired on Parliament Hill!")
    b = make_claim(1, "shots fired on parliament hill")
    c = make_claim(3, "Something else entirely happened")
    kept = deduplicate([a, b, c])
    assert [k.claim_id for k in kept] == ["test:1", "test:3"]
    assert kept[0].metadata["n_duplicates"] == 1
    assert kept[0].metadata["duplicate_accounts"] == ["acct2"]
    assert normalise_for_dedup("Hello, World!") == "hello world"


def test_split_sentences():
    assert split_sentences("He ran. She saw him! Was it 5 p.m.? Yes.") == [
        "He ran.", "She saw him!", "Was it 5 p.m.?", "Yes."]


# ---------------------------------------------------------------- PHEME
def test_convert_veracity():
    assert convert_veracity({"misinformation": 0, "true": "1"}) == "true"
    assert convert_veracity({"misinformation": 1, "true": "0"}) == "false"
    assert convert_veracity({"misinformation": 0, "true": 0}) == "unverified"
    assert convert_veracity({"misinformation": 1}) == "false"
    assert convert_veracity({"misinformation": 0}) == "unverified"
    assert convert_veracity({"misinformation": 1, "true": 1}) is None
    assert convert_veracity({"is_rumour": "nonrumour"}) is None


def test_parse_timestamp():
    assert parse_timestamp("Wed Jan 07 11:24:15 +0000 2015") == "2015-01-07T11:24:15+00:00"
    assert parse_timestamp("garbage") is None and parse_timestamp(None) is None


def test_pheme_loader_source_tweets(fake_pheme):
    assert list_events(fake_pheme) == ["testevent"]
    claims = load_pheme_event(fake_pheme, "testevent")
    # French tweet filtered, duplicated soldier tweet folded, junk ._ files ignored
    assert len(claims) == 4
    by_account = {c.account_id: c for c in claims}
    assert "personB" not in by_account and "newsD" not in by_account
    soldier = by_account["newsA"]
    assert soldier.text == "BREAKING: A soldier has been shot at the war memorial"
    assert soldier.proxy_label == "true"
    assert soldier.metadata["rumour_category"] == "Soldier shot"
    assert soldier.metadata["n_duplicates"] == 1
    assert by_account["newsC"].metadata["is_turnaround"] is True
    assert by_account["personA"].proxy_label is None
    assert all(c.metadata["is_source"] for c in claims)


def test_pheme_loader_options(fake_pheme):
    with_replies = load_pheme_event(fake_pheme, "testevent", include_replies=True, dedup=False)
    assert sum(not c.metadata["is_source"] for c in with_replies) == 6  # replies are filtered on their own language
    assert all(c.proxy_label is None for c in with_replies if not c.metadata["is_source"])
    all_langs = load_pheme_event(fake_pheme, "testevent", languages=None, dedup=False)
    assert len(all_langs) == 6
    assert len(load_pheme_event(fake_pheme, "testevent", rumours_only=True)) == 3
    assert len(load_pheme_event(fake_pheme, "testevent", max_claims=2)) == 2
    with pytest.raises(FileNotFoundError):
        load_pheme_event(fake_pheme, "nope")


# ---------------------------------------------------------------- MultiNLI
def test_multinli_loader(tmp_path):
    rows = [
        {"gold_label": "entailment", "sentence1": "A man eats.", "sentence2": "Someone eats.", "genre": "g", "pairID": "1"},
        {"gold_label": "-", "sentence1": "x", "sentence2": "y", "genre": "g", "pairID": "2"},
        {"gold_label": "contradiction", "sentence1": "It rains.", "sentence2": "It is dry.", "genre": "g", "pairID": "3"},
    ]
    (tmp_path / "multinli_1.0_dev_matched.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    examples = load_multinli(tmp_path, "dev_matched")
    assert [e.label for e in examples] == ["entailment", "contradiction"]
    assert len(load_multinli(tmp_path, "dev_matched", max_examples=1)) == 1
    with pytest.raises(ValueError):
        load_multinli(tmp_path, "test")


# ---------------------------------------------------------------- proxy labels
def _rumour(i, story, veracity, turnaround=False):
    return make_claim(i, f"text number {i} about {story}", proxy_label=veracity,
                      rumour_category=story, is_turnaround=turnaround, is_source=True)


def test_pair_proxy_label():
    t1, t2 = _rumour(1, "A", "true"), _rumour(2, "A", "true")
    f1, f_turn = _rumour(3, "B", "false"), _rumour(4, "B", "false", turnaround=True)
    non_rumour = make_claim(5, "no story here at all")
    assert pair_proxy_label(t1, t2) == "entailment"
    assert pair_proxy_label(f1, f_turn) == "contradiction"
    assert pair_proxy_label(t1, f1) == "neutral"
    assert pair_proxy_label(f1, _rumour(6, "B", "false")) is None  # same false story, no debunk
    assert pair_proxy_label(t1, non_rumour) is None


def test_cluster_proxies():
    t, f, f_turn = _rumour(1, "A", "true"), _rumour(2, "B", "false"), _rumour(3, "B", "false", True)
    u = _rumour(4, "C", "unverified")
    assert cluster_proxy_positive([t, f])
    assert cluster_proxy_positive([f, f_turn])
    assert not cluster_proxy_positive([t, u])
    assert not cluster_proxy_positive([f, _rumour(5, "B", "false")])
    assert cluster_proxy_contested([f, u]) and not cluster_proxy_contested([t, u])


def test_build_proxy_pairs_balanced_and_seeded():
    claims = ([_rumour(i, "A", "true") for i in range(6)]
              + [_rumour(10 + i, "B", "false") for i in range(4)]
              + [_rumour(20, "B", "false", turnaround=True)])
    pairs = build_proxy_pairs(claims, per_class=5, seed=1)
    labels = [label for _, _, label in pairs]
    assert labels.count("entailment") == 5 and labels.count("neutral") == 5
    assert labels.count("contradiction") == 4  # only 4 debunk pairs exist
    again = build_proxy_pairs(claims, per_class=5, seed=1)
    assert [(a.claim_id, b.claim_id) for a, b, _ in pairs] == [(a.claim_id, b.claim_id) for a, b, _ in again]


# ---------------------------------------------------------------- PHEME RTE
def test_pheme_rte_loader(tmp_path):
    from data.pheme_rte import load_pheme_rte

    xml = ('<root><pair entailment="CONTRADICTION" id="1.json_2.json" rte_type="x" '
           'rumours="Story 1: a; Story 2: b"><t>148 people on board URL</t><h>#Germanwings: 150 on board</h></pair>'
           '<pair entailment="ENTAILMENT" id="1.json_3.json" rte_type="x" rumours="a"><t>Plane crashed &amp; burned</t>'
           '<h>A plane has crashed</h></pair>'
           '<pair entailment="UNKNOWN" id="2.json_3.json" rte_type="x" rumours="a; b"><t>Same text</t><h>Same text</h></pair>'
           '<pair entailment="ENTAILMENT" id="1.json_3.json" rte_type="x" rumours="a"><t>dup</t><h>dup id</h></pair></root>')
    (tmp_path / "pheme_rte_gw_en.xml").write_text(xml, encoding="utf-8")
    pairs = load_pheme_rte(tmp_path, "germanwings-crash")
    assert [p.label for p in pairs] == ["contradiction", "entailment"]  # identical + duplicate id dropped
    first = pairs[0]
    assert (first.tweet_a, first.tweet_b) == ("1", "2")
    assert first.premise == "148 people on board" and first.hypothesis == "Germanwings: 150 on board"
    assert pairs[1].premise == "Plane crashed & burned"
    assert len(load_pheme_rte(tmp_path, "germanwings-crash", drop_identical=False)) == 3
    with pytest.raises(ValueError):
        load_pheme_rte(tmp_path, "ferguson")
    with pytest.raises(FileNotFoundError):
        load_pheme_rte(tmp_path, "ottawashooting")
