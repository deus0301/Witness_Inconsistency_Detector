"""Shared fixtures. Tests never touch the network: they use mock models and synthetic data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.schema import Claim  # noqa: E402
from pipeline.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """Sample config with mock models and all outputs redirected to a temp folder."""
    config = load_config(sample=True)
    config["models"]["mock"] = True
    config["paths"]["cache_dir"] = str(tmp_path / "cache")
    config["paths"]["output_dir"] = str(tmp_path / "out")
    config["paths"]["figures_dir"] = str(tmp_path / "fig")
    return config


def make_claim(i: int, text: str, account: str | None = None, **meta) -> Claim:
    return Claim(claim_id=f"test:{i}", event="test", thread_id=str(i),
                 account_id=account or f"acct{i}", text=text,
                 timestamp=f"2015-01-01T00:00:{i:02d}+00:00",
                 proxy_label=meta.pop("proxy_label", None), metadata=meta)


@pytest.fixture
def toy_claims() -> list[Claim]:
    texts = [
        "The soldier shot at the war memorial has died",
        "The soldier shot at the war memorial has not died",
        "Police confirm the soldier at the war memorial died in hospital",
        "The soldier at the war memorial is dead according to police",
        "There were 148 people on board the plane that crashed",
        "There were 150 people on board the plane that crashed",
        "The plane that crashed had 150 people on board",
        "Officials say 150 people were on board the crashed plane",
        "The hockey game tonight has been postponed",
        "Tonight's hockey game is postponed after the shooting",
        "The hockey game has been postponed by the league tonight",
    ]
    return [make_claim(i, t) for i, t in enumerate(texts)]


def _tweet(tid: int, text: str, user: str, lang: str = "en", reply_to: int | None = None) -> dict:
    return {"id": tid, "id_str": str(tid), "text": text, "lang": lang,
            "created_at": "Wed Oct 22 14:00:00 +0000 2014",
            "in_reply_to_status_id_str": str(reply_to) if reply_to else None,
            "user": {"id": tid, "id_str": str(tid), "screen_name": user}}


@pytest.fixture
def fake_pheme(tmp_path) -> Path:
    """A tiny PHEME-v2-shaped tree, including macOS ``._`` junk files."""
    root = tmp_path / "pheme"
    threads = [
        ("rumours", 100, "BREAKING: A soldier has been shot at the war memorial http://t.co/x", "newsA",
         {"is_rumour": "rumour", "category": "Soldier shot", "misinformation": 0, "true": "1",
          "is_turnaround": 0}),
        ("rumours", 101, "RT @x: Three separate shooting incidents reported in the city #Ottawa", "newsB",
         {"is_rumour": "rumour", "category": "Three shootings", "misinformation": 1, "true": "0",
          "is_turnaround": 0}),
        ("rumours", 102, "Police now say there were two shooting incidents, not three", "newsC",
         {"is_rumour": "rumour", "category": "Three shootings", "misinformation": 1, "true": "0",
          "is_turnaround": 1}),
        ("rumours", 103, "BREAKING: A soldier has been shot at the war memorial http://t.co/y", "newsD",
         {"is_rumour": "rumour", "category": "Soldier shot", "misinformation": 0, "true": "1",
          "is_turnaround": 0}),
        ("non-rumours", 200, "Our thoughts are with everyone in the city today", "personA",
         {"is_rumour": "nonrumour"}),
        ("non-rumours", 201, "Une fusillade a eu lieu ce matin au parlement", "personB",
         {"is_rumour": "nonrumour"}),
    ]
    for kind, tid, text, user, ann in threads:
        thread = root / "testevent-all-rnr-threads" / kind / str(tid)
        (thread / "source-tweets").mkdir(parents=True)
        (thread / "reactions").mkdir()
        lang = "fr" if tid == 201 else "en"
        (thread / "source-tweets" / f"{tid}.json").write_text(
            json.dumps(_tweet(tid, text, user, lang)), encoding="utf-8")
        (thread / "source-tweets" / f"._{tid}.json").write_bytes(b"\x00\x05junk")
        (thread / "reactions" / f"{tid + 1000}.json").write_text(
            json.dumps(_tweet(tid + 1000, f"@{user} is this really confirmed by anyone?", "replier",
                              reply_to=tid)), encoding="utf-8")
        (thread / "annotation.json").write_text(json.dumps(ann), encoding="utf-8")
        (thread / "structure.json").write_text(json.dumps({str(tid): {str(tid + 1000): []}}),
                                               encoding="utf-8")
    return root
