from __future__ import annotations

import json

from membrane.cli.main import main


def test_profile_explain_coding(capsys):
    code = main(["profile", "explain", "coding"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Profile: coding" in out
    assert "deduplication" in out


def test_profile_list_json(capsys):
    code = main(["profile", "list", "--json"])

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert code == 0
    assert set(payload) == {"minimal", "coding", "conversational"}
    assert payload["minimal"]["purpose"]


def test_inspect_json(capsys):
    code = main(
        [
            "inspect",
            "--user-message",
            "Explain retries.",
            "--memory",
            "policy|2|Retries use exponential backoff.",
            "--max-tokens",
            "2048",
            "--json",
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert code == 0
    assert payload["memory_count"] == 1
    assert payload["token_count"] > 0
    assert payload["segments_offered"] >= payload["segments_kept"]


def test_quote_reports_cost(capsys):
    code = main(
        [
            "quote",
            "--user-message",
            "Explain retries.",
            "--memory",
            "policy|2|Retries use exponential backoff.",
            "--max-tokens",
            "2048",
            "--price-per-1k-input-tokens",
            "0.001",
            "--json",
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert code == 0
    assert payload["total_tokens"] > 0
    assert payload["estimated_cost_usd"] > 0
    assert payload["trimming_strategies"]
