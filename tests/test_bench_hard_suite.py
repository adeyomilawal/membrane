"""Smoke test for benchmarks/hard_suite.py: scenarios build and Membrane holds its guarantees."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import hard_suite as hs

BY_NAME = {s.name: s for s in hs.SCENARIOS}


def test_scenarios_are_deterministic_and_cover_every_question_kind():
    kinds = set()
    for sc in hs.SCENARIOS:
        turns = hs.build_scenario(sc, 1)
        assert [t.user for t in turns] == [t.user for t in hs.build_scenario(sc, 1)]
        kinds |= {t.kind for t in turns}
    assert kinds == {"memory", "history", "tool", "paste", "multi"}


def test_orphaned_tools_detects_split_pairs():
    call = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]}
    result = {"role": "tool", "tool_call_id": "c1", "content": "r"}
    user = {"role": "user", "content": "hi"}
    assert not hs.orphaned_tools([user, call, result, user])
    assert hs.orphaned_tools([result, user])
    assert hs.orphaned_tools([user, call, user])


def test_membrane_stays_in_budget_valid_and_caches_more_than_naive():
    strategies = {n: hs.STRATEGIES[n] for n in ("naive", "membrane", "membrane-lean")}
    for name in ("tools", "tiny"):
        sc = BY_NAME[name]
        scores = hs.run_scenario(sc, 1, strategies)
        table = {n: hs.summarize(s, scores["naive"]) for n, s in scores.items()}
        assert hs.report(sc, table, 1)
        for n in ("membrane", "membrane-lean"):
            assert table[n]["over"] == 0
            assert table[n]["invalid"] == 0
            assert table[n]["cached"] > table["naive"]["cached"]


def test_membrane_budget_is_never_scaled():
    """Every strategy gets the same real budget: Membrane's token scale is pinned to 1."""
    for name in ("membrane", "membrane-lean"):
        strategy = hs.STRATEGIES[name](4_000, BY_NAME["tiny"])
        assert strategy._compiler.token_scale == 1.0
    turn = hs.build_scenario(BY_NAME["tiny"], 1)[0]
    strategy = hs.STRATEGIES["membrane"](4_000, BY_NAME["tiny"])
    strategy.build(turn, [])
    assert strategy.last_result.metadata["token_scale"] == 1.0


def test_spread_fact_scenarios_need_every_sentence_of_the_fact():
    for name in ("spread-fact", "spread-reworded"):
        turns = hs.build_scenario(BY_NAME[name], 1)
        asked = [t for t in turns if t.kind == "history"]
        assert asked and all(len(t.evidence) == 2 for t in asked)
        for n, t in enumerate(turns):
            # The trip and the code never share a sentence.
            sentences = t.user.split(". ")
            assert not any(f"trip {n}" in s.lower() and t.user[-8:-1] in s for s in sentences)
    reworded = [t for t in hs.build_scenario(BY_NAME["spread-reworded"], 1) if t.kind == "history"]
    assert all(t.user.startswith("Which storage combination went with journey") for t in reworded)


def test_existing_scenarios_did_not_change():
    """New scenarios must not shift the random stream of the nine older ones."""
    import hashlib

    digest = hashlib.sha256()
    for name in list(BY_NAME)[:9]:
        for seed in (1, 5):
            for t in hs.build_scenario(BY_NAME[name], seed):
                contents = [m.content for m in t.memories]
                row = (t.user, t.follow, t.needles, t.kind, contents, t.evidence)
                digest.update(repr(row).encode())
    assert digest.hexdigest() == (
        "e81d16eda3300ca8ff033a27aa8ad5c653a9e54a8f710aedde16435b8a7d639f"
    )


def test_priompt_node_builds_from_the_cutoff_the_package_returns(monkeypatch):
    """No Node needed: the package's answer is faked, the request is assembled here."""
    sc = BY_NAME["tools"]
    turns = hs.build_scenario(sc, 1)
    history: list[hs.Message] = []
    for turn in turns[:6]:
        history = [*history, {"role": "user", "content": turn.user}, *turn.follow]
    asked: list[dict] = []

    def fake(cls, request):
        asked.append(request)
        return {"cutoff": 1_004, "tokens": 0}

    monkeypatch.setattr(hs.PriomptNode, "_ask", classmethod(fake))
    strategy = hs.PriomptNode.__new__(hs.PriomptNode)
    strategy.budget = 1_000_000
    messages, glued = strategy.build(turns[6], history)
    assert not glued and not hs.orphaned_tools(messages)
    # Turns 4 and 5 (priority 1004, 1005) and every memory are at or above the cutoff.
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        turns[4].user, turns[5].user, turns[6].user
    ]
    assert all(m.content in messages[0]["content"] for m in turns[6].memories)
    request = asked[0]
    assert [t["prio"] for t in request["history"]] == [1_000 + i for i in range(6)]
    roles = [m["role"] for m in request["history"][0]["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert "lookup_badge" in request["history"][0]["messages"][1]["content"]

    # Over budget after the package's cutoff: the shared glue trims and reports it.
    strategy.budget = hs.billed(messages) - 1
    trimmed, glued = strategy.build(turns[6], history)
    assert glued and hs.billed(trimmed) <= strategy.budget


def test_priompt_node_is_skipped_without_node(monkeypatch):
    monkeypatch.setattr(hs.shutil, "which", lambda name: None)
    ready, missing = hs.available_strategies()
    assert "priompt-node" in missing and "priompt-node-recent" in missing
    assert "priompt" in ready and "membrane" in ready


def test_priompt_node_agrees_with_the_reimplementation_when_installed():
    try:
        real = hs.PriomptNode(3_000, BY_NAME["tiny"])
    except ImportError:
        import pytest

        pytest.skip("node or benchmarks/priompt_node/node_modules missing")
    sc = BY_NAME["tiny"]
    scores = hs.run_scenario(sc, 1, {"naive": hs.Naive, "py": hs.Priompt, "node": type(real)})
    table = {n: hs.summarize(s, scores["naive"]) for n, s in scores.items()}
    assert table["node"]["invalid"] == 0
    # Same priorities, slightly different token accounting: within 2% of each other.
    assert abs(table["node"]["tokens"] - table["py"]["tokens"]) <= 0.02 * table["py"]["tokens"]
    assert table["node"]["recall"] == table["py"]["recall"]
