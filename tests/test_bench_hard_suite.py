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
