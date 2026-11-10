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
