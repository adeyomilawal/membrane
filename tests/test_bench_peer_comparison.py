"""Smoke test for benchmarks/peer_comparison.py (offline mode)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import peer_comparison as pc


def test_offline_comparison_runs_and_end_placement_caches_more():
    turns = pc.build_scenario()
    scores = {n: pc.run(n, turns, pc.BUDGET) for n in pc.STRATEGIES}
    report = pc.report(scores, live=False, budget=pc.BUDGET, memories=pc.MEMORIES_PER_TURN)
    assert "membrane-end" in report
    for name in ("naive", "priompt"):
        assert scores[name].over_budget == 0
    rate = {n: s.cached_tokens / s.input_tokens for n, s in scores.items()}
    assert rate["membrane-end"] > rate["naive"]
    assert all(s.recall["memory"] for s in scores.values())
