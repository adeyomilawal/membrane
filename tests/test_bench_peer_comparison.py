"""Smoke test for benchmarks/peer_comparison.py (offline mode)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import peer_comparison as pc


def test_offline_comparison_runs_and_end_placement_caches_more():
    results = pc.run_all((1,), pc.BUDGET, pc.MEMORIES_PER_TURN)
    report = pc.report(results, False, pc.BUDGET, pc.MEMORIES_PER_TURN, 1)
    assert "membrane-end" in report
    scores = {n: sc[0] for n, sc in results.items()}
    for name in ("naive", "priompt", "priompt-recent", "membrane-end", "membrane-system"):
        assert scores[name].over_budget == 0
    rate = {n: s.cached_tokens / s.input_tokens for n, s in scores.items()}
    assert rate["membrane-end"] > rate["naive"]
    assert all(all(s.recall["memory"]) for s in scores.values())
