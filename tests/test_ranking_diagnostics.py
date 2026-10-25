"""Ranking diagnostic formula (Phase 0 roadmap) — explainability only."""

from membrane.firewall import TIER_EPISODIC, TIER_PROCEDURAL, TIER_SEMANTIC
from membrane.memory_protocol import MemoryItem
from membrane.ranking_diagnostics import (
    RankingWeights,
    compute_ranking_diagnostics,
    firewall_weights_aligned,
    tier_priority_01,
)


def test_tier_priority_ordering():
    assert tier_priority_01(TIER_PROCEDURAL) > tier_priority_01(TIER_SEMANTIC)
    assert tier_priority_01(TIER_SEMANTIC) > tier_priority_01(TIER_EPISODIC)


def test_compute_ranking_diagnostics_no_rerank_neutral_relevance():
    m1 = MemoryItem(id="a", content="x", memory_type="semantic", priority=2)
    m2 = MemoryItem(id="b", content="y", memory_type="episodic", priority=3)
    retrieved = [(m1, TIER_SEMANTIC), (m2, TIER_EPISODIC)]
    out = compute_ranking_diagnostics(retrieved, segment_weights=None)
    assert out["memory_segment_count"] == 2
    assert all(s["relevance_norm"] == 0.5 for s in out["segments"])
    assert out["segments"][0]["tier_priority_01"] == 0.5
    assert out["segments"][1]["tier_priority_01"] == 0.0


def test_compute_ranking_diagnostics_with_weights_and_freshness():
    m1 = MemoryItem(id="a", content="x", memory_type="semantic", priority=2)
    setattr(m1, "_membrane_freshness_score", 1.0)
    m2 = MemoryItem(id="b", content="y", memory_type="procedural", priority=1)
    setattr(m2, "_membrane_freshness_score", 0.5)
    retrieved = [(m1, TIER_SEMANTIC), (m2, TIER_PROCEDURAL)]
    out = compute_ranking_diagnostics(
        retrieved,
        segment_weights=[0.0, 10.0],
        weights=RankingWeights(w_relevance=1.0, w_tier=1.0, w_age=1.0),
    )
    assert out["segments"][0]["relevance_norm"] == 0.0
    assert out["segments"][1]["relevance_norm"] == 1.0
    assert out["segments"][0]["age_decay"] == 0.0
    assert out["segments"][1]["age_decay"] == 0.5


def test_firewall_weights_aligned():
    assert firewall_weights_aligned(2, [0.1, 0.2]) is True
    assert firewall_weights_aligned(1, [0.1, 0.2]) is False
    assert firewall_weights_aligned(2, None) is False
