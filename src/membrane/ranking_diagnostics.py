"""Unified per-segment ranking diagnostics (Phase 0 roadmap formula).

The compiler still runs the staged pipeline (freshness → dedup → rerank → firewall).
This module computes a **diagnostic** score aligned with::

    FinalScore = (relevance_norm × W₁) + (tier_priority × W₂) − (age_decay × W₃)

- ``relevance_norm``: min–max normalized reranker densities when a reranker ran;
  otherwise 0.5 per segment (neutral).
- ``tier_priority``: procedural=1.0, semantic=0.5, episodic=0.0 (higher = more
  preferred in tier ordering).
- ``age_decay``: ``1 - freshness_score`` when :attr:`_membrane_freshness_score`
  was set by :class:`~membrane.freshness.manager.FreshnessManager`; otherwise 0
  (no extra penalty — freshness may still be folded into ``relevance_score``).

These values explain *why* segments were ordered; they do not replace firewall logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from membrane.firewall import TIER_EPISODIC, TIER_PROCEDURAL, TIER_SEMANTIC


@dataclass(frozen=True)
class RankingWeights:
    """Weights for the diagnostic formula (tuning explainability, not firewall math)."""

    w_relevance: float = 1.0
    w_tier: float = 1.0
    w_age: float = 1.0


def tier_priority_01(tier: int) -> float:
    """Higher = procedural-first preference, on [0, 1]."""
    if tier == TIER_PROCEDURAL:
        return 1.0
    if tier == TIER_SEMANTIC:
        return 0.5
    if tier == TIER_EPISODIC:
        return 0.0
    return 0.0


def _normalize_relevance(raw: Optional[Sequence[float]], n: int) -> List[float]:
    if raw is None or len(raw) != n or n == 0:
        return [0.5] * n
    vals = [float(x) for x in raw]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0.5] * n
    return [(v - lo) / (hi - lo) for v in vals]


def _age_decay_term(obj: Any) -> float:
    """Penalty in [0, 1]; 0 if no explicit freshness score on the object."""
    fs = getattr(obj, "_membrane_freshness_score", None)
    if fs is None:
        return 0.0
    try:
        f = float(fs)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - f))


def compute_ranking_diagnostics(
    retrieved: Sequence[Tuple[Any, int]],
    *,
    segment_weights: Optional[Sequence[float]] = None,
    weights: Optional[RankingWeights] = None,
) -> Dict[str, Any]:
    """
    Build per-memory diagnostic rows and metadata.

    ``retrieved`` is the ``(memory_object, tier)`` list feeding ``build_middle``,
    in firewall order before any middle merge from compression.
    """
    rw = weights or RankingWeights()
    n = len(retrieved)
    rel = _normalize_relevance(segment_weights, n)
    segments: List[Dict[str, Any]] = []
    for i, (obj, tier) in enumerate(retrieved):
        r = rel[i]
        tp = tier_priority_01(tier)
        age = _age_decay_term(obj)
        final = rw.w_relevance * r + rw.w_tier * tp - rw.w_age * age
        fs = getattr(obj, "_membrane_freshness_score", None)
        segments.append(
            {
                "index": i,
                "segment_id": str(getattr(obj, "id", f"seg_{i}")),
                "tier": tier,
                "relevance_norm": r,
                "tier_priority_01": tp,
                "age_decay": age,
                "freshness_score": fs,
                "final_score_diagnostic": final,
            }
        )

    return {
        "formula": (
            "final_score_diagnostic = w_relevance*relevance_norm + "
            "w_tier*tier_priority_01 - w_age*age_decay"
        ),
        "weights": {
            "w_relevance": rw.w_relevance,
            "w_tier": rw.w_tier,
            "w_age": rw.w_age,
        },
        "note": (
            "Diagnostic only. The firewall still sorts by (-rerank_weight, tier) when "
            "rerank weights align with middle segments, else (tier, insertion). "
            "If compression merges the middle into one block, weights may no longer align "
            "with segment count — see firewall_weights_aligned on the compile debug_trace."
        ),
        "segments": segments,
        "memory_segment_count": n,
    }


def firewall_weights_aligned(
    middle_segment_count: int,
    segment_weights: Optional[Sequence[float]],
) -> bool:
    """Whether rerank weights apply to firewall ordering (same length as middle segments)."""
    if not segment_weights:
        return False
    return len(segment_weights) == middle_segment_count
