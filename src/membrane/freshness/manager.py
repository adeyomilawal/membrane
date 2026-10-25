"""Apply freshness weights to memories."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from membrane.freshness.decay import ExponentialDecay


@dataclass
class FreshnessResult:
    segment_id: str
    age_seconds: float
    freshness_score: float
    is_stale: bool
    last_verified: Optional[float] = None


class FreshnessManager:
    def __init__(
        self,
        *,
        decay_fn: Any = None,
        staleness_threshold: float = 0.3,
        reference_time: Optional[float] = None,
    ) -> None:
        self._decay = decay_fn or ExponentialDecay()
        self._staleness_threshold = staleness_threshold
        self._ref = reference_time

    def _now(self) -> float:
        return self._ref if self._ref is not None else time.time()

    def score(
        self,
        memories: Sequence[Any],
        *,
        timestamp_fn: Optional[Callable[[Any], Optional[float]]] = None,
        id_fn: Optional[Callable[[Any], str]] = None,
    ) -> List[FreshnessResult]:
        tf = timestamp_fn or (lambda m: getattr(m, "timestamp", None))
        idf = id_fn or (lambda m: str(getattr(m, "id", id(m))))
        now = self._now()
        out: List[FreshnessResult] = []
        for m in memories:
            ts = tf(m)
            age = 0.0 if ts is None else max(0.0, now - float(ts))
            sc = self._decay.compute(age)
            stale = sc < self._staleness_threshold
            out.append(
                FreshnessResult(
                    segment_id=idf(m),
                    age_seconds=age,
                    freshness_score=sc,
                    is_stale=stale,
                    last_verified=float(ts) if ts is not None else None,
                )
            )
        return out

    def apply_weights(
        self,
        memories: Sequence[Any],
        *,
        timestamp_fn: Optional[Callable[[Any], Optional[float]]] = None,
        weight_fn: Optional[Callable[[Any], float]] = None,
        weight_setter: Optional[Callable[[Any, float], None]] = None,
    ) -> Sequence[Any]:
        scores = self.score(memories, timestamp_fn=timestamp_fn)
        wfn = weight_fn or (lambda m: float(getattr(m, "relevance_score", 1.0) or 1.0))
        setter = weight_setter or self._default_set_weight
        for m, fr in zip(memories, scores):
            setattr(m, "_membrane_freshness_score", fr.freshness_score)
            setattr(m, "_membrane_is_stale", bool(fr.is_stale))
            base = wfn(m)
            setter(m, base * fr.freshness_score)
        return memories

    @staticmethod
    def _default_set_weight(m: Any, w: float) -> None:
        if hasattr(m, "relevance_score"):
            try:
                m.relevance_score = w
            except Exception:
                setattr(m, "relevance_score", w)
        else:
            setattr(m, "_membrane_freshness_weight", w)

    def flag_stale(
        self,
        memories: Sequence[Any],
        *,
        timestamp_fn: Optional[Callable[[Any], Optional[float]]] = None,
    ) -> List[Any]:
        scores = self.score(memories, timestamp_fn=timestamp_fn)
        return [m for m, fr in zip(memories, scores) if fr.is_stale]
