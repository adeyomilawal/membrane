"""Decay functions for memory freshness."""

from __future__ import annotations

import math
from typing import Protocol


class DecayFunction(Protocol):
    def compute(self, age_seconds: float) -> float: ...


class ExponentialDecay:
    def __init__(self, half_life_seconds: float = 86400.0) -> None:
        self._half = max(half_life_seconds, 1e-6)

    def compute(self, age_seconds: float) -> float:
        return math.exp(-math.log(2) * age_seconds / self._half)


class LinearDecay:
    def __init__(self, max_age_seconds: float = 604800.0) -> None:
        self._max = max(max_age_seconds, 1e-6)

    def compute(self, age_seconds: float) -> float:
        return max(0.0, 1.0 - age_seconds / self._max)


class StepDecay:
    def __init__(self, thresholds: list[tuple[float, float]]) -> None:
        self._thresholds = sorted(thresholds, key=lambda x: x[0])

    def compute(self, age_seconds: float) -> float:
        for max_age, weight in self._thresholds:
            if age_seconds <= max_age:
                return weight
        return self._thresholds[-1][1] if self._thresholds else 0.0
