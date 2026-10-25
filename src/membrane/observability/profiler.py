"""LatencyProfiler — per-stage timing for ContextCompiler."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import Iterator


@dataclass
class CompileProfile:
    """Wall time and per-stage breakdown for one compile."""

    total_ms: float
    stages: dict[str, float] = field(default_factory=dict)

    @property
    def percentage(self) -> dict[str, float]:
        if self.total_ms <= 0:
            return {k: 0.0 for k in self.stages}
        return {k: (v / self.total_ms) * 100.0 for k, v in self.stages.items()}

    def to_waterfall(self) -> str:
        """ASCII bar chart for terminal debugging."""
        lines: list[str] = []
        total = self.total_ms or 1.0
        for name, ms in sorted(self.stages.items(), key=lambda x: -x[1]):
            pct = (ms / total) * 100.0
            bar_len = max(1, int(pct / 5))
            bar = "#" * bar_len
            lines.append(f"{name:24s} {ms:8.2f} ms {pct:5.1f}% {bar}")
        lines.append(f"{'TOTAL':24s} {self.total_ms:8.2f} ms")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        return {"total_ms": self.total_ms, **self.stages}


class LatencyProfiler:
    """Records time spent in named pipeline stages."""

    def __init__(self) -> None:
        self._stage_ms: dict[str, float] = {}
        self._t0: float = 0.0

    def start_total(self) -> None:
        self._t0 = time.perf_counter()

    def finish_total(self) -> CompileProfile:
        wall = (time.perf_counter() - self._t0) * 1000.0
        return CompileProfile(total_ms=wall, stages=dict(self._stage_ms))

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            self._stage_ms[name] = self._stage_ms.get(name, 0.0) + dt
