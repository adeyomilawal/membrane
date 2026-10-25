"""MetricsCollector — counters, gauges, histograms via a StructuredMetricsCallback."""

from __future__ import annotations

from typing import Optional

from membrane.observability.callbacks import LoggingStructuredCallback, StructuredMetricsCallback
from membrane.observability.events import MetricEvent, MetricKind


class MetricsCollector:
    """Collects compile-time metrics and forwards them to a callback."""

    def __init__(self, callback: Optional[StructuredMetricsCallback] = None) -> None:
        self._callback: StructuredMetricsCallback = callback or LoggingStructuredCallback()

    @property
    def callback(self) -> StructuredMetricsCallback:
        return self._callback

    def set_callback(self, callback: StructuredMetricsCallback) -> None:
        self._callback = callback

    def _emit(self, event: MetricEvent) -> None:
        try:
            self._callback.emit(event)
        except Exception:
            pass

    def counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        self._emit(
            MetricEvent(
                name=name,
                kind=MetricKind.COUNTER,
                value=value,
                labels=dict(labels),
            )
        )

    def gauge(self, name: str, value: float, **labels: str) -> None:
        self._emit(
            MetricEvent(
                name=name,
                kind=MetricKind.GAUGE,
                value=value,
                labels=dict(labels),
            )
        )

    def histogram(self, name: str, value: float, **labels: str) -> None:
        self._emit(
            MetricEvent(
                name=name,
                kind=MetricKind.HISTOGRAM,
                value=value,
                labels=dict(labels),
            )
        )
