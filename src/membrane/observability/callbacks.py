"""Protocols and default implementations for structured metrics emission."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from membrane.observability.events import MetricEvent

logger = logging.getLogger(__name__)


@runtime_checkable
class StructuredMetricsCallback(Protocol):
    """Receives structured MetricEvent instances (distinct from
    membrane.metrics.MetricsCallback).
    """

    def emit(self, event: MetricEvent) -> None: ...


class LoggingStructuredCallback:
    """Emit one DEBUG line per metric event."""

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log or logger

    def emit(self, event: MetricEvent) -> None:
        self._log.debug(
            "membrane_metric name=%s kind=%s value=%s labels=%s",
            event.name,
            event.kind.value,
            event.value,
            event.labels,
        )
