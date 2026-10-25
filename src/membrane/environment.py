"""Request-scoped environment snapshot for epigenetic tagging (somatic recall)."""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentSnapshot:
    """Diagnostic state captured at memory write time."""

    latency_ms: float | None = None
    cpu_load: float | None = None
    tool_failure_count: int | None = None
    loop_retries: int | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "cpu_load": self.cpu_load,
            "tool_failure_count": self.tool_failure_count,
            "loop_retries": self.loop_retries,
            "timestamp": self.timestamp,
        }


_current_snapshot: ContextVar[EnvironmentSnapshot | None] = ContextVar(
    "membrane_environment_snapshot",
    default=None,
)


def set_environment_snapshot(
    latency_ms: float | None = None,
    cpu_load: float | None = None,
    tool_failure_count: int | None = None,
    loop_retries: int | None = None,
    timestamp: str | None = None,
) -> None:
    """Set request-scoped environment state before creating episodic writes."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    snap = EnvironmentSnapshot(
        latency_ms=latency_ms,
        cpu_load=cpu_load,
        tool_failure_count=tool_failure_count,
        loop_retries=loop_retries,
        timestamp=ts,
    )
    _current_snapshot.set(snap)


def set_environment_snapshot_from_state(state: EnvironmentSnapshot) -> None:
    """Set snapshot from an existing EnvironmentSnapshot."""
    _current_snapshot.set(state)


def get_current_environment_state() -> EnvironmentSnapshot | None:
    """Return current request-scoped snapshot, or None."""
    return _current_snapshot.get()


def clear_environment_snapshot() -> None:
    """Clear snapshot (e.g. after request end)."""
    with contextlib.suppress(LookupError):
        _current_snapshot.set(None)


def capture_cpu_load() -> float | None:
    """Return current CPU load 0.0–1.0, or None if unavailable."""
    try:
        import psutil

        return psutil.cpu_percent(interval=None) / 100.0
    except Exception as e:
        logger.debug("Could not sample CPU load: %s", e)
        return None


def snapshot_to_stress_inputs(snapshot: EnvironmentSnapshot | None) -> dict[str, Any] | None:
    """Convert snapshot to dict for compute_stress_from_environment."""
    if snapshot is None:
        return None
    d: dict[str, Any] = {}
    if snapshot.latency_ms is not None:
        d["latency_ms"] = snapshot.latency_ms
    if snapshot.cpu_load is not None:
        d["cpu_load"] = snapshot.cpu_load
    if snapshot.tool_failure_count is not None:
        d["tool_failure_count"] = snapshot.tool_failure_count
    if snapshot.loop_retries is not None:
        d["loop_retries"] = snapshot.loop_retries
    return d if d else None
