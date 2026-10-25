"""Compile-time metrics and optional callbacks (no network I/O in the free core)."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileMetrics:
    """Structured metrics for one compile call (no prompt content, no memory text)."""

    compile_id: str
    timestamp: float
    wall_time_ms: float
    tokens_in: int
    tokens_trimmed: int
    tokens_budget: int
    segments_offered: int
    segments_kept: int
    ref_map_size: int
    cache_hit: bool
    is_anonymous: bool
    model_id: Optional[str]
    user_id: Optional[str]
    conversation_id: Optional[str]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_serializable_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict for metrics export (extra must be JSON-serializable)."""
        return {
            "compile_id": self.compile_id,
            "timestamp": self.timestamp,
            "wall_time_ms": self.wall_time_ms,
            "tokens_in": self.tokens_in,
            "tokens_trimmed": self.tokens_trimmed,
            "tokens_budget": self.tokens_budget,
            "segments_offered": self.segments_offered,
            "segments_kept": self.segments_kept,
            "ref_map_size": self.ref_map_size,
            "cache_hit": self.cache_hit,
            "is_anonymous": self.is_anonymous,
            "model_id": self.model_id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "extra": dict(self.extra),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_serializable_dict(), separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )


@runtime_checkable
class MetricsCallback(Protocol):
    def on_compile(self, metrics: CompileMetrics) -> None: ...


class NoOpMetricsCallback:
    """Default: no-op (no overhead)."""

    def on_compile(self, metrics: CompileMetrics) -> None:
        return None


class LoggingMetricsCallback:
    """Emit one INFO log line per compile with key numeric fields."""

    def __init__(self, log: Optional[logging.Logger] = None) -> None:
        self._log = log or logger

    def on_compile(self, metrics: CompileMetrics) -> None:
        self._log.info(
            "membrane_compile compile_id=%s wall_ms=%.2f tokens_in=%s trimmed=%s anonymous=%s",
            metrics.compile_id,
            metrics.wall_time_ms,
            metrics.tokens_in,
            metrics.tokens_trimmed,
            metrics.is_anonymous,
        )


def new_compile_id() -> str:
    return str(uuid.uuid4())


def build_compile_metrics(
    *,
    wall_time_ms: float,
    token_count: int,
    tokens_trimmed: int,
    tokens_budget: int,
    segments_offered: int,
    segments_kept: int,
    ref_map_size: int,
    cache_hit: bool,
    is_anonymous: bool,
    model_id: Optional[str],
    user_id: Optional[str],
    conversation_id: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
    compile_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> CompileMetrics:
    return CompileMetrics(
        compile_id=compile_id or new_compile_id(),
        timestamp=timestamp if timestamp is not None else time.time(),
        wall_time_ms=wall_time_ms,
        tokens_in=token_count,
        tokens_trimmed=tokens_trimmed,
        tokens_budget=tokens_budget,
        segments_offered=segments_offered,
        segments_kept=segments_kept,
        ref_map_size=ref_map_size,
        cache_hit=cache_hit,
        is_anonymous=is_anonymous,
        model_id=model_id,
        user_id=user_id,
        conversation_id=conversation_id,
        extra=dict(extra or {}),
    )
