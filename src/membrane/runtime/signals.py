"""Signal bus for interrupt, LOD shift, and confidence-gap notifications."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


class SignalKind(str, Enum):
    INTERRUPT = "interrupt"
    LOD_SHIFT = "lod_shift"
    CONFIDENCE_GAP = "confidence_gap"
    STALE_CONTENT_WARNING = "stale_content_warning"


@dataclass
class Signal:
    kind: SignalKind
    query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class SignalHandler(Protocol):
    async def handle(self, signal: Signal) -> Optional[str]: ...


SyncHandler = Callable[[Signal], Optional[str]]


class SignalBus:
    """Subscribe handlers per SignalKind; emit returns first non-None response."""

    def __init__(self) -> None:
        self._handlers: Dict[SignalKind, List[Any]] = {}

    def subscribe(self, kind: SignalKind, handler: Any) -> None:
        self._handlers.setdefault(kind, []).append(handler)

    def unsubscribe(self, kind: SignalKind, handler: Any) -> None:
        lst = self._handlers.get(kind)
        if not lst:
            return
        try:
            lst.remove(handler)
        except ValueError:
            pass

    async def emit(self, signal: Signal) -> Optional[str]:
        import asyncio

        for h in self._handlers.get(signal.kind, []):
            try:
                if hasattr(h, "handle"):
                    result = await h.handle(signal)
                elif asyncio.iscoroutinefunction(h):
                    result = await h(signal)
                else:
                    result = h(signal)
                if result is not None:
                    return result
            except Exception:
                continue
        return None
