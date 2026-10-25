"""Pluggable Pre-Flight Analyst (compression) adapters."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class PreFlightAnalyst(Protocol):
    """Async contract for compressing retrieved memory text."""

    async def compress_retrieved_memory(
        self,
        raw_middle_text: str,
        *,
        target_tokens: int,
        user_message_context: str,
        model: str,
    ) -> str:
        ...


class CallablePreFlightAnalyst:
    """Wrap an async callable as PreFlightAnalyst."""

    def __init__(
        self,
        fn: Callable[..., Awaitable[str]],
    ):
        self._fn = fn

    async def compress_retrieved_memory(
        self,
        raw_middle_text: str,
        *,
        target_tokens: int,
        user_message_context: str,
        model: str,
    ) -> str:
        return await self._fn(
            raw_middle_text,
            target_tokens=target_tokens,
            user_message_context=user_message_context,
            model=model,
        )


def as_preflight_analyst(
    fn: Optional[Callable[..., Awaitable[str]]],
) -> Optional[PreFlightAnalyst]:
    """Convert async function to PreFlightAnalyst, or None."""
    if fn is None:
        return None
    return CallablePreFlightAnalyst(fn)
