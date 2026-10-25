"""Stale survivor metadata and STALE_CONTENT_WARNING on compile_async SignalBus."""

from __future__ import annotations

import pytest

from membrane.compiler import ContextCompiler
from membrane.freshness import ExponentialDecay, FreshnessManager
from membrane.profile import MembraneProfile
from membrane.runtime.signals import Signal, SignalKind, SignalBus


def _two_memories_with_timestamps(ref_time: float):
    """Two semantic memories: one very old (will score stale), one recent."""

    class Mem:
        def __init__(self, mid: str, content: str, ts: float) -> None:
            self.id = mid
            self.content = content
            self.memory_type = "semantic"
            self.priority = 2
            self.timestamp = ts

    stale_m = Mem("stale-id", "stale fact content " * 20, ref_time - 90 * 86400)
    fresh_m = Mem("fresh-id", "fresh fact content " * 20, ref_time - 3600)
    return [stale_m, fresh_m]


def _freshness_profile(ref_time: float) -> MembraneProfile:
    fm = FreshnessManager(
        decay_fn=ExponentialDecay(half_life_seconds=86400.0),
        staleness_threshold=0.3,
        reference_time=ref_time,
    )
    p = MembraneProfile.minimal()
    p.freshness = fm
    return p


def test_compile_sync_populates_stale_survivors_metadata():
    ref_time = 1_700_000_000.0
    memories = _two_memories_with_timestamps(ref_time)
    profile = _freshness_profile(ref_time)

    compiler = ContextCompiler(
        max_tokens=32_000,
        model_id="gpt-4o-mini",
        core_content="You are helpful.",
    )
    result = compiler.compile(
        memories=memories,
        user_message="What do we know?",
        profile=profile,
    )
    survivors = result.metadata.get("stale_survivors", [])
    assert isinstance(survivors, list)
    assert result.metadata.get("stale_survivor_count", 0) == len(survivors)
    assert result.metadata["stale_survivor_count"] > 0
    assert "stale-id" in survivors


@pytest.mark.asyncio
async def test_compile_async_emits_stale_content_warning_on_signal_bus():
    ref_time = 1_700_000_000.0
    memories = _two_memories_with_timestamps(ref_time)
    profile = _freshness_profile(ref_time)

    received: list[Signal] = []

    async def on_stale(sig: Signal) -> None:
        received.append(sig)

    bus = SignalBus()
    bus.subscribe(SignalKind.STALE_CONTENT_WARNING, on_stale)
    profile.signal_bus = bus

    async def passthrough_compress(raw: str, target_tokens: int, query: str) -> str:
        return raw

    compiler = ContextCompiler(
        max_tokens=32_000,
        model_id="gpt-4o-mini",
        core_content="You are helpful.",
    )
    result = await compiler.compile_async(
        memories=memories,
        user_message="What do we know?",
        profile=profile,
        compress_async=passthrough_compress,
    )

    assert result.metadata.get("stale_survivor_count", 0) > 0
    assert len(received) == 1
    assert received[0].kind == SignalKind.STALE_CONTENT_WARNING
    assert received[0].metadata.get("count") == result.metadata["stale_survivor_count"]
    assert received[0].metadata.get("segment_ids") == result.metadata["stale_survivors"]
