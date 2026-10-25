"""README-style happy path for ContextCompiler.compile()."""

from __future__ import annotations

from membrane import CompileResult, ContextCompiler, MemoryItem


def test_compile_readme_quickstart():
    """Mirrors the Quickstart block in README.md."""
    compiler = ContextCompiler(
        max_tokens=8192,
        model_id="openai/gpt-4o",
        core_content="You are a helpful assistant.",
    )

    # Two semantic rows: both serialize with `[Mem:N] ...` so ref_map preserves 1 and 2
    # (episodic lines use `[Mem:N|date]:` which the firewall ref extractor does not index).
    memories = [
        MemoryItem(
            id="1", content="User prefers concise answers.", memory_type="semantic", priority=2
        ),
        MemoryItem(
            id="2", content="Also discussed asyncio yesterday.", memory_type="semantic", priority=2
        ),
    ]

    result: CompileResult = compiler.compile(
        memories=memories,
        user_message="Show a producer-consumer example.",
    )

    assert isinstance(result, CompileResult)
    assert result.messages
    # Default end placement: system holds the core; the last message holds memories + query.
    assert result.messages[0]["role"] == "system"
    assert "helpful assistant" in result.messages[0]["content"]
    assert result.messages[-1]["role"] == "user"
    assert "Show a producer-consumer example." in result.messages[-1]["content"]
    assert result.ref_map.get(1) == "1"
    assert result.ref_map.get(2) == "2"
    assert "token_count" in result.metadata
    assert isinstance(result.metadata["token_count"], int)
    assert result.metadata["token_count"] > 0
