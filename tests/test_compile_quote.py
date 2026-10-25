"""CompileQuote / estimate() — same pipeline as compile(), optional USD from caller rate."""

from __future__ import annotations

import pytest

from membrane import CompileQuote, ContextCompiler, MemoryItem, compile_quote_from_result


def test_estimate_matches_compile_and_includes_segments():
    compiler = ContextCompiler(
        max_tokens=8192,
        model_id="openai/gpt-4o",
        core_content="You are a helpful assistant.",
    )
    memories = [
        MemoryItem(id="1", content="User prefers concise answers.", memory_type="semantic", priority=2),
    ]
    msg = "Hello."

    quote = compiler.estimate(memories=memories, user_message=msg)
    assert isinstance(quote, CompileQuote)
    direct = compiler.compile(memories=memories, user_message=msg)

    assert quote.compile_result.messages == direct.messages
    assert quote.total_tokens == direct.metadata["token_count"]
    assert quote.tokens_trimmed == direct.metadata["tokens_trimmed"]
    assert quote.tokens_budget == direct.metadata["tokens_budget"]
    assert "segments_offered" in direct.metadata
    assert "segments_kept" in direct.metadata
    assert quote.segments_offered == direct.metadata["segments_offered"]
    assert quote.segments_kept == direct.metadata["segments_kept"]
    assert quote.trimmed_count == max(
        0, quote.segments_offered - quote.segments_kept
    )


def test_estimate_cost_usd_from_caller_rate():
    compiler = ContextCompiler(
        max_tokens=4096,
        model_id="openai/gpt-4o",
        core_content="System.",
    )
    memories = [MemoryItem(id="1", content="x" * 200, memory_type="semantic", priority=2)]
    quote = compiler.estimate(
        memories=memories,
        user_message="y" * 100,
        price_per_1k_input_tokens=0.005,
    )
    expected = (quote.total_tokens / 1000.0) * 0.005
    assert quote.estimated_cost_usd is not None
    assert abs(quote.estimated_cost_usd - expected) < 1e-9


def test_estimate_no_price_yields_none_cost():
    compiler = ContextCompiler(max_tokens=4096, model_id="openai/gpt-4o", core_content="S")
    q = compiler.estimate(
        memories=[MemoryItem(id="1", content="hi", memory_type="semantic", priority=2)],
        user_message="yo",
    )
    assert q.estimated_cost_usd is None


def test_compile_quote_from_result_rejects_negative_price():
    compiler = ContextCompiler(max_tokens=4096, model_id="openai/gpt-4o", core_content="S")
    result = compiler.compile(
        memories=[MemoryItem(id="1", content="a", memory_type="semantic", priority=2)],
        user_message="b",
    )
    with pytest.raises(ValueError, match="price_per_1k_input_tokens"):
        compile_quote_from_result(result, price_per_1k_input_tokens=-1.0)


def test_target_token_budget_adds_strategy_when_over():
    compiler = ContextCompiler(max_tokens=8192, model_id="openai/gpt-4o", core_content="Core " * 50)
    memories = [MemoryItem(id="1", content="m" * 500, memory_type="semantic", priority=2)]
    quote = compiler.estimate(
        memories=memories,
        user_message="short",
        target_token_budget=50,
    )
    assert quote.total_tokens > 50
    assert any("target_token_budget=50" in s for s in quote.trimming_strategies)
