"""count_tokens and PreFlightContextFirewall when the budget is exceeded."""

from __future__ import annotations

from membrane import (
    TIER_SEMANTIC,
    PreFlightContextFirewall,
    count_tokens,
)


def test_count_tokens_monotonic_and_positive_for_text():
    assert count_tokens("") == 0
    short = "Hi"
    long_text = "Hello world, this is a longer piece of text for token counting."
    assert count_tokens(long_text) >= count_tokens(short) >= 1


def test_firewall_trims_middle_when_budget_tight():
    """Huge middle segment must be dropped so total stays within max_tokens."""
    core = "Core."
    end = "User: hello"
    small = "[Mem:1] (a|b|c|0.9)"
    huge = "[Mem:2] " + ("x" * 5000)
    max_tokens = 100
    fw = PreFlightContextFirewall(max_tokens=max_tokens)
    middle = [(small, TIER_SEMANTIC), (huge, TIER_SEMANTIC)]
    final, ref_map_kept, total, _stats = fw.run(core, end, middle, ref_map={1: "u1", 2: "u2"})

    assert "Core." in final
    assert "User: hello" in final
    assert total <= max_tokens
    assert total == count_tokens(final)
    assert 2 not in ref_map_kept or "[Mem:2]" not in final
