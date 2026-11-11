"""Calibrated budgets: a per-model scale between counted tokens and the provider's tokens."""

from __future__ import annotations

import random

import pytest

from membrane import ContextCompiler, MemoryItem
from membrane.provider_registry import get_token_scale
from membrane.tokenizers import TiktokenAdapter, TokenizerRegistry

CLAUDE = "anthropic/claude-opus-5-5"
GPT = "openai/gpt-4o"


def _mems(n: int = 40) -> list[MemoryItem]:
    return [MemoryItem(id=str(i), content=f"Fact {i}: " + "detail " * 30) for i in range(n)]


def _compile(cc: ContextCompiler, **kw):
    return cc.compile(memories=_mems(), user_message="What do you know?", **kw)


def _billed(result) -> int:
    return result.metadata["token_count"] + result.metadata["message_framing_tokens"]


def test_the_registry_seeds_the_scale_per_model():
    assert get_token_scale("gpt-4o") == 1.0
    # Measured with benchmarks/measure_token_scale.py (prose): two tokenizer generations.
    assert get_token_scale("claude-opus-5-5") == 1.51
    assert get_token_scale("claude-haiku-5-5") == 1.51
    assert get_token_scale("claude-opus-4-7") == 1.51
    assert get_token_scale("claude-opus-4-6") == 1.11
    assert get_token_scale("claude-haiku-4-5-20251001") == 1.11
    assert get_token_scale("anthropic/claude-sonnet-4.5") == 1.11
    assert get_token_scale("claude-3-opus") == 1.0  # retired, no entry
    assert get_token_scale("some-local-model") == 1.0
    assert get_token_scale(None) == 1.0


def test_the_first_compile_already_budgets_in_the_providers_tokens():
    r = _compile(ContextCompiler(max_tokens=1500, model_id=CLAUDE))
    md = r.metadata
    assert md["token_scale"] == 1.51 and md["token_scale_samples"] == 0
    assert md["tokens_budget"] == 1500 and md["tokens_budget_counted"] == int(1500 / 1.51)
    assert _billed(r) <= int(1500 / 1.51)
    # Without the scale the same request would have filled 1,500 counted tokens.
    plain = _compile(ContextCompiler(max_tokens=1500, model_id=CLAUDE, token_scale=1.0))
    assert 1000 < _billed(plain) <= 1500
    gpt = _compile(ContextCompiler(max_tokens=1500, model_id=GPT)).metadata
    assert gpt["token_scale"] == 1.0 and "tokens_budget_counted" not in gpt


def test_observe_pulls_the_scale_towards_what_the_provider_bills():
    cc = ContextCompiler(max_tokens=1500, model_id=CLAUDE)
    r = _compile(cc)
    for _ in range(30):
        scale = cc.observe(r, round(_billed(r) * 1.3))
        r = _compile(cc)
    # It settles on the grid at or just above the ratio, never more than 0.1 over.
    assert scale == r.metadata["token_scale"]
    assert 1.3 <= scale <= 1.4
    assert r.metadata["token_scale_samples"] == 30
    assert r.metadata["tokens_budget_counted"] == int(1500 / scale)
    # What the provider would bill for this prompt now fits the budget the user set.
    assert _billed(r) * 1.3 <= 1500


def test_the_scale_rises_at_once_and_falls_only_with_slack():
    cc = ContextCompiler(max_tokens=4000, model_id=GPT)
    r = _compile(cc)
    assert cc.observe(r, _billed(r) * 2) > 1.0, "an under-count is corrected immediately"
    for _ in range(40):
        high = cc.observe(r, _billed(r) * 2)
    assert 2.0 <= high <= 2.1
    # One slightly lower reading does not bring it down; many low ones do.
    assert cc.observe(r, round(_billed(r) * 1.97)) == high
    for _ in range(60):
        low = cc.observe(r, _billed(r))
    assert 1.0 <= low <= 1.1


def test_noisy_readings_leave_the_budget_alone():
    rng = random.Random(7)
    cc = ContextCompiler(max_tokens=4000, model_id=CLAUDE)
    r = _compile(cc)
    scales = [cc.observe(r, round(_billed(r) * rng.uniform(1.38, 1.42))) for _ in range(200)]
    settled = scales[20:]
    assert len(set(settled)) <= 2, "the scale sits on a grid and does not chase the noise"
    assert all(1.40 <= s <= 1.50 for s in settled), "mean plus a margin from the spread"
    assert all(abs(s / 0.05 - round(s / 0.05)) < 1e-6 for s in scales)


def test_a_fixed_scale_is_never_changed_by_observations():
    cc = ContextCompiler(max_tokens=3000, model_id=CLAUDE, token_scale=1.0)
    r = _compile(cc)
    assert cc.observe(r, _billed(r) * 3) == 1.0
    assert _compile(cc).metadata["token_scale"] == 1.0
    with pytest.raises(ValueError):
        ContextCompiler(max_tokens=100, token_scale=0)


def test_bad_readings_are_ignored_and_wild_ones_are_bounded():
    cc = ContextCompiler(max_tokens=3000, model_id=GPT)
    r = _compile(cc)
    assert cc.observe(r, 0) == 1.0 and cc.observe(r, -5) == 1.0
    assert _compile(cc).metadata["token_scale_samples"] == 0
    assert cc.observe(r, _billed(r) * 1000) <= 4.0


def test_with_a_tokenizer_registry_the_counts_are_taken_as_exact():
    registry = TokenizerRegistry()
    registry.register("anthropic/*", TiktokenAdapter("cl100k_base"))
    cc = ContextCompiler(max_tokens=1500, model_id=CLAUDE)
    r = _compile(cc, tokenizer_registry=registry)
    assert r.metadata["token_scale"] == 1.0
    # Observations for the registry's counts do not leak into the plain estimate's scale.
    cc.observe(r, _billed(r) * 2)
    assert _compile(cc, tokenizer_registry=registry).metadata["token_scale"] > 1.0
    assert _compile(cc).metadata["token_scale"] == 1.51


def test_the_cacheable_minimum_is_checked_in_the_providers_tokens():
    core = "\n".join(f"Rule {i}: answer precisely." for i in range(50))  # ~400 counted tokens
    scaled = ContextCompiler(max_tokens=8000, model_id=CLAUDE, core_content=core)
    plain = ContextCompiler(max_tokens=8000, model_id=CLAUDE, core_content=core, token_scale=1.0)
    # 512-token minimum: about 470 counted tokens are about 700 of the provider's.
    assert [p["reason"] for p in _compile(scaled).metadata["cache_breakpoints"]] == ["core"]
    assert _compile(plain).metadata["cache_breakpoints"] == []
