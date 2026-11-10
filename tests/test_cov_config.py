"""Coverage tests for membrane.config."""

import logging

import pytest

from membrane import config

ALL_VARS = [
    "MEMORY_PREFLIGHT_TOKEN_THRESHOLD",
    "MEMORY_PREFLIGHT_TARGET_TOKENS",
    "PREFLIGHT_ANALYST_MODEL",
    "LOD_ATTENTION_SHIFT_COSINE_THRESHOLD",
    "STRESS_RECENT_TTL_SECONDS",
    "MAX_CACHEABLE_PREFIX_TOKENS",
    "MEMBRANE_PREFLIGHT_CACHE_TTL_SECONDS",
    "MEMORY_OS_KERNEL_PREFLIGHT_CACHE_TTL_SECONDS",
    "MEMBRANE_STRESS_KEY_PREFIX",
    "MEMORY_OS_KERNEL_STRESS_KEY_PREFIX",
    "MEMBRANE_PREFLIGHT_CACHE_PREFIX",
    "MEMORY_OS_KERNEL_PREFLIGHT_CACHE_PREFIX",
    "MEMBRANE_REF_MAP_PREFIX",
    "MEMORY_OS_KERNEL_REF_MAP_PREFIX",
    "MEMBRANE_REF_MAP_TTL_SECONDS",
    "MEMORY_OS_KERNEL_REF_MAP_TTL_SECONDS",
    "MEMBRANE_REDIS_URL",
    "MEMORY_OS_KERNEL_REDIS_URL",
    "REDIS_URL",
    "MEMBRANE_MEMORY_TEXT_MAX_CHARS",
    "LLM_CAPABILITY_MAP_PATH",
]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ALL_VARS:
        monkeypatch.delenv(k, raising=False)


def test_defaults():
    assert config.get_preflight_token_threshold() == 8000
    assert config.get_preflight_target_tokens() == 500
    assert config.get_preflight_analyst_model() == "openai/gpt-4o-mini"
    assert config.get_lod_attention_shift_cosine_threshold() == 0.35
    assert config.get_stress_recent_ttl_seconds() == 300
    assert config.get_max_cacheable_prefix_tokens() == 2000
    assert config.get_preflight_cache_ttl_seconds() == 60
    assert config.get_stress_redis_key_prefix() == "membrane:stress:"
    assert config.get_preflight_cache_key_prefix() == "membrane:preflight:"
    assert config.get_ref_map_cache_key_prefix() == "membrane:ref_map:"
    assert config.get_ref_map_cache_ttl_seconds() == 600
    assert config.get_default_redis_url() is None
    assert config.get_memory_text_max_chars() is None
    assert config.get_llm_capability_map_path() is None


def test_int_env_valid_and_whitespace(monkeypatch):
    monkeypatch.setenv("MEMORY_PREFLIGHT_TOKEN_THRESHOLD", " 123 ")
    assert config.get_preflight_token_threshold() == 123


def test_int_env_invalid_warns_and_defaults(monkeypatch, caplog):
    monkeypatch.setenv("MEMORY_PREFLIGHT_TARGET_TOKENS", "abc")
    with caplog.at_level(logging.WARNING, logger="membrane.config"):
        assert config.get_preflight_target_tokens() == 500
    assert "MEMORY_PREFLIGHT_TARGET_TOKENS" in caplog.text


def test_float_env(monkeypatch, caplog):
    monkeypatch.setenv("LOD_ATTENTION_SHIFT_COSINE_THRESHOLD", " 0.5 ")
    assert config.get_lod_attention_shift_cosine_threshold() == 0.5
    monkeypatch.setenv("LOD_ATTENTION_SHIFT_COSINE_THRESHOLD", "x")
    with caplog.at_level(logging.WARNING, logger="membrane.config"):
        assert config.get_lod_attention_shift_cosine_threshold() == 0.35
    assert "LOD_ATTENTION_SHIFT_COSINE_THRESHOLD" in caplog.text


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("TRUE", True), (" yes ", True), ("on", True), ("0", False), ("nope", False)],
)
def test_bool_env(monkeypatch, raw, expected):
    monkeypatch.setenv("SOME_FLAG", raw)
    assert config._bool_env("SOME_FLAG", not expected) is expected


def test_bool_env_default(monkeypatch):
    monkeypatch.delenv("SOME_FLAG", raising=False)
    assert config._bool_env("SOME_FLAG", True) is True
    assert config._bool_env("SOME_FLAG", False) is False


def test_analyst_model_override_and_blank(monkeypatch):
    monkeypatch.setenv("PREFLIGHT_ANALYST_MODEL", "  anthropic/claude  ")
    assert config.get_preflight_analyst_model() == "anthropic/claude"
    monkeypatch.setenv("PREFLIGHT_ANALYST_MODEL", "   ")
    assert config.get_preflight_analyst_model() == "openai/gpt-4o-mini"


def test_legacy_int_fallback_and_precedence(monkeypatch, caplog):
    monkeypatch.setenv("MEMORY_OS_KERNEL_REF_MAP_TTL_SECONDS", "11")
    assert config.get_ref_map_cache_ttl_seconds() == 11
    monkeypatch.setenv("MEMBRANE_REF_MAP_TTL_SECONDS", "22")
    assert config.get_ref_map_cache_ttl_seconds() == 22
    monkeypatch.setenv("MEMBRANE_REF_MAP_TTL_SECONDS", "bad")
    with caplog.at_level(logging.WARNING, logger="membrane.config"):
        assert config.get_ref_map_cache_ttl_seconds() == 600
    assert "MEMBRANE_REF_MAP_TTL_SECONDS" in caplog.text


def test_legacy_ttl_preflight(monkeypatch):
    monkeypatch.setenv("MEMORY_OS_KERNEL_PREFLIGHT_CACHE_TTL_SECONDS", "7")
    assert config.get_preflight_cache_ttl_seconds() == 7


def test_str_prefixes_legacy_and_blank(monkeypatch):
    monkeypatch.setenv("MEMORY_OS_KERNEL_STRESS_KEY_PREFIX", " old: ")
    assert config.get_stress_redis_key_prefix() == "old:"
    monkeypatch.setenv("MEMBRANE_STRESS_KEY_PREFIX", "new:")
    assert config.get_stress_redis_key_prefix() == "new:"
    monkeypatch.setenv("MEMBRANE_PREFLIGHT_CACHE_PREFIX", "pf:")
    assert config.get_preflight_cache_key_prefix() == "pf:"
    monkeypatch.setenv("MEMBRANE_REF_MAP_PREFIX", "   ")
    # whitespace-only value is truthy for ``or`` but stripped to empty -> default
    assert config.get_ref_map_cache_key_prefix() == "membrane:ref_map:"


def test_other_int_getters(monkeypatch):
    monkeypatch.setenv("STRESS_RECENT_TTL_SECONDS", "9")
    monkeypatch.setenv("MAX_CACHEABLE_PREFIX_TOKENS", "77")
    assert config.get_stress_recent_ttl_seconds() == 9
    assert config.get_max_cacheable_prefix_tokens() == 77


def test_default_redis_url_precedence(monkeypatch):
    monkeypatch.setenv("REDIS_URL", " redis://c ")
    assert config.get_default_redis_url() == "redis://c"
    monkeypatch.setenv("MEMORY_OS_KERNEL_REDIS_URL", "redis://b")
    assert config.get_default_redis_url() == "redis://b"
    monkeypatch.setenv("MEMBRANE_REDIS_URL", "redis://a")
    assert config.get_default_redis_url() == "redis://a"


def test_memory_text_max_chars(monkeypatch):
    monkeypatch.setenv("MEMBRANE_MEMORY_TEXT_MAX_CHARS", "150")
    assert config.get_memory_text_max_chars() == 150
    monkeypatch.setenv("MEMBRANE_MEMORY_TEXT_MAX_CHARS", "0")
    assert config.get_memory_text_max_chars() == 0
    monkeypatch.setenv("MEMBRANE_MEMORY_TEXT_MAX_CHARS", "-5")
    assert config.get_memory_text_max_chars() == 0
    monkeypatch.setenv("MEMBRANE_MEMORY_TEXT_MAX_CHARS", "junk")
    assert config.get_memory_text_max_chars() == 0


def test_capability_map_path(monkeypatch):
    monkeypatch.setenv("LLM_CAPABILITY_MAP_PATH", " /tmp/x.json ")
    assert config.get_llm_capability_map_path() == "/tmp/x.json"
    monkeypatch.setenv("LLM_CAPABILITY_MAP_PATH", "")
    assert config.get_llm_capability_map_path() is None
