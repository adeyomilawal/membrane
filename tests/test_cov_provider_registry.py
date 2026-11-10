"""Coverage tests for membrane.provider_registry."""

import json
import logging

import pytest

from membrane import provider_registry as pr


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv("LLM_CAPABILITY_MAP_PATH", raising=False)
    monkeypatch.setattr(pr, "_capabilities_map", None)
    monkeypatch.setattr(pr, "_context_caps_map", None)
    monkeypatch.setattr(pr, "_warned_unknown", set())
    yield
    pr._capabilities_map = None
    pr._context_caps_map = None


def _use_map(monkeypatch, tmp_path, data, raw=None):
    p = tmp_path / "caps.json"
    p.write_text(raw if raw is not None else json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("LLM_CAPABILITY_MAP_PATH", str(p))
    pr.reload_capability_maps()


SAMPLE = {
    "capabilities": {
        "_default": ["streaming_interrupt", "no_caching"],
        "openai/gpt-4o": ["streaming_interrupt", "prompt_caching"],
        "openai/gpt-4": ["no_caching"],
        "anthropic/claude-3-5-sonnet": ["prompt_caching"],
    },
    "context_caps": {
        "_default": 4096,
        "openai/gpt-4o": 128000,
        "openai/gpt-4": 8192,
        "anthropic/claude-3-5-sonnet": 200000,
    },
}


def test_default_maps_shape():
    caps, ctx = pr._default_maps()
    assert caps == {"_default": ["streaming_interrupt", "no_caching"]}
    assert ctx == {"_default": 8192}


def test_bundled_map_loads():
    caps = pr.get_full_capability_map()
    assert "_default" in caps
    assert pr.get_full_context_caps()["_default"] > 0


def test_normalize():
    assert pr._normalize("  GPT-4.1-Mini ") == "gpt-4-1-mini"
    assert pr._normalize("openai/gpt-4o-2024-08-06") == "openai/gpt-4o"
    assert pr._normalize("openai/gpt-4o-20240806") == "openai/gpt-4o"
    assert pr._normalize("claude-3-opus-latest") == "claude-3-opus"


def test_resolve_key_variants(monkeypatch, tmp_path):
    _use_map(monkeypatch, tmp_path, SAMPLE)
    keys = list(SAMPLE["capabilities"])
    assert pr._resolve_key(None, keys) is None
    assert pr._resolve_key("", keys) is None
    assert pr._resolve_key("openai/gpt-4o", keys) == "openai/gpt-4o"  # exact
    assert pr._resolve_key("gpt-4o", keys) == "openai/gpt-4o"  # bare name
    assert pr._resolve_key("openai/gpt-4o-2024-08-06", keys) == "openai/gpt-4o"  # date suffix
    assert pr._resolve_key("gpt-4o-mini", keys) == "openai/gpt-4o"  # prefix w/ dash
    assert pr._resolve_key("gpt-4-turbo", keys) == "openai/gpt-4"  # longest prefix wins
    assert pr._resolve_key("gpt-4o-latest", keys) == "openai/gpt-4o"
    assert pr._resolve_key("claude-3-5-sonnet-20241022", keys) == "anthropic/claude-3-5-sonnet"
    # must not match on partial word: gpt-40 is not gpt-4 + "-"
    assert pr._resolve_key("gpt-40", keys) is None
    # _default is never a fuzzy-match target
    assert pr._resolve_key("_default-2", keys) is None


def test_resolve_key_unknown_warns_once(monkeypatch, tmp_path, caplog):
    _use_map(monkeypatch, tmp_path, SAMPLE)
    keys = list(SAMPLE["capabilities"])
    with caplog.at_level(logging.WARNING, logger="membrane.provider_registry"):
        assert pr._resolve_key("mystery-1", keys) is None
        assert pr._resolve_key("mystery-1", keys) is None
    assert caplog.text.count("mystery-1") == 1


def test_capabilities_and_caps(monkeypatch, tmp_path):
    _use_map(monkeypatch, tmp_path, SAMPLE)
    assert pr.has_prompt_caching("gpt-4o") is True
    assert pr.has_prompt_caching("gpt-4") is False
    assert pr.has_streaming_interrupt("gpt-4o") is True
    assert pr.has_streaming_interrupt("gpt-4") is False
    assert pr.has_prompt_caching(None) is False
    assert pr.get_capabilities_for_model("unknown-x") == ["streaming_interrupt", "no_caching"]
    assert pr.lookup_context_cap("gpt-4o") == 128000
    assert pr.lookup_context_cap("unknown-x") is None
    assert pr.get_context_cap_for_model("gpt-4o") == 128000
    assert pr.get_context_cap_for_model("unknown-x") == 4096


def test_get_capabilities_returns_copy(monkeypatch, tmp_path):
    _use_map(monkeypatch, tmp_path, SAMPLE)
    got = pr.get_capabilities_for_model("gpt-4o")
    got.append("mutated")
    assert "mutated" not in pr.get_capabilities_for_model("gpt-4o")
    full = pr.get_full_capability_map()
    full["openai/gpt-4o"].append("x")
    assert "x" not in pr.get_full_capability_map()["openai/gpt-4o"]


def test_malformed_sections_are_sanitised(monkeypatch, tmp_path):
    _use_map(
        monkeypatch,
        tmp_path,
        {
            "capabilities": {"a/m": "not-a-list", "a/n": [1, "x"]},
            "context_caps": {"a/m": "bad", "a/n": "300", "a/o": None},
        },
    )
    caps = pr.get_full_capability_map()
    assert caps["a/m"] == ["streaming_interrupt", "no_caching"]
    assert caps["a/n"] == ["1", "x"]
    ctx = pr.get_full_context_caps()
    assert ctx["a/m"] == 8192
    assert ctx["a/n"] == 300
    assert ctx["a/o"] == 8192
    assert ctx["_default"] == 8192  # injected when missing


def test_non_dict_sections_use_defaults(monkeypatch, tmp_path):
    _use_map(monkeypatch, tmp_path, {"capabilities": [1], "context_caps": 5})
    assert pr.get_full_capability_map() == {"_default": ["streaming_interrupt", "no_caching"]}
    assert pr.get_full_context_caps() == {"_default": 8192}


def test_invalid_override_falls_back_to_bundled(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="membrane.provider_registry"):
        _use_map(monkeypatch, tmp_path, None, raw="{not json")
    assert "Invalid capability map" in caplog.text
    assert "_default" in pr.get_full_capability_map()


def test_missing_override_file_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_CAPABILITY_MAP_PATH", str(tmp_path / "nope.json"))
    pr.reload_capability_maps()
    assert "_default" in pr.get_full_capability_map()


def test_everything_fails_uses_builtin_defaults(monkeypatch):
    monkeypatch.setattr(pr, "_load_bundled", lambda: None)
    pr.reload_capability_maps()
    assert pr.get_full_capability_map() == {"_default": ["streaming_interrupt", "no_caching"]}
    assert pr.get_context_cap_for_model("gpt-4o") == 8192


def test_load_json_from_path_non_dict_ok_and_errors(tmp_path):
    good = tmp_path / "g.json"
    good.write_text('{"a": 1}')
    assert pr._load_json_from_path(good) == {"a": 1}
    assert pr._load_json_from_path(tmp_path / "missing.json") is None


def test_load_bundled_failure(monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(pr.Path, "open", boom)
    with caplog.at_level(logging.WARNING, logger="membrane.provider_registry"):
        assert pr._load_bundled() is None
    assert "Bundled" in caplog.text


def test_load_once_is_cached(monkeypatch, tmp_path):
    _use_map(monkeypatch, tmp_path, SAMPLE)
    caps1, ctx1 = pr._load_once()
    caps2, ctx2 = pr._load_once()
    assert caps1 is caps2 and ctx1 is ctx2
    # changing env without reload does not change the cached maps
    monkeypatch.setenv("LLM_CAPABILITY_MAP_PATH", str(tmp_path / "other.json"))
    assert pr.lookup_context_cap("gpt-4o") == 128000
