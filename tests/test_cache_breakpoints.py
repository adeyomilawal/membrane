"""Price-aware layout: cache prices in the registry, breakpoints in metadata, Anthropic helper."""

from __future__ import annotations

import json

from membrane import ContextCompiler, MemoryItem, anthropic_request, get_cache_pricing
from membrane.cache_format import to_anthropic_params
from membrane import compiler as comp
from membrane.compiler import _INDEX_HEADER
from membrane.provider_registry import CachePricing

CLAUDE = "anthropic/claude-opus-5-5"
GPT = "openai/gpt-4o"
CORE = "\n".join(f"Rule {i}: answer precisely and cite what you rely on." for i in range(120))
MARK = {"type": "ephemeral"}


def _history(turns: int, words: int = 40) -> list[dict]:
    out = []
    for i in range(turns):
        out.append({"role": "user", "content": f"question {i} " + "word " * words})
        out.append({"role": "assistant", "content": f"answer {i}. " + "reply " * words})
    return out


def _compile(
    model: str = CLAUDE, max_tokens: int = 8000, turns: int = 12, words: int = 40, **kw
):
    kw.setdefault("token_scale", 1.0)  # budgets in counted tokens, whatever the model
    cc = ContextCompiler(max_tokens=max_tokens, model_id=model, core_content=CORE, **kw)
    return cc.compile(
        memories=[MemoryItem(id="m", content="The user is in Lagos.")],
        user_message="What time is it?",
        conversation_messages=_history(turns, words) if turns else None,
    )


def _markers(params: dict) -> int:
    blocks = [*params.get("system", []), *(b for m in params["messages"] for b in m["content"])]
    return sum("cache_control" in b for b in blocks)


def test_registry_has_cache_prices_per_model():
    opus = get_cache_pricing("claude-opus-5-5")
    assert (opus.mode, opus.write, opus.read, opus.min_tokens) == ("marked", 1.25, 0.05, 512)
    assert get_cache_pricing("anthropic/claude-opus-4-6").min_tokens == 4096
    assert get_cache_pricing("claude-sonnet-4-5-20250929").read == 0.1
    gpt = get_cache_pricing("gpt-4o")
    assert (gpt.mode, gpt.write, gpt.read, gpt.min_tokens) == ("automatic", 1.0, 0.5, 1024)
    assert not gpt.marked and opus.marked
    assert get_cache_pricing("llama3.2") is None
    assert get_cache_pricing(None) is None


def test_breakpoints_after_the_core_and_after_the_history():
    r = _compile()
    points = r.metadata["cache_breakpoints"]
    assert [p["reason"] for p in points] == ["core", "history"]
    core, history = points
    assert core["message_index"] == 0
    assert core["chars"] == len(r.messages[0]["content"])  # no index yet: all of it is core
    assert history["message_index"] == len(r.messages) - 2
    assert history["tokens"] == r.metadata["cacheable_prefix_tokens"] > core["tokens"]
    assert r.metadata["cache_mode"] == "marked"


def test_anthropic_request_places_the_markers():
    r = _compile()
    params = anthropic_request(r)
    assert params["system"] == [
        {"type": "text", "text": r.messages[0]["content"], "cache_control": MARK}
    ]
    assert [m["role"] for m in params["messages"]] == ["user", "assistant"] * 12 + ["user"]
    assert params["messages"][-2]["content"][-1]["cache_control"] == MARK
    assert "cache_control" not in params["messages"][-1]["content"][-1]
    assert params["messages"][-1]["content"][0]["text"] == r.messages[-1]["content"]
    assert _markers(params) == 2
    assert anthropic_request(r, ttl="1h")["system"][0]["cache_control"] == {**MARK, "ttl": "1h"}


def test_the_eviction_index_sits_in_its_own_system_block_after_the_marked_core():
    r = _compile(max_tokens=5000, turns=40)
    assert r.metadata["history_index_lines"] > 0
    system = anthropic_request(r)["system"]
    assert len(system) == 2
    assert system[0]["cache_control"] == MARK and "cache_control" not in system[1]
    assert system[0]["text"] + system[1]["text"] == r.messages[0]["content"]
    assert system[1]["text"].lstrip().startswith(_INDEX_HEADER)


def test_history_is_always_marked_whatever_the_prices(monkeypatch):
    # The marker used to be held back when a cut was expected before the history could be
    # read back. Measured, that saved about 0.1% of cost, so the history is always marked.
    dear = CachePricing(mode="marked", write=2.5, read=0.1, min_tokens=512)
    monkeypatch.setattr(comp, "get_cache_pricing", lambda model_id: dear)
    for turns in range(14, 22):
        r = _compile(max_tokens=3600, turns=turns, words=290)
        md = r.metadata
        assert md["history_messages_dropped"] > 0
        assert "history_cache_expected_reads" not in md
        assert [p["reason"] for p in md["cache_breakpoints"]] == ["core", "history"]
        assert _markers(anthropic_request(r)) == 2
    monkeypatch.undo()
    auto = _compile(GPT, max_tokens=3600, turns=20, words=290)
    assert [p["reason"] for p in auto.metadata["cache_breakpoints"]] == ["core", "history"]
    assert auto.metadata["cache_mode"] == "automatic"


def test_nothing_is_marked_below_the_models_minimum():
    r = _compile("anthropic/claude-opus-4-6", turns=2)  # 4,096-token minimum
    assert r.metadata["cache_breakpoints"] == []
    assert "prompt_cache_used" not in r.metadata
    assert _markers(anthropic_request(r)) == 0
    assert "cache_breakpoints" not in _compile("llama3.2", max_tokens=4000).metadata
    off = ContextCompiler(max_tokens=8000, model_id=CLAUDE, core_content=CORE).compile(
        memories=[], user_message="q", use_prompt_caching_split=False
    )
    assert "cache_breakpoints" not in off.metadata


def test_system_placement_reports_the_core_breakpoint():
    r = _compile(memory_placement="system")
    (point,) = r.metadata["cache_breakpoints"]
    assert point["reason"] == "core" and point["chars"] == r.metadata["cacheable_prefix_chars"]


def test_tool_calls_and_results_become_anthropic_blocks():
    call = lambda i: {  # noqa: E731
        "id": f"c{i}", "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"n": i})},
    }
    messages = [
        {"role": "system", "content": "core"},
        {"role": "user", "content": "look up 1 and 2"},
        {"role": "assistant", "content": "On it.", "tool_calls": [call(1), call(2)]},
        {"role": "tool", "tool_call_id": "c1", "content": "one"},
        {"role": "tool", "tool_call_id": "c2", "content": "two"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "thanks"},
    ]
    params = to_anthropic_params(messages, [{"message_index": 4}, {"message_index": 0}])
    assert params["system"] == [{"type": "text", "text": "core", "cache_control": MARK}]
    roles = [m["role"] for m in params["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assistant = params["messages"][1]["content"]
    assert assistant[0] == {"type": "text", "text": "On it."}
    assert assistant[1] == {"type": "tool_use", "id": "c1", "name": "lookup", "input": {"n": 1}}
    results = params["messages"][2]["content"]  # both results share one user message
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"]
    assert "cache_control" not in results[0] and results[1]["cache_control"] == MARK
    assert messages[4] == {"role": "tool", "tool_call_id": "c2", "content": "two"}  # untouched
    assert "system" not in to_anthropic_params(messages[1:2])
