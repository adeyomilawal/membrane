"""compile_anonymous honours the ContextCompiler options that apply to it."""

from __future__ import annotations

import pytest

from membrane import ContextCompiler, anthropic_request
from membrane.compiler import _INDEX_HEADER, compile_anonymous
from membrane.metrics import CompileMetrics

MODEL = "openai/gpt-4o"
CORE = "\n".join(f"Rule {i}: answer precisely and cite what you rely on." for i in range(120))


def _history(turns: int) -> list[dict]:
    out = []
    for i in range(turns):
        out.append({"role": "user", "content": f"The gate code for site {i} is G{7000 + i}. Ok?"})
        out.append({"role": "assistant", "content": f"Stored {i}. " + "reply " * 60})
    return out


def _anonymous(**kw):
    kw.setdefault("max_tokens", 4000)
    kw.setdefault("model_id", MODEL)
    return compile_anonymous(
        user_message="What is the gate code for site 2?",
        conversation_messages=_history(40),
        core_content=CORE,
        **kw,
    )


def test_end_placement_matches_the_compiler_with_no_memories():
    r = _anonymous(memory_placement="end")
    cc = ContextCompiler(
        max_tokens=4000, model_id=MODEL, core_content=CORE, include_interrupt_instruction=False
    )
    same = cc.compile(
        memories=[],
        user_message="What is the gate code for site 2?",
        conversation_messages=_history(40),
    )
    assert r.messages == same.messages
    md = r.metadata
    assert md["security_flag"] == "anonymous"
    assert md["history_messages_dropped"] > 0 and md["history_index_lines"] > 0
    assert md["history_messages_recalled"] > 0
    assert md["token_count"] + md["message_framing_tokens"] <= 4000
    assert "G7002" in r.messages[0]["content"] or "G7002" in r.messages[-1]["content"]
    assert [p["reason"] for p in md["cache_breakpoints"]] == ["core", "history"]
    assert anthropic_request(r)["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_end_placement_options_are_passed_through():
    plain = _anonymous(memory_placement="end", eviction_index=False, recall_evicted_history=False)
    assert _INDEX_HEADER not in plain.messages[0]["content"]
    assert "history_messages_recalled" not in plain.metadata
    assert "G7002" not in "".join(m["content"] for m in plain.messages)
    kept_all = _anonymous(memory_placement="end", trim_history=False)
    assert kept_all.metadata["history_messages_dropped"] == 0
    assert kept_all.metadata["budget_exceeded"] is True
    small = _anonymous(memory_placement="end", history_share=0.3)
    assert small.metadata["history_messages_dropped"] > _anonymous(
        memory_placement="end"
    ).metadata["history_messages_dropped"]
    assert _anonymous(memory_placement="end", message_overhead_tokens=0).metadata.get(
        "message_framing_tokens"
    ) is None


def test_system_placement_keeps_its_layout_and_reserves_framing():
    r = compile_anonymous(
        user_message="hi", conversation_messages=_history(2), max_tokens=4000, model_id=MODEL,
        core_content="Be brief.",
    )
    assert len(r.messages) == 1 and r.messages[0]["role"] == "system"
    assert "--- Recent conversation ---" in r.messages[0]["content"]
    assert r.metadata["message_framing_tokens"] == 6
    assert "budget_exceeded" not in r.metadata
    tokens = r.metadata["token_count"]
    # A budget the content fits but content plus framing does not.
    tight = compile_anonymous(
        user_message="hi", conversation_messages=_history(2), max_tokens=tokens + 3,
        model_id=MODEL, core_content="Be brief.",
    )
    assert tight.metadata["budget_exceeded"] is True
    loose = compile_anonymous(
        user_message="hi", conversation_messages=_history(2), max_tokens=tokens + 3,
        model_id=MODEL, core_content="Be brief.", message_overhead_tokens=0,
    )
    assert "budget_exceeded" not in loose.metadata


def test_the_budget_is_scaled_for_models_that_count_more():
    kw = dict(user_message="hi", conversation_messages=_history(4), core_content="Be brief.")
    tokens = compile_anonymous(max_tokens=10_000, model_id=MODEL, **kw).metadata["token_count"]
    budget = tokens + 50
    assert "budget_exceeded" not in compile_anonymous(
        max_tokens=budget, model_id=MODEL, **kw
    ).metadata
    claude = compile_anonymous(max_tokens=budget, model_id="claude-opus-5-5", **kw).metadata
    assert claude["token_scale"] == 1.5 and claude["tokens_budget_counted"] == int(budget / 1.5)
    assert claude["budget_exceeded"] is True
    fixed = compile_anonymous(max_tokens=budget, model_id="claude-opus-5-5", token_scale=1.0, **kw)
    assert "budget_exceeded" not in fixed.metadata


def test_metrics_and_bad_arguments():
    seen: list[CompileMetrics] = []

    class Callback:
        def on_compile(self, metrics: CompileMetrics) -> None:
            seen.append(metrics)

    _anonymous(memory_placement="end", metrics_callback=Callback())
    assert len(seen) == 1 and seen[0].is_anonymous is True
    with pytest.raises(ValueError):
        _anonymous(memory_placement="middle")
    with pytest.raises(ValueError):
        _anonymous(token_scale=0)
