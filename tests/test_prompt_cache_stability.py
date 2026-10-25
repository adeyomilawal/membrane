"""Prompt-cache prefix stability across consecutive turns.

Providers cache on an exact byte prefix of the request. Retrieved memories change
every turn, so anything placed after them cannot be served from cache.
"""

from __future__ import annotations

from typing import Any, Dict, List

from membrane import ContextCompiler, MemoryItem
from membrane.firewall import count_tokens

MODEL = "openai/gpt-4o"
CORE = "\n".join(
    f"Rule {i}: answer precisely, cite [Mem:N] when you rely on memory." for i in range(60)
)
HISTORY: List[Dict[str, Any]] = [
    {"role": "user", "content": "Hi, I am planning a trip to Lisbon."},
    {"role": "assistant", "content": "Great choice. What dates are you considering?"},
]
U1 = "Mid-May, for five days."
A1 = "May is lovely there. Do you want food or museum suggestions?"
U2 = "Food, mostly seafood."

MEMS_TURN1 = [
    MemoryItem(id="10", content="User is vegetarian on weekdays.", memory_type="semantic", priority=2),
    MemoryItem(id="11", content="User dislikes crowded tours.", memory_type="semantic", priority=2),
]
MEMS_TURN2 = [
    MemoryItem(id="20", content="User loves grilled sardines.", memory_type="semantic", priority=2),
    MemoryItem(id="21", content="User has a shellfish allergy.", memory_type="semantic", priority=2),
]


def _serialize(messages: List[Dict[str, Any]]) -> str:
    return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def _common_prefix(a: str, b: str) -> str:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def _two_turns(**compiler_kw: Any):
    cc = ContextCompiler(max_tokens=8192, model_id=MODEL, core_content=CORE, **compiler_kw)
    t1 = cc.compile(memories=MEMS_TURN1, user_message=U1, conversation_messages=list(HISTORY))
    history2 = HISTORY + [
        {"role": "user", "content": U1},
        {"role": "assistant", "content": A1},
    ]
    t2 = cc.compile(memories=MEMS_TURN2, user_message=U2, conversation_messages=history2)
    return t1, t2


def _stable_prefix_tokens(t1, t2) -> int:
    return count_tokens(_common_prefix(_serialize(t1.messages), _serialize(t2.messages)))


def test_default_placement_breaks_cache_at_first_memory():
    """Confirms the issue: the shared prefix ends where turn-specific memories begin."""
    t1, t2 = _two_turns()
    prefix = _common_prefix(_serialize(t1.messages), _serialize(t2.messages))
    assert len(t1.messages) == 1 and t1.messages[0]["role"] == "system"
    # Prefix is the core and nothing past it; history is never part of a shared prefix.
    assert CORE in prefix
    assert MEMS_TURN1[0].content not in prefix
    assert HISTORY[0]["content"] not in prefix
    assert U1 not in prefix


def test_end_placement_keeps_system_and_history_byte_stable():
    t1, t2 = _two_turns(memory_placement="end")
    # Everything but turn 1's final user message is replayed verbatim in turn 2.
    assert t2.messages[: len(t1.messages) - 1] == t1.messages[:-1]
    assert t1.messages[0] == {"role": "system", "content": t1.messages[0]["content"]}
    assert "[Mem:1]" not in t1.messages[0]["content"]
    assert t2.messages[1:5] == [
        *HISTORY,
        {"role": "user", "content": U1},
        {"role": "assistant", "content": A1},
    ]
    last = t2.messages[-1]
    assert last["role"] == "user"
    # U-shape end: memories first, current user message last.
    assert last["content"].index("[Mem:1]") < last["content"].index(U2)
    assert last["content"].rstrip().endswith(U2)
    assert MEMS_TURN1[0].content not in last["content"]
    assert t2.ref_map == {1: "20", 2: "21"}
    assert t2.metadata["memory_placement"] == "end"
    assert t2.metadata["stable_prefix_messages"] == 5
    assert t2.metadata["prompt_cache_used"] is True


def test_end_placement_no_duplicate_last_turn():
    _, t2 = _two_turns(memory_placement="end")
    assert "--- Last turn ---" not in t2.messages[-1]["content"]


def test_end_placement_directives_leave_system_message():
    cc = ContextCompiler(max_tokens=8192, model_id=MODEL, core_content=CORE, memory_placement="end")
    r = cc.compile(
        memories=MEMS_TURN1,
        user_message=U1,
        conversation_messages=list(HISTORY),
        dynamic_directives=["stress=high: be brief"],
    )
    assert "ACTIVE_SYSTEM_DIRECTIVES" not in r.messages[0]["content"]
    assert "stress=high: be brief" in r.messages[-1]["content"]


def test_end_placement_firewall_trims_and_prunes_ref_map():
    big = [
        MemoryItem(id=str(i), content=f"Fact {i}: " + "detail " * 40, memory_type="semantic", priority=2)
        for i in range(10)
    ]
    roomy = ContextCompiler(max_tokens=100_000, model_id=MODEL, core_content=CORE, memory_placement="end")
    full = roomy.compile(memories=big, user_message=U1, conversation_messages=list(HISTORY))
    budget = full.metadata["token_count"] - 150
    tight = ContextCompiler(max_tokens=budget, model_id=MODEL, core_content=CORE, memory_placement="end")
    r = tight.compile(memories=big, user_message=U1, conversation_messages=list(HISTORY))
    assert r.messages[:-1] == full.messages[:-1]  # history and system never trimmed
    assert 0 < len(r.ref_map) < len(full.ref_map)
    assert r.metadata["tokens_trimmed"] > 0
    assert r.metadata["token_count"] <= budget
    tail = r.messages[-1]["content"]
    for n in full.ref_map:
        assert (f"[Mem:{n}]" in tail) == (n in r.ref_map)


def test_end_placement_async_path_matches_sync():
    import asyncio

    async def never_called(raw: str, target: int, user_msg: str) -> str:
        raise AssertionError("middle is under the pre-flight threshold")

    cc = ContextCompiler(max_tokens=8192, model_id=MODEL, core_content=CORE, memory_placement="end")
    kw = dict(memories=MEMS_TURN1, user_message=U1, conversation_messages=list(HISTORY))
    a = asyncio.run(cc.compile_async(compress_async=never_called, **kw))
    s = cc.compile(**kw)
    assert a.messages == s.messages
    assert a.ref_map == s.ref_map


def test_invalid_memory_placement_rejected():
    import pytest

    with pytest.raises(ValueError):
        ContextCompiler(max_tokens=100, memory_placement="middle")


def _long_turns(n_exchanges: int, **compiler_kw: Any):
    cc = ContextCompiler(max_tokens=100_000, model_id=MODEL, core_content=CORE, **compiler_kw)
    history: List[Dict[str, Any]] = []
    for i in range(n_exchanges):
        history.append({"role": "user", "content": f"Question {i}: what about district {i}?"})
        history.append({"role": "assistant", "content": f"District {i} is known for " + "tiles " * 30})
    t1 = cc.compile(memories=MEMS_TURN1, user_message=U1, conversation_messages=list(history))
    history += [{"role": "user", "content": U1}, {"role": "assistant", "content": A1}]
    t2 = cc.compile(memories=MEMS_TURN2, user_message=U2, conversation_messages=history)
    return t1, t2


def test_stable_prefix_benchmark(capsys):
    """Reports cache-reusable prefix tokens between two consecutive turns."""
    lines = []
    for n in (1, 10, 30):
        before = _stable_prefix_tokens(*_long_turns(n))
        t1, t2 = _long_turns(n, memory_placement="end")
        after = _stable_prefix_tokens(t1, t2)
        total = count_tokens(_serialize(t2.messages))
        lines.append(
            f"[stable-prefix] {2 * n:>3} history msgs | turn-2 request {total:>5} tok | "
            f"reusable: system={before:>5} tok, end={after:>5} tok "
            f"({100 * after / total:.0f}% of request)"
        )
        assert after > before
        # The shared prefix covers the full system + history that turn 1 sent.
        assert after >= count_tokens(_serialize(t1.messages[:-1]))
    with capsys.disabled():
        print("\n" + "\n".join(lines))


def test_end_placement_preserves_tool_call_history():
    history = [
        {"role": "user", "content": "Weather in Lisbon?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "wx", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "22C, sunny"},
        {"role": "assistant", "content": [{"type": "text", "text": "It is 22C and sunny."}]},
    ]
    cc = ContextCompiler(max_tokens=8192, model_id=MODEL, core_content=CORE, memory_placement="end")
    r = cc.compile(memories=MEMS_TURN1, user_message=U1, conversation_messages=history)
    assert r.messages[1:5] == history


def test_end_placement_flags_history_over_budget():
    history = [{"role": "user", "content": "word " * 3000}]
    cc = ContextCompiler(max_tokens=1500, model_id=MODEL, core_content=CORE, memory_placement="end")
    r = cc.compile(memories=MEMS_TURN1, user_message=U1, conversation_messages=history)
    assert r.metadata["budget_exceeded"] is True
    assert r.messages[1] == history[0]  # never trimmed
    assert r.ref_map == {}
    ok = ContextCompiler(max_tokens=8192, model_id=MODEL, core_content=CORE, memory_placement="end")
    assert "budget_exceeded" not in ok.compile(
        memories=MEMS_TURN1, user_message=U1, conversation_messages=list(HISTORY)
    ).metadata
