"""End placement trims old history in large, stable chunks instead of overrunning the budget."""

from __future__ import annotations

from itertools import pairwise

from membrane import ContextCompiler, MemoryItem
from membrane.compiler import MEMORY_PLACEMENT_END, MEMORY_PLACEMENT_SYSTEM


def _history(turns: int) -> list:
    out = []
    for i in range(turns):
        out.append({"role": "user", "content": f"question {i} " + "word " * 40})
        out.append({"role": "assistant", "content": f"answer {i} " + "word " * 40})
    return out


def _mems() -> list:
    return [MemoryItem(id=f"m{i}", content=f"fact {i}") for i in range(3)]


def _cc(**kw) -> ContextCompiler:
    kw.setdefault("memory_placement", MEMORY_PLACEMENT_END)
    return ContextCompiler(max_tokens=2000, include_interrupt_instruction=False, **kw)


def _first_history_message(result):
    roles = [m["role"] for m in result.messages]
    start = 1 if roles[0] == "system" else 0
    return result.messages[start]["content"]


def test_default_placement_is_end():
    assert ContextCompiler(max_tokens=1000).memory_placement == MEMORY_PLACEMENT_END


def test_long_history_is_trimmed_to_budget():
    r = _cc(core_content="Be helpful.").compile(
        memories=_mems(), user_message="now", conversation_messages=_history(60)
    )
    md = r.metadata
    assert md["token_count"] <= 2000
    assert not md.get("budget_exceeded")
    assert md["history_messages_dropped"] > 0
    # Kept history starts on a user turn and ends with the most recent exchange.
    assert r.messages[1]["role"] == "user"
    assert "answer 59" in r.messages[-2]["content"]


def test_memories_survive_a_huge_history():
    r = _cc().compile(memories=_mems(), user_message="now", conversation_messages=_history(200))
    assert r.ref_map, "history must not crowd out every memory"


def test_trim_point_is_stable_across_turns():
    starts = []
    for turns in range(30, 70):
        r = _cc().compile(
            memories=_mems(), user_message="now", conversation_messages=_history(turns)
        )
        starts.append(_first_history_message(r))
    changes = sum(1 for a, b in pairwise(starts) if a != b)
    # The cut moves in big jumps, so most consecutive turns share a cacheable prefix.
    assert changes <= len(starts) // 4


def test_short_history_is_untouched():
    r = _cc().compile(memories=_mems(), user_message="now", conversation_messages=_history(2))
    assert r.metadata.get("history_messages_dropped", 0) == 0
    assert len([m for m in r.messages if m["role"] != "system"]) == 5


def test_trimming_can_be_turned_off():
    r = _cc(trim_history=False).compile(
        memories=_mems(), user_message="now", conversation_messages=_history(60)
    )
    assert r.metadata["budget_exceeded"] is True
    assert r.metadata.get("history_messages_dropped", 0) == 0


def test_tool_results_are_not_orphaned():
    hist = []
    for i in range(40):
        hist += [
            {"role": "user", "content": f"q{i} " + "word " * 30},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"c{i}", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": f"c{i}", "content": "result " * 30},
            {"role": "assistant", "content": f"a{i} " + "word " * 30},
        ]
    r = _cc().compile(memories=_mems(), user_message="now", conversation_messages=hist)
    body = [m for m in r.messages if m["role"] != "system"]
    assert body[0]["role"] == "user"


def test_system_placement_still_available():
    r = ContextCompiler(max_tokens=2000, memory_placement=MEMORY_PLACEMENT_SYSTEM).compile(
        memories=_mems(), user_message="now", conversation_messages=_history(5)
    )
    assert len(r.messages) == 1
