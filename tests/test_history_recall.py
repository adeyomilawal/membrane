"""Stable history cut and recall of evicted history in end placement."""

from __future__ import annotations

import random

import pytest

from membrane import ContextCompiler, MemoryItem

MODEL = "openai/gpt-4o"


def _history(n: int) -> list[dict]:
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"question {i} " + "word " * 30})
        out.append({"role": "assistant", "content": f"answer {i} [Mem:1] " + "reply " * 60})
    return out


def _mems(k: int, size: int = 20) -> list[MemoryItem]:
    return [
        MemoryItem(id=str(i), content=f"Fact {i}: " + "detail " * size, memory_type="semantic")
        for i in range(k)
    ]


def test_dropped_count_never_decreases_while_turn_sizes_vary():
    rng = random.Random(3)
    cc = ContextCompiler(max_tokens=3000, model_id=MODEL, core_content="Be brief.")
    dropped = []
    for turns in range(5, 60):
        r = cc.compile(
            memories=_mems(rng.randint(0, 12), rng.randint(5, 40)),
            user_message="now " * rng.randint(1, 80),
            conversation_messages=_history(turns),
        )
        dropped.append(r.metadata["history_messages_dropped"])
    assert dropped == sorted(dropped)
    assert dropped[-1] > 0


def test_evicted_history_is_recalled_in_the_tail_chronologically():
    cc = ContextCompiler(max_tokens=3000, model_id=MODEL, core_content="Be brief.")
    r = cc.compile(memories=_mems(2), user_message="now", conversation_messages=_history(40))
    md = r.metadata
    assert md["history_messages_dropped"] > 0
    assert md["history_messages_recalled"] > 0
    tail = r.messages[-1]["content"]
    assert tail.index("--- Earlier conversation ---") < tail.index("--- Memories ---")
    earlier = [ln for ln in tail.splitlines() if ln.startswith("[Earlier ")]
    numbers = [int(ln.split()[3]) for ln in earlier]
    assert numbers == sorted(numbers)
    # The recalled chunk is the one just before the kept history.
    first_kept = int(r.messages[1]["content"].split()[1])
    assert max(numbers) == first_kept - 1
    # Old citations never claim refs in this turn's ref_map.
    assert "[Mem:1] reply" not in tail
    assert set(r.ref_map.values()) <= {"0", "1"}


def test_recall_yields_to_memories_under_pressure():
    cc = ContextCompiler(max_tokens=2500, model_id=MODEL, core_content="Be brief.")
    r = cc.compile(memories=_mems(60), user_message="now", conversation_messages=_history(40))
    assert "history_messages_recalled" not in r.metadata or (
        r.metadata["history_messages_recalled"] == 0
    )
    assert len(r.ref_map) > 0


def test_tool_messages_are_not_recalled():
    history = [
        {"role": "user", "content": "weather? " + "pad " * 400},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "22C " + "pad " * 400},
        {"role": "assistant", "content": "It is 22C. " + "pad " * 400},
        *_history(20),
    ]
    cc = ContextCompiler(max_tokens=3000, model_id=MODEL, core_content="Be brief.")
    r = cc.compile(memories=[], user_message="now", conversation_messages=history)
    tail = r.messages[-1]["content"]
    assert "[Earlier tool]" not in tail
    assert "22C pad" not in tail


def test_recall_can_be_turned_off():
    cc = ContextCompiler(
        max_tokens=3000, model_id=MODEL, core_content="Be brief.", recall_evicted_history=False
    )
    r = cc.compile(memories=_mems(2), user_message="now", conversation_messages=_history(40))
    assert "[Earlier " not in r.messages[-1]["content"]
    assert "history_messages_recalled" not in r.metadata


def test_history_share_is_validated():
    with pytest.raises(ValueError):
        ContextCompiler(max_tokens=100, history_share=1.5)
