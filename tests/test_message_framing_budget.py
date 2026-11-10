"""The budget covers provider message framing, not just content."""

from __future__ import annotations

from membrane import ContextCompiler, MemoryItem
from membrane.compiler import DEFAULT_MESSAGE_OVERHEAD_TOKENS
from membrane.firewall import count_tokens


def _billed(messages: list[dict]) -> int:
    per = DEFAULT_MESSAGE_OVERHEAD_TOKENS
    return per + sum(per + count_tokens(str(m.get("content") or "")) for m in messages)


def _history(n: int) -> list[dict]:
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"question {i} " + "word " * 30})
        out.append({"role": "assistant", "content": f"answer {i} " + "reply " * 60})
    return out


MEMS = [
    MemoryItem(id=str(i), content=f"Fact {i}: " + "detail " * 25, memory_type="semantic")
    for i in range(30)
]


def test_billed_tokens_stay_under_budget_end_placement():
    for budget in (1500, 2500, 4000):
        cc = ContextCompiler(max_tokens=budget, model_id="openai/gpt-4o", core_content="Be brief.")
        for turns in (0, 5, 20):
            r = cc.compile(memories=MEMS, user_message="hi", conversation_messages=_history(turns))
            assert _billed(r.messages) <= budget
            assert r.metadata["message_framing_tokens"] == 3 * (len(r.messages) + 1)
            assert "budget_exceeded" not in r.metadata


def test_billed_tokens_stay_under_budget_system_placement():
    cc = ContextCompiler(max_tokens=1200, model_id="openai/gpt-4o", memory_placement="system")
    r = cc.compile(memories=MEMS, user_message="hi")
    assert _billed(r.messages) <= 1200
    assert r.metadata["token_count"] == count_tokens(r.messages[0]["content"])


def test_zero_overhead_budgets_content_only():
    cc = ContextCompiler(max_tokens=1500, model_id="openai/gpt-4o", message_overhead_tokens=0)
    r = cc.compile(memories=MEMS, user_message="hi", conversation_messages=_history(5))
    assert "message_framing_tokens" not in r.metadata
    assert sum(count_tokens(str(m["content"])) for m in r.messages) <= 1500
