"""Stable history cut and recall of evicted history in end placement."""

from __future__ import annotations

import random
import re

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
    numbers = [int(n) for n in re.findall(r"\[Earlier \w+\] (?:question|answer) (\d+)", tail)]
    assert numbers and numbers == sorted(numbers)
    # With nothing in the question to match, the most recently dropped turns come back.
    first_kept = int(r.messages[1]["content"].split()[1])
    assert max(numbers) == first_kept - 1
    # Old citations never claim refs in this turn's ref_map.
    assert "[Mem:1] reply" not in tail
    assert set(r.ref_map.values()) <= {"0", "1"}


def test_recall_yields_to_memories_under_pressure():
    cc = ContextCompiler(max_tokens=2500, model_id=MODEL, core_content="Be brief.")
    r = cc.compile(memories=_mems(60), user_message="now", conversation_messages=_history(40))
    # Every memory is offered its place first; a passage only takes what is left over, so
    # recall costs the memories at most the room of its two header lines.
    without = ContextCompiler(
        max_tokens=2500, model_id=MODEL, core_content="Be brief.", recall_evicted_history=False
    ).compile(memories=_mems(60), user_message="now", conversation_messages=_history(40))
    assert len(without.ref_map) - 1 <= len(r.ref_map) < 60
    assert r.metadata.get("history_messages_recalled", 0) <= 1


def test_relevant_old_passage_beats_recent_ones():
    history = [
        {"role": "user", "content": "The warehouse gate code is 7741-Zulu. " + "pad " * 50},
        {"role": "assistant", "content": "Stored. " + "reply " * 50},
        *_history(60),
    ]
    cc = ContextCompiler(max_tokens=3000, model_id=MODEL, core_content="Be brief.")
    r = cc.compile(
        memories=[], user_message="What is the warehouse gate code?", conversation_messages=history
    )
    assert r.metadata["history_messages_dropped"] > 100
    assert "7741-Zulu" in r.messages[-1]["content"]
    # Without a matching question the oldest turn does not come back.
    r = cc.compile(memories=[], user_message="now", conversation_messages=history)
    assert "7741-Zulu" not in r.messages[-1]["content"]


def test_long_paste_is_recalled_by_passage():
    clauses = [f"Clause {i} stays in force." for i in range(400)]
    clauses.insert(217, "The vault code in this document is V-90210.")
    history = [
        {"role": "user", "content": "Here is the contract: " + " ".join(clauses)},
        {"role": "assistant", "content": "Read it."},
        *_history(30),
    ]
    cc = ContextCompiler(max_tokens=3000, model_id=MODEL, core_content="Be brief.")
    r = cc.compile(
        memories=[], user_message="What is the vault code?", conversation_messages=history
    )
    tail = r.messages[-1]["content"]
    assert "V-90210" in tail
    assert tail.count("Clause") < 200  # passages, not the whole paste


def test_tool_results_are_recalled_as_text_and_calls_are_not():
    history = [
        {"role": "user", "content": "weather in Lisbon? " + "pad " * 100},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Lisbon forecast: 22C, sunny."},
        {"role": "assistant", "content": "It is warm. " + "pad " * 100},
        *_history(40),
    ]
    cc = ContextCompiler(max_tokens=3000, model_id=MODEL, core_content="Be brief.")
    r = cc.compile(
        memories=[], user_message="What was the Lisbon forecast?", conversation_messages=history
    )
    tail = r.messages[-1]["content"]
    assert "[Earlier tool result] Lisbon forecast: 22C, sunny." in tail
    assert all(m["role"] != "tool" for m in r.messages if "c1" in str(m.get("tool_call_id")))
    assert "tool_calls" not in tail


def test_identifiers_match_whole():
    from membrane.compiler import _terms

    assert _terms("call_12 and ABC-123, trip 12.") == {"call_12", "and", "abc-123", "trip", "12"}


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


def test_a_recalled_passage_that_does_not_fit_is_thinned_to_the_sentence_that_matches():
    history = []
    for i in range(40):
        fact = f"The gate code for site {i} is G{7000 + i}. "
        history.append({"role": "user", "content": "filler words here " * 20 + fact + "more " * 20})
        history.append({"role": "assistant", "content": f"Stored {i}. " + "reply " * 60})
    def run(**kw):
        cc = ContextCompiler(
            max_tokens=2600, model_id=MODEL, core_content="Be brief.", eviction_index=False, **kw
        )
        return cc.compile(
            memories=_mems(30, 30),
            user_message="What is the gate code for site 3?",
            conversation_messages=history,
        )

    r = run()
    tail = r.messages[-1]["content"]
    assert "[Earlier user] The gate code for site 3 is G7003. …" in tail
    assert tail.startswith("(Lines ending in … are excerpts.)")
    assert tail.index("--- Earlier conversation ---") < tail.index("--- Memories ---")
    assert r.metadata["history_messages_recalled"] > 0
    alone = run(recall_evicted_history=False)
    # Passages only take what the memories leave; the one-line excerpt note is the only cost.
    assert len(r.ref_map) >= len(alone.ref_map) - 1


def test_ranking_through_postings_matches_the_plain_ranking():
    import random

    from membrane.compiler import _rank_postings, _rank_recall, _recall_postings, _relevance

    rng = random.Random(5)
    words = [f"w{i}" for i in range(60)]
    for _ in range(50):
        candidates = [
            (f"passage {i}", frozenset(rng.sample(words, rng.randint(1, 12))))
            for i in range(rng.randint(1, 80))
        ]
        query = " ".join(rng.sample(words, rng.randint(0, 10)))
        fast = _rank_postings(_recall_postings(candidates), len(candidates), query, 40)
        plain = _rank_recall(candidates, query)[:40]
        scores = _relevance([terms for _, terms in candidates], query)
        assert len(fast) == len(plain) == min(40, len(candidates))
        # Same order, up to candidates whose scores differ only by rounding.
        for a, b in zip(fast, plain, strict=True):
            assert a == b or abs(scores[a] - scores[b]) < 1e-9
