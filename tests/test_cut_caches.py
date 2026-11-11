"""Work done at a history cut is done once: indexes grow cut by cut, and a compiler that
has seen the conversation gives exactly what a fresh one gives."""

from __future__ import annotations

import random
from itertools import pairwise

from membrane import ContextCompiler, MemoryItem
from membrane.compiler import _rank_postings, _recall_candidates, _recall_postings, _RecallIndex
from membrane.firewall import count_tokens

MODEL = "openai/gpt-4o"
CORE = "Be brief and exact.\nCite memories by ref."
WORDS = ["gate", "locker", "badge", "trip", "Lisbon", "Porto", "code", "door", "office", "ferry"]


def _conversation(turns: int, seed: int = 3, paste_at: int = -1) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(turns):
        text = f"Is trip {i} fine? Trip {i} used gate G{7000 + i}. It ended at locker L{i}. "
        text += " ".join(rng.choices(WORDS, k=rng.randint(5, 60)))
        if i == paste_at:
            text += " " + " ".join(f"Clause {k} stays in force." for k in range(1500))
        out.append({"role": "user", "content": text})
        call = {"id": f"c{i}", "function": {"name": "check", "arguments": f'{{"trip": {i}}}'}}
        out.append({"role": "assistant", "content": None, "tool_calls": [call]})
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"Trip {i} is fine."})
        out.append({"role": "assistant", "content": f"Trip {i} noted. " + "All good. " * 12})
    return out


def _run(cc: ContextCompiler, history: list[dict], question: str):
    mems = [MemoryItem(id=str(i), content=f"Office {i} door code: {4000 + i}.") for i in range(6)]
    return cc.compile(memories=mems, user_message=question, conversation_messages=history)


def test_a_compiler_that_saw_every_turn_matches_a_fresh_one_on_every_turn(monkeypatch):
    """Also on turns where a long question pushes the cut ahead and the next turn takes it
    back: earlier cuts stay cached and are not rebuilt or corrupted."""
    import membrane.compiler as compiler

    # The index's size is decided when a cut is first seen, which a fresh compiler sees on a
    # different turn. With the floor at the whole share the size is the same for both.
    monkeypatch.setattr(compiler, "_INDEX_FLOOR_PART", 1)
    full = _conversation(45)
    seen = ContextCompiler(max_tokens=3000, model_id=MODEL, core_content=CORE)
    dropped = []
    for turn in range(2, 45):
        history = full[: 4 * turn]
        question = f"Which locker for trip {turn // 2}?"
        if turn % 9 == 4:
            question += " Context: " + "padding words " * 500
        fresh = ContextCompiler(max_tokens=3000, model_id=MODEL, core_content=CORE)
        a, b = _run(seen, history, question), _run(fresh, history, question)
        assert a.messages == b.messages
        assert a.metadata["token_count"] == b.metadata["token_count"]
        dropped.append(a.metadata["history_messages_dropped"])
    assert any(later < earlier for earlier, later in pairwise(dropped))


def test_recall_index_grown_by_cuts_ranks_like_one_built_at_once():
    full = _conversation(30)
    grown = _RecallIndex()
    for cut in (4, 12, 12, 40, 28, 96):
        evicted = full[:cut]
        key = tuple((m["role"], m.get("content"), 1) for m in evicted)
        assert grown.covers(key)
        candidates, postings, known = grown.upto(key, evicted)
        fresh = _recall_candidates(evicted)
        assert candidates[:known] == fresh
        for query in ("locker for trip 3", "gate G7005 ferry", "nothing shared at all"):
            assert _rank_postings(postings, known, query, 25) == _rank_postings(
                _recall_postings(fresh), len(fresh), query, 25
            )
    other = tuple((m["role"], m.get("content"), 1) for m in _conversation(5, seed=9))
    assert not grown.covers(other)


def test_system_message_counted_line_by_line_is_the_exact_count():
    for seed in range(4):
        cc = ContextCompiler(max_tokens=2600, model_id=MODEL, core_content=CORE)
        r = _run(cc, _conversation(40, seed=seed), "now")
        assert r.metadata["history_index_lines"] > 3
        system = r.messages[0]["content"]
        core = count_tokens(system[: r.metadata["core_chars"]])
        assert core + r.metadata["history_index_tokens"] == count_tokens(system)


def test_a_long_message_is_tokenised_once_while_it_stays_in_history(monkeypatch):
    import membrane.firewall as fw

    full = _conversation(8, paste_at=6)
    paste = full[24]["content"]
    assert len(paste) > fw._COUNT_CACHE_MAX_CHARS
    encoded: list[int] = []
    encoding = fw._get_encoding()

    class Spy:
        def encode(self, text, *args, **kwargs):
            encoded.append(len(text))
            return encoding.encode(text, *args, **kwargs)

    monkeypatch.setattr(fw, "_tiktoken_encoding", Spy())
    cc = ContextCompiler(max_tokens=30_000, model_id=MODEL, core_content=CORE)
    for turn in range(7, 12):
        history = [*full, *_conversation(turn - 7, seed=11)]
        _run(cc, history, f"question {turn}")
    assert encoded.count(len(paste)) == 1
