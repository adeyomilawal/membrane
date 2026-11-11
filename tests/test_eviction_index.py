"""Eviction index: one keyframe line per dropped message, changing only when the cut moves."""

from __future__ import annotations

from membrane import ContextCompiler, MemoryItem
from membrane.compiler import _INDEX_HEADER, _cut_starts, _eviction_index
from membrane.firewall import count_tokens

MODEL = "openai/gpt-4o"
CORE = "Be brief."


def _history(n: int, tools: bool = False) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        fact = f"Thanks for that. The gate code for site {i} is G{7000 + i}. "
        out.append({"role": "user", "content": fact + "Please keep it in mind. " * 6})
        if tools:
            call = {"id": f"c{i}", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            out.append({"role": "assistant", "content": None, "tool_calls": [call]})
            out.append(
                {"role": "tool", "tool_call_id": f"c{i}",
                 "content": f"Badge for site {i}: B{9000 + i}. " + "Record field. " * 10}
            )
        out.append({"role": "assistant", "content": f"Stored for site {i}. " + "Understood. " * 20})
    return out


def _cc(**kw) -> ContextCompiler:
    kw.setdefault("max_tokens", 2500)
    return ContextCompiler(model_id=MODEL, core_content=CORE, **kw)


def _index(result) -> list[str]:
    system = result.messages[0]["content"]
    if _INDEX_HEADER not in system:
        return []
    return system.split(_INDEX_HEADER + "\n", 1)[1].splitlines()


def test_every_dropped_message_leaves_its_keyframe_after_the_core():
    r = _cc(max_tokens=4000).compile(
        memories=[], user_message="now", conversation_messages=_history(24)
    )
    dropped = r.metadata["history_messages_dropped"]
    lines = _index(r)
    assert dropped > 0 and len(lines) == dropped == r.metadata["history_index_lines"]
    system = r.messages[0]["content"]
    assert system.startswith(CORE)
    assert r.metadata["core_chars"] == len(system.split("\n\n" + _INDEX_HEADER)[0])
    assert lines[0] == "user: The gate code for site 0 is G7000."
    assert lines[1] == "assistant: Stored for site 0."
    # Kept history follows untouched, starting at the first message that was not dropped.
    assert r.messages[1] == _history(24)[dropped]
    assert r.metadata["token_count"] + r.metadata["message_framing_tokens"] <= 4000


def test_index_changes_only_when_the_cut_moves_and_only_by_appending():
    cc = _cc(max_tokens=6000)
    seen: dict[int, str] = {}
    previous: list[str] = []
    cuts = 0
    for turns in range(4, 90):
        r = cc.compile(
            memories=[MemoryItem(id="m", content=f"Fact for turn {turns}.")],
            user_message=f"question {turns} " + "word " * (turns % 7),
            conversation_messages=_history(turns),
        )
        dropped = r.metadata["history_messages_dropped"]
        system = r.messages[0]["content"]
        assert seen.setdefault(dropped, system) == system, "same cut, same system message"
        lines = _index(r)
        if lines != previous:
            cuts += 1
            # A cut appends lines; over the index's share it also drops from the front.
            kept = len(previous)
            gone = next(k for k in range(kept + 1) if previous[k:] == lines[: kept - k])
            assert len(lines) > len(previous) - gone, "a cut adds at least one line"
            previous = lines
    assert cuts >= 3


def test_tool_results_get_a_line_and_tool_call_stubs_do_not():
    r = _cc(max_tokens=5000).compile(
        memories=[], user_message="now", conversation_messages=_history(20, tools=True)
    )
    lines = _index(r)
    assert "tool result: Badge for site 0: B9000." in lines
    assert not any("tool_calls" in line or line.startswith("assistant: None") for line in lines)
    roles = [line.split(":")[0] for line in lines[:3]]
    assert roles == ["user", "tool result", "assistant"]


def test_over_its_share_the_oldest_lines_go():
    r = _cc(index_share=0.03).compile(
        memories=[], user_message="now", conversation_messages=_history(60)
    )
    lines = _index(r)
    assert 0 < len(lines) < r.metadata["history_messages_dropped"]
    assert r.metadata["history_index_tokens"] <= int(0.03 * 2500) + 2
    newest_dropped = r.metadata["history_messages_dropped"] // 2 - 1
    assert lines[-1] == f"assistant: Stored for site {newest_dropped}."
    assert "G7000" not in "\n".join(lines)


def test_index_room_comes_from_what_history_leaves_not_from_history():
    """With and without the index the same history is kept word for word."""
    for turns in range(10, 60, 7):
        history = _history(turns)
        r = _cc().compile(memories=[], user_message="now", conversation_messages=history)
        plain = _cc(eviction_index=False).compile(
            memories=[], user_message="now", conversation_messages=history
        )
        dropped = r.metadata["history_messages_dropped"]
        assert dropped == plain.metadata["history_messages_dropped"]
        base = 2500 - 9 - count_tokens(CORE)
        kept = sum(count_tokens(str(m["content"])) + 3 for m in r.messages[1:-1])
        assert kept <= 0.75 * base
        assert r.metadata.get("history_index_tokens", 0) <= 0.125 * base + 2
        assert r.metadata["token_count"] + r.metadata["message_framing_tokens"] <= 2500
    without = _cc(eviction_index=False).compile(
        memories=[], user_message="now", conversation_messages=_history(30)
    )
    assert "history_index_lines" not in without.metadata
    assert _INDEX_HEADER not in without.messages[0]["content"]


def _fat_memories(n: int) -> list[MemoryItem]:
    return [MemoryItem(id=f"m{i}", content=f"Site {i} note. " + "Detail. " * 30) for i in range(n)]


def test_index_shrinks_to_half_its_share_when_memories_crowd_the_budget():
    history = _history(60)
    base = 2500 - 9 - count_tokens(CORE)
    roomy = _cc().compile(memories=[], user_message="now", conversation_messages=history)
    crowded = _cc().compile(
        memories=_fat_memories(12), user_message="now", conversation_messages=history
    )
    cap = int(0.125 * base)
    assert cap // 2 < roomy.metadata["history_index_tokens"] <= cap + 2
    assert 0 < crowded.metadata["history_index_tokens"] <= cap // 2 + 2
    # Every memory is still there, at least as its keyframe.
    assert all(f"Site {i} note." in crowded.messages[-1]["content"] for i in range(12))
    # The newest dropped messages are the ones that keep their line.
    assert _index(crowded) == _index(roomy)[-len(_index(crowded)) :]
    assert crowded.metadata["history_messages_dropped"] == roomy.metadata[
        "history_messages_dropped"
    ]


def test_index_size_is_fixed_at_the_cut_whatever_later_turns_bring():
    cc = _cc()
    history = _history(60)
    first = cc.compile(memories=[], user_message="now", conversation_messages=history)
    later = cc.compile(
        memories=_fat_memories(12), user_message="and now", conversation_messages=history
    )
    assert later.messages[0] == first.messages[0]
    assert later.metadata["token_count"] + later.metadata["message_framing_tokens"] <= 2500


def test_no_index_when_history_takes_the_whole_budget():
    r = _cc(history_share=1.0).compile(
        memories=[], user_message="now", conversation_messages=_history(60)
    )
    assert r.metadata["history_messages_dropped"] > 0
    assert _INDEX_HEADER not in r.messages[0]["content"]


def test_old_citations_in_the_index_never_claim_refs():
    history = _history(30)
    history[1]["content"] = "Per [Mem:1] the gate is open. " + "Understood. " * 20
    r = _cc().compile(
        memories=[MemoryItem(id="real", content="A fact.")],
        user_message="now",
        conversation_messages=history,
    )
    assert "[Mem:" not in "\n".join(_index(r))
    assert r.ref_map == {1: "real"}


def test_no_index_without_trimming_or_in_system_placement():
    history = _history(30)
    assert _index(_cc(trim_history=False).compile(
        memories=[], user_message="q", conversation_messages=history)) == []
    r = _cc(memory_placement="system").compile(
        memories=[], user_message="q", conversation_messages=history
    )
    assert _INDEX_HEADER not in r.messages[0]["content"]
    roomy = _cc(max_tokens=50_000).compile(
        memories=[], user_message="q", conversation_messages=history
    )
    assert _index(roomy) == [] and "history_index_lines" not in roomy.metadata


def test_index_works_without_a_core_prompt():
    cc = ContextCompiler(max_tokens=2500, model_id=MODEL, include_interrupt_instruction=False)
    r = cc.compile(memories=[], user_message="now", conversation_messages=_history(30))
    assert r.messages[0]["role"] == "system"
    assert r.messages[0]["content"].startswith(_INDEX_HEADER)
    assert r.metadata["core_chars"] == 0


def test_a_line_is_scored_at_its_own_cut_so_later_turns_cannot_change_it():
    evicted = [("user", "Alpha beta gamma. Delta epsilon.", 9), ("assistant", "Fine. Noted.", 5)]
    more = [*evicted, ("user", "Delta epsilon again. Delta twice.", 9), ("assistant", "Ok.", 2)]
    first = _eviction_index(evicted, (2,), count_tokens, 40, 500)
    later = _eviction_index(more, (2, 4), count_tokens, 40, 500)
    assert later[: len(first)] == first


def test_cut_starts_replays_the_grid():
    history = _history(10)
    sizes = [100] * len(history)
    assert _cut_starts(history, sizes, 400, 12) == (4, 8, 12)
    assert _cut_starts(history, sizes, 400, 4) == (4,)


def test_a_long_message_keeps_one_keyframe_per_ten_caps_of_length():
    clauses = [f"Clause {i} of the agreement stays in force." for i in range(120)]
    clauses.insert(70, "The vault code in document 5 is V4417.")
    paste = "Here is the contract. " + " ".join(clauses) + " My badge number is B2290."
    evicted = [("user", paste, count_tokens(paste)), ("assistant", "Got it. Thanks.", 5)]
    lines = _eviction_index(evicted, (2,), count_tokens, 40, 2000)
    assert lines[1] == "assistant: Got it."
    frames = lines[0].removeprefix("user: ").split(" … ")
    assert len(frames) == 1 + count_tokens(paste) // 400
    assert "The vault code in document 5 is V4417." in frames
    assert "My badge number is B2290." in frames
    assert frames.index("The vault code in document 5 is V4417.") < frames.index(
        "My badge number is B2290."
    ), "keyframes keep the order they had in the message"
