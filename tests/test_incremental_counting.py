"""Incremental counting: nothing already counted is tokenised again, and counts stay exact."""

from __future__ import annotations

import random

from membrane import ContextCompiler, MemoryItem
from membrane import firewall as fw
from membrane.firewall import count_parts, count_tokens, parts_add_up

MODEL = "openai/gpt-4o"
CORE = "\n".join(f"Guideline {i}: answer precisely and cite memories." for i in range(400))


def _turn(i: int) -> list[dict]:
    ask = f"Question {i}? My code for trip {i} is L{1000 + i}. " + "pad " * 40
    return [
        {"role": "user", "content": ask},
        {"role": "assistant", "content": f"Answer {i}. " + "Certainly noted. " * 30},
    ]


def _mems(i: int) -> list[MemoryItem]:
    detail = "Ops detail. " * 18
    return [
        MemoryItem(id=f"{i}-{k}", content=f"Project {k} deploy code: D{i}{k}. " + detail)
        for k in range(8)
    ]


class Spy:
    """Wraps the tokenizer and records what it is asked to encode."""

    def __init__(self, encoding):
        self._encoding, self.texts = encoding, []

    def encode(self, text):
        self.texts.append(text)
        return self._encoding.encode(text)


def _exact(messages: list[dict]) -> int:
    return sum(len(fw._get_encoding().encode(m["content"])) for m in messages)


def test_token_count_is_exact_on_every_turn_of_a_long_conversation():
    cc = ContextCompiler(max_tokens=9000, model_id=MODEL, core_content=CORE)
    history: list[dict] = []
    seen = set()
    for i in range(60):
        r = cc.compile(memories=_mems(i), user_message=f"What now, turn {i}?",
                       conversation_messages=history)
        md = r.metadata
        assert md["token_count"] == _exact(r.messages)
        assert md["token_count"] + md["message_framing_tokens"] <= 9000
        seen.add((md["history_messages_dropped"] > 0, md.get("history_messages_recalled", 0) > 0))
        history = [*history, *_turn(i)]
    assert (True, True) in seen, "the conversation was long enough to cut and recall"


def test_a_turn_only_tokenises_what_is_new(monkeypatch):
    cc = ContextCompiler(max_tokens=9000, model_id=MODEL, core_content=CORE)
    history: list[dict] = []
    for i in range(30):
        cc.compile(memories=_mems(i), user_message=f"What now, turn {i}?",
                   conversation_messages=history)
        history = [*history, *_turn(i)]
    spy = Spy(fw._get_encoding())
    monkeypatch.setattr(fw, "_tiktoken_encoding", spy)
    r = cc.compile(memories=_mems(30), user_message="What now, turn 30?",
                   conversation_messages=history)
    assert r.metadata["history_messages_dropped"] > 0
    encoded = sum(len(t) for t in spy.texts)
    request = sum(len(m["content"]) for m in r.messages)
    # The core (over the count cache's size limit), the kept history and the recalled
    # passages were all counted on earlier turns.
    assert CORE not in spy.texts and not any(CORE in t for t in spy.texts)
    assert not any(m["content"] in spy.texts for m in r.messages[1:-3])
    assert encoded < request / 8, (encoded, request)


def test_joined_history_is_never_tokenised(monkeypatch):
    cc = ContextCompiler(max_tokens=20_000, model_id=MODEL, core_content="Be brief.")
    history = [m for i in range(40) for m in _turn(i)]
    spy = Spy(fw._get_encoding())
    monkeypatch.setattr(fw, "_tiktoken_encoding", spy)
    fw._count_exact_cached.cache_clear()
    cc.compile(memories=_mems(0), user_message="now", conversation_messages=history)
    longest_message = max(len(m["content"]) for m in history)
    assert max(len(t) for t in spy.texts) <= longest_message + 2000


def test_counting_in_parts_is_exact_for_the_built_in_tokenizer():
    rng = random.Random(11)
    alphabet = ["word", "Ünïcode", "12", "3.5", "note7-3", "—", "(x)", "[Mem:4]", ".", ",", "?",
                "…", "  ", " ", "\t", "'s", "😀", "http://a.b/c?d=1", "--- Header ---", "日本語"]
    for _ in range(400):
        parts = []
        for _ in range(rng.randint(2, 8)):
            words = [rng.choice(alphabet) + rng.choice(["", " "]) for _ in range(12)]
            parts.append("".join(words[: rng.randint(1, 12)]).lstrip() or "x")
        breaks = ["\n", "\n\n", " \n", ".\n", "\r\n"]
        parts = [p + rng.choice(breaks) for p in parts[:-1]] + parts[-1:]
        assert parts_add_up(parts)
        joined, total = count_parts(parts, count_tokens)
        assert joined == "".join(parts)
        assert total == len(fw._get_encoding().encode(joined)), parts


def test_parts_that_do_not_split_cleanly_are_counted_whole():
    assert not parts_add_up(["no line break", "next"])
    assert not parts_add_up(["line\n", " starts with a space"])
    assert not parts_add_up(["line\n", ""])
    assert parts_add_up(["only one part"])
    calls = []

    def count(text: str) -> int:
        calls.append(text)
        return len(text)

    assert count_parts(["ab", "cd"], count) == ("abcd", 4) and calls == ["abcd"]
    assert count_parts("plain", count) == ("plain", 5)


def test_a_custom_tokenizer_counts_the_tail_as_one_text():
    from membrane.tokenizers import TiktokenAdapter, TokenizerRegistry

    registry = TokenizerRegistry()
    registry.register("openai/*", TiktokenAdapter("o200k_base"))
    cc = ContextCompiler(max_tokens=3000, model_id=MODEL, core_content="Be brief.")
    r = cc.compile(memories=_mems(1), user_message="now", tokenizer_registry=registry,
                   conversation_messages=[m for i in range(30) for m in _turn(i)])
    exact = sum(registry.count(MODEL, m["content"]) for m in r.messages)
    assert r.metadata["token_count"] == exact
