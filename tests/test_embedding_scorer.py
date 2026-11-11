"""Optional embedder: meaning, not only shared words, picks what gets its full text."""

from __future__ import annotations

import logging
import re

import pytest

from membrane import ContextCompiler, MemoryItem, compile_anonymous

MODEL = "openai/gpt-4o"

# A fake embedder: one dimension per concept, words of a concept count towards it.
CONCEPTS = [
    {"salary", "pay", "wage", "earns", "earn"},
    {"car", "automobile", "vehicle", "drives", "drive"},
    {"doctor", "physician", "clinic"},
]


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            out.append([float(len(words & concept)) for concept in CONCEPTS] + [0.01])
        return out


TOPICS = ["garden", "kitchen", "roof", "fence", "cellar", "attic", "porch", "shed"] * 5


def _mems() -> list[MemoryItem]:
    mems = [
        MemoryItem(
            id=str(i),
            content=f"The {topic} job {i} is booked. " + "Site detail. " * 20,
            memory_type="semantic",
        )
        for i, topic in enumerate(TOPICS)
    ]
    # The last-ranked memory answers the question below without sharing a word with it.
    mems[-1] = MemoryItem(
        id="39",
        content="Dana earns 4100 a month at the depot. " + "Site detail. " * 20,
        memory_type="semantic",
    )
    return mems


QUESTION = "What is the wage of our colleague?"


def _compile(embedder=None, max_tokens: int = 900, **kw):
    kw.setdefault("thin_before_drop", False)
    cc = ContextCompiler(
        max_tokens=max_tokens, model_id=MODEL, core_content="Be brief.", embedder=embedder, **kw
    )
    return cc.compile(memories=_mems(), user_message=QUESTION)


def test_default_is_word_matching_and_the_reworded_memory_is_dropped():
    r = _compile()
    assert "4100" not in r.messages[-1]["content"]
    assert "semantic_ranking" not in r.metadata


def test_an_embedder_keeps_the_memory_that_means_the_same():
    r = _compile(FakeEmbedder())
    tail = r.messages[-1]["content"]
    assert "Dana earns 4100" in tail
    assert r.metadata["semantic_ranking"] is True
    assert 0 < len(r.ref_map) < 40
    refs = [int(n) for n in re.findall(r"\[Mem:(\d+)", tail)]
    assert refs == sorted(refs), "survivors keep the order they were passed in"


def test_an_embedder_decides_full_text_not_keyframes():
    plain = _compile(thin_before_drop=True, max_tokens=1200)
    meant = _compile(FakeEmbedder(), thin_before_drop=True, max_tokens=1200)

    def full(result) -> set[str]:
        lines = result.messages[-1]["content"].splitlines()
        return {m.group(1) for ln in lines if not ln.endswith("…") for m in [_REF.match(ln)] if m}

    def thin(result) -> dict[str, str]:
        lines = result.messages[-1]["content"].splitlines()
        return {m.group(1): ln for ln in lines if ln.endswith("…") for m in [_REF.match(ln)] if m}

    assert "40" in full(meant) and "40" not in full(plain)
    # A memory thinned in both runs has the same keyframe in both.
    shared = set(thin(plain)) & set(thin(meant))
    assert shared and all(thin(plain)[ref] == thin(meant)[ref] for ref in shared)


_REF = re.compile(r"\[Mem:(\d+)")


def _history(n: int) -> list[dict]:
    out = []
    for i in range(n):
        fact = "My automobile is a green Volvo. " if i == 2 else ""
        out.append({"role": "user", "content": f"{fact}note {i} " + "word " * 30})
        out.append({"role": "assistant", "content": f"answer {i} " + "reply " * 60})
    return out


def _recall(embedder=None):
    cc = ContextCompiler(
        max_tokens=3000,
        model_id=MODEL,
        core_content="Be brief.",
        embedder=embedder,
        eviction_index=False,
    )
    return cc.compile(
        memories=[],
        user_message="Which vehicle do I own?",
        conversation_messages=_history(40),
    )


def test_an_embedder_recalls_dropped_history_that_shares_no_word_with_the_question():
    plain, meant = _recall(), _recall(FakeEmbedder())
    assert plain.metadata["history_messages_dropped"] > 6
    assert "green Volvo" not in "\n".join(str(m["content"]) for m in plain.messages)
    assert "[Earlier user] My automobile is a green Volvo." in meant.messages[-1]["content"]
    assert meant.metadata["semantic_ranking"] is True
    # The cut and the cached prefix do not depend on the embedder.
    assert meant.messages[:-1] == plain.messages[:-1]


def test_each_text_is_embedded_once_per_compiler():
    embedder = FakeEmbedder()
    cc = ContextCompiler(
        max_tokens=3000, model_id=MODEL, core_content="Be brief.", embedder=embedder
    )
    history = _history(40)
    cc.compile(memories=[], user_message="Which vehicle do I own?", conversation_messages=history)
    first = sum(len(batch) for batch in embedder.calls)
    cc.compile(memories=[], user_message="And the doctor?", conversation_messages=history)
    again = [text for batch in embedder.calls[1:] for text in batch]
    assert first > 10
    assert again == ["And the doctor?"]


def test_a_failing_embedder_falls_back_to_words(caplog):
    def broken(texts):
        raise RuntimeError("model missing")

    with caplog.at_level(logging.WARNING, logger="membrane.compiler"):
        r = _compile(broken)
        again = _compile(lambda texts: [[1.0]])  # wrong number of vectors
    assert r.messages == _compile().messages == again.messages
    assert "semantic_ranking" not in r.metadata
    assert "ranking by words only" in caplog.text


def test_nothing_changes_when_everything_fits():
    fits = _compile(FakeEmbedder(), max_tokens=20_000)
    assert fits.messages == _compile(max_tokens=20_000).messages


def test_compile_anonymous_takes_the_embedder():
    kw = dict(
        core_content="Be brief.",
        user_message="Which vehicle do I own?",
        conversation_messages=_history(40),
        max_tokens=3000,
        model_id=MODEL,
        memory_placement="end",
        eviction_index=False,
    )
    assert "green Volvo" not in compile_anonymous(**kw).messages[-1]["content"]
    assert "green Volvo" in compile_anonymous(embedder=FakeEmbedder(), **kw).messages[-1]["content"]


def test_fastembed_embedder_says_how_to_install(monkeypatch):
    import builtins

    from membrane import lod

    real_import = builtins.__import__

    def no_fastembed(name, *args, **kwargs):
        if name.startswith("fastembed"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_fastembed)
    with pytest.raises(ImportError, match=r"membrane-aol\[lod\]"):
        lod.fastembed_embedder()
