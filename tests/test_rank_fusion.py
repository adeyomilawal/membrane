"""Which memories survive trimming: retrieval rank fused with relevance to the question."""

from __future__ import annotations

import re

from membrane import ContextCompiler, MemoryItem

MODEL = "openai/gpt-4o"
CITIES = ["Lisbon", "Porto", "Faro", "Braga", "Coimbra", "Evora", "Sintra", "Tavira"] * 5


def _mems() -> list[MemoryItem]:
    return [
        MemoryItem(
            id=str(i),
            content=f"Office {city}-{i} door code: {1000 + i}. " + "Site detail. " * 20,
            memory_type="semantic",
        )
        for i, city in enumerate(CITIES)
    ]


def _compile(question: str, max_tokens: int = 900, **kw):
    kw.setdefault("thin_before_drop", False)
    cc = ContextCompiler(max_tokens=max_tokens, model_id=MODEL, core_content="Be brief.", **kw)
    return cc.compile(memories=_mems(), user_message=question)


def _kept(result) -> list[int]:
    """Door codes of the memories that kept their full text."""
    lines = result.messages[-1]["content"].splitlines()
    full = "\n".join(line for line in lines if not line.endswith("…"))
    return [int(n) for n in re.findall(r"door code: (\d+)", full)]


def test_last_ranked_memory_survives_when_the_question_names_it():
    r = _compile("What is the door code for office Tavira-39?")
    kept = _kept(r)
    assert 0 < len(kept) < 40
    assert 1039 in kept
    assert kept == sorted(kept), "survivors keep the order they were passed in"
    assert r.ref_map[40] == "39"


def test_without_lexical_signal_retrieval_rank_decides():
    kept = _kept(_compile("anything new?"))
    assert kept == list(range(1000, 1000 + len(kept)))


def test_rank_fusion_can_be_turned_off():
    kept = _kept(_compile("What is the door code for office Tavira-39?", rank_fusion=False))
    assert 1039 not in kept
    assert kept == list(range(1000, 1000 + len(kept)))


def test_no_effect_when_everything_fits():
    def run(**kw):
        cc = ContextCompiler(max_tokens=20_000, model_id=MODEL, core_content="Be brief.", **kw)
        return cc.compile(memories=_mems(), user_message="door code for office Tavira-39?")

    assert run().messages == run(rank_fusion=False).messages


def test_fusion_decides_which_memories_keep_their_full_text_when_thinning():
    question = "What is the door code for office Tavira-39?"
    r = _compile(question, thin_before_drop=True, max_tokens=1200)
    full = _kept(r)
    assert 0 < len(full) < 40
    assert 1039 in full
    assert len(r.ref_map) == 40, "the rest are thinned, not dropped"
