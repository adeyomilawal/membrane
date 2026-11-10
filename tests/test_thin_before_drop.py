"""Level-of-detail packing: every memory is thinned to its keyframe before any is dropped."""

from __future__ import annotations

from membrane import ContextCompiler, MemoryItem
from membrane.firewall import TIER_EPISODIC, TIER_SEMANTIC, PreFlightContextFirewall, count_tokens
from membrane.keyframes import keyframe, sentence_frequency

MODEL = "openai/gpt-4o"
CITIES = ["Lisbon", "Porto", "Faro", "Braga", "Coimbra", "Evora", "Sintra", "Tavira"] * 5


def _mems() -> list[MemoryItem]:
    return [
        MemoryItem(
            id=str(i),
            content="Site detail. " * 8 + f"Office {city}-{i} door code: {1000 + i}. "
            + "Site detail. " * 8,
            memory_type="semantic",
        )
        for i, city in enumerate(CITIES)
    ]


def _compile(max_tokens: int = 1300, **kw):
    cc = ContextCompiler(max_tokens=max_tokens, model_id=MODEL, core_content="Be brief.", **kw)
    return cc.compile(memories=_mems(), user_message="anything new?")


def test_no_memory_is_dropped_while_another_is_whole():
    r = _compile()
    lines = [x for x in r.messages[-1]["content"].splitlines() if x.startswith("[Mem:")]
    thin = [x for x in lines if x.endswith("…")]
    assert len(lines) == 40 and 0 < len(thin) < 40
    assert len(r.ref_map) == 40
    assert r.metadata["segments_thinned"] == len(thin)
    assert r.metadata["segments_kept"] == 40
    assert r.metadata["token_count"] + r.metadata["message_framing_tokens"] <= 1300
    assert "excerpts" in r.messages[-1]["content"]
    # Full text goes to the best-ranked memories; the thinned ones are the tail of the ranking.
    assert [x.endswith("…") for x in lines] == sorted(x.endswith("…") for x in lines)


def test_keyframe_is_the_sentence_with_the_rare_words_not_the_first_one():
    r = _compile()
    for line in r.messages[-1]["content"].splitlines():
        if line.startswith("[Mem:") and line.endswith("…"):
            assert "door code" in line and "Site detail" not in line


def test_keyframes_alone_are_dropped_from_the_bottom_when_even_they_do_not_fit():
    r = _compile(max_tokens=400)
    kept = sorted(r.ref_map)
    assert 0 < len(kept) < 40
    assert kept == list(range(1, len(kept) + 1))
    assert r.metadata["segments_thinned"] == len(kept)


def test_nothing_changes_when_everything_fits_or_when_turned_off():
    assert _compile(20_000).messages == _compile(20_000, thin_before_drop=False).messages
    off = _compile(thin_before_drop=False)
    assert off.metadata["segments_thinned"] == 0
    assert "…" not in off.messages[-1]["content"]
    assert len(off.ref_map) < 40


def test_system_placement_is_untouched():
    r = _compile(memory_placement="system")
    assert r.metadata["segments_thinned"] == 0
    assert "…" not in r.messages[0]["content"]


def test_a_memory_with_no_sentence_under_the_cap_is_kept_whole_or_dropped():
    mems = [
        MemoryItem(id=str(i), content=f"Fact {i} " + "and more " * 60, memory_type="semantic")
        for i in range(10)
    ]
    cc = ContextCompiler(max_tokens=500, model_id=MODEL, core_content="Be brief.")
    r = cc.compile(memories=mems, user_message="hi")
    assert r.metadata["segments_thinned"] == 0
    assert 0 < len(r.ref_map) < 10


def test_keyframe_scoring_rules():
    texts = ["The door code is 4417. Thanks a lot.", "Thanks a lot. See you.", "Thanks. Bye."]
    df, n = sentence_frequency(texts)
    assert keyframe(texts[0], df, n, count_tokens) == "The door code is 4417."
    # A sentence over the cap cannot win, however rare its words.
    long = "zebra quokka " * 30 + "end. Short one."
    assert keyframe(long, df, n, count_tokens, cap=10) == "Short one."
    assert keyframe("zebra quokka " * 30, df, n, count_tokens, cap=10) is None
    # Ties go to the earlier sentence; labels and cut-off fragments are not sentences.
    assert keyframe("Alpha beta. Gamma delta.", {}, 1, count_tokens) == "Alpha beta."
    assert keyframe("[fact] kind=note | text=Real one. Cut off he... | conf=1.00", {}, 1,
                    count_tokens) == "Real one."


def test_firewall_packs_thin_forms_first_then_upgrades_in_priority_order():
    segs = [(f"[Mem:{i}] " + "word " * 30, TIER_SEMANTIC) for i in range(1, 5)]
    thin = [f"[Mem:{i}] w …" for i in range(1, 5)]
    fw = PreFlightContextFirewall(max_tokens=100)
    _, refs, total, stats = fw.run(
        "core", "end", segs, {i: f"id{i}" for i in range(1, 5)}, thin_segments=thin
    )
    assert total <= 100
    assert sorted(refs) == [1, 2, 3, 4]
    assert stats["kept_segments"] == [segs[0][0], segs[1][0], thin[2], thin[3]]
    assert stats["segments_thinned"] == 2
    assert stats["thinned_segments"] == {thin[2]: segs[2][0], thin[3]: segs[3][0]}


def test_firewall_gives_pinned_tiers_their_full_text_before_any_thin_form():
    segs = [("[Mem:1] " + "word " * 30, TIER_SEMANTIC), ("[Mem:2] " + "word " * 60, TIER_EPISODIC)]
    thin = ["[Mem:1] w …", "[Mem:2] w …"]
    fw = PreFlightContextFirewall(max_tokens=80)
    _, _, _, stats = fw.run(
        "core", "end", segs, {1: "a", 2: "b"}, thin_segments=thin, pinned_tiers={TIER_EPISODIC}
    )
    assert stats["kept_segments"] == [segs[1][0], thin[0]]


def test_segments_without_a_thin_form_are_all_or_nothing():
    segs = [("[Mem:1] " + "word " * 30, TIER_SEMANTIC), ("passage " * 30, TIER_EPISODIC)]
    fw = PreFlightContextFirewall(max_tokens=50)
    _, _, _, stats = fw.run("core", "end", segs, {1: "a"}, thin_segments=["[Mem:1] w …", None])
    assert stats["kept_segments"] == [segs[0][0]]


def test_a_phrase_repeated_on_every_line_of_one_long_text_is_not_rare():
    doc = " ".join(f"Clause {i} of the agreement stays in force." for i in range(100))
    doc += " The vault code is V4417."
    df, n = sentence_frequency([doc, "Thanks.", "The weather is fine."])
    assert keyframe(doc, df, n, count_tokens) == "The vault code is V4417."


def test_a_segment_that_is_its_own_thin_form_cannot_be_pushed_out_by_an_excerpt_below_it():
    whole = "[Mem:1] " + "word " * 30
    segs = [(whole, TIER_SEMANTIC), ("passage " * 30, TIER_EPISODIC)]
    fw = PreFlightContextFirewall(max_tokens=45)
    _, refs, _, stats = fw.run("core", "end", segs, {1: "a"}, thin_segments=[whole, "p …"])
    assert stats["kept_segments"] == [whole, "p …"]
    assert stats["segments_thinned"] == 1 and refs == {1: "a"}
