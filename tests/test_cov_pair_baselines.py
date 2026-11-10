import pytest

from membrane.firewall import TIER_EPISODIC, TIER_PROCEDURAL, TIER_SEMANTIC, count_tokens
from membrane.pair_baselines import (
    _ref_submap_from_text,
    marker_survival_score,
    pack_middle_order,
    pad_prefix_to_token_count,
    procedural_recall_vs_fifo,
    tier_vs_fifo_demo_segments,
)


def test_ref_submap():
    assert _ref_submap_from_text("[Mem:1] a [Mem:3] b [Mem:9]", {1: "a", 3: "c"}) == {
        1: "a", 3: "c"}


def test_pack_unknown_order():
    with pytest.raises(ValueError, match="unknown order"):
        pack_middle_order("c", "e", [], 100, order="bogus")


def test_pack_as_given_keeps_order_and_refs():
    segs = [("[Mem:1] aaa", TIER_EPISODIC), ("[Mem:2] bbb", TIER_PROCEDURAL)]
    final, refs, total = pack_middle_order(
        "core", "end", segs, 1000, order="as_given", ref_map={1: "g1", 2: "g2", 3: "g3"})
    assert final == "core\n\n[Mem:1] aaa\n[Mem:2] bbb\n\nend"
    assert refs == {1: "g1", 2: "g2"}
    assert total == count_tokens(final)


def test_pack_tier_desc_orders_episodic_first():
    segs = [("[Mem:1] P", TIER_PROCEDURAL), ("[Mem:2] E", TIER_EPISODIC),
            ("[Mem:3] S", TIER_SEMANTIC)]
    final, _, _ = pack_middle_order("c", "e", segs, 1000, order="tier_desc")
    assert final.index("] E") < final.index("] S") < final.index("] P")


def test_pack_reserved_exceeds_budget():
    final, refs, total = pack_middle_order(
        "[Mem:1] core", "end", [("[Mem:2] x", 2)], 1, order="as_given", ref_map={1: "a", 2: "b"})
    assert final == "[Mem:1] core\n\nend"
    assert refs == {1: "a"}
    assert total == count_tokens("[Mem:1] core") + count_tokens("end")


def test_pack_skips_oversized_and_caps_procedural():
    big = "word " * 500
    segs = [(big, TIER_SEMANTIC)] + [(f"P{i}", TIER_PROCEDURAL) for i in range(5)]
    final, _, _ = pack_middle_order("c", "e", segs, 200, order="as_given")
    assert "word" not in final
    assert "P0" in final and "P2" in final
    assert "P3" not in final and "P4" not in final


def test_pack_tier_delegates_to_firewall():
    segs = [("[Mem:1] sem", TIER_SEMANTIC), ("[Mem:2] proc", TIER_PROCEDURAL)]
    final, refs, total = pack_middle_order("c", "e", segs, 1000, order="tier",
                                           ref_map={1: "a", 2: "b"})
    assert "proc" in final and "sem" in final
    assert total > 0 and set(refs) <= {1, 2}


def test_pad_prefix():
    prefix = "hello "
    pad = pad_prefix_to_token_count(prefix, 40)
    assert count_tokens(prefix + pad) <= 40
    assert count_tokens(prefix + pad + "x") > 40 or len(pad) == 40 * 8
    assert pad_prefix_to_token_count("word " * 100, 5) == ""


def test_marker_survival_score():
    assert marker_survival_score(["A", "B", "C"], "xx A yy C") == (2, ["B"])
    assert marker_survival_score([], "t") == (0, [])


def test_demo_segments_shape():
    core, end, middle, ref_map, markers, cap = tier_vs_fifo_demo_segments()
    assert len(middle) == 7 and markers == ["MARK_PROC_0", "MARK_PROC_1", "MARK_PROC_2"]
    assert sorted(ref_map) == list(range(1, 8))
    assert [t for _, t in middle].count(TIER_PROCEDURAL) == 3
    assert cap == count_tokens(core) + count_tokens(end) + 4 * 50
    for text, tier in middle:
        if tier == TIER_SEMANTIC:
            assert count_tokens(text) <= 50


def test_procedural_recall_vs_fifo_demo():
    core, end, middle, ref_map, markers, cap = tier_vs_fifo_demo_segments()
    res = procedural_recall_vs_fifo(core, end, middle, cap, markers, ref_map)
    assert res["tier_markers_kept"] == 3 and res["tier_missing"] == []
    assert res["fifo_markers_kept"] < 3
    assert res["membrane_beats_fifo_on_markers"] is True
    assert res["membrane_beats_tier_desc_on_markers"] is True
    assert res["tier_tokens"] <= cap
