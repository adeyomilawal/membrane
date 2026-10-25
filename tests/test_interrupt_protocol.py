"""parse_interrupt and strip_all_interrupts (host integration surface)."""

from __future__ import annotations

from membrane import parse_interrupt, strip_all_interrupts


def test_parse_interrupt_extracts_query_and_strip_removes_tag():
    text = (
        "I need more info. <SYSTEM_INTERRUPT_FETCH[user's timezone preference]> "
        "Let me check."
    )
    parsed = parse_interrupt(text)
    assert parsed is not None
    before, query, after = parsed
    assert "I need more info." in before
    assert query == "user's timezone preference"
    assert "Let me check." in after

    stripped = strip_all_interrupts(text)
    assert "<SYSTEM_INTERRUPT_FETCH" not in (stripped or "")
    assert "Let me check." in stripped
