"""Tier 2 interrupt protocol: parse <SYSTEM_INTERRUPT_FETCH[Query]> from model output."""

from __future__ import annotations

import re
from typing import Any

INTERRUPT_TAG_START = "<SYSTEM_INTERRUPT_FETCH["
INTERRUPT_TAG_END = "]>"

_INTERRUPT_PATTERN = re.compile(
    re.escape(INTERRUPT_TAG_START) + r"(.*?)" + re.escape(INTERRUPT_TAG_END),
    re.DOTALL,
)


def parse_interrupt(text: Any) -> tuple[str, str, str] | None:
    """
    Scan text for the first occurrence of <SYSTEM_INTERRUPT_FETCH[Query]>.

    Returns:
        (before, query, after) if found; otherwise None.
    """
    if not text or not isinstance(text, str):
        return None
    m = _INTERRUPT_PATTERN.search(text)
    if not m:
        return None
    before = text[: m.start()]
    query = m.group(1).strip()
    after = text[m.end() :]
    return (before, query, after)


def contains_interrupt(text: Any) -> bool:
    """True if text contains at least one interrupt tag."""
    if not text or not isinstance(text, str):
        return False
    return _INTERRUPT_PATTERN.search(text) is not None


def strip_all_interrupts(text: Any) -> Any:
    """Remove all interrupt tags from text (e.g. when page-fault limit reached)."""
    if not text:
        return text
    return _INTERRUPT_PATTERN.sub("", text)
