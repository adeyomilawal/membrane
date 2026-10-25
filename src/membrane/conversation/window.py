"""Sliding window over recent turns."""

from __future__ import annotations



def sliding_turns(
    turns: list[tuple[str, str]],
    *,
    max_turns: int,
) -> list[tuple[str, str]]:
    if max_turns <= 0:
        return []
    return turns[-max_turns:]


def serialize_turns(turns: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"role": r, "content": c} for r, c in turns]
