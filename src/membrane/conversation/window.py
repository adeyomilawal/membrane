"""Sliding window over recent turns."""

from __future__ import annotations

from typing import Dict, List, Tuple


def sliding_turns(
    turns: List[Tuple[str, str]],
    *,
    max_turns: int,
) -> List[Tuple[str, str]]:
    if max_turns <= 0:
        return []
    return turns[-max_turns:]


def serialize_turns(turns: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    return [{"role": r, "content": c} for r, c in turns]
