"""Extractive summarization for older turns (no LLM required)."""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple


def score_sentences(text: str) -> List[Tuple[str, float]]:
    """Very simple TF-like scoring: longer sentences with digits/entities score higher."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    scored: List[Tuple[str, float]] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        w = len(p.split())
        bonus = 0.15 if re.search(r"\d", p) else 0.0
        scored.append((p, float(w) + bonus))
    scored.sort(key=lambda x: -x[1])
    return scored


def extractive_summary(
    text: str,
    *,
    max_sentences: int = 5,
    max_chars: int = 1200,
) -> str:
    ranked = score_sentences(text)
    out: List[str] = []
    n = 0
    total = 0
    for sent, _ in ranked:
        if n >= max_sentences:
            break
        if total + len(sent) > max_chars:
            break
        out.append(sent)
        total += len(sent)
        n += 1
    return " ".join(out) if out else text[:max_chars]


def summarize_turns(
    turns: Sequence[Tuple[str, str]],
    summarization_fn: Optional[Callable[[str], str]],
) -> str:
    blob = "\n".join(f"{r}: {c}" for r, c in turns)
    if summarization_fn is not None:
        return summarization_fn(blob)
    return extractive_summary(blob)
