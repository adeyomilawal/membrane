"""Extractive summarization for older turns (no LLM required)."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence


def score_sentences(text: str) -> list[tuple[str, float]]:
    """Very simple TF-like scoring: longer sentences with digits/entities score higher."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    scored: list[tuple[str, float]] = []
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
    out: list[str] = []
    total = 0
    for n, (sent, _) in enumerate(ranked):
        if n >= max_sentences:
            break
        if total + len(sent) > max_chars:
            break
        out.append(sent)
        total += len(sent)
    return " ".join(out) if out else text[:max_chars]


def summarize_turns(
    turns: Sequence[tuple[str, str]],
    summarization_fn: Callable[[str], str] | None,
) -> str:
    blob = "\n".join(f"{r}: {c}" for r, c in turns)
    if summarization_fn is not None:
        return summarization_fn(blob)
    return extractive_summary(blob)
