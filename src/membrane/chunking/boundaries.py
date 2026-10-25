"""Text boundary detection for chunking strategies."""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple, cast


def detect_sentences(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) spans for sentence-like segments."""
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out: List[Tuple[int, int]] = []
    pos = 0
    for p in parts:
        if not p:
            continue
        start = text.find(p, pos)
        if start < 0:
            start = pos
        end = start + len(p)
        out.append((start, end))
        pos = end
    return out if out else [(0, len(text))]


def detect_paragraphs(text: str) -> List[Tuple[int, int]]:
    blocks = text.split("\n\n")
    out: List[Tuple[int, int]] = []
    pos = 0
    for b in blocks:
        if not b:
            pos += 2
            continue
        start = text.find(b, pos)
        if start < 0:
            start = pos
        end = start + len(b)
        out.append((start, end))
        pos = end + 2
    return out if out else [(0, len(text))]


def detect_headings(text: str) -> List[Tuple[int, int, int]]:
    """Markdown headings: (start, end, level)."""
    out: List[Tuple[int, int, int]] = []
    for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
        level = len(m.group(1))
        out.append((m.start(), m.end(), level))
    return out


def detect_topic_shifts(
    text: str,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    threshold: float = 0.25,
) -> List[int]:
    """
    Return character indices where topic boundaries are suggested.

    If ``embed_fn`` is None, falls back to paragraph boundaries.
    """
    if embed_fn is None:
        return [a for a, _ in detect_paragraphs(text)][1:]

    sents = detect_sentences(text)
    if len(sents) < 2:
        return []
    embeddings: List[List[float]] = []
    for start, end in sents:
        chunk = text[start:end]
        vec = embed_fn(chunk)
        embeddings.append(cast(List[float], vec))
    boundaries: List[int] = []
    for i in range(1, len(embeddings)):
        d = _cosine_dist(embeddings[i - 1], embeddings[i])
        if d > threshold:
            boundaries.append(sents[i][0])
    return boundaries


def _cosine_dist(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return 1.0 - dot / (na * nb)
