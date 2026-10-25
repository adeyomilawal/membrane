"""Chunking strategies."""

from __future__ import annotations

from typing import Callable, List, Optional

from membrane.chunking.boundaries import (
    detect_headings,
    detect_paragraphs,
    detect_sentences,
    detect_topic_shifts,
)


class Chunk:
    __slots__ = ("content", "token_count", "start_char", "end_char", "metadata")

    def __init__(
        self,
        content: str,
        token_count: int,
        start_char: int,
        end_char: int,
        metadata: Optional[dict] = None,
    ) -> None:
        self.content = content
        self.token_count = token_count
        self.start_char = start_char
        self.end_char = end_char
        self.metadata = metadata or {}


def _default_tok() -> Callable[[str], int]:
    from membrane.firewall import count_tokens

    return count_tokens


def _pack_units(
    text: str,
    spans: List[tuple],
    max_tokens: int,
    tok: Callable[[str], int],
    label: str,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf_parts: List[str] = []
    buf_start: Optional[int] = None
    buf_end = 0
    buf_tokens = 0
    for start, end in spans:
        piece = text[start:end].strip()
        if not piece:
            continue
        t = tok(piece)
        if buf_start is None:
            buf_start = start
        if buf_tokens + t > max_tokens and buf_parts:
            content = "\n".join(buf_parts).strip()
            start_char = buf_start if buf_start is not None else start
            chunks.append(
                Chunk(
                    content,
                    tok(content),
                    start_char,
                    buf_end,
                    {"strategy": label},
                )
            )
            buf_parts = []
            buf_start = start
            buf_tokens = 0
        buf_parts.append(piece)
        buf_tokens += t
        buf_end = end
    if buf_parts and buf_start is not None:
        content = "\n".join(buf_parts).strip()
        chunks.append(
            Chunk(
                content,
                tok(content),
                buf_start,
                buf_end,
                {"strategy": label},
            )
        )
    return chunks


class SentenceStrategy:
    def chunk(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        tokenizer: Optional[Callable[[str], int]] = None,
    ) -> List[Chunk]:
        tok = tokenizer or _default_tok()
        spans = detect_sentences(text)
        if not spans:
            spans = [(0, len(text))]
        return _pack_units(text, spans, max_tokens, tok, "sentence")


class ParagraphStrategy:
    def chunk(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        tokenizer: Optional[Callable[[str], int]] = None,
    ) -> List[Chunk]:
        tok = tokenizer or _default_tok()
        spans = detect_paragraphs(text)
        if not spans:
            spans = [(0, len(text))]
        return _pack_units(text, spans, max_tokens, tok, "paragraph")


class FixedStrategy:
    def chunk(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        tokenizer: Optional[Callable[[str], int]] = None,
    ) -> List[Chunk]:
        tok = tokenizer or _default_tok()
        chunks: List[Chunk] = []
        if max_tokens <= 0:
            return chunks
        start = 0
        n = len(text)
        while start < n:
            lo, hi = start, n
            best = start + 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if tok(text[start:mid]) <= max_tokens:
                    best = mid
                    lo = mid
                else:
                    hi = mid - 1
            end = max(start + 1, best)
            seg = text[start:end]
            chunks.append(Chunk(seg, tok(seg), start, end, {"strategy": "fixed"}))
            if end >= n:
                break
            adv = max(1, end - start - overlap_tokens)
            start += adv
        return chunks


class RecursiveStrategy:
    def chunk(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        tokenizer: Optional[Callable[[str], int]] = None,
    ) -> List[Chunk]:
        tok = tokenizer or _default_tok()
        hd = detect_headings(text)
        if not hd:
            return ParagraphStrategy().chunk(
                text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                tokenizer=tok,
            )
        chunks: List[Chunk] = []
        for i, (s, e, level) in enumerate(hd):
            nxt = hd[i + 1][0] if i + 1 < len(hd) else len(text)
            block = text[s:nxt]
            sub = ParagraphStrategy().chunk(
                block,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                tokenizer=tok,
            )
            for c in sub:
                c.metadata["heading_level"] = level
            chunks.extend(sub)
        return chunks


class SemanticStrategy:
    def __init__(self, embed_fn: Optional[Callable[[str], List[float]]] = None) -> None:
        self._embed_fn = embed_fn

    def chunk(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        tokenizer: Optional[Callable[[str], int]] = None,
    ) -> List[Chunk]:
        tok = tokenizer or _default_tok()
        boundaries = [0] + detect_topic_shifts(text, self._embed_fn) + [len(text)]
        spans = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        return _pack_units(text, spans, max_tokens, tok, "semantic")
