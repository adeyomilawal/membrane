"""Chunking strategies."""

from __future__ import annotations

from collections.abc import Callable

from membrane.chunking.boundaries import (
    detect_headings,
    detect_paragraphs,
    detect_sentences,
    detect_topic_shifts,
)


class Chunk:
    __slots__ = ("content", "end_char", "metadata", "start_char", "token_count")

    def __init__(
        self,
        content: str,
        token_count: int,
        start_char: int,
        end_char: int,
        metadata: dict | None = None,
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
    spans: list[tuple],
    max_tokens: int,
    tok: Callable[[str], int],
    label: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf_parts: list[str] = []
    buf_start: int | None = None
    buf_end = 0
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf_parts, buf_start, buf_tokens
        if buf_parts and buf_start is not None:
            content = "\n".join(buf_parts).strip()
            chunks.append(Chunk(content, tok(content), buf_start, buf_end, {"strategy": label}))
        buf_parts = []
        buf_start = None
        buf_tokens = 0

    for start, end in spans:
        raw = text[start:end]
        piece = raw.strip()
        if not piece:
            continue
        t = tok(piece)
        if t > max_tokens:
            # unit alone exceeds the budget: flush the buffer, then split the unit itself
            flush()
            base = start + (len(raw) - len(raw.lstrip()))
            for sub in FixedStrategy().chunk(piece, max_tokens=max_tokens, tokenizer=tok):
                sub.start_char += base
                sub.end_char += base
                sub.metadata["strategy"] = label
                chunks.append(sub)
            continue
        if buf_parts and buf_tokens + t > max_tokens:
            flush()
        if buf_start is None:
            buf_start = start
        buf_parts.append(piece)
        buf_tokens += t
        buf_end = end
    flush()
    return chunks


class SentenceStrategy:
    def chunk(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        tokenizer: Callable[[str], int] | None = None,
    ) -> list[Chunk]:
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
        tokenizer: Callable[[str], int] | None = None,
    ) -> list[Chunk]:
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
        tokenizer: Callable[[str], int] | None = None,
    ) -> list[Chunk]:
        tok = tokenizer or _default_tok()
        chunks: list[Chunk] = []
        if max_tokens <= 0:
            return chunks
        ov_tokens = max(0, min(overlap_tokens, max_tokens - 1))
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
            # overlap is measured in tokens but applied in characters: find the tail of this
            # chunk holding at most ``ov_tokens`` tokens, and never overlap the whole chunk
            seg_len = end - start
            ov_chars = 0
            if ov_tokens > 0:
                lo, hi = 0, seg_len
                while lo < hi:
                    mid = (lo + hi) // 2
                    if tok(seg[mid:]) <= ov_tokens:
                        hi = mid
                    else:
                        lo = mid + 1
                ov_chars = min(seg_len - lo, seg_len - 1)
            start += max(1, seg_len - ov_chars)
        return chunks


class RecursiveStrategy:
    def chunk(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        tokenizer: Callable[[str], int] | None = None,
    ) -> list[Chunk]:
        tok = tokenizer or _default_tok()
        hd = detect_headings(text)
        if not hd:
            return ParagraphStrategy().chunk(
                text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                tokenizer=tok,
            )
        chunks: list[Chunk] = []
        if hd[0][0] > 0 and text[: hd[0][0]].strip():
            # text before the first heading
            for c in ParagraphStrategy().chunk(
                text[: hd[0][0]],
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                tokenizer=tok,
            ):
                chunks.append(c)
        for i, (s, _e, level) in enumerate(hd):
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
                # offsets are block-relative; report them against the whole document
                c.start_char += s
                c.end_char += s
            chunks.extend(sub)
        return chunks


class SemanticStrategy:
    def __init__(self, embed_fn: Callable[[str], list[float]] | None = None) -> None:
        self._embed_fn = embed_fn

    def chunk(
        self,
        text: str,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        tokenizer: Callable[[str], int] | None = None,
    ) -> list[Chunk]:
        tok = tokenizer or _default_tok()
        boundaries = [0, *detect_topic_shifts(text, self._embed_fn), len(text)]
        spans = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        return _pack_units(text, spans, max_tokens, tok, "semantic")
