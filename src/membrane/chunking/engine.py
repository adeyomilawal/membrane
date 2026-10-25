"""ChunkingEngine — batch and single-document chunking."""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from membrane.chunking.strategies import Chunk


class ChunkingEngine:
    def __init__(
        self,
        strategy: Any,
        *,
        tokenizer: Callable[[str], int] | None = None,
        model_id: str | None = None,
    ) -> None:
        self._strategy = strategy
        self._tokenizer = tokenizer
        self._model_id = model_id

    def chunk(
        self,
        text: str,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ) -> list[Chunk]:
        tok = self._tokenizer
        if tok is None and self._model_id:
            try:
                from membrane.tokenizers import TokenizerRegistry

                reg = TokenizerRegistry.get()

                def _tok(s: str) -> int:
                    return int(reg.count(self._model_id or "", s))

                tok = _tok
            except Exception:
                tok = None
        return self._strategy.chunk(
            text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, tokenizer=tok
        )

    def chunk_batch(
        self,
        texts: list[str],
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ) -> list[list[Chunk]]:
        return [self.chunk(t, max_tokens=max_tokens, overlap_tokens=overlap_tokens) for t in texts]
