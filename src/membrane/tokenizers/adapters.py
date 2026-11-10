"""Concrete tokenizer adapters."""

from __future__ import annotations

from typing import Any, cast

import tiktoken


class TiktokenAdapter:
    def __init__(self, encoding_name: str) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self._name = encoding_name

    @property
    def name(self) -> str:
        return f"tiktoken:{self._name}"

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


class SentencePieceAdapter:
    """Lazy sentencepiece wrapper (optional dependency)."""

    def __init__(self, model_path: str) -> None:
        import sentencepiece as spm  # type: ignore

        # sentencepiece ships incomplete stubs; treat processor as dynamically shaped.
        sp: Any = spm.SentencePieceProcessor()
        sp.load(model_path)
        self._sp: Any = sp
        self._model_path = model_path

    @property
    def name(self) -> str:
        return f"sentencepiece:{self._model_path}"

    def encode(self, text: str) -> list[int]:
        return cast(list[int], self._sp.encode_as_ids(text))

    def decode(self, tokens: list[int]) -> str:
        return cast(str, self._sp.decode_ids(tokens))

    def count(self, text: str) -> int:
        return len(self.encode(text))


class HuggingFaceAdapter:
    """Transformers AutoTokenizer wrapper (optional dependency)."""

    def __init__(self, model_id: str) -> None:
        from transformers import AutoTokenizer  # type: ignore

        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model_id = model_id

    @property
    def name(self) -> str:
        return f"hf:{self._model_id}"

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self._tok.decode(tokens, skip_special_tokens=True)

    def count(self, text: str) -> int:
        return len(self.encode(text))


class EstimationAdapter:
    """Fast character-based estimate when no exact tokenizer is registered."""

    def __init__(self, chars_per_token: float = 3.5) -> None:
        self._cpt = chars_per_token

    @property
    def name(self) -> str:
        return "estimation"

    def encode(self, text: str) -> list[int]:
        return list(range(self.count(text)))

    def decode(self, tokens: list[int]) -> str:
        return ""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self._cpt))
