"""Singleton registry mapping model name patterns to tokenizer adapters."""

from __future__ import annotations

import fnmatch
from typing import Protocol, runtime_checkable

from membrane.firewall import count_tokens


@runtime_checkable
class TokenizerAdapter(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...

    def count(self, text: str) -> int: ...

    @property
    def name(self) -> str: ...


class TokenizerRegistry:
    _instance: TokenizerRegistry | None = None

    def __init__(self) -> None:
        self._patterns: list[tuple[str, TokenizerAdapter]] = []
        self._default: TokenizerAdapter | None = None

    @classmethod
    def get(cls) -> TokenizerRegistry:
        if cls._instance is None:
            inst = cls()
            inst._bootstrap_defaults()
            cls._instance = inst
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        cls._instance = None

    def _bootstrap_defaults(self) -> None:
        from membrane.tokenizers.adapters import EstimationAdapter, TiktokenAdapter

        # defaults are appended (lowest precedence); first match wins, so specific before broad
        defaults = [
            ("gpt-3.5*", TiktokenAdapter("cl100k_base")),
            ("gpt-4o*", TiktokenAdapter("o200k_base")),
            ("gpt-4.1*", TiktokenAdapter("o200k_base")),
            ("gpt-4*", TiktokenAdapter("cl100k_base")),
            ("anthropic/*", TiktokenAdapter("cl100k_base")),
            ("claude*", TiktokenAdapter("cl100k_base")),
        ]
        self._patterns.extend(defaults)
        self._default = EstimationAdapter()

    def register(self, model_pattern: str, adapter: TokenizerAdapter) -> None:
        # user patterns take precedence over defaults (and over earlier user patterns)
        self._patterns.insert(0, (model_pattern, adapter))

    def _match(self, model_id: str) -> TokenizerAdapter | None:
        for pat, adapter in self._patterns:
            if fnmatch.fnmatch(model_id.lower(), pat.lower()):
                return adapter
        return None

    def count(self, model_id: str, text: str) -> int:
        adapter = self._match(model_id)
        if adapter is not None:
            return adapter.count(text)
        if self._default is not None:
            return self._default.count(text)
        return count_tokens(text)

    def encode(self, model_id: str, text: str) -> list[int]:
        adapter = self._match(model_id)
        if adapter is not None:
            return adapter.encode(text)
        if self._default is not None:
            return self._default.encode(text)
        return []

    def decode(self, model_id: str, tokens: list[int]) -> str:
        adapter = self._match(model_id)
        if adapter is not None:
            return adapter.decode(tokens)
        if self._default is not None:
            return self._default.decode(tokens)
        return ""

    def list_models(self) -> list[str]:
        return [p for p, _ in self._patterns]
