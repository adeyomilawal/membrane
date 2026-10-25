"""Singleton registry mapping model name patterns to tokenizer adapters."""

from __future__ import annotations

import fnmatch
from typing import List, Optional, Protocol, runtime_checkable

from membrane.firewall import count_tokens


@runtime_checkable
class TokenizerAdapter(Protocol):
    def encode(self, text: str) -> List[int]: ...

    def decode(self, tokens: List[int]) -> str: ...

    def count(self, text: str) -> int: ...

    @property
    def name(self) -> str: ...


class TokenizerRegistry:
    _instance: Optional["TokenizerRegistry"] = None

    def __init__(self) -> None:
        self._patterns: List[tuple[str, TokenizerAdapter]] = []
        self._default: Optional[TokenizerAdapter] = None

    @classmethod
    def get(cls) -> "TokenizerRegistry":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._bootstrap_defaults()
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        cls._instance = None

    def _bootstrap_defaults(self) -> None:
        from membrane.tokenizers.adapters import EstimationAdapter, TiktokenAdapter

        self.register("gpt-3.5*", TiktokenAdapter("cl100k_base"))
        self.register("gpt-4*", TiktokenAdapter("o200k_base"))
        self.register("gpt-4o*", TiktokenAdapter("o200k_base"))
        self.register("anthropic/*", TiktokenAdapter("cl100k_base"))
        self.register("claude*", TiktokenAdapter("cl100k_base"))
        self._default = EstimationAdapter()

    def register(self, model_pattern: str, adapter: TokenizerAdapter) -> None:
        self._patterns.append((model_pattern, adapter))

    def _match(self, model_id: str) -> Optional[TokenizerAdapter]:
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

    def encode(self, model_id: str, text: str) -> List[int]:
        adapter = self._match(model_id)
        if adapter is not None:
            return adapter.encode(text)
        if self._default is not None:
            return self._default.encode(text)
        return []

    def decode(self, model_id: str, tokens: List[int]) -> str:
        adapter = self._match(model_id)
        if adapter is not None:
            return adapter.decode(tokens)
        if self._default is not None:
            return self._default.decode(tokens)
        return ""

    def list_models(self) -> List[str]:
        return [p for p, _ in self._patterns]
