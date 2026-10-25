"""Haystack: document conversion and prompt assembly hook."""

from __future__ import annotations

from typing import Any, Literal, cast
from collections.abc import Sequence

from membrane.memory_protocol import MemoryItem

MemoryType = Literal["semantic", "episodic", "procedural"]


def _memory_type(value: Any) -> MemoryType:
    text = str(value)
    if text in ("semantic", "episodic", "procedural"):
        return cast(MemoryType, text)
    return "semantic"


def haystack_document_to_memory(doc: Any) -> MemoryItem:
    """Convert Haystack ``Document`` to :class:`~membrane.memory_protocol.MemoryItem`."""
    try:
        from haystack import Document  # type: ignore
    except ImportError as e:
        raise ImportError("membrane-aol[haystack] requires haystack-ai") from e
    if not isinstance(doc, Document):
        raise TypeError("expected haystack.Document")
    meta = doc.meta or {}
    return MemoryItem(
        id=str(meta.get("id", doc.id or "doc")),
        content=doc.content or "",
        memory_type=_memory_type(meta.get("memory_type", "semantic")),
        priority=int(meta.get("priority", 2)),
    )


class MembranePromptBuilder:
    """Callable that returns Membrane-compiled system text for Haystack pipelines."""

    def __init__(self, compiler: Any, memories: Sequence[Any], *, profile: Any = None) -> None:
        self._compiler = compiler
        self._memories = memories
        self._profile = profile

    def __call__(self, query: str) -> str:
        fn = getattr(self._compiler, "compile", None)
        if fn is None:
            raise TypeError("compiler must have compile()")
        res = fn(memories=self._memories, user_message=query, profile=self._profile)
        msgs = getattr(res, "messages", []) or []
        return "\n".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))
