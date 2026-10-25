"""CrewAI: agent-level context hook pattern."""

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


def document_to_memory(doc: Any) -> MemoryItem:
    """Map a generic CrewAI / dict-like document to
    :class:`~membrane.memory_protocol.MemoryItem`.
    """
    if isinstance(doc, dict):
        return MemoryItem(
            id=str(doc.get("id", "doc")),
            content=str(doc.get("content", "")),
            memory_type=_memory_type(doc.get("memory_type", "semantic")),
            priority=int(doc.get("priority", 2)),
        )
    content = getattr(doc, "content", None) or getattr(doc, "raw", str(doc))
    return MemoryItem(
        id=str(getattr(doc, "id", "doc")),
        content=str(content),
        memory_type="semantic",
        priority=2,
    )


def compile_for_agent(
    compiler: Any,
    *,
    memories: Sequence[Any],
    user_message: str,
    profile: Any = None,
) -> Any:
    """Run :meth:`~membrane.compiler.ContextCompiler.compile` for one agent turn."""
    fn = getattr(compiler, "compile", None)
    if fn is None:
        raise TypeError("compiler must have compile()")
    return fn(memories=memories, user_message=user_message, profile=profile)
