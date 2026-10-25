"""
LangChain integration.

**Collision prevention:** Membrane replaces LangChain's conversation memory when you use
:class:`MembranePromptTemplate`. Do not combine with ``ConversationBufferMemory`` /
``ConversationSummaryMemory`` on the same chain — call :func:`disable_langchain_memory` on the
chain or raise :class:`membrane.exceptions.MemoryCollisionError` if both are detected.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, cast

from membrane.exceptions import MemoryCollisionError
from membrane.memory_protocol import MemoryItem

MemoryType = Literal["semantic", "episodic", "procedural"]


def _memory_type(value: Any) -> MemoryType:
    text = str(value)
    if text in ("semantic", "episodic", "procedural"):
        return cast(MemoryType, text)
    return "semantic"


def langchain_doc_to_memory(doc: Any) -> MemoryItem:
    """Convert a LangChain ``Document`` to :class:`~membrane.memory_protocol.MemoryItem`."""
    try:
        from langchain_core.documents import Document  # type: ignore
    except ImportError as e:
        raise ImportError("membrane[langchain] requires langchain-core") from e
    if not isinstance(doc, Document):
        raise TypeError("expected langchain_core.documents.Document")
    return MemoryItem(
        id=str(doc.metadata.get("id", doc.metadata.get("source", "doc"))),
        content=doc.page_content,
        memory_type=_memory_type(doc.metadata.get("memory_type", "semantic")),
        priority=int(doc.metadata.get("priority", 2)),
    )


class MembranePromptTemplate:
    """
    Wraps formatting so ``ContextCompiler.compile`` runs before prompt construction.

    Pass ``compiler`` (:class:`~membrane.compiler.ContextCompiler`), ``memories``, and
    ``profile`` / kwargs supported by :meth:`~membrane.compiler.ContextCompiler.compile`.
    """

    def __init__(
        self,
        compiler: Any,
        template: str,
        *,
        memories: Sequence[Any],
        profile: Any = None,
        **compile_kwargs: Any,
    ) -> None:
        self._compiler = compiler
        self._template = template
        self._memories = memories
        self._profile = profile
        self._compile_kwargs = compile_kwargs

    def format(self, **kwargs: Any) -> str:
        user_message = kwargs.get("user_message") or kwargs.get("input") or ""
        if not user_message:
            raise ValueError("MembranePromptTemplate.format requires user_message or input")
        compile_fn = getattr(self._compiler, "compile", None)
        if compile_fn is None:
            raise TypeError("compiler must have compile()")
        res = compile_fn(
            memories=self._memories,
            user_message=str(user_message),
            profile=self._profile,
            **self._compile_kwargs,
        )
        msgs = getattr(res, "messages", []) or []
        core = "\n".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))
        return self._template.format(compiled_context=core, **kwargs)


def disable_langchain_memory(chain: Any) -> Any:
    """Remove common LangChain memory components from a chain (best-effort)."""
    mem_attrs = ("memory", "memory_key")
    for a in mem_attrs:
        if hasattr(chain, a):
            try:
                setattr(chain, a, None)
            except Exception:
                pass
    return chain


def detect_memory_collision(chain: Any, membrane_active: bool) -> None:
    if not membrane_active:
        return
    if hasattr(chain, "memory") and getattr(chain, "memory", None) is not None:
        raise MemoryCollisionError(
            "LangChain memory is set while Membrane manages context. "
            "Remove ConversationBufferMemory or disable Membrane."
        )
