"""LlamaIndex: convert retrieved nodes to Membrane memories and post-process."""

from __future__ import annotations

import contextlib
from typing import Any

from membrane.memory_protocol import KernelMemoryObject


def node_to_memory(node: Any) -> KernelMemoryObject:
    """Convert a LlamaIndex ``NodeWithScore`` (or similar) to
    :class:`~membrane.memory_protocol.KernelMemoryObject`.

    The retrieval score becomes ``relevance_score``; the firewall tier is always 2 (semantic).
    """
    text = getattr(node, "text", None) or getattr(node, "get_content", lambda: "")()
    if callable(text):
        text = text()
    nid = getattr(node, "id_", None) or getattr(node, "node_id", None) or "node"
    score = getattr(node, "score", None)
    if score is None and hasattr(node, "metadata"):
        score = node.metadata.get("score")
    relevance = 1.0
    if score is not None:
        with contextlib.suppress(TypeError, ValueError):
            relevance = max(0.0, float(score))
    return KernelMemoryObject(
        id=str(nid),
        payload=str(text),
        relevance_score=relevance,
        semantic_tier=2,
    )


try:
    from llama_index.core.bridge.pydantic import PrivateAttr  # type: ignore
    from llama_index.core.postprocessor.types import BaseNodePostprocessor  # type: ignore
    from llama_index.core.schema import NodeWithScore, TextNode  # type: ignore

    _HAS_LLAMA_INDEX = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAS_LLAMA_INDEX = False
    BaseNodePostprocessor = object  # type: ignore[assignment,misc]

    def PrivateAttr(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return None


class MembraneNodePostprocessor(BaseNodePostprocessor):  # type: ignore[misc,valid-type]
    """
    Postprocessor that runs Membrane's compiler on retrieved nodes.

    Requires ``llama-index-core`` and a configured :class:`~membrane.compiler.ContextCompiler`.
    """

    _compiler: Any = PrivateAttr()
    _max_tokens: int = PrivateAttr()
    _profile: Any = PrivateAttr()

    def __init__(self, compiler: Any, *, max_tokens: int = 8192, profile: Any = None) -> None:
        if not _HAS_LLAMA_INDEX:
            raise ImportError("membrane-aol[llamaindex] requires llama-index-core")
        super().__init__()
        self._compiler = compiler
        self._max_tokens = max_tokens
        self._profile = profile

    @classmethod
    def class_name(cls) -> str:
        return "MembraneNodePostprocessor"

    def _postprocess_nodes(self, nodes: list[Any], query_bundle: Any = None) -> list[Any]:
        memories = [node_to_memory(n) for n in nodes]
        q = ""
        if query_bundle is not None:
            q = str(getattr(query_bundle, "query_str", None) or getattr(query_bundle, "query", ""))
        compile_fn = getattr(self._compiler, "compile", None)
        if compile_fn is None:
            return nodes
        res = compile_fn(
            memories=memories,
            user_message=q or "context",
            profile=self._profile,
        )
        # Return a single synthetic node carrying compiled text (caller interprets)
        text = "\n".join(
            str(m.get("content", "")) for m in getattr(res, "messages", []) or []
        )
        return [NodeWithScore(node=TextNode(text=text, metadata={"membrane": True}))]
