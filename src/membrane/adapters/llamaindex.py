"""LlamaIndex: convert retrieved nodes to Membrane memories and post-process."""

from __future__ import annotations

from typing import Any, List

from membrane.memory_protocol import MemoryItem


def node_to_memory(node: Any) -> MemoryItem:
    """Convert a LlamaIndex ``NodeWithScore`` (or similar) to :class:`~membrane.memory_protocol.MemoryItem`."""
    text = getattr(node, "text", None) or getattr(node, "get_content", lambda: "")()
    if callable(text):
        text = text()
    nid = getattr(node, "id_", None) or getattr(node, "node_id", None) or "node"
    score = getattr(node, "score", None)
    if score is None and hasattr(node, "metadata"):
        score = node.metadata.get("score")
    priority = 2
    if score is not None:
        try:
            priority = max(1, min(3, int(float(score) * 3)))
        except (TypeError, ValueError):
            pass
    return MemoryItem(
        id=str(nid),
        content=str(text),
        memory_type="semantic",
        priority=priority,
    )


class MembraneNodePostprocessor:
    """
    Postprocessor that runs Membrane's compiler on retrieved nodes.

    Requires ``llama-index-core`` and a configured :class:`~membrane.compiler.ContextCompiler`.
    """

    def __init__(self, compiler: Any, *, max_tokens: int = 8192, profile: Any = None) -> None:
        self._compiler = compiler
        self._max_tokens = max_tokens
        self._profile = profile

    def postprocess_nodes(self, nodes: List[Any], query_bundle: Any) -> List[Any]:
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
        try:
            from llama_index.core.schema import TextNode  # type: ignore
        except ImportError as e:
            raise ImportError("membrane[llamaindex] requires llama-index-core") from e
        text = "\n".join(
            str(m.get("content", "")) for m in getattr(res, "messages", []) or []
        )
        return [TextNode(text=text, metadata={"membrane": True})]
