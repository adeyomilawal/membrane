"""Render parsed templates to OpenAI-style message dicts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from membrane.templates.parser import parse_document
from membrane.templates.types import (
    BlockNode,
    ForNode,
    IfNode,
    RoleBlockNode,
    TextNode,
    VarNode,
)


class TemplateMissingVariable(KeyError):
    """Raised when a required template variable is unbound."""


def merge_block_nodes(nodes: List[Any], overrides: Dict[str, List[Any]]) -> List[Any]:
    """Apply child ``block`` overrides onto a parent AST (single ``extends`` level; nested blocks supported)."""
    out: List[Any] = []
    for n in nodes:
        if isinstance(n, BlockNode):
            rep = overrides.get(n.name)
            ch: List[Any] = list(rep) if rep is not None else list(n.children)
            out.append(BlockNode(n.name, merge_block_nodes(ch, overrides)))
        elif isinstance(n, IfNode):
            out.append(IfNode(n.condition, merge_block_nodes(n.children, overrides)))
        elif isinstance(n, ForNode):
            out.append(ForNode(n.item, n.seq, merge_block_nodes(n.children, overrides)))
        elif isinstance(n, RoleBlockNode):
            out.append(RoleBlockNode(n.role, merge_block_nodes(n.children, overrides)))
        else:
            out.append(n)
    return out


class PromptTemplate:
    """
    Lightweight template engine (variable substitution, if/for, role blocks).

    Use :meth:`render` with ``compile_result=`` to inject ``compiled_context`` from
    :class:`~membrane.compiler.CompileResult`.
    """

    def __init__(
        self,
        template: str,
        *,
        variables: Optional[List[Any]] = None,
        parent: Optional["PromptTemplate"] = None,
        loader: Optional[Dict[str, str]] = None,
    ) -> None:
        self._source = template
        self._loader = loader
        doc = parse_document(template)
        if doc.extends_parent:
            if not loader or doc.extends_parent not in loader:
                raise KeyError(
                    f"parent template {doc.extends_parent!r} not found in loader "
                    "(pass loader={{'id': source, ...}} matching extends name)"
                )
            parent_src = loader[doc.extends_parent]
            parent_doc = parse_document(parent_src)
            if parent_doc.extends_parent:
                raise ValueError(
                    "Chained {% extends %} is not supported; provide a flat parent template."
                )
            self._ast = merge_block_nodes(parent_doc.nodes, doc.block_overrides)
        else:
            self._ast = doc.nodes
        self._variables = variables
        self._parent = parent

    @classmethod
    def from_file(cls, path: str) -> "PromptTemplate":
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    @property
    def variables(self) -> List[Any]:
        return list(self._variables or [])

    def validate(self, **kwargs: Any) -> List[str]:
        return [name for name in self._collect_vars(self._ast) if name not in kwargs]

    def _collect_vars(self, nodes: List[Any]) -> List[str]:
        names: List[str] = []
        for n in nodes:
            if isinstance(n, VarNode):
                names.append(n.name)
            elif isinstance(n, IfNode):
                names.extend(self._collect_vars(n.children))
                names.append(n.condition.strip())
            elif isinstance(n, ForNode):
                names.append(n.seq)
                names.extend(self._collect_vars(n.children))
            elif isinstance(n, RoleBlockNode):
                names.extend(self._collect_vars(n.children))
            elif isinstance(n, BlockNode):
                names.extend(self._collect_vars(n.children))
        return names

    def render(
        self,
        *,
        compile_result: Any = None,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        ctx = dict(kwargs)
        if compile_result is not None:
            msgs = getattr(compile_result, "messages", None) or []
            text = "\n".join(
                str(m.get("content", "")) for m in msgs if isinstance(m, dict)
            )
            ctx.setdefault("compiled_context", text)
        missing = self.validate(**ctx)
        if missing:
            raise TemplateMissingVariable(f"Missing template variables: {missing}")
        return self._render_nodes(self._ast, ctx)

    def _render_nodes(self, nodes: List[Any], ctx: Dict[str, Any]) -> List[Dict[str, str]]:
        by_role: Dict[str, List[str]] = {"system": [], "user": [], "assistant": []}
        for n in nodes:
            self._render_node(n, ctx, by_role)
        out: List[Dict[str, str]] = []
        for role in ("system", "user", "assistant"):
            if by_role[role]:
                out.append({"role": role, "content": "".join(by_role[role]).strip()})
        return out

    def _render_node(
        self,
        n: Any,
        ctx: Dict[str, Any],
        by_role: Dict[str, List[str]],
        current_role: str = "system",
    ) -> None:
        if isinstance(n, TextNode):
            by_role[current_role].append(n.text)
        elif isinstance(n, VarNode):
            val = ctx.get(n.name, "")
            by_role[current_role].append(str(val))
        elif isinstance(n, IfNode):
            if self._truthy(n.condition, ctx):
                for c in n.children:
                    self._render_node(c, ctx, by_role, current_role)
        elif isinstance(n, ForNode):
            seq = ctx.get(n.seq, [])
            if not isinstance(seq, list):
                seq = list(seq) if seq is not None else []
            for item in seq:
                sub = dict(ctx)
                sub[n.item] = item
                for c in n.children:
                    self._render_node(c, sub, by_role, current_role)
        elif isinstance(n, RoleBlockNode):
            for c in n.children:
                self._render_node(c, ctx, by_role, n.role)
        elif isinstance(n, BlockNode):
            for c in n.children:
                self._render_node(c, ctx, by_role, current_role)

    def _truthy(self, cond: str, ctx: Dict[str, Any]) -> bool:
        c = cond.strip()
        if c in ctx:
            return bool(ctx[c])
        parts = c.split()
        if len(parts) == 2 and parts[0] == "not":
            return not bool(ctx.get(parts[1]))
        return bool(ctx.get(c, False))
