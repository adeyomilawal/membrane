"""Render parsed templates to OpenAI-style message dicts."""

from __future__ import annotations

from typing import Any

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


def merge_block_nodes(nodes: list[Any], overrides: dict[str, list[Any]]) -> list[Any]:
    """Apply child ``block`` overrides onto a parent AST (single ``extends`` level; nested
    blocks supported).
    """
    out: list[Any] = []
    for n in nodes:
        if isinstance(n, BlockNode):
            rep = overrides.get(n.name)
            ch: list[Any] = list(rep) if rep is not None else list(n.children)
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
        variables: list[Any] | None = None,
        parent: PromptTemplate | None = None,
        loader: dict[str, str] | None = None,
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
    def from_file(cls, path: str) -> PromptTemplate:
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    @property
    def variables(self) -> list[Any]:
        return list(self._variables or [])

    def validate(self, **kwargs: Any) -> list[str]:
        return [name for name in self._collect_vars(self._ast) if name not in kwargs]

    def _collect_vars(self, nodes: list[Any], bound: frozenset = frozenset()) -> list[str]:
        names: list[str] = []
        for n in nodes:
            if isinstance(n, VarNode):
                if n.name not in bound:
                    names.append(n.name)
            elif isinstance(n, IfNode):
                # conditions are optional (absent = falsy); only body variables are required
                names.extend(self._collect_vars(n.children, bound))
            elif isinstance(n, ForNode):
                if n.seq not in bound:
                    names.append(n.seq)
                names.extend(self._collect_vars(n.children, bound | {n.item}))
            elif isinstance(n, (RoleBlockNode, BlockNode)):
                names.extend(self._collect_vars(n.children, bound))
        return names

    def render(
        self,
        *,
        compile_result: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
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

    def _render_nodes(self, nodes: list[Any], ctx: dict[str, Any]) -> list[dict[str, str]]:
        # each segment: [role, parts, is_role_block]; role blocks never merge with neighbours
        segments: list[list[Any]] = []
        for n in nodes:
            self._render_node(n, ctx, segments)
        out: list[dict[str, str]] = []
        for role, parts, _ in segments:
            content = "".join(parts).strip()
            if content:
                out.append({"role": role, "content": content})
        return out

    def _render_node(
        self,
        n: Any,
        ctx: dict[str, Any],
        segments: list[list[Any]],
    ) -> None:
        if isinstance(n, TextNode):
            self._current(segments).append(n.text)
        elif isinstance(n, VarNode):
            val = ctx.get(n.name, "")
            self._current(segments).append(str(val))
        elif isinstance(n, IfNode):
            if self._truthy(n.condition, ctx):
                for c in n.children:
                    self._render_node(c, ctx, segments)
        elif isinstance(n, ForNode):
            seq = ctx.get(n.seq, [])
            if not isinstance(seq, list):
                seq = list(seq) if seq is not None else []
            for item in seq:
                sub = dict(ctx)
                sub[n.item] = item
                for c in n.children:
                    self._render_node(c, sub, segments)
        elif isinstance(n, RoleBlockNode):
            segments.append([n.role, [], True])
            for c in n.children:
                self._render_node(c, ctx, segments)
            # text after the block starts a fresh implicit system segment
            segments.append(["system", [], False])
        elif isinstance(n, BlockNode):
            for c in n.children:
                self._render_node(c, ctx, segments)

    @staticmethod
    def _current(segments: list[list[Any]]) -> list[str]:
        if not segments:
            segments.append(["system", [], False])
        return segments[-1][1]

    def _truthy(self, cond: str, ctx: dict[str, Any]) -> bool:
        c = cond.strip()
        if c in ctx:
            return bool(ctx[c])
        parts = c.split()
        if len(parts) == 2 and parts[0] == "not":
            return not bool(ctx.get(parts[1]))
        return bool(ctx.get(c, False))
