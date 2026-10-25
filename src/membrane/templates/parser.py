"""Tokenizer + parser for Membrane prompt templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from membrane.templates.types import ParsedDocument

_EXTENDS_RE = re.compile(r"^extends\s+(?:\"([^\"]+)\"|'([^']+)'|(\w+))\s*$")


class TokType(Enum):
    TEXT = auto()
    VAR = auto()
    TAG = auto()
    EOF = auto()


@dataclass
class Token:
    kind: TokType
    value: str = ""


def tokenize(source: str) -> List[Token]:
    out: List[Token] = []
    i = 0
    n = len(source)
    while i < n:
        if source.startswith("{{", i):
            j = source.find("}}", i + 2)
            if j < 0:
                raise ValueError("unclosed {{")
            inner = source[i + 2 : j].strip()
            if re.match(r"^\w+$", inner):
                out.append(Token(TokType.VAR, inner))
            else:
                out.append(Token(TokType.TEXT, source[i : j + 2]))
            i = j + 2
            continue
        if source.startswith("{%", i):
            j = source.find("%}", i + 2)
            if j < 0:
                raise ValueError("unclosed {%")
            out.append(Token(TokType.TAG, source[i + 2 : j].strip()))
            i = j + 2
            continue
        j = i
        while j < n and not source.startswith("{{", j) and not source.startswith("{%", j):
            j += 1
        if j > i:
            out.append(Token(TokType.TEXT, source[i:j]))
        i = j
    out.append(Token(TokType.EOF))
    return out


def _extends_name(tag: str) -> Optional[str]:
    m = _EXTENDS_RE.match(tag.strip())
    if not m:
        return None
    return m.group(1) or m.group(2) or m.group(3)


def parse_nodes(tokens: List[Token], pos: list, end_pred) -> List[Any]:
    from membrane.templates.types import BlockNode, ForNode, IfNode, RoleBlockNode, TextNode, VarNode

    nodes: List[Any] = []
    while pos[0] < len(tokens):
        t = tokens[pos[0]]
        if t.kind == TokType.EOF:
            break
        if end_pred(t):
            break
        if t.kind == TokType.TEXT:
            nodes.append(TextNode(t.value))
            pos[0] += 1
            continue
        if t.kind == TokType.VAR:
            nodes.append(VarNode(t.value))
            pos[0] += 1
            continue
        if t.kind == TokType.TAG:
            tag = t.value.strip()
            pos[0] += 1
            if tag.startswith("if "):
                cond = tag[3:].strip()
                inner = parse_nodes(tokens, pos, lambda tt: tt.kind == TokType.TAG and tt.value.strip() == "endif")
                if pos[0] >= len(tokens) or tokens[pos[0]].value.strip() != "endif":
                    raise ValueError("expected endif")
                pos[0] += 1
                nodes.append(IfNode(condition=cond, children=inner))
                continue
            if tag.startswith("for "):
                rest = tag[4:].strip()
                if " in " not in rest:
                    raise ValueError("for needs 'in'")
                item, seq = rest.split(" in ", 1)
                item, seq = item.strip(), seq.strip()
                inner = parse_nodes(tokens, pos, lambda tt: tt.kind == TokType.TAG and tt.value.strip() == "endfor")
                if pos[0] >= len(tokens) or tokens[pos[0]].value.strip() != "endfor":
                    raise ValueError("expected endfor")
                pos[0] += 1
                nodes.append(ForNode(item=item, seq=seq, children=inner))
                continue
            if tag.startswith("block "):
                rest = tag[6:].strip()
                block_name = rest.split()[0] if rest else ""
                if not block_name:
                    raise ValueError("block needs a name")
                inner = parse_nodes(
                    tokens,
                    pos,
                    lambda tt: tt.kind == TokType.TAG and tt.value.strip() == "endblock",
                )
                if pos[0] >= len(tokens) or tokens[pos[0]].value.strip() != "endblock":
                    raise ValueError("expected endblock")
                pos[0] += 1
                nodes.append(BlockNode(name=block_name, children=inner))
                continue
            if tag in ("system", "user", "assistant"):
                role = tag
                end = f"end_{role}"
                inner = parse_nodes(
                    tokens,
                    pos,
                    lambda tt, r=role, e=end: tt.kind == TokType.TAG and tt.value.strip() == e,
                )
                if pos[0] >= len(tokens) or tokens[pos[0]].value.strip() != end:
                    raise ValueError(f"expected {end}")
                pos[0] += 1
                nodes.append(RoleBlockNode(role=role, children=inner))
                continue
            nodes.append(TextNode("{% " + tag + " %}"))
            continue
    return nodes


def parse_document(source: str) -> ParsedDocument:
    """Parse template; supports ``{% extends \"id\" %}`` + child ``{% block %}`` overrides."""
    toks = tokenize(source)
    pos = [0]
    while pos[0] < len(toks) and toks[pos[0]].kind == TokType.TEXT:
        if toks[pos[0]].value.strip():
            break
        pos[0] += 1
    if pos[0] < len(toks) and toks[pos[0]].kind == TokType.TAG:
        ext = _extends_name(toks[pos[0]].value)
        if ext:
            pos[0] += 1
            overrides: Dict[str, List[Any]] = {}

            while pos[0] < len(toks) and toks[pos[0]].kind != TokType.EOF:
                tt = toks[pos[0]]
                if tt.kind == TokType.TEXT and not tt.value.strip():
                    pos[0] += 1
                    continue
                if tt.kind != TokType.TAG or not tt.value.strip().startswith("block "):
                    raise ValueError("after {% extends %} only {% block name %} ... {% endblock %} are allowed")
                tag = tt.value.strip()
                rest = tag[6:].strip()
                bname = rest.split()[0] if rest else ""
                if not bname:
                    raise ValueError("block needs a name")
                pos[0] += 1
                inner = parse_nodes(
                    toks,
                    pos,
                    lambda t: t.kind == TokType.TAG and t.value.strip() == "endblock",
                )
                if pos[0] >= len(toks) or toks[pos[0]].value.strip() != "endblock":
                    raise ValueError("expected endblock")
                pos[0] += 1
                overrides[bname] = inner
            return ParsedDocument(extends_parent=ext, block_overrides=overrides, nodes=[])

    pos = [0]
    body = parse_nodes(toks, pos, lambda t: t.kind == TokType.EOF)
    return ParsedDocument(nodes=body)


def parse_template(source: str) -> List[Any]:
    doc = parse_document(source)
    if doc.extends_parent:
        raise ValueError(
            "This template uses {% extends %}; use PromptTemplate(..., loader={...}) "
            "with parent sources, or parse_document() for structured access."
        )
    return doc.nodes
