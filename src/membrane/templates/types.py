"""Template AST node types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedDocument:
    """Result of parsing a template source (optional ``extends`` + block overrides)."""

    extends_parent: str | None = None
    block_overrides: dict[str, list[Any]] = field(default_factory=dict)
    nodes: list[Any] = field(default_factory=list)


@dataclass
class TemplateVariable:
    name: str
    type: str = "str"
    default: Any = None
    required: bool = True


@dataclass
class TextNode:
    text: str


@dataclass
class VarNode:
    name: str


@dataclass
class IfNode:
    condition: str
    children: list[Any]


@dataclass
class ForNode:
    item: str
    seq: str
    children: list[Any]


@dataclass
class RoleBlockNode:
    role: str
    children: list[Any]


@dataclass
class BlockNode:
    """Named region ``{% block name %}...{% endblock %}`` (override via ``{% extends %}``)."""

    name: str
    children: list[Any]
