"""Template AST node types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedDocument:
    """Result of parsing a template source (optional ``extends`` + block overrides)."""

    extends_parent: Optional[str] = None
    block_overrides: Dict[str, List[Any]] = field(default_factory=dict)
    nodes: List[Any] = field(default_factory=list)


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
    children: List[Any]


@dataclass
class ForNode:
    item: str
    seq: str
    children: List[Any]


@dataclass
class RoleBlockNode:
    role: str
    children: List[Any]


@dataclass
class BlockNode:
    """Named region ``{% block name %}...{% endblock %}`` (override via ``{% extends %}``)."""

    name: str
    children: List[Any]
