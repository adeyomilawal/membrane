"""Pluggable memory types for prompt injection (protocol + concrete dataclasses)."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from typing import Any, Literal, Protocol, runtime_checkable
from collections.abc import Mapping, Sequence


@runtime_checkable
class KernelMemoryRecord(Protocol):
    """Minimum surface for objects serialized into the sacrifice zone."""

    @property
    def id(self) -> str | None:
        ...

    @property
    def memory_kind(self) -> Literal["semantic", "episodic", "procedural"]:
        ...


KERNEL_MEMORY_SCHEMA_VERSION = 1
KERNEL_MEMORY_SCHEMA_ID = "https://membrane.consilience.icu/schemas/kernel-memory-object/v1"


@dataclass
class KernelMemoryObject:
    """
    Versioned input contract for :meth:`membrane.compiler.ContextCompiler.compile`.

    ``semantic_tier`` matches firewall tiers: 1 = procedural, 2 = semantic, 3 = episodic
    (same convention as :class:`MemoryItem` ``priority``).

    JSON interchange uses the same field names; see packaged schema
    ``membrane.schemas.kernel_memory_object.schema.json``.
    """

    payload: str
    relevance_score: float = 1.0
    semantic_tier: int = 2
    temporal_weight: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if self.semantic_tier not in (1, 2, 3):
            raise ValueError("semantic_tier must be 1 (procedural), 2 (semantic), or 3 (episodic)")
        if self.relevance_score < 0.0:
            raise ValueError("relevance_score must be >= 0")
        if self.temporal_weight < 0.0:
            raise ValueError("temporal_weight must be >= 0")

    @property
    def content(self) -> str:
        """Alias for ``payload`` — compiler and serializers use ``content``."""
        return self.payload

    @property
    def priority(self) -> int:
        """Matches :class:`MemoryItem` / firewall tier for
        :func:`membrane.compiler._tier_for_object`.
        """
        return self.semantic_tier

    @property
    def memory_kind(self) -> Literal["semantic", "episodic", "procedural"]:
        if self.semantic_tier == 1:
            return "procedural"
        if self.semantic_tier == 2:
            return "semantic"
        return "episodic"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KernelMemoryObject:
        """Build from a mapping (e.g. JSON). Requires ``payload``; coerces types safely."""
        if "payload" not in data:
            raise ValueError("KernelMemoryObject requires 'payload'")
        raw = data["payload"]
        payload_s = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True, default=str)
        st = max(1, min(3, int(data.get("semantic_tier", 2))))
        rs = float(data.get("relevance_score", 1.0))
        tw = float(data.get("temporal_weight", 1.0))
        prov = data.get("provenance")
        if prov is not None and not isinstance(prov, dict):
            raise ValueError("provenance must be an object when present")
        pid = data.get("id")
        return cls(
            id=str(pid) if pid is not None else str(uuid.uuid4()),
            payload=payload_s,
            relevance_score=rs,
            semantic_tier=st,
            temporal_weight=tw,
            provenance=dict(prov or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict matching the packaged schema."""
        return {
            "id": self.id,
            "payload": self.payload,
            "relevance_score": self.relevance_score,
            "semantic_tier": self.semantic_tier,
            "temporal_weight": self.temporal_weight,
            "provenance": dict(self.provenance),
        }


def normalize_compile_memories(memories: Sequence[Any]) -> list[Any]:
    """
    Normalize inputs for compilation.

    - :class:`KernelMemoryObject` instances pass through.
    - ``dict`` values with a ``payload`` key become :class:`KernelMemoryObject`
      (JSON interchange / retrieval systems).
    - All other objects pass through unchanged (duck-typed memories).
    """
    out: list[Any] = []
    for m in memories:
        if isinstance(m, KernelMemoryObject):
            out.append(m)
        elif isinstance(m, dict) and "payload" in m:
            out.append(KernelMemoryObject.from_dict(m))
        else:
            out.append(m)
    return out


def load_kernel_memory_json_schema() -> dict[str, Any]:
    """Load the packaged JSON Schema for :class:`KernelMemoryObject` (draft 2020-12)."""
    pkg = resources.files("membrane.schemas").joinpath("kernel_memory_object.schema.json")
    return json.loads(pkg.read_text(encoding="utf-8"))


@dataclass
class MemoryItem:
    """
    Minimal memory row for RAG-style pipelines (shipping plan public API).

    ``priority`` matches firewall tiers: 1=procedural, 2=semantic, 3=episodic.
    """

    id: str
    content: str
    memory_type: Literal["semantic", "episodic", "procedural"] = "semantic"
    priority: int = 2

    @property
    def memory_kind(self) -> Literal["semantic", "episodic", "procedural"]:
        return self.memory_type


def memory_item_to_kernel_memory(item: MemoryItem) -> KernelMemoryObject:
    """Bridge legacy :class:`MemoryItem` rows to :class:`KernelMemoryObject`."""
    return KernelMemoryObject(
        id=item.id,
        payload=item.content,
        relevance_score=1.0,
        semantic_tier=item.priority,
        temporal_weight=1.0,
        provenance={"bridge": "memory_item"},
    )


def kernel_memory_to_memory_item(obj: KernelMemoryObject) -> MemoryItem:
    """Bridge back to :class:`MemoryItem` for APIs that still expect it."""
    return MemoryItem(
        id=obj.id,
        content=obj.payload,
        memory_type=obj.memory_kind,
        priority=obj.semantic_tier,
    )


@dataclass
class SemanticMemory:
    """Structured semantic fact."""

    id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0

    @property
    def memory_kind(self) -> Literal["semantic", "episodic", "procedural"]:
        return "semantic"


@dataclass
class EpisodicMemory:
    """Structured episodic memory."""

    id: str
    query: str = ""
    stage3_synthesis: str = ""
    timestamp: datetime | str | None = None
    domain: str | None = None
    learned_insights: list[str] | None = None

    @property
    def memory_kind(self) -> Literal["semantic", "episodic", "procedural"]:
        return "episodic"


@dataclass
class ProceduralStep:
    """Single step in a procedure."""

    step_number: int
    action: str = ""
    description: str = ""
    expected_outcome: str = ""


@dataclass
class ProceduralMemory:
    """Structured procedural memory."""

    id: str
    name: str
    description: str = ""
    steps: list[ProceduralStep] = field(default_factory=list)
    success_rate: float | None = None

    @property
    def memory_kind(self) -> Literal["semantic", "episodic", "procedural"]:
        return "procedural"


def infer_memory_kind(obj: Any) -> Literal["semantic", "episodic", "procedural"] | None:
    """
    Infer memory kind for duck-typed objects (no memory_kind attribute).

    Order: memory_kind, memory_type string, semantic triple, procedural, episodic.
    """
    if obj is None:
        return None
    if hasattr(obj, "memory_kind"):
        try:
            mk = obj.memory_kind
            if mk in ("semantic", "episodic", "procedural"):
                return mk  # type: ignore[return-value]
        except Exception:
            pass
    mt = getattr(obj, "memory_type", None)
    if mt in ("semantic", "episodic", "procedural"):
        return mt  # type: ignore[return-value]
    subj = getattr(obj, "subject", None)
    pred = getattr(obj, "predicate", None)
    objv = getattr(obj, "object", None)
    if subj is not None and pred is not None and objv is not None:
        return "semantic"
    steps = getattr(obj, "steps", None)
    name = getattr(obj, "name", None)
    if steps is not None and name is not None:
        return "procedural"
    if (
        getattr(obj, "stage3_synthesis", None) is not None
        or getattr(obj, "query", None) is not None
    ):
        return "episodic"
    return None
