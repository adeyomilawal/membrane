"""Resolve MembraneProfile and apply Phase 1 preprocessing to memories."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from membrane.classifier.classifier import MemoryClassifier, MemoryKind
from membrane.firewall import count_tokens
from membrane.memory_protocol import MemoryItem


def resolve_profile(
    profile: Any,
    *,
    reranker: Any = None,
    deduplicator: Any = None,
    compression: Any = None,
    policies: Any = None,
    metrics: Any = None,
    embedding_cache: Any = None,
    classifier: Any = None,
    freshness: Any = None,
    assertions: Any = None,
    recorder: Any = None,
    tokenizer_registry: Any = None,
    trace_exporters: Any = None,
    enable_lineage: Optional[bool] = None,
    enable_profiler: Optional[bool] = None,
    metrics_collector: Any = None,
    lineage_tracker: Any = None,
) -> Any:
    """Return a MembraneProfile; raise if both ``profile`` and component kwargs are set."""
    from membrane.profile import MembraneProfile

    component_kwargs: Dict[str, Any] = {
        "reranker": reranker,
        "deduplicator": deduplicator,
        "compression": compression,
        "policies": policies,
        "metrics": metrics,
        "embedding_cache": embedding_cache,
        "classifier": classifier,
        "freshness": freshness,
        "assertions": assertions,
        "recorder": recorder,
        "tokenizer_registry": tokenizer_registry,
        "metrics_collector": metrics_collector,
        "lineage_tracker": lineage_tracker,
    }
    if trace_exporters is not None:
        component_kwargs["trace_exporters"] = trace_exporters
    non_none = {k: v for k, v in component_kwargs.items() if v is not None}
    if enable_lineage is not None:
        non_none["enable_lineage"] = enable_lineage
    if enable_profiler is not None:
        non_none["enable_profiler"] = enable_profiler

    if profile is not None:
        if non_none:
            raise ValueError(
                "Pass either profile= or individual Pro component kwargs, not both"
            )
        return profile

    if not non_none:
        return MembraneProfile.minimal()

    base = MembraneProfile.minimal()
    for k, v in non_none.items():
        setattr(base, k, v)
    return base


def _kind_to_memory_type(kind: MemoryKind) -> str:
    if kind == MemoryKind.PROCEDURAL:
        return "procedural"
    if kind in (MemoryKind.EPISODIC, MemoryKind.CONVERSATIONAL):
        return "episodic"
    return "semantic"


def apply_classifier(
    classifier: Optional[MemoryClassifier],
    memories: Sequence[Any],
) -> List[Any]:
    if classifier is None:
        return list(memories)
    out: List[Any] = []
    for m in memories:
        text = str(getattr(m, "content", m))
        cr = classifier.classify(text)
        mt = _kind_to_memory_type(cr.kind)
        if isinstance(m, MemoryItem):
            m = MemoryItem(
                id=m.id,
                content=m.content,
                memory_type=mt,  # type: ignore[arg-type]
                priority=m.priority,
            )
        else:
            if hasattr(m, "memory_type"):
                try:
                    m.memory_type = mt  # type: ignore[attr-defined]
                except Exception:
                    setattr(m, "memory_type", mt)
            else:
                setattr(m, "_membrane_inferred_type", mt)
        out.append(m)
    return out


def estimate_memory_tokens(memories: Sequence[Any]) -> int:
    total = 0
    for m in memories:
        total += count_tokens(str(getattr(m, "content", m)))
    return total


def policy_feature_disabled(policy_disable: Optional[Sequence[str]], name: str) -> bool:
    if not policy_disable:
        return False
    return name in policy_disable
