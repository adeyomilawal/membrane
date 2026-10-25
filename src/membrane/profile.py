"""MembraneProfile — single configuration surface for ContextCompiler and Phase 1 modules."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, TypeVar

T = TypeVar("T", bound="MembraneProfile")


@dataclass
class MembraneProfile:
    """
    Bundles optional Phase 1 components and flags for ``ContextCompiler.compile()``.

    Prefer presets (``minimal``, ``coding``, ``conversational``) or ``from_yaml`` over
    ad-hoc kwargs on ``compile()``.
    """

    reranker: Any = None
    deduplicator: Any = None
    compression: Any = None
    policies: Any = None
    metrics: Any = None
    embedding_cache: Any = None
    classifier: Any = None
    freshness: Any = None
    assertions: Any = None
    recorder: Any = None
    tokenizer_registry: Any = None
    trace_exporters: List[Any] = field(default_factory=list)
    enable_lineage: bool = False
    enable_profiler: bool = False
    lineage_tracker: Any = None
    latency_profiler: Any = None
    metrics_collector: Any = None
    # Explainability: documented FinalScore-style diagnostics on CompileResult.debug_trace
    include_ranking_diagnostics: bool = True
    ranking_weights: Any = None  # Optional[RankingWeights] from membrane.ranking_diagnostics
    # Async observability: optional :class:`membrane.runtime.signals.SignalBus` for compile_async
    signal_bus: Any = None

    @classmethod
    def minimal(cls) -> MembraneProfile:
        """Phase 0 behavior: no optional intelligence modules."""
        return cls()

    @classmethod
    def coding(cls) -> MembraneProfile:
        """Coding-oriented preset: classifier, dedup, freshness, policies, profiler."""
        from membrane.classifier import MemoryClassifier
        from membrane.dedup import SemanticDeduplicator
        from membrane.freshness import FreshnessManager, StepDecay
        from membrane.policies import PolicyEngine

        return cls(
            classifier=MemoryClassifier(),
            deduplicator=SemanticDeduplicator(similarity_threshold=0.90),
            freshness=FreshnessManager(
                decay_fn=StepDecay(
                    [
                        (86400.0, 1.0),
                        (604800.0, 0.8),
                        (float("inf"), 0.5),
                    ]
                )
            ),
            policies=PolicyEngine.from_dict(
                {
                    "policies": [
                        {
                            "name": "code_boost",
                            "condition": "always",
                            "actions": [
                                {"set_budget": {"procedural": 0.45, "semantic": 0.20}},
                            ],
                        }
                    ]
                }
            ),
            enable_profiler=True,
        )

    @classmethod
    def conversational(cls) -> MembraneProfile:
        """Chat-oriented preset."""
        from membrane.classifier import MemoryClassifier
        from membrane.dedup import SemanticDeduplicator
        from membrane.freshness import FreshnessManager, LinearDecay

        return cls(
            classifier=MemoryClassifier(),
            deduplicator=SemanticDeduplicator(similarity_threshold=0.92),
            freshness=FreshnessManager(decay_fn=LinearDecay(max_age_seconds=3600.0)),
            enable_profiler=True,
        )

    @classmethod
    def from_yaml(cls, path: str) -> MembraneProfile:
        """Load a partial profile from YAML; unknown keys are ignored."""
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "MembraneProfile.from_yaml requires PyYAML. Install with: pip install pyyaml"
            ) from e

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MembraneProfile:
        """Build profile from a plain dict (e.g. JSON)."""
        base = cls.minimal()
        known = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k in known and k != "trace_exporters":
                setattr(base, k, v)
        return base

    def override(self: T, **kwargs: Any) -> T:
        """Return a shallow copy with selected fields replaced."""
        data = {f.name: copy.copy(getattr(self, f.name)) for f in fields(self)}
        for k, v in kwargs.items():
            if k not in data:
                raise TypeError(f"MembraneProfile has no field {k!r}")
            data[k] = v
        return self.__class__(**data)
