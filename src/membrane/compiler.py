"""Context compilation: U-shaped layout, pre-flight compression, firewall, glaciation.

Pipeline stages are documented in ``membrane/future.md`` (Compile Pipeline Order) and in
:class:`ContextCompiler` — memory shaping, then layout→trim→format, then
:class:`CompileResult`, then assertions and observability on the complete result.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

import membrane.config as os_config
from membrane.async_preflight import get_preflight_cache, set_preflight_cache
from membrane.compile_wiring import (
    apply_classifier,
    estimate_memory_tokens,
    policy_feature_disabled,
    resolve_profile,
)
from membrane.exceptions import FeatureNotAvailable
from membrane.firewall import (
    TIER_EPISODIC,
    TIER_PROCEDURAL,
    TIER_SEMANTIC,
    PreFlightContextFirewall,
    _ref_map_from_text,
    count_tokens,
)
from membrane.layout import ContextLayoutEngine
from membrane.observability.profiler import LatencyProfiler
from membrane.lod import compute_attention_shift
from membrane.memory_protocol import infer_memory_kind, normalize_compile_memories
from membrane.metrics import CompileMetrics, MetricsCallback, build_compile_metrics
from membrane.provider_registry import has_prompt_caching, lookup_context_cap
from membrane.ref_map import set_ref_map_for_conversation
from membrane.runtime.signals import Signal, SignalKind

logger = logging.getLogger(__name__)


def _record_jta_lineage_from_compression(
    lineage: Any,
    *,
    compression_strategy_used: Optional[str],
    compression_details: Dict[str, Any],
    segment_id: str = "middle_compressed",
) -> None:
    """Record per-alias substitution events when JTA compression ran."""
    if lineage is None:
        return
    record = getattr(lineage, "record_event", None)
    if not callable(record):
        return

    jta_aux = compression_details.get("jta_lineage_details")
    if isinstance(jta_aux, dict):
        _record_jta_lineage_from_compression(
            lineage,
            compression_strategy_used="jta_deterministic",
            compression_details=jta_aux,
            segment_id=segment_id,
        )

    if not compression_strategy_used or not str(compression_strategy_used).startswith(
        "jta"
    ):
        return
    for decision in compression_details.get("alias_decisions") or []:
        if not isinstance(decision, dict):
            continue
        record(
            segment_id,
            stage="compression",
            action="jta_alias_map",
            detail={
                "phrase": decision.get("phrase"),
                "alias": decision.get("alias"),
                "token_savings": decision.get("token_savings"),
                "roi": decision.get("token_savings"),
                "frequency": decision.get("frequency"),
                "phrase_token_cost": decision.get("phrase_token_cost"),
                "header_token_cost": decision.get("header_token_cost"),
            },
        )
    aliases_reused = int(compression_details.get("aliases_reused", 0) or 0)
    if aliases_reused > 0 or compression_details.get("incremental_cache_hit"):
        record(
            segment_id,
            stage="compression",
            action="jta_incremental_cache",
            detail={
                "aliases_reused": aliases_reused,
                "incremental_cache_hit": bool(
                    compression_details.get("incremental_cache_hit")
                ),
                "delta_chars_scanned": compression_details.get("delta_chars_scanned"),
                "new_aliases_mapped": compression_details.get("new_aliases_mapped"),
            },
        )


def _stage(profiler: Optional[LatencyProfiler], name: str):
    if profiler is None:
        return contextlib.nullcontext()
    return profiler.stage(name)


def _build_ranking_diagnostics_for_compile(
    prof: Any,
    retrieved: List[Tuple[Any, int]],
    segment_weights: Optional[List[float]],
    middle_segments: List[Tuple[str, int]],
    *,
    ranking_weights: Optional[Any] = None,
    include_ranking_diagnostics: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Document-style per-memory scores; pipeline behavior is unchanged."""
    from membrane.ranking_diagnostics import (
        RankingWeights,
        compute_ranking_diagnostics,
        firewall_weights_aligned,
    )

    inc = (
        include_ranking_diagnostics
        if include_ranking_diagnostics is not None
        else getattr(prof, "include_ranking_diagnostics", True)
    )
    if not inc:
        return None
    rw = ranking_weights
    if rw is None:
        rw = getattr(prof, "ranking_weights", None)
    if rw is None:
        rw = RankingWeights()
    diag = compute_ranking_diagnostics(
        retrieved, segment_weights=segment_weights, weights=rw
    )
    diag["firewall_weights_aligned"] = firewall_weights_aligned(
        len(middle_segments), segment_weights
    )
    return diag


def _merge_compile_debug_trace(
    policy_kw: Dict[str, Any],
    segment_weights_used: bool,
    ranking_diag: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "policy_kw": {
            k: v
            for k, v in policy_kw.items()
            if k.startswith("policy_") or k in ("max_tokens",)
        },
        "segment_weights_used": segment_weights_used,
    }
    if ranking_diag:
        out["ranking_diagnostics"] = ranking_diag
    return out


def _run_trace_exporters(prof: Any, result: CompileResult) -> None:
    """Invoke optional Langfuse/LangSmith exporters attached to the profile."""
    exporters = getattr(prof, "trace_exporters", None) or []
    for exp in exporters:
        fn = getattr(exp, "export_compile_run", None)
        if not callable(fn):
            continue
        try:
            fn(result)
        except Exception as exc:
            logger.debug("trace exporter export_compile_run failed: %s", exc, exc_info=True)


def _emit_compile_metrics(
    callback: Optional[MetricsCallback],
    metrics: CompileMetrics,
) -> None:
    if callback is None:
        return
    try:
        callback.on_compile(metrics)
    except Exception as exc:
        logger.warning("metrics_callback failed: %s", exc, exc_info=True)


PREFLIGHT_ANALYST_SYSTEM = """You are an Intelligence Analyst. Your task is to compress retrieved long-term memory into a dense summary for the middle "sacrifice zone" of a prompt. Rules:
- Output ONLY the compressed summary; no preamble, no "Summary:" label.
- Use triples where possible: (Subject|Predicate|Object|Confidence).
- Use bracket-dense episodes when relevant: [Mem:N|YYYY-MM-DD]: one-line event.
- Preserve key entities, dates, and causal links.
- Stay strictly under the requested token limit (dense, no filler)."""

MEMORY_PLACEMENT_SYSTEM = "system"
MEMORY_PLACEMENT_END = "end"
MEMORY_PLACEMENTS = (MEMORY_PLACEMENT_SYSTEM, MEMORY_PLACEMENT_END)

INTERRUPT_INSTRUCTION = """If a detail is missing from the provided context, do not hallucinate. Output exactly: <SYSTEM_INTERRUPT_FETCH[Your precise query for the missing detail]> so the system can fetch it. Use request_memory_expansion(local_id) for [Mem:N] references when you need full prose; use search_cognitive_memory(query) for new searches."""


class SecurityFlag(str, Enum):
    ANONYMOUS_MODE = "anonymous"
    AUTHENTICATED = "authenticated"


@dataclass
class CompileResult:
    messages: List[Dict[str, Any]]
    ref_map: Dict[int, str]
    ref_map_type: Optional[Dict[int, str]] = None
    system_alerts: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Phase 1 optional attachments
    debug_trace: Optional[Dict[str, Any]] = None
    latency_profile: Optional[Any] = None
    dedup_report: Optional[Any] = None
    lineage_report: Optional[Any] = None
    assertion_report: Optional[Any] = None
    freshness_flags: Optional[List[str]] = None

    def to_debug_dict(self) -> Dict[str, Any]:
        """Structured snapshot for visualizer / export."""
        return {
            "messages": self.messages,
            "ref_map": self.ref_map,
            "ref_map_type": self.ref_map_type,
            "metadata": self.metadata,
            "dedup_report": getattr(self.dedup_report, "__dict__", self.dedup_report),
            "lineage_report": (
                self.lineage_report.to_dict()
                if self.lineage_report is not None
                and hasattr(self.lineage_report, "to_dict")
                else self.lineage_report
            ),
            "latency_profile": (
                self.latency_profile.to_dict() if self.latency_profile is not None else None
            ),
            "assertion_report": getattr(
                self.assertion_report, "__dict__", self.assertion_report
            ),
        }

    def diff(self, other: "CompileResult") -> Any:
        raise FeatureNotAvailable("CompileResult.diff")

    def explain(self, segment_id: str) -> str:
        if self.lineage_report is None:
            return "No lineage_report on this CompileResult."
        return self.lineage_report.explain(segment_id)


@dataclass
class CompileQuote:
    """
    Pre-flight style summary from a full :meth:`ContextCompiler.compile` run.

    :meth:`ContextCompiler.estimate` runs the same pipeline as :meth:`~ContextCompiler.compile`
    and returns this object plus the compiled :class:`CompileResult`, so token counts match
    execution. There is no cheaper “approximate” path.

    ``estimated_cost_usd`` is **input-side only** (compiled prompt tokens × your rate). It is
    ``None`` unless you pass ``price_per_1k_input_tokens`` to :meth:`~ContextCompiler.estimate`.
    Provider list prices change often; the library does not ship authoritative USD rates.
    """

    total_tokens: int
    tokens_trimmed: int
    trimmed_count: int
    segments_offered: int
    segments_kept: int
    tokens_budget: int
    estimated_cost_usd: Optional[float]
    trimming_strategies: List[str]
    compile_result: CompileResult


def _build_trimming_strategies(
    *,
    total_tokens: int,
    tokens_trimmed: int,
    trimmed_count: int,
    tokens_budget: int,
    target_token_budget: Optional[int],
) -> List[str]:
    """Deterministic suggestions; structural checks only (no content safety)."""
    out: List[str] = []
    if tokens_trimmed > 0 and trimmed_count > 0:
        out.append(
            f"Firewall dropped {trimmed_count} middle segment(s) "
            f"({tokens_trimmed} token(s)) to stay within budget {tokens_budget}."
        )
        out.append(
            "Raise ContextCompiler.max_tokens (or pass a higher policy cap) to keep more context."
        )
        out.append(
            "Enable compression and/or deduplication on MembraneProfile to shrink the "
            "sacrifice zone before trimming."
        )
    elif tokens_trimmed > 0:
        out.append(
            f"Middle zone was compressed or trimmed by {tokens_trimmed} token(s); "
            "review preflight/compression settings."
        )
    if target_token_budget is not None and total_tokens > target_token_budget:
        over = total_tokens - target_token_budget
        out.append(
            f"Compiled prompt is {over} token(s) above target_token_budget={target_token_budget}; "
            "reduce memories or max_tokens to meet the target."
        )
    if not out and total_tokens > 0:
        out.append("No trimming occurred; context fits the current budget.")
    return out


def compile_quote_from_result(
    result: CompileResult,
    *,
    price_per_1k_input_tokens: Optional[float] = None,
    target_token_budget: Optional[int] = None,
    compiler_max_tokens: int = 0,
) -> CompileQuote:
    """
    Build a :class:`CompileQuote` from an existing :class:`CompileResult` (same numbers as
    :meth:`ContextCompiler.estimate`).
    """
    md = result.metadata
    total = int(md.get("token_count", 0))
    tokens_trimmed = int(md.get("tokens_trimmed", 0))
    raw_budget = md.get("tokens_budget")
    tokens_budget = int(raw_budget) if raw_budget is not None else int(compiler_max_tokens)
    segments_offered = int(md.get("segments_offered", 0))
    segments_kept = int(md.get("segments_kept", 0))
    trimmed_count = max(0, segments_offered - segments_kept)

    cost: Optional[float] = None
    if price_per_1k_input_tokens is not None:
        if price_per_1k_input_tokens < 0:
            raise ValueError("price_per_1k_input_tokens must be non-negative")
        cost = (total / 1000.0) * float(price_per_1k_input_tokens)

    strategies = _build_trimming_strategies(
        total_tokens=total,
        tokens_trimmed=tokens_trimmed,
        trimmed_count=trimmed_count,
        tokens_budget=tokens_budget,
        target_token_budget=target_token_budget,
    )

    return CompileQuote(
        total_tokens=total,
        tokens_trimmed=tokens_trimmed,
        trimmed_count=trimmed_count,
        segments_offered=segments_offered,
        segments_kept=segments_kept,
        tokens_budget=tokens_budget,
        estimated_cost_usd=cost,
        trimming_strategies=strategies,
        compile_result=result,
    )


def _last_turn_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("stage3_synthesis") or m.get("content") or m.get("stage3")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, dict) and content.get("text"):
                return str(content["text"])
    return None


def _last_user_messages(messages: List[Dict[str, Any]], cap: int = 3) -> List[str]:
    user_contents: List[str] = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, str):
                user_contents.append(content)
            elif isinstance(content, dict) and "text" in content:
                user_contents.append(str(content["text"]))
    return user_contents[-cap:] if user_contents else []


_INTERNAL_MESSAGE_KEYS = ("stage3", "stage3_synthesis")


def _normalize_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Provider-ready history: shallow copies with every provider key kept as given.

    ``tool_calls``, ``tool_call_id``, ``name`` and content-part lists pass through
    untouched. Only internal keys (``stage3``, ``stage3_synthesis``) are dropped, with
    ``stage3_synthesis`` used as ``content`` when content is missing. Deterministic, so
    the same history always serializes to the same bytes.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        msg = {k: v for k, v in m.items() if k not in _INTERNAL_MESSAGE_KEYS}
        msg.setdefault("role", "user")
        content = msg.get("content")
        if content is None or content == "":
            synth = m.get("stage3_synthesis")
            if isinstance(synth, str) and synth:
                msg["content"] = synth
        elif isinstance(content, dict) and "text" in content:
            msg["content"] = str(content["text"])
        out.append(msg)
    return out


def _message_text(message: Dict[str, Any]) -> str:
    """Text used for token counting of a history message."""
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif content is None:
        text = ""
    else:
        text = json.dumps(content, sort_keys=True, default=str)
    extras = {k: v for k, v in message.items() if k not in ("role", "content")}
    if extras:
        text += json.dumps(extras, sort_keys=True, default=str)
    return text


def _flatten_messages_for_display(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content") or m.get("stage3_synthesis") or ""
        if isinstance(content, dict):
            content = content.get("text", "") or str(content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_retrieval_summary(
    n_episodic: int,
    n_semantic: int,
    n_procedural: int,
    user_msg: str,
) -> str:
    total = n_episodic + n_semantic + n_procedural
    if total == 0:
        return "No long-term memories were attached for this turn."
    topic = (user_msg or "").strip().replace("\n", " ")[:50]
    if topic:
        topic = topic.rstrip()
        if len((user_msg or "").strip()) > 50:
            topic = topic + "…"
    else:
        topic = "your query"
    parts: List[str] = []
    if n_episodic:
        parts.append(f"{n_episodic} episode{'s' if n_episodic != 1 else ''}")
    if n_semantic:
        parts.append(f"{n_semantic} semantic fact{'s' if n_semantic != 1 else ''}")
    if n_procedural:
        parts.append(f"{n_procedural} procedure{'s' if n_procedural != 1 else ''}")
    return f"Remembered {', '.join(parts)} regarding {topic}."


def _tier_for_object(obj: Any) -> int:
    p = getattr(obj, "priority", None)
    if p is not None:
        try:
            return int(p)
        except (TypeError, ValueError):
            pass
    kind = infer_memory_kind(obj)
    if kind == "procedural":
        return TIER_PROCEDURAL
    if kind == "semantic":
        return TIER_SEMANTIC
    return TIER_EPISODIC


def _stale_survivor_memory_ids(
    retrieved: List[Tuple[Any, int]],
    ref_map_kept: Dict[int, str],
) -> List[str]:
    """
    After the firewall, ``ref_map_kept`` maps local [Mem:N] indices to global memory ids.
    Return ids for kept refs whose source objects were marked ``_membrane_is_stale`` by
    :class:`~membrane.freshness.manager.FreshnessManager`.
    """
    id_to_obj: Dict[str, Any] = {}
    for i, (obj, _) in enumerate(retrieved):
        mid = str(getattr(obj, "id", f"seg_{i}"))
        id_to_obj[mid] = obj
    survivors: List[str] = []
    for mid in sorted(set(ref_map_kept.values())):
        obj = id_to_obj.get(mid)
        if obj is not None and bool(getattr(obj, "_membrane_is_stale", False)):
            survivors.append(mid)
    return survivors


LOD_FOCUS_EPISODIC = 2
LOD_FOCUS_SEMANTIC = 2
LOD_FOCUS_PROCEDURAL = 1


def _lod_focus_merge(
    main_retrieved: List[Tuple[Any, int]],
    focus_retrieved: List[Tuple[Any, int]],
) -> List[Tuple[Any, int]]:
    caps = {
        TIER_PROCEDURAL: LOD_FOCUS_PROCEDURAL,
        TIER_SEMANTIC: LOD_FOCUS_SEMANTIC,
        TIER_EPISODIC: LOD_FOCUS_EPISODIC,
    }
    counts = {TIER_PROCEDURAL: 0, TIER_SEMANTIC: 0, TIER_EPISODIC: 0}
    focus_list: List[Tuple[Any, int]] = []
    seen_ids: Set[Any] = set()
    for obj, tier in focus_retrieved:
        if tier not in caps:
            continue
        if counts[tier] >= caps[tier]:
            continue
        pid = getattr(obj, "id", None)
        if pid in seen_ids:
            continue
        focus_list.append((obj, tier))
        seen_ids.add(pid)
        counts[tier] += 1
    focus_ids = {getattr(o, "id", None) for o, _ in focus_list}
    main_deduped = [(o, t) for o, t in main_retrieved if getattr(o, "id", None) not in focus_ids]
    return focus_list + main_deduped


_KIND_TIER = {
    "procedural": TIER_PROCEDURAL,
    "semantic": TIER_SEMANTIC,
    "episodic": TIER_EPISODIC,
}


def _policy_disabled_features(policy_kw: Dict[str, Any]) -> List[str]:
    """Features disabled by policies, minus any a policy re-enabled."""
    enabled = set(policy_kw.get("policy_enable") or [])
    return [f for f in (policy_kw.get("policy_disable") or []) if f not in enabled]


def _policy_tier_settings(
    policy_kw: Dict[str, Any],
) -> Tuple[Optional[Dict[int, float]], Optional[Set[int]]]:
    """``set_budget`` kind shares and ``pin_memory_kind`` kinds, as firewall tiers."""
    shares = {
        _KIND_TIER[k]: float(v)
        for k, v in (policy_kw.get("policy_budget_override") or {}).items()
        if k in _KIND_TIER
    }
    pinned = {_KIND_TIER[k] for k in (policy_kw.get("policy_pin_kinds") or []) if k in _KIND_TIER}
    return shares or None, pinned or None


@dataclass
class ContextCompiler:
    """
    Orchestrates the compile pipeline: policies and memory shaping, layout (core →
    middle → end), optional compression, firewall, then provider messages (with
    glaciation when prompt caching applies).

    After a :class:`CompileResult` exists, optional assertions, lineage finalize,
    recorder, metrics callbacks, and trace exporters run — in that order — so
    structural validation and telemetry never observe a half-built result.

    ``max_tokens`` is the hard context budget (defaults from provider registry when
    ``model_id`` is set via ``resolve_max_tokens_from_model``).

    ``memory_placement`` controls where retrieved memories go. ``"system"`` (default)
    emits one system message: core, memories, last turn, user message. Memories change
    every turn, so the provider prompt cache cannot hit past the core. ``"end"`` emits
    the system message (core only) and ``conversation_messages`` as given, then one user
    message with memories, the current user message and dynamic directives. System +
    history then form a byte-stable prefix that prompt caching can reuse across turns.
    History is sent in full and never trimmed; if it overruns ``max_tokens`` the result
    carries ``metadata["budget_exceeded"] = True`` and the caller must shorten it.
    """

    max_tokens: int
    model_id: Optional[str] = None
    core_content: str = ""
    include_interrupt_instruction: bool = True
    preflight_analyst_model: Optional[str] = None
    metrics_callback: Optional[MetricsCallback] = None
    memory_placement: str = MEMORY_PLACEMENT_SYSTEM

    def __post_init__(self) -> None:
        if self.memory_placement not in MEMORY_PLACEMENTS:
            raise ValueError(
                f"memory_placement must be one of {MEMORY_PLACEMENTS}, "
                f"got {self.memory_placement!r}"
            )

    def _place_and_trim(
        self,
        *,
        layout: ContextLayoutEngine,
        core_text: str,
        middle_segments: List[Tuple[str, int]],
        ref_map: Dict[int, str],
        segment_weights: Optional[List[float]],
        conversation_messages: List[Dict[str, Any]],
        last_assistant_turn: Optional[str],
        user_message: str,
        max_tok: int,
        use_split: bool,
        count_fn: Callable[[str], int],
        tier_shares: Optional[Dict[int, float]] = None,
        pinned_tiers: Optional[Set[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[int, str], int, Dict[str, Any]]:
        """Build end block, trim the middle under ``max_tok``, and format provider messages.

        ``memory_placement="system"``: one system message (core, memories, end, directives).
        ``memory_placement="end"``: system (core only) + history as given + one user message
        holding memories, the current user message and directives, so the system message and
        history form a byte-stable prefix across turns.
        """
        caching = use_split and bool(self.model_id and has_prompt_caching(self.model_id))
        firewall = PreFlightContextFirewall(max_tokens=max_tok, count_fn=count_fn)
        directives_block = layout.build_directives_block()

        if self.memory_placement == MEMORY_PLACEMENT_SYSTEM:
            end_text = layout.build_end(
                last_turn_content=last_assistant_turn
                or _last_turn_from_messages(conversation_messages),
                user_message=user_message,
            )
            # Directives go last but are reserved with the end block so they fit the budget.
            final_text, ref_map_kept, _, fw_stats = firewall.run(
                core_text=core_text,
                end_text=end_text + directives_block,
                middle_segments=middle_segments,
                ref_map=ref_map,
                segment_weights=segment_weights,
                tier_shares=tier_shares,
                pinned_tiers=pinned_tiers,
            )
            fw_stats.pop("kept_segments", None)
            total_tokens = count_fn(final_text)
            messages, cache_meta = _build_output_messages(
                self.model_id, final_text, caching, prefix_text=core_text
            )
            if total_tokens > max_tok:
                cache_meta["budget_exceeded"] = True
            return messages, cache_meta, ref_map_kept, total_tokens, fw_stats

        history = _normalize_history(conversation_messages)
        end_text = layout.build_end(
            # The last assistant turn is already in the history messages.
            last_turn_content=None if history else last_assistant_turn,
            user_message=user_message,
        )
        history_text = "\n\n".join(_message_text(m) for m in history)
        # History is part of the cached prefix: it is reserved, never trimmed.
        reserved_end = history_text + "\n\n" + end_text if history_text else end_text
        _, ref_map_kept, _, fw_stats = firewall.run(
            core_text=core_text,
            end_text=reserved_end,
            middle_segments=middle_segments,
            ref_map=ref_map,
            segment_weights=segment_weights,
            tier_shares=tier_shares,
            pinned_tiers=pinned_tiers,
        )
        kept = list(fw_stats.pop("kept_segments", []))
        tail = ("\n".join(kept) + "\n\n" + end_text) if kept else end_text
        tail += directives_block
        # Some providers reject an empty system message.
        system = [{"role": "system", "content": core_text}] if core_text.strip() else []
        messages = [*system, *history]
        messages.append({"role": "user", "content": tail})
        prefix_tokens = count_fn(core_text) + sum(count_fn(_message_text(m)) for m in history)
        tail_tokens = count_fn(tail)
        cache_meta: Dict[str, Any] = {
            "memory_placement": MEMORY_PLACEMENT_END,
            "stable_prefix_messages": len(system) + len(history),
        }
        if prefix_tokens + tail_tokens > max_tok:
            # History is never trimmed (that would break the cache); report the overrun.
            logger.warning(
                "memory_placement=end: system+history+end (%s tokens) exceeds max_tokens (%s); "
                "trim conversation_messages before compiling",
                prefix_tokens + tail_tokens,
                max_tok,
            )
            cache_meta["budget_exceeded"] = True
        if caching and prefix_tokens >= MIN_CACHEABLE_PREFIX_TOKENS:
            cache_meta.update(
                prompt_cache_used=True,
                cacheable_prefix_tokens=prefix_tokens,
                dynamic_tail_tokens=tail_tokens,
            )
        return messages, cache_meta, ref_map_kept, prefix_tokens + tail_tokens, fw_stats

    def _policy_max_tokens(self, policy_kw: Dict[str, Any]) -> int:
        """Budget after policies: ``set_max_tokens`` / ``override.max_tokens`` replace it,
        bounded by the model's context cap when the model is known."""
        policy_max = policy_kw.get("policy_max_tokens")
        if policy_max is None and policy_kw.get("max_tokens") != self.max_tokens:
            policy_max = policy_kw.get("max_tokens")
        if policy_max is None:
            return self.effective_max_tokens()
        cap = lookup_context_cap(self.model_id) if self.model_id else None
        return min(int(policy_max), cap) if cap else int(policy_max)

    def effective_max_tokens(self) -> int:
        if self.model_id:
            # Unknown models never shrink the caller's budget.
            cap = lookup_context_cap(self.model_id)
            if cap and cap > 0:
                return min(self.max_tokens, cap)
        return self.max_tokens

    def estimate(
        self,
        *,
        memories: Sequence[Any],
        user_message: str,
        price_per_1k_input_tokens: Optional[float] = None,
        target_token_budget: Optional[int] = None,
        **kwargs: Any,
    ) -> CompileQuote:
        """
        Same work as :meth:`compile` — returns a :class:`CompileQuote` with token/cost summaries
        and the full :attr:`CompileQuote.compile_result` so you do not need to compile twice.

        ``estimated_cost_usd`` is ``(total_tokens / 1000) * price_per_1k_input_tokens`` when you
        pass a non-negative rate; otherwise ``None``. This is an **input** (prompt) estimate only.
        """
        result = self.compile(
            memories=memories,
            user_message=user_message,
            **kwargs,
        )
        return compile_quote_from_result(
            result,
            price_per_1k_input_tokens=price_per_1k_input_tokens,
            target_token_budget=target_token_budget,
            compiler_max_tokens=self.max_tokens,
        )

    async def estimate_async(
        self,
        *,
        memories: Sequence[Any],
        user_message: str,
        price_per_1k_input_tokens: Optional[float] = None,
        target_token_budget: Optional[int] = None,
        **kwargs: Any,
    ) -> CompileQuote:
        """Async counterpart to :meth:`estimate`; delegates to :meth:`compile_async`."""
        result = await self.compile_async(
            memories=memories,
            user_message=user_message,
            **kwargs,
        )
        return compile_quote_from_result(
            result,
            price_per_1k_input_tokens=price_per_1k_input_tokens,
            target_token_budget=target_token_budget,
            compiler_max_tokens=self.max_tokens,
        )

    def compile(
        self,
        *,
        memories: Sequence[Any],
        user_message: str,
        last_assistant_turn: Optional[str] = None,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        agent_persona_prompt: Optional[str] = None,
        dynamic_directives: Optional[Sequence[str]] = None,
        middle_extras: Optional[Sequence[Tuple[str, int]]] = None,
        compress_sync: Optional[Callable[[str, int, str], str]] = None,
        tier_by_memory_id: Optional[Dict[int, int]] = None,
        use_prompt_caching_split: bool = True,
        profile: Any = None,
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
        trace_exporters: Optional[List[Any]] = None,
        enable_lineage: Optional[bool] = None,
        enable_profiler: Optional[bool] = None,
        metrics_collector: Any = None,
        lineage_tracker: Any = None,
        ranking_weights: Any = None,
        include_ranking_diagnostics: Optional[bool] = None,
    ) -> CompileResult:
        """Synchronous compile: optional sync ``compress_sync`` for sacrifice zone; optional modules via ``profile``.

        Stages follow ``membrane/future.md`` (Compile Pipeline Order): policies →
        classify → freshness → dedup → layout core → reranker → lineage segment
        registration → build middle → compression / preflight compress → build end →
        firewall → directives → format output (cacheable prefix = core) →
        :class:`CompileResult` → assertions → lineage finalize → recorder → metrics →
        trace exporters.
        """
        prof = resolve_profile(
            profile,
            reranker=reranker,
            deduplicator=deduplicator,
            compression=compression,
            policies=policies,
            metrics=metrics,
            embedding_cache=embedding_cache,
            classifier=classifier,
            freshness=freshness,
            assertions=assertions,
            recorder=recorder,
            tokenizer_registry=tokenizer_registry,
            trace_exporters=trace_exporters,
            enable_lineage=enable_lineage,
            enable_profiler=enable_profiler,
            metrics_collector=metrics_collector,
            lineage_tracker=lineage_tracker,
        )

        memories = normalize_compile_memories(memories)

        latency_prof: Optional[Any] = None
        profiler: Optional[LatencyProfiler] = None
        if prof.enable_profiler:
            profiler = LatencyProfiler()
            profiler.start_total()

        def _count_ctx(text: str) -> int:
            if prof.tokenizer_registry is not None and self.model_id:
                try:
                    return int(
                        prof.tokenizer_registry.count(self.model_id, text)
                    )
                except Exception:
                    pass
            return count_tokens(text)

        policy_kw: Dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "model_id": self.model_id,
        }
        est_tok = estimate_memory_tokens(memories)
        if prof.policies is not None:
            with _stage(profiler, "policies"):
                policy_kw = prof.policies.apply(
                    query=user_message,
                    memories=memories,
                    compiler_kwargs=policy_kw,
                    estimated_tokens=est_tok,
                    model_id=self.model_id,
                )
        policy_disable = _policy_disabled_features(policy_kw)
        tier_shares, pinned_tiers = _policy_tier_settings(policy_kw)
        max_tok = self._policy_max_tokens(policy_kw)

        mems: List[Any] = list(memories)
        dedup_res = None
        lineage = prof.lineage_tracker
        if prof.enable_lineage and lineage is None:
            raise FeatureNotAvailable("enable_lineage (pass your own lineage_tracker instead)")

        if prof.classifier is not None:
            with _stage(profiler, "classify"):
                mems = apply_classifier(prof.classifier, mems)

        if prof.freshness is not None:
            with _stage(profiler, "freshness"):
                mems = list(prof.freshness.apply_weights(mems))

        if prof.deduplicator is not None:
            with _stage(profiler, "dedup"):
                dedup_res = prof.deduplicator.deduplicate(
                    mems, text_fn=lambda x: str(getattr(x, "content", x))
                )
                mems = list(dedup_res.kept)

        t0 = time.perf_counter()
        layout = ContextLayoutEngine(core_content=self.core_content)
        if self.include_interrupt_instruction:
            layout.core_content = layout.core_content + "\n\n" + INTERRUPT_INSTRUCTION
        if agent_persona_prompt:
            layout.core_content = layout.core_content + "\n\n" + agent_persona_prompt
        if dynamic_directives:
            layout.set_system_directives("\n".join(dynamic_directives))
        core_text = layout.build_core()

        if tier_by_memory_id:
            retrieved = [
                (m, tier_by_memory_id.get(id(m), _tier_for_object(m))) for m in mems
            ]
        else:
            retrieved = [(m, _tier_for_object(m)) for m in mems]
        segment_weights: Optional[List[float]] = None
        if (
            prof.reranker is not None
            and not policy_feature_disabled(policy_disable, "reranker")
        ):
            with _stage(profiler, "reranker"):
                ranked = prof.reranker.rerank(
                    user_message, [m for m, _ in retrieved]
                )
                dens = {id(m): float(d) for m, d in ranked}
                tier_of = {id(m): t for m, t in retrieved}
                retrieved = [
                    (m, tier_of.get(id(m), _tier_for_object(m))) for m, _ in ranked
                ]
                segment_weights = [
                    dens.get(id(obj), 0.0) for obj, _ in retrieved
                ]

        msgs = conversation_messages or []
        attention_shift = False
        if msgs:
            attention_shift = compute_attention_shift(
                _last_user_messages(msgs), user_message
            )

        if lineage is not None:
            for i, (obj, _) in enumerate(retrieved):
                sid = str(getattr(obj, "id", f"seg_{i}"))
                lineage.register_segment(
                    sid,
                    str(getattr(obj, "content", obj)),
                    source_origin="compile",
                    source_metadata={"tier": _tier_for_object(obj)},
                )

        middle_segments, ref_map, ref_map_type = layout.build_middle(
            retrieved, start_index=1
        )
        if middle_extras:
            middle_segments = list(middle_extras) + middle_segments
            if segment_weights is not None:
                # Caller-supplied extras rank ahead of reranked memories.
                segment_weights = [float("inf")] * len(middle_extras) + segment_weights

        n_episodic = sum(1 for _, t in retrieved if t == TIER_EPISODIC)
        n_semantic = sum(1 for _, t in retrieved if t == TIER_SEMANTIC)
        n_procedural = sum(1 for _, t in retrieved if t == TIER_PROCEDURAL)
        retrieval_summary = _build_retrieval_summary(
            n_episodic, n_semantic, n_procedural, user_message
        )

        threshold = os_config.get_preflight_token_threshold()
        target_tokens = os_config.get_preflight_target_tokens()
        raw_middle_text = "\n\n".join(seg[0] for seg in middle_segments)
        middle_tokens = _count_ctx(raw_middle_text)
        compression_strategy_used: Optional[str] = None
        compressed_middle: Optional[str] = None
        compression_details: Dict[str, Any] = {}
        compression_token_delta: int = 0

        if (
            prof.compression is not None
            and not policy_feature_disabled(policy_disable, "compression")
            and raw_middle_text.strip()
        ):
            with _stage(profiler, "compression"):
                if self.model_id:
                    set_model_id = getattr(prof.compression, "set_model_id", None)
                    if callable(set_model_id):
                        set_model_id(self.model_id)
                ct = policy_kw.get("policy_compression_target")
                tgt = (
                    int(middle_tokens * float(ct))
                    if ct is not None
                    else target_tokens
                )
                cres = prof.compression.compress(
                    raw_middle_text,
                    target_tokens=max(1, tgt),
                    query_context=user_message,
                )
                compression_strategy_used = cres.strategy_used
                if isinstance(getattr(cres, "details", {}), dict):
                    compression_details = dict(cres.details)
                compression_token_delta = int(cres.original_tokens - cres.compressed_tokens)
                if cres.compressed and cres.compressed.strip():
                    middle_segments = [(cres.compressed.strip(), TIER_SEMANTIC)]
                    segment_weights = None
                    compressed_middle = cres.compressed.strip()
                    raw_middle_text = cres.compressed.strip()
                    middle_tokens = _count_ctx(raw_middle_text)
                _record_jta_lineage_from_compression(
                    lineage,
                    compression_strategy_used=compression_strategy_used,
                    compression_details=compression_details,
                )

        if middle_tokens > threshold and raw_middle_text.strip() and compress_sync:
            try:
                compressed = compress_sync(raw_middle_text, target_tokens, user_message)
            except Exception as e:
                logger.warning("compress_sync failed: %s", e, exc_info=True)
                compressed = None
            if compressed and compressed.strip():
                middle_segments = [(compressed.strip(), TIER_SEMANTIC)]
                segment_weights = None
                compressed_middle = compressed.strip()

        ranking_diag = _build_ranking_diagnostics_for_compile(
            prof,
            list(retrieved),
            segment_weights,
            list(middle_segments),
            ranking_weights=ranking_weights,
            include_ranking_diagnostics=include_ranking_diagnostics,
        )

        segments_offered_pre = len(middle_segments)
        with _stage(profiler, "firewall"):
            messages, cache_meta, ref_map_kept, total_tokens, fw_stats = (
                self._place_and_trim(
                    layout=layout,
                    core_text=core_text,
                    middle_segments=middle_segments,
                    ref_map=ref_map,
                    segment_weights=segment_weights,
                    conversation_messages=msgs,
                    last_assistant_turn=last_assistant_turn,
                    user_message=user_message,
                    max_tok=max_tok,
                    use_split=use_prompt_caching_split,
                    count_fn=_count_ctx,
                    tier_shares=tier_shares,
                    pinned_tiers=pinned_tiers,
                )
            )
        if compressed_middle is not None and fw_stats.get("segments_kept"):
            # Compressed text keeps [Mem:N] references; resolve every one it still carries.
            ref_map_kept = _ref_map_from_text(compressed_middle, ref_map)
        stale_survivors = _stale_survivor_memory_ids(list(retrieved), ref_map_kept)

        ref_map_type_kept: Dict[int, str] = {}
        if ref_map_type:
            for k in ref_map_kept:
                if k in ref_map_type:
                    ref_map_type_kept[k] = ref_map_type[k]

        wall_ms = (time.perf_counter() - t0) * 1000.0
        if profiler is not None:
            latency_prof = profiler.finish_total()

        cache_hit = bool(
            cache_meta.get("prompt_cache_used")
            and use_prompt_caching_split
            and bool(self.model_id and has_prompt_caching(self.model_id))
        )

        memory_kinds_present = list(
            {ref_map_type_kept[k] for k in ref_map_kept if k in ref_map_type_kept}
        )

        result = CompileResult(
            messages=messages,
            ref_map=ref_map_kept,
            ref_map_type=ref_map_type_kept or None,
            system_alerts=[],
            metadata={
                "security_flag": SecurityFlag.AUTHENTICATED.value,
                "token_count": total_tokens,
                "attention_shift": attention_shift,
                "retrieval_summary": retrieval_summary,
                "tokens_trimmed": int(fw_stats.get("tokens_trimmed", 0)),
                "tokens_budget": max_tok,
                "segments_offered": int(fw_stats.get("segments_offered", segments_offered_pre)),
                "segments_kept": int(fw_stats.get("segments_kept", 0)),
                "memory_kinds_present": memory_kinds_present,
                "memories_truncated": layout.last_truncated_count,
                "stale_survivors": stale_survivors,
                "stale_survivor_count": len(stale_survivors),
                "compression_strategy_used": compression_strategy_used,
                "compression_token_delta": compression_token_delta,
                "compression_details": compression_details,
                **cache_meta,
            },
            dedup_report=dedup_res,
            latency_profile=latency_prof,
            debug_trace=_merge_compile_debug_trace(
                policy_kw,
                segment_weights is not None,
                ranking_diag,
            ),
        )
        if compression_strategy_used is not None:
            result.debug_trace = dict(result.debug_trace or {})
            result.debug_trace["compression"] = {
                "strategy_used": compression_strategy_used,
                "token_delta": compression_token_delta,
                "details": compression_details,
            }

        if prof.assertions is not None:
            with _stage(profiler, "assertions"):
                result.assertion_report = prof.assertions.run(result)

        if lineage is not None:
            included = set(ref_map_kept.values())
            result.lineage_report = lineage.finalize(
                included_ids=included,
                positions={},
            )

        if prof.recorder is not None:
            try:
                prof.recorder.record(
                    mems,
                    user_message,
                    result,
                    {
                        "max_tokens": self.max_tokens,
                        "model_id": self.model_id,
                    },
                )
            except Exception as exc:
                logger.debug("recorder.record failed: %s", exc)

        metrics_extra: Dict[str, Any] = {}
        if result.assertion_report is not None:
            metrics_extra["assertions_passed"] = getattr(
                result.assertion_report, "passed", None
            )
        _emit_compile_metrics(
            self.metrics_callback,
            build_compile_metrics(
                wall_time_ms=wall_ms,
                token_count=total_tokens,
                tokens_trimmed=int(fw_stats.get("tokens_trimmed", 0)),
                tokens_budget=max_tok,
                segments_offered=int(fw_stats.get("segments_offered", segments_offered_pre)),
                segments_kept=int(fw_stats.get("segments_kept", 0)),
                ref_map_size=len(ref_map_kept),
                cache_hit=cache_hit,
                is_anonymous=False,
                model_id=self.model_id,
                user_id=None,
                conversation_id=None,
                extra=metrics_extra,
            ),
        )

        collector = prof.metrics_collector or prof.metrics
        if collector is not None:
            try:
                collector.histogram("compile_latency_ms", wall_ms)
                collector.counter("compile_tokens_total", float(total_tokens))
                collector.counter(
                    "compile_tokens_trimmed",
                    float(fw_stats.get("tokens_trimmed", 0)),
                )
            except Exception:
                pass

        _run_trace_exporters(prof, result)

        return result

    async def compile_async(
        self,
        *,
        memories: Sequence[Any],
        user_message: str,
        last_assistant_turn: Optional[str] = None,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        agent_persona_prompt: Optional[str] = None,
        dynamic_directives: Optional[Sequence[str]] = None,
        middle_extras: Optional[Sequence[Tuple[str, int]]] = None,
        compress_async: Optional[Callable[[str, int, str], Awaitable[str]]] = None,
        compress_sync: Optional[Callable[[str, int, str], str]] = None,
        redis_client: Optional[Any] = None,
        conversation_id: Optional[str] = None,
        tier_by_memory_id: Optional[Dict[int, int]] = None,
        use_prompt_caching_split: bool = True,
        profile: Any = None,
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
        trace_exporters: Optional[List[Any]] = None,
        enable_lineage: Optional[bool] = None,
        enable_profiler: Optional[bool] = None,
        metrics_collector: Any = None,
        lineage_tracker: Any = None,
        ranking_weights: Any = None,
        include_ranking_diagnostics: Optional[bool] = None,
    ) -> CompileResult:
        """
        Async compile with optional ``compress_async`` and/or Redis pre-flight cache.

        If neither ``compress_async`` nor ``redis_client`` is set, delegates to
        :meth:`compile` (sync compress only).

        Pipeline order matches :meth:`compile` (see ``membrane/future.md``).
        """
        if compress_async is None and redis_client is None:
            return self.compile(
                memories=memories,
                user_message=user_message,
                last_assistant_turn=last_assistant_turn,
                conversation_messages=conversation_messages,
                agent_persona_prompt=agent_persona_prompt,
                dynamic_directives=dynamic_directives,
                middle_extras=middle_extras,
                compress_sync=compress_sync,
                tier_by_memory_id=tier_by_memory_id,
                use_prompt_caching_split=use_prompt_caching_split,
                profile=profile,
                reranker=reranker,
                deduplicator=deduplicator,
                compression=compression,
                policies=policies,
                metrics=metrics,
                embedding_cache=embedding_cache,
                classifier=classifier,
                freshness=freshness,
                assertions=assertions,
                recorder=recorder,
                tokenizer_registry=tokenizer_registry,
                trace_exporters=trace_exporters,
                enable_lineage=enable_lineage,
                enable_profiler=enable_profiler,
                metrics_collector=metrics_collector,
                lineage_tracker=lineage_tracker,
                ranking_weights=ranking_weights,
                include_ranking_diagnostics=include_ranking_diagnostics,
            )

        prof = resolve_profile(
            profile,
            reranker=reranker,
            deduplicator=deduplicator,
            compression=compression,
            policies=policies,
            metrics=metrics,
            embedding_cache=embedding_cache,
            classifier=classifier,
            freshness=freshness,
            assertions=assertions,
            recorder=recorder,
            tokenizer_registry=tokenizer_registry,
            trace_exporters=trace_exporters,
            enable_lineage=enable_lineage,
            enable_profiler=enable_profiler,
            metrics_collector=metrics_collector,
            lineage_tracker=lineage_tracker,
        )

        memories = normalize_compile_memories(memories)

        latency_prof: Optional[Any] = None
        profiler: Optional[LatencyProfiler] = None
        if prof.enable_profiler:
            profiler = LatencyProfiler()
            profiler.start_total()

        def _count_ctx(text: str) -> int:
            if prof.tokenizer_registry is not None and self.model_id:
                try:
                    return int(
                        prof.tokenizer_registry.count(self.model_id, text)
                    )
                except Exception:
                    pass
            return count_tokens(text)

        policy_kw: Dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "model_id": self.model_id,
        }
        est_tok = estimate_memory_tokens(memories)
        if prof.policies is not None:
            with _stage(profiler, "policies"):
                policy_kw = prof.policies.apply(
                    query=user_message,
                    memories=memories,
                    compiler_kwargs=policy_kw,
                    estimated_tokens=est_tok,
                    model_id=self.model_id,
                )
        policy_disable = _policy_disabled_features(policy_kw)
        tier_shares, pinned_tiers = _policy_tier_settings(policy_kw)
        max_tok = self._policy_max_tokens(policy_kw)

        mems: List[Any] = list(memories)
        dedup_res = None
        lineage = prof.lineage_tracker
        if prof.enable_lineage and lineage is None:
            raise FeatureNotAvailable("enable_lineage (pass your own lineage_tracker instead)")

        if prof.classifier is not None:
            with _stage(profiler, "classify"):
                mems = apply_classifier(prof.classifier, mems)

        if prof.freshness is not None:
            with _stage(profiler, "freshness"):
                mems = list(prof.freshness.apply_weights(mems))

        if prof.deduplicator is not None:
            with _stage(profiler, "dedup"):
                dedup_res = prof.deduplicator.deduplicate(
                    mems, text_fn=lambda x: str(getattr(x, "content", x))
                )
                mems = list(dedup_res.kept)

        t0 = time.perf_counter()
        layout = ContextLayoutEngine(core_content=self.core_content)
        if self.include_interrupt_instruction:
            layout.core_content = layout.core_content + "\n\n" + INTERRUPT_INSTRUCTION
        if agent_persona_prompt:
            layout.core_content = layout.core_content + "\n\n" + agent_persona_prompt
        if dynamic_directives:
            layout.set_system_directives("\n".join(dynamic_directives))
        core_text = layout.build_core()

        if tier_by_memory_id:
            retrieved = [
                (m, tier_by_memory_id.get(id(m), _tier_for_object(m))) for m in mems
            ]
        else:
            retrieved = [(m, _tier_for_object(m)) for m in mems]

        segment_weights: Optional[List[float]] = None
        if (
            prof.reranker is not None
            and not policy_feature_disabled(policy_disable, "reranker")
        ):
            with _stage(profiler, "reranker"):
                ranked = prof.reranker.rerank(
                    user_message, [m for m, _ in retrieved]
                )
                dens = {id(m): float(d) for m, d in ranked}
                tier_of = {id(m): t for m, t in retrieved}
                retrieved = [
                    (m, tier_of.get(id(m), _tier_for_object(m))) for m, _ in ranked
                ]
                segment_weights = [
                    dens.get(id(obj), 0.0) for obj, _ in retrieved
                ]

        msgs = conversation_messages or []
        attention_shift = False
        if msgs:
            attention_shift = compute_attention_shift(
                _last_user_messages(msgs), user_message
            )

        if lineage is not None:
            for i, (obj, _) in enumerate(retrieved):
                sid = str(getattr(obj, "id", f"seg_{i}"))
                lineage.register_segment(
                    sid,
                    str(getattr(obj, "content", obj)),
                    source_origin="compile_async",
                    source_metadata={"tier": _tier_for_object(obj)},
                )

        middle_segments, ref_map, ref_map_type = layout.build_middle(
            retrieved, start_index=1
        )
        if middle_extras:
            middle_segments = list(middle_extras) + middle_segments
            if segment_weights is not None:
                # Caller-supplied extras rank ahead of reranked memories.
                segment_weights = [float("inf")] * len(middle_extras) + segment_weights

        n_episodic = sum(1 for _, t in retrieved if t == TIER_EPISODIC)
        n_semantic = sum(1 for _, t in retrieved if t == TIER_SEMANTIC)
        n_procedural = sum(1 for _, t in retrieved if t == TIER_PROCEDURAL)
        retrieval_summary = _build_retrieval_summary(
            n_episodic, n_semantic, n_procedural, user_message
        )

        threshold = os_config.get_preflight_token_threshold()
        target_tokens = os_config.get_preflight_target_tokens()
        raw_middle_text = "\n\n".join(seg[0] for seg in middle_segments)
        middle_tokens = _count_ctx(raw_middle_text)
        compression_strategy_used: Optional[str] = None
        compressed_middle: Optional[str] = None
        compression_details: Dict[str, Any] = {}
        compression_token_delta: int = 0

        if (
            prof.compression is not None
            and not policy_feature_disabled(policy_disable, "compression")
            and raw_middle_text.strip()
        ):
            with _stage(profiler, "compression"):
                if self.model_id:
                    set_model_id = getattr(prof.compression, "set_model_id", None)
                    if callable(set_model_id):
                        set_model_id(self.model_id)
                ct = policy_kw.get("policy_compression_target")
                tgt = (
                    int(middle_tokens * float(ct))
                    if ct is not None
                    else target_tokens
                )
                cres = prof.compression.compress(
                    raw_middle_text,
                    target_tokens=max(1, tgt),
                    query_context=user_message,
                )
                compression_strategy_used = cres.strategy_used
                if isinstance(getattr(cres, "details", {}), dict):
                    compression_details = dict(cres.details)
                compression_token_delta = int(cres.original_tokens - cres.compressed_tokens)
                if cres.compressed and cres.compressed.strip():
                    middle_segments = [(cres.compressed.strip(), TIER_SEMANTIC)]
                    segment_weights = None
                    compressed_middle = cres.compressed.strip()
                    raw_middle_text = cres.compressed.strip()
                    middle_tokens = _count_ctx(raw_middle_text)
                _record_jta_lineage_from_compression(
                    lineage,
                    compression_strategy_used=compression_strategy_used,
                    compression_details=compression_details,
                )

        if middle_tokens > threshold and raw_middle_text.strip():
            compressed: Optional[str] = None
            if redis_client and conversation_id:
                try:
                    compressed = await get_preflight_cache(
                        redis_client, conversation_id, user_message
                    )
                except Exception as e:
                    logger.debug("Pre-flight cache lookup failed: %s", e)
            if not compressed and compress_async is not None:
                try:
                    compressed = await compress_async(
                        raw_middle_text, target_tokens, user_message
                    )
                except Exception as e:
                    logger.warning("compress_async failed: %s", e, exc_info=True)
                    compressed = None
                if compressed and redis_client and conversation_id:
                    try:
                        await set_preflight_cache(
                            redis_client,
                            conversation_id,
                            user_message,
                            compressed,
                        )
                    except Exception as e:
                        logger.debug("Pre-flight cache set failed: %s", e)
            if not compressed and compress_sync is not None:
                try:
                    compressed = compress_sync(
                        raw_middle_text, target_tokens, user_message
                    )
                except Exception as e:
                    logger.warning("compress_sync failed: %s", e, exc_info=True)
                    compressed = None
            if compressed and compressed.strip():
                middle_segments = [(compressed.strip(), TIER_SEMANTIC)]
                segment_weights = None
                compressed_middle = compressed.strip()

        ranking_diag = _build_ranking_diagnostics_for_compile(
            prof,
            list(retrieved),
            segment_weights,
            list(middle_segments),
            ranking_weights=ranking_weights,
            include_ranking_diagnostics=include_ranking_diagnostics,
        )

        segments_offered_pre = len(middle_segments)
        with _stage(profiler, "firewall"):
            messages, cache_meta, ref_map_kept, total_tokens, fw_stats = (
                self._place_and_trim(
                    layout=layout,
                    core_text=core_text,
                    middle_segments=middle_segments,
                    ref_map=ref_map,
                    segment_weights=segment_weights,
                    conversation_messages=msgs,
                    last_assistant_turn=last_assistant_turn,
                    user_message=user_message,
                    max_tok=max_tok,
                    use_split=use_prompt_caching_split,
                    count_fn=_count_ctx,
                    tier_shares=tier_shares,
                    pinned_tiers=pinned_tiers,
                )
            )
        if compressed_middle is not None and fw_stats.get("segments_kept"):
            # Compressed text keeps [Mem:N] references; resolve every one it still carries.
            ref_map_kept = _ref_map_from_text(compressed_middle, ref_map)
        stale_survivors = _stale_survivor_memory_ids(list(retrieved), ref_map_kept)

        ref_map_type_kept: Dict[int, str] = {}
        if ref_map_type:
            for k in ref_map_kept:
                if k in ref_map_type:
                    ref_map_type_kept[k] = ref_map_type[k]

        wall_ms = (time.perf_counter() - t0) * 1000.0
        if profiler is not None:
            latency_prof = profiler.finish_total()

        cache_hit = bool(
            cache_meta.get("prompt_cache_used")
            and use_prompt_caching_split
            and bool(self.model_id and has_prompt_caching(self.model_id))
        )

        memory_kinds_present = list(
            {ref_map_type_kept[k] for k in ref_map_kept if k in ref_map_type_kept}
        )

        result = CompileResult(
            messages=messages,
            ref_map=ref_map_kept,
            ref_map_type=ref_map_type_kept or None,
            system_alerts=[],
            metadata={
                "security_flag": SecurityFlag.AUTHENTICATED.value,
                "token_count": total_tokens,
                "attention_shift": attention_shift,
                "retrieval_summary": retrieval_summary,
                "tokens_trimmed": int(fw_stats.get("tokens_trimmed", 0)),
                "tokens_budget": max_tok,
                "segments_offered": int(fw_stats.get("segments_offered", segments_offered_pre)),
                "segments_kept": int(fw_stats.get("segments_kept", 0)),
                "memory_kinds_present": memory_kinds_present,
                "memories_truncated": layout.last_truncated_count,
                "stale_survivors": stale_survivors,
                "stale_survivor_count": len(stale_survivors),
                "compression_strategy_used": compression_strategy_used,
                "compression_token_delta": compression_token_delta,
                "compression_details": compression_details,
                **cache_meta,
            },
            dedup_report=dedup_res,
            latency_profile=latency_prof,
            debug_trace=_merge_compile_debug_trace(
                policy_kw,
                segment_weights is not None,
                ranking_diag,
            ),
        )
        if compression_strategy_used is not None:
            result.debug_trace = dict(result.debug_trace or {})
            result.debug_trace["compression"] = {
                "strategy_used": compression_strategy_used,
                "token_delta": compression_token_delta,
                "details": compression_details,
            }

        if prof.assertions is not None:
            with _stage(profiler, "assertions"):
                result.assertion_report = prof.assertions.run(result)

        if lineage is not None:
            included = set(ref_map_kept.values())
            result.lineage_report = lineage.finalize(
                included_ids=included,
                positions={},
            )

        if prof.recorder is not None:
            try:
                prof.recorder.record(
                    mems,
                    user_message,
                    result,
                    {
                        "max_tokens": self.max_tokens,
                        "model_id": self.model_id,
                    },
                )
            except Exception as exc:
                logger.debug("recorder.record failed: %s", exc)

        metrics_extra: Dict[str, Any] = {}
        if result.assertion_report is not None:
            metrics_extra["assertions_passed"] = getattr(
                result.assertion_report, "passed", None
            )
        _emit_compile_metrics(
            self.metrics_callback,
            build_compile_metrics(
                wall_time_ms=wall_ms,
                token_count=total_tokens,
                tokens_trimmed=int(fw_stats.get("tokens_trimmed", 0)),
                tokens_budget=max_tok,
                segments_offered=int(fw_stats.get("segments_offered", segments_offered_pre)),
                segments_kept=int(fw_stats.get("segments_kept", 0)),
                ref_map_size=len(ref_map_kept),
                cache_hit=cache_hit,
                is_anonymous=False,
                model_id=self.model_id,
                user_id=None,
                conversation_id=None,
                extra=metrics_extra,
            ),
        )

        collector = prof.metrics_collector or prof.metrics
        if collector is not None:
            try:
                collector.histogram("compile_latency_ms", wall_ms)
                collector.counter("compile_tokens_total", float(total_tokens))
                collector.counter(
                    "compile_tokens_trimmed",
                    float(fw_stats.get("tokens_trimmed", 0)),
                )
            except Exception:
                pass

        _run_trace_exporters(prof, result)

        bus = getattr(prof, "signal_bus", None)
        if bus is not None and stale_survivors:
            await bus.emit(
                Signal(
                    kind=SignalKind.STALE_CONTENT_WARNING,
                    metadata={
                        "segment_ids": stale_survivors,
                        "count": len(stale_survivors),
                    },
                )
            )

        return result


# OpenAI and Anthropic only cache prefixes of at least this many tokens.
MIN_CACHEABLE_PREFIX_TOKENS = 1024


def _build_output_messages(
    model_id: Optional[str],
    final_text: str,
    use_split: bool,
    prefix_text: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One system message; report the byte-stable prefix (``prefix_text``) when it is cacheable.

    ``cacheable_prefix_chars`` lets callers split the content for explicit cache markers
    (for example :func:`membrane.cache_format.anthropic_system_blocks`).
    """
    cache_meta: Dict[str, Any] = {}
    messages: List[Dict[str, Any]] = [{"role": "system", "content": final_text}]
    if not (model_id and use_split and has_prompt_caching(model_id)):
        return messages, cache_meta
    prefix = prefix_text if prefix_text.strip() and final_text.startswith(prefix_text) else ""
    prefix_tokens = count_tokens(prefix)
    if prefix_tokens >= MIN_CACHEABLE_PREFIX_TOKENS:
        cache_meta = {
            "prompt_cache_used": True,
            "cacheable_prefix_chars": len(prefix),
            "cacheable_prefix_tokens": prefix_tokens,
            "dynamic_tail_tokens": count_tokens(final_text[len(prefix):]),
        }
    return messages, cache_meta


def compile_anonymous(
    *,
    user_message: str,
    conversation_messages: List[Dict[str, Any]],
    max_tokens: int,
    model_id: Optional[str] = None,
    core_content: str = "",
    agent_persona_prompt: Optional[str] = None,
    use_prompt_caching_split: bool = True,
    metrics_callback: Optional[MetricsCallback] = None,
) -> CompileResult:
    """No long-term memory: core + conversation + user; firewall if over cap."""
    t0 = time.perf_counter()
    cap = max_tokens
    if model_id:
        mc = lookup_context_cap(model_id)
        if mc:
            cap = min(cap, mc)
    layout = ContextLayoutEngine(core_content=core_content)
    if agent_persona_prompt:
        layout.core_content = layout.core_content + "\n\n" + agent_persona_prompt
    core_text = layout.build_core()
    end_text = layout.build_end(
        last_turn_content=_last_turn_from_messages(conversation_messages),
        user_message=user_message,
    )
    raw_history = _flatten_messages_for_display(conversation_messages)
    if raw_history:
        end_text = "--- Recent conversation ---\n" + raw_history + "\n\n" + end_text
    full = core_text + "\n\n" + end_text
    total = count_tokens(full)
    fw_stats: Dict[str, Any] = {
        "segments_offered": 0,
        "segments_kept": 0,
        "tokens_trimmed": 0,
    }
    if total > cap:
        firewall = PreFlightContextFirewall(max_tokens=cap)
        full, ref_map, total, fw_stats = firewall.run(
            core_text=core_text,
            end_text=end_text,
            middle_segments=[],
            ref_map={},
        )
    else:
        ref_map = {}
    messages, cache_meta = _build_output_messages(
        model_id,
        full,
        use_prompt_caching_split and bool(model_id and has_prompt_caching(model_id)),
        prefix_text=core_text,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    cache_hit = bool(
        cache_meta.get("prompt_cache_used")
        and use_prompt_caching_split
        and bool(model_id and has_prompt_caching(model_id))
    )
    result = CompileResult(
        messages=messages,
        ref_map=ref_map,
        ref_map_type=None,
        system_alerts=[],
        metadata={
            "security_flag": SecurityFlag.ANONYMOUS_MODE.value,
            "token_count": total,
            "tokens_trimmed": int(fw_stats.get("tokens_trimmed", 0)),
            "tokens_budget": cap,
            "segments_offered": int(fw_stats.get("segments_offered", 0)),
            "segments_kept": int(fw_stats.get("segments_kept", 0)),
            **cache_meta,
        },
    )
    if total > cap:
        # History is never trimmed here; report the overrun instead of hiding it.
        result.metadata["budget_exceeded"] = True
    _emit_compile_metrics(
        metrics_callback,
        build_compile_metrics(
            wall_time_ms=wall_ms,
            token_count=total,
            tokens_trimmed=int(fw_stats.get("tokens_trimmed", 0)),
            tokens_budget=cap,
            segments_offered=int(fw_stats.get("segments_offered", 0)),
            segments_kept=int(fw_stats.get("segments_kept", 0)),
            ref_map_size=len(ref_map),
            cache_hit=cache_hit,
            is_anonymous=True,
            model_id=model_id,
            user_id=None,
            conversation_id=None,
        ),
    )
    return result


async def compile_prompt_async(
    *,
    user_msg: str,
    messages_list: List[Dict[str, Any]],
    user_id: Optional[str],
    max_tokens: int,
    model_id: Optional[str] = None,
    core_content: str = "",
    agent_persona_prompt: Optional[str] = None,
    dynamic_directives: Optional[Sequence[str]] = None,
    retrieved_objects: Optional[List[Tuple[Any, int]]] = None,
    current_stress: Optional[float] = None,
    redis_client: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    compress_fn: Optional[Callable[[str, int, str], Awaitable[str]]] = None,
    middle_segments_prefix: Optional[List[Tuple[str, int]]] = None,
    lod_focus_retrieve: Optional[Callable[[], Awaitable[List[Tuple[Any, int]]]]] = None,
    persist_ref_map_to_redis: bool = False,
    redis_url_for_ref_map: Optional[str] = None,
    use_prompt_caching_split: bool = True,
    metrics_callback: Optional[MetricsCallback] = None,
    memory_placement: str = MEMORY_PLACEMENT_SYSTEM,
    profile: Any = None,
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
    trace_exporters: Optional[List[Any]] = None,
    enable_lineage: Optional[bool] = None,
    enable_profiler: Optional[bool] = None,
    metrics_collector: Any = None,
    lineage_tracker: Any = None,
) -> CompileResult:
    """
    Async compile pipeline for pre-retrieved memory pairs.

    Pass ``retrieved_objects`` as (memory_obj, tier) from your retriever.
    ``lod_focus_retrieve`` optional: when attention shifts, called to prepend focus memories.
    ``compress_fn`` optional: Pre-Flight Analyst; if None and over threshold, middle is kept raw.
    ``memory_placement``: see :class:`ContextCompiler` (``"end"`` keeps system + history cacheable).

    Optional Phase 1 kwargs (``profile``, ``reranker``, …) are forwarded to
    :meth:`ContextCompiler.compile_async` so optional modules apply to the same path.
    """
    plugins = (
        profile, reranker, deduplicator, compression, policies, metrics, embedding_cache,
        classifier, freshness, assertions, recorder, tokenizer_registry, trace_exporters,
        enable_lineage, enable_profiler, metrics_collector, lineage_tracker,
    )
    needs_full_path = bool(
        dynamic_directives
        or retrieved_objects
        or middle_segments_prefix
        or compress_fn
        or memory_placement != MEMORY_PLACEMENT_SYSTEM
        or any(p is not None for p in plugins)
    )
    if user_id is None and not needs_full_path:
        return compile_anonymous(
            user_message=user_msg,
            conversation_messages=messages_list,
            max_tokens=max_tokens,
            model_id=model_id,
            core_content=core_content,
            agent_persona_prompt=agent_persona_prompt,
            use_prompt_caching_split=use_prompt_caching_split,
            metrics_callback=metrics_callback,
        )

    attention_shift = compute_attention_shift(_last_user_messages(messages_list), user_msg)

    retrieved: List[Tuple[Any, int]] = list(retrieved_objects or [])
    if attention_shift and lod_focus_retrieve and retrieved:
        try:
            focus_list = await lod_focus_retrieve()
            if focus_list:
                retrieved = _lod_focus_merge(retrieved, focus_list)
        except Exception as e:
            logger.debug("LOD focus retrieval failed: %s", e)

    memories = [o for o, _ in retrieved]
    tier_by_memory_id: Optional[Dict[int, int]] = (
        {id(o): t for o, t in retrieved} if retrieved else None
    )

    cc = ContextCompiler(
        max_tokens=max_tokens,
        model_id=model_id,
        core_content=core_content,
        include_interrupt_instruction=True,
        metrics_callback=metrics_callback,
        memory_placement=memory_placement,
    )
    result = await cc.compile_async(
        memories=memories,
        user_message=user_msg,
        conversation_messages=messages_list,
        agent_persona_prompt=agent_persona_prompt,
        dynamic_directives=dynamic_directives,
        middle_extras=middle_segments_prefix,
        compress_async=compress_fn,
        redis_client=redis_client,
        conversation_id=conversation_id,
        tier_by_memory_id=tier_by_memory_id,
        use_prompt_caching_split=use_prompt_caching_split,
        profile=profile,
        reranker=reranker,
        deduplicator=deduplicator,
        compression=compression,
        policies=policies,
        metrics=metrics,
        embedding_cache=embedding_cache,
        classifier=classifier,
        freshness=freshness,
        assertions=assertions,
        recorder=recorder,
        tokenizer_registry=tokenizer_registry,
        trace_exporters=trace_exporters,
        enable_lineage=enable_lineage,
        enable_profiler=enable_profiler,
        metrics_collector=metrics_collector,
        lineage_tracker=lineage_tracker,
    )

    result.metadata["current_stress"] = current_stress
    result.metadata["user_id"] = user_id
    if user_id is None:
        result.metadata["security_flag"] = SecurityFlag.ANONYMOUS_MODE.value
    if conversation_id is not None:
        result.metadata["conversation_id"] = conversation_id

    if persist_ref_map_to_redis and conversation_id and user_id is not None:
        ref_type = result.ref_map_type or {}
        await set_ref_map_for_conversation(
            conversation_id,
            result.ref_map,
            ref_type or None,
            user_id=user_id,
            redis_client=redis_client,
            redis_url=redis_url_for_ref_map,
        )

    return result


compile_prompt_payload = compile_prompt_async
