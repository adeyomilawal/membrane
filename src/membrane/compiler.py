"""Context compilation: U-shaped layout, pre-flight compression, firewall, glaciation.

Pipeline stages are documented in ``membrane/future.md`` (Compile Pipeline Order) and in
:class:`ContextCompiler` — memory shaping, then layout→trim→format, then
:class:`CompileResult`, then assertions and observability on the complete result.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import time
from bisect import bisect_left
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from itertools import islice
from typing import Any

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
    _COUNT_CACHE_MAX_CHARS,
    _get_encoding,
    _ref_map_from_text,
    count_parts,
    count_tokens,
    parts_add_up,
)
from membrane.keyframes import (
    DEFAULT_KEYFRAME_TOKENS,
    THIN_MARK,
    add_sentences,
    keyframe,
    keyframes,
    matching_sentence,
)
from membrane.keyframes import terms as _terms
from membrane.layout import ContextLayoutEngine
from membrane.observability.profiler import LatencyProfiler
from membrane.lod import compute_attention_shift
from membrane.memory_protocol import infer_memory_kind, normalize_compile_memories
from membrane.metrics import CompileMetrics, MetricsCallback, build_compile_metrics
from membrane.provider_registry import (
    CachePricing,
    get_cache_pricing,
    get_token_scale,
    has_prompt_caching,
    lookup_context_cap,
)
from membrane.ref_map import set_ref_map_for_conversation
from membrane.runtime.signals import Signal, SignalKind

logger = logging.getLogger(__name__)


def _record_jta_lineage_from_compression(
    lineage: Any,
    *,
    compression_strategy_used: str | None,
    compression_details: dict[str, Any],
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


def _stage(profiler: LatencyProfiler | None, name: str):
    if profiler is None:
        return contextlib.nullcontext()
    return profiler.stage(name)


def _build_ranking_diagnostics_for_compile(
    prof: Any,
    retrieved: list[tuple[Any, int]],
    segment_weights: list[float] | None,
    middle_segments: list[tuple[str, int]],
    *,
    ranking_weights: Any | None = None,
    include_ranking_diagnostics: bool | None = None,
) -> dict[str, Any] | None:
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
    policy_kw: dict[str, Any],
    segment_weights_used: bool,
    ranking_diag: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
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
    callback: MetricsCallback | None,
    metrics: CompileMetrics,
) -> None:
    if callback is None:
        return
    try:
        callback.on_compile(metrics)
    except Exception as exc:
        logger.warning("metrics_callback failed: %s", exc, exc_info=True)


PREFLIGHT_ANALYST_SYSTEM = (
    "You are an Intelligence Analyst. Your task is to compress retrieved long-term memory into a "
    'dense summary for the middle "sacrifice zone" of a prompt. Rules:\n'
    "- Output ONLY the compressed summary; no preamble, no \"Summary:\" label.\n"
    "- Use triples where possible: (Subject|Predicate|Object|Confidence).\n"
    "- Use bracket-dense episodes when relevant: [Mem:N|YYYY-MM-DD]: one-line event.\n"
    "- Preserve key entities, dates, and causal links.\n"
    "- Stay strictly under the requested token limit (dense, no filler)."
)

# OpenAI chat framing: tokens billed per message on top of content (and once for priming).
DEFAULT_MESSAGE_OVERHEAD_TOKENS = 3
# End placement: share of the budget after the core that history may hold in the prefix.
DEFAULT_HISTORY_SHARE = 0.75
_EARLIER_HEADER = "--- Earlier conversation ---"
_MEMORIES_HEADER = "--- Memories ---"
_QUESTION_HEADER = "--- Current user message ---\n"  # as ContextLayoutEngine.build_end writes it
_INDEX_HEADER = "--- Earlier conversation: one line per dropped message ---"
DEFAULT_INDEX_SHARE = 0.125
_INDEX_CACHE_SIZE = 64
_EXCERPT_NOTE = "(Lines ending in … are excerpts.)"
# One keyframe per this many keyframe caps of message: a long paste keeps several lines.
_INDEX_COMPRESSION = 10
_INDEX_FLOOR_PART = 2  # the index keeps at least this part of its share under memory pressure
_THIN_LINE_EXTRA = 8  # a thinned memory line: its keyframe plus the ref, the mark and a line break

MEMORY_PLACEMENT_SYSTEM = "system"
MEMORY_PLACEMENT_END = "end"
MEMORY_PLACEMENTS = (MEMORY_PLACEMENT_SYSTEM, MEMORY_PLACEMENT_END)

INTERRUPT_INSTRUCTION = (
    "If a detail is missing from the provided context, do not hallucinate. Output exactly: "
    "<SYSTEM_INTERRUPT_FETCH[Your precise query for the missing detail]> so the system can "
    "fetch it. Use request_memory_expansion(local_id) for [Mem:N] references when you need "
    "full prose; use search_cognitive_memory(query) for new searches."
)


_SCALE_GRID = 0.05  # the applied scale moves in steps of this size
_SCALE_SLACK = 0.1  # and only comes down once it is this much too high
_SCALE_SMOOTHING = 0.2
_SCALE_BOUNDS = (0.25, 4.0)


@dataclass
class _TokenScale:
    """What one model has billed against what Membrane counted, smoothed.

    ``applied`` is the scale budgets are divided by. It follows ``mean + 2 * spread`` but on
    a grid and with slack on the way down, because every change moves the budget and with
    it the history cut, which breaks the cached prefix once.
    """

    mean: float
    applied: float
    spread: float = 0.0
    samples: int = 0

    def add(self, ratio: float) -> None:
        ratio = min(max(ratio, _SCALE_BOUNDS[0]), _SCALE_BOUNDS[1])
        self.samples += 1
        # The seed counts as one sample, so the first few observations move it quickly.
        weight = max(_SCALE_SMOOTHING, 1.0 / (self.samples + 1))
        self.spread += weight * (abs(ratio - self.mean) - self.spread)
        self.mean += weight * (ratio - self.mean)
        target = min(self.mean + 2.0 * self.spread, _SCALE_BOUNDS[1])
        if target > self.applied or target < self.applied - _SCALE_SLACK:
            self.applied = round(math.ceil(target / _SCALE_GRID - 1e-9) * _SCALE_GRID, 4)


class SecurityFlag(str, Enum):
    ANONYMOUS_MODE = "anonymous"
    AUTHENTICATED = "authenticated"


@dataclass
class CompileResult:
    messages: list[dict[str, Any]]
    ref_map: dict[int, str]
    ref_map_type: dict[int, str] | None = None
    system_alerts: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 1 optional attachments
    debug_trace: dict[str, Any] | None = None
    latency_profile: Any | None = None
    dedup_report: Any | None = None
    lineage_report: Any | None = None
    assertion_report: Any | None = None
    freshness_flags: list[str] | None = None

    def to_debug_dict(self) -> dict[str, Any]:
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

    def diff(self, other: CompileResult) -> Any:
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
    estimated_cost_usd: float | None
    trimming_strategies: list[str]
    compile_result: CompileResult


def _build_trimming_strategies(
    *,
    total_tokens: int,
    tokens_trimmed: int,
    trimmed_count: int,
    tokens_budget: int,
    target_token_budget: int | None,
) -> list[str]:
    """Deterministic suggestions; structural checks only (no content safety)."""
    out: list[str] = []
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
    price_per_1k_input_tokens: float | None = None,
    target_token_budget: int | None = None,
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

    cost: float | None = None
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


def _last_turn_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("stage3_synthesis") or m.get("content") or m.get("stage3")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, dict) and content.get("text"):
                return str(content["text"])
    return None


def _last_user_messages(messages: list[dict[str, Any]], cap: int = 3) -> list[str]:
    user_contents: list[str] = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, str):
                user_contents.append(content)
            elif isinstance(content, dict) and "text" in content:
                user_contents.append(str(content["text"]))
    return user_contents[-cap:] if user_contents else []


_INTERNAL_MESSAGE_KEYS = ("stage3", "stage3_synthesis")


def _normalize_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provider-ready history: shallow copies with every provider key kept as given.

    ``tool_calls``, ``tool_call_id``, ``name`` and content-part lists pass through
    untouched. Only internal keys (``stage3``, ``stage3_synthesis``) are dropped, with
    ``stage3_synthesis`` used as ``content`` when content is missing. Deterministic, so
    the same history always serializes to the same bytes.
    """
    out: list[dict[str, Any]] = []
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


def _trim_history_chunked(
    history: list[dict[str, Any]],
    limit: int,
    count_fn: Callable[[str], int],
    step: int | None = None,
    sizes: list[int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Drop the oldest messages so the rest fits ``limit``; return (kept, dropped count).

    The cut lands on a grid of half the limit, measured in tokens from the start of the
    conversation. History is append-only, so the cut stays put while new turns arrive and
    only jumps when another half-limit of history has built up: the cached prefix breaks
    once per jump instead of every turn. The kept history always starts on a user message
    so tool results are never separated from their call. Pass ``step`` to keep the grid
    fixed while ``limit`` varies.
    """
    if sizes is None:
        sizes = [count_fn(_message_text(m)) for m in history]
    total = sum(sizes)
    if total <= limit or not history:
        return history, 0
    if limit <= 0:
        return [], len(history)
    step = max(1, step if step is not None else limit // 2)
    boundary = -(-(total - limit) // step) * step  # round up to the grid
    start, seen = 0, 0
    while start < len(history) and seen < boundary:
        seen += sizes[start]
        start += 1
    while start < len(history) and history[start].get("role") != "user":
        start += 1
    return history[start:], start


def _cut_starts(
    history: list[dict[str, Any]], sizes: list[int], step: int, dropped: int
) -> tuple[int, ...]:
    """Where each cut on the grid landed on the way to the current one, ending at ``dropped``.

    The same walk as :func:`_trim_history_chunked`, one grid step at a time, so the answer
    depends only on the dropped messages and never on what came after them.
    """
    starts: list[int] = []
    walked, seen, boundary = 0, 0, step
    while True:
        while walked < len(history) and seen < boundary:
            seen += sizes[walked]
            walked += 1
        start = walked
        while start < len(history) and history[start].get("role") != "user":
            start += 1
        if start >= dropped:
            break
        if not starts or start > starts[-1]:
            starts.append(start)
        boundary += step
    return (*starts, dropped)


@dataclass
class _IndexState:
    """The eviction index as built so far: every line, and the sentence counts it was scored
    with. A later cut continues from here instead of starting over."""

    lines: list[str] = field(default_factory=list)
    df: dict[str, int] = field(default_factory=dict)
    n: int = 0
    budget: int = 0  # tokens the block may hold, fixed at the cut that built this state
    trimmed: dict[int, tuple[str, ...]] = field(default_factory=dict)

    def copy(self) -> _IndexState:
        """A state the next cut can add to without changing this one."""
        return _IndexState(lines=list(self.lines), df=dict(self.df), n=self.n)

    def add_cut(
        self,
        batch: Sequence[tuple[Any, ...]],
        count_fn: Callable[[str], int],
        cap: int,
    ) -> None:
        """Add the messages one cut dropped: first to the counts, then their lines. Each is
        (role, text, tokens) and, for a tool result, the call that produced it."""
        texts = [
            (_RECALL_ROLES[role], text, size, call[0] if call else None)
            for role, text, size, *call in batch
            if role in _RECALL_ROLES and text and text.strip()
        ]
        for _, text, _, _ in texts:
            self.n += add_sentences(self.df, text)
        for label, text, size, call in texts:
            limit = 1 + size // (_INDEX_COMPRESSION * max(1, cap))
            frames = keyframes(text, self.df, self.n, count_fn, cap, limit)
            if frames:
                if call:
                    # A tool result answers a call; without it the line may mean nothing.
                    name = call.split(" ", 1)[0]
                    label = f"{label} ({call if count_fn(call) <= cap else name})"
                # An old citation must not claim a ref in this turn's ref_map.
                line = f"{label}: " + f"{THIN_MARK} ".join(frames)
                self.lines.append(line.replace("[Mem:", "[mem:"))
        self.trimmed.clear()

    def within(self, budget: int, count_fn: Callable[[str], int]) -> tuple[str, ...]:
        """The newest lines that fit ``budget`` tokens, header included."""
        lines = self.trimmed.get(budget)
        if lines is None:
            room = budget - count_fn(_INDEX_HEADER) - 1
            keep = len(self.lines)
            while keep > 0 and room >= count_fn(self.lines[keep - 1]) + 1:
                room -= count_fn(self.lines[keep - 1]) + 1
                keep -= 1
            lines = self.trimmed[budget] = tuple(self.lines[keep:])
        return lines


def _tool_calls(history: Sequence[dict[str, Any]]) -> dict[int, str]:
    """For each tool result in ``history`` (by position): the call that produced it, as its
    name and arguments. Only OpenAI-style ``tool_calls`` are read; anything else has none."""
    calls: dict[str, str] = {}
    out: dict[int, str] = {}
    for position, message in enumerate(history):
        for call in message.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and function.get("name") and call.get("id"):
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, sort_keys=True) if arguments else ""
                calls[str(call["id"])] = " ".join(f"{function['name']} {arguments}".split())
        if message.get("role") == "tool":
            found = calls.get(str(message.get("tool_call_id")))
            if found:
                out[position] = found
    return out


def _eviction_index(
    evicted: Sequence[tuple[Any, ...]],
    starts: tuple[int, ...],
    count_fn: Callable[[str], int],
    cap: int,
    budget: int,
) -> tuple[str, ...]:
    """One line per dropped message: its role and its keyframe, oldest first.

    ``evicted`` is (role, text, tokens) for every dropped message, plus the call for a tool
    result (see :func:`_tool_calls`); ``starts`` are the cuts that dropped them. A tool result's
    line names its call, so the result still means something on its own. A message's keyframe is
    scored over the conversation up to the cut that dropped it, so its line never changes
    afterwards and the index only grows at a cut. A long message keeps one keyframe for every
    ten caps of its length, so a pasted document is not reduced to a single sentence. Over
    ``budget`` (header included), the oldest lines go.
    """
    state, previous = _IndexState(), 0
    for start in starts:
        state.add_cut(evicted[previous:start], count_fn, cap)
        previous = start
    return state.within(budget, count_fn)


def _recall_postings(
    candidates: Sequence[tuple[str, frozenset[str]]],
    postings: dict[str, list[int]] | None = None,
    start: int = 0,
) -> dict[str, list[int]]:
    """For each term, the candidates that contain it, in ascending order. With ``postings``
    and ``start``, the candidates from ``start`` on are added to what is already there."""
    postings = {} if postings is None else postings
    for i in range(start, len(candidates)):
        for term in candidates[i][1]:
            postings.setdefault(term, []).append(i)
    return postings


class _RecallIndex:
    """Passages of a conversation's dropped messages and their postings, grown cut by cut.

    Dropped history only grows at its end, so a later cut adds its messages' passages to
    the same lists. ``ends[k]`` is how many passages the first ``k`` messages gave, so an
    earlier cut (a shorter prefix of the same messages) reads the same lists up to its own
    count.
    """

    def __init__(self) -> None:
        self.key: tuple[tuple[Any, ...], ...] = ()
        self.candidates: list[tuple[str, frozenset[str]]] = []
        self.postings: dict[str, list[int]] = {}
        self.ends: list[int] = [0]

    def covers(self, key: tuple[tuple[Any, ...], ...]) -> bool:
        """Whether ``key`` and the messages indexed so far are one conversation."""
        shared = min(len(key), len(self.key))
        return key[:shared] == self.key[:shared]

    def upto(
        self, key: tuple[tuple[Any, ...], ...], evicted: list[dict[str, Any]]
    ) -> tuple[list[tuple[str, frozenset[str]]], dict[str, list[int]], int]:
        """Candidates, postings and how many of the candidates belong to ``key``."""
        for message in evicted[len(self.key) : len(key)]:
            start = len(self.candidates)
            self.candidates.extend(_recall_candidates([message]))
            _recall_postings(self.candidates, self.postings, start)
            self.ends.append(len(self.candidates))
        if len(key) > len(self.key):
            self.key = key
        return self.candidates, self.postings, self.ends[len(key)]


def _rank_postings(postings: dict[str, list[int]], n: int, query: str, limit: int) -> list[int]:
    """The first ``limit`` candidate indices in the order :func:`_rank_recall` gives, found
    through the postings so only candidates that share a term with the query are scored."""
    hits: dict[int, float] = {}
    for term in sorted(_terms(query)):
        found = postings.get(term)
        # The postings may already hold later candidates (see _RecallIndex): stop at ``n``.
        known = bisect_left(found, n) if found else 0
        if known:
            weight = math.log(1.0 + n / known)
            for i in islice(found, known):
                hits[i] = hits.get(i, 0.0) + weight
    ranked = sorted(hits, key=lambda i: (-hits[i], -i))[:limit]
    newest = n - 1
    while len(ranked) < limit and newest >= 0:
        if newest not in hits:
            ranked.append(newest)
        newest -= 1
    return ranked


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_PASSAGE_WORDS = 60
_RECALL_SCAN = 256  # ranked passages token-counted per compile, at most
_LEADING_REF = re.compile(r"^\s*\[Mem:\d+[^\]]*\]:?\s*")
_EARLIER_LABEL = re.compile(r"^\[Earlier [^\]]+\]\s*")
_RECALL_ROLES = {"user": "user", "assistant": "assistant", "tool": "tool result"}


@lru_cache(maxsize=8192)
def _passages(label: str, content: str) -> tuple[tuple[str, frozenset[str]], ...]:
    """Split a message into passages of about 60 words, each with its term set.

    ``[Mem:`` is neutralized so an old citation never claims a ref in this turn's ``ref_map``.
    """
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(content.strip().replace("[Mem:", "[mem:")):
        words = sentence.split()
        # A run-on without punctuation is cut by length, so no passage is too big to recall.
        pieces += [
            " ".join(words[i : i + _PASSAGE_WORDS]) for i in range(0, len(words), _PASSAGE_WORDS)
        ]
    passages: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        n = len(piece.split())
        if current and size + n > _PASSAGE_WORDS:
            passages.append(" ".join(current))
            current, size = [], 0
        current.append(piece)
        size += n
    if current:
        passages.append(" ".join(current))
    return tuple((f"[Earlier {label}] {text}", _terms(text)) for text in passages)


def _recall_candidates(evicted: list[dict[str, Any]]) -> list[tuple[str, frozenset[str]]]:
    """Passages of evicted history, oldest first. Tool results come back as plain text;
    tool-call stubs and non-text content are skipped."""
    out: list[tuple[str, frozenset[str]]] = []
    for message in evicted:
        label = _RECALL_ROLES.get(str(message.get("role")))
        content = message.get("content")
        if label and isinstance(content, str) and content.strip():
            out.extend(_passages(label, content))
    return out


def _relevance(term_sets: Sequence[frozenset[str]], query: str) -> list[float]:
    """Lexical relevance of each text to the query: the summed IDF (over these texts) of the
    query terms it contains, so rare shared words count most."""
    wanted = _terms(query)
    shared = [terms & wanted for terms in term_sets]
    df: dict[str, int] = {}
    for hit in shared:
        for term in hit:
            df[term] = df.get(term, 0) + 1
    n = len(term_sets)
    idf = {term: math.log(1.0 + n / count) for term, count in df.items()}
    return [sum(idf[t] for t in hit) if hit else 0.0 for hit in shared]


def _rank_recall(candidates: list[tuple[str, frozenset[str]]], query: str) -> list[int]:
    """Candidate indices, best first: by relevance to the query; ties and non-matches go
    to the most recently evicted."""
    scores = _relevance([terms for _, terms in candidates], query)
    return sorted(range(len(candidates)), key=lambda i: (-scores[i], -i))


_RRF_K = 60  # the usual reciprocal-rank-fusion constant


def _fused_keep_order(segments: Sequence[tuple[str, int]], query: str) -> list[int]:
    """Segment indices in the order they should be offered to the firewall.

    Reciprocal rank fusion of the given order (the retriever's rank) and lexical relevance
    to the query. Texts with equal relevance share the last rank of their group: a tie says
    nothing, so it earns nothing, and with no lexical signal this is the given order.
    """
    scores = _relevance([_terms(_LEADING_REF.sub("", text)) for text, _ in segments], query)
    by_relevance = sorted(range(len(segments)), key=lambda i: -scores[i])
    lexical_rank: dict[int, int] = {}
    for position in range(len(by_relevance) - 1, -1, -1):
        i = by_relevance[position]
        below = by_relevance[position + 1] if position + 1 < len(by_relevance) else None
        tied = below is not None and scores[below] == scores[i]
        lexical_rank[i] = lexical_rank[below] if tied else position  # type: ignore[index]
    return sorted(
        range(len(segments)),
        key=lambda i: (-(1.0 / (_RRF_K + i) + 1.0 / (_RRF_K + lexical_rank[i])), i),
    )


class _ConversationCounts:
    """Sentence counts of a conversation, kept up to date as it grows.

    Conversations are append-only, so each compile only adds the messages it has not seen.
    If an earlier message changed, the counts are rebuilt.
    """

    def __init__(self) -> None:
        self.seen: list[Any] = []
        self.df: dict[str, int] = {}
        self.n = 0

    def update(self, history: Sequence[dict[str, Any]]) -> tuple[dict[str, int], int]:
        contents = [m.get("content") for m in history]
        known = len(self.seen)
        same = known <= len(contents) and all(
            a is b or a == b for a, b in zip(self.seen, contents, strict=False)
        )
        if not same:
            self.df, self.n, known = {}, 0, 0
        for content in contents[known:]:
            if isinstance(content, str):
                self.n += add_sentences(self.df, content)
        self.seen = contents
        return self.df, self.n


def _thin_forms(
    segments: Sequence[tuple[str, int]],
    recalled: set[str],
    conversation: tuple[dict[str, int], int],
    query: str,
    count_fn: Callable[[str], int],
    cap: int,
    cost_fn: Callable[[str], int] | None = None,
) -> list[str | None]:
    """Thin form of each segment: its ``[Mem:N]`` ref, its keyframe and an ellipsis.

    A memory's keyframe is its densest sentence, measured over the memories on offer and
    the conversation so far. A recalled passage (in ``recalled``) was picked for the words
    it shares with the question, so its thin form is the sentence that shares them.
    A memory with no sentence under the cap, or one that thinning would not shorten, is its
    own thin form; a recalled passage with no matching sentence has ``None``.
    """
    bodies: list[tuple[str, str]] = []
    for text, _ in segments:
        lead = (_EARLIER_LABEL if text in recalled else _LEADING_REF).match(text)
        prefix = lead.group(0) if lead else ""
        bodies.append((prefix, text[len(prefix) :]))
    # Recalled passages are history, which is already in the conversation's counts.
    df, n = dict(conversation[0]), conversation[1]
    for (text, _), (_, body) in zip(segments, bodies, strict=True):
        if text not in recalled:
            n += add_sentences(df, body)
    wanted = _terms(query)
    cost = cost_fn or count_fn
    out: list[str | None] = []
    for (text, _), (prefix, body) in zip(segments, bodies, strict=True):
        if text in recalled:
            frame = matching_sentence(body, wanted, count_fn, cap)
        else:
            frame = keyframe(body, df, n, count_fn, cap)
        if frame is None:
            # A memory that cannot be thinned is its own thin form: it is placed with the
            # keyframes, so no excerpt of anything ranked below it can push it out. A
            # recalled passage with nothing to show waits until the memories are done.
            out.append(None if text in recalled else text)
            continue
        thin = (prefix.strip() + " " if prefix.strip() else "") + frame + THIN_MARK
        out.append(thin if cost(thin) < cost(text) else text)
    return out


def _history_cache_pays(pricing: CachePricing, expected_reads: float) -> bool:
    """Whether writing the history to the cache is expected to cost less than not doing so.

    A cut rewrites everything after the core, so history written to the cache is only worth
    its write premium if it is read back before the next cut. With automatic caching there
    is no premium and nothing to decide.
    """
    if not pricing.marked or pricing.write <= 1.0:
        return True
    return expected_reads * (1.0 - pricing.read) >= pricing.write - 1.0


def _turns_until_cut(
    history: list[dict[str, Any]], sizes: list[int], dropped: int, limit: int, step: int
) -> list[float]:
    """For each request sent since the current cut, oldest first and ending with this one:
    how many more turns its history was expected to last before the next cut (the room left
    on the grid over the mean size of a kept turn).

    A request is one user message with the history before it, so the earlier requests are
    the prefixes of ``history`` that end just before a user message.
    """

    def boundary(total: int) -> int:
        return 0 if total <= limit else -(-(total - limit) // step) * step

    total = sum(sizes)
    now = boundary(total)
    kept_from = total - sum(sizes[dropped:])
    turns = sum(1 for m in history[dropped:] if m.get("role") == "user")
    end = len(history)
    out: list[float] = []
    while turns > 0 and boundary(total) == now:
        out.append((limit + now - total) / max(1.0, (total - kept_from) / turns))
        # Step back one request: everything from the last user message on was not there yet.
        last = end - 1
        while last > dropped and history[last].get("role") != "user":
            last -= 1
        total -= sum(sizes[last:end])
        end = last
        turns -= 1
    return out[::-1]


def _message_text(message: dict[str, Any]) -> str:
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


def _flatten_messages_for_display(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
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
    parts: list[str] = []
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
    retrieved: list[tuple[Any, int]],
    ref_map_kept: dict[int, str],
) -> list[str]:
    """
    After the firewall, ``ref_map_kept`` maps local [Mem:N] indices to global memory ids.
    Return ids for kept refs whose source objects were marked ``_membrane_is_stale`` by
    :class:`~membrane.freshness.manager.FreshnessManager`.
    """
    id_to_obj: dict[str, Any] = {}
    for i, (obj, _) in enumerate(retrieved):
        mid = str(getattr(obj, "id", f"seg_{i}"))
        id_to_obj[mid] = obj
    survivors: list[str] = []
    for mid in sorted(set(ref_map_kept.values())):
        obj = id_to_obj.get(mid)
        if obj is not None and bool(getattr(obj, "_membrane_is_stale", False)):
            survivors.append(mid)
    return survivors


LOD_FOCUS_EPISODIC = 2
LOD_FOCUS_SEMANTIC = 2
LOD_FOCUS_PROCEDURAL = 1


def _lod_focus_merge(
    main_retrieved: list[tuple[Any, int]],
    focus_retrieved: list[tuple[Any, int]],
) -> list[tuple[Any, int]]:
    caps = {
        TIER_PROCEDURAL: LOD_FOCUS_PROCEDURAL,
        TIER_SEMANTIC: LOD_FOCUS_SEMANTIC,
        TIER_EPISODIC: LOD_FOCUS_EPISODIC,
    }
    counts = {TIER_PROCEDURAL: 0, TIER_SEMANTIC: 0, TIER_EPISODIC: 0}
    focus_list: list[tuple[Any, int]] = []
    seen_ids: set[Any] = set()
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


def _policy_disabled_features(policy_kw: dict[str, Any]) -> list[str]:
    """Features disabled by policies, minus any a policy re-enabled."""
    enabled = set(policy_kw.get("policy_enable") or [])
    return [f for f in (policy_kw.get("policy_disable") or []) if f not in enabled]


def _policy_tier_settings(
    policy_kw: dict[str, Any],
) -> tuple[dict[int, float] | None, set[int] | None]:
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

    ``memory_placement`` controls where retrieved memories go. ``"end"`` (default) emits
    the system message (core only) and ``conversation_messages`` as given, then one user
    message with memories, the current user message and dynamic directives. System +
    history then form a byte-stable prefix that prompt caching can reuse across turns.
    ``"system"`` emits one system message: core, memories, last turn, user message. It
    sends less (only the last turn of history) but memories change every turn, so the
    provider cache cannot hit past the core; it suits local models and one-off calls.

    With ``trim_history`` (default) end placement keeps history within ``history_share`` of
    the budget left after the core, dropping the oldest messages in chunks of half that
    share. The cut depends only on the budget and the core, never on this turn's memories or
    question, so it stays put while the conversation grows and the cached prefix breaks only
    when another chunk is dropped. ``metadata["history_messages_dropped"]`` reports how
    many. Turn it off to manage history yourself; an overrun then sets
    ``metadata["budget_exceeded"] = True``.

    With ``recall_evicted_history`` (default) dropped messages are not lost. They are split
    into passages of about 60 words, tool results included, and ranked by the rare words
    they share with the current user message, then by recency. The best passages, up to one
    chunk (half the history limit), join the sacrifice zone as low-priority episodic
    segments in whatever room the memories leave: every memory is offered its place before
    any passage is, so a passage costs the memories nothing but the two header lines that
    introduce the passages. A passage that does not fit whole is cut to the sentence that
    matches the question (see ``thin_before_drop``). They never enter ``ref_map``;
    ``metadata["history_messages_recalled"]`` reports how many survived.

    With ``rank_fusion`` (default, end placement) the memories that survive trimming are
    chosen by reciprocal rank fusion of the order you passed (your retriever's rank) and
    lexical relevance to the user message, within each tier. Survivors keep the order you
    passed, so nothing changes when everything fits. It is skipped when a reranker supplies
    weights.

    With ``eviction_index`` (default, end placement with ``trim_history``) every dropped
    message leaves one line behind: its keyframe (see below), in a block appended to the
    system message after the core. A line is written when the cut that drops its message
    happens and never changes afterwards, so the block only changes at a cut, where the
    cached prefix after the core is rewritten anyway: it adds no cache breaks. The block
    has up to ``index_share`` (default one eighth) of the budget the core leaves, taken
    from the room history does not use (never out of ``history_share``). At each cut it
    gets what the question and a keyframe of every memory on offer leave of that room, at
    least half its share, and keeps that size until the next cut; a compiler that has not
    seen the cut sizes it from the turn at hand. Over its size the oldest lines go. Tool results
    get a line, tool-call stubs do not. ``metadata["history_index_lines"]`` and
    ``metadata["history_index_tokens"]`` report it, and ``metadata["core_chars"]`` is
    where the core ends inside the system message. Passage recall above still returns
    matching passages in full.

    When the model caches prompts, ``metadata["cache_breakpoints"]`` lists where a cached
    prefix ends, as ``{"message_index", "tokens", "reason"}`` (plus ``"chars"`` when the
    prefix ends inside the message): after the core, and after the history. Providers that
    cache automatically need nothing more. For providers that need markers (Anthropic),
    :func:`membrane.cache_format.anthropic_request` builds the request with them placed.
    The history is marked only when the cache prices say it pays: a cut rewrites the
    history, so it must be read back before the next cut to earn its write premium. The
    expected number of reads is the room left before the next cut over the mean size of a
    kept turn (``metadata["history_cache_expected_reads"]``). Once a request since the cut
    has marked it, later ones keep the marker, which then reads what is cached. Prices and
    minimum cacheable sizes come from the capability map (``cache_pricing``).

    Budgets are calibrated to the provider's own token counts. Membrane counts with
    tiktoken for every model, and other tokenizers count the same text differently (Claude
    counts more). The budget is divided by a per-model ``token_scale``: it starts from the
    capability map's seed and is corrected by :meth:`observe`, which you call after each
    request with the prompt tokens the provider reported. ``metadata["token_scale"]`` and
    ``metadata["token_scale_samples"]`` report it; ``metadata["tokens_budget_counted"]`` is
    the budget in counted tokens when the scale is not 1. Pass ``token_scale=`` to fix the
    scale yourself (1.0 turns calibration off). No tokenizer is downloaded and nothing is
    sent anywhere.

    With ``thin_before_drop`` (default, end placement) no memory is dropped while another
    still has its full text. Every memory has a keyframe: its densest sentence under
    ``keyframe_tokens`` (mean rarity of its words over the sentences of the memories on offer
    and of the conversation). Under pressure the keyframes of all memories are placed first, then
    memories get their full text back in priority order while room lasts. A thinned memory
    keeps its ``[Mem:N]`` ref and ends in ``…``; ``metadata["segments_thinned"]`` counts
    them. A memory with no sentence under the cap is kept whole or dropped, as before.
    Recalled passages take part too: their thin form is the sentence that shares the most
    with the question, placed after every memory's keyframe and before memories get their
    full text back.

    ``message_overhead_tokens`` is what the provider bills per chat message on top of its
    content (3 for OpenAI chat models), plus the same again once for reply priming. The
    budget reserves it so the billed request stays under ``max_tokens``;
    ``metadata["token_count"]`` stays content-only and ``metadata["message_framing_tokens"]``
    reports the reserve. Set it to 0 to budget content only.
    """

    max_tokens: int
    model_id: str | None = None
    core_content: str = ""
    include_interrupt_instruction: bool = True
    preflight_analyst_model: str | None = None
    metrics_callback: MetricsCallback | None = None
    memory_placement: str = MEMORY_PLACEMENT_END
    trim_history: bool = True
    history_share: float = DEFAULT_HISTORY_SHARE
    recall_evicted_history: bool = True
    rank_fusion: bool = True
    message_overhead_tokens: int = DEFAULT_MESSAGE_OVERHEAD_TOKENS
    thin_before_drop: bool = True
    keyframe_tokens: int = DEFAULT_KEYFRAME_TOKENS
    eviction_index: bool = True
    index_share: float = DEFAULT_INDEX_SHARE
    token_scale: float | None = None
    _conversations: dict[str | None, _ConversationCounts] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _big_counts: dict[tuple[Any, str], int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _scales: dict[tuple[str | None, bool], _TokenScale] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _index_cache: dict[Any, _IndexState] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _recall_cache: dict[Any, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.history_share <= 1.0:
            raise ValueError(f"history_share must be within [0, 1], got {self.history_share!r}")
        if not 0.0 <= self.index_share <= 1.0:
            raise ValueError(f"index_share must be within [0, 1], got {self.index_share!r}")
        if self.token_scale is not None and self.token_scale <= 0:
            raise ValueError(f"token_scale must be positive, got {self.token_scale!r}")
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
        middle_segments: list[tuple[str, int]],
        ref_map: dict[int, str],
        segment_weights: list[float] | None,
        conversation_messages: list[dict[str, Any]],
        last_assistant_turn: str | None,
        user_message: str,
        max_tok: int,
        use_split: bool,
        count_fn: Callable[[str], int],
        tier_shares: dict[int, float] | None = None,
        pinned_tiers: set[int] | None = None,
        thin: bool = True,
        count_key: Any = None,
        token_scale: float = 1.0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, str], int, dict[str, Any]]:
        """Build end block, trim the middle under ``max_tok``, and format provider messages.

        ``memory_placement="system"``: one system message (core, memories, end, directives).
        ``memory_placement="end"``: system (core only) + history as given + one user message
        holding memories, the current user message and directives, so the system message and
        history form a byte-stable prefix across turns.
        """
        caching = use_split and bool(self.model_id and has_prompt_caching(self.model_id))
        overhead = max(0, self.message_overhead_tokens)
        directives_block = layout.build_directives_block()

        if self.memory_placement == MEMORY_PLACEMENT_SYSTEM:
            end_text = layout.build_end(
                last_turn_content=last_assistant_turn
                or _last_turn_from_messages(conversation_messages),
                user_message=user_message,
            )
            # One message plus reply priming.
            framing = 2 * overhead
            firewall = PreFlightContextFirewall(max_tokens=max_tok - framing, count_fn=count_fn)
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
            if framing:
                cache_meta["message_framing_tokens"] = framing
            if total_tokens + framing > max_tok:
                cache_meta["budget_exceeded"] = True
            return messages, cache_meta, ref_map_kept, total_tokens, fw_stats

        history = _normalize_history(conversation_messages)
        end_text = layout.build_end(
            # The last assistant turn is already in the history messages.
            last_turn_content=None if history else last_assistant_turn,
            user_message=user_message,
        )
        history_dropped = 0
        end_with_directives = end_text + directives_block
        # With the built-in tokenizer the tail is counted in parts that add up exactly (see
        # firewall.parts_add_up): a line and its line break, the question. Parts seen on an
        # earlier turn, and the question when it becomes history, are not tokenised again.
        in_parts = count_key is None and _get_encoding() is not False
        end_parts = [end_with_directives]
        if in_parts and end_with_directives.startswith(_QUESTION_HEADER):
            end_parts = [_QUESTION_HEADER, end_with_directives[len(_QUESTION_HEADER) :]]
        end_tokens = count_parts(end_parts, count_fn)[1]

        def line_cost(text: str) -> int:
            """What one line costs in the tail."""
            return count_fn(text + "\n") if in_parts else count_fn(text)

        indexing = bool(self.eviction_index and self.trim_history and history)
        core_tokens = self._big_count(core_text, count_fn, count_key)
        # System message (when there is one, or the index may need one), the final user
        # message, and reply priming.
        fixed_framing = overhead * ((1 if core_text.strip() or indexing else 0) + 2)
        evicted: list[dict[str, Any]] = []
        evicted_key: tuple[tuple[Any, ...], ...] = ()
        history_tokens: int | None = None
        full_history = history
        index_lines: tuple[str, ...] = ()
        until_cut: list[float] | None = None  # see _turns_until_cut; None when never trimmed
        if self.trim_history and history:
            # Stable inputs only (budget, core, framing), so the grid never moves between turns.
            base = max(0, max_tok - fixed_framing - core_tokens)
            history_limit = max(0, int(base * self.history_share))
            # The index has its own room in what history leaves, never out of history's share.
            leftover = base - history_limit
            index_cap = min(int(base * self.index_share), leftover) if indexing else 0
            # Extra grid steps only when this turn's own end block would not fit.
            limit = min(history_limit, base - index_cap - end_tokens)
            step = max(1, history_limit // 2)
            # A long message (a pasted document) is over the shared count cache's size
            # limit; remembered here, it is tokenised once instead of on every turn.
            sizes = [
                (
                    self._big_count(text, count_fn, count_key)
                    if len(text := _message_text(m)) > _COUNT_CACHE_MAX_CHARS
                    else count_fn(text)
                )
                + overhead
                for m in history
            ]
            history, history_dropped = _trim_history_chunked(
                history, limit, count_fn, step=step, sizes=sizes
            )
            evicted = full_history[:history_dropped]
            history_tokens = sum(sizes[history_dropped:]) - overhead * len(history)
            calls = _tool_calls(evicted) if indexing else {}
            evicted_key = tuple(
                (
                    str(m.get("role")),
                    c if isinstance(c := m.get("content"), str) else None,
                    size,
                    *((calls[i],) if i in calls else ()),
                )
                for i, (m, size) in enumerate(zip(evicted, sizes, strict=False))
            )
            if history:
                until_cut = _turns_until_cut(
                    full_history, sizes, history_dropped, history_limit, step
                )
            if indexing and evicted and index_cap > 0:

                def index_budget() -> int:
                    """Asked once per cut: what the question and a keyframe of every memory
                    on offer leave of the leftover, between half the index's share and all
                    of it. Memories give up their full text before the index gives up a
                    line, but the index never costs a memory its keyframe."""
                    thin = self.keyframe_tokens + _THIN_LINE_EXTRA
                    wanted = end_tokens + sum(
                        min(thin, line_cost(text)) for text, _ in middle_segments
                    )
                    return min(index_cap, max(index_cap // _INDEX_FLOOR_PART, leftover - wanted))

                index_lines = self._index_lines(
                    evicted_key, full_history, sizes, step, count_fn, index_budget, count_key
                )
        system_text, system_tokens = core_text, core_tokens
        if index_lines:
            block = "\n".join((_INDEX_HEADER, *index_lines))
            system_text = core_text + "\n\n" + block if core_text.strip() else block
            if in_parts:
                # Line by line (exact, see firewall.parts_add_up): a cut only tokenises the
                # lines it adds, not the core and the whole index again.
                lead = [core_text + "\n\n"] if core_text.strip() else []
                lines = [_INDEX_HEADER, *index_lines]
                parts = [*lead, *(f"{line}\n" for line in lines[:-1]), lines[-1]]
                if parts_add_up(parts):
                    system_tokens = sum(count_fn(part) for part in parts[len(lead) :])
                    if lead:
                        system_tokens += self._big_count(lead[0], count_fn, count_key)
                else:
                    system_tokens = self._big_count(system_text, count_fn, count_key)
            else:
                system_tokens = self._big_count(system_text, count_fn, count_key)
        # Under pressure, which memories survive is decided by the retriever's rank fused
        # with relevance to this turn; the survivors are still emitted in the given order.
        given_order: dict[str, int] = {}
        if self.rank_fusion and segment_weights is None and len(middle_segments) > 1:
            for i, (text, _) in enumerate(middle_segments):
                given_order.setdefault(text, i)
            middle_segments = [
                middle_segments[i] for i in _fused_keep_order(middle_segments, user_message)
            ]
        recalled_texts: set[str] = set()
        recall_order: dict[str, int] = {}
        if self.recall_evicted_history and evicted:
            # Passages of dropped history, picked by relevance to this turn and then recency.
            # Capped at one chunk so recall does not inflate every request to the full budget.
            candidates, postings, known = self._recall_index(evicted_key, evicted)
            recall_cap, recalled = max(1, history_limit // 2), []
            ranked = _rank_postings(postings, known, user_message, _RECALL_SCAN)
            for i in ranked:
                text = candidates[i][0]
                cost = line_cost(text)
                if cost > recall_cap or text in recall_order:
                    continue
                recall_cap -= cost
                recall_order[text] = i
                recalled.append(text)
            recalled_texts = set(recalled)
            middle_segments = [*middle_segments, *((t, TIER_EPISODIC) for t in recalled)]
            if segment_weights is not None:
                floor = min(segment_weights, default=0.0) - 1.0
                segment_weights = [*segment_weights, *([floor] * len(recalled))]
        framing = fixed_framing + overhead * len(history)
        headers = 0
        if recalled_texts:
            headers = count_fn(f"{_EARLIER_HEADER}\n{_MEMORIES_HEADER}\n")
        if history_tokens is None:
            history_tokens = sum(count_fn(_message_text(m)) for m in history)
        thin_forms: list[str | None] | None = None
        if thin and self.thin_before_drop and middle_segments:
            # Keyframes are only worked out when the memories do not all fit.
            room = max_tok - framing - headers - system_tokens - history_tokens
            room -= end_tokens + len(middle_segments) + 8
            if sum(line_cost(text) for text, _ in middle_segments) > room:
                thin_forms = _thin_forms(
                    middle_segments,
                    recalled_texts,
                    self._conversation_counts(full_history),
                    user_message,
                    count_fn,
                    self.keyframe_tokens,
                    line_cost,
                )
                pairs = zip(thin_forms, middle_segments, strict=True)
                if any(form and form != text for form, (text, _) in pairs):
                    headers += count_fn(_EXCERPT_NOTE + "\n")
                else:
                    thin_forms = None
        earlier_kept = 0

        def render(kept: list[str], thinned: dict[str, str]) -> str | list[str]:
            """The final user message: recalled passages, memories, the question, directives."""
            nonlocal earlier_kept
            whole = {seg: thinned.get(seg, seg) for seg in kept}
            earlier = sorted(
                (seg for seg in kept if whole[seg] in recalled_texts),
                key=lambda seg: recall_order[whole[seg]],
            )  # chronological
            earlier_kept = len(earlier)
            lines = [seg for seg in kept if whole[seg] not in recalled_texts]
            if given_order:
                lines.sort(key=lambda seg: given_order.get(whole[seg], len(given_order)))
            if earlier:
                lines = [_EARLIER_HEADER, *earlier, _MEMORIES_HEADER, *lines]
            if thinned:
                lines = [_EXCERPT_NOTE, *lines]
            if not lines:
                return end_parts if in_parts else end_with_directives
            if in_parts:
                head = [f"{line}\n" for line in lines[:-1]]
                return [*head, lines[-1] + "\n\n", *end_parts]
            return "\n".join(lines) + "\n\n" + end_with_directives

        # The prefix (system message and history) is counted message by message, exactly as
        # it is sent; the firewall packs the tail into what the prefix leaves and counts
        # the finished tail once.
        firewall = PreFlightContextFirewall(
            max_tokens=max_tok - framing - system_tokens - history_tokens, count_fn=count_fn
        )
        tail, ref_map_kept, tail_tokens, fw_stats = firewall.run(
            core_text="",
            end_text="",  # the end block is in what render returns and in the reserve
            middle_segments=middle_segments,
            ref_map=ref_map,
            segment_weights=segment_weights,
            tier_shares=tier_shares,
            pinned_tiers=pinned_tiers,
            thin_segments=thin_forms,
            render=render,
            reserve_tokens=headers + end_tokens,
            segment_count=line_cost,
        )
        fw_stats.pop("kept_segments", None)
        fw_stats.pop("thinned_segments", None)
        # Some providers reject an empty system message.
        system = [{"role": "system", "content": system_text}] if system_text.strip() else []
        messages = [*system, *history]
        messages.append({"role": "user", "content": tail})
        prefix_tokens = system_tokens + history_tokens
        cache_meta: dict[str, Any] = {
            "memory_placement": MEMORY_PLACEMENT_END,
            "stable_prefix_messages": len(system) + len(history),
            "history_messages_dropped": history_dropped,
        }
        if recalled_texts:
            cache_meta["history_messages_recalled"] = earlier_kept
        if index_lines:
            cache_meta["history_index_lines"] = len(index_lines)
            cache_meta["history_index_tokens"] = system_tokens - core_tokens
            cache_meta["core_chars"] = len(core_text)
        if framing:
            cache_meta["message_framing_tokens"] = framing
        if prefix_tokens + tail_tokens + framing > max_tok:
            logger.warning(
                "memory_placement=end: system+history+end (%s tokens) exceeds max_tokens (%s); "
                "shorten conversation_messages or enable trim_history",
                prefix_tokens + tail_tokens,
                max_tok,
            )
            cache_meta["budget_exceeded"] = True
        pricing = (get_cache_pricing(self.model_id) or CachePricing()) if caching else None
        # The provider's minimum is in its own tokens.
        minimum = pricing.min_tokens / token_scale if pricing is not None else 0.0
        if pricing is not None and prefix_tokens >= minimum:
            cache_meta.update(
                prompt_cache_used=True,
                cacheable_prefix_tokens=prefix_tokens,
                dynamic_tail_tokens=tail_tokens,
            )
        if pricing is not None:
            # Where a cached prefix ends: after the core (stable for the whole conversation)
            # and after the history (stable until the next cut).
            points: list[dict[str, Any]] = []
            if core_text.strip() and core_tokens >= minimum:
                points.append(
                    {
                        "message_index": 0,
                        "chars": len(core_text),
                        "tokens": core_tokens,
                        "reason": "core",
                    }
                )
            if history and prefix_tokens >= minimum:
                # Mark when this history is expected to be read before the next cut, or when
                # an earlier request since the cut already wrote it: then the marker reads
                # that entry and only writes the turn added since.
                pays = not until_cut or any(_history_cache_pays(pricing, t) for t in until_cut)
                if until_cut:
                    cache_meta["history_cache_expected_reads"] = round(until_cut[-1], 2)
                if pays:
                    points.append(
                        {
                            "message_index": len(messages) - 2,
                            "tokens": prefix_tokens,
                            "reason": "history",
                        }
                    )
            cache_meta["cache_mode"] = pricing.mode
            cache_meta["cache_breakpoints"] = points
        return messages, cache_meta, ref_map_kept, prefix_tokens + tail_tokens, fw_stats

    def _scale_state(self, exact_tokenizer: bool = False) -> _TokenScale:
        """Calibration state for this model. With a tokenizer registry the counts are taken
        to be the provider's own, so the seed is 1."""
        key = (self.model_id, exact_tokenizer)
        state = self._scales.get(key)
        if state is None:
            seed = 1.0 if exact_tokenizer else get_token_scale(self.model_id)
            state = self._scales[key] = _TokenScale(mean=seed, applied=seed)
        return state

    def observe(self, result: CompileResult, actual_prompt_tokens: int) -> float:
        """Report what the provider billed for a compiled prompt; returns the scale now in use.

        ``actual_prompt_tokens`` is the provider's prompt-token count for the request built
        from ``result``: everything on the input side, cached or not (for Anthropic,
        ``input_tokens + cache_read_input_tokens + cache_creation_input_tokens``). The
        ratio to Membrane's own count (content plus message framing) is smoothed per model,
        a margin of twice its spread is added, and later compiles divide the budget by the
        result. Tool definitions and anything else you add to the request count towards the
        ratio, which is what keeps the whole request inside the budget.

        The scale in use changes in steps of 0.05, rises at once and falls only when it is
        0.1 too high, because each change moves the history cut once. Nothing is stored
        outside this compiler and nothing is sent anywhere.
        """
        md = result.metadata
        counted = int(md.get("token_count", 0)) + int(md.get("message_framing_tokens", 0))
        state = self._scale_state(bool(md.get("token_scale_exact_tokenizer")))
        if counted > 0 and actual_prompt_tokens > 0:
            state.add(actual_prompt_tokens / counted)
        return self.token_scale if self.token_scale is not None else state.applied

    def _conversation_counts(
        self, history: Sequence[dict[str, Any]]
    ) -> tuple[dict[str, int], int]:
        """Sentence counts of this conversation, updated with the messages added since the
        last compile. Conversations are told apart by their first message."""
        first = history[0].get("content") if history else None
        key = first if isinstance(first, str) else None
        counts = self._conversations.get(key)
        if counts is None:
            while len(self._conversations) >= _INDEX_CACHE_SIZE:
                self._conversations.pop(next(iter(self._conversations)))
            counts = self._conversations[key] = _ConversationCounts()
        return counts.update(history)

    def _big_count(self, text: str, count_fn: Callable[[str], int], count_key: Any) -> int:
        """Token count of a long text that repeats across compiles (the core, the system
        message), remembered on the compiler: such texts are over the shared count cache's
        size limit and would otherwise be tokenised again on every turn."""
        key = (count_key, text)
        tokens = self._big_counts.get(key)
        if tokens is None:
            tokens = count_fn(text)
            while len(self._big_counts) >= _INDEX_CACHE_SIZE:
                self._big_counts.pop(next(iter(self._big_counts)))
            self._big_counts[key] = tokens
        return tokens

    def _index_lines(
        self,
        evicted: tuple[tuple[Any, ...], ...],
        full_history: list[dict[str, Any]],
        sizes: list[int],
        step: int,
        count_fn: Callable[[str], int],
        budget: Callable[[], int],
        count_key: Any,
    ) -> tuple[str, ...]:
        """Eviction index for the current cut. Built once per cut, and from the previous
        cut's index when that is still at hand, so a cut only scores the messages it drops.
        Its token budget is asked for when the cut is first seen and kept until the next cut,
        so the block stays byte-stable between cuts whatever later turns bring."""
        cap = self.keyframe_tokens
        key = (evicted, step, cap, count_key)
        state = self._index_cache.get(key)
        if state is None:
            starts = _cut_starts(full_history, sizes, step, len(evicted))
            # Continue from the latest earlier cut still at hand. Earlier cuts stay cached:
            # a turn with a long question can push the cut ahead for that turn only.
            state, previous, done = None, 0, 0
            for position in range(len(starts) - 2, -1, -1):
                earlier = self._index_cache.get((evicted[: starts[position]], step, cap, count_key))
                if earlier is not None:
                    state, previous, done = earlier.copy(), starts[position], position + 1
                    break
            if state is None:
                state = _IndexState()
            for start in starts[done:]:
                state.add_cut(evicted[previous:start], count_fn, cap)
                previous = start
            while len(self._index_cache) >= _INDEX_CACHE_SIZE:
                self._index_cache.pop(next(iter(self._index_cache)))
            self._index_cache[key] = state
            state.budget = budget()
        return state.within(state.budget, count_fn)

    def _recall_index(
        self,
        key: tuple[tuple[Any, ...], ...],
        evicted: list[dict[str, Any]],
    ) -> tuple[list[tuple[str, frozenset[str]]], dict[str, list[int]], int]:
        """Passages of the dropped history, their postings, and how many passages count.
        Kept per conversation (told apart by its first dropped message) and grown by the
        messages each cut adds, so a cut only splits what it drops."""
        index = self._recall_cache.get(key[0])
        if index is None or not index.covers(key):
            self._recall_cache.pop(key[0], None)
            while len(self._recall_cache) >= _INDEX_CACHE_SIZE:
                self._recall_cache.pop(next(iter(self._recall_cache)))
            index = self._recall_cache[key[0]] = _RecallIndex()
        return index.upto(key, evicted)

    def _policy_max_tokens(self, policy_kw: dict[str, Any]) -> int:
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
        price_per_1k_input_tokens: float | None = None,
        target_token_budget: int | None = None,
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
        price_per_1k_input_tokens: float | None = None,
        target_token_budget: int | None = None,
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
        last_assistant_turn: str | None = None,
        conversation_messages: list[dict[str, Any]] | None = None,
        agent_persona_prompt: str | None = None,
        dynamic_directives: Sequence[str] | None = None,
        middle_extras: Sequence[tuple[str, int]] | None = None,
        compress_sync: Callable[[str, int, str], str] | None = None,
        tier_by_memory_id: dict[int, int] | None = None,
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
        trace_exporters: list[Any] | None = None,
        enable_lineage: bool | None = None,
        enable_profiler: bool | None = None,
        metrics_collector: Any = None,
        lineage_tracker: Any = None,
        ranking_weights: Any = None,
        include_ranking_diagnostics: bool | None = None,
    ) -> CompileResult:
        """Synchronous compile: optional sync ``compress_sync`` for sacrifice zone; optional
        modules via ``profile``.

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

        latency_prof: Any | None = None
        profiler: LatencyProfiler | None = None
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

        policy_kw: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "model_id": self.model_id,
        }
        # Only policies read the estimate.
        est_tok = estimate_memory_tokens(memories) if prof.policies is not None else 0
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
        scale_state = self._scale_state(prof.tokenizer_registry is not None)
        token_scale = self.token_scale if self.token_scale is not None else scale_state.applied
        # The budget is in the provider's tokens; counting is in ours.
        counted_budget = max(1, int(max_tok / token_scale))

        mems: list[Any] = list(memories)
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
        segment_weights: list[float] | None = None
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
        # Only a compressor needs the size of the joined middle; skip tokenising it otherwise.
        compressing = prof.compression is not None or compress_sync is not None
        middle_tokens = _count_ctx(raw_middle_text) if compressing else 0
        compression_strategy_used: str | None = None
        compressed_middle: str | None = None
        compression_details: dict[str, Any] = {}
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
                    max_tok=counted_budget,
                    token_scale=token_scale,
                    use_split=use_prompt_caching_split,
                    count_fn=_count_ctx,
                    tier_shares=tier_shares,
                    pinned_tiers=pinned_tiers,
                    thin=compressed_middle is None,
                    count_key=id(prof.tokenizer_registry) if prof.tokenizer_registry else None,
                )
            )
        if compressed_middle is not None and fw_stats.get("segments_kept"):
            # Compressed text keeps [Mem:N] references; resolve every one it still carries.
            ref_map_kept = _ref_map_from_text(compressed_middle, ref_map)
        stale_survivors = _stale_survivor_memory_ids(list(retrieved), ref_map_kept)

        ref_map_type_kept: dict[int, str] = {}
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
                "token_scale": token_scale,
                "token_scale_samples": scale_state.samples,
                **({"tokens_budget_counted": counted_budget} if counted_budget != max_tok else {}),
                **(
                    {"token_scale_exact_tokenizer": True}
                    if prof.tokenizer_registry is not None
                    else {}
                ),
                "segments_offered": int(fw_stats.get("segments_offered", segments_offered_pre)),
                "segments_kept": int(fw_stats.get("segments_kept", 0)),
                "segments_thinned": int(fw_stats.get("segments_thinned", 0)),
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

        metrics_extra: dict[str, Any] = {}
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
        last_assistant_turn: str | None = None,
        conversation_messages: list[dict[str, Any]] | None = None,
        agent_persona_prompt: str | None = None,
        dynamic_directives: Sequence[str] | None = None,
        middle_extras: Sequence[tuple[str, int]] | None = None,
        compress_async: Callable[[str, int, str], Awaitable[str]] | None = None,
        compress_sync: Callable[[str, int, str], str] | None = None,
        redis_client: Any | None = None,
        conversation_id: str | None = None,
        tier_by_memory_id: dict[int, int] | None = None,
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
        trace_exporters: list[Any] | None = None,
        enable_lineage: bool | None = None,
        enable_profiler: bool | None = None,
        metrics_collector: Any = None,
        lineage_tracker: Any = None,
        ranking_weights: Any = None,
        include_ranking_diagnostics: bool | None = None,
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

        latency_prof: Any | None = None
        profiler: LatencyProfiler | None = None
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

        policy_kw: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "model_id": self.model_id,
        }
        # Only policies read the estimate.
        est_tok = estimate_memory_tokens(memories) if prof.policies is not None else 0
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
        scale_state = self._scale_state(prof.tokenizer_registry is not None)
        token_scale = self.token_scale if self.token_scale is not None else scale_state.applied
        # The budget is in the provider's tokens; counting is in ours.
        counted_budget = max(1, int(max_tok / token_scale))

        mems: list[Any] = list(memories)
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

        segment_weights: list[float] | None = None
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
        compression_strategy_used: str | None = None
        compressed_middle: str | None = None
        compression_details: dict[str, Any] = {}
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
            compressed: str | None = None
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
                    max_tok=counted_budget,
                    token_scale=token_scale,
                    use_split=use_prompt_caching_split,
                    count_fn=_count_ctx,
                    tier_shares=tier_shares,
                    pinned_tiers=pinned_tiers,
                    thin=compressed_middle is None,
                    count_key=id(prof.tokenizer_registry) if prof.tokenizer_registry else None,
                )
            )
        if compressed_middle is not None and fw_stats.get("segments_kept"):
            # Compressed text keeps [Mem:N] references; resolve every one it still carries.
            ref_map_kept = _ref_map_from_text(compressed_middle, ref_map)
        stale_survivors = _stale_survivor_memory_ids(list(retrieved), ref_map_kept)

        ref_map_type_kept: dict[int, str] = {}
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
                "token_scale": token_scale,
                "token_scale_samples": scale_state.samples,
                **({"tokens_budget_counted": counted_budget} if counted_budget != max_tok else {}),
                **(
                    {"token_scale_exact_tokenizer": True}
                    if prof.tokenizer_registry is not None
                    else {}
                ),
                "segments_offered": int(fw_stats.get("segments_offered", segments_offered_pre)),
                "segments_kept": int(fw_stats.get("segments_kept", 0)),
                "segments_thinned": int(fw_stats.get("segments_thinned", 0)),
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

        metrics_extra: dict[str, Any] = {}
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


# The usual shortest cacheable prefix; the capability map has each model's own minimum.
MIN_CACHEABLE_PREFIX_TOKENS = 1024


def _build_output_messages(
    model_id: str | None,
    final_text: str,
    use_split: bool,
    prefix_text: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One system message; report the byte-stable prefix (``prefix_text``) when it is cacheable.

    ``cacheable_prefix_chars`` lets callers split the content for explicit cache markers
    (for example :func:`membrane.cache_format.anthropic_system_blocks`).
    """
    cache_meta: dict[str, Any] = {}
    messages: list[dict[str, Any]] = [{"role": "system", "content": final_text}]
    if not (model_id and use_split and has_prompt_caching(model_id)):
        return messages, cache_meta
    prefix = prefix_text if prefix_text.strip() and final_text.startswith(prefix_text) else ""
    prefix_tokens = count_tokens(prefix)
    pricing = get_cache_pricing(model_id) or CachePricing()
    if prefix_tokens >= pricing.min_tokens:
        cache_meta = {
            "prompt_cache_used": True,
            "cacheable_prefix_chars": len(prefix),
            "cacheable_prefix_tokens": prefix_tokens,
            "dynamic_tail_tokens": count_tokens(final_text[len(prefix):]),
            "cache_mode": pricing.mode,
            "cache_breakpoints": [
                {
                    "message_index": 0,
                    "chars": len(prefix),
                    "tokens": prefix_tokens,
                    "reason": "core",
                }
            ],
        }
    return messages, cache_meta


def compile_anonymous(
    *,
    user_message: str,
    conversation_messages: list[dict[str, Any]],
    max_tokens: int,
    model_id: str | None = None,
    core_content: str = "",
    agent_persona_prompt: str | None = None,
    use_prompt_caching_split: bool = True,
    metrics_callback: MetricsCallback | None = None,
    memory_placement: str = MEMORY_PLACEMENT_SYSTEM,
    trim_history: bool = True,
    history_share: float = DEFAULT_HISTORY_SHARE,
    recall_evicted_history: bool = True,
    eviction_index: bool = True,
    index_share: float = DEFAULT_INDEX_SHARE,
    keyframe_tokens: int = DEFAULT_KEYFRAME_TOKENS,
    message_overhead_tokens: int = DEFAULT_MESSAGE_OVERHEAD_TOKENS,
    token_scale: float | None = None,
) -> CompileResult:
    """No long-term memory: core + conversation + user message.

    ``memory_placement="system"`` (default) puts everything into one system message and
    never trims history; an overrun sets ``metadata["budget_exceeded"]``. It honours
    ``message_overhead_tokens`` (framing is reserved in the budget) and ``token_scale``
    (the budget is divided by it; ``None`` takes the capability map's seed for the model).

    ``memory_placement="end"`` gives the layout :class:`ContextCompiler` uses by default:
    system message, history as chat messages, the question last. It honours every history
    option of the compiler (``trim_history``, ``history_share``, ``recall_evicted_history``,
    ``eviction_index``, ``index_share``, ``keyframe_tokens``) and reports the same metadata,
    cache breakpoints included. Each call is independent, so for a long conversation a
    ``ContextCompiler`` you keep is faster: it remembers what it has already counted.
    """
    if memory_placement not in MEMORY_PLACEMENTS:
        raise ValueError(
            f"memory_placement must be one of {MEMORY_PLACEMENTS}, got {memory_placement!r}"
        )
    t0 = time.perf_counter()
    if memory_placement == MEMORY_PLACEMENT_END:
        compiler = ContextCompiler(
            max_tokens=max_tokens,
            model_id=model_id,
            core_content=core_content,
            include_interrupt_instruction=False,
            memory_placement=MEMORY_PLACEMENT_END,
            trim_history=trim_history,
            history_share=history_share,
            recall_evicted_history=recall_evicted_history,
            eviction_index=eviction_index,
            index_share=index_share,
            keyframe_tokens=keyframe_tokens,
            message_overhead_tokens=message_overhead_tokens,
            token_scale=token_scale,
        )
        result = compiler.compile(
            memories=[],
            user_message=user_message,
            conversation_messages=conversation_messages,
            agent_persona_prompt=agent_persona_prompt,
            use_prompt_caching_split=use_prompt_caching_split,
        )
        md = result.metadata
        md["security_flag"] = SecurityFlag.ANONYMOUS_MODE.value
        _emit_compile_metrics(
            metrics_callback,
            build_compile_metrics(
                wall_time_ms=(time.perf_counter() - t0) * 1000.0,
                token_count=int(md["token_count"]),
                tokens_trimmed=int(md["tokens_trimmed"]),
                tokens_budget=int(md["tokens_budget"]),
                segments_offered=int(md["segments_offered"]),
                segments_kept=int(md["segments_kept"]),
                ref_map_size=0,
                cache_hit=bool(md.get("prompt_cache_used")),
                is_anonymous=True,
                model_id=model_id,
                user_id=None,
                conversation_id=None,
            ),
        )
        return result

    cap = max_tokens
    if model_id:
        mc = lookup_context_cap(model_id)
        if mc:
            cap = min(cap, mc)
    scale = token_scale if token_scale is not None else get_token_scale(model_id)
    if scale <= 0:
        raise ValueError(f"token_scale must be positive, got {token_scale!r}")
    # One message plus reply priming; the budget is in the provider's tokens.
    framing = 2 * max(0, message_overhead_tokens)
    counted_cap = max(1, int(cap / scale))
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
    fw_stats: dict[str, Any] = {
        "segments_offered": 0,
        "segments_kept": 0,
        "tokens_trimmed": 0,
    }
    if total + framing > counted_cap:
        firewall = PreFlightContextFirewall(max_tokens=counted_cap - framing)
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
            "token_scale": scale,
            **({"tokens_budget_counted": counted_cap} if counted_cap != cap else {}),
            "segments_offered": int(fw_stats.get("segments_offered", 0)),
            "segments_kept": int(fw_stats.get("segments_kept", 0)),
            **({"message_framing_tokens": framing} if framing else {}),
            **cache_meta,
        },
    )
    if total + framing > counted_cap:
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
    messages_list: list[dict[str, Any]],
    user_id: str | None,
    max_tokens: int,
    model_id: str | None = None,
    core_content: str = "",
    agent_persona_prompt: str | None = None,
    dynamic_directives: Sequence[str] | None = None,
    retrieved_objects: list[tuple[Any, int]] | None = None,
    current_stress: float | None = None,
    redis_client: Any | None = None,
    conversation_id: str | None = None,
    compress_fn: Callable[[str, int, str], Awaitable[str]] | None = None,
    middle_segments_prefix: list[tuple[str, int]] | None = None,
    lod_focus_retrieve: Callable[[], Awaitable[list[tuple[Any, int]]]] | None = None,
    persist_ref_map_to_redis: bool = False,
    redis_url_for_ref_map: str | None = None,
    use_prompt_caching_split: bool = True,
    metrics_callback: MetricsCallback | None = None,
    memory_placement: str = MEMORY_PLACEMENT_END,
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
    trace_exporters: list[Any] | None = None,
    enable_lineage: bool | None = None,
    enable_profiler: bool | None = None,
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

    retrieved: list[tuple[Any, int]] = list(retrieved_objects or [])
    if attention_shift and lod_focus_retrieve and retrieved:
        try:
            focus_list = await lod_focus_retrieve()
            if focus_list:
                retrieved = _lod_focus_merge(retrieved, focus_list)
        except Exception as e:
            logger.debug("LOD focus retrieval failed: %s", e)

    memories = [o for o, _ in retrieved]
    tier_by_memory_id: dict[int, int] | None = (
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
