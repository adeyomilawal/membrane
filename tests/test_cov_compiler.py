"""Coverage-oriented behaviour tests for ``membrane.compiler``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import membrane.compiler as comp
from membrane.compiler import (
    ContextCompiler,
    SecurityFlag,
    compile_prompt_async,
    compile_quote_from_result,
)
from membrane.compiler import CompileResult
from membrane.exceptions import FeatureNotAvailable
from membrane.firewall import TIER_EPISODIC, TIER_PROCEDURAL, TIER_SEMANTIC, count_tokens
from membrane.memory_protocol import MemoryItem
from membrane.profile import MembraneProfile
from membrane.runtime.signals import SignalBus, SignalKind


def _cc(max_tokens: int = 2000, **kw: Any) -> ContextCompiler:
    return ContextCompiler(max_tokens=max_tokens, include_interrupt_instruction=False, **kw)


def _items(n: int, kind: str = "semantic", prio: int = 2) -> list[MemoryItem]:
    return [
        MemoryItem(id=f"m{i}", content=f"fact number {i}", memory_type=kind, priority=prio)  # type: ignore[arg-type]
        for i in range(n)
    ]


def _all_text(result: CompileResult) -> str:
    return "\n".join(str(m["content"]) for m in result.messages)


@dataclass
class FakeCompressionResult:
    compressed: str
    strategy_used: str = "fake"
    details: Any = field(default_factory=dict)
    original_tokens: int = 100
    compressed_tokens: int = 40


class FakeCompression:
    def __init__(self, result: FakeCompressionResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self.model_ids: list[str] = []

    def set_model_id(self, model_id: str) -> None:
        self.model_ids.append(model_id)

    def compress(self, text: str, *, target_tokens: int, query_context: str):
        self.calls.append(
            {"text": text, "target_tokens": target_tokens, "query_context": query_context}
        )
        return self.result


class FakeLineage:
    def __init__(self) -> None:
        self.registered: list[tuple[str, str, str]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.finalized: dict[str, Any] | None = None

    def register_segment(self, sid, content, *, source_origin, source_metadata):
        self.registered.append((sid, content, source_origin))

    def record_event(self, segment_id, **kw):
        self.events.append((segment_id, kw))

    def finalize(self, *, included_ids, positions):
        self.finalized = {"included": set(included_ids), "positions": positions}
        return SimpleNamespace(
            to_dict=lambda: {"included": sorted(included_ids)},
            explain=lambda seg: f"explained {seg}",
        )


# --- helpers ----------------------------------------------------------------


def test_jta_lineage_noop_without_recorder():
    comp._record_jta_lineage_from_compression(
        None, compression_strategy_used="jta", compression_details={}
    )
    comp._record_jta_lineage_from_compression(
        object(), compression_strategy_used="jta", compression_details={}
    )


def test_jta_lineage_records_aliases_and_cache_event():
    lin = FakeLineage()
    details = {
        "alias_decisions": [
            {"phrase": "foo bar", "alias": "A1", "token_savings": 7, "frequency": 3},
            "not-a-dict",
        ],
        "aliases_reused": 2,
        "incremental_cache_hit": True,
        "delta_chars_scanned": 10,
        "new_aliases_mapped": 1,
    }
    comp._record_jta_lineage_from_compression(
        lin, compression_strategy_used="jta_deterministic", compression_details=details
    )
    actions = [kw["action"] for _, kw in lin.events]
    assert actions == ["jta_alias_map", "jta_incremental_cache"]
    alias_detail = lin.events[0][1]["detail"]
    assert alias_detail["alias"] == "A1"
    assert alias_detail["roi"] == 7
    assert lin.events[0][0] == "middle_compressed"
    assert lin.events[1][1]["detail"]["aliases_reused"] == 2
    assert lin.events[1][1]["detail"]["incremental_cache_hit"] is True


def test_jta_lineage_nested_details_recorded_even_for_non_jta_strategy():
    lin = FakeLineage()
    details = {"jta_lineage_details": {"alias_decisions": [{"phrase": "p", "alias": "Z"}]}}
    comp._record_jta_lineage_from_compression(
        lin, compression_strategy_used="llm", compression_details=details, segment_id="seg"
    )
    assert [(s, kw["action"]) for s, kw in lin.events] == [("seg", "jta_alias_map")]


def test_jta_lineage_skips_non_jta_strategy():
    lin = FakeLineage()
    comp._record_jta_lineage_from_compression(
        lin,
        compression_strategy_used="llm",
        compression_details={"alias_decisions": [{"phrase": "p"}]},
    )
    comp._record_jta_lineage_from_compression(
        lin, compression_strategy_used=None, compression_details={}
    )
    assert lin.events == []


def test_stage_without_profiler_is_nullcontext():
    with comp._stage(None, "x"):
        pass


def test_ranking_diagnostics_disabled_returns_none():
    prof = SimpleNamespace(include_ranking_diagnostics=False)
    assert comp._build_ranking_diagnostics_for_compile(prof, [], None, []) is None
    prof2 = SimpleNamespace(include_ranking_diagnostics=True)
    assert (
        comp._build_ranking_diagnostics_for_compile(
            prof2, [], None, [], include_ranking_diagnostics=False
        )
        is None
    )


def test_merge_debug_trace_filters_policy_keys():
    out = comp._merge_compile_debug_trace(
        {"policy_x": 1, "max_tokens": 5, "model_id": "m", "other": 2}, True, {"d": 1}
    )
    assert out["policy_kw"] == {"policy_x": 1, "max_tokens": 5}
    assert out["segment_weights_used"] is True
    assert out["ranking_diagnostics"] == {"d": 1}
    assert "ranking_diagnostics" not in comp._merge_compile_debug_trace({}, False, None)


def test_trace_exporters_failures_are_swallowed(caplog):
    seen = []
    good = SimpleNamespace(export_compile_run=seen.append)

    def boom(_r):
        raise RuntimeError("exporter down")

    bad = SimpleNamespace(export_compile_run=boom)
    prof = SimpleNamespace(trace_exporters=[object(), bad, good])
    sentinel = object()
    with caplog.at_level(logging.DEBUG, logger="membrane.compiler"):
        comp._run_trace_exporters(prof, sentinel)  # type: ignore[arg-type]
    assert seen == [sentinel]
    assert any("exporter down" in r.getMessage() for r in caplog.records)


def test_emit_metrics_callback_failure_warns(caplog):
    class Bad:
        def on_compile(self, metrics):
            raise RuntimeError("cb broke")

    with caplog.at_level(logging.WARNING, logger="membrane.compiler"):
        comp._emit_compile_metrics(Bad(), object())  # type: ignore[arg-type]
    assert any(
        r.levelno == logging.WARNING and "metrics_callback failed" in r.getMessage()
        for r in caplog.records
    )
    comp._emit_compile_metrics(None, object())  # type: ignore[arg-type]


# --- CompileResult ----------------------------------------------------------


def test_compile_result_helpers():
    r = CompileResult(messages=[], ref_map={})
    assert r.explain("x") == "No lineage_report on this CompileResult."
    with pytest.raises(FeatureNotAvailable):
        r.diff(r)
    r.lineage_report = SimpleNamespace(
        explain=lambda s: f"why {s}", to_dict=lambda: {"k": 1}
    )
    r.latency_profile = SimpleNamespace(to_dict=lambda: {"total": 1})
    r.assertion_report = SimpleNamespace(passed=True)
    r.dedup_report = SimpleNamespace(removed=2)
    assert r.explain("seg") == "why seg"
    d = r.to_debug_dict()
    assert d["lineage_report"] == {"k": 1}
    assert d["latency_profile"] == {"total": 1}
    assert d["assertion_report"] == {"passed": True}
    assert d["dedup_report"] == {"removed": 2}


# --- quotes / trimming strategies -------------------------------------------


def test_trimming_strategies_dropped_segments():
    out = comp._build_trimming_strategies(
        total_tokens=90, tokens_trimmed=50, trimmed_count=2, tokens_budget=100,
        target_token_budget=None,
    )
    assert len(out) == 3
    assert "dropped 2 middle segment(s)" in out[0]
    assert "budget 100" in out[0]


def test_trimming_strategies_compressed_and_over_target():
    out = comp._build_trimming_strategies(
        total_tokens=120, tokens_trimmed=10, trimmed_count=0, tokens_budget=200,
        target_token_budget=100,
    )
    assert "compressed or trimmed by 10" in out[0]
    assert "20 token(s) above target_token_budget=100" in out[1]


def test_trimming_strategies_no_trim():
    out = comp._build_trimming_strategies(
        total_tokens=10, tokens_trimmed=0, trimmed_count=0, tokens_budget=100,
        target_token_budget=None,
    )
    assert out == ["No trimming occurred; context fits the current budget."]
    assert comp._build_trimming_strategies(
        total_tokens=0, tokens_trimmed=0, trimmed_count=0, tokens_budget=1,
        target_token_budget=None,
    ) == []


def test_quote_negative_price_rejected_and_fallback_budget():
    r = CompileResult(messages=[], ref_map={}, metadata={"token_count": 2000})
    with pytest.raises(ValueError):
        compile_quote_from_result(r, price_per_1k_input_tokens=-1.0)
    q = compile_quote_from_result(r, price_per_1k_input_tokens=0.5, compiler_max_tokens=777)
    assert q.tokens_budget == 777
    assert q.estimated_cost_usd == pytest.approx(1.0)


def test_estimate_real_compile_quote_with_trimming():
    cc = _cc(300)
    q = cc.estimate(
        memories=_items(80), user_message="hi", price_per_1k_input_tokens=1.0,
        target_token_budget=10,
    )
    assert q.total_tokens == q.compile_result.metadata["token_count"] <= 300
    assert q.segments_offered == 80
    assert q.segments_kept < 80
    assert q.trimmed_count == q.segments_offered - q.segments_kept
    assert q.estimated_cost_usd == pytest.approx(q.total_tokens / 1000.0)
    assert any("above target_token_budget=10" in s for s in q.trimming_strategies)


async def test_estimate_async_matches_sync():
    cc = _cc(2000)
    sync = cc.estimate(memories=_items(3), user_message="hello")
    aq = await cc.estimate_async(
        memories=_items(3), user_message="hello", price_per_1k_input_tokens=2.0
    )
    assert aq.total_tokens == sync.total_tokens
    assert aq.estimated_cost_usd == pytest.approx(aq.total_tokens * 2.0 / 1000.0)
    assert aq.tokens_budget == 2000


# --- history normalisation / trimming ---------------------------------------


def test_normalize_history_variants():
    out = comp._normalize_history(
        [
            {"content": "no role"},
            {"role": "assistant", "content": "", "stage3_synthesis": "synth"},
            {"role": "assistant", "content": {"text": "dict text"}},
            {"role": "tool", "content": "r", "tool_call_id": "c1", "stage3": "x"},
        ]
    )
    assert out[0] == {"content": "no role", "role": "user"}
    assert out[1]["content"] == "synth" and "stage3_synthesis" not in out[1]
    assert out[2]["content"] == "dict text"
    assert out[3] == {"role": "tool", "content": "r", "tool_call_id": "c1"}


def test_message_text_handles_non_string_content_and_extras():
    assert comp._message_text({"role": "assistant", "content": None}) == ""
    txt = comp._message_text(
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}], "name": "bob"}
    )
    assert '"type": "text"' in txt and '"name": "bob"' in txt


def test_trim_history_chunked_edge_cases():
    count = count_tokens
    hist = [{"role": "user", "content": "word " * 50}, {"role": "assistant", "content": "ok"}]
    assert comp._trim_history_chunked(hist, 10_000, count) == (hist, 0)
    assert comp._trim_history_chunked([], 5, count) == ([], 0)
    kept, dropped = comp._trim_history_chunked(hist, 0, count)
    assert kept == [] and dropped == 2


def test_trim_history_chunked_starts_on_user_message():
    count = count_tokens
    hist = []
    for i in range(10):
        hist.append({"role": "user", "content": f"q{i} " + "word " * 40})
        hist.append({"role": "assistant", "content": f"a{i} " + "word " * 40})
    total = sum(count(comp._message_text(m)) for m in hist)
    kept, dropped = comp._trim_history_chunked(hist, total // 2, count)
    assert dropped > 0
    assert kept == hist[dropped:]
    assert kept[0]["role"] == "user"
    assert sum(count(comp._message_text(m)) for m in kept) <= total // 2


def test_flatten_messages_for_display():
    text = comp._flatten_messages_for_display(
        [
            {"role": "user", "content": {"text": "hello"}},
            {"role": "assistant", "stage3_synthesis": "synth"},
            {"content": {"other": 1}},
        ]
    )
    assert text.splitlines()[0] == "user: hello"
    assert text.splitlines()[1] == "assistant: synth"
    assert text.splitlines()[2].startswith("unknown: ")


def test_last_turn_and_user_helpers():
    msgs = [
        {"role": "user", "content": {"text": "u1"}},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": {"text": "dict answer"}},
        {"role": "user", "content": "u3"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "   "},
    ]
    assert comp._last_turn_from_messages(msgs) == "dict answer"
    assert comp._last_turn_from_messages([{"role": "user", "content": "x"}]) is None
    assert comp._last_user_messages(msgs) == ["u2", "u3", "u4"]
    assert comp._last_user_messages([]) == []


def test_retrieval_summary_variants():
    assert "No long-term" in comp._build_retrieval_summary(0, 0, 0, "q")
    s = comp._build_retrieval_summary(2, 1, 1, "a" * 80)
    assert s == "Remembered 2 episodes, 1 semantic fact, 1 procedure regarding " + "a" * 50 + "…."
    assert "regarding your query" in comp._build_retrieval_summary(1, 0, 0, "   ")
    assert "1 episode " in comp._build_retrieval_summary(1, 0, 0, "short")


def test_tier_for_object_fallbacks():
    assert comp._tier_for_object(SimpleNamespace(priority="bad", memory_type="procedural")) == (
        TIER_PROCEDURAL
    )
    assert comp._tier_for_object(SimpleNamespace(priority="3")) == TIER_EPISODIC
    assert comp._tier_for_object(SimpleNamespace(memory_type="semantic")) == TIER_SEMANTIC
    assert comp._tier_for_object(SimpleNamespace(memory_type="episodic")) == TIER_EPISODIC


def test_stale_survivor_ids_only_kept_and_marked():
    stale = SimpleNamespace(id="a", _membrane_is_stale=True)
    fresh = SimpleNamespace(id="b", _membrane_is_stale=False)
    unkept = SimpleNamespace(id="c", _membrane_is_stale=True)
    out = comp._stale_survivor_memory_ids(
        [(stale, 2), (fresh, 2), (unkept, 2)], {1: "a", 2: "b"}
    )
    assert out == ["a"]


def test_lod_focus_merge_caps_dedupes_and_prepends():
    def m(i):
        return SimpleNamespace(id=i)

    focus = [(m("f1"), TIER_EPISODIC), (m("f1"), TIER_EPISODIC)]
    focus += [(m(f"e{i}"), TIER_EPISODIC) for i in range(3)]
    focus += [(m("p1"), TIER_PROCEDURAL), (m("p2"), TIER_PROCEDURAL), (m("odd"), 99)]
    main = [(m("f1"), TIER_EPISODIC), (m("keep"), TIER_SEMANTIC)]
    merged = comp._lod_focus_merge(main, focus)
    ids = [o.id for o, _ in merged]
    # 2 episodic cap (f1, e0), 1 procedural cap (p1), unknown tier skipped
    assert ids == ["f1", "e0", "p1", "keep"]


def test_policy_helpers():
    assert comp._policy_disabled_features(
        {"policy_disable": ["a", "b"], "policy_enable": ["b"]}
    ) == ["a"]
    shares, pinned = comp._policy_tier_settings(
        {
            "policy_budget_override": {"procedural": 0.5, "bogus": 1},
            "policy_pin_kinds": ["semantic", "x"],
        }
    )
    assert shares == {TIER_PROCEDURAL: 0.5}
    assert pinned == {TIER_SEMANTIC}
    assert comp._policy_tier_settings({}) == (None, None)


# --- constructor / budgets ---------------------------------------------------


def test_invalid_memory_placement_rejected():
    with pytest.raises(ValueError, match="memory_placement"):
        ContextCompiler(max_tokens=100, memory_placement="middle")


def test_effective_and_policy_max_tokens(monkeypatch):
    monkeypatch.setattr(comp, "lookup_context_cap", lambda mid: 500 if mid == "known" else None)
    assert ContextCompiler(max_tokens=1000, model_id="known").effective_max_tokens() == 500
    assert ContextCompiler(max_tokens=100, model_id="known").effective_max_tokens() == 100
    assert ContextCompiler(max_tokens=1000, model_id="other").effective_max_tokens() == 1000
    cc = ContextCompiler(max_tokens=1000, model_id="known")
    assert cc._policy_max_tokens({"policy_max_tokens": 9000}) == 500
    assert cc._policy_max_tokens({"policy_max_tokens": 200}) == 200
    assert cc._policy_max_tokens({"max_tokens": 300}) == 300
    assert cc._policy_max_tokens({"max_tokens": 1000}) == 500
    plain = ContextCompiler(max_tokens=1000)
    assert plain._policy_max_tokens({"policy_max_tokens": 4000}) == 4000


# --- compile(): plugin stages ------------------------------------------------


class FakePolicies:
    def __init__(self, **extra: Any) -> None:
        self.extra = extra
        self.seen: dict[str, Any] = {}

    def apply(self, *, query, memories, compiler_kwargs, estimated_tokens, model_id):
        self.seen = {"query": query, "n": len(memories), "est": estimated_tokens}
        return {**compiler_kwargs, **self.extra}


def test_policies_override_budget_and_appear_in_debug_trace():
    pol = FakePolicies(policy_max_tokens=250)
    r = _cc(5000).compile(memories=_items(60), user_message="q", policies=pol)
    assert pol.seen["n"] == 60 and pol.seen["query"] == "q" and pol.seen["est"] > 0
    assert r.metadata["tokens_budget"] == 250
    assert r.metadata["token_count"] <= 250
    assert r.debug_trace["policy_kw"]["policy_max_tokens"] == 250


def test_policy_disable_skips_reranker_and_compression():
    class Boom:
        def rerank(self, *a):
            raise AssertionError("reranker must not run")

        def compress(self, *a, **k):
            raise AssertionError("compression must not run")

    pol = FakePolicies(policy_disable=["reranker", "compression"])
    r = _cc().compile(
        memories=_items(3), user_message="q", policies=pol, reranker=Boom(), compression=Boom()
    )
    assert r.metadata["compression_strategy_used"] is None
    assert r.metadata["segments_kept"] == 3


def test_reranker_orders_memories_and_sets_weights():
    class Rev:
        def rerank(self, query, mems):
            return [(m, float(i)) for i, m in enumerate(reversed(mems))]

    r = _cc().compile(memories=_items(3), user_message="q", reranker=Rev())
    body = r.messages[-1]["content"]
    # Rev gives the highest weight to the originally-first memory; local refs follow rerank order.
    assert body.index("fact number 0") < body.index("fact number 1") < body.index("fact number 2")
    assert r.debug_trace["segment_weights_used"] is True
    assert r.ref_map == {1: "m2", 2: "m1", 3: "m0"}


def test_reranker_with_middle_extras_ranks_extras_first():
    class Same:
        def rerank(self, query, mems):
            return [(m, 1.0) for m in mems]

    r = _cc().compile(
        memories=_items(2), user_message="q", reranker=Same(),
        middle_extras=[("EXTRA_SEGMENT", TIER_SEMANTIC)],
    )
    body = r.messages[-1]["content"]
    assert body.index("EXTRA_SEGMENT") < body.index("fact number 0")


def test_tier_by_memory_id_overrides_priority():
    mems = _items(2, prio=2)
    r = _cc().compile(
        memories=mems, user_message="q", tier_by_memory_id={id(mems[1]): TIER_PROCEDURAL}
    )
    assert "1 procedure" in r.metadata["retrieval_summary"]
    assert "1 semantic fact" in r.metadata["retrieval_summary"]


def test_classifier_and_freshness_and_dedup_stages_run():
    calls: list[str] = []

    class Fresh:
        def apply_weights(self, mems):
            calls.append("fresh")
            return mems[:2]

    class Dedup:
        def deduplicate(self, mems, text_fn):
            calls.append("dedup")
            assert text_fn(mems[0]) == mems[0].content
            return SimpleNamespace(kept=mems[:1], removed=mems[1:])

    r = _cc().compile(
        memories=_items(4), user_message="q", freshness=Fresh(), deduplicator=Dedup()
    )
    assert calls == ["fresh", "dedup"]
    assert r.metadata["segments_kept"] == 1
    assert r.dedup_report.kept[0].id == "m0"


def test_enable_lineage_without_tracker_raises():
    with pytest.raises(FeatureNotAvailable):
        _cc().compile(memories=_items(1), user_message="q", enable_lineage=True)


async def test_enable_lineage_without_tracker_raises_async():
    async def ca(*a):
        return "x"

    with pytest.raises(FeatureNotAvailable):
        await _cc().compile_async(
            memories=_items(1), user_message="q", enable_lineage=True, compress_async=ca
        )


def test_lineage_tracker_registers_and_finalizes():
    lin = FakeLineage()
    r = _cc().compile(memories=_items(2), user_message="q", lineage_tracker=lin)
    assert [s[0] for s in lin.registered] == ["m0", "m1"]
    assert all(s[2] == "compile" for s in lin.registered)
    assert lin.finalized["included"] == {"m0", "m1"}
    assert r.lineage_report.to_dict() == {"included": ["m0", "m1"]}
    assert r.explain("m0") == "explained m0"


def test_profiler_assertions_recorder_metrics_collector_and_exporters():
    recorded: list[Any] = []
    exported: list[Any] = []
    hist: list[tuple[str, float]] = []
    counters: list[tuple[str, float]] = []

    class Rec:
        def record(self, mems, user_message, result, cfg):
            recorded.append((len(mems), user_message, cfg))

    class Asserts:
        def run(self, result):
            return SimpleNamespace(passed=True)

    class Collector:
        def histogram(self, name, v):
            hist.append((name, v))

        def counter(self, name, v):
            counters.append((name, v))

    cb_seen: list[Any] = []
    cc = _cc(metrics_callback=SimpleNamespace(on_compile=cb_seen.append))
    r = cc.compile(
        memories=_items(2),
        user_message="q",
        enable_profiler=True,
        assertions=Asserts(),
        recorder=Rec(),
        metrics_collector=Collector(),
        trace_exporters=[SimpleNamespace(export_compile_run=exported.append)],
    )
    assert r.latency_profile is not None
    assert r.assertion_report.passed is True
    assert recorded == [(2, "q", {"max_tokens": 2000, "model_id": None})]
    assert [h[0] for h in hist] == ["compile_latency_ms"]
    assert [c[0] for c in counters] == ["compile_tokens_total", "compile_tokens_trimmed"]
    assert counters[0][1] == float(r.metadata["token_count"])
    assert exported == [r]
    assert len(cb_seen) == 1
    assert r.to_debug_dict()["latency_profile"] is not None


def test_recorder_and_collector_failures_do_not_break_compile():
    class BadRec:
        def record(self, *a):
            raise RuntimeError("nope")

    class BadCollector:
        def histogram(self, *a):
            raise RuntimeError("nope")

        def counter(self, *a):
            raise RuntimeError("nope")

    r = _cc().compile(
        memories=_items(1), user_message="q", recorder=BadRec(), metrics_collector=BadCollector()
    )
    assert r.metadata["segments_kept"] == 1


def test_tokenizer_registry_used_and_failure_falls_back():
    class Reg:
        def __init__(self, fail):
            self.fail = fail
            self.calls = 0

        def count(self, model_id, text):
            self.calls += 1
            if self.fail:
                raise RuntimeError("no tokenizer")
            return 1

    ok = Reg(False)
    r = _cc(model_id="unknown-model-x").compile(
        memories=_items(2), user_message="q", tokenizer_registry=ok
    )
    assert ok.calls > 0
    assert r.metadata["token_count"] < 20
    bad = Reg(True)
    r2 = _cc(model_id="unknown-model-x").compile(
        memories=_items(2), user_message="q", tokenizer_registry=bad
    )
    assert bad.calls > 0
    assert r2.metadata["token_count"] > r.metadata["token_count"]


# --- compression -------------------------------------------------------------


def test_compression_replaces_middle_and_records_debug_trace():
    comp_res = FakeCompressionResult(
        compressed="  [Mem:2] squashed summary  ", details={"k": "v"},
        original_tokens=100, compressed_tokens=30,
    )
    fc = FakeCompression(comp_res)
    lin = FakeLineage()
    r = _cc(model_id="unknown-model-x").compile(
        memories=_items(3), user_message="what?", compression=fc, lineage_tracker=lin,
        enable_profiler=True,
    )
    assert fc.model_ids == ["unknown-model-x"]
    assert fc.calls[0]["query_context"] == "what?"
    assert "fact number 0" in fc.calls[0]["text"]
    body = r.messages[-1]["content"]
    assert "[Mem:2] squashed summary" in body
    assert "fact number 1" not in body
    assert r.metadata["compression_strategy_used"] == "fake"
    assert r.metadata["compression_token_delta"] == 70
    assert r.metadata["compression_details"] == {"k": "v"}
    assert r.debug_trace["compression"]["strategy_used"] == "fake"
    assert r.debug_trace["compression"]["token_delta"] == 70
    # ref_map is rebuilt from the [Mem:N] references the compressed text still carries
    assert r.ref_map == {2: "m1"}
    assert r.metadata["segments_offered"] == 1


def test_compression_target_from_policy_ratio_and_empty_output_keeps_raw():
    fc = FakeCompression(FakeCompressionResult(compressed="   ", details=None))
    pol = FakePolicies(policy_compression_target=0.5)
    r = _cc().compile(memories=_items(3), user_message="q", compression=fc, policies=pol)
    assert fc.calls[0]["target_tokens"] >= 1
    assert fc.calls[0]["target_tokens"] <= count_tokens(fc.calls[0]["text"])
    assert r.metadata["compression_details"] == {}
    assert "fact number 2" in r.messages[-1]["content"]
    assert r.metadata["segments_kept"] == 3


def test_compress_sync_triggered_over_threshold(monkeypatch):
    monkeypatch.setattr(comp.os_config, "get_preflight_token_threshold", lambda: 1)
    seen = {}

    def cs(text, target, query):
        seen.update(text=text, target=target, query=query)
        return "  [Mem:1] tiny  "

    r = _cc().compile(memories=_items(3), user_message="ask", compress_sync=cs)
    assert seen["query"] == "ask" and "fact number 0" in seen["text"]
    assert "[Mem:1] tiny" in r.messages[-1]["content"]
    assert r.debug_trace["segment_weights_used"] is False
    assert r.ref_map == {1: "m0"}


def test_compress_sync_failure_warns_and_keeps_raw(monkeypatch, caplog):
    monkeypatch.setattr(comp.os_config, "get_preflight_token_threshold", lambda: 1)

    def cs(*a):
        raise RuntimeError("compressor exploded")

    with caplog.at_level(logging.WARNING, logger="membrane.compiler"):
        r = _cc().compile(memories=_items(2), user_message="q", compress_sync=cs)
    assert "fact number 1" in r.messages[-1]["content"]
    assert any("compress_sync failed" in rec.getMessage() for rec in caplog.records)


# --- compile(): layout -------------------------------------------------------


def test_end_layout_core_persona_history_and_directives():
    cc = ContextCompiler(max_tokens=3000, core_content="CORE_RULES")
    hist = [
        {"role": "user", "content": "first q"},
        {"role": "assistant", "content": "first a"},
    ]
    r = cc.compile(
        memories=_items(2),
        user_message="second q",
        conversation_messages=hist,
        agent_persona_prompt="PERSONA_X",
        dynamic_directives=["DIR_A", "DIR_B"],
    )
    roles = [m["role"] for m in r.messages]
    assert roles == ["system", "user", "assistant", "user"]
    sys = r.messages[0]["content"]
    assert "CORE_RULES" in sys and "PERSONA_X" in sys and "SYSTEM_INTERRUPT_FETCH" in sys
    assert r.messages[1] == hist[0] and r.messages[2] == hist[1]
    tail = r.messages[-1]["content"]
    assert "second q" in tail and "[Mem:1]" in tail
    assert r.metadata["stable_prefix_messages"] == 3
    assert r.metadata["memory_placement"] == "end"
    assert r.metadata["security_flag"] == SecurityFlag.AUTHENTICATED.value


def test_end_layout_without_core_omits_empty_system_message():
    r = _cc().compile(memories=_items(1), user_message="q")
    assert [m["role"] for m in r.messages] == ["user"]
    assert r.metadata["stable_prefix_messages"] == 0


def test_last_assistant_turn_included_when_no_history():
    r = _cc().compile(memories=[], user_message="q", last_assistant_turn="PREV_ANSWER")
    assert "PREV_ANSWER" in r.messages[-1]["content"]
    assert r.metadata["retrieval_summary"].startswith("No long-term")


def test_history_trimming_reports_dropped_and_respects_budget():
    hist = []
    for i in range(20):
        hist.append({"role": "user", "content": f"q{i} " + "word " * 60})
        hist.append({"role": "assistant", "content": f"a{i} " + "word " * 60})
    r = _cc(800).compile(memories=_items(10), user_message="q", conversation_messages=hist)
    assert r.metadata["history_messages_dropped"] > 0
    assert r.metadata["token_count"] <= 800
    assert "budget_exceeded" not in r.metadata


def test_untrimmed_history_overrun_warns(caplog):
    hist = [{"role": "user", "content": "word " * 600}, {"role": "assistant", "content": "ok"}]
    with caplog.at_level(logging.WARNING, logger="membrane.compiler"):
        r = _cc(200, trim_history=False).compile(
            memories=_items(1), user_message="q", conversation_messages=hist
        )
    assert r.metadata["budget_exceeded"] is True
    assert r.metadata["history_messages_dropped"] == 0
    assert any("exceeds max_tokens" in rec.getMessage() for rec in caplog.records)


def test_system_placement_layout():
    cc = _cc(memory_placement="system", core_content="CORE")
    r = cc.compile(
        memories=_items(2), user_message="the question", last_assistant_turn="earlier reply",
        dynamic_directives=["DIRECTIVE_Z"],
    )
    assert len(r.messages) == 1 and r.messages[0]["role"] == "system"
    text = r.messages[0]["content"]
    assert text.index("CORE") < text.index("[Mem:1]") < text.index("the question")
    assert "earlier reply" in text and "DIRECTIVE_Z" in text
    assert r.metadata["token_count"] == count_tokens(text)


def test_system_placement_uses_last_turn_from_history():
    r = _cc(memory_placement="system").compile(
        memories=[], user_message="q",
        conversation_messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "HIST_REPLY"},
        ],
    )
    assert "HIST_REPLY" in r.messages[0]["content"]


def test_prompt_caching_metadata_with_large_prefix(monkeypatch):
    monkeypatch.setattr(comp, "has_prompt_caching", lambda mid: True)
    big_core = "stable rule " * 800
    cc = ContextCompiler(
        max_tokens=20000, model_id="cachey", core_content=big_core,
        include_interrupt_instruction=False,
    )
    monkeypatch.setattr(comp, "lookup_context_cap", lambda mid: None)
    r = cc.compile(memories=_items(2), user_message="q")
    assert r.metadata["prompt_cache_used"] is True
    assert r.metadata["cacheable_prefix_tokens"] >= comp.MIN_CACHEABLE_PREFIX_TOKENS
    assert r.metadata["dynamic_tail_tokens"] > 0
    off = cc.compile(memories=_items(2), user_message="q", use_prompt_caching_split=False)
    assert "prompt_cache_used" not in off.metadata

    sys_cc = ContextCompiler(
        max_tokens=20000, model_id="cachey", core_content=big_core,
        include_interrupt_instruction=False, memory_placement="system",
    )
    rs = sys_cc.compile(memories=_items(2), user_message="q")
    assert rs.metadata["prompt_cache_used"] is True
    assert rs.metadata["cacheable_prefix_chars"] > 0
    assert rs.messages[0]["content"].startswith(big_core.strip()[:50])


def test_build_output_messages_prefix_must_match():
    msgs, meta = comp._build_output_messages(None, "text", True, prefix_text="text")
    assert msgs == [{"role": "system", "content": "text"}] and meta == {}


def test_build_output_messages_caching_branches(monkeypatch):
    monkeypatch.setattr(comp, "has_prompt_caching", lambda mid: True)
    big = "stable " * 1500
    _, meta = comp._build_output_messages("m", big + "tail", True, prefix_text=big)
    assert meta["cacheable_prefix_chars"] == len(big)
    assert meta["dynamic_tail_tokens"] == count_tokens("tail")
    _, meta2 = comp._build_output_messages("m", "different", True, prefix_text=big)
    assert meta2 == {}
    _, meta3 = comp._build_output_messages("m", "small", True, prefix_text="small")
    assert meta3 == {}


# --- compile_async -----------------------------------------------------------


class FakeRedis:
    def __init__(self, store=None, fail_get=False, fail_set=False):
        self.store = dict(store or {})
        self.fail_get = fail_get
        self.fail_set = fail_set
        self.sets: list[tuple[str, str]] = []

    async def get(self, key):
        if self.fail_get:
            raise RuntimeError("get failed")
        return self.store.get(key)

    async def set(self, key, val, ex=None):
        if self.fail_set:
            raise RuntimeError("set failed")
        self.sets.append((key, val))
        self.store[key] = val


async def test_compile_async_without_async_hooks_delegates_to_sync():
    r = await _cc().compile_async(memories=_items(2), user_message="q")
    assert r.metadata["segments_kept"] == 2


async def test_compile_async_compress_async_and_cache_write(monkeypatch):
    monkeypatch.setattr(comp.os_config, "get_preflight_token_threshold", lambda: 1)
    redis = FakeRedis()
    seen = {}

    async def ca(text, target, query):
        seen.update(text=text, query=query)
        return " [Mem:2] async squashed "

    r = await _cc().compile_async(
        memories=_items(3), user_message="ask", compress_async=ca,
        redis_client=redis, conversation_id="conv1",
    )
    assert seen["query"] == "ask"
    assert "[Mem:2] async squashed" in r.messages[-1]["content"]
    assert r.ref_map == {2: "m1"}
    assert len(redis.sets) == 1 and redis.sets[0][1].strip() == "[Mem:2] async squashed"


async def test_compile_async_uses_redis_cache_hit_and_skips_compressor(monkeypatch):
    monkeypatch.setattr(comp.os_config, "get_preflight_token_threshold", lambda: 1)

    async def get_cache(client, conv, msg):
        assert (conv, msg) == ("c", "q")
        return "[Mem:1] cached summary"

    monkeypatch.setattr(comp, "get_preflight_cache", get_cache)

    async def ca(*a):
        raise AssertionError("must not be called on cache hit")

    r = await _cc().compile_async(
        memories=_items(3), user_message="q", compress_async=ca,
        redis_client=object(), conversation_id="c",
    )
    assert "cached summary" in r.messages[-1]["content"]


async def test_compile_async_cache_lookup_failure_falls_through(monkeypatch):
    monkeypatch.setattr(comp.os_config, "get_preflight_token_threshold", lambda: 1)

    async def bad_get(*a):
        raise RuntimeError("redis down")

    monkeypatch.setattr(comp, "get_preflight_cache", bad_get)

    async def ca(*a):
        return "[Mem:1] fallback summary"

    r = await _cc().compile_async(
        memories=_items(2), user_message="q", compress_async=ca,
        redis_client=object(), conversation_id="c",
    )
    assert "fallback summary" in r.messages[-1]["content"]


async def test_compile_async_cache_write_failure_is_tolerated(monkeypatch):
    monkeypatch.setattr(comp.os_config, "get_preflight_token_threshold", lambda: 1)

    async def bad_set(*a):
        raise RuntimeError("redis down")

    monkeypatch.setattr(comp, "set_preflight_cache", bad_set)

    async def ca(*a):
        return "[Mem:1] summary"

    r = await _cc().compile_async(
        memories=_items(2), user_message="q", compress_async=ca,
        redis_client=FakeRedis(), conversation_id="c",
    )
    assert "[Mem:1] summary" in r.messages[-1]["content"]


async def test_compile_async_failure_falls_back_to_compress_sync(monkeypatch, caplog):
    monkeypatch.setattr(comp.os_config, "get_preflight_token_threshold", lambda: 1)

    async def ca(*a):
        raise RuntimeError("async compressor failed")

    def cs(text, target, query):
        return "[Mem:1] sync fallback"

    with caplog.at_level(logging.WARNING, logger="membrane.compiler"):
        r = await _cc().compile_async(
            memories=_items(2), user_message="q", compress_async=ca, compress_sync=cs
        )
    assert "sync fallback" in r.messages[-1]["content"]
    assert any("compress_async failed" in rec.getMessage() for rec in caplog.records)


async def test_compile_async_both_compressors_fail_keeps_raw(monkeypatch, caplog):
    monkeypatch.setattr(comp.os_config, "get_preflight_token_threshold", lambda: 1)

    async def ca(*a):
        raise RuntimeError("a")

    def cs(*a):
        raise RuntimeError("s")

    with caplog.at_level(logging.WARNING, logger="membrane.compiler"):
        r = await _cc().compile_async(
            memories=_items(2), user_message="q", compress_async=ca, compress_sync=cs
        )
    assert "fact number 1" in r.messages[-1]["content"]
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("compress_async failed" in m for m in msgs)
    assert any("compress_sync failed" in m for m in msgs)


async def test_compile_async_full_plugin_stack():
    recorded: list[Any] = []
    hist: list[str] = []
    exported: list[Any] = []
    lin = FakeLineage()

    class Rev:
        def rerank(self, query, mems):
            return [(m, float(i)) for i, m in enumerate(reversed(mems))]

    class Fresh:
        def apply_weights(self, mems):
            return mems

    class Dedup:
        def deduplicate(self, mems, text_fn):
            return SimpleNamespace(kept=mems, removed=[])

    class Asserts:
        def run(self, result):
            return SimpleNamespace(passed=False)

    class Collector:
        def histogram(self, name, v):
            hist.append(name)

        def counter(self, name, v):
            hist.append(name)

    class Rec:
        def record(self, *a):
            recorded.append(a)

    async def ca(*a):
        return None

    fc = FakeCompression(
        FakeCompressionResult(compressed="[Mem:1] squashed", strategy_used="jta_deterministic",
                              details={"alias_decisions": [{"phrase": "p", "alias": "A"}],
                                       "aliases_reused": 1})
    )
    cb: list[Any] = []
    cc = _cc(
        model_id="unknown-model-x",
        metrics_callback=SimpleNamespace(on_compile=cb.append),
    )
    r = await cc.compile_async(
        memories=_items(3),
        user_message="q",
        compress_async=ca,
        enable_profiler=True,
        reranker=Rev(),
        freshness=Fresh(),
        deduplicator=Dedup(),
        assertions=Asserts(),
        recorder=Rec(),
        metrics_collector=Collector(),
        lineage_tracker=lin,
        compression=fc,
        policies=FakePolicies(policy_compression_target=0.5),
        tier_by_memory_id={},
        trace_exporters=[SimpleNamespace(export_compile_run=exported.append)],
        middle_extras=[("EXTRA", TIER_SEMANTIC)],
    )
    assert fc.model_ids == ["unknown-model-x"]
    assert r.latency_profile is not None
    assert r.assertion_report.passed is False
    assert len(recorded) == 1 and exported == [r] and len(cb) == 1
    assert hist == ["compile_latency_ms", "compile_tokens_total", "compile_tokens_trimmed"]
    assert [s[2] for s in lin.registered] == ["compile_async"] * 3
    assert [kw["action"] for _, kw in lin.events] == ["jta_alias_map", "jta_incremental_cache"]
    assert r.debug_trace["compression"]["strategy_used"] == "jta_deterministic"
    assert r.metadata["compression_token_delta"] == 60
    assert r.lineage_report is not None


async def test_compile_async_reranker_extras_and_classifier():
    class Same:
        def rerank(self, query, mems):
            return [(m, 1.0) for m in mems]

    class Cls:
        def classify(self, text):
            from membrane.classifier.classifier import MemoryKind

            return SimpleNamespace(kind=MemoryKind.PROCEDURAL)

    async def ca(*a):
        return None

    r = await _cc().compile_async(
        memories=_items(2), user_message="q", reranker=Same(), classifier=Cls(),
        compress_async=ca, middle_extras=[("EXTRA_SEG", TIER_SEMANTIC)],
    )
    body = r.messages[-1]["content"]
    assert body.index("EXTRA_SEG") < body.index("fact number 0")
    assert r.metadata["memory_kinds_present"] == ["procedural"]
    assert set(r.ref_map_type.values()) == {"procedural"}


async def test_compile_async_tokenizer_registry_and_failure():
    class Reg:
        def __init__(self, fail):
            self.fail = fail

        def count(self, model_id, text):
            if self.fail:
                raise RuntimeError("x")
            return 1

    async def ca(*a):
        return None

    ok = await _cc(model_id="unknown-model-x").compile_async(
        memories=_items(2), user_message="q", compress_async=ca, tokenizer_registry=Reg(False)
    )
    bad = await _cc(model_id="unknown-model-x").compile_async(
        memories=_items(2), user_message="q", compress_async=ca, tokenizer_registry=Reg(True)
    )
    assert ok.metadata["token_count"] < bad.metadata["token_count"]


async def test_compile_async_recorder_and_collector_failures_tolerated():
    class BadRec:
        def record(self, *a):
            raise RuntimeError("x")

    class BadCollector:
        def histogram(self, *a):
            raise RuntimeError("x")

        def counter(self, *a):
            raise RuntimeError("x")

    async def ca(*a):
        return None

    r = await _cc().compile_async(
        memories=_items(1), user_message="q", compress_async=ca, recorder=BadRec(),
        metrics_collector=BadCollector(),
    )
    assert r.metadata["segments_kept"] == 1


async def test_compile_async_attention_shift_and_history(monkeypatch):
    seen = {}

    def fake_shift(last_users, current):
        seen["args"] = (last_users, current)
        return True

    monkeypatch.setattr(comp, "compute_attention_shift", fake_shift)

    async def ca(*a):
        return None

    r = await _cc().compile_async(
        memories=_items(1), user_message="new topic", compress_async=ca,
        conversation_messages=[{"role": "user", "content": "old topic"}],
    )
    assert seen["args"] == (["old topic"], "new topic")
    assert r.metadata["attention_shift"] is True


async def test_compile_async_emits_stale_signal_only_with_survivors():
    def Mem(id_, stale):
        m = MemoryItem(id=id_, content=f"content {id_}")
        m._membrane_is_stale = stale  # type: ignore[attr-defined]
        return m

    received: list[Any] = []
    bus = SignalBus()

    async def on_stale(sig):
        received.append(sig)

    bus.subscribe(SignalKind.STALE_CONTENT_WARNING, on_stale)
    prof = MembraneProfile.minimal()
    prof.signal_bus = bus

    async def ca(*a):
        return None

    r = await _cc().compile_async(
        memories=[Mem("s1", True), Mem("f1", False)], user_message="q", compress_async=ca,
        profile=prof,
    )
    assert r.metadata["stale_survivors"] == ["s1"]
    assert len(received) == 1 and received[0].metadata["segment_ids"] == ["s1"]

    received.clear()
    await _cc().compile_async(
        memories=[Mem("f1", False)], user_message="q", compress_async=ca, profile=prof
    )
    assert received == []


# --- compile_prompt_async ----------------------------------------------------


async def test_compile_prompt_async_anonymous_fast_path():
    r = await compile_prompt_async(
        user_msg="hello", messages_list=[], user_id=None, max_tokens=1000,
        memory_placement="system",
    )
    assert r.metadata["security_flag"] == SecurityFlag.ANONYMOUS_MODE.value
    assert "hello" in r.messages[0]["content"]


async def test_compile_prompt_async_full_path_metadata_and_ref_map_persist(monkeypatch):
    saved: dict[str, Any] = {}

    async def fake_set(conv, ref_map, ref_type, **kw):
        saved.update(conv=conv, ref_map=ref_map, ref_type=ref_type, **kw)

    monkeypatch.setattr(comp, "set_ref_map_for_conversation", fake_set)
    mems = [(m, TIER_SEMANTIC) for m in _items(2)]
    r = await compile_prompt_async(
        user_msg="q", messages_list=[], user_id="u1", max_tokens=2000,
        retrieved_objects=mems, current_stress=0.3, conversation_id="conv9",
        persist_ref_map_to_redis=True, redis_url_for_ref_map="redis://x",
    )
    assert r.metadata["user_id"] == "u1"
    assert r.metadata["current_stress"] == 0.3
    assert r.metadata["conversation_id"] == "conv9"
    assert saved["conv"] == "conv9"
    assert saved["ref_map"] == {1: "m0", 2: "m1"}
    assert saved["ref_type"] == {1: "semantic", 2: "semantic"}
    assert saved["user_id"] == "u1" and saved["redis_url"] == "redis://x"
    assert "SYSTEM_INTERRUPT_FETCH" in r.messages[0]["content"]


async def test_compile_prompt_async_anonymous_full_path_flagged():
    r = await compile_prompt_async(
        user_msg="q", messages_list=[], user_id=None, max_tokens=2000,
        dynamic_directives=["BE_BRIEF"],
    )
    assert r.metadata["security_flag"] == SecurityFlag.ANONYMOUS_MODE.value
    assert r.metadata["user_id"] is None
    assert "conversation_id" not in r.metadata
    assert "BE_BRIEF" in r.messages[-1]["content"]


async def test_compile_prompt_async_lod_focus_merge(monkeypatch):
    monkeypatch.setattr(comp, "compute_attention_shift", lambda *a, **k: True)
    main = [(m, TIER_SEMANTIC) for m in _items(2)]
    focus_mem = MemoryItem(id="focus", content="focused fact", memory_type="semantic")

    async def focus():
        return [(focus_mem, TIER_SEMANTIC)]

    r = await compile_prompt_async(
        user_msg="q", messages_list=[{"role": "user", "content": "old"}], user_id="u",
        max_tokens=2000, retrieved_objects=main, lod_focus_retrieve=focus,
    )
    assert r.ref_map[1] == "focus"
    assert set(r.ref_map.values()) == {"focus", "m0", "m1"}


async def test_compile_prompt_async_lod_focus_failure_is_tolerated(monkeypatch):
    monkeypatch.setattr(comp, "compute_attention_shift", lambda *a, **k: True)
    main = [(m, TIER_SEMANTIC) for m in _items(2)]

    async def focus():
        raise RuntimeError("focus retrieval down")

    r = await compile_prompt_async(
        user_msg="q", messages_list=[], user_id="u", max_tokens=2000,
        retrieved_objects=main, lod_focus_retrieve=focus,
    )
    assert set(r.ref_map.values()) == {"m0", "m1"}


# --- remaining branches ------------------------------------------------------


def test_trim_history_skips_leading_non_user_messages():
    hist = [
        {"role": "user", "content": "word " * 100},
        {"role": "assistant", "content": "word " * 100},
        {"role": "tool", "content": "result", "tool_call_id": "c"},
        {"role": "user", "content": "short"},
    ]
    kept, dropped = comp._trim_history_chunked(hist, 60, count_tokens)
    assert kept[0] == hist[3] and dropped == 3


def test_system_placement_overrun_flags_budget_exceeded():
    r = _cc(10, memory_placement="system", core_content="core " * 50).compile(
        memories=_items(2), user_message="q"
    )
    assert r.metadata["token_count"] > 10
    assert r.metadata["budget_exceeded"] is True


def test_classifier_retypes_memories_in_sync_compile():
    from membrane.classifier.classifier import MemoryKind

    class Cls:
        def classify(self, text):
            return SimpleNamespace(kind=MemoryKind.PROCEDURAL)

    mems = _items(2)
    r = _cc().compile(memories=mems, user_message="q", classifier=Cls())
    assert set(r.ref_map_type.values()) == {"procedural"}
    assert all(m.memory_type == "semantic" for m in mems)  # caller objects untouched


async def test_async_compile_persona_directives_and_interrupt_instruction():
    async def ca(*a):
        return None

    cc = ContextCompiler(max_tokens=3000, core_content="CORE_ASYNC")
    r = await cc.compile_async(
        memories=_items(2), user_message="q", compress_async=ca,
        agent_persona_prompt="PERSONA_ASYNC", dynamic_directives=["DIR_ASYNC"],
        tier_by_memory_id={0: TIER_EPISODIC},
    )
    sys = r.messages[0]["content"]
    assert "CORE_ASYNC" in sys and "PERSONA_ASYNC" in sys and "SYSTEM_INTERRUPT_FETCH" in sys
    assert "DIR_ASYNC" in r.messages[-1]["content"]


async def test_async_compile_classifier_stage_and_profiler():
    from membrane.classifier.classifier import MemoryKind

    class Cls:
        def classify(self, text):
            return SimpleNamespace(kind=MemoryKind.EPISODIC)

    async def ca(*a):
        return None

    r = await _cc().compile_async(
        memories=_items(2), user_message="q", compress_async=ca, classifier=Cls(),
        enable_profiler=True,
    )
    assert set(r.ref_map_type.values()) == {"episodic"}
    assert r.latency_profile is not None


def test_compile_anonymous_caps_persona_history_and_overrun(monkeypatch):
    monkeypatch.setattr(comp, "lookup_context_cap", lambda mid: 150)
    seen: list[Any] = []
    hist = [
        {"role": "user", "content": "word " * 400},
        {"role": "assistant", "content": "answer"},
    ]
    r = comp.compile_anonymous(
        user_message="q", conversation_messages=hist, max_tokens=5000, model_id="capped",
        core_content="CORE", agent_persona_prompt="PERSONA",
        metrics_callback=SimpleNamespace(on_compile=seen.append),
    )
    assert r.metadata["tokens_budget"] == 150
    assert r.metadata["budget_exceeded"] is True
    assert r.metadata["security_flag"] == SecurityFlag.ANONYMOUS_MODE.value
    text = r.messages[0]["content"]
    assert "PERSONA" in text and "--- Recent conversation ---" in text
    assert len(seen) == 1


def test_compile_anonymous_under_budget_is_not_flagged():
    r = comp.compile_anonymous(
        user_message="q", conversation_messages=[{"role": "user", "content": "hi"}],
        max_tokens=1000,
    )
    assert "budget_exceeded" not in r.metadata
    assert r.metadata["token_count"] == count_tokens(r.messages[0]["content"])
    assert r.ref_map == {}
