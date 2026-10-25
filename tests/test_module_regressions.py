"""Regression tests for module-level bug fixes (templates, adapters, conversation, ...)."""

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import pytest


# 1. templates ---------------------------------------------------------------


def test_template_for_loop_var_not_required():
    from membrane.templates.engine import PromptTemplate

    t = PromptTemplate("{% for x in xs %}{{x}},{% endfor %}")
    assert t.render(xs=[1, 2])[0]["content"] == "1,2,"
    assert t.validate(xs=[1]) == []


def test_template_if_condition_optional():
    from membrane.templates.engine import PromptTemplate

    t = PromptTemplate("a{% if flag %}B{% endif %}")
    assert t.render()[0]["content"] == "a"
    assert t.render(flag=True)[0]["content"] == "aB"
    n = PromptTemplate("{% if not flag %}off{% endif %}")
    assert n.render(flag=False)[0]["content"] == "off"
    assert n.render(flag=True) == []


def test_template_role_blocks_not_merged():
    from membrane.templates.engine import PromptTemplate

    src = (
        "{% system %}S{% end_system %}\n{% user %}u1{% end_user %}\n"
        "{% assistant %}a{% end_assistant %}\n{% user %}u2{% end_user %}"
    )
    msgs = PromptTemplate(src).render()
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("system", "S"),
        ("user", "u1"),
        ("assistant", "a"),
        ("user", "u2"),
    ]


def test_template_whitespace_between_blocks_no_empty_system():
    from membrane.templates.engine import PromptTemplate

    msgs = PromptTemplate(
        "\n{% user %}hi{% end_user %}\n\n{% assistant %}yo{% end_assistant %}\n"
    ).render()
    assert [m["role"] for m in msgs] == ["user", "assistant"]


# 2. adapters ----------------------------------------------------------------


def test_llamaindex_node_to_memory_constant_tier():
    from membrane.adapters.llamaindex import node_to_memory

    node = SimpleNamespace(text="hello", id_="n1", score=0.1)
    m = node_to_memory(node)
    assert m.semantic_tier == 2
    assert m.relevance_score == pytest.approx(0.1)


def test_llamaindex_postprocessor_is_base_postprocessor():
    pytest.importorskip("llama_index.core")
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    from membrane.adapters.llamaindex import MembraneNodePostprocessor

    class FakeCompiler:
        def compile(self, **kw):
            return SimpleNamespace(messages=[{"role": "user", "content": "compiled"}])

    pp = MembraneNodePostprocessor(FakeCompiler())
    assert isinstance(pp, BaseNodePostprocessor)
    out = pp.postprocess_nodes(
        [NodeWithScore(node=TextNode(text="x"), score=0.5)], QueryBundle("q")
    )
    assert isinstance(out[0], NodeWithScore)
    assert out[0].node.get_content() == "compiled"


def test_langchain_format_with_compiled_context_kwarg():
    from membrane.adapters.langchain import MembranePromptTemplate

    class FakeCompiler:
        def compile(self, **kw):
            return SimpleNamespace(messages=[{"role": "user", "content": "CORE"}])

    tpl = MembranePromptTemplate(FakeCompiler(), "{compiled_context}|{input}", memories=[])
    assert tpl.format(input="q", compiled_context="ignored") == "CORE|q"


def test_langchain_doc_non_numeric_priority():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document

    from membrane.adapters.langchain import langchain_doc_to_memory

    m = langchain_doc_to_memory(Document(page_content="x", metadata={"priority": "high"}))
    assert m.priority == 2


# 3. conversation ------------------------------------------------------------


def _manager(strategy, calls, trigger=4):
    from membrane.conversation.manager import ConversationManager

    def summ(blob):
        calls.append(blob)
        return f"S{len(calls)}"

    return ConversationManager(
        compiler=None, window_strategy=strategy, summary_trigger=trigger, summarization_fn=summ
    )


@pytest.mark.parametrize("strategy", ["hybrid", "summarize"])
def test_conversation_summary_folds_and_trims(strategy):
    calls = []
    m = _manager(strategy, calls)
    for i in range(12):
        m.add_turn("user", f"t{i}")
    # summarizer fires once per trigger, not every turn
    assert len(calls) < 6
    assert len(m._turns) <= 4
    assert m._turns[-1] == ("user", "t11")
    # every turn is either summarized or retained: first turn is in the first blob
    assert "t0" in calls[0]
    # later summaries fold in the previous summary
    assert any("Previous summary: S1" in c for c in calls[1:])
    # no turn is silently dropped between summaries
    seen = "".join(calls) + "".join(c for _, c in m._turns)
    for i in range(12):
        assert f"t{i}" in seen


def test_conversation_deserialize_restores_config():
    from membrane.conversation.manager import ConversationManager

    m = ConversationManager(None, max_turns=7, summary_trigger=3, topic_aware=False)
    m.add_turn("user", "a")
    m2 = ConversationManager.deserialize(m.serialize(), None)
    assert (m2._max_turns, m2._summary_trigger, m2._topic_aware) == (7, 3, False)
    assert m2._turns == [("user", "a")]


# 4. policies ----------------------------------------------------------------


def test_policy_condition_ands_all_keys():
    from membrane.policies.engine import PolicyCondition

    c = PolicyCondition({"query_contains_any": ["foo"], "memory_count_above": 5})
    kw = dict(memories=[1], estimated_tokens=0, model_id=None)
    assert c.matches(query="foo", **kw) is False
    assert c.matches(query="foo", memories=[1] * 6, estimated_tokens=0, model_id=None) is True
    assert c.matches(query="bar", memories=[1] * 6, estimated_tokens=0, model_id=None) is False


@pytest.mark.parametrize(
    "cond",
    [{"bogus": 1}, {}, {"query_length_below": "abc"}, {"query_contains_any": "foo"}, 42],
)
def test_policy_from_dict_rejects_bad_condition(cond):
    from membrane.policies.engine import PolicyEngine

    with pytest.raises(ValueError):
        PolicyEngine.from_dict({"policies": [{"name": "p", "condition": cond}]})


# 5. tokenizers --------------------------------------------------------------


def test_tokenizer_registry_precedence_and_encodings():
    from membrane.tokenizers.registry import TokenizerRegistry

    TokenizerRegistry.reset_for_testing()
    try:
        reg = TokenizerRegistry.get()
        assert reg._match("gpt-4o-mini").name == "tiktoken:o200k_base"
        assert reg._match("gpt-4").name == "tiktoken:cl100k_base"
        assert reg._match("gpt-4-turbo").name == "tiktoken:cl100k_base"
        custom = SimpleNamespace(name="custom")
        reg.register("gpt-4*", custom)
        assert reg._match("gpt-4") is custom
    finally:
        TokenizerRegistry.reset_for_testing()


def test_tokenizer_registry_instance_set_only_after_bootstrap(monkeypatch):
    from membrane.tokenizers.registry import TokenizerRegistry

    TokenizerRegistry.reset_for_testing()

    def boom(self):
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(TokenizerRegistry, "_bootstrap_defaults", boom)
    with pytest.raises(RuntimeError):
        TokenizerRegistry.get()
    assert TokenizerRegistry._instance is None


# 6. freshness ---------------------------------------------------------------


def test_freshness_zero_relevance_stays_zero_and_idempotent():
    from membrane.freshness.manager import FreshnessManager

    class Mem:
        def __init__(self, rel):
            self.relevance_score = rel
            self.timestamp = 0.0

    fm = FreshnessManager(reference_time=3600.0)
    zero, one = Mem(0.0), Mem(1.0)
    fm.apply_weights([zero, one])
    assert zero.relevance_score == 0.0
    first = one.relevance_score
    assert 0.0 < first < 1.0
    fm.apply_weights([zero, one])
    fm.apply_weights([zero, one])
    assert one.relevance_score == pytest.approx(first)


def test_freshness_tolerates_frozen_objects():
    from dataclasses import dataclass

    from membrane.freshness.manager import FreshnessManager

    @dataclass(frozen=True)
    class Frozen:
        relevance_score: float = 1.0
        timestamp: float = 0.0

    @dataclass(frozen=True)
    class Slotted:
        __slots__ = ("timestamp",)
        timestamp: float

    FreshnessManager(reference_time=10.0).apply_weights([Frozen(), Slotted(0.0)])


# 7. chunking ----------------------------------------------------------------


def test_recursive_strategy_preface_and_absolute_offsets():
    from membrane.chunking.strategies import RecursiveStrategy

    text = (
        "Intro paragraph before any heading.\n\n# Title\n\nBody text here.\n\n## Sub\n\nMore body."
    )
    chunks = RecursiveStrategy().chunk(text, max_tokens=50)
    assert chunks[0].content.startswith("Intro")
    assert chunks[0].start_char == 0
    for c in chunks:
        assert text[c.start_char : c.end_char].split() == c.content.split()


def test_fixed_strategy_large_overlap_does_not_explode():
    from membrane.chunking.strategies import FixedStrategy

    text = "abcdefghij" * 100
    chunks = FixedStrategy().chunk(
        text, max_tokens=5, overlap_tokens=50, tokenizer=lambda s: len(s) // 2
    )
    assert len(chunks) < 600
    assert chunks[-1].end_char == len(text)
    starts = [c.start_char for c in chunks]
    assert starts == sorted(set(starts))


def test_fixed_strategy_overlap_in_chars_from_tokens():
    from membrane.chunking.strategies import FixedStrategy

    text = "x" * 100
    chunks = FixedStrategy().chunk(text, max_tokens=10, overlap_tokens=2, tokenizer=len)
    assert chunks[1].start_char == 8
    assert chunks[0].end_char == 10


def test_pack_units_splits_oversized_units():
    from membrane.chunking.strategies import ParagraphStrategy

    text = "short one.\n\n" + ("y" * 100) + "\n\nshort two."
    chunks = ParagraphStrategy().chunk(text, max_tokens=20, tokenizer=len)
    assert all(c.token_count <= 20 for c in chunks)
    assert "".join(c.content for c in chunks).count("y") == 100


# 8. embedding cache ---------------------------------------------------------


class FakeRedis:
    def __init__(self):
        self.d = {}
        self.gets = 0

    def get(self, k):
        self.gets += 1
        return self.d.get(k)

    def set(self, k, v):
        self.d[k] = v

    def setex(self, k, ttl, v):
        self.d[k] = v

    def delete(self, k):
        self.d.pop(k, None)


def test_embedding_cache_batch_uses_redis_and_dedupes():
    from membrane.cache.embedding_cache import EmbeddingCache

    r = FakeRedis()
    cache = EmbeddingCache(backend="redis", redis_client=r)
    seen = []

    def embed(batch):
        seen.append(list(batch))
        return [[float(len(t))] for t in batch]

    out = cache.get_or_compute_batch(["a", "bb", "a"], embed)
    assert out == [[1.0], [2.0], [1.0]]
    assert seen == [["a", "bb"]]
    assert len(r.d) == 2
    assert cache.stats.misses == 3
    cache.get_or_compute_batch(["a", "bb"], embed)
    assert len(seen) == 1
    assert cache.stats.hits == 2


def test_embedding_cache_model_id_namespaces_keys():
    from membrane.cache.embedding_cache import EmbeddingCache

    r = FakeRedis()
    a = EmbeddingCache(backend="redis", redis_client=r, model_id="m1")
    b = EmbeddingCache(backend="redis", redis_client=r, model_id="m2")
    a.get_or_compute("t", lambda t: [1.0])
    assert b.get_or_compute("t", lambda t: [2.0]) == [2.0]


def test_embedding_cache_embed_fn_runs_outside_lock():
    from membrane.cache.embedding_cache import EmbeddingCache

    cache = EmbeddingCache()
    started = threading.Event()
    release = threading.Event()

    def slow(t):
        started.set()
        release.wait(2)
        return [1.0]

    th = threading.Thread(target=lambda: cache.get_or_compute("slow", slow))
    th.start()
    assert started.wait(2)
    t0 = time.time()
    assert cache.get_or_compute("fast", lambda t: [2.0]) == [2.0]
    release.set()
    th.join()
    assert time.time() - t0 < 1.0


def test_embedding_cache_batch_counts_misses_memory():
    from membrane.cache.embedding_cache import EmbeddingCache

    cache = EmbeddingCache()
    cache.get_or_compute_batch(["a", "b"], lambda xs: [[1.0] for _ in xs])
    assert cache.stats.misses == 2


# 9. classifier --------------------------------------------------------------


def test_code_feature_ignores_prose_and_user_prefix_matches():
    from membrane.classifier.features import extract_features

    assert extract_features("I got this from John yesterday")[0] == 0.0
    assert extract_features("The first class of the import duties")[0] == 0.0
    assert extract_features("def foo(x):\n    return x")[0] == 1.0
    assert extract_features("```py\nx\n```")[0] == 1.0
    assert extract_features("user: hello there")[2] > 0.0


def test_classifier_logs_weights_load_failure(monkeypatch, tmp_path, caplog):
    from membrane.classifier import classifier as mod

    bad = tmp_path / "weights.json"
    bad.write_text("{not json")
    monkeypatch.setattr(mod, "_load_weights_path", lambda: bad)
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        mod.MemoryClassifier()
    assert any("weights" in r.getMessage() for r in caplog.records)


# 10. dedup ------------------------------------------------------------------


def test_dedup_not_transitive():
    from membrane.dedup.deduplicator import SemanticDeduplicator

    vecs = {"a": [1.0, 0.0], "b": [0.8, 0.6], "c": [0.28, 0.96]}  # a~b, b~c, a!~c at 0.75
    d = SemanticDeduplicator(
        similarity_threshold=0.75, embedding_fn=lambda ts: [vecs[t] for t in ts]
    )
    res = d.deduplicate(["a", "b", "c"])
    assert res.merge_map == {1: 0}
    assert "c" in res.kept


def test_dedup_warns_once_when_fastembed_missing(monkeypatch, caplog):
    import sys

    from membrane.dedup import deduplicator as mod

    monkeypatch.setattr(mod, "_MODEL", None)
    monkeypatch.setattr(mod, "_MODEL_FAILED", False)
    monkeypatch.setitem(sys.modules, "fastembed", None)  # import raises ImportError
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        out = mod._default_embed_batch(["x"])
        mod._default_embed_batch(["y"])
    assert len(out[0]) == 32
    warnings = [r for r in caplog.records if "hash embeddings" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].exc_info is None


def test_dedup_model_loaded_once(monkeypatch):
    import sys

    from membrane.dedup import deduplicator as mod

    monkeypatch.setattr(mod, "_MODEL", None)
    monkeypatch.setattr(mod, "_MODEL_FAILED", False)
    count = []

    class FakeTE:
        def __init__(self, model_name):
            count.append(1)

        def embed(self, texts):
            return [[1.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTE))
    mod._default_embed_batch(["a"])
    mod._default_embed_batch(["b"])
    assert len(count) == 1


# 11. mcp server -------------------------------------------------------------


def test_mcp_per_call_profile_wins_and_is_reported():
    from membrane.adapters.mcp_server import MembraneMCPServer
    from membrane.profile import MembraneProfile

    seen = {}

    class FakeCompiler:
        def compile(self, **kw):
            seen["profile"] = kw["profile"]
            return SimpleNamespace(metadata={}, to_debug_dict=lambda: {"ok": True})

    default = MembraneProfile.preset("minimal")
    srv = MembraneMCPServer(FakeCompiler(), profile=default)
    asyncio.run(srv._compile_tool({"query": "q", "profile": "coding"}))
    assert seen["profile"] is not default
    assert srv._metrics_snapshot["last_profile"] == "coding"
    asyncio.run(srv._compile_tool({"query": "q"}))
    assert seen["profile"] is default


def test_mcp_sse_non_loopback_warns(caplog):
    from membrane.adapters import mcp_server as mod

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        mod.MembraneMCPServer(None, host="0.0.0.0")._warn_if_exposed()
        assert caplog.records
        caplog.clear()
        mod.MembraneMCPServer(None, host="127.0.0.1")._warn_if_exposed()
        assert not caplog.records


# 12. signals / metrics ------------------------------------------------------


def test_signal_bus_logs_handler_exception_and_supports_sync_and_async(caplog):
    from membrane.runtime.signals import Signal, SignalBus, SignalKind

    bus = SignalBus()

    def bad(sig):
        raise RuntimeError("boom")

    async def good(sig):
        return "ok"

    bus.subscribe(SignalKind.INTERRUPT, bad)
    bus.subscribe(SignalKind.INTERRUPT, good)
    with caplog.at_level(logging.ERROR):
        res = asyncio.run(bus.emit(Signal(kind=SignalKind.INTERRUPT)))
    assert res == "ok"
    assert any(r.exc_info for r in caplog.records)


def test_signal_bus_sync_returning_awaitable_and_handle_object():
    from membrane.runtime.signals import Signal, SignalBus, SignalKind

    class H:
        async def handle(self, sig):
            return "h"

    bus = SignalBus()
    bus.subscribe(SignalKind.LOD_SHIFT, H())
    assert asyncio.run(bus.emit(Signal(kind=SignalKind.LOD_SHIFT))) == "h"

    bus2 = SignalBus()
    bus2.subscribe(SignalKind.LOD_SHIFT, lambda s: asyncio.sleep(0, result="aw"))
    assert asyncio.run(bus2.emit(Signal(kind=SignalKind.LOD_SHIFT))) == "aw"


def test_metrics_collector_logs_callback_exception(caplog):
    from membrane.observability.collectors import MetricsCollector

    class Bad:
        def emit(self, event):
            raise RuntimeError("x")

    with caplog.at_level(logging.ERROR):
        MetricsCollector(Bad()).counter("c")
    assert any(r.exc_info for r in caplog.records)
