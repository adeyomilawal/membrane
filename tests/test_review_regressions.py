"""Regression tests for bugs found in the October 2026 code review."""

from __future__ import annotations

import pytest

from membrane.compile_wiring import apply_classifier
from membrane.compiler import ContextCompiler, compile_anonymous, compile_prompt_async
from membrane.firewall import PreFlightContextFirewall, count_tokens
from membrane.memory_protocol import KernelMemoryObject, MemoryItem


def _cc(max_tokens: int = 2000, **kw) -> ContextCompiler:
    return ContextCompiler(max_tokens=max_tokens, include_interrupt_instruction=False, **kw)


def _facts(n: int) -> list:
    return [KernelMemoryObject(payload=f"fact number {i}", semantic_tier=2) for i in range(n)]


def _text(result) -> str:
    return "\n".join(str(m["content"]) for m in result.messages)


# --- Token budget -----------------------------------------------------------


@pytest.mark.parametrize("max_tokens", [300, 400, 700])
@pytest.mark.parametrize("directives", [None, ["be terse " * 30]])
def test_system_mode_stays_within_budget(max_tokens, directives):
    r = _cc(max_tokens, memory_placement="system").compile(
        memories=_facts(60), user_message="hello there", dynamic_directives=directives
    )
    assert r.metadata["token_count"] == count_tokens(r.messages[0]["content"])
    assert r.metadata["token_count"] <= max_tokens


@pytest.mark.parametrize("max_tokens", [300, 400, 700])
@pytest.mark.parametrize("directives", [None, ["be terse " * 30]])
def test_end_mode_stays_within_budget(max_tokens, directives):
    r = _cc(max_tokens).compile(
        memories=_facts(60), user_message="hello there", dynamic_directives=directives
    )
    assert r.metadata["token_count"] <= max_tokens
    if directives:
        assert "be terse" in r.messages[-1]["content"]


def test_directives_survive_trimming():
    r = _cc(400).compile(memories=_facts(60), user_message="hi", dynamic_directives=["KEEP_ME"])
    assert "KEEP_ME" in r.messages[0]["content"]


def test_anonymous_overrun_is_flagged():
    hist = [{"role": "user", "content": "word " * 3000}, {"role": "assistant", "content": "ok"}]
    r = compile_anonymous(user_message="q", conversation_messages=hist, max_tokens=1000)
    assert r.metadata["token_count"] > 1000
    assert r.metadata["budget_exceeded"] is True


def test_reserved_overrun_is_flagged_in_system_mode():
    r = _cc(50).compile(memories=_facts(3), user_message="word " * 200)
    assert r.metadata["budget_exceeded"] is True


def test_within_budget_is_not_flagged():
    r = _cc(2000).compile(memories=_facts(3), user_message="hi")
    assert not r.metadata.get("budget_exceeded")


# --- ref_map ----------------------------------------------------------------


def test_ref_inside_kept_memory_does_not_pull_in_dropped_memory():
    fw = PreFlightContextFirewall(max_tokens=60)
    segs = [("[Mem:1] see [Mem:2] for details", 1), ("[Mem:2] " + "long text " * 80, 3)]
    _, kept_refs, _, stats = fw.run("core", "end", segs, {1: "a", 2: "b"})
    assert stats["segments_kept"] == 1
    assert kept_refs == {1: "a"}


def test_compressed_middle_keeps_its_refs(monkeypatch):
    monkeypatch.setenv("MEMORY_PREFLIGHT_TOKEN_THRESHOLD", "1")
    mems = [MemoryItem(id="a", content="alpha"), MemoryItem(id="b", content="beta")]
    r = _cc().compile(
        memories=mems, user_message="x", compress_sync=lambda raw, t, u: "sum [Mem:1] [Mem:2]"
    )
    assert r.ref_map == {1: "a", 2: "b"}


# --- Serialization ----------------------------------------------------------


def test_procedural_kernel_object_keeps_its_payload():
    k = KernelMemoryObject(payload="Always run migrations before deploy", semantic_tier=1)
    r = _cc().compile(memories=[k], user_message="hi")
    assert "Always run migrations before deploy" in r.messages[0]["content"]


# --- Middle extras and reranker ---------------------------------------------


def test_middle_extras_keep_their_order():
    r = _cc().compile(memories=[], user_message="x", middle_extras=[("EXTRA_A", 2), ("EXTRA_B", 2)])
    text = r.messages[0]["content"]
    assert text.index("EXTRA_A") < text.index("EXTRA_B")


class _IndexReranker:
    """Scores later memories higher."""

    def rerank(self, query, memories):
        return [(m, float(i)) for i, m in enumerate(memories)]


def test_reranker_weights_apply_with_extras():
    mems = [MemoryItem(id="a", content="alpha low"), MemoryItem(id="b", content="beta high")]
    r = _cc().compile(
        memories=mems, user_message="x", reranker=_IndexReranker(), middle_extras=[("EX", 2)]
    )
    text = r.messages[0]["content"]
    assert text.index("beta high") < text.index("alpha low")


class _FlatReranker:
    def rerank(self, query, memories):
        return [(m, 1.0) for m in memories]


def test_reranker_keeps_tier_overrides():
    objs = _facts(5)
    tiers = {id(o): 1 for o in objs}
    plain = _cc().compile(memories=objs, user_message="x", tier_by_memory_id=tiers)
    ranked = _cc().compile(
        memories=objs, user_message="x", tier_by_memory_id=tiers, reranker=_FlatReranker()
    )
    assert plain.metadata["segments_kept"] == ranked.metadata["segments_kept"] == 3


# --- Classifier -------------------------------------------------------------


class _ProceduralClassifier:
    class _R:
        kind = "procedural"

    def classify(self, text):
        return self._R()


class _Obj:
    def __init__(self, content: str, memory_type: str = "semantic"):
        self.content = content
        self.memory_type = memory_type


def test_classifier_does_not_mutate_caller_objects():
    obj = _Obj("deploy steps")
    out = apply_classifier(_ProceduralClassifier(), [obj])
    assert obj.memory_type == "semantic"
    assert out[0].memory_type == "procedural"


# --- Lineage ----------------------------------------------------------------


class _Tracker:
    def __init__(self):
        self.included = None

    def record_event(self, *a, **k):
        pass

    def register_segment(self, *a, **k):
        pass

    def finalize(self, included_ids=None, **kwargs):
        self.included = set(included_ids or ())
        return None


def test_lineage_lists_only_kept_memories():
    tracker = _Tracker()
    mems = [MemoryItem(id=f"m{i}", content="word " * 60) for i in range(5)]
    r = _cc(300).compile(memories=mems, user_message="x", lineage_tracker=tracker)
    kept = set(r.ref_map.values())
    assert kept and len(kept) < 5
    assert tracker.included == kept


# --- Async compression fallback ---------------------------------------------


async def test_async_compression_falls_back_to_sync(monkeypatch):
    monkeypatch.setenv("MEMORY_PREFLIGHT_TOKEN_THRESHOLD", "1")

    async def broken(raw, target, user):
        raise RuntimeError("down")

    r = await _cc().compile_async(
        memories=[MemoryItem(id="a", content="alpha beta gamma")],
        user_message="x",
        compress_async=broken,
        compress_sync=lambda raw, t, u: "SYNCSUMMARY",
    )
    assert "SYNCSUMMARY" in r.messages[0]["content"]


# --- Provider registry ------------------------------------------------------


@pytest.mark.parametrize(
    "model", ["openai/gpt-4.1", "anthropic/claude-sonnet-4-5-20250929", "some/unknown-model"]
)
def test_unknown_model_does_not_shrink_user_budget(model):
    assert ContextCompiler(max_tokens=100_000, model_id=model).effective_max_tokens() == 100_000


def test_registry_returns_copies():
    from membrane import provider_registry as pr

    caps = pr.get_full_context_caps()
    caps["poison"] = 1
    assert "poison" not in pr.get_full_context_caps()


# --- compile_prompt_async without a user id ---------------------------------


async def test_anonymous_async_keeps_directives_and_placement():
    r = await compile_prompt_async(
        user_msg="q",
        messages_list=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok"},
        ],
        user_id=None,
        max_tokens=4000,
        dynamic_directives=["DIRECTIVE"],
        memory_placement="end",
    )
    assert "DIRECTIVE" in _text(r)
    assert [m["role"] for m in r.messages] == ["system", "user", "assistant", "user"]


# --- End placement ----------------------------------------------------------


def test_end_mode_skips_empty_system_message():
    r = _cc(memory_placement="end").compile(memories=_facts(2), user_message="hi")
    assert all(m["content"] for m in r.messages)
    assert r.messages[0]["role"] != "system" or r.messages[0]["content"]


# --- Redis ref_map key -------------------------------------------------------


class _FakeRedis:
    def __init__(self):
        self.store = {}

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def get(self, key):
        return self.store.get(key)


async def test_ref_map_key_is_scoped_to_user():
    from membrane.ref_map import get_ref_map_for_conversation, set_ref_map_for_conversation

    redis = _FakeRedis()
    await set_ref_map_for_conversation("c1", {1: "alice-mem"}, user_id="alice", redis_client=redis)
    await set_ref_map_for_conversation("c1", {1: "bob-mem"}, user_id="bob", redis_client=redis)
    alice, _ = await get_ref_map_for_conversation("c1", user_id="alice", redis_client=redis)
    assert alice == {1: "alice-mem"}


# --- Prompt-cache metadata ---------------------------------------------------


def test_cache_prefix_is_exactly_the_core():
    core = "Stable instructions. " * 400
    r = ContextCompiler(
        max_tokens=20000, model_id="openai/gpt-4o", core_content=core, memory_placement="system"
    ).compile(
        memories=_facts(5), user_message="hi"
    )
    md = r.metadata
    assert md["prompt_cache_used"] is True
    content = r.messages[0]["content"]
    prefix = content[: md["cacheable_prefix_chars"]]
    assert prefix.startswith("Stable instructions.")
    assert "fact number" not in prefix
    assert md["cacheable_prefix_tokens"] == count_tokens(prefix)


def test_short_core_does_not_claim_caching():
    r = ContextCompiler(
        max_tokens=4000, model_id="openai/gpt-4o", core_content="Be brief."
    ).compile(
        memories=_facts(3), user_message="hi"
    )
    assert not r.metadata.get("prompt_cache_used")
