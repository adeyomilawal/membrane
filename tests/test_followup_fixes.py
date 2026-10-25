"""Follow-ups to the October 2026 review: tokenizer, truncation, policy actions."""

from __future__ import annotations

import pytest

from membrane.compiler import ContextCompiler
from membrane.firewall import PreFlightContextFirewall
from membrane.memory_protocol import KernelMemoryObject, MemoryItem
from membrane.policies import PolicyEngine
from membrane.profile import MembraneProfile


def _cc(max_tokens: int = 2000, **kw) -> ContextCompiler:
    return ContextCompiler(
        max_tokens=max_tokens, model_id="openai/gpt-4o", include_interrupt_instruction=False, **kw
    )


def _policy(actions, condition="always"):
    return PolicyEngine.from_dict(
        {"policies": [{"name": "p", "condition": condition, "actions": actions}]}
    )


# --- Firewall uses the profile's tokenizer -----------------------------------


class _CharRegistry:
    """One token per character."""

    def count(self, model_id, text):
        return len(text)


def test_firewall_budget_uses_profile_tokenizer():
    mems = [MemoryItem(id=str(i), content="word " * 20) for i in range(20)]
    prof = MembraneProfile(tokenizer_registry=_CharRegistry())
    r = _cc(600).compile(memories=mems, user_message="hi", profile=prof)
    assert r.metadata["token_count"] == len(r.messages[0]["content"])
    assert r.metadata["token_count"] <= 600


def test_firewall_accepts_count_fn():
    fw = PreFlightContextFirewall(max_tokens=50, count_fn=len)
    final, _, total, _ = fw.run("core", "end", [("x" * 30, 2), ("y" * 30, 2)])
    assert total == len(final) <= 50


# --- Memory text truncation is configurable ----------------------------------


def test_memory_text_limit_can_be_disabled(monkeypatch):
    long_text = "detail " * 100
    monkeypatch.setenv("MEMBRANE_MEMORY_TEXT_MAX_CHARS", "0")
    r = _cc(5000).compile(memories=[MemoryItem(id="a", content=long_text)], user_message="hi")
    assert long_text.strip() in r.messages[0]["content"]


def test_memory_text_limit_default_unchanged():
    long_text = "detail " * 100
    r = _cc(5000).compile(memories=[MemoryItem(id="a", content=long_text)], user_message="hi")
    assert long_text.strip() not in r.messages[0]["content"]


def test_truncated_memories_are_reported():
    r = _cc(5000).compile(memories=[MemoryItem(id="a", content="detail " * 100)], user_message="hi")
    assert r.metadata["memories_truncated"] == 1


# --- Policy actions ------------------------------------------------------------


def _mixed():
    procs = [
        KernelMemoryObject(payload=f"step {i} " * 15, semantic_tier=1, id=f"p{i}")
        for i in range(3)
    ]
    facts = [
        KernelMemoryObject(payload=f"fact {i} " * 15, semantic_tier=2, id=f"f{i}")
        for i in range(10)
    ]
    return procs + facts


def test_set_budget_gives_each_kind_its_share_first():
    # Without a share, procedural (tier 1) is offered first and eats the budget.
    base = _cc(500).compile(memories=_mixed(), user_message="hi")
    shared = _cc(500).compile(
        memories=_mixed(),
        user_message="hi",
        profile=MembraneProfile(policies=_policy([{"set_budget": {"procedural": 0.2}}])),
    )
    procs_base = sum(1 for v in base.ref_map.values() if v.startswith("p"))
    procs_shared = sum(1 for v in shared.ref_map.values() if v.startswith("p"))
    assert procs_shared < procs_base
    assert len(shared.ref_map) >= len(base.ref_map)


def test_set_budget_unused_share_is_not_wasted():
    mems = [KernelMemoryObject(payload=f"fact {i}", semantic_tier=2, id=f"f{i}") for i in range(5)]
    r = _cc(2000).compile(
        memories=mems,
        user_message="hi",
        profile=MembraneProfile(policies=_policy([{"set_budget": {"semantic": 0.01}}])),
    )
    assert len(r.ref_map) == 5


def test_pin_memory_kind_is_kept_first():
    r = _cc(400).compile(
        memories=_mixed(),
        user_message="hi",
        profile=MembraneProfile(policies=_policy([{"pin_memory_kind": "semantic"}])),
    )
    kept = list(r.ref_map.values())
    assert kept and all(v.startswith("f") for v in kept)


def test_enable_reverses_a_disable():
    class Reranker:
        called = False

        def rerank(self, query, memories):
            Reranker.called = True
            return [(m, 1.0) for m in memories]

    pol = PolicyEngine.from_dict(
        {
            "policies": [
                {
                    "name": "off",
                    "condition": "always",
                    "priority": 2,
                    "actions": [{"disable": ["reranker"]}],
                },
                {
                    "name": "on",
                    "condition": "always",
                    "priority": 1,
                    "actions": [{"enable": ["reranker"]}],
                },
            ]
        }
    )
    _cc().compile(
        memories=_mixed(),
        user_message="hi",
        profile=MembraneProfile(policies=pol, reranker=Reranker()),
    )
    assert Reranker.called


@pytest.mark.parametrize(
    "action", [{"set_max_tokens": 3000}, {"override": {"max_tokens": 3000}}]
)
def test_policy_can_raise_budget_within_model_cap(action):
    r = _cc(500).compile(
        memories=_mixed(), user_message="hi", profile=MembraneProfile(policies=_policy([action]))
    )
    assert r.metadata["tokens_budget"] == 3000


def test_policy_budget_never_exceeds_model_cap():
    r = _cc(500).compile(
        memories=_mixed(),
        user_message="hi",
        profile=MembraneProfile(policies=_policy([{"set_max_tokens": 10_000_000}])),
    )
    assert r.metadata["tokens_budget"] == 128_000


def test_policy_without_condition_is_rejected():
    with pytest.raises(ValueError, match="condition"):
        PolicyEngine.from_dict(
            {"policies": [{"name": "p", "actions": [{"disable": ["reranker"]}]}]}
        )


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError, match="action"):
        _policy([{"set_budgte": {"semantic": 0.5}}])
