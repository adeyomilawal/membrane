"""benchmarks/hard_suite_live.py with a fake client: request shape and scoring, no network."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import hard_suite as hs
from benchmarks import hard_suite_live as live
from benchmarks.peer_comparison import CORE

BY_NAME = {s.name: s for s in hs.SCENARIOS}


class FakeClient:
    """Answers with the first code-like value in the last user message's context."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params):
        with self._lock:
            self.requests.append(params)
        prompt = str(params.get("system")) + str(params["messages"])
        reply = "D000000" if "release key" in prompt else prompt
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=reply)],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=100, cache_creation_input_tokens=20, cache_read_input_tokens=880
            ),
        )


def test_to_claude_converts_tools_and_places_breakpoints():
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "lookup_badge", "arguments": '{"trip": 3}'},
    }
    params = live.to_claude(
        [
            {"role": "system", "content": CORE + "\n\nRelevant memories:\n[1] fact"},
            {"role": "user", "content": "badge for trip 3?"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "B123456"},
            {"role": "assistant", "content": "It is B123456."},
            {"role": "user", "content": "and again?"},
        ],
        with_tools=True,
    )
    assert params["system"][0] == {"type": "text", "text": CORE, "cache_control": live.CACHE}
    assert "cache_control" not in params["system"][1]
    roles = [m["role"] for m in params["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert params["messages"][1]["content"][0] == {
        "type": "tool_use",
        "id": "c1",
        "name": "lookup_badge",
        "input": {"trip": 3},
    }
    assert params["messages"][2]["content"][0]["tool_use_id"] == "c1"
    # Memories in the system message: a history breakpoint would never be read.
    assert all("cache_control" not in b for m in params["messages"] for b in m["content"])
    assert params["tools"] == live.TOOLS


def test_core_only_system_gets_a_history_breakpoint():
    params = live.to_claude(
        [
            {"role": "system", "content": CORE},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
        with_tools=False,
    )
    assert params["system"] == [{"type": "text", "text": CORE, "cache_control": live.CACHE}]
    assert params["messages"][-2]["content"][-1]["cache_control"] == live.CACHE
    assert "cache_control" not in params["messages"][-1]["content"][-1]
    assert "tools" not in params


def test_live_run_scores_and_sums_usage_across_threads():
    sc = BY_NAME["tools"]
    client = FakeClient()
    strategies = {n: hs.STRATEGIES[n] for n in ("naive", "membrane")}
    scores = live.run_scenario(sc, 2, strategies, client, "claude-opus-5-5", workers=4)
    table = {n: live.summarize(s, scores["naive"]) for n, s in scores.items()}
    assert live.report(sc, table, 2, "claude-opus-5-5")
    assert len(client.requests) == 2 * 2 * sc.turns
    for row in table.values():
        assert row["requests"] == 2 * sc.turns
        assert row["prompt"] == 1000 * sc.turns
        assert row["read"] == pytest.approx(0.88)
        assert row["accuracy"]["memory"] == 1.0  # the fake echoes the prompt
    assert all(r["output_config"] == {"effort": "low"} for r in client.requests)
    assert all(r["tools"] == live.TOOLS for r in client.requests)


def test_paraphrase_memory_questions_are_sent_but_not_graded():
    sc = BY_NAME["paraphrase"]
    client = FakeClient()
    scores = live.run_scenario(sc, 1, {"membrane": hs.Membrane}, client, "m", workers=1)
    assert len(client.requests) == sc.turns
    row = live.summarize(scores["membrane"], scores["membrane"])
    assert set(row["accuracy"]) == {"history"}
    graded = sum(len(v) for s in scores["membrane"] for v in s.correct.values())
    assert graded < sc.turns
    assert row["wrong"] <= graded


def test_membrane_requests_carry_the_breakpoints_the_compiler_reports():
    sc = BY_NAME["tiny"]
    client = FakeClient()
    live.run_scenario(sc, 1, {"membrane": hs.Membrane}, client, "claude-opus-5-5", workers=1)
    assert live.for_model(hs.Membrane, "claude-opus-5-5").model == "anthropic/claude-opus-5-5"
    assert live.for_model(hs.Naive, "claude-opus-5-5") is hs.Naive
    marked_history = 0
    for request in client.requests:
        system = request["system"]
        assert system[0] == {"type": "text", "text": CORE, "cache_control": live.CACHE}
        assert all("cache_control" not in block for block in system[1:])
        blocks = [b for m in request["messages"] for b in m["content"]]
        assert "cache_control" not in blocks[-1]
        marked_history += sum("cache_control" in b for b in blocks)
    # At 3,000 tokens a cut comes every turn or two: history is marked on some turns only.
    assert 0 < marked_history < len(client.requests)
