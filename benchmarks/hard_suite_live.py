"""Live run of the hard suite against Claude: real answers and real prompt-cache usage.

Sends every request the offline suite builds (``benchmarks/hard_suite.py``) to the Claude
API and reports, per scenario and strategy:

- ``accuracy``: the reply contains the fact asked for, by where the fact lives.
- ``wrong``: the reply states a different code (a confident wrong answer, not "I don't know").
- ``prompt``: prompt tokens per conversation as billed (uncached + cache writes + cache reads).
- ``read`` / ``write``: share of prompt tokens read from, and written to, the prompt cache.
- ``cost``: uncached + 1.25 x written + 0.1 x read, relative to the first strategy = 100.
  Cache reads are cheaper than 0.1x on some models, which widens the gaps.
- ``errors``: requests the API rejected (for example a tool result split from its call).

Every strategy gets a cache breakpoint after the core system prompt. Strategies whose
system message is the core alone also get one after the last history message; where
memories sit in the system message that breakpoint would be written every turn and never
read, so it is left out. The ``paraphrase`` scenario's memory questions name projects by an
alias no model can resolve, so they are sent (to keep the conversation intact) but not graded.

Assistant turns stay scripted so all strategies see the same
history. Budgets are enforced in tiktoken tokens for every strategy; Claude's own counts are
higher, by the same factor for everyone. Refusal fallbacks are left off on purpose: a
refusal counts as a wrong answer instead of being rerouted to another model.

Needs credentials the Anthropic SDK can find (``ANTHROPIC_API_KEY`` or ``ant auth login``).
Default: 4 strategies, 1 seed, about 1,300 requests and 7M prompt tokens.

Run: ``uv run --extra langchain --extra llamaindex --extra llm-anthropic \\
      python -m benchmarks.hard_suite_live [--model ID] [--seeds N] [--scenario NAME]
      [--strategies a,b,c] [--workers N]``
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from benchmarks import hard_suite as hs
from benchmarks.peer_comparison import CORE

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_STRATEGIES = "langchain,llamaindex-memory,membrane,membrane-lean"
CACHE = {"type": "ephemeral"}
TOOLS = [
    {
        "name": "lookup_badge",
        "description": "Look up the badge issued for a trip.",
        "input_schema": {
            "type": "object",
            "properties": {"trip": {"type": "integer"}},
            "required": ["trip"],
        },
    }
]
_CODE = re.compile(r"\b[DLBV]\d{6}\b")


def to_claude(messages: list[hs.Message], with_tools: bool) -> dict[str, Any]:
    """OpenAI-style chat messages as Messages API parameters, with cache breakpoints."""
    params: dict[str, Any] = {}
    rest = messages
    if messages and messages[0].get("role") == "system":
        text, rest = str(messages[0]["content"]), messages[1:]
        if text.startswith(CORE) and text[len(CORE) :].strip():
            # Core is stable across turns; whatever follows it (memories) is not.
            params["system"] = [
                {"type": "text", "text": CORE, "cache_control": CACHE},
                {"type": "text", "text": text[len(CORE) :]},
            ]
        else:
            params["system"] = [{"type": "text", "text": text, "cache_control": CACHE}]
    out: list[dict[str, Any]] = []
    for m in rest:
        if m.get("role") == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": str(m.get("content") or ""),
            }
            out.append({"role": "user", "content": [block]})
        elif m.get("tool_calls"):
            calls = [
                {
                    "type": "tool_use",
                    "id": c["id"],
                    "name": c["function"]["name"],
                    "input": json.loads(c["function"]["arguments"]),
                }
                for c in m["tool_calls"]
            ]
            out.append({"role": "assistant", "content": calls})
        else:
            text_block = {"type": "text", "text": str(m.get("content") or "")}
            out.append({"role": m["role"], "content": [text_block]})
    # A breakpoint after history only pays when everything before it is stable. With
    # memories in the system message it would be written every turn and never read.
    if len(out) >= 2 and len(params.get("system", [])) <= 1:
        out[-2]["content"][-1]["cache_control"] = CACHE
    params["messages"] = out
    if with_tools:
        params["tools"] = TOOLS
    return params


@dataclass
class LiveScore:
    uncached: int = 0
    written: int = 0
    read: int = 0
    correct: dict[str, list[bool]] = field(default_factory=dict)
    wrong: int = 0
    errors: int = 0
    refusals: int = 0
    requests: int = 0

    @property
    def prompt(self) -> int:
        return self.uncached + self.written + self.read

    @property
    def cost(self) -> float:
        return self.uncached + 1.25 * self.written + 0.1 * self.read


def run_live(
    cls: type, sc: hs.Scenario, turns: list[hs.Turn], client: Any, model: str
) -> LiveScore:
    """One conversation, turn by turn, so each request can read the previous one's cache."""
    import anthropic

    strategy, score = cls(sc.budget, sc), LiveScore()
    history: list[hs.Message] = []
    for turn in turns:
        messages, _ = strategy.build(turn, history)
        score.requests += 1
        reply = ""
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                output_config={"effort": "low"},
                **to_claude(messages, with_tools=bool(sc.tool_every)),
            )
        except anthropic.BadRequestError:
            score.errors += 1
        else:
            usage = response.usage
            score.uncached += usage.input_tokens
            score.written += usage.cache_creation_input_tokens or 0
            score.read += usage.cache_read_input_tokens or 0
            score.refusals += response.stop_reason == "refusal"
            reply = " ".join(b.text for b in response.content if b.type == "text")
        if not (sc.paraphrase and turn.kind == "memory"):
            right = all(n in reply for n in turn.needles)
            score.correct.setdefault(turn.kind, []).append(right)
            score.wrong += (not right) and bool(_CODE.search(reply))
        history = [*history, {"role": "user", "content": turn.user}, *turn.follow]
    return score


def summarize(scores: list[LiveScore], baseline: list[LiveScore]) -> dict[str, Any]:
    prompt = sum(s.prompt for s in scores) or 1
    kinds = sorted({k for s in scores for k in s.correct})
    accuracy = {}
    for kind in kinds:
        hits = [x for s in scores for x in s.correct.get(kind, [])]
        accuracy[kind] = sum(hits) / len(hits)
    everything = [x for s in scores for hits in s.correct.values() for x in hits]
    return {
        "prompt": prompt // len(scores),
        "read": sum(s.read for s in scores) / prompt,
        "write": sum(s.written for s in scores) / prompt,
        "cost": 100 * sum(s.cost for s in scores) / (sum(s.cost for s in baseline) or 1),
        "accuracy": accuracy,
        "accuracy_all": sum(everything) / len(everything),
        "wrong": sum(s.wrong for s in scores),
        "errors": sum(s.errors for s in scores),
        "refusals": sum(s.refusals for s in scores),
        "requests": sum(s.requests for s in scores),
    }


def report(sc: hs.Scenario, table: dict[str, dict[str, Any]], seeds: int, model: str) -> str:
    kinds = sorted({k for row in table.values() for k in row["accuracy"]})
    out = [
        f"\n== {sc.name}: {sc.about}",
        f"   {sc.turns} turns, budget {sc.budget:,}, {seeds} seeds, {model}",
        f"{'strategy':<18} {'prompt':>8} {'read':>5} {'write':>6} {'cost':>5} "
        + " ".join(f"{k:>8}" for k in kinds)
        + f" {'wrong':>6} {'errors':>7}",
    ]
    for name, row in table.items():
        out.append(
            f"{name:<18} {row['prompt']:>8,} {100 * row['read']:>4.0f}% "
            f"{100 * row['write']:>5.0f}% {row['cost']:>5.0f} "
            + " ".join(f"{100 * row['accuracy'][k]:>7.0f}%" for k in kinds)
            + f" {row['wrong']:>6} {row['errors']:>7}"
        )
    return "\n".join(out)


def overall(results: dict[str, dict[str, dict[str, Any]]]) -> str:
    names = list(next(iter(results.values())))
    out = [
        "\n== overall: mean over scenarios",
        f"{'strategy':<18} {'cost':>5} {'accuracy':>9} {'wrong':>6} {'errors':>7} {'refusals':>9}",
    ]
    for name in names:
        rows = [table[name] for table in results.values()]
        out.append(
            f"{name:<18} {sum(r['cost'] for r in rows) / len(rows):>5.0f} "
            f"{100 * sum(r['accuracy_all'] for r in rows) / len(rows):>8.0f}% "
            f"{sum(r['wrong'] for r in rows):>6} {sum(r['errors'] for r in rows):>7} "
            f"{sum(r['refusals'] for r in rows):>9}"
        )
    return "\n".join(out)


def run_scenario(
    sc: hs.Scenario, seeds: int, strategies: dict[str, type], client: Any, model: str, workers: int
) -> dict[str, list[LiveScore]]:
    """Conversations run in parallel; the turns inside each one stay in order."""
    scenarios = [hs.build_scenario(sc, seed) for seed in range(1, seeds + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            name: [pool.submit(run_live, cls, sc, turns, client, model) for turns in scenarios]
            for name, cls in strategies.items()
        }
        return {name: [f.result() for f in jobs] for name, jobs in futures.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES, help="comma-separated")
    parser.add_argument("--scenario", action="append", choices=[s.name for s in hs.SCENARIOS])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args(argv)

    import anthropic

    ready, missing = hs.available_strategies()
    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in ready]
    if unknown:
        parser.error(f"not available: {', '.join(unknown)} (not installed: {missing or 'none'})")
    strategies = {name: ready[name] for name in wanted}
    scenarios = [s for s in hs.SCENARIOS if not args.scenario or s.name in args.scenario]
    requests = sum(s.turns for s in scenarios) * len(strategies) * args.seeds
    print(
        f"{requests:,} requests to {args.model}: {len(strategies)} strategies, {args.seeds} seed(s)"
    )

    client = anthropic.Anthropic(max_retries=6)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for sc in scenarios:
        scores = run_scenario(sc, args.seeds, strategies, client, args.model, args.workers)
        first = next(iter(scores.values()))
        results[sc.name] = {n: summarize(s, first) for n, s in scores.items()}
        print(report(sc, results[sc.name], args.seeds, args.model), flush=True)
        if args.json:  # rewritten after every scenario, so a stopped run keeps its results
            with open(args.json, "w") as fh:
                json.dump({"model": args.model, "results": results}, fh, indent=2)
    if len(results) > 1:
        print(overall(results))


if __name__ == "__main__":
    main()
