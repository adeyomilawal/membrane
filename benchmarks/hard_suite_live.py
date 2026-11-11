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

Cache breakpoints: Membrane's requests are built by ``membrane.anthropic_request`` from the
breakpoints the compiler reports (after the core and after the history), compiled with the
live model's id so its minimum cacheable size applies.
The other strategies get a breakpoint after the core system prompt, and one after the last
history message when their system message is the core alone; where memories sit in the
system message that breakpoint would be written every turn and never read, so it is left
out. The ``paraphrase`` scenario's memory questions name projects by an alias no model can
resolve, so they are sent (to keep the conversation intact) but not graded.

Assistant turns stay scripted so all strategies see the same
history. Budgets are enforced in tiktoken tokens for every strategy; Claude's own counts are
higher, by the same factor for everyone. Refusal fallbacks are left off on purpose: a
refusal counts as a wrong answer instead of being rerouted to another model.

Needs credentials the Anthropic SDK can find (``ANTHROPIC_API_KEY`` or ``ant auth login``).
Default: 4 strategies, 1 seed, about 1,700 requests.

``--prices`` (dollars per million tokens: input, cache write, cache read, output) makes the
run report what it spent, and ``--max-usd`` stops it there: no request starts once the total
reaches the limit, and the scenario in progress is discarded.

Run: ``uv run --extra langchain --extra llamaindex --extra llm-anthropic \\
      python -m benchmarks.hard_suite_live [--model ID] [--seeds N] [--scenario NAME]
      [--strategies a,b,c] [--workers N] [--prices 4,5,0.2,20 --max-usd 30]``
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from benchmarks import hard_suite as hs
from benchmarks.peer_comparison import CORE
from membrane.cache_format import to_anthropic_params

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_STRATEGIES = "langchain,llamaindex-memory,membrane,membrane-lean"
CACHE = {"type": "ephemeral"}  # what membrane.cache_format places at a breakpoint
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


def peer_breakpoints(messages: list[hs.Message]) -> list[dict[str, Any]]:
    """Breakpoints for a strategy that reports none: after the core, and after the history
    when everything before it is stable."""
    if not messages or messages[0].get("role") != "system":
        return [{"message_index": len(messages) - 2}] if len(messages) >= 2 else []
    text = str(messages[0]["content"])
    if text.startswith(CORE) and text[len(CORE) :].strip():
        # Core is stable across turns; whatever follows it (memories) is not, and a
        # breakpoint after history would be written every turn and never read.
        return [{"message_index": 0, "chars": len(CORE)}]
    points: list[dict[str, Any]] = [{"message_index": 0}]
    if len(messages) >= 3:
        points.append({"message_index": len(messages) - 2})
    return points


def to_claude(
    messages: list[hs.Message],
    with_tools: bool,
    breakpoints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """OpenAI-style chat messages as Messages API parameters, with cache breakpoints:
    the ones given (Membrane reports its own), else :func:`peer_breakpoints`."""
    if breakpoints is None:
        breakpoints = peer_breakpoints(messages)
    params = to_anthropic_params(messages, breakpoints)
    if with_tools:
        params["tools"] = TOOLS
    return params


def for_model(cls: type, model: str) -> type:
    """A Membrane strategy compiled for the live model, so the breakpoints it reports use
    that model's cache prices and minimum cacheable size."""
    if not issubclass(cls, hs.Membrane):
        return cls
    model_id = model if "/" in model else f"anthropic/{model}"
    # Budgets stay in tiktoken tokens for every strategy, so the scale is fixed at 1.
    options = {**cls.options, "token_scale": 1.0}
    return type(cls.__name__, (cls,), {"model": model_id, "options": options})


class SpendLimitReached(RuntimeError):
    """The run's dollar limit was reached; no further request is sent."""


@dataclass
class Spend:
    """Running cost of a live run in dollars, shared by every conversation thread.

    ``prices`` are dollars per million tokens: plain input, cache write, cache read, output.
    With a ``limit`` no request starts once the total has reached it; requests already in
    flight still finish, so the run can end a little over.
    """

    prices: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    limit: float | None = None
    usd: float = 0.0
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        if self.limit is not None and self.usd >= self.limit:
            raise SpendLimitReached(f"${self.usd:.2f} spent, limit ${self.limit:.2f}")

    def add(self, usage: Any) -> None:
        plain, write, read, output = self.prices
        cost = (
            usage.input_tokens * plain
            + (usage.cache_creation_input_tokens or 0) * write
            + (usage.cache_read_input_tokens or 0) * read
            + (getattr(usage, "output_tokens", 0) or 0) * output
        ) / 1e6
        with self._lock:
            self.usd += cost


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
    cls: type,
    sc: hs.Scenario,
    turns: list[hs.Turn],
    client: Any,
    model: str,
    spend: Spend | None = None,
) -> LiveScore:
    """One conversation, turn by turn, so each request can read the previous one's cache."""
    import anthropic

    strategy, score = for_model(cls, model)(sc.budget, sc), LiveScore()
    history: list[hs.Message] = []
    for turn in turns:
        messages, _ = strategy.build(turn, history)
        result = getattr(strategy, "last_result", None)
        breakpoints = result.metadata.get("cache_breakpoints", []) if result else None
        if spend is not None:
            spend.check()
        score.requests += 1
        reply = ""
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                output_config={"effort": "low"},
                **to_claude(messages, bool(sc.tool_every), breakpoints),
            )
        except anthropic.BadRequestError:
            score.errors += 1
        else:
            usage = response.usage
            if spend is not None:
                spend.add(usage)
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
        f"{'strategy':<20} {'prompt':>8} {'read':>5} {'write':>6} {'cost':>5} "
        + " ".join(f"{k:>8}" for k in kinds)
        + f" {'wrong':>6} {'errors':>7}",
    ]
    for name, row in table.items():
        out.append(
            f"{name:<20} {row['prompt']:>8,} {100 * row['read']:>4.0f}% "
            f"{100 * row['write']:>5.0f}% {row['cost']:>5.0f} "
            + " ".join(f"{100 * row['accuracy'][k]:>7.0f}%" for k in kinds)
            + f" {row['wrong']:>6} {row['errors']:>7}"
        )
    return "\n".join(out)


def overall(results: dict[str, dict[str, dict[str, Any]]]) -> str:
    names = list(next(iter(results.values())))
    out = [
        "\n== overall: mean over scenarios",
        f"{'strategy':<20} {'cost':>5} {'accuracy':>9} {'wrong':>6} {'errors':>7} {'refusals':>9}",
    ]
    for name in names:
        rows = [table[name] for table in results.values()]
        out.append(
            f"{name:<20} {sum(r['cost'] for r in rows) / len(rows):>5.0f} "
            f"{100 * sum(r['accuracy_all'] for r in rows) / len(rows):>8.0f}% "
            f"{sum(r['wrong'] for r in rows):>6} {sum(r['errors'] for r in rows):>7} "
            f"{sum(r['refusals'] for r in rows):>9}"
        )
    return "\n".join(out)


def run_scenario(
    sc: hs.Scenario,
    seeds: int,
    strategies: dict[str, type],
    client: Any,
    model: str,
    workers: int,
    spend: Spend | None = None,
) -> dict[str, list[LiveScore]]:
    """Conversations run in parallel; the turns inside each one stay in order."""
    scenarios = [hs.build_scenario(sc, seed) for seed in range(1, seeds + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            name: [
                pool.submit(run_live, cls, sc, turns, client, model, spend) for turns in scenarios
            ]
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
    parser.add_argument(
        "--prices",
        help="dollars per million tokens as input,cache_write,cache_read,output; "
        "with it the run reports what it spent",
    )
    parser.add_argument(
        "--max-usd", type=float, help="stop sending requests at this spend (needs --prices)"
    )
    args = parser.parse_args(argv)
    if args.max_usd is not None and not args.prices:
        parser.error("--max-usd needs --prices")
    spend = None
    if args.prices:
        prices = tuple(float(x) for x in args.prices.split(","))
        if len(prices) != 4:
            parser.error("--prices takes four numbers")
        spend = Spend(prices=prices, limit=args.max_usd)  # type: ignore[arg-type]

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
        try:
            scores = run_scenario(
                sc, args.seeds, strategies, client, args.model, args.workers, spend
            )
        except SpendLimitReached as stop:
            print(f"\nStopped in {sc.name}: {stop}. Scenarios before it are complete.")
            break
        first = next(iter(scores.values()))
        results[sc.name] = {n: summarize(s, first) for n, s in scores.items()}
        print(report(sc, results[sc.name], args.seeds, args.model), flush=True)
        if args.json:  # rewritten after every scenario, so a stopped run keeps its results
            payload: dict[str, Any] = {"model": args.model, "results": results}
            if spend is not None:
                payload["spent_usd"] = round(spend.usd, 2)
            with open(args.json, "w") as fh:
                json.dump(payload, fh, indent=2)
        if spend is not None:
            print(f"   spent so far: ${spend.usd:.2f}", flush=True)
    if len(results) > 1:
        print(overall(results))


if __name__ == "__main__":
    main()
