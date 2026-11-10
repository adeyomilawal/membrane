"""Membrane vs naive concatenation vs Priompt-style priority packing, under a token budget.

A deterministic 24-turn chat with a 6,000-token budget, so every strategy has to drop
something by the second half. Each turn the user asks for one fact (the "needle"):

- most turns: a deploy code that sits in one of 8 retrieved memories, at a random
  retrieval rank (usually near the top, sometimes not);
- every 4th turn: a locker number the user stated in their own message 10 turns earlier,
  so the needle sits in history.

Strategies, each given the same core prompt, memories, history and budget:

- ``naive``: core + memories in the system message, then history, then the question.
  Over budget, the oldest history messages go first, then the lowest-ranked memories
  (the usual ``trim_messages(strategy="last")`` pattern).
- ``priompt``: Priompt's selection algorithm, reimplemented here: every element has a
  priority and the renderer keeps everything at or above the lowest priority cutoff that
  fits (binary search). Core and question are pinned, memories rank above history, newer
  history ranks above older. Layout as typically written: memories in the system message.
  This is a reimplementation, not the ``priompt`` package itself. These priorities
  reproduce naive's drop order, so ``priompt`` and ``naive`` tie by construction.
- ``priompt-recent``: Priompt with the last 6 history messages ranked above all memories
  and older history below them, a common chat scheme.
- ``membrane-end``: Membrane defaults (``memory_placement="end"``, chunked history trim).
- ``membrane-system``: Membrane with ``memory_placement="system"`` (last turn only).

Offline metrics (no network), averaged over 5 seeds: input tokens (OpenAI chat framing:
content plus 3 tokens per message plus 3 for reply priming), simulated prompt-cache hits
(OpenAI rules: longest prefix shared with the previous request, at least 1,024 tokens, in 128-token
steps), relative input cost, and needle recall (is the answer anywhere in the prompt).
Recall is an upper bound on answer quality: a model cannot answer from text it never saw.

Live mode (``--live``, needs ``OPENAI_API_KEY``) sends every request to OpenAI and reports
answer accuracy (the reply contains the code) and the provider's real ``cached_tokens``.
Assistant turns stay scripted so all strategies see identical history. Cost with the
default ``gpt-4o-mini`` is roughly 5 strategies x 5 seeds x 24 requests x ~5k tokens,
about $0.50. Live mode has not been run by the maintainers yet: until it is, recall is the
only quality number and it is an upper bound.

Run: ``uv run python -m benchmarks.peer_comparison [--budget N] [--memories N] [--live]``

``--memories 40`` adds memory pressure: memories alone no longer fit, so the strategies'
priority rules decide which ones survive. The Priompt priorities here (memories by rank,
all above history) are one reasonable choice; other priority schemes give other results.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from membrane import ContextCompiler, MemoryItem
from membrane.firewall import count_tokens

MODEL = "openai/gpt-4o"  # Membrane model id: enables the caching path
TURNS = 24
BUDGET = 6_000
MEMORIES_PER_TURN = 8
HISTORY_QUESTION_EVERY = 4
HISTORY_LOOKBACK = 10
SEED = 1
CACHED_PRICE = {"cache reads 0.5x (OpenAI)": 0.5, "cache reads 0.1x (Anthropic)": 0.1}
STRATEGIES = ("naive", "priompt", "priompt-recent", "membrane-end", "membrane-system")
SEEDS = (1, 2, 3, 4, 5)

CORE = "\n".join(
    f"Guideline {i}: answer precisely, cite [Mem:N] when you rely on memory, "
    f"and keep code examples runnable." for i in range(60)
) + "\n\nWhen asked for a code or number, reply with that value only."

PROJECTS = [
    "Atlas", "Borealis", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Helix", "Iris",
    "Juniper", "Kestrel", "Lumen", "Mistral", "Nimbus", "Onyx", "Pulsar", "Quartz", "Raven",
    "Sierra", "Tundra", "Umber", "Vertex", "Willow", "Xenon", "Yarrow", "Zephyr",
]


# --- scenario ---------------------------------------------------------------------------


@dataclass
class Turn:
    user: str
    assistant: str
    needle: str
    needle_in: str  # "memory" | "history"
    memories: list[MemoryItem] = field(default_factory=list)


def _code(rng: random.Random) -> str:
    return f"{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"


def build_scenario(seed: int = SEED, memories_per_turn: int = MEMORIES_PER_TURN) -> list[Turn]:
    rng = random.Random(seed)
    deploy = {p: _code(rng) for p in PROJECTS}
    lockers = {t: str(rng.randint(100, 999)) for t in range(TURNS)}
    turns: list[Turn] = []
    for t in range(TURNS):
        mentions_locker = f" By the way, my locker number for trip {t} is {lockers[t]}."
        filler = " ".join(f"note{t}-{k}" for k in range(25))
        source = t - HISTORY_LOOKBACK
        if t % HISTORY_QUESTION_EVERY == HISTORY_QUESTION_EVERY - 1 and source >= 0:
            user = f"What locker number did I give you for trip {source}?"
            needle, needle_in, target = lockers[source], "history", None
        else:
            target = PROJECTS[t % len(PROJECTS)]
            user = f"What is the deploy code for project {target}?"
            needle, needle_in = deploy[target], "memory"
        user = f"{user} Context for later: {filler}.{mentions_locker}"

        others = [p for p in PROJECTS if p != target]
        rng.shuffle(others)
        # Retrieval may return a project more than once across turns; wrap to fill the list.
        picked = (others * (1 + memories_per_turn // len(others)))[
            : memories_per_turn - (1 if target else 0)
        ]
        if target:
            # Needle rank: usually near the top, with a long tail.
            weights = [max(1, 40 // (r + 1)) for r in range(memories_per_turn)]
            rank = rng.choices(range(memories_per_turn), weights=weights)[0]
            picked.insert(rank, target)
        memories = [
            MemoryItem(
                id=f"t{t}-{i}-{p}",
                content=f"Project {p} deploy code: {deploy[p]}. " + "Ops detail. " * 18,
                memory_type="semantic",
                priority=2,
            )
            for i, p in enumerate(picked)
        ]
        assistant = f"Noted for turn {t}. " + "Here is a careful, complete answer. " * 22
        turns.append(Turn(user, assistant, needle, needle_in, memories))
    return turns


# --- strategies -------------------------------------------------------------------------


# OpenAI chat framing (cookbook ``num_tokens_from_messages``): every message costs 3 tokens
# on top of its content, 1 more with a ``name``, and every reply is primed with 3 tokens.
TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
REPLY_PRIMING = 3


def _tokens(messages: list[dict]) -> int:
    """Billed prompt tokens for a chat request, the same count for every strategy."""
    total = REPLY_PRIMING
    for m in messages:
        total += TOKENS_PER_MESSAGE + count_tokens(str(m.get("content") or ""))
        total += TOKENS_PER_NAME if m.get("name") else 0
    return total


def _serialize(messages: list[dict]) -> str:
    return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def _system_with_memories(memories: list[MemoryItem]) -> dict:
    lines = "\n".join(f"[{i + 1}] {m.content}" for i, m in enumerate(memories))
    return {"role": "system", "content": CORE + "\n\nRelevant memories:\n" + lines}


def naive(turn: Turn, history: list[dict], budget: int) -> list[dict]:
    hist, mems = list(history), list(turn.memories)

    def build() -> list[dict]:
        return [_system_with_memories(mems), *hist, {"role": "user", "content": turn.user}]

    while _tokens(build()) > budget and hist:
        hist.pop(0)
    while _tokens(build()) > budget and mems:
        mems.pop()
    return build()


def priompt(turn: Turn, history: list[dict], budget: int) -> list[dict]:
    elements: list[tuple[int, str, object]] = [
        (3_000 - 10 * rank, "mem", m) for rank, m in enumerate(turn.memories)
    ]
    elements += [(1_000 + i, "hist", m) for i, m in enumerate(history)]
    return _priompt_render(turn, elements, budget)


def _priompt_render(
    turn: Turn, elements: list[tuple[int, str, object]], budget: int
) -> list[dict]:
    pinned = 1_000_000

    def render(cutoff: int) -> list[dict]:
        mems = [o for p, k, o in elements if k == "mem" and p >= cutoff]
        hist = [o for p, k, o in elements if k == "hist" and p >= cutoff]
        return [_system_with_memories(mems), *hist, {"role": "user", "content": turn.user}]

    # Priompt: lowest cutoff whose render fits; binary search over distinct priorities.
    levels = [*sorted({p for p, _, _ in elements}), pinned]
    lo, hi = 0, len(levels) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _tokens(render(levels[mid])) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return render(levels[lo])


def priompt_recent(turn: Turn, history: list[dict], budget: int) -> list[dict]:
    """Priompt with a common chat scheme: the last 6 history messages outrank all memories,
    memories rank by retrieval order, older history comes last."""
    recent = 6
    n = len(history)
    elements: list[tuple[int, str, object]] = [
        (3_000 - 10 * rank, "mem", m) for rank, m in enumerate(turn.memories)
    ]
    elements += [
        ((5_000 if i >= n - recent else 1_000) + i, "hist", m) for i, m in enumerate(history)
    ]
    return _priompt_render(turn, elements, budget)


def _membrane(placement: str, budget: int) -> Callable[[Turn, list[dict], int], list[dict]]:
    cc = ContextCompiler(
        max_tokens=budget,
        model_id=MODEL,
        core_content=CORE,
        include_interrupt_instruction=False,
        memory_placement=placement,
    )

    def run(turn: Turn, history: list[dict], budget: int) -> list[dict]:
        return cc.compile(
            memories=turn.memories, user_message=turn.user, conversation_messages=history
        ).messages

    return run


def strategy(name: str, budget: int) -> Callable[[Turn, list[dict], int], list[dict]]:
    if name.startswith("membrane-"):
        return _membrane(name.removeprefix("membrane-"), budget)
    return {"naive": naive, "priompt": priompt, "priompt-recent": priompt_recent}[name]


# --- measurement ------------------------------------------------------------------------


def _cache_hit(previous: str, request: str) -> int:
    n = 0
    for x, y in zip(previous, request, strict=False):
        if x != y:
            break
        n += 1
    shared = count_tokens(request[:n])
    if shared < 1024:
        return 0
    return 1024 + (shared - 1024) // 128 * 128


def _chat(messages: list[dict], model: str, key: str) -> tuple[str, int, int]:
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": 0, "max_tokens": 20}
    ).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            usage = data.get("usage", {})
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            return data["choices"][0]["message"]["content"] or "", usage["prompt_tokens"], cached
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


@dataclass
class Score:
    input_tokens: int = 0
    cached_tokens: int = 0
    over_budget: int = 0
    recall: dict[str, list[bool]] = field(default_factory=lambda: {"memory": [], "history": []})
    live_correct: dict[str, list[bool]] = field(
        default_factory=lambda: {"memory": [], "history": []}
    )
    live_input: int = 0
    live_cached: int = 0

    def cost(self, cached_price: float, live: bool = False) -> float:
        total, cached = (
            (self.live_input, self.live_cached) if live else (self.input_tokens, self.cached_tokens)
        )
        return (total - cached) + cached * cached_price


def run(name: str, turns: list[Turn], budget: int, live_model: str | None = None) -> Score:
    build = strategy(name, budget)
    key = os.environ.get("OPENAI_API_KEY", "") if live_model else ""
    score, history, previous = Score(), [], ""
    for turn in turns:
        messages = build(turn, history, budget)
        request = _serialize(messages)
        tokens = _tokens(messages)
        score.input_tokens += tokens
        score.over_budget += tokens > budget
        if previous:
            score.cached_tokens += _cache_hit(previous, request)
        previous = request
        score.recall[turn.needle_in].append(turn.needle in request)
        if live_model:
            reply, prompt_tokens, cached = _chat(messages, live_model, key)
            score.live_correct[turn.needle_in].append(turn.needle in reply)
            score.live_input += prompt_tokens
            score.live_cached += cached
        history += [
            {"role": "user", "content": turn.user},
            {"role": "assistant", "content": turn.assistant},
        ]
    return score


def _rate(xs: list[bool]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_all(
    seeds: tuple[int, ...], budget: int, memories: int, live_model: str | None = None
) -> dict[str, list[Score]]:
    """Every strategy on every seed: {strategy: [Score per seed]}."""
    scenarios = [build_scenario(seed, memories) for seed in seeds]
    return {n: [run(n, turns, budget, live_model) for turns in scenarios] for n in STRATEGIES}


def report(
    results: dict[str, list[Score]], live: bool, budget: int, memories: int, seeds: int
) -> str:
    first = next(iter(results.values()))[0]
    out = [
        f"{TURNS}-turn chat, budget {budget:,} tokens, {memories} memories per turn, "
        f"{seeds} seeds ({len(first.recall['memory'])} memory + "
        f"{len(first.recall['history'])} history questions per seed)",
        "Recall: mean over seeds (worst seed). Tokens counted with OpenAI chat framing.\n",
        f"{'strategy':<16} {'input tok':>10} {'cached':>7} {'over budget':>12} "
        f"{'recall mem':>13} {'recall hist':>13}",
    ]

    def recall(scores: list[Score], kind: str) -> str:
        rates = [_rate(s.recall[kind]) for s in scores]
        return f"{100 * sum(rates) / len(rates):.0f}% ({100 * min(rates):.0f}%)"

    for name, scores in results.items():
        tokens = sum(s.input_tokens for s in scores)
        cached = sum(s.cached_tokens for s in scores)
        over = sum(s.over_budget for s in scores)
        out.append(
            f"{name:<16} {tokens // len(scores):>10,} {100 * cached / tokens:>6.0f}% "
            f"{over:>12} {recall(scores, 'memory'):>13} {recall(scores, 'history'):>13}"
        )
    out.append("\nRelative input cost (naive = 100), simulated cache:")
    for label, price in CACHED_PRICE.items():
        base = sum(s.cost(price) for s in results["naive"])
        cells = "  ".join(
            f"{n}={100 * sum(s.cost(price) for s in sc) / base:.0f}" for n, sc in results.items()
        )
        out.append(f"  {label:<30} {cells}")
    if live:
        out.append("\nLive (OpenAI):")
        out.append(
            f"{'strategy':<16} {'prompt tok':>10} {'cached':>7} {'acc mem':>8} {'acc hist':>9}"
        )
        for name, scores in results.items():
            prompt = sum(s.live_input for s in scores)
            cached = sum(s.live_cached for s in scores)
            mem = [x for s in scores for x in s.live_correct["memory"]]
            hist = [x for s in scores for x in s.live_correct["history"]]
            out.append(
                f"{name:<16} {prompt // len(scores):>10,} {100 * cached / max(1, prompt):>6.0f}% "
                f"{100 * _rate(mem):>7.0f}% {100 * _rate(hist):>8.0f}%"
            )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live", action="store_true", help="call OpenAI (needs OPENAI_API_KEY)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for --live")
    parser.add_argument("--budget", type=int, default=BUDGET, help="token budget per request")
    parser.add_argument(
        "--memories", type=int, default=MEMORIES_PER_TURN, help="memories retrieved per turn"
    )
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="number of seeds")
    args = parser.parse_args()
    if args.live and not os.environ.get("OPENAI_API_KEY"):
        parser.error("--live needs OPENAI_API_KEY")
    seeds = tuple(range(1, args.seeds + 1))
    live_model = args.model if args.live else None
    results = run_all(seeds, args.budget, args.memories, live_model)
    print(report(results, args.live, args.budget, args.memories, len(seeds)))


if __name__ == "__main__":
    main()
