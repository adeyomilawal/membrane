"""Prompt-cache savings over a multi-turn chat.

Simulates a deterministic 20-turn conversation. Every turn retrieves different memories,
history grows, and the provider caches the longest byte prefix shared with the previous
request (at least 1024 tokens). Compares:

- ``naive``: the common pattern without Membrane: core + memories in the system message,
  then the full history and the question.
- ``end``: Membrane's default placement (full history, memories next to the question).
- ``system``: Membrane's ``"system"`` placement, which sends only the last turn of history.

Run: ``uv run python -m benchmarks.cache_savings``
"""

from __future__ import annotations

from membrane import ContextCompiler, MemoryItem
from membrane.firewall import count_tokens

MODEL = "openai/gpt-4o"
TURNS = 20
# Cached input price as a fraction of the normal input price.
CACHED_PRICE = {"Anthropic (0.1x)": 0.1, "OpenAI (0.5x)": 0.5}

CORE = "\n".join(
    f"Guideline {i}: answer precisely, cite [Mem:N] when you rely on memory, "
    f"and keep code examples runnable." for i in range(80)
)


def _memories(turn: int) -> list[MemoryItem]:
    return [
        MemoryItem(id=f"t{turn}m{k}", content=f"Fact {k} for turn {turn}: " + "detail " * 30)
        for k in range(5)
    ]


def _user(turn: int) -> str:
    return f"Question {turn}: " + "please explain " * 15


def _assistant(turn: int) -> str:
    return f"Answer {turn}: " + "here is the explanation " * 40


def _serialize(messages: list[dict]) -> str:
    return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def _common_prefix_tokens(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return count_tokens(a[:n])


def _naive_messages(turn: int, history: list[dict]) -> list[dict]:
    memories = "\n".join(m.content for m in _memories(turn))
    return [
        {"role": "system", "content": CORE + "\n\nRelevant memories:\n" + memories},
        *history,
        {"role": "user", "content": _user(turn)},
    ]


def run(placement: str) -> tuple[int, int]:
    """Return (total input tokens, total cache-served tokens) over the conversation."""
    cc = None
    if placement != "naive":
        cc = ContextCompiler(
            max_tokens=32_000, model_id=MODEL, core_content=CORE, memory_placement=placement
        )
    history: list[dict] = []
    previous = ""
    total = cached = 0
    for turn in range(TURNS):
        if cc is None:
            messages = _naive_messages(turn, history)
        else:
            messages = cc.compile(
                memories=_memories(turn), user_message=_user(turn), conversation_messages=history
            ).messages
        request = _serialize(messages)
        total += count_tokens(request)
        if previous:
            hit = _common_prefix_tokens(previous, request)
            if hit >= 1024:  # provider minimum for a cache hit
                cached += hit
        previous = request
        history += [
            {"role": "user", "content": _user(turn)},
            {"role": "assistant", "content": _assistant(turn)},
        ]
    return total, cached


def main() -> None:
    rows = {p: run(p) for p in ("naive", "end", "system")}
    print(f"{TURNS}-turn chat, {MODEL}, 5 new memories per turn\n")
    print(f"{'placement':<10} {'input tok':>10} {'cached tok':>11} {'cached %':>9}")
    for p, (total, cached) in rows.items():
        print(f"{p:<10} {total:>10,} {cached:>11,} {100 * cached / total:>8.0f}%")
    print("\nRelative input cost (naive = 100):")
    for label, price in CACHED_PRICE.items():
        costs = {p: (total - cached) + cached * price for p, (total, cached) in rows.items()}
        base = costs["naive"]
        cells = "  ".join(f"{p}={100 * c / base:.0f}" for p, c in costs.items())
        print(f"  {label:<17} {cells}")


if __name__ == "__main__":
    main()
