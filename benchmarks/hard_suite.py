"""Hard benchmark suite: Membrane against real context-management products.

Eleven stress scenarios, five seeds each, every strategy fed the same core prompt, retrieved
memories, conversation history and token budget. Budgets are in tiktoken tokens for every
strategy: Membrane runs with ``token_scale=1.0``, so no per-model calibration shrinks or
grows its budget. All offline and deterministic.

Products under test:

- ``naive``: core + memories in the system message, then history, then the question; over
  budget it pops the oldest message until it fits (a hand-rolled loop, no library).
- ``langchain``: LangChain ``trim_messages(strategy="last", start_on="human",
  include_system=True)`` from ``langchain-core``, with an exact billed-token counter.
- ``llamaindex-buffer``: LlamaIndex ``ChatMemoryBuffer`` with a token limit.
- ``llamaindex-memory``: LlamaIndex ``Memory`` (the current API): 70% of the limit for chat
  history, flushed in chunks. No long-term memory blocks are configured, because those
  need an LLM or a vector store.
- ``priompt`` / ``priompt-recent``: Priompt's priority-cutoff search, reimplemented (the
  package is JavaScript). One element per conversation turn. ``priompt`` ranks memories
  above history; ``priompt-recent`` ranks the last three turns above memories.
- ``priompt-node`` / ``priompt-node-recent``: the real ``@anysphere/priompt`` package, run
  through Node with the same elements and priorities. Needs ``node`` on the path and
  ``npm install`` run once in ``benchmarks/priompt_node``; skipped otherwise. The package
  decides the priority cutoff with its own renderer and token counts.
- ``membrane``: ``ContextCompiler`` defaults.
- ``membrane-lean``: ``recall_evicted_history=False``.
- ``membrane-embed`` / ``membrane-lean-embed``: the two above with ``embedder=`` set to
  fastembed's BAAI/bge-small-en-v1.5 (``--extra lod``; the model is downloaded on first
  use). Not part of the default run: name them in ``--strategies``.

For LangChain and LlamaIndex the retrieved memories go into the system message, as in
their RAG chat patterns. A user of either can move memories next to the question by hand
and get the same caching Membrane gets by default; this suite measures the defaults.

No library other than Membrane enforces the whole budget (memories included), so each
gets the same small glue an app would need: if the request is still over budget, drop the
oldest turns, then the lowest-ranked memories. ``over`` counts the turns where the product
needed that glue.

Metrics per scenario:

- ``tokens``: billed prompt tokens per conversation (OpenAI chat framing).
- ``cached``: share served from a simulated prompt cache (longest shared prefix with the
  previous request, at least 1,024 tokens, in 128-token steps).
- ``cost``: input cost relative to naive = 100, with cache reads at 0.5x and 0.1x.
- ``recall``: share of questions whose answer is somewhere in the prompt, by where the
  fact lives. An upper bound on accuracy, not accuracy.

Membrane picks which memories and which dropped history to keep partly by the words they
share with the question. In most scenarios the question names what it asks for ("project
Atlas", "trip 12"), which helps that. ``paraphrase`` removes the help: its questions share
no rare word with the facts, so only retrieval rank and recency are left.

Facts in most scenarios are one sentence, which flatters anything that keeps one sentence
per message. ``split-fact`` spreads each history fact over two sentences, ``spread-fact``
over two or three in varied wording (sometimes with an unrelated sentence in between), and
``spread-reworded`` also asks for it in other words ("journey twelve" for "trip 12"). In
those the recall check needs every sentence that carries part of the fact.
- ``over``: turns over budget before glue. ``invalid``: requests with a tool result
  separated from its call, which providers reject.
- ``ms``: median build time per turn. The harness pre-counts tokens for naive, LangChain
  and Priompt, so their times are lower bounds; Membrane and LlamaIndex tokenize inside.
  ``priompt-node`` times include a round trip to the Node process.

Run: ``uv run --extra langchain --extra llamaindex python -m benchmarks.hard_suite``
Options: ``--scenario NAME`` (repeatable), ``--seeds N``, ``--strategies a,b``, ``--json PATH``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

from benchmarks.peer_comparison import CORE, PROJECTS
from membrane import ContextCompiler, MemoryItem
from membrane.firewall import count_tokens

MODEL = "openai/gpt-4o"
Message = dict[str, Any]

# --- scenarios --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    about: str
    turns: int
    budget: int
    memories: int
    lookbacks: tuple[int, ...] = (10,)
    history_every: int = 4  # every Nth turn asks for a fact from history
    tool_every: int = 0  # every Nth turn is answered through a tool call
    uniform_rank: bool = False  # the needed memory is equally likely at any rank
    multi: bool = False  # every question needs a memory fact and a history fact
    paste_turns: tuple[int, ...] = ()  # turns where the user pastes a long document
    paraphrase: bool = False  # questions share no rare word with the fact they ask for
    split_fact: bool = False  # the history fact is spread over two sentences of its message
    spread_fact: bool = False  # ... over two or three sentences, in one of several wordings
    reworded: bool = False  # history questions share no rare word with the fact


SCENARIOS = (
    Scenario(
        "marathon", "80 turns, facts asked 3 to 30 turns later", 80, 8_000, 8,
        lookbacks=(3, 8, 15, 30),
    ),
    Scenario(
        "tools", "every other turn uses a tool; facts live in tool results", 40, 6_000, 8,
        lookbacks=(1, 6, 11), history_every=2, tool_every=2,
    ),
    Scenario(
        "flood", "up to 40 memories per turn, the needed one at a random rank", 30, 5_000, 40,
        lookbacks=(4,), uniform_rank=True,
    ),
    Scenario(
        "multi-fact", "every answer needs one memory fact and one history fact", 30, 6_000, 12,
        lookbacks=(3, 8), multi=True,
    ),
    Scenario(
        "paste", "three 2,500-token pastes, asked about 2 and 8 turns later", 30, 6_000, 8,
        lookbacks=(5,), paste_turns=(5, 14, 23),
    ),
    Scenario(
        "tiny", "3,000-token budget, half of it system prompt", 30, 3_000, 8, lookbacks=(2, 5)
    ),
    Scenario(
        "paraphrase", "flood, but questions share no wording with the facts", 30, 5_000, 40,
        lookbacks=(4, 12), paraphrase=True,
    ),
    Scenario("roomy", "24,000-token budget, nothing needs dropping", 40, 24_000, 8),
    Scenario(
        "split-fact", "history facts spread over two sentences of one message", 40, 6_000, 8,
        lookbacks=(3, 8, 15), history_every=2, split_fact=True,
    ),
    Scenario(
        "spread-fact", "history facts spread over two or three sentences, varied wording",
        40, 6_000, 8, lookbacks=(3, 8, 15), history_every=2, spread_fact=True,
    ),
    Scenario(
        "spread-reworded", "spread-fact, and history questions use other words", 40, 6_000, 8,
        lookbacks=(3, 8, 15), history_every=2, spread_fact=True, reworded=True,
    ),
)

# Ways a person spreads one fact over several sentences. {t} trip, {g} gate, {c} locker code.
# The trip is only named in the first sentence and the code only in the last.
_SPREAD = (
    "Trip {t} went through gate {g}. It ended at a locker whose code is",
    "Trip {t} went through gate {g}. The queue was short that morning. My locker code there was",
    "For trip {t} I used gate {g}. My bags went into a locker. Its code is",
    "I took gate {g} on trip {t}. That was the quiet one. Afterwards I wrote down the locker "
    "code, which is",
    "Gate {g} was the one for trip {t}. The locker I got there opens with",
)


@dataclass
class Turn:
    user: str
    follow: list[Message]  # what joins history after the user message
    needles: tuple[str, ...]
    kind: str  # memory | history | tool | paste | multi
    memories: list[MemoryItem] = field(default_factory=list)
    # What must be in the prompt for the answer to be derivable, when that is more than the
    # answer itself (a fact split over two sentences needs both).
    evidence: tuple[str, ...] = ()


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
_ONES += ["eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen"]
_ONES += ["eighteen", "nineteen"]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _spell(n: int) -> str:
    """0-99 in words, so a paraphrased question never repeats a number from the fact."""
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10 - 2] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])


def build_scenario(sc: Scenario, seed: int) -> list[Turn]:
    rng = random.Random(f"{sc.name}-{seed}")
    used: set[str] = set()

    def fresh(prefix: str) -> str:
        while True:
            value = f"{prefix}{rng.randrange(10**5, 10**6)}"
            if value not in used:
                used.add(value)
                return value

    deploy = {p: fresh("D") for p in PROJECTS}
    locker = [fresh("L") for _ in range(sc.turns)]
    badge = [fresh("B") for _ in range(sc.turns)]
    vault = {p: fresh("V") for p in sc.paste_turns}
    gate = [fresh("G") for _ in range(sc.turns)] if sc.split_fact or sc.spread_fact else []

    def is_tool_turn(t: int) -> bool:
        return bool(sc.tool_every) and t % sc.tool_every == 0

    turns: list[Turn] = []
    for t in range(sc.turns):
        target: str | None = PROJECTS[t % len(PROJECTS)]
        source = t - sc.lookbacks[(t // sc.history_every) % len(sc.lookbacks)]
        pasted = next((p for p in sc.paste_turns if t - p in (2, 8)), None)
        asks_history = source >= 0 and t % sc.history_every == sc.history_every - 1
        if pasted is not None:
            ask, needles, kind, target = (
                f"What is the vault code in document {pasted}?", (vault[pasted],), "paste", None
            )
        elif sc.multi and source >= 0:
            ask = (
                f"What is the deploy code for project {target}, and what locker code did I "
                f"give you for trip {source}?"
            )
            needles, kind = (deploy[target], locker[source]), "multi"
        elif asks_history and is_tool_turn(source):
            ask, needles, kind, target = (
                f"What badge did the lookup return for trip {source}?", (badge[source],), "tool",
                None,
            )
        elif asks_history:
            ask, needles, kind, target = (
                f"What locker code did I give you for trip {source}?", (locker[source],),
                "history", None,
            )
        else:
            ask = f"What is the deploy code for project {target}?"
            needles, kind = (deploy[target],), "memory"
        if sc.paraphrase:
            # Same facts, asked in words that appear nowhere in the memory or the history.
            ask = (
                f"Which storage combination went with journey {_spell(source)}?"
                if kind == "history"
                else f"Which release key belongs to initiative {_spell(PROJECTS.index(target))}?"
            )
        if sc.reworded and kind == "history":
            ask = f"Which storage combination went with journey {_spell(source)}?"

        filler = " ".join(f"note{t}-{k}" for k in range(rng.randint(5, 60)))
        evidence: tuple[str, ...] = ()
        if sc.split_fact:
            # The trip is named in one sentence and the code in the next; either alone is
            # useless, so both must reach the prompt.
            fact = f"Trip {t} went through gate {gate[t]}. The locker code I used there is"
            if kind == "history":
                evidence = (gate[source], locker[source])
        elif sc.spread_fact:
            fact = rng.choice(_SPREAD).format(t=t, g=gate[t])
            if kind == "history":
                evidence = (gate[source], locker[source])
        else:
            fact = f"My locker code for trip {t} is"
        user = f"{ask} Context for later: {filler}. {fact} {locker[t]}."
        if t in sc.paste_turns:
            clauses = [f"Clause {i} of the agreement stays in force." for i in range(250)]
            clauses.insert(rng.randrange(250), f"The vault code in document {t} is {vault[t]}.")
            user += " Here is document " + str(t) + ": " + " ".join(clauses)

        reply = f"Noted for turn {t}. " + "A careful, complete answer. " * rng.randint(8, 30)
        follow: list[Message] = [{"role": "assistant", "content": reply}]
        if is_tool_turn(t):
            call = {
                "id": f"call_{t}",
                "type": "function",
                "function": {"name": "lookup_badge", "arguments": json.dumps({"trip": t})},
            }
            result = f"Badge for trip {t}: {badge[t]}. " + "Record field. " * rng.randint(20, 80)
            follow = [
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": result},
                *follow,
            ]

        count = rng.randint(max(1, sc.memories // 2), sc.memories)
        others = [p for p in PROJECTS if p != target]
        rng.shuffle(others)
        picked = (others * (1 + count // len(others)))[: count - (1 if target else 0)]
        if target:
            weights = [1 if sc.uniform_rank else max(1, 40 // (r + 1)) for r in range(count)]
            picked.insert(rng.choices(range(count), weights=weights)[0], target)
        memories = [
            MemoryItem(
                id=f"t{t}-{i}-{p}",
                content=f"Project {p} deploy code: {deploy[p]}. " + "Ops detail. " * 18,
                memory_type="semantic",
                priority=2,
            )
            for i, p in enumerate(picked)
        ]
        turns.append(Turn(user, follow, needles, kind, memories, evidence))
    return turns


# --- token accounting -------------------------------------------------------------------

_count = lru_cache(maxsize=None)(count_tokens)


def message_tokens(m: Message) -> int:
    """Billed tokens for one chat message: 3 for framing, content, and any tool calls."""
    n = 3 + _count(str(m.get("content") or ""))
    if m.get("tool_calls"):
        n += _count(json.dumps(m["tool_calls"], sort_keys=True))
    return n


def billed(messages: list[Message]) -> int:
    return 3 + sum(message_tokens(m) for m in messages)


def cache_hit(previous: list[Message], request: list[Message]) -> int:
    """Tokens a prefix cache would serve: whole shared messages plus the shared start of
    the first message that differs."""
    shared = 0
    for a, b in zip(previous, request, strict=False):
        if a == b:
            shared += message_tokens(a)
            continue
        if a.get("role") == b.get("role"):
            x, y = str(a.get("content") or ""), str(b.get("content") or "")
            shared += _count(os.path.commonprefix([x, y]))
        break
    if shared < 1024:
        return 0
    return 1024 + (shared - 1024) // 128 * 128


def orphaned_tools(messages: list[Message]) -> bool:
    """True when a tool result is missing its call or a call is missing its result."""
    open_calls: set[str] = set()
    for m in messages:
        if m.get("role") == "tool":
            if m.get("tool_call_id") not in open_calls:
                return True
            open_calls.discard(m["tool_call_id"])
            continue
        if open_calls:
            return True
        open_calls = {c["id"] for c in m.get("tool_calls") or []}
    return bool(open_calls)


# --- strategies -------------------------------------------------------------------------


def system_with_memories(memories: list[MemoryItem]) -> Message:
    if not memories:
        return {"role": "system", "content": CORE}
    lines = "\n".join(f"[{i + 1}] {m.content}" for i, m in enumerate(memories))
    return {"role": "system", "content": CORE + "\n\nRelevant memories:\n" + lines}


def _assemble(memories: list[MemoryItem], history: list[Message], turn: Turn) -> list[Message]:
    return [system_with_memories(memories), *history, {"role": "user", "content": turn.user}]


def _turn_starts(history: list[Message]) -> list[int]:
    return [i for i, m in enumerate(history) if m.get("role") == "user"]


def _fit(
    memories: list[MemoryItem], history: list[Message], turn: Turn, budget: int
) -> tuple[list[Message], bool]:
    """Glue around a product that cannot enforce the whole budget: drop the oldest turns,
    then the lowest-ranked memories (naive's order). Returns (messages, glue was needed)."""
    messages = _assemble(memories, history, turn)
    if billed(messages) <= budget:
        return messages, False
    while history and billed(_assemble(memories, history, turn)) > budget:
        starts = _turn_starts(history)
        history = history[starts[1] :] if len(starts) > 1 else []
    memories = list(memories)
    while memories and billed(_assemble(memories, history, turn)) > budget:
        memories.pop()
    return _assemble(memories, history, turn), True


class Naive:
    def __init__(self, budget: int, sc: Scenario) -> None:
        self.budget = budget

    def build(self, turn: Turn, history: list[Message]) -> tuple[list[Message], bool]:
        history, memories = list(history), list(turn.memories)
        fixed = billed(_assemble(memories, [], turn))
        room = self.budget - fixed
        sizes = [message_tokens(m) for m in history]
        total, start = sum(sizes), 0
        while start < len(history) and total > room:
            total -= sizes[start]
            start += 1
        history = history[start:]
        while memories and billed(_assemble(memories, history, turn)) > self.budget:
            memories.pop()
        return _assemble(memories, history, turn), False


class LangChainTrim:
    """``langchain_core.messages.trim_messages`` with the settings its docs recommend."""

    def __init__(self, budget: int, sc: Scenario) -> None:
        from langchain_core.messages import convert_to_messages, trim_messages

        self.budget = budget
        self._convert, self._trim = convert_to_messages, trim_messages
        self._lc: list[Any] = []  # converted history, grows with the conversation

    def build(self, turn: Turn, history: list[Message]) -> tuple[list[Message], bool]:
        for i in range(len(self._lc), len(history)):
            self._lc.extend(self._convert([{**history[i], "id": f"h{i}"}]))
        system = system_with_memories(turn.memories)
        question = {"role": "user", "content": turn.user}
        edges = self._convert([{**system, "id": "system"}, {**question, "id": "question"}])
        size = {"system": message_tokens(system), "question": message_tokens(question)}
        size.update({f"h{i}": message_tokens(m) for i, m in enumerate(history)})
        kept = self._trim(
            [edges[0], *self._lc, edges[1]],
            max_tokens=self.budget,
            token_counter=lambda msgs: 3 + sum(size[m.id] for m in msgs),
            strategy="last",
            start_on="human",
            include_system=True,
        )
        ids = [m.id for m in kept if m.id not in ("system", "question")]
        return _fit(turn.memories, [history[int(i[1:])] for i in ids], turn, self.budget)


def _llama_messages(history: list[Message], start: int = 0) -> list[Any]:
    from llama_index.core.llms import ChatMessage

    return [
        ChatMessage(
            role=m["role"],
            content=m.get("content") or "",
            additional_kwargs={
                "idx": i,
                **{k: m[k] for k in ("tool_calls", "tool_call_id") if k in m},
            },
        )
        for i, m in enumerate(history[start:], start)
    ]


class LlamaIndexBuffer:
    """``llama_index.core.memory.ChatMemoryBuffer`` with ``token_limit`` = the budget."""

    def __init__(self, budget: int, sc: Scenario) -> None:
        import tiktoken
        from llama_index.core.memory import ChatMemoryBuffer

        self.budget = budget
        self._buffer = ChatMemoryBuffer
        self._encode = tiktoken.get_encoding("cl100k_base").encode
        self._li: list[Any] = []

    def build(self, turn: Turn, history: list[Message]) -> tuple[list[Message], bool]:
        self._li.extend(_llama_messages(history, len(self._li)))
        buffer = self._buffer.from_defaults(
            chat_history=list(self._li), token_limit=self.budget, tokenizer_fn=self._encode
        )
        reserved = billed(_assemble(turn.memories, [], turn))
        try:
            kept = buffer.get(initial_token_count=reserved)
        except ValueError:  # system + question alone exceed the limit
            kept = []
        chosen = [history[m.additional_kwargs["idx"]] for m in kept]
        return _fit(turn.memories, chosen, turn, self.budget)


class LlamaIndexMemory:
    """``llama_index.core.memory.Memory``: history gets 70% of what the core leaves, and is
    flushed in chunks of half that. No long-term memory blocks."""

    _sessions = 0

    def __init__(self, budget: int, sc: Scenario) -> None:
        import tiktoken
        from llama_index.core.memory import Memory

        self.budget = budget
        limit = max(200, budget - _count(CORE))
        LlamaIndexMemory._sessions += 1
        self._memory = Memory.from_defaults(
            session_id=f"bench-{LlamaIndexMemory._sessions}",
            token_limit=limit,
            chat_history_token_ratio=0.7,
            token_flush_size=max(100, int(limit * 0.7 / 2)),
            tokenizer_fn=tiktoken.get_encoding("cl100k_base").encode,
        )
        self._seen = 0

    def build(self, turn: Turn, history: list[Message]) -> tuple[list[Message], bool]:
        if len(history) > self._seen:
            self._memory.put_messages(_llama_messages(history, self._seen))
            self._seen = len(history)
        kept = [m for m in self._memory.get() if "idx" in m.additional_kwargs]
        chosen = [history[m.additional_kwargs["idx"]] for m in kept]
        return _fit(turn.memories, chosen, turn, self.budget)


class Priompt:
    """Priompt's renderer: keep every element at or above the lowest priority cutoff that
    fits. One element per turn, so a tool call is never split from its result."""

    recent_turns = 0  # turns ranked above all memories

    def __init__(self, budget: int, sc: Scenario) -> None:
        self.budget = budget

    def _elements(self, turn: Turn, history: list[Message]) -> list[tuple[int, str, Any]]:
        starts = [*_turn_starts(history), len(history)]
        groups = [history[a:b] for a, b in pairwise(starts)]
        first_recent = len(groups) - self.recent_turns
        elements: list[tuple[int, str, Any]] = [
            (3_000 - 10 * rank, "mem", m) for rank, m in enumerate(turn.memories)
        ]
        elements += [
            ((5_000 if i >= first_recent else 1_000) + i, "hist", g) for i, g in enumerate(groups)
        ]
        return elements

    @staticmethod
    def _render(elements: list[tuple[int, str, Any]], cutoff: float, turn: Turn) -> list[Message]:
        mems = [o for p, k, o in elements if k == "mem" and p >= cutoff]
        hist = [m for p, k, g in elements if k == "hist" and p >= cutoff for m in g]
        return _assemble(mems, hist, turn)

    def build(self, turn: Turn, history: list[Message]) -> tuple[list[Message], bool]:
        elements = self._elements(turn, history)
        levels = [*sorted({p for p, _, _ in elements}), 1_000_000]
        lo, hi = 0, len(levels) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if billed(self._render(elements, levels[mid], turn)) <= self.budget:
                hi = mid
            else:
                lo = mid + 1
        return self._render(elements, levels[lo], turn), False


class PriomptRecent(Priompt):
    recent_turns = 3


class PriomptNode(Priompt):
    """The real ``@anysphere/priompt`` package, run through Node (``benchmarks/priompt_node``).

    The same elements and priorities as :class:`Priompt`, but the cutoff comes from the
    package's own ``render`` with its own tokenizer (cl100k) and message accounting. The
    request is then assembled here from that cutoff, and gets the same glue as the other
    products when the package's count and the billed count disagree."""

    _dir = Path(__file__).resolve().parent / "priompt_node"
    _proc: ClassVar[Any] = None
    _lock: ClassVar[Any] = threading.Lock()

    def __init__(self, budget: int, sc: Scenario) -> None:
        super().__init__(budget, sc)
        if not shutil.which("node") or not (self._dir / "node_modules/@anysphere/priompt").is_dir():
            raise ImportError("node and `npm install` in benchmarks/priompt_node are needed")

    @classmethod
    def _ask(cls, request: dict[str, Any]) -> dict[str, Any]:
        with cls._lock:  # one Node process for the whole run; requests are answered in order
            if cls._proc is None or cls._proc.poll() is not None:
                cls._proc = subprocess.Popen(
                    ["node", "render.cjs"], cwd=cls._dir, text=True, encoding="utf-8",
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                )
                atexit.register(cls._proc.terminate)
            cls._proc.stdin.write(json.dumps(request) + "\n")
            cls._proc.stdin.flush()
            return json.loads(cls._proc.stdout.readline())

    @staticmethod
    def _plain(m: Message) -> dict[str, str]:
        """A chat message as role, text and name: a tool call is sent as its JSON."""
        content = m.get("content")
        if m.get("tool_calls"):
            content = json.dumps(m["tool_calls"], sort_keys=True)
        out = {"role": m["role"], "content": str(content or "")}
        return {**out, "name": "lookup_badge"} if m["role"] == "tool" else out

    def build(self, turn: Turn, history: list[Message]) -> tuple[list[Message], bool]:
        elements = self._elements(turn, history)
        memories = [(p, o) for p, k, o in elements if k == "mem"]
        reply = self._ask(
            {
                "budget": self.budget,
                "core": CORE,
                "header": "\n\nRelevant memories:",
                "memories": [
                    {"prio": p, "text": f"\n[{i + 1}] {m.content}"}
                    for i, (p, m) in enumerate(memories)
                ],
                "history": [
                    {"prio": p, "messages": [self._plain(m) for m in g]}
                    for p, k, g in elements
                    if k == "hist"
                ],
                "question": turn.user,
            }
        )
        # An error means even the core and the question are over the limit: keep nothing else.
        cutoff = reply.get("cutoff", 1_000_000)
        kept = self._render(elements, cutoff, turn)
        if billed(kept) <= self.budget:
            return kept, False
        mems = [o for p, k, o in elements if k == "mem" and p >= cutoff]
        hist = [m for p, k, g in elements if k == "hist" and p >= cutoff for m in g]
        return _fit(mems, hist, turn, self.budget)


class PriomptNodeRecent(PriomptNode):
    recent_turns = 3


class Membrane:
    # Every strategy's budget is in tiktoken tokens, so Membrane's own budget calibration is
    # pinned off: whatever the model id, it gets the same real budget as the others.
    options: ClassVar[dict[str, Any]] = {"token_scale": 1.0}
    model: ClassVar[str] = MODEL

    def __init__(self, budget: int, sc: Scenario) -> None:
        self._compiler = ContextCompiler(
            max_tokens=budget,
            model_id=self.model,
            core_content=CORE,
            include_interrupt_instruction=False,
            **self.options,
        )
        self.last_result: Any = None  # the live suite reads its cache breakpoints

    def build(self, turn: Turn, history: list[Message]) -> tuple[list[Message], bool]:
        self.last_result = self._compiler.compile(
            memories=turn.memories, user_message=turn.user, conversation_messages=history
        )
        return self.last_result.messages, False


class MembraneLean(Membrane):
    options: ClassVar[dict[str, Any]] = {**Membrane.options, "recall_evicted_history": False}


_EMBEDDED: dict[str, list[float]] = {}
_EMBED_MODEL: Any = None


def _shared_embedder(texts: Any) -> list[list[float]]:
    """fastembed's default model, loaded on first use; vectors shared by every run so a
    text is embedded once per process."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from membrane.lod import fastembed_embedder

        _EMBED_MODEL = fastembed_embedder()
    missing = [t for t in dict.fromkeys(texts) if t not in _EMBEDDED]
    if missing:
        _EMBEDDED.update(zip(missing, _EMBED_MODEL(missing), strict=True))
    return [_EMBEDDED[t] for t in texts]


class MembraneEmbed(Membrane):
    """Membrane with ``embedder=`` set to fastembed's BAAI/bge-small-en-v1.5 (the ``lod``
    extra). Only run when named in ``--strategies``; its ``ms`` includes the model."""

    options: ClassVar[dict[str, Any]] = {**Membrane.options, "embedder": _shared_embedder}

    def __init__(self, budget: int, sc: Scenario) -> None:
        import fastembed  # noqa: F401  (ImportError marks the strategy as not installed)

        super().__init__(budget, sc)


class MembraneLeanEmbed(MembraneEmbed):
    options: ClassVar[dict[str, Any]] = {**MembraneEmbed.options, "recall_evicted_history": False}


OPT_IN = ("membrane-embed", "membrane-lean-embed")

STRATEGIES: dict[str, type] = {
    "naive": Naive,
    "langchain": LangChainTrim,
    "llamaindex-buffer": LlamaIndexBuffer,
    "llamaindex-memory": LlamaIndexMemory,
    "priompt": Priompt,
    "priompt-recent": PriomptRecent,
    "priompt-node": PriomptNode,
    "priompt-node-recent": PriomptNodeRecent,
    "membrane": Membrane,
    "membrane-lean": MembraneLean,
    "membrane-embed": MembraneEmbed,
    "membrane-lean-embed": MembraneLeanEmbed,
}


def available_strategies() -> tuple[dict[str, type], list[str]]:
    """Strategies whose library is installed, and the names of those that are not."""
    ready, missing = {}, []
    probe = Scenario("probe", "", 1, 4_000, 1)
    for name, cls in STRATEGIES.items():
        try:
            cls(4_000, probe)
            ready[name] = cls
        except ImportError:
            missing.append(name)
    return ready, missing


# --- measurement ------------------------------------------------------------------------


@dataclass
class Score:
    tokens: int = 0
    cached: int = 0
    over: int = 0
    invalid: int = 0
    recall: dict[str, list[bool]] = field(default_factory=dict)
    ms: list[float] = field(default_factory=list)

    def cost(self, cached_price: float) -> float:
        return (self.tokens - self.cached) + self.cached * cached_price


def _text(messages: list[Message]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def run(cls: type, sc: Scenario, turns: list[Turn]) -> Score:
    strategy, score = cls(sc.budget, sc), Score()
    history: list[Message] = []
    previous: list[Message] = []
    for turn in turns:
        started = time.perf_counter()
        messages, glued = strategy.build(turn, history)
        score.ms.append((time.perf_counter() - started) * 1000)
        tokens = billed(messages)
        score.tokens += tokens
        score.over += glued or tokens > sc.budget
        score.invalid += orphaned_tools(messages)
        score.cached += cache_hit(previous, messages)
        previous = messages
        text = _text(messages)
        wanted = turn.evidence or turn.needles
        score.recall.setdefault(turn.kind, []).append(all(n in text for n in wanted))
        history = [*history, {"role": "user", "content": turn.user}, *turn.follow]
    return score


def run_scenario(sc: Scenario, seeds: int, strategies: dict[str, type]) -> dict[str, list[Score]]:
    scenarios = [build_scenario(sc, seed) for seed in range(1, seeds + 1)]
    return {name: [run(cls, sc, turns) for turns in scenarios] for name, cls in strategies.items()}


def summarize(scores: list[Score], baseline: list[Score]) -> dict[str, Any]:
    tokens = sum(s.tokens for s in scores)
    kinds = sorted({k for s in scores for k in s.recall})
    recall = {}
    for kind in kinds:
        hits = [x for s in scores for x in s.recall.get(kind, [])]
        recall[kind] = sum(hits) / len(hits)
    everything = [x for s in scores for hits in s.recall.values() for x in hits]
    return {
        "tokens": tokens // len(scores),
        "cached": sum(s.cached for s in scores) / tokens,
        "cost": {
            label: 100 * sum(s.cost(p) for s in scores) / sum(s.cost(p) for s in baseline)
            for label, p in (("0.5x", 0.5), ("0.1x", 0.1))
        },
        "recall": recall,
        "recall_all": sum(everything) / len(everything),
        "over": sum(s.over for s in scores),
        "invalid": sum(s.invalid for s in scores),
        "ms": statistics.median(ms for s in scores for ms in s.ms),
    }


def report(sc: Scenario, table: dict[str, dict[str, Any]], seeds: int) -> str:
    kinds = sorted({k for row in table.values() for k in row["recall"]})
    head = (
        f"{'strategy':<20} {'tokens':>8} {'cached':>7} {'cost 0.5x':>10} {'cost 0.1x':>10} "
        + " ".join(f"{k:>8}" for k in kinds)
        + f" {'over':>5} {'invalid':>8} {'ms':>6}"
    )
    out = [
        f"\n== {sc.name}: {sc.about}",
        f"   {sc.turns} turns, budget {sc.budget:,}, up to {sc.memories} memories per turn, "
        f"{seeds} seeds",
        head,
    ]
    for name, row in table.items():
        out.append(
            f"{name:<20} {row['tokens']:>8,} {100 * row['cached']:>6.0f}% "
            f"{row['cost']['0.5x']:>10.0f} {row['cost']['0.1x']:>10.0f} "
            + " ".join(f"{100 * row['recall'][k]:>7.0f}%" for k in kinds)
            + f" {row['over']:>5} {row['invalid']:>8} {row['ms']:>6.1f}"
        )
    return "\n".join(out)


def overall(results: dict[str, dict[str, dict[str, Any]]]) -> str:
    names = list(next(iter(results.values())))
    out = [
        "\n== overall: mean over scenarios",
        f"{'strategy':<20} {'cost 0.5x':>10} {'cost 0.1x':>10} {'recall':>7} {'over':>5} "
        f"{'invalid':>8}",
    ]
    for name in names:
        rows = [table[name] for table in results.values()]
        out.append(
            f"{name:<20} {statistics.mean(r['cost']['0.5x'] for r in rows):>10.0f} "
            f"{statistics.mean(r['cost']['0.1x'] for r in rows):>10.0f} "
            f"{100 * statistics.mean(r['recall_all'] for r in rows):>6.0f}% "
            f"{sum(r['over'] for r in rows):>5} {sum(r['invalid'] for r in rows):>8}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", action="append", choices=[s.name for s in SCENARIOS])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--json", help="also write the results to this file")
    parser.add_argument(
        "--strategies", help="comma-separated; naive is always run (cost is relative to it)"
    )
    args = parser.parse_args(argv)
    strategies, missing = available_strategies()
    if missing:
        print(f"Not installed, skipped: {', '.join(missing)}")
    if args.strategies:
        wanted = {"naive", *(s.strip() for s in args.strategies.split(","))}
        unknown = sorted(wanted - set(strategies))
        if unknown:
            parser.error(f"not available: {', '.join(unknown)}")
        strategies = {n: c for n, c in strategies.items() if n in wanted}
    else:
        strategies = {n: c for n, c in strategies.items() if n not in OPT_IN}
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for sc in SCENARIOS:
        if args.scenario and sc.name not in args.scenario:
            continue
        scores = run_scenario(sc, args.seeds, strategies)
        results[sc.name] = {n: summarize(s, scores["naive"]) for n, s in scores.items()}
        print(report(sc, results[sc.name], args.seeds), flush=True)
    if len(results) > 1:
        print(overall(results))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
