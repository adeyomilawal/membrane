# Membrane

**Treat the context window like RAM, not a scrap heap of strings.**

[![PyPI](https://img.shields.io/pypi/v/membrane-aol.svg)](https://pypi.org/project/membrane-aol/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[Website](https://consilience.icu/products/membrane) · [PyPI](https://pypi.org/project/membrane-aol/) · [Issues](https://github.com/adeyomilawal/membrane/issues)

Membrane is a prompt compiler for LLM applications. You give it your system rules, the memories
your retriever found, the conversation so far and the current user turn. It returns a
provider-ready message list that fits a hard token budget, puts the important content where
models pay attention, keeps a byte-stable prefix so provider prompt caching actually hits, and
gives you a map from every `[Mem:N]` citation back to your own record IDs. Like an operating
system managing RAM, it has fixed capacity, explicit placement, eviction by priority, and
caching. Prompts are compiled, never concatenated. The only core dependency is `tiktoken`.
Membrane itself never calls an LLM or sends your prompts anywhere.

```text
memories + rules + history + user turn
        │
        ▼
 Layout ──▶ Pre-flight ──▶ Firewall ──▶ Glaciator ──▶ Provider
 (U-shape)  (compress,     (hard token   (stable prefix  (messages,
            optional)      budget)       vs dynamic tail) cache hints)
        │
        ▼
 CompileResult: messages · ref_map · metadata (tokens, trimming, caching)
```

---

## Contents

- [Why Membrane](#why-membrane)
- [Install](#install)
- [Quickstart](#quickstart)
- [Core concepts](#core-concepts)
- [Providers and adapters](#providers-and-adapters)
- [Composing the stages yourself](#composing-the-stages-yourself)
- [Async usage](#async-usage)
- [Configuration reference](#configuration-reference)
- [More tools](#more-tools)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [License](#license)

---

## Why Membrane

Most LLM apps join system instructions, retrieved memories, history and the user message into
one big string. That breaks in four predictable ways:

| Problem | What goes wrong | What Membrane does |
| --- | --- | --- |
| **Context overflow** | The prompt exceeds the model's limit and gets truncated or rejected. | A firewall enforces a hard token budget. It counts with tiktoken and trims retrieved memories by priority tier. Rules and the user turn are never cut. |
| **Lost in the middle** | Models attend weakly to the middle of long prompts ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)), so facts placed there go unused. | U-shaped layout: rules at the start, the current turn at the end, and expendable memories in the middle, where they are trimmed first. |
| **Cache misses** | Memories that change every turn sit in front of the history, so the provider's prefix cache breaks on every request. | System rules and history form a byte-stable prefix. Changing content goes into a dynamic tail after it. |
| **Opaque cost** | You can't see what each section costs until you hit the limit or the bill. | Every compile reports token counts, trimmed tokens, kept and offered segments, and cache-prefix sizes. `estimate()` adds an input-cost quote. |

---

## Install

Requires Python 3.10+. The package is published as `membrane-aol` and imported as `membrane`.

```bash
pip install membrane-aol
```

Core features need only `tiktoken`. Optional features come as extras:

```bash
pip install "membrane-aol[all]"
```

```bash
pip install "membrane-aol[lod,redis]"
```

| Extra | Installs | Enables |
| --- | --- | --- |
| `lod` | numpy, fastembed | Level-of-detail attention-shift detection (embeddings run locally; fastembed downloads its model on first use) |
| `redis` | redis | Redis-backed ref-map persistence, the pre-flight compression cache and recent-stress storage |
| `stress` | psutil | CPU-load sampling for environment snapshots |
| `pydantic` | pydantic | Pydantic model support |
| `llm-openai` | openai | OpenAI SDK, for writing a pre-flight compressor |
| `llm-anthropic` | anthropic | Anthropic SDK, for writing a pre-flight compressor |
| `langchain` | langchain-core | `membrane.adapters.langchain` |
| `llamaindex` | llama-index-core | `membrane.adapters.llamaindex` |
| `crewai` | crewai | `membrane.adapters.crewai` |
| `haystack` | haystack-ai | `membrane.adapters.haystack` |
| `tokenizers` | sentencepiece, transformers | SentencePiece and Hugging Face tokenizer adapters |
| `mcp` | mcp | The Membrane MCP server (`python -m membrane.adapters.mcp_server`) |
| `all` | numpy, fastembed, redis, psutil, pydantic, openai | `lod` + `redis` + `stress` + `pydantic` + `llm-openai` |

`all` does **not** include `llm-anthropic`, `tokenizers`, `mcp` or the framework adapters.
Install those by name.

Two downloads happen on first use. tiktoken fetches its encoding files once; for offline
machines, pre-seed `TIKTOKEN_CACHE_DIR`. With `lod` installed, fastembed downloads its embedding
model the first time topic-shift detection or the `coding`/`conversational` deduplicator runs.

---

## Quickstart

```python
from membrane import CompileResult, ContextCompiler, MemoryItem

compiler = ContextCompiler(
    max_tokens=8192,
    model_id="openai/gpt-4o",
    core_content="You are a helpful assistant.",
)

memories = [
    MemoryItem(id="1", content="User prefers concise answers.", memory_type="semantic", priority=2),
    MemoryItem(id="2", content="Also discussed asyncio yesterday.", memory_type="semantic", priority=2),
]

result: CompileResult = compiler.compile(
    memories=memories,
    user_message="Show a producer-consumer example.",
)

print(result.messages)                   # send these to your provider
print(result.ref_map[1])                 # "1": [Mem:1] in the prompt is your record "1"
print(result.metadata["token_count"])    # exact size of the compiled prompt
```

`result.messages` is an OpenAI-style list. The system message holds your rules. The final user
message holds the kept memories followed by the current turn:

```text
[system] You are a helpful assistant.

         If a detail is missing from the provided context, do not hallucinate. Output exactly:
         <SYSTEM_INTERRUPT_FETCH[Your precise query for the missing detail]> ...
[user]   [Mem:1] [fact] kind=note | text=User prefers concise answers. | conf=1.00
         [Mem:2] [fact] kind=note | text=Also discussed asyncio yesterday. | conf=1.00

         --- Current user message ---
         Show a producer-consumer example.
```

The second paragraph of the system message is the [interrupt protocol](#interrupt-protocol)
instruction. Pass `include_interrupt_instruction=False` to leave it out.

Pass the conversation so far as `conversation_messages`. It is sent as real messages between the
system message and the final user message, with tool calls and content-part lists kept as given:

```python
from membrane import ContextCompiler, MemoryItem

compiler = ContextCompiler(max_tokens=8192, model_id="openai/gpt-4o",
                           core_content="You are a helpful assistant.")
history = [
    {"role": "user", "content": "What's a good queue for asyncio?"},
    {"role": "assistant", "content": "asyncio.Queue is the standard choice."},
]
result = compiler.compile(
    memories=[MemoryItem(id="pref-42", content="User prefers concise answers.")],
    conversation_messages=history,
    user_message="Show me a producer-consumer example.",
)
assert [m["role"] for m in result.messages] == ["system", "user", "assistant", "user"]
```

---

## Core concepts

### The U-shaped layout

Models attend most strongly to the start and end of their context
([Liu et al., 2023](https://arxiv.org/abs/2307.03172)). `ContextLayoutEngine` builds every prompt
in three regions:

| Region | Content | Attention | Trimmed? |
| --- | --- | --- | --- |
| **Start (core)** | System rules, persona, interrupt instruction | High | Never |
| **Middle (sacrifice zone)** | Retrieved memories, each tagged `[Mem:N]` | Low | First, by tier |
| **End** | Last turn and current user message, then any dynamic directives | High | Never |

With the default `memory_placement="end"`, history goes between the start and the middle, so the
memories sit right next to the question. See [docs/u-shaped-layout.md](docs/u-shaped-layout.md).

### Token budgets and priority tiers

`max_tokens` is a hard budget for the compiled prompt. When `model_id` is in the capability map,
the budget is also capped at that model's context window. Unknown models never shrink your
budget.

Every memory carries a priority tier. When the prompt is over budget, the firewall drops
memories from the lowest-priority tier first. The start and end regions are always kept.

| Tier | Constant | `MemoryItem.priority` | Typical content | Trimmed |
| --- | --- | --- | --- | --- |
| 1 | `TIER_PROCEDURAL` | `1` | Procedures, instructions | Last (at most 3 are kept) |
| 2 | `TIER_SEMANTIC` | `2` (default) | Facts, preferences | Second |
| 3 | `TIER_EPISODIC` | `3` | Past events and conversations | First |

Kept memories are emitted in tier order (procedural, then semantic, then episodic). `[Mem:N]`
numbers follow your input order, so a citation always points at the same input record. Each
memory's text is shortened by default: 200 characters for facts, 280 for episodes and 500 for
procedures. Set `MEMBRANE_MEMORY_TEXT_MAX_CHARS` to change this (`0` disables it).
`metadata["memories_truncated"]` reports how many were shortened.

```python
from membrane import ContextCompiler, MemoryItem

compiler = ContextCompiler(max_tokens=200, core_content="You are a support agent.")
memories = [
    MemoryItem(id="howto", content="Escalate refunds over $500 to a human.", memory_type="procedural", priority=1),
    MemoryItem(id="fact", content="Customer is on the Pro plan.", memory_type="semantic", priority=2),
    *[
        MemoryItem(id=f"ep{i}", content=f"Earlier ticket #{i} about a billing question, resolved by email.",
                   memory_type="episodic", priority=3)
        for i in range(10)
    ],
]
result = compiler.compile(memories=memories, user_message="Can I get a refund?")

md = result.metadata
print(md["token_count"], "<=", md["tokens_budget"])            # always within budget
print(md["segments_kept"], "of", md["segments_offered"], "memories kept")
print(md["tokens_trimmed"], "tokens trimmed")
assert "howto" in result.ref_map.values()                      # procedural kept
assert "fact" in result.ref_map.values()                       # semantic kept
```

Token counts use tiktoken's `cl100k_base` encoding by default, for every model. That is exact for
GPT-4 and GPT-3.5 but only approximate for other tokenizers. To count with a model's own tokenizer,
pass a `TokenizerRegistry` with patterns for your model ids. `TokenizerRegistry.get()` returns a
shared registry preloaded with patterns for bare ids (`gpt-4o*` → `o200k_base`, `claude*` →
`cl100k_base`). `TokenizerRegistry()` starts empty:

```python
from membrane import ContextCompiler, MemoryItem
from membrane.tokenizers import TiktokenAdapter, TokenizerRegistry

registry = TokenizerRegistry()
registry.register("openai/gpt-4o*", TiktokenAdapter("o200k_base"))

compiler = ContextCompiler(max_tokens=8192, model_id="openai/gpt-4o")
result = compiler.compile(
    memories=[MemoryItem(id="1", content="User prefers concise answers.")],
    user_message="Hi",
    tokenizer_registry=registry,
)
```

Patterns are matched against the full `model_id` with `fnmatch`, so the preloaded `gpt-4o*` does
not match `openai/gpt-4o`. Register a pattern for the ids you actually use. When no pattern
matches, `TokenizerRegistry.get()` falls back to a character-based estimate, and a fresh
`TokenizerRegistry()` falls back to `cl100k_base`. `SentencePieceAdapter` and `HuggingFaceAdapter` need the
`tokenizers` extra.

### The pipeline

Every compile runs the same deterministic stages:

1. **Layout.** `ContextLayoutEngine` serialises memories into `[Mem:N]` lines and builds the core,
   middle and end regions.
2. **Pre-flight.** If the middle exceeds `MEMORY_PREFLIGHT_TOKEN_THRESHOLD` (8,000 tokens by
   default) and you passed a compressor (`compress_sync`, or `compress_async` /
   `compress_fn` on the async paths), it is summarised to about `MEMORY_PREFLIGHT_TARGET_TOKENS`
   (500). Without a compressor the stage is skipped and the firewall trims as usual.
3. **Firewall.** `PreFlightContextFirewall` enforces the hard budget by trimming the middle by
   tier, then prunes `ref_map` to the memories that survived.
4. **Glaciator.** The output is split into a byte-stable prefix (system rules, then history) and
   a dynamic tail (memories, current turn, directives). Prompt caching reuses the prefix.
5. **Provider.** Messages are formatted for the target model, with caching metadata when the
   model supports prompt caching.

Optional modules (dedup, freshness decay, classifier, policies, reranker, profiler) plug in
through a `MembraneProfile`. The built-in presets are `minimal` (the default), `coding` and
`conversational`:

```python
from membrane import ContextCompiler, MemoryItem
from membrane.profile import MembraneProfile

compiler = ContextCompiler(max_tokens=8192, model_id="openai/gpt-4o")
result = compiler.compile(
    memories=[MemoryItem(id="1", content="Use pytest for tests.", memory_type="procedural", priority=1)],
    user_message="Write a test for parse_date().",
    profile=MembraneProfile.coding(),
)
```

### Glaciation and prompt caching

Providers cache the longest prefix a request shares with an earlier one. If memories sit before
the conversation history, they change every turn and the cache never gets past them. Membrane
"glaciates" the prompt: the frozen part comes first and the moving part last.

With the default `memory_placement="end"`, the compiler emits:

1. a system message with only the core rules,
2. `conversation_messages` exactly as given,
3. one user message with the kept memories, the current turn and any dynamic directives.

Items 1 and 2 are byte-identical from turn to turn, so the provider serves most of each request
from cache. History may hold `history_share` (default 0.75) of the budget left after the core.
Past that, the oldest turns are dropped in chunks of half that share
(`metadata["history_messages_dropped"]`). The cut depends only on the budget and the core, never
on this turn's memories or question, so it stays put as the chat grows and the cache breaks only
when another chunk goes. Dropped turns are not lost: the most recently dropped chunk returns in
the dynamic tail as low-priority recall (`recall_evicted_history`), trimmed before any memory and
kept out of `ref_map`. Pass `trim_history=False` to manage history yourself. An overrun then sets
`metadata["budget_exceeded"] = True` instead of dropping turns.

`memory_placement="system"` puts core, memories, the last turn and the user message into a single
system message. It sends fewer tokens because it includes only the last turn of history, but the
cache can never get past the core. Use it for local models without caching, and for one-off
calls.

| Metadata key | Meaning |
| --- | --- |
| `memory_placement` | `"end"` (end placement only) |
| `stable_prefix_messages` | Number of leading messages that form the stable prefix |
| `history_messages_dropped` | Oldest history messages dropped to fit the budget |
| `history_messages_recalled` | Dropped messages that came back in the tail as recall |
| `message_framing_tokens` | Per-message framing reserved in the budget (`token_count` is content only) |
| `prompt_cache_used` | `True` when the model supports caching and the prefix is at least 1,024 tokens |
| `cacheable_prefix_tokens` / `dynamic_tail_tokens` | Sizes of the two parts (set with `prompt_cache_used`) |
| `budget_exceeded` | Reserved content (rules, history, current turn) alone exceeded `max_tokens` |

To split text you assemble yourself, use the standalone `glaciate()` helper. See
[Composing the stages yourself](#composing-the-stages-yourself) and
[docs/prompt-caching.md](docs/prompt-caching.md).

### Ref-map citations

Each memory gets a short local index in the prompt, such as `[Mem:3]`. `result.ref_map` maps each
index that survived trimming back to your record ID, and `result.ref_map_type` gives its kind.
Ask the model to cite `[Mem:N]` and you can resolve every citation to a real record:

```python
import re

from membrane import ContextCompiler, MemoryItem

compiler = ContextCompiler(max_tokens=4096)
result = compiler.compile(
    memories=[
        MemoryItem(id="doc-381", content="The refund window is 30 days."),
        MemoryItem(id="doc-112", content="Refunds go back to the original card."),
    ],
    user_message="How long do I have to request a refund?",
)

answer = "You have 30 days [Mem:1], refunded to your original card [Mem:2]."  # model output
cited = [result.ref_map[int(n)] for n in re.findall(r"\[Mem:(\d+)\]", answer)]
assert cited == ["doc-381", "doc-112"]
assert result.ref_map_type[1] == "semantic"
```

Trimmed memories are removed from the map, so a citation that doesn't resolve shows the model
cited something it was never shown. Across turns, you can persist the map in Redis with
`set_ref_map_for_conversation` and `get_ref_map_for_conversation` (requires the `redis` extra), or
pass `persist_ref_map_to_redis=True` to `compile_prompt_async`.

---

## Providers and adapters

### Capability map

Membrane ships a capability map ([`llm_capabilities.json`](src/membrane/llm_capabilities.json))
that sets the context cap and prompt-caching support for each model:

| Model IDs | Context cap | Prompt caching |
| --- | --- | --- |
| `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-4-turbo`, `openai/gpt-5.1` | 128,000 | Yes |
| `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-7-sonnet`, `anthropic/claude-sonnet-4.5`, `anthropic/claude-3-opus` | 200,000 | Yes |
| `google/gemini-3-pro-preview`, `google/gemini-pro-1.5` | 1,000,000 | Yes |
| `x-ai/grok-4` | 131,072 | Yes |
| `qwen2.5:7b`, `llama3.2`, `llama3.1:8b`, `llama-3` (local) | 4,096–32,768 | No |

Lookups ignore case, a missing provider prefix and date suffixes, so `gpt-4o` and
`openai/gpt-4o-2024-08-06` both resolve to `openai/gpt-4o`. Unknown models get no caching and no
context cap, and log a warning once. To use your own map, point `LLM_CAPABILITY_MAP_PATH` at a
JSON file with the same `capabilities` and `context_caps` keys; it replaces the bundled map.

```python
from membrane import get_context_cap_for_model, has_prompt_caching

assert has_prompt_caching("gpt-4o")
assert get_context_cap_for_model("anthropic/claude-sonnet-4.5") == 200_000
```

### OpenAI and OpenAI-compatible APIs

`result.messages` already has the Chat Completions shape:

```python
from membrane import ContextCompiler, MemoryItem

compiler = ContextCompiler(max_tokens=8192, model_id="openai/gpt-4o-mini",
                           core_content="You are a helpful assistant.")
result = compiler.compile(memories=[MemoryItem(id="1", content="User is in Lagos.")],
                          user_message="What time zone am I in?")

request = {"model": "gpt-4o-mini", "messages": result.messages}
# from openai import OpenAI
# OpenAI().chat.completions.create(**request)
```

OpenAI caches long shared prefixes automatically, so end placement is all you need.

### Anthropic

The Anthropic Messages API takes the system prompt as a separate `system` parameter. Split it
off and mark it cacheable with `anthropic_system_blocks`:

```python
from membrane import ContextCompiler, MemoryItem, anthropic_system_blocks

compiler = ContextCompiler(max_tokens=8192, model_id="anthropic/claude-sonnet-4.5",
                           core_content="You are a helpful assistant.")
result = compiler.compile(memories=[MemoryItem(id="1", content="User is in Lagos.")],
                          user_message="What time zone am I in?")

system, *messages = result.messages
assert system["role"] == "system"
request = {
    "model": "claude-sonnet-4-5",  # the API's model name, not Membrane's model_id
    "max_tokens": 1024,
    "system": anthropic_system_blocks(system["content"], ""),  # adds cache_control
    "messages": messages,
}
# import anthropic
# anthropic.Anthropic().messages.create(**request)
```

This caches the system prompt. To cache the conversation history as well, add a `cache_control`
breakpoint to the last history message yourself. Membrane does not add one. Anthropic only
caches prefixes above a model-specific minimum length.

### Local models

Local models in the map (Ollama-style ids such as `llama3.1:8b`) report no caching. Consider
`memory_placement="system"` for them: it sends fewer tokens, and there is no cache to protect.

### Framework adapters

| Framework | Extra | Entry points |
| --- | --- | --- |
| LangChain | `langchain` | `membrane.adapters.langchain`: `MembranePromptTemplate`, `langchain_doc_to_memory`, `disable_langchain_memory` |
| LlamaIndex | `llamaindex` | `membrane.adapters.llamaindex`: `MembraneNodePostprocessor`, `node_to_memory` |
| CrewAI | `crewai` | `membrane.adapters.crewai`: `compile_for_agent`, `document_to_memory` |
| Haystack | `haystack` | `membrane.adapters.haystack`: `MembranePromptBuilder`, `haystack_document_to_memory` |
| MCP | `mcp` | `python -m membrane.adapters.mcp_server` (stdio server) |

Use Membrane *instead of* a framework's own conversation memory, not alongside it. With
LangChain, `disable_langchain_memory(chain)` removes the chain's memory, and
`detect_memory_collision(chain, membrane_active=True)` raises `MemoryCollisionError` if one is
still set. See [examples/drop3_mcp_and_adapters.md](examples/drop3_mcp_and_adapters.md).

### Memory types

`compile()` accepts `MemoryItem` rows or any object with an `id` and a `memory_kind` property (the
`KernelMemoryRecord` protocol). Typed records are also available: `SemanticMemory`,
`EpisodicMemory`, `ProceduralMemory` and `KernelMemoryObject`, with a JSON schema from
`load_kernel_memory_json_schema()`.

---

## Composing the stages yourself

Every stage is a public building block. Use them separately when you already have your own
prompt assembly:

```python
from membrane import (
    TIER_EPISODIC,
    TIER_PROCEDURAL,
    TIER_SEMANTIC,
    ContextLayoutEngine,
    PreFlightContextFirewall,
    SemanticMemory,
    anthropic_system_blocks,
    build_system_messages,
    count_tokens,
    glaciate,
)

# 1. Layout: serialise (memory, tier) pairs into [Mem:N] segments.
layout = ContextLayoutEngine(core_content="You are a precise research assistant.")
retrieved = [
    (SemanticMemory(id="s1", subject="Membrane", predicate="license", object="Apache-2.0"), TIER_SEMANTIC),
]
middle, ref_map, ref_map_type = layout.build_middle(retrieved)
middle.append(("[Mem:2] Earlier session: user compared prompt compilers.", TIER_EPISODIC))
middle.append(("[Mem:3] Always cite sources as [Mem:N].", TIER_PROCEDURAL))
ref_map.update({2: "e7", 3: "p1"})

core = layout.build_core()
end = layout.build_end(last_turn_content=None, user_message="What license does Membrane use?")

# 2. Firewall: enforce the budget; trims the middle by tier and prunes the ref map.
firewall = PreFlightContextFirewall(max_tokens=4096)
text, kept_refs, total, stats = firewall.run(
    core_text=core, end_text=end, middle_segments=middle, ref_map=ref_map
)
assert total == count_tokens(text) <= 4096
print(stats["segments_kept"], "of", stats["segments_offered"], "segments kept")

# 3. Glaciator: split at a line boundary into a cacheable prefix and a dynamic tail.
g = glaciate(text, max_cacheable_tokens=2000)

# 4. Provider formatting.
openai_messages, cache_hint = build_system_messages("openai/gpt-4o", g.cacheable_prefix, g.dynamic_tail)
anthropic_system = anthropic_system_blocks(g.cacheable_prefix, g.dynamic_tail)
```

`glaciate()` never cuts mid-line. Its `use_caching` flag is `True` once the prefix reaches 256
tokens. Check the provider's own minimum cacheable length too.

---

## Async usage

`ContextCompiler.compile_async` takes the same arguments as `compile`, plus an async
pre-flight compressor and optional Redis caching of compressed output:

```python
import asyncio

from membrane import ContextCompiler, MemoryItem


async def compress(raw_middle: str, target_tokens: int, user_message: str) -> str:
    # Call your LLM here. Runs only when the middle exceeds MEMORY_PREFLIGHT_TOKEN_THRESHOLD.
    return raw_middle[: target_tokens * 4]


async def main() -> None:
    compiler = ContextCompiler(max_tokens=8192, model_id="openai/gpt-4o")
    result = await compiler.compile_async(
        memories=[MemoryItem(id="1", content="User prefers concise answers.")],
        user_message="Summarise our plan.",
        compress_async=compress,
    )
    print(result.metadata["token_count"])


asyncio.run(main())
```

`compile_prompt_async` (alias `compile_prompt_payload`) is a higher-level entry point for chat
backends. It takes pre-retrieved `(memory, tier)` pairs and the raw message list, detects topic
shifts (with the `lod` extra) and can persist the ref map to Redis:

```python
import asyncio

from membrane import TIER_EPISODIC, TIER_SEMANTIC, MemoryItem, compile_prompt_async


async def main() -> None:
    result = await compile_prompt_async(
        user_msg="Can you remind me what we decided?",
        messages_list=[
            {"role": "user", "content": "Let's pick a database."},
            {"role": "assistant", "content": "Postgres fits your workload."},
        ],
        user_id="user-1",
        conversation_id="conv-9",
        max_tokens=8192,
        model_id="openai/gpt-4o",
        core_content="You are a helpful assistant.",
        retrieved_objects=[
            (MemoryItem(id="d1", content="Team chose Postgres over MongoDB."), TIER_SEMANTIC),
            (MemoryItem(id="e1", content="Discussed sharding last week.", memory_type="episodic"), TIER_EPISODIC),
        ],
    )
    print(result.ref_map)


asyncio.run(main())
```

To write the compressor against a class instead of a bare function, implement the
`PreFlightAnalyst` protocol (`compress_retrieved_memory(...)`), or wrap an async function with
`as_preflight_analyst`. Membrane does not ship provider SDK clients. The `llm-openai` and
`llm-anthropic` extras only install the SDKs for your own compressor.

---

## Configuration reference

### `ContextCompiler`

| Parameter | Default | Description |
| --- | --- | --- |
| `max_tokens` | required | Hard token budget for the compiled prompt (capped at the model's context window when known) |
| `model_id` | `None` | Model id for capability lookup (context cap, prompt caching) |
| `core_content` | `""` | System rules placed at the start; never trimmed |
| `include_interrupt_instruction` | `True` | Append the interrupt-protocol instruction to the system message |
| `memory_placement` | `"end"` | `"end"` (cache-friendly) or `"system"` (single system message) |
| `trim_history` | `True` | With end placement, drop the oldest history in chunks when over budget |
| `history_share` | `0.75` | Share of the budget after the core that history may hold in the cached prefix |
| `recall_evicted_history` | `True` | Bring the most recently dropped history chunk back as low-priority recall |
| `message_overhead_tokens` | `3` | Tokens billed per message on top of content (OpenAI chat framing), reserved in the budget |
| `metrics_callback` | `None` | Receives a `CompileMetrics` after each compile (token counts, never prompt text) |
| `preflight_analyst_model` | `None` | Model hint for your pre-flight compressor |

### `compile()` / `compile_async()` arguments

| Argument | Description |
| --- | --- |
| `memories` | `MemoryItem` rows or `KernelMemoryRecord`-compatible objects |
| `user_message` | The current user turn (always kept) |
| `conversation_messages` | Prior messages; sent as-is with end placement |
| `last_assistant_turn` | Last assistant reply, when you don't pass full history |
| `agent_persona_prompt` | Appended to `core_content` |
| `dynamic_directives` | Per-turn instructions, emitted last in an `<ACTIVE_SYSTEM_DIRECTIVES>` block |
| `middle_extras` | Extra `(text, tier)` segments for the middle, trimmed by tier like memories (ranked first when a reranker is active) |
| `tier_by_memory_id` | Override tiers, keyed by Python `id()` of each memory object |
| `use_prompt_caching_split` | `False` disables cache metadata |
| `compress_sync` / `compress_async` | Pre-flight compressor `(raw_middle, target_tokens, user_message) -> str` |
| `profile` | A `MembraneProfile` (`minimal`, `coding`, `conversational`) |
| `tokenizer_registry` | Per-model token counting |
| `redis_client`, `conversation_id` | Async only: cache pre-flight output in Redis |

### `CompileResult`

| Field | Description |
| --- | --- |
| `messages` | Provider-ready message list |
| `ref_map` | `{N: your_id}` for every `[Mem:N]` that survived trimming |
| `ref_map_type` | `{N: "semantic" \| "episodic" \| "procedural"}` |
| `metadata` | `token_count`, `tokens_budget`, `tokens_trimmed`, `segments_offered`, `segments_kept`, `memories_truncated`, `retrieval_summary`, the caching keys above, and more |
| `debug_trace` | Ranking diagnostics, when enabled |
| `to_debug_dict()` | JSON-friendly snapshot for logging |

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `MEMORY_PREFLIGHT_TOKEN_THRESHOLD` | `8000` | Middle size (tokens) that triggers the pre-flight compressor |
| `MEMORY_PREFLIGHT_TARGET_TOKENS` | `500` | Target size passed to the compressor |
| `PREFLIGHT_ANALYST_MODEL` | `openai/gpt-4o-mini` | Model hint for the compressor |
| `MEMBRANE_MEMORY_TEXT_MAX_CHARS` | unset | Per-memory text cap; unset = 200/280/500 by kind; `0` = no cap |
| `MAX_CACHEABLE_PREFIX_TOKENS` | `2000` | Default `max_cacheable_tokens` for `glaciate()` |
| `LOD_ATTENTION_SHIFT_COSINE_THRESHOLD` | `0.35` | Cosine distance that counts as a topic shift |
| `LLM_CAPABILITY_MAP_PATH` | bundled | Path to a replacement capability map JSON |
| `MEMBRANE_REDIS_URL` (or `REDIS_URL`) | unset | Default Redis URL for ref-map persistence |
| `MEMBRANE_REF_MAP_PREFIX` / `MEMBRANE_REF_MAP_TTL_SECONDS` | `membrane:ref_map:` / `600` | Ref-map Redis keys |
| `MEMBRANE_PREFLIGHT_CACHE_PREFIX` / `MEMBRANE_PREFLIGHT_CACHE_TTL_SECONDS` | `membrane:preflight:` / `60` | Pre-flight cache Redis keys |
| `MEMBRANE_STRESS_KEY_PREFIX` / `STRESS_RECENT_TTL_SECONDS` | `membrane:stress:` / `300` | Recent-stress Redis keys |

---

## More tools

### Cost estimates

`estimate()` runs the full compile and adds an input-side quote. Membrane ships no price list, so
you pass your provider's rate:

```python
from membrane import ContextCompiler, MemoryItem

compiler = ContextCompiler(max_tokens=8192, model_id="openai/gpt-4o-mini")
quote = compiler.estimate(
    memories=[MemoryItem(id="1", content="User prefers concise answers.")],
    user_message="Hello",
    price_per_1k_input_tokens=0.00015,
)
print(quote.total_tokens, quote.estimated_cost_usd, quote.trimming_strategies)
messages = quote.compile_result.messages   # no need to compile twice
```

### Interrupt protocol

When the context is missing a detail, the default system instruction tells the model to emit a
fetch tag instead of guessing. Your host detects it, retrieves, and continues:

```python
from membrane import contains_interrupt, parse_interrupt, strip_all_interrupts

text = "Let me check. <SYSTEM_INTERRUPT_FETCH[user's timezone preference]> One moment."
if contains_interrupt(text):
    before, query, after = parse_interrupt(text)
    assert query == "user's timezone preference"
clean = strip_all_interrupts(text)
```

See [docs/interrupt-protocol.md](docs/interrupt-protocol.md) for streaming integration.

### Topic-shift detection (LOD)

With the `lod` extra, `compute_attention_shift` compares the current message with the centroid
of recent user messages, using local fastembed embeddings:

```python
from membrane import compute_attention_shift

shifted = compute_attention_shift(
    ["What's the best Python testing framework?", "How does pytest discover tests?"],
    "How do I deploy to Kubernetes?",
)
```

It returns `False` when numpy or fastembed is missing. To use your own embedder, call
`compute_attention_shift_with_embeddings`.

### CLI diagnostics

The `membrane` command runs the compiler locally and calls no LLM:

```bash
membrane inspect --user-message "Explain the retry policy." --memory "policy|2|Retries use exponential backoff with jitter." --max-tokens 2048
```

```bash
membrane quote --user-message "Summarize the release plan." --memory "release|2|Version 0.2 ships after the integration tests pass." --price-per-1k-input-tokens 0.00015
```

```bash
membrane profile explain coding
```

`--memory` takes `id|tier|content` and can be repeated. Add `--json` for machine-readable output.

---

## Benchmarks

**Prompt-cache savings** ([`benchmarks/cache_savings.py`](benchmarks/cache_savings.py)): a
20-turn chat on `openai/gpt-4o` with a 2,000-token system prompt and 5 new memories retrieved
every turn. The simulated provider caches the longest prefix shared with the previous request
(minimum 1,024 tokens).

| Setup | Input tokens | Served from cache | Relative cost, cache reads at 0.1× | Relative cost, cache reads at 0.5× |
| --- | ---: | ---: | ---: | ---: |
| Naive: memories in the system prompt, full history | 84,380 | 45% | 100 | 100 |
| **Membrane, `memory_placement="end"` (default)** | 87,460 | **86%** | **40** | **77** |
| Membrane, `memory_placement="system"` | 51,091 | 78% | 31 | 48 |

With the same full history, end placement cuts input cost by 23–60%, because memories no longer
break the cache every turn. Anthropic bills cache reads at about 0.1× and OpenAI at about 0.5×.
`"system"` is cheaper only because it sends just the last turn of history. The simulation is
deterministic; your savings depend on prompt sizes and pricing.

```bash
python -m benchmarks.cache_savings
```

**Peer comparison** ([`benchmarks/peer_comparison.py`](benchmarks/peer_comparison.py)): a
24-turn chat under a hard budget, 5 seeds. Each turn asks for one fact that sits either in a
retrieved memory or in a user message 10 turns back. Tokens are counted with OpenAI chat framing.
Recall means the answer was in the prompt, an upper bound on accuracy. The `--live` mode measures
real accuracy and `cached_tokens` against OpenAI but has not been run yet. `priompt` reimplements
Priompt's priority-cutoff search rather than importing the package.

| 6,000-token budget, 8 memories per turn | Cached | Over budget | Recall (memory / history) | Cost at 0.5× / 0.1× cache reads |
| --- | ---: | ---: | ---: | ---: |
| Naive concatenation | 27% | 0 | 100% / 100% | 100 / 100 |
| Priompt-style, either priority scheme | 27% | 0 | 100% / 100% | 100 / 100 |
| **Membrane, defaults** | **58%** | 0 | 100% / 100% | **82 / 63** |
| Membrane, `memory_placement="system"` | 56% | 0 | 100% / 0% | 41 / 32 |

With a 12,000-token budget Membrane caches 81% and costs 68 / 35. Under memory pressure
(4,000 tokens, 40 memories) it still costs less (88 / 75), but it keeps fewer memories: 76%
memory recall against 95% for naive, which makes room by discarding all history. Lower
`history_share` for memory-heavy apps.

```bash
python -m benchmarks.peer_comparison [--budget N] [--memories N] [--live]
```

Also in `benchmarks/`: `token_savings_bench.py` (savings from compression and trimming) and
`trimming_bench.py` (trimming keeps high-priority content and the U-shaped layout).

---

## Contributing

Bug reports, fixes, docs and new adapters are welcome. For anything bigger than a small fix,
open an issue first. Every pull request needs the [Contributor License Agreement](CLA.md) box
ticked. See [CONTRIBUTING.md](CONTRIBUTING.md) for details. Report security issues privately as
described in [SECURITY.md](SECURITY.md).

```bash
git clone https://github.com/adeyomilawal/membrane.git
```

```bash
cd membrane && uv sync --locked --extra dev
```

Run the same checks as CI:

```bash
uv lock --check && uv run ruff check src tests && uv run pytest && uv build
```

Without uv, run `pip install -e ".[dev]"` and then `pytest`. Tests that need optional stacks
(MCP, LangChain and so on) are skipped when those packages aren't installed.

Further reading: [Architecture](docs/architecture.md) ·
[U-shaped layout](docs/u-shaped-layout.md) · [Prompt caching](docs/prompt-caching.md) ·
[Interrupt protocol](docs/interrupt-protocol.md) · [Changelog](CHANGELOG.md)

---

## License

Membrane is licensed under the [Apache License, Version 2.0](LICENSE). You can use, modify and
distribute it in commercial and non-commercial products. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
