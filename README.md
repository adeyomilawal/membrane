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

## Results at a glance

Eleven stress scenarios (long chats, tool calls, memory floods, long pastes, tight budgets,
paraphrased questions) sent to the Claude API on 2026-11-11, with the same token budget for every
library. Accuracy means the reply contained the fact asked for. "Wrong" counts replies that stated
a different value instead of saying it was unknown. Cost is input cost relative to LangChain = 100.

| | Opus 5.5 accuracy | Wrong | Cost | Haiku 5.5 accuracy | Wrong | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LangChain `trim_messages` | 85% | 21 | 100 | 84% | 25 | 100 |
| LlamaIndex `Memory` | 79% | 40 | 81 | 77% | 42 | 81 |
| Priompt | not run | | | 83% | 26 | 100 |
| **Membrane, defaults** | **99%** | **2** | **63** | **96%** | **7** | **59** |
| Membrane, `recall_evicted_history=False` | 97% | 9 | 43 | 95% | 12 | 45 |

Read this with its limits: one seed per model, synthetic tasks, Claude models only, and the other
libraries run with their documented defaults. The suite, the raw results and the scenarios where
Membrane got worse or stayed behind are under [Benchmarks](#benchmarks).

---

## Contents

- [Results at a glance](#results-at-a-glance)
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

With end placement, nothing is dropped while something else still has its full text
(`thin_before_drop`). Every memory has a keyframe: its densest sentence under
`keyframe_tokens` (40), where dense means its words are rare across the sentences of the
memories on offer and of the conversation. Under pressure the keyframes of all memories are
placed first, then memories get their full text back in priority order while room lasts. A
thinned memory keeps its `[Mem:N]` ref and ends in `…`, so the model can still cite it or ask
for the full text. `metadata["segments_thinned"]` counts them. A memory with no sentence under
the cap is kept whole or dropped, as before. A keyframe is one sentence: a fact spread over two
sentences of a memory is only half there once that memory is thinned.

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
GPT-4 and GPT-3.5 but other tokenizers count the same text differently (Claude counts more). So
the budget is divided by a per-model `token_scale`. It starts from a seed in the capability map
(1.0 for OpenAI, 1.11 for Claude Opus 4.5 and 4.6, Sonnet 4.6 and Haiku 4.5, 1.51 for Claude
Opus 4.7 and every later Claude model) and you correct it with what the provider actually billed:

```python
from membrane import ContextCompiler, MemoryItem

compiler = ContextCompiler(max_tokens=8000, model_id="anthropic/claude-opus-5-5")
result = compiler.compile(memories=[MemoryItem(id="1", content="User is in Lagos.")],
                          user_message="What time zone am I in?")
# ... send the request, then report the prompt tokens the provider counted:
# usage = response.usage
# compiler.observe(result, usage.input_tokens + usage.cache_read_input_tokens
#                          + usage.cache_creation_input_tokens)
print(result.metadata["token_scale"], result.metadata["tokens_budget_counted"])  # 1.51 5298
```

`observe` smooths the ratio per model, adds a margin from its spread, and later compiles use the
result. The scale in use moves in steps of 0.05, rises at once and falls only when it is 0.1 too
high, because every change moves the history cut and breaks the cache once. It lives on the
compiler object: nothing is downloaded, stored or sent. Pass `token_scale=1.0` to turn it off.
The Claude seeds were measured on 2026-11-11 with Anthropic's token-count endpoint
([`benchmarks/measure_token_scale.py`](benchmarks/measure_token_scale.py)) and are the ratio for
English prose. Code counts higher (1.65 on Opus 4.7 and later, 1.27 on the older tokenizer) and
digit-heavy text lower (1.11 and 1.05), so a first request that is mostly code can run about 9%
over the budget until `observe` has seen it. Models other than Claude and OpenAI start at 1.0,
unmeasured. `observe` is what makes the budget right.

To count with a model's own tokenizer instead,
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
when another chunk goes.

Dropped turns are not lost. Each dropped message leaves one line in an index placed in the
system message right after the core (`eviction_index`): its role and its keyframe (see
[Token budgets](#token-budgets-and-priority-tiers)), scored over the conversation up to the cut
that dropped it. A line never changes after it is written, so the index only changes when the
cut moves, and at that moment the cache after the core is rewritten anyway: the index adds no
cache breaks. It may hold up to `index_share` (one eighth) of the budget the core leaves. That
room is not taken from history: it comes from what history, the question and one keyframe per
memory on offer leave, and never less than half the share. The size is decided when a cut is
first seen and kept until the next cut, so the block stays byte-stable in between; a new
compiler object can size it differently, which costs one cache miss. Over its room, the oldest
lines go, again only at a cut. Under memory pressure the index shrinks, and with a very small
budget that loses old history (see [Benchmarks](#benchmarks)). Tool results get a line,
tool-call stubs do not, and a long message keeps one keyframe per 400 tokens of its length.

Passages come back in full where they match (`recall_evicted_history`): dropped messages are
split into passages of about 60 words, tool results included, and ranked by the rare words they
share with the current user message, then by recency. The best passages, up to one chunk, return
in the dynamic tail and stay out of `ref_map`. Every memory is offered its place before any
passage is, so passages cost the memories at most the two header lines that introduce them
(which can be one memory line under a very tight budget). A passage that does not fit whole is
cut to the sentence that matches the question.

When memories do not all fit, the ones that survive are chosen by reciprocal rank fusion of the
order you passed (your retriever's rank) and the same lexical relevance (`rank_fusion`). The
survivors keep your order, so nothing changes when everything fits. Both are word matching, not
semantic search: they help when the question names what it asks for. To rank by meaning as
well, pass an embedding function (see [Ranking by meaning](#ranking-by-meaning-optional)). Pass `trim_history=False` to manage history yourself. An overrun then sets
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
| `history_messages_recalled` | Dropped passages that came back in the tail as recall |
| `history_index_lines` / `history_index_tokens` | Size of the eviction index in the system message |
| `core_chars` | Where the core ends inside the system message (set when there is an index) |
| `segments_thinned` | Memories and passages cut to their keyframe |
| `cache_breakpoints` | Where a cached prefix ends: `{"message_index", "tokens", "reason"}`, plus `"chars"` inside a message |
| `cache_mode` | `"automatic"` (the provider caches by itself) or `"marked"` (the request must mark the prefix) |
| `token_scale` / `token_scale_samples` | Provider tokens per counted token, and how many `observe` calls shaped it |
| `tokens_budget_counted` | The budget in counted tokens (`tokens_budget / token_scale`), when the scale is not 1 |
| `message_framing_tokens` | Per-message framing reserved in the budget (`token_count` is content only) |
| `prompt_cache_used` | `True` when the model supports caching and the prefix reaches the model's minimum cacheable size |
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
that sets the context cap, prompt-caching support, cache prices and token scale for each model:

| Model IDs | Context cap | Prompt caching |
| --- | --- | --- |
| `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-5.1` | 128,000 | Yes |
| `anthropic/claude-sonnet-4.5` (deprecated by Anthropic, retires 2026-11-30) | 200,000 | Yes |
| `anthropic/claude-opus-*`, `claude-sonnet-*`, `claude-haiku-*`, `claude-fable-*` (matched by family, so `claude-opus-5-5` resolves) | 200,000, or 1,000,000 for Opus 4.6+, Sonnet 4.6+, Haiku 5+ and Fable | Yes |
| `google/gemini-3-pro-preview`, `google/gemini-pro-1.5` | 1,000,000 | Yes |
| `x-ai/grok-4` | 131,072 | Yes |
| `qwen2.5:7b`, `llama3.2`, `llama3.1:8b`, `llama-3` (local) | 4,096–32,768 | No |

Lookups ignore case, a missing provider prefix and date suffixes, so `gpt-4o` and
`openai/gpt-4o-2024-08-06` both resolve to `openai/gpt-4o`. Unknown models get no caching and no
context cap, and log a warning once. To use your own map, point `LLM_CAPABILITY_MAP_PATH` at a
JSON file with the same `capabilities` and `context_caps` keys; it replaces the bundled map.

Two more sections of the map drive the layout. `cache_pricing` gives, per model, the cache write
and read prices relative to plain input, the shortest prefix the provider caches, and whether
caching is `automatic` (OpenAI) or `marked` (Anthropic). `token_scale` is the seed for
[calibrated budgets](#token-budgets-and-priority-tiers). Both are optional in your own map and
both are lists of prices and ratios that providers change, so check them against your provider's
current pricing page. They were last checked against the OpenAI and Anthropic pricing pages on
2026-11-11; models retired by then (`gpt-4-turbo`, `claude-3-5-sonnet`, `claude-3-7-sonnet`,
`claude-3-opus`) were removed from the map and are now treated as unknown models. The map does
not yet list OpenAI models newer than `gpt-5.1`; add them through your own map.

```python
from membrane import get_cache_pricing, get_token_scale

print(get_cache_pricing("claude-opus-5-5"))  # mode='marked', write=1.25, read=0.05, min_tokens=512
print(get_token_scale("gpt-4o"))             # 1.0
```

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

The Anthropic Messages API takes the system prompt as a separate `system` parameter and only
caches what the request marks with `cache_control`. `anthropic_request` builds both from a
compile result:

```python
from membrane import ContextCompiler, MemoryItem, anthropic_request

compiler = ContextCompiler(max_tokens=8192, model_id="anthropic/claude-opus-5-5",
                           core_content="You are a helpful assistant.")
result = compiler.compile(memories=[MemoryItem(id="1", content="User is in Lagos.")],
                          user_message="What time zone am I in?")

request = {
    "model": "claude-opus-5-5",  # the API's model name, not Membrane's model_id
    "max_tokens": 1024,
    **anthropic_request(result),  # "system" blocks and "messages", cache_control placed
}
# import anthropic
# response = anthropic.Anthropic().messages.create(**request)
# usage = response.usage
# compiler.observe(result, usage.input_tokens + usage.cache_read_input_tokens
#                          + usage.cache_creation_input_tokens)
```

The markers go where `metadata["cache_breakpoints"]` says a cached prefix ends: after the core
system prompt, and after the conversation history. Tool calls and tool results in the history are
converted to `tool_use` and `tool_result` blocks. Nothing is marked below the model's minimum
cacheable size (512 to 4,096 tokens depending on the model).

The history is always marked. An earlier version held the marker back when a cut was expected
before the history could be read again; in a simulation of the 3,000-token scenario that saved
about 0.1% of input cost, so the rule was removed. Pass `ttl="1h"` for the one-hour cache. `anthropic_request` needs end placement (the
default); `anthropic_system_blocks(prefix, tail)` is still there for text you split yourself.

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
| `recall_evicted_history` | `True` | Bring passages of dropped history back as low-priority recall, picked by relevance to the question, then recency. `False` is the cheap mode: lower cost, a few points less accurate, and not recommended under very small budgets (about 3,000 tokens) |
| `rank_fusion` | `True` | When memories must be trimmed, choose survivors by your order fused with lexical relevance to the question |
| `message_overhead_tokens` | `3` | Tokens billed per message on top of content (OpenAI chat framing), reserved in the budget |
| `thin_before_drop` | `True` | With end placement, place every memory's keyframe before dropping any, then restore full text by priority |
| `keyframe_tokens` | `40` | Longest sentence that can be a keyframe |
| `eviction_index` | `True` | Keep one keyframe line per dropped history message in the system message, changed only when the cut moves |
| `index_share` | `0.125` | Most of the budget after the core that the eviction index may hold. Its room comes from what history, the question and memory keyframes leave (at least half the share), sized once per cut |
| `embedder` | `None` | Optional function from a list of texts to one vector each. Adds similarity to the question when choosing which memories keep their full text and which dropped passages come back. `None`: word matching only |
| `token_scale` | `None` | Provider tokens per counted token. `None`: seed from the capability map, corrected by `observe()`; a number fixes it |
| `metrics_callback` | `None` | Receives a `CompileMetrics` after each compile (token counts, never prompt text) |
| `preflight_analyst_model` | `None` | Model hint for your pre-flight compressor |

`compiler.observe(result, actual_prompt_tokens)` reports what the provider billed for a compiled
prompt and returns the scale now in use. `compile_anonymous(...)` takes the same history options
when called with `memory_placement="end"`.

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

### Ranking by meaning (experimental)

> Off by default, and in our own measurements it did not improve results (details below). It is
> here for you to test on your data, not as a recommended setting.

By default Membrane decides what gets its full text by the words an item shares with the
question. Pass `embedder=` and it also uses similarity of meaning, for two choices only: which
memories keep their full text under pressure, and which passages of dropped history come back.
Keyframes, the index of dropped history and the history cut do not use it, so the cached prefix
is the same with or without it.

```python
from membrane import ContextCompiler
from membrane.lod import fastembed_embedder  # pip install "membrane-aol[lod]"

compiler = ContextCompiler(max_tokens=8000, model_id="openai/gpt-4o",
                           embedder=fastembed_embedder())
```

`embedder` is any function from a list of texts to one vector each, so you can pass your own
model or an API client instead. `fastembed_embedder()` runs BAAI/bge-small-en-v1.5 locally on
the CPU; fastembed downloads the model (about 130 MB) the first time. Membrane itself still
makes no network calls and has no new dependency. Each text is embedded once per compiler and
the question once per turn; if the function raises, that compile falls back to word matching.
`metadata["semantic_ranking"]` is set when it was used. The similarity is fused with the lexical
order by reciprocal rank fusion (and with your retriever's order for memories), so it is one
vote, not a replacement.

Measured on the offline hard suite (eleven scenarios, 5 seeds, BAAI/bge-small-en-v1.5; run it
with `--strategies membrane,membrane-embed`), **it did not help**. Ten scenarios are identical
with and without it, in both modes. One got worse: history recall in the reworded spread-fact
scenario fell from 93.8% to 91.2% (two more of 80 questions missed); the likely cause, not verified, is a
passage that resembles the question displacing one that carried part of the fact. Cost did not move. Each
turn took roughly 40 to 220 ms longer on a laptop CPU (the model, not Membrane). The suite is a
weak test of it: memory recall is already 100% there, and the history it still misses is lost
when a fact is spread over sentences, not when the wrong passage is ranked first. It is likelier
to pay off where questions and stored text use different words and many memories compete, which
this suite's paraphrase scenarios only partly model. Treat it as untested on real conversations
and measure before turning it on.

Separately from `embedder`: with the `lod` extra installed, every compile that has history also
embeds the last user messages for topic-shift detection (below), which cost 100 to 200 ms per
compile on the same (busy) machine whether or not you pass `embedder`.

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
| **Membrane, defaults** | **60%** | 0 | 100% / 100% | **81 / 61** |
| Membrane, `memory_placement="system"` | 56% | 0 | 100% / 0% | 41 / 32 |

Recall costs tokens: with `recall_evicted_history=False` the 6,000-token run caches 69% and costs
66 / 44. It now finds every history fact too, through the index of dropped history (before the
index it found none); this benchmark states each fact in one sentence, the best case for the
index. With a 12,000-token budget Membrane caches 81% and costs 68 / 35. Under memory pressure
(4,000 tokens, 40 memories) it costs 87 / 73 and keeps the needed memory every time, against
95% for naive.

```bash
python -m benchmarks.peer_comparison [--budget N] [--memories N] [--live]
```

**Hard suite** ([`benchmarks/hard_suite.py`](benchmarks/hard_suite.py)): eleven stress
scenarios, 5 seeds each, against LangChain `trim_messages`, LlamaIndex `ChatMemoryBuffer` and
`Memory`, Priompt and a naive loop. Offline; recall means the answer was in the prompt, an upper
bound on accuracy. Every strategy gets the same budget in tiktoken tokens (Membrane runs with
`token_scale=1.0`). Mean over all scenarios (cost relative to naive = 100):

| | Cost at 0.5× / 0.1× cache reads | Recall | Turns over budget | Invalid tool requests |
| --- | ---: | ---: | ---: | ---: |
| Naive loop | 100 / 100 | 92% | 0 | 10 |
| LangChain `trim_messages` | 99 / 98 | 92% | 0 | 0 |
| LlamaIndex `ChatMemoryBuffer` | 99 / 98 | 92% | 409 | 0 |
| LlamaIndex `Memory` | 82 / 80 | 86% | 239 | 0 |
| Priompt-style cutoff (Python reimplementation) | 99 / 98 | 92% | 0 | 0 |
| Priompt (`@anysphere/priompt` 0.2.1 through Node) | 98 / 98 | 92% | 0 | 0 |
| **Membrane, defaults** | **82 / 62** | **99%** | 0 | 0 |
| Membrane, `recall_evicted_history=False` | 69 / 47 | 96% | 0 | 0 |

The Priompt row runs the real package with one element per conversation turn and memories
ranked above history; its recall equals the Python reimplementation's in all eleven scenarios.
Ranking the last three turns above memories instead changes its recall by less than a point.

Per scenario, Membrane defaults against the best of the others. "Before" is the same suite on
the previous state of the code (commit `cb43a15`: thin-before-drop, the first eviction index,
incremental counting), so the two columns show what the later changes did (index room, neighbour
sentences in keyframes, questions ranked behind statements, tool calls named in the index):

| Scenario | Membrane recall | Before | Best peer recall | Membrane cost | Before |
| --- | ---: | ---: | ---: | ---: | ---: |
| 80-turn chat, facts asked up to 30 turns later | history 100% | 100% | 76% | 76 / 52 | 75 / 54 |
| Tool calls, facts in old tool results | tool 100%, history 100% | 100%, 97% | 65%, 100% | 81 / 62 | 81 / 65 |
| Up to 40 memories per turn, needed one at a random rank | memory 100%, history 100% | 100%, 93% | 100%, 50% | 87 / 70 | 88 / 74 |
| Every answer needs a memory fact and a history fact | 100% | 97% | 100% | 81 / 61 | 83 / 65 |
| Three 2,500-token pastes | paste 100%, history 100% | 100%, 100% | 60%, 60% | 91 / 80 | 86 / 75 |
| 3,000-token budget | history 97% | 100% | 57% | 95 / 86 | 96 / 89 |
| Memory flood with paraphrased questions | memory 100%, history 96% | 100%, 72% | 100%, 32% | 86 / 70 | 89 / 75 |
| 24,000-token budget | 100% | 100% | 100% | 62 / 26 | 62 / 26 |
| History fact spread over two sentences of one message | history 96% | 94% | 75% | 80 / 58 | 80 / 62 |
| History fact spread over two or three sentences, varied wording | history 94% | 90% | 75% | 80 / 59 | 80 / 62 |
| The same, asked for in other words | history 94% | 86% | 75% | 80 / 58 | 80 / 62 |

With `recall_evicted_history=False` (the cheap mode: the index alone carries old history),
history recall before and after: 80-turn chat 92% to 95%, tool scenario 30% to 97%, memory
flood 83% to 100%, two-fact 87% to 99%, 3,000-token budget 86% to 89%, paraphrase 48% to 96%,
two-sentence fact 49% to 72%, spread fact 45% to 70%, spread and reworded 50% to 72%; pastes
and the 24,000-token budget stay at 100%. Mean recall 91% to 96%.

Where Membrane got worse or is behind:

- With a 3,000-token budget, history recall fell from 100% to 97%. The index now shrinks when
  memories need the room; without shrinking it stays at 100% but 13 of 9,890 offered memories
  lose their line.
- The paste scenario costs more: 91 / 80 against 86 / 75, because more history is kept word
  for word. The 80-turn chat costs one point more at 0.5× cache reads (76 against 75).
- The cheap mode costs more in five scenarios (for example 80-turn chat 55 / 29 against
  51 / 28, tool scenario 63 / 41 against 59 / 40), 69 / 47 on average against 69 / 50.
- Median compile time rose by about 0.5 ms in most scenarios (80-turn chat 0.9 to 1.9 ms,
  memory flood 2.6 to 3.3 ms), from keyframes that keep neighbour sentences. Pastes fell from
  3.3 to 0.8 ms.
- A fact spread over several sentences is still the weak spot: 94% to 96% with recall on, 70%
  to 72% in the cheap mode. A keyframe keeps a neighbour sentence only when that sentence opens
  with a pronoun or is very short; the two-sentence scenario's second sentence opens with
  "The", so the rule does not fire there.
- The rule that ranks questions behind statements is flattered by this suite: every user
  message in it opens with a question and then states a fact. Expect less on real
  conversations. It accounts for most of the paraphrase gain (72% to 96%).
- LangChain, Priompt and the naive loop keep 100% of two-fact answers and of old history in
  the tool scenario; Membrane now matches them there, at lower cost, but does not beat them.

Membrane picks dropped history and surviving memories partly by the words they share with the
question; most scenarios name what they ask for, which helps that. The paraphrase and
reworded scenarios remove the help. The benchmark text is synthetic and templated; nothing in
the keyframe scoring was fitted to it.

The peers put retrieved memories in the system message, as their RAG chat patterns do. Moving
them next to the question by hand gives the same caching in any of them. Median compile time is
0.5 to 3.6 ms per turn at these sizes (1.9 ms on the 80-turn chat); a turn that moves the
history cut takes 2 to 9 ms. LangChain and the Priompt reimplementation take 0.1 to 0.2 ms
with token counts supplied by the harness, real Priompt 7 to 16 ms including the round trip to
Node, LlamaIndex 2 to 80 ms.

```bash
uv run --extra langchain --extra llamaindex python -m benchmarks.hard_suite
```

The real Priompt rows need `node` and one `npm install` in `benchmarks/priompt_node`; without
them the suite skips those rows.

**Live run** ([`benchmarks/hard_suite_live.py`](benchmarks/hard_suite_live.py)): the suite's
requests sent to the Claude API. Accuracy means the reply contained the fact. "Wrong" counts
replies that stated a different code instead of saying it was unknown. Cost is uncached input
+ 1.25 × cache writes + 0.1 × cache reads, relative to LangChain = 100 (Claude Opus 5.5 bills
reads at 0.05×, which widens the gaps).

Latest run, 2026-11-11, **one seed** (seed 1) on each model, all eleven scenarios, 430 requests
per strategy, on the code at commit `61b8df6`. `claude-opus-5-5` ran four strategies;
`claude-haiku-5-5` also ran real Priompt. Mean over scenarios:

| | Opus 5.5 accuracy | Wrong | Read from cache | Cost | Haiku 5.5 accuracy | Wrong | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LangChain `trim_messages` | 85.1% | 21 | 33% | 100 | 83.6% | 25 | 100 |
| LlamaIndex `Memory` | 78.6% | 40 | 39% | 81 | 76.5% | 42 | 81 |
| Priompt (`@anysphere/priompt` through Node) | not run | | | | 82.8% | 26 | 100 |
| **Membrane, defaults** | **99.3%** | **2** | 65% | 63 | **96.3%** | **7** | 59 |
| Membrane, `recall_evicted_history=False` | 97.0% | 9 | 75% | 43 | 95.0% | 12 | 45 |

Before and after on `claude-opus-5-5`, same seed, the nine scenarios both runs have. "Before"
is the one-seed run on `cb43a15` described below. Accuracy, then cost against LangChain = 100:

| Scenario | Membrane before | after | Cheap mode before | after | LangChain after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 80-turn chat | 100%, 57 | 100%, 53 | 99%, 21 | 99%, 27 | 95% |
| Tool calls | 100%, 66 | 100%, 64 | 90%, 42 | 100%, 37 | 88% |
| Memory flood | 100%, 68 | 100%, 69 | 97%, 73 | 100%, 59 | 90% |
| Two-fact answers | 97%, 68 | 100%, 65 | 90%, 46 | 100%, 37 | 100% |
| Three pastes | 100%, 81 | 100%, 76 | 100%, 38 | 100%, 47 | 83% |
| 3,000-token budget | 100%, 102 | 100%, 92 | 97%, 77 | 93%, 58 | 90% |
| Paraphrased questions (history only) | 80%, 73 | 100%, 60 | 60%, 60 | 100%, 65 | 20% |
| 24,000-token budget | 100%, 23 | 100%, 26 | 100%, 27 | 100%, 24 | 100% |
| Fact over two sentences | 98%, 65 | 98%, 54 | 82%, 29 | 90%, 42 | 90% |
| **Mean of nine** | 97.1%, 67 | 99.7%, 62 | 90.5%, 46 | 98.0%, 44 | 84.0% |
| Confident wrong answers, nine scenarios | 2 | 1 | 15 | 5 | 19 |

The two new scenarios (no "before"): Membrane 98% in both at cost 64, cheap mode 92% at 46 and
32, LangChain 90%, LlamaIndex `Memory` 78%.

What got worse or stayed behind in the live runs:

- Cheap mode, 3,000-token budget: 97% to 93% on Opus (history 86% to 71%), and 93% on Haiku.
- Cheap mode costs more than before in four scenarios on Opus: 80-turn chat 21 to 27, pastes
  38 to 47, paraphrase 60 to 65, two-sentence fact 29 to 42. Defaults cost more in two: memory
  flood 68 to 69, 24,000-token budget 23 to 26.
- On Haiku 5.5 the defaults were below the cheap mode in the tool scenario (tool facts 92%
  against 100%) and below LangChain and Priompt on two-fact answers (96% against 100%), and gave
  7 confident wrong answers, against 2 on Opus.
- Facts spread over sentences: 94% of those history questions on Opus, 88% on Haiku.
- The paraphrase jump (80% to 100%) rests on 10 history questions in one seed.

This is one seed per model: 30 to 80 questions per scenario, so a difference of one or two
questions (3 to 7 points in the smaller scenarios) is within what the same strategy shows
between runs. LangChain's requests did not change between the two Opus runs, yet its accuracy
moved from 85% to 88% in the tool scenario and from 97% to 100% on two-fact answers. Three
seeds were planned; one was run to keep the spend under the $35 the run was allowed (about $29
for Opus and $1.30 for Haiku at list prices).

The run before that: **one seed**, nine scenarios, 350 requests per strategy, `claude-opus-5-5`
on 2026-11-10, on the code at commit `cb43a15` (before the index-room, neighbour-sentence,
question and tool-call changes above). Mean over scenarios:

| | Accuracy | Confident wrong answers | Prompt tokens read from cache | Cost |
| --- | ---: | ---: | ---: | ---: |
| LangChain `trim_messages` | 83.3% | 19 | 33% | 100 |
| LlamaIndex `Memory` | 78.9% | 36 | 38% | 83 |
| **Membrane, defaults** | **97.1%** | **2** | 61% | 67 |
| Membrane, `recall_evicted_history=False` | 90.5% | 15 | 73% | 46 |

In that run Membrane had the best or equal accuracy in every scenario and cost less in eight of
nine; under a 3,000-token budget it cost 102. With `recall_evicted_history=False` it was below
LangChain in two-fact questions (90% against 97%) and the two-sentence scenario (82% against
90%). One seed means 30 to 80 questions per scenario, so single-scenario differences of a few
points are noise.

Things to know when reading the live numbers. The paraphrase scenario's memory questions are
left out of accuracy, because they name projects by an alias no model can resolve. Assistant
turns are scripted, so every strategy sees the same history. An earlier run on the release
before these mechanisms (eight scenarios, one seed: Membrane 92% at cost 69) is kept in
[`benchmarks/results/`](benchmarks/results/) with the other raw results.

```bash
uv run --extra langchain --extra llamaindex --extra llm-anthropic python -m benchmarks.hard_suite_live
```

The run needs Claude API credentials and costs API credits: about $26 per seed at list prices
for the default four strategies on Claude Opus 5.5 over eleven scenarios, about $10 more with
`priompt-node`.

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
