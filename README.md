# Membrane

> Prompt engineering infrastructure — token budgets, priority trimming, U-shaped layout, and prompt caching for LLM apps.

---

## The problem

Every LLM application eventually hits the same wall: you concatenate system instructions, retrieved memories, conversation history, and the user's message into one big string — and things break.

- **Context overflow** — you exceed the model's token limit and get truncated or rejected.
- **Lost in the middle** — research shows LLMs attend weakly to the middle of long prompts, so important facts placed there are effectively invisible (Liu et al., 2023).
- **No caching** — every request recomputes the full prompt, even when 80% of the system message is identical across calls.
- **No budget** — you have no idea how many tokens each section costs until you hit the limit.

Membrane treats the LLM context window like an operating system treats RAM: **fixed capacity**, **eviction policies**, **explicit placement**, and **caching**. Every prompt is _compiled_ — never concatenated.

---

## Install

Membrane is not on PyPI yet. Install from source:

```bash
git clone https://github.com/adeyomilawal/membrane.git
cd membrane
pip install -e .            # core
pip install -e ".[all]"     # common optional extras
```

For development:

```bash
uv sync --locked --extra dev
```

| Extra           | What it adds                                                     |
| --------------- | ---------------------------------------------------------------- |
| `lod`           | Level-of-detail attention shift (numpy, fastembed)               |
| `redis`         | Redis-backed ref_map cache, stress persistence, async pre-flight |
| `stress`        | CPU load sampling for environment snapshots (psutil)             |
| `pydantic`      | Pydantic model support                                           |
| `llm-openai`    | OpenAI SDK for pre-flight analyst adapters                       |
| `llm-anthropic` | Anthropic SDK for pre-flight analyst adapters                    |
| `all`           | Everything in the `all` group in `pyproject.toml`                |

---

## Quickstart

Compile a prompt in 10 lines:

```python
from membrane import ContextCompiler, MemoryItem, CompileResult

compiler = ContextCompiler(
    max_tokens=8192,
    model_id="openai/gpt-4o",
    core_content="You are a helpful assistant.",
)

memories = [
    MemoryItem(id="1", content="User prefers concise answers.", memory_type="semantic", priority=2),
    MemoryItem(id="2", content="Discussed asyncio yesterday.", memory_type="episodic", priority=3),
]

result: CompileResult = compiler.compile(
    memories=memories,
    user_message="Show a producer-consumer example.",
)

print(result.messages)       # API-ready message list
print(result.ref_map)        # {1: "1", 2: "2"}
print(result.metadata["token_count"])
```

For async workflows with pre-retrieved memory pairs and optional LLM compression:

```python
from membrane import compile_prompt_async, TIER_SEMANTIC

async def compress(raw: str, target: int, user_msg: str) -> str:
    ...  # your LLM call

result = await compile_prompt_async(
    user_msg="Hello",
    messages_list=[{"role": "user", "content": "Hi"}],
    user_id="user-1",
    max_tokens=8192,
    retrieved_objects=[(my_fact, TIER_SEMANTIC)],
    compress_fn=compress,
)
```

---

## Core concepts

The kernel runs a deterministic pipeline: **Layout → Pre-flight compression → Firewall → Glaciator → Provider format**.

1. **`ContextLayoutEngine`** builds three positional regions (core / middle / end).
2. An optional **pre-flight analyst** (your LLM adapter) compresses the middle when it exceeds a token threshold.
3. **`PreFlightContextFirewall`** enforces the hard token budget with exact tiktoken counting and priority-tier trimming.
4. **`glaciate()`** splits the system message into a cacheable prefix and dynamic tail for prompt caching.
5. **Provider formatters** (`build_system_messages`, `anthropic_system_blocks`) produce API-ready messages.

See [Architecture](docs/architecture.md) for the full pipeline diagram.

---

## U-shaped layout

LLMs attend most strongly to the **start** and **end** of their context window. The middle is a "sacrifice zone" — content placed there is the least influential (Liu et al., 2023 — _Lost in the Middle: How Language Models Use Long Contexts_).

The kernel encodes this by construction:

| Region     | Content                                    | Attention           |
| ---------- | ------------------------------------------ | ------------------- |
| **Start**  | System rules, persona, interrupt protocol  | High                |
| **Middle** | Retrieved memories (`[Mem:N]` indexed)     | Low — trimmed first |
| **End**    | Last assistant turn + current user message | High                |

See [U-shaped layout](docs/u-shaped-layout.md) for the full rationale and academic citation.

---

## Priority trimming

When the compiled prompt exceeds `max_tokens`, the firewall trims the middle by **tier**:

| Tier | Constant          | Priority               |
| ---- | ----------------- | ---------------------- |
| 1    | `TIER_PROCEDURAL` | Highest — trimmed last |
| 2    | `TIER_SEMANTIC`   | Medium                 |
| 3    | `TIER_EPISODIC`   | Lowest — trimmed first |

Assign tiers when you build your `(memory_object, tier)` pairs. The firewall removes the lowest-priority segments first, preserving the U-shaped positional layout. The `ref_map` is automatically pruned to match surviving `[Mem:N]` references.

---

## Prompt caching

Different providers implement prompt caching differently:

| Provider           | Mechanism                                 | Kernel support                                                                                |
| ------------------ | ----------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Anthropic**      | `cache_control` blocks on system messages | `anthropic_system_blocks(prefix, tail)` — first block gets `cache_control: {type: ephemeral}` |
| **OpenAI**         | Cached prefix matching                    | `build_system_messages(model_id, prefix, tail)` — returns caching hint                        |
| **Ollama / local** | No prompt caching                         | Full system string as one message (glaciation skipped)                                        |

`glaciate()` splits the system text at line boundaries so the cacheable prefix is stable across requests. `provider_registry.has_prompt_caching(model_id)` checks whether a model supports caching.

By default (`memory_placement="system"`) retrieved memories sit inside the system message, so the cache cannot hit past the core: memories change every turn. Pass `ContextCompiler(..., memory_placement="end")` to emit the system message (core only), then `conversation_messages` as given (tool calls and content-part lists kept), then one user message with the kept memories, the current user message and dynamic directives. The system message plus history then form a byte-stable prefix across turns, and the firewall still trims memories and prunes `ref_map` (history is reserved, never trimmed; an overrun sets `metadata["budget_exceeded"]`). End mode sends the full history, which the default mode does not, so requests are larger but mostly cache-served. `tests/test_prompt_cache_stability.py` prints the reusable prefix for both modes.

See [Prompt caching](docs/prompt-caching.md) for per-provider details.

---

## Interrupt protocol

When the compiled context is incomplete, the kernel can instruct the model to emit a machine-readable tag instead of hallucinating:

```
<SYSTEM_INTERRUPT_FETCH[precise query describing what information is needed]>
```

The host detects this tag in the output stream, runs retrieval with the query, injects results, and continues generation.

```python
from membrane import parse_interrupt, contains_interrupt, strip_all_interrupts

text = "I need more info. <SYSTEM_INTERRUPT_FETCH[user's timezone preference]> Let me check."
if contains_interrupt(text):
    result = parse_interrupt(text)
    # result = ("I need more info. ", "user's timezone preference", " Let me check.")
```

`strip_all_interrupts(text)` removes all tags after a maximum number of "page faults".

See [Interrupt protocol](docs/interrupt-protocol.md) for host integration patterns.

---

## LOD / attention shift

The Level-of-Detail engine detects topic shifts by comparing the embedding centroid of the current query against the recent conversation window:

```python
from membrane import compute_attention_shift

shifted = compute_attention_shift(
    current_query="How do I deploy to Kubernetes?",
    recent_queries=["What's the best Python testing framework?", "How does pytest work?"],
)
# shifted = True — topic changed from testing to deployment
```

When a shift is detected (cosine distance above `LOD_ATTENTION_SHIFT_COSINE_THRESHOLD`, default 0.35), the host should re-retrieve memories at a different level of detail. Requires the `lod` extra (`numpy`, `fastembed`).

---

## Benchmarks

Benchmark scripts are included in `benchmarks/`:

- **`token_savings_bench.py`** — measures token reduction from pre-flight compression and priority trimming across prompt sizes.
- **`trimming_bench.py`** — validates that trimming preserves high-priority content and maintains the U-shaped positional layout.

Run benchmarks:

```bash
uv run python benchmarks/token_savings_bench.py
uv run python benchmarks/trimming_bench.py
```

---

## CLI diagnostics

Compiler diagnostics run locally, with no network access:

```bash
membrane inspect \
  --user-message "Explain the retry policy." \
  --memory "policy|2|Retries use exponential backoff with jitter." \
  --max-tokens 2048

membrane quote \
  --user-message "Summarize the release plan." \
  --memory "release|2|Version 0.2 ships after the integration tests pass." \
  --price-per-1k-input-tokens 0.00015

membrane profile list
membrane profile explain coding
```

`inspect` shows token/layout diagnostics. `quote` runs the same compile pipeline
and adds input-token cost guidance when you provide a price. `profile` lists the
built-in profiles and what each one enables.

---

## API reference

### Core pipeline

| Symbol                   | Type           | Purpose                                                            |
| ------------------------ | -------------- | ------------------------------------------------------------------ |
| `ContextCompiler`        | class          | Synchronous prompt compiler (layout → firewall → glaciate)         |
| `compile_prompt_async`   | async function | Full async pipeline with pre-flight compression, LOD, stress       |
| `compile_prompt_payload` | async function | Alias for `compile_prompt_async`                                   |
| `compile_anonymous`      | function       | Stateless compile (no user/conversation context)                   |
| `CompileResult`          | dataclass      | Return type: `.messages`, `.ref_map`, `.ref_map_type`, `.metadata` |

### Layout & serialisation

| Symbol                                                   | Type      | Purpose                                           |
| -------------------------------------------------------- | --------- | ------------------------------------------------- |
| `ContextLayoutEngine`                                    | class     | U-shaped layout builder (aliases: `LayoutEngine`) |
| `DenseSerializer`                                        | class     | Memory → compact string formatter                 |
| `MemoryItem`                                             | dataclass | Concrete memory record for RAG rows               |
| `KernelMemoryRecord`                                     | protocol  | Duck-typing protocol for memory objects           |
| `SemanticMemory` / `EpisodicMemory` / `ProceduralMemory` | dataclass | Typed memory records                              |

### Firewall

| Symbol                                                | Type     | Purpose                                               |
| ----------------------------------------------------- | -------- | ----------------------------------------------------- |
| `PreFlightContextFirewall`                            | class    | Token budget enforcement (alias: `PreFlightFirewall`) |
| `count_tokens`                                        | function | Exact tiktoken token count                            |
| `TIER_PROCEDURAL` / `TIER_SEMANTIC` / `TIER_EPISODIC` | int      | Priority tier constants                               |

### Caching

| Symbol                    | Type      | Purpose                                                |
| ------------------------- | --------- | ------------------------------------------------------ |
| `glaciate`                | function  | Split system text into cacheable prefix + dynamic tail |
| `GlaciateResult`          | dataclass | Return type: `.prefix`, `.tail`, `.use_caching`        |
| `build_system_messages`   | function  | OpenAI-style system message with caching hint          |
| `anthropic_system_blocks` | function  | Anthropic content blocks with `cache_control`          |

### Interrupt

| Symbol                 | Type     | Purpose                                           |
| ---------------------- | -------- | ------------------------------------------------- |
| `parse_interrupt`      | function | Extract first `<SYSTEM_INTERRUPT_FETCH[...]>` tag |
| `contains_interrupt`   | function | Boolean check for interrupt tags                  |
| `strip_all_interrupts` | function | Remove all interrupt tags from text               |

### LOD

| Symbol                                    | Type     | Purpose                                                      |
| ----------------------------------------- | -------- | ------------------------------------------------------------ |
| `compute_attention_shift`                 | function | Detect topic shift from query strings (requires `lod` extra) |
| `compute_attention_shift_with_embeddings` | function | Detect topic shift from pre-computed embedding arrays        |
| `cosine_distance`                         | function | Cosine distance between two vectors                          |

### Environment & stress

| Symbol                                                       | Type      | Purpose                                   |
| ------------------------------------------------------------ | --------- | ----------------------------------------- |
| `EnvironmentSnapshot`                                        | dataclass | Request-scoped environment state          |
| `compute_stress_from_environment`                            | function  | Environment signals → scalar stress score |
| `set_environment_snapshot` / `get_current_environment_state` | function  | `ContextVar`-based snapshot management    |

### Provider registry

| Symbol                       | Type     | Purpose                             |
| ---------------------------- | -------- | ----------------------------------- |
| `get_capabilities_for_model` | function | Full capability dict for a model ID |
| `has_prompt_caching`         | function | Check prompt caching support        |
| `has_streaming_interrupt`    | function | Check streaming interrupt support   |
| `get_context_cap_for_model`  | function | Context window size for a model     |

### Redis-backed features (optional)

| Symbol                                                                | Type           | Purpose                                   |
| --------------------------------------------------------------------- | -------------- | ----------------------------------------- |
| `set_ref_map_for_conversation` / `get_ref_map_for_conversation`       | async function | Persist / retrieve `ref_map` in Redis     |
| `set_recent_stress_for_user` / `get_recent_stress_for_user`           | async function | Persist / retrieve stress scores in Redis |
| `run_async_preflight` / `get_preflight_cache` / `set_preflight_cache` | async function | Async pre-flight compression cache        |

---

## Tests

From this directory, with the package on `PYTHONPATH` (editable install is easiest):

```bash
pip install -e ".[dev]"
pytest
```

`pyproject.toml` sets `pythonpath = ["src"]` so `import membrane` resolves without manual `PYTHONPATH` when dependencies (including `tiktoken`) are installed.

---

## Documentation

- [Architecture](docs/architecture.md) — pipeline diagram, OS metaphor, stage details
- [U-shaped layout](docs/u-shaped-layout.md) — rationale with academic citation
- [Prompt caching](docs/prompt-caching.md) — per-provider caching guide
- [Interrupt protocol](docs/interrupt-protocol.md) — mid-generation retrieval

---

## License

Membrane is licensed under the [Apache License, Version 2.0](LICENSE). You can use it, modify it, and ship it in commercial and noncommercial products. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
