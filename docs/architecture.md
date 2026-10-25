# Architecture

The Memory OS Kernel treats the LLM context window like an operating system treats RAM: fixed capacity, eviction policies, and explicit placement of content. Every prompt runs through a deterministic pipeline before reaching the model.

## The OS metaphor

| OS concept                   | Kernel analogue                                                                                                             |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **RAM capacity**             | `max_tokens` budget enforced by `PreFlightContextFirewall` with tiktoken                                                    |
| **Page eviction**            | Priority-tier trimming — low-tier segments are evicted first when the budget is exceeded                                    |
| **Virtual memory layout**    | U-shaped layout positions high-importance content at the start and end of the context (where LLM attention is strongest)    |
| **Disk cache**               | Glaciator splits the system message into a cacheable prefix and dynamic tail for provider prompt-caching                    |
| **Page faults / interrupts** | `<SYSTEM_INTERRUPT_FETCH[query]>` tag lets the model request missing data mid-generation                                    |
| **Process environment**      | `EnvironmentSnapshot` captures request-scoped signals (latency, CPU, tool failures) as a stress score for somatic filtering |

## Pipeline diagram

```
                    ┌────────────────────────────────--─┐
                    │       compile_prompt_payload()    │
                    └───────────────┬───────────────--──┘
                                    │
                    ┌───────────────▼────────────────--─┐
                    │       Layout Engine               │
                    │  ┌───────────────────────────-──┐ │
                    │  │ START: Core + Persona + Rules│ │  ← High attention
                    │  ├────────────────────────────-─┤ │
                    │  │ MIDDLE: Sacrifice Zone       │ │  ← Low attention
                    │  │ [Mem:1] Semantic facts       │ │    (trimmed first)
                    │  │ [Mem:2] Episodic memories    │ │
                    │  │ [Mem:3] Procedural steps     │ │
                    │  ├────────────────────────────-─┤ │
                    │  │ END: Last turn + User query  │ │  ← High attention
                    │  │ + ACTIVE_SYSTEM_DIRECTIVES   │ │
                    │  └─────────────────────────────-┘ │
                    └───────────────┬────────────────--─┘
                                    │
                    ┌───────────────▼────────────────--─┐
                    │       PreFlight Firewall          │
                    │  Exact tiktoken counting          │
                    │  Priority-tier trimming           │
                    │  Reference map pruning            │
                    └───────────────┬────────────────--─┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼──────────┐  ┌─── ───▼─────────┐  ┌─────--─▼────────┐
    │   Glaciator        │  │ LOD Engine      │  │  Stress Provider│
    │   Cache prefix     │  │ Attention shift │  │  Somatic filter │
    │   + dynamic tail   │  │ detection       │  │  (epigenetic)   │
    └─────────┬──────────┘  └─────────────────┘  └─────────────────┘
              │
    ┌─────────▼─────────--─┐
    │ Provider Formatter   │
    │ Anthropic / OpenAI   │
    │ / Ollama message fmt │
    └─────────┬────────--──┘
              │
              ▼
    API-ready messages[]
```

## Stage details

### 1. Layout (`ContextLayoutEngine`)

Builds three positional regions following the U-shaped attention hypothesis:

- **`build_core()`** — Assembles the system prompt: persona text, `INTERRUPT_RULE` injection (if the model supports streaming interrupts), and any static system directives.
- **`build_directives_block()`** — Appends `ACTIVE_SYSTEM_DIRECTIVES` after the core system text when present.
- **`build_middle()`** — Serialises retrieved memory objects using `DenseSerializer` into compact `[Mem:N]` indexed lines. Returns a `ref_map` (local index → global UUID) and `ref_map_type` (index → memory kind) for downstream resolution.
- **`build_end()`** — Formats the recent conversation tail: last assistant turn (if any) and the current user message.
- **`assemble()`** — Concatenates core + directives + middle + end in the correct order.

### 2. Pre-flight compression (optional)

When the raw middle zone exceeds `MEMORY_PREFLIGHT_TOKEN_THRESHOLD` (default 8 000 tokens), an optional `PreFlightAnalyst` — your LLM adapter — compresses it to `PREFLIGHT_TARGET_TOKENS`. The async pre-flight cache (`async_preflight.py`, Redis-backed) stores compression results keyed by `(conversation_id, user_text_hash)` so that compression can run speculatively while the user is typing.

### 3. Firewall (`PreFlightContextFirewall`)

Enforces a hard token budget using **tiktoken** (exact BPE token counting, not approximations):

1. Measures `core_tokens + end_tokens`.
2. Allocates remaining budget to the middle.
3. If the middle exceeds the budget, trims by tier in ascending priority order (`TIER_PROCEDURAL` < `TIER_SEMANTIC` < `TIER_EPISODIC`). Within a tier, segments furthest from the U-shaped edges are removed first.
4. Prunes `ref_map` to contain only surviving `[Mem:N]` indices.

### 4. Glaciator

For models that support provider prompt-caching (Anthropic `cache_control`, OpenAI cached prefixes), `glaciate()` walks the final system string **line by line** until the cacheable portion fits under `MAX_CACHEABLE_PREFIX_TOKENS`. The remainder becomes the dynamic tail. Line-boundary splitting avoids unstable cache keys.

### 5. Provider formatter (`cache_format.py`)

- **OpenAI-style** — `build_system_messages(model_id, prefix, tail)` returns `[{"role": "system", "content": prefix + "\n\n" + tail}]` with a caching hint boolean.
- **Anthropic** — `anthropic_system_blocks(prefix, tail)` returns content blocks where the first block carries `cache_control: {type: ephemeral}` when the prefix is non-empty.
- **Ollama / local** — Models without `prompt_caching` in the capability map skip glaciation; the full system string is sent as one message.

### 6. LOD engine (optional)

`compute_attention_shift()` compares the embedding centroid of the current query against the centroid of the recent conversation window. A cosine distance above `LOD_ATTENTION_SHIFT_COSINE_THRESHOLD` signals a topic shift — the host can re-retrieve memories at a different "level of detail".

### 7. Stress provider (optional)

`compute_stress_from_environment()` converts `EnvironmentSnapshot` signals (latency, CPU load, tool failure count, loop retries) into a scalar `[0, 1]` stress score. This can be used by the host for somatic-style filtering of episodic memories (downweight calm memories during high-stress requests, or vice versa).

## Boundaries

- **No bundled LLM HTTP client** — you pass `compress_fn` or a `PreFlightAnalyst` adapter. The kernel never calls an LLM directly.
- **No vector database or MIRIX** — memory objects satisfy duck-typing via `KernelMemoryRecord` protocol, `MemoryItem` dataclass, or `SemanticMemory` / `EpisodicMemory` / `ProceduralMemory` classes.
- **No CMG or REM** — inject extra middle segments yourself if needed.
- **No Redis required for core pipeline** — Redis features (stress persistence, ref_map caching, async pre-flight) are all optional; the core compile path works with zero external dependencies beyond `tiktoken`.

## Configuration

All pipeline knobs are environment-variable-driven via `membrane.config`:

| Variable                               | Default              | Purpose                                                 |
| -------------------------------------- | -------------------- | ------------------------------------------------------- |
| `MEMORY_PREFLIGHT_TOKEN_THRESHOLD`     | 8000                 | Token count above which pre-flight compression triggers |
| `PREFLIGHT_TARGET_TOKENS`              | 500                  | Target token count for compression output               |
| `MAX_CACHEABLE_PREFIX_TOKENS`          | 2000                 | Maximum tokens for the glaciated cacheable prefix       |
| `LOD_ATTENTION_SHIFT_COSINE_THRESHOLD` | 0.35                 | Cosine distance threshold for topic shift detection     |
| `MEMORY_OS_KERNEL_REDIS_URL`           | None                 | Redis connection URL for optional features              |
| `LLM_CAPABILITY_MAP_PATH`              | None                 | Override path for `llm_capabilities.json`               |
| `PREFLIGHT_ANALYST_MODEL`              | `openai/gpt-4o-mini` | Model hint for the pre-flight analyst                   |

See `membrane.config` for the full set including TTLs and key prefixes.
