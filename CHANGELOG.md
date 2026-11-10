# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Default output changes for end placement (history cut, tail layout, system message), so release as 0.3.0.

- **Thin before drop** (`thin_before_drop=True`, end placement). Every memory has a keyframe: its densest sentence under `keyframe_tokens` (40), by how rare its words are across the sentences of the memories on offer and of the conversation. Under pressure all keyframes are placed first, then memories get their full text back in priority order. A thinned memory keeps its `[Mem:N]` ref and ends in `…`; `metadata["segments_thinned"]` counts them. `memory_placement="system"` is unchanged.
- **Eviction index** (`eviction_index=True`, `index_share=0.125`). Each dropped history message leaves one keyframe line in the system message after the core (tool results included, tool-call stubs not; one keyframe per 400 tokens of a long message). A line is scored at the cut that drops its message and never changes, so the index only changes when the cut moves and adds no cache breaks. Its room comes out of `history_share`, so less history is kept word for word than in 0.2.x. New metadata: `history_index_lines`, `history_index_tokens`, `core_chars`.
- Recalled passages that do not fit whole are cut to the sentence that matches the question. No memory is dropped to make room for a passage.
- **Price-aware layout.** The capability map has `cache_pricing` per model (write and read multipliers, minimum cacheable tokens, automatic or marked caching) and matches the current Claude families (`claude-opus-*`, `claude-sonnet-*`, `claude-haiku-*`, `claude-fable-*`). Compile reports `metadata["cache_breakpoints"]` and `cache_mode`; `anthropic_request(result)` builds Anthropic Messages API parameters with `cache_control` placed and tool calls converted. The history is marked only when it is expected to be read before the next cut. `prompt_cache_used` now uses the model's own minimum instead of 1,024 for every model. New exports: `anthropic_request`, `get_cache_pricing`.
- **Calibrated budgets.** `compiler.observe(result, actual_prompt_tokens)` keeps a per-model scale between counted tokens and the provider's tokens; the budget is divided by it. Seeds come from the map's `token_scale` (1.0 OpenAI, 1.2 older Claude, 1.5 Claude Opus 4.7 and later), so **budgets for Claude models are tighter by default than in 0.2.x**; pass `token_scale=1.0` for the old behaviour. New metadata: `token_scale`, `token_scale_samples`, `tokens_budget_counted`. New export: `get_token_scale`.
- **Incremental counting.** The joined history is no longer tokenised each compile; the tail is counted in parts that add up exactly with the built-in tokenizer; the core, the index and recall ranking are computed once per cut. Median compile time on an 80-turn chat: 8.6 ms to 0.9 ms. A compiler object now keeps small per-conversation caches, so reuse one compiler per model.
- `compile_anonymous` takes `memory_placement="end"` with the compiler's history options, and its system layout reserves message framing and applies `token_scale`.
- Hard suite: new `split-fact` scenario (a history fact spread over two sentences of one message). Offline results over nine scenarios: recall 95% to 99% at the same cost (82 / 65). Worse than before: history recall in the memory-flood scenario (100% to 93%), two-fact questions (98% to 97%), cost up 1 to 3 points in four scenarios, and with `recall_evicted_history=False` history recall fell in four scenarios (tool scenario 57% to 30%). The live run in the README has not been repeated since these changes.

- The budget now covers provider message framing: `message_overhead_tokens` (default 3, OpenAI chat framing) per message plus reply priming, so the billed request stays under `max_tokens`. `token_count` stays content-only; `metadata["message_framing_tokens"]` reports the reserve.
- End placement: the history cut depends only on the budget, the core and framing (`history_share`, default 0.75), on a fixed chunk grid. It no longer shifts with each turn's memories or question, so the cached prefix breaks less often.
- End placement: the most recently dropped history chunk returns in the tail as low-priority recall (`recall_evicted_history=True`, `metadata["history_messages_recalled"]`). It is trimmed before any memory, skips tool messages, and never enters `ref_map`.
- End placement: recalled history is now picked by relevance. Dropped messages are split into passages of about 60 words, tool results included, and ranked by the rare words they share with the user message, then by recency. A long paste comes back as its relevant passage instead of not at all.
- End placement: `rank_fusion=True` chooses which memories survive trimming by reciprocal rank fusion of the order you pass and lexical relevance to the user message. Survivors keep your order; nothing changes when everything fits or when a reranker supplies weights.
- `count_tokens` caches counts for texts up to 8,000 characters. Compile time on an 80-turn chat dropped from about 40 ms to about 9 ms.
- New `benchmarks/hard_suite_live.py`: runs the hard suite against the Claude API for real answer accuracy and prompt-cache usage. First run (`claude-opus-5-5`, one seed) is in the README and `benchmarks/results/`.
- New `benchmarks/hard_suite.py`: eight stress scenarios against LangChain `trim_messages`, LlamaIndex `ChatMemoryBuffer` and `Memory`, a Priompt-style cutoff and a naive loop. Results are in the README.
- New `benchmarks/peer_comparison.py`: naive concatenation, a Priompt-style priority cutoff, and Membrane under a hard budget, with an opt-in `--live` mode against OpenAI.

## [0.2.1] — 2026-11-10

- Raise test coverage from 68% to 88%; CI now fails below 80%.
- Fix step numbering in `DenseSerializer` full prose: steps without a `step_number` were numbered by output line instead of position.
- `EstimationAdapter.encode("")` returns no tokens, matching `count("")`.
- Stored ref maps that are valid JSON but not an object are ignored instead of raising.
- Ship a `py.typed` marker so type checkers use Membrane's annotations.
- Link the project website from the package metadata and README.

## [0.2.0] — 2026-11-10

First open-source release, published on PyPI as **`membrane-aol`** (imports as `membrane`).

- **Default `memory_placement` is now `"end"`**, so system message and history stay cacheable across turns. Pass `memory_placement="system"` for the previous layout (better for local models and one-off calls).
- End placement trims the oldest history in large, stable chunks when it would crowd out the budget, leaving room for memories (`trim_history=True`, `metadata["history_messages_dropped"]`). Directives are now reserved inside the budget in end placement too.

- Relicense under **Apache-2.0** (previously PolyForm Noncommercial).
- Remove license activation, tier gates, portal analytics, and the CLI `login`, `logout`, `whoami`, `status`, `refresh`, and `token` commands. Every feature is available without activation.
- Drop the `PyJWT`, `httpx`, and `keyring` runtime dependencies.
- Remove `CompileMetrics.tier`.
- Remove from the open-source edition: JTA and other compression strategies, lineage, assertions, replay, diff/version store, export, batch, reranker, visualizer, testing utilities, Langfuse/LangSmith exporters, and the interrupt runner. The `research`, `full`, and `enterprise` profiles are removed with them.
- Add `FeatureNotAvailable`, raised by `CompileResult.diff()` and `enable_lineage=True` without a `lineage_tracker`.
- Add `memory_placement="end"` for prompt-cache-stable compiles.

### Fixed (code review)

- Token budget: the firewall now checks the assembled text, and dynamic directives are reserved inside the budget, so `token_count <= max_tokens` unless `metadata["budget_exceeded"]` is set. That flag is now set in every mode, including `compile_anonymous`. Tight budgets may trim one more memory than before.
- Prompt-cache metadata reports the real byte-stable prefix (the core in system mode; system plus history in end mode) and `cacheable_prefix_chars`. `prompt_cache_used` is only set when that prefix reaches 1024 tokens, the provider minimum.
- Unknown or dated model ids (`openai/gpt-4o-2024-08-06`, `gpt-4o`, `claude-sonnet-4-5-20250929`) resolve to their registry entry. Models not in the registry no longer cap `max_tokens` at 8192; a warning is logged once. New `lookup_context_cap()`.
- `compile_prompt_async(user_id=None)` no longer drops directives, `memory_placement`, retrieved objects, compression or plug-ins.
- `[Mem:N]` mentioned inside a kept memory no longer pulls a dropped memory into `ref_map`; compressed middles keep the refs they still contain.
- Redis ref maps are keyed by user and conversation (`user_id=` on `set_/get_ref_map_for_conversation`).
- Procedural memories without structured steps (e.g. `KernelMemoryObject` tier 1) keep their text.
- Reranked memories keep caller tier overrides; `middle_extras` keep their order and rank ahead of reranked memories.
- The classifier step returns copies instead of mutating caller objects, and retiers `KernelMemoryObject`s.
- Lineage `included_ids` lists only memories that survived trimming.
- Async compile falls back to `compress_sync` when `compress_async` fails; `compress_sync` errors are logged instead of raised.
- End placement omits the system message when the core is empty.
- The capability registry returns copies and publishes reloaded maps atomically.

- Removed-profile references: the MCP server defaulted to the removed `research` profile and crashed; MCP and the CLI now use `MembraneProfile.preset()` and offer `minimal`, `coding` and `conversational` (MCP default `coding`).
- Templates: `for` loop variables and `if` conditions are no longer required arguments; each role block is its own message.
- Adapters: LlamaIndex no longer turns a low score into the highest-priority tier and returns `NodeWithScore` from a real `BaseNodePostprocessor`; LangChain `format(compiled_context=...)` and non-numeric priorities no longer raise.
- Conversation manager: hybrid and summarize strategies no longer drop turns, keep earlier summaries, and summarize once per trigger instead of every turn.
- Policies: condition keys are ANDed; unknown or malformed conditions raise instead of matching everything.
- Tokenizers: user registrations take precedence; `gpt-4`/`gpt-4-turbo` use `cl100k_base`; a failed bootstrap is not cached.
- Freshness: relevance 0.0 stays 0.0 and decay is idempotent across compiles.
- Chunking: recursive chunking keeps text before the first heading with document offsets; fixed chunking no longer mixes tokens and characters; oversized sentences and paragraphs are split.
- Embedding cache: the batch path uses Redis, keys can be namespaced by `model_id`, embedding runs outside the lock, and duplicates in a batch are embedded once.
- Classifier features: code detection no longer fires on prose; `user:`/`assistant:` turns are detected.
- Dedup: no transitive merging; the embedding model is cached; hash fallback is logged.
- MCP server: documented as single-client; warns when SSE binds to a non-loopback host; per-call profiles override the default.
- Signal bus and metrics collector: handler errors are logged, and sync handlers work.

- The firewall counts with the profile's tokenizer registry when one is set, so the budget matches `token_count`. `PreFlightContextFirewall(count_fn=...)` accepts any counter.
- Memory text limits are configurable: `MEMBRANE_MEMORY_TEXT_MAX_CHARS` overrides the built-in 200/280/500-character limits, and `0` disables truncation. `metadata["memories_truncated"]` reports how many memories were shortened.
- Policy actions now take effect: `set_budget` gives each memory kind a first-pass share of the middle (unused space goes to the others), `pin_memory_kind` keeps that kind first, `enable` reverses an earlier `disable`, and `set_max_tokens` / `override: {max_tokens}` set the budget, bounded by the model's context cap. **Breaking:** a policy without a `condition` now raises `ValueError` (write `condition: always`), and unknown action keys or memory kinds raise.

## [0.1.1] — 2026-04-11

- Add an in-package `tests/` tree (README-style compile, firewall budget edge, interrupt helpers) and document running `pytest` from the package directory.
- Add GitHub Actions workflow: editable install, **Ruff** on `src`/`tests`, **pytest** on push/PR for `membrane/**`.
- Patch release after test and packaging verification.

[0.1.1]: https://github.com/adeyomilawal/membrane/compare/v0.1.0...v0.1.1
