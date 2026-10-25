# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — Unreleased

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
