# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] — 2026-11-11

Default output changes for end placement (history cut, tail layout, system message), so this is a minor release: prompts compiled by 0.3.0 differ from 0.2.x for the same input.

- **Thin before drop** (`thin_before_drop=True`, end placement). Every memory has a keyframe: its densest sentence under `keyframe_tokens` (40), by how rare its words are across the sentences of the memories on offer and of the conversation. Under pressure all keyframes are placed first, then memories get their full text back in priority order. A thinned memory keeps its `[Mem:N]` ref and ends in `…`; `metadata["segments_thinned"]` counts them. `memory_placement="system"` is unchanged.
- **Eviction index** (`eviction_index=True`, `index_share=0.125`). Each dropped history message leaves one keyframe line in the system message after the core (tool results included, tool-call stubs not; one keyframe per 400 tokens of a long message). A line is scored at the cut that drops its message and never changes, so the index only changes when the cut moves and adds no cache breaks. `index_share` is its upper bound, not a reservation: its room comes from what history, the question and one keyframe per memory on offer leave (at least half the share), and `history_share` is history's alone, as in 0.2.x. The size is decided when a cut is first seen and kept until the next cut; a new compiler object can size it differently (one cache miss). A tool result's line names the call that produced it (`tool result (find_orders {"customer": 14}): …`). New metadata: `history_index_lines`, `history_index_tokens`, `core_chars`.
- **Keyframes keep context.** A sentence that opens with a pronoun or has at most three words stays with the sentence before it, so a keyframe can be up to three sentences under the cap. A sentence ending in a question mark is a keyframe only when nothing else fits: a question says what someone wanted to know, rarely the thing itself.
- Recalled passages that do not fit whole are cut to the sentence that matches the question. Every memory is offered its place before any passage; the two header lines that introduce passages can cost one memory line under a very tight budget.
- **Optional ranking by meaning** (`embedder=None`). Pass a function from texts to vectors and the two choices made by shared words also use similarity to the question: which memories keep their full text and which dropped passages come back (reciprocal rank fusion with the lexical order). Keyframes, the index and the history cut do not use it. `membrane.lod.fastembed_embedder()` gives a local one with the `lod` extra; the core stays free of dependencies and network calls, and the default is still word matching. Measured on the offline suite with BAAI/bge-small-en-v1.5 it did not help: ten of eleven scenarios are identical, the reworded spread-fact scenario got worse (history 93.8% to 91.2%), cost is unchanged and each turn takes roughly 40 to 220 ms longer. New metadata: `semantic_ranking`.
- **Price-aware layout.** The capability map has `cache_pricing` per model (write and read multipliers, minimum cacheable tokens, automatic or marked caching) and matches the current Claude families (`claude-opus-*`, `claude-sonnet-*`, `claude-haiku-*`, `claude-fable-*`). Compile reports `metadata["cache_breakpoints"]` and `cache_mode`; `anthropic_request(result)` builds Anthropic Messages API parameters with `cache_control` placed and tool calls converted. The history is always marked when the prefix reaches the model's minimum (an earlier rule that held the marker back before an expected cut saved about 0.1% in simulation and was removed). `prompt_cache_used` now uses the model's own minimum instead of 1,024 for every model. New exports: `anthropic_request`, `get_cache_pricing`.
- **Calibrated budgets.** `compiler.observe(result, actual_prompt_tokens)` keeps a per-model scale between counted tokens and the provider's tokens; the budget is divided by it. Seeds come from the map's `token_scale`: 1.0 OpenAI, 1.11 for Claude Opus 4.5/4.6, Sonnet 4.6 and Haiku 4.5, 1.51 for Claude Opus 4.7 and later, measured on prose with Anthropic's token-count endpoint on 2026-11-11 (`benchmarks/measure_token_scale.py`; code measures 1.27 and 1.65, digits 1.05 and 1.11; Sonnet 4.5 is deprecated and could not be measured, so it takes 1.11 from its generation), so **budgets for Claude models are tighter by default than in 0.2.x**; pass `token_scale=1.0` for the old behaviour. New metadata: `token_scale`, `token_scale_samples`, `tokens_budget_counted`. New export: `get_token_scale`.
- **Capability map checked** against the OpenAI and Anthropic pricing pages on 2026-11-11. Claude Sonnet 5.5 cache reads are 0.05× (the map said 0.1×); Claude Fable 5.2 added (reads 0.025×). `gpt-4-turbo`, `claude-3-5-sonnet`, `claude-3-7-sonnet` and `claude-3-opus` are retired and were removed: they are now unknown models (no caching assumed, no context cap, one warning). Not verified: the 1,024-token minimum for `gpt-4o`, `gpt-4o-mini` and `gpt-5.1` (OpenAI's page no longer states a number for them). OpenAI models newer than `gpt-5.1` are not in the map.
- **Incremental counting.** The joined history is no longer tokenised each compile; the tail is counted in parts that add up exactly with the built-in tokenizer; the core, the index and recall ranking are computed once per cut. Median compile time on an 80-turn chat: 8.6 ms to 0.9 ms, then back up to 1.9 ms with the neighbour-sentence keyframes. A turn that moves the history cut grows the recall and index state instead of rebuilding it (80-turn chat 6.1 to 4.3 ms on those turns, pastes 12.6 to 8.9 ms). A compiler object now keeps small per-conversation caches, so reuse one compiler per model.
- `compile_anonymous` takes `memory_placement="end"` with the compiler's history options, and its system layout reserves message framing and applies `token_scale`.
- Hard suite: three new scenarios (`split-fact`, `spread-fact`, `spread-reworded`: a history fact spread over two or three sentences, the last also asked for in other words), the real `@anysphere/priompt` package run through Node as strategies `priompt-node` and `priompt-node-recent`, `--strategies`, and Membrane pinned to `token_scale=1.0` so every strategy has the same budget. Offline results over eleven scenarios, five seeds: Membrane recall 99.3% at cost 82 / 62; LangChain, Priompt (reimplemented and real) 92% at 98 to 99; LlamaIndex `Memory` 86% at 82 / 80. With `recall_evicted_history=False`: 96.4% at 69 / 47.
- Against the first version of these mechanisms (commit `cb43a15`, mean recall 98.1%, cheap mode 90.9%), history recall per scenario: tools 97% to 100%, memory flood 93% to 100%, two-fact 97% to 100%, paraphrase 72% to 96%, split fact 94% to 96%, spread fact 90% to 94%, spread and reworded 86% to 94%. **Worse:** 3,000-token budget 100% to 97% (the index shrinks under memory pressure); paste scenario cost 86 / 75 to 91 / 80; 80-turn chat cost 75 to 76 at 0.5× reads; the cheap mode costs up to 4 points more in five scenarios; median compile time up about 0.5 ms in most scenarios. Caveats: the question rule is flattered by the suite (every user message opens with a question, then states a fact); the two-sentence scenario barely moved because its second sentence opens with "The".
- Live runs on 2026-11-11, **one seed each**, eleven scenarios, code at `61b8df6` (`benchmarks/results/live_claude-opus-5-5_2026-11-11.json`, `live_claude-haiku-5-5_2026-11-11.json`). `claude-opus-5-5`: Membrane 99.3% accuracy at cost 63 with 2 confident wrong answers; `recall_evicted_history=False` 97.0% at 43 with 9; LangChain 85.1% at 100 with 21; LlamaIndex `Memory` 78.6% at 81 with 40. `claude-haiku-5-5`: Membrane 96.3% at 59 with 7; cheap mode 95.0% at 45 with 12; LangChain 83.6% with 25; real Priompt 82.8% with 26; LlamaIndex `Memory` 76.5% with 42. On the nine scenarios of the earlier Opus run, same seed: Membrane 97.1% to 99.7% (cost 67 to 62), cheap mode 90.5% to 98.0% (46 to 44). **Worse:** cheap mode under a 3,000-token budget 97% to 93%; cheap-mode cost up in four scenarios (two-sentence fact 29 to 42, pastes 38 to 47, 80-turn chat 21 to 27, paraphrase 60 to 65); default cost up in two (24,000-token budget 23 to 26, memory flood 68 to 69); on Haiku the defaults missed tool facts the cheap mode found (92% against 100%) and were below LangChain on two-fact answers (96% against 100%). One seed is 30 to 80 questions per scenario; differences of a few points are noise. Three seeds were planned and not run (spend limit); real Priompt was not run on Opus.
- `benchmarks/hard_suite_live.py`: `--prices` reports what a run spent and `--max-usd` stops it there.
- Earlier live run on `claude-opus-5-5`, **one seed, nine scenarios, code at `cb43a15`** (`benchmarks/results/live_claude-opus-5-5_2026-11-10_after-five-mechanisms.json`): Membrane 97.1% accuracy at cost 67 with 2 confident wrong answers; `recall_evicted_history=False` 90.5% at 46 with 15; LangChain 83.3% at 100 with 19; LlamaIndex `Memory` 78.9% at 83 with 36.

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
