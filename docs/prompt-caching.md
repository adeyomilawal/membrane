# Prompt caching

## Detection

`provider_registry.has_prompt_caching(model_id)` reads capabilities from bundled `llm_capabilities.json` (override path via `LLM_CAPABILITY_MAP_PATH`).

## Splitting

`glaciate(full_system_text)` walks **line by line** until the cacheable prefix fits under `MAX_CACHEABLE_PREFIX_TOKENS` (env). The remainder is the dynamic tail. This avoids splitting mid-line, which would break cache keys unpredictably.

## OpenAI-style APIs

`build_system_messages(model_id, prefix, tail)` returns `[{"role":"system","content": prefix + "\n\n" + tail}]` and a boolean hint when caching is meaningful (non-empty prefix + model supports caching).

## Anthropic

`anthropic_system_blocks(prefix, tail)` returns a list of content blocks: the first text block includes `cache_control: {type: ephemeral}` when the prefix is non-empty; the tail is a separate block without cache control.

## Ollama / local models

Models without `prompt_caching` in the capability map skip glaciated splitting; the full system string is sent as one message.
