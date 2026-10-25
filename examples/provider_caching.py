"""Glaciate + build_system_messages for cache-friendly system prompt."""

from membrane import build_system_messages, glaciate, has_prompt_caching


def main() -> None:
    model = "anthropic/claude-3-7-sonnet"
    long_system = "Rule line A\n" * 100 + "\nVolatile tail for this turn.\n"
    g = glaciate(long_system, max_cacheable_tokens=500)
    print("supports_caching", has_prompt_caching(model))
    msgs, hint = build_system_messages(model, g.cacheable_prefix, g.dynamic_tail)
    print("messages", len(msgs), "cache_hint", hint)


if __name__ == "__main__":
    main()
