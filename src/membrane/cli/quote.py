"""`membrane quote` command."""

from __future__ import annotations

from typing import Any

from membrane.cli.diagnostics import (
    build_compiler,
    load_memories,
    print_json,
    print_lines,
    resolve_profile_name,
)


def run(args: Any) -> int:
    """Run the compile pipeline and print a token/cost quote."""
    try:
        memories = load_memories(args.memory, args.memory_json)
        profile = resolve_profile_name(args.profile)
        compiler = build_compiler(
            max_tokens=args.max_tokens,
            model=args.model,
            core=args.core,
        )
        quote = compiler.estimate(
            memories=memories,
            user_message=args.user_message,
            profile=profile,
            price_per_1k_input_tokens=args.price_per_1k_input_tokens,
            target_token_budget=args.target_token_budget,
            use_prompt_caching_split=not args.no_prompt_cache,
        )
    except Exception as exc:
        print(f"membrane quote failed: {exc}")
        return 1

    payload = {
        "model": args.model,
        "profile": args.profile or "default",
        "memory_count": len(memories),
        "total_tokens": quote.total_tokens,
        "tokens_budget": quote.tokens_budget,
        "tokens_trimmed": quote.tokens_trimmed,
        "trimmed_count": quote.trimmed_count,
        "segments_offered": quote.segments_offered,
        "segments_kept": quote.segments_kept,
        "estimated_cost_usd": quote.estimated_cost_usd,
        "trimming_strategies": quote.trimming_strategies,
    }

    if args.json:
        print_json(payload)
        return 0

    lines = [
        "Membrane quote",
        f"  model: {args.model or 'default'}",
        f"  profile: {args.profile or 'default'}",
        f"  memories: {len(memories)}",
        f"  total_tokens: {quote.total_tokens}",
        f"  tokens_budget: {quote.tokens_budget}",
        f"  tokens_trimmed: {quote.tokens_trimmed}",
        f"  trimmed_count: {quote.trimmed_count}",
    ]
    if quote.estimated_cost_usd is not None:
        lines.append(f"  estimated_input_cost_usd: {quote.estimated_cost_usd:.8f}")
    lines.append("  guidance:")
    for item in quote.trimming_strategies:
        lines.append(f"    - {item}")
    print_lines(lines)
    return 0
