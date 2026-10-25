"""`membrane inspect` command."""

from __future__ import annotations

from typing import Any

from membrane.cli.diagnostics import (
    build_compiler,
    load_memories,
    print_json,
    print_lines,
    resolve_profile_name,
    result_summary,
)


def run(args: Any) -> int:
    """Compile a prompt and print token/layout diagnostics."""
    try:
        memories = load_memories(args.memory, args.memory_json)
        profile = resolve_profile_name(args.profile)
        compiler = build_compiler(
            max_tokens=args.max_tokens,
            model=args.model,
            core=args.core,
        )
        result = compiler.compile(
            memories=memories,
            user_message=args.user_message,
            profile=profile,
            use_prompt_caching_split=not args.no_prompt_cache,
        )
    except Exception as exc:
        print(f"membrane inspect failed: {exc}")
        return 1

    summary = result_summary(result)
    summary["profile"] = args.profile or "default"
    summary["memory_count"] = len(memories)
    summary["model"] = args.model

    if args.json:
        print_json(summary)
        return 0

    lines = [
        "Membrane inspect",
        f"  model: {args.model or 'default'}",
        f"  profile: {args.profile or 'default'}",
        f"  memories: {len(memories)}",
        f"  token_count: {summary['token_count']}",
        f"  tokens_budget: {summary['tokens_budget']}",
        f"  tokens_trimmed: {summary['tokens_trimmed']}",
        f"  segments: {summary['segments_kept']} kept / {summary['segments_offered']} offered",
        f"  ref_map_size: {len(result.ref_map)}",
        f"  message_roles: {', '.join(str(r) for r in summary['message_roles'])}",
    ]
    if result.system_alerts:
        lines.append("  alerts:")
        for alert in result.system_alerts:
            lines.append(f"    - {alert}")
    print_lines(lines)
    return 0
