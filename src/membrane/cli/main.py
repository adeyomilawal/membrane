"""argparse entrypoint for `membrane` CLI."""

from __future__ import annotations

import argparse
import sys


def _version_string() -> str:
    try:
        from importlib.metadata import version

        return version("membrane-aol")
    except Exception:
        return "0.0.0"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="membrane",
        description="Membrane CLI for compiler diagnostics.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_version_string()}",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_compile_diag_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--user-message", required=True, help="Current user message to compile")
        p.add_argument(
            "--memory",
            action="append",
            default=[],
            help="Memory as 'id|tier|content' or plain content; repeatable",
        )
        p.add_argument(
            "--memory-json",
            default=None,
            help="Path to JSON list of memory strings or objects",
        )
        p.add_argument("--core", default="You are a helpful assistant.", help="Core system content")
        p.add_argument("--model", default="gpt-4o-mini", help="Model id used for capability lookup")
        p.add_argument("--max-tokens", type=int, default=8192, help="Hard context token budget")
        p.add_argument(
            "--profile",
            default=None,
            help="Profile: minimal, coding, conversational",
        )
        p.add_argument(
            "--no-prompt-cache",
            action="store_true",
            help="Disable provider prompt-cache split",
        )
        p.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    p_inspect = sub.add_parser("inspect", help="Compile a prompt and show layout/token diagnostics")
    add_compile_diag_args(p_inspect)

    p_quote = sub.add_parser(
        "quote",
        help="Estimate input tokens, trimming, and optional input cost",
    )
    add_compile_diag_args(p_quote)
    p_quote.add_argument(
        "--price-per-1k-input-tokens",
        type=float,
        default=None,
        help="Optional input-token price used for local cost estimate",
    )
    p_quote.add_argument(
        "--target-token-budget",
        type=int,
        default=None,
        help="Optional target budget for guidance without changing compile max_tokens",
    )

    p_profile = sub.add_parser("profile", help="Profile diagnostics")
    profile_sub = p_profile.add_subparsers(dest="profile_cmd", required=True)
    p_profile_list = profile_sub.add_parser("list", help="List built-in profiles")
    p_profile_list.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p_profile_explain = profile_sub.add_parser("explain", help="Explain a built-in profile")
    p_profile_explain.add_argument("name", help="Profile name")
    p_profile_explain.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )

    args = parser.parse_args(argv)

    if args.cmd == "inspect":
        from membrane.cli import inspect as inspect_mod

        return inspect_mod.run(args)
    if args.cmd == "quote":
        from membrane.cli import quote as quote_mod

        return quote_mod.run(args)
    if args.cmd == "profile":
        from membrane.cli import profile_cmd

        return profile_cmd.run(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
