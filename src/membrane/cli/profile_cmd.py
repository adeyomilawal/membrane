"""`membrane profile` command group."""

from __future__ import annotations

from typing import Any

from membrane.cli.diagnostics import PROFILE_DESCRIPTIONS, print_json, print_lines


def run(args: Any) -> int:
    if args.profile_cmd == "list":
        return run_list(args)
    if args.profile_cmd == "explain":
        return run_explain(args)
    print("unknown profile command")
    return 2


def run_list(args: Any) -> int:
    payload = {
        name: {
            "purpose": desc["purpose"],
        }
        for name, desc in PROFILE_DESCRIPTIONS.items()
    }
    if args.json:
        print_json(payload)
        return 0
    print_lines(["Membrane profiles:"])
    for name, desc in PROFILE_DESCRIPTIONS.items():
        print(f"  {name}: {desc['purpose']}")
    return 0


def run_explain(args: Any) -> int:
    name = args.name.strip().lower()
    desc = PROFILE_DESCRIPTIONS.get(name)
    if desc is None:
        print(f"unknown profile {args.name!r}")
        return 1
    if args.json:
        print_json({"name": name, **desc})
        return 0
    lines = [
        f"Profile: {name}",
        f"  purpose: {desc['purpose']}",
        "  features:",
    ]
    for feature in desc["features"]:
        lines.append(f"    - {feature}")
    deps = desc.get("optional_dependencies") or []
    lines.append("  optional_dependencies:")
    if deps:
        for dep in deps:
            lines.append(f"    - {dep}")
    else:
        lines.append("    - none")
    print_lines(lines)
    return 0
