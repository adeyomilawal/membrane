"""Shared helpers for Membrane CLI diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from membrane import ContextCompiler, MemoryItem
from membrane.profile import MembraneProfile


PROFILE_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "minimal": {
        "purpose": "Phase 0 compiler behavior with no optional intelligence modules.",
        "features": ["layout", "firewall", "prompt caching", "ref_map"],
        "optional_dependencies": [],
    },
    "coding": {
        "purpose": "Coding-oriented context shaping for procedural memory and fresh code facts.",
        "features": ["classifier", "deduplication", "freshness", "policy budget shift", "profiler"],
        "optional_dependencies": [],
    },
    "conversational": {
        "purpose": "Chat-oriented context shaping with short recency decay.",
        "features": ["classifier", "deduplication", "freshness", "profiler"],
        "optional_dependencies": [],
    },
}


def parse_memory_arg(raw: str, index: int) -> MemoryItem:
    """Parse ``id|tier|content`` or plain content into a MemoryItem."""
    parts = raw.split("|", 2)
    if len(parts) == 3:
        mid, tier_raw, content = parts
        try:
            tier = int(tier_raw)
        except ValueError as exc:
            raise ValueError(f"memory tier must be an integer in {raw!r}") from exc
        if tier not in (1, 2, 3):
            raise ValueError(f"memory tier must be 1, 2, or 3 in {raw!r}")
        memory_type = "procedural" if tier == 1 else "semantic" if tier == 2 else "episodic"
        return MemoryItem(
            id=mid or f"mem-{index}",
            content=content,
            priority=tier,
            memory_type=memory_type,
        )
    return MemoryItem(id=f"mem-{index}", content=raw, priority=2, memory_type="semantic")


def load_memories(memory_args: Sequence[str] | None, memory_json: str | None) -> list[MemoryItem]:
    memories: list[MemoryItem] = []
    for i, raw in enumerate(memory_args or [], start=1):
        memories.append(parse_memory_arg(raw, i))

    if memory_json:
        data = json.loads(Path(memory_json).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("memory JSON must contain a list")
        for i, item in enumerate(data, start=len(memories) + 1):
            if isinstance(item, str):
                memories.append(parse_memory_arg(item, i))
                continue
            if not isinstance(item, dict):
                raise ValueError("memory JSON items must be strings or objects")
            content = item.get("content", item.get("payload", ""))
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True, default=str)
            tier = int(item.get("priority", item.get("semantic_tier", 2)))
            if tier not in (1, 2, 3):
                raise ValueError("memory priority/semantic_tier must be 1, 2, or 3")
            default_type = "procedural" if tier == 1 else "semantic" if tier == 2 else "episodic"
            memory_type = str(item.get("memory_type") or default_type)
            memories.append(
                MemoryItem(
                    id=str(item.get("id", f"mem-{i}")),
                    content=content,
                    priority=tier,
                    memory_type=memory_type,  # type: ignore[arg-type]
                )
            )
    return memories


def resolve_profile_name(name: str | None) -> Any:
    if not name:
        return None
    return MembraneProfile.preset(name)


def build_compiler(*, max_tokens: int, model: str | None, core: str) -> ContextCompiler:
    return ContextCompiler(max_tokens=max_tokens, model_id=model, core_content=core)


def result_summary(result: Any) -> dict[str, Any]:
    md = result.metadata or {}
    messages = result.messages or []
    return {
        "token_count": int(md.get("token_count", 0)),
        "tokens_budget": md.get("tokens_budget"),
        "tokens_trimmed": int(md.get("tokens_trimmed", 0)),
        "segments_offered": int(md.get("segments_offered", 0)),
        "segments_kept": int(md.get("segments_kept", 0)),
        "ref_map": result.ref_map,
        "ref_map_type": result.ref_map_type,
        "system_alerts": result.system_alerts,
        "message_roles": [m.get("role") for m in messages],
        "metadata": md,
    }


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)
