"""MCP resource payloads for profiles, last compile, and metrics (session state)."""

from __future__ import annotations

import json
from typing import Any

from membrane.profile import PRESET_NAMES


_PROFILE_NAMES = PRESET_NAMES


def list_profile_resource_text() -> str:
    return json.dumps({"presets": list(_PROFILE_NAMES)}, indent=2)


def last_compile_resource_text(last: Any | None) -> str:
    if last is None:
        return json.dumps({"error": "no_compilation_yet"})
    if hasattr(last, "to_debug_dict"):
        return json.dumps(last.to_debug_dict(), default=str, indent=2)
    return json.dumps({"raw": str(last)})


def metrics_resource_text(snapshot: dict[str, Any] | None) -> str:
    return json.dumps(snapshot or {"session": "membrane-mcp", "metrics": "none"}, indent=2)


def resource_uris() -> list[str]:
    return [
        "membrane://profiles",
        "membrane://last_compile",
        "membrane://metrics",
    ]
