"""Model capabilities: context caps and prompt_caching support."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_capabilities_map: Optional[Dict[str, List[str]]] = None
_context_caps_map: Optional[Dict[str, int]] = None


def _default_maps() -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    return (
        {"_default": ["streaming_interrupt", "no_caching"]},
        {"_default": 8192},
    )


def _load_json_from_path(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.error("Invalid capability map at %s: %s", path, e)
        return None


def _load_bundled() -> Optional[dict]:
    """Load JSON shipped beside this module (included in the wheel)."""
    path = Path(__file__).resolve().parent / "llm_capabilities.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Bundled llm_capabilities.json load failed: %s", e)
        return None


def _load_once() -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    global _capabilities_map, _context_caps_map
    if _capabilities_map is not None and _context_caps_map is not None:
        return _capabilities_map, _context_caps_map

    from membrane import config as membrane_config

    data = None
    override = membrane_config.get_llm_capability_map_path()
    if override:
        data = _load_json_from_path(Path(override))
    if data is None:
        data = _load_bundled()
    if data is None:
        _capabilities_map, _context_caps_map = _default_maps()
        return _capabilities_map, _context_caps_map

    caps = data.get("capabilities")
    if not isinstance(caps, dict):
        caps = {"_default": ["streaming_interrupt", "no_caching"]}
    _capabilities_map = {}
    for k, v in caps.items():
        if isinstance(v, list):
            _capabilities_map[k] = [str(x) for x in v]
        else:
            _capabilities_map[k] = ["streaming_interrupt", "no_caching"]

    caps_int = data.get("context_caps")
    if not isinstance(caps_int, dict):
        caps_int = {"_default": 8192}
    _context_caps_map = {}
    for k, v in caps_int.items():
        try:
            _context_caps_map[k] = int(v)
        except (TypeError, ValueError):
            _context_caps_map[k] = 8192
    if "_default" not in _context_caps_map:
        _context_caps_map["_default"] = 8192

    return _capabilities_map, _context_caps_map


def reload_capability_maps() -> None:
    """Force reload from disk (tests and hot-reload)."""
    global _capabilities_map, _context_caps_map
    _capabilities_map = None
    _context_caps_map = None
    _load_once()


def get_capabilities_for_model(model_id: Optional[str]) -> List[str]:
    cap_map, _ = _load_once()
    if model_id and model_id in cap_map:
        return cap_map[model_id]
    return cap_map.get("_default", ["streaming_interrupt", "no_caching"])


def get_context_cap_for_model(model_id: Optional[str]) -> int:
    _, ctx_map = _load_once()
    if model_id and model_id in ctx_map:
        return ctx_map[model_id]
    return ctx_map.get("_default", 8192)


def has_prompt_caching(model_id: Optional[str]) -> bool:
    return "prompt_caching" in get_capabilities_for_model(model_id)


def has_streaming_interrupt(model_id: Optional[str]) -> bool:
    return "streaming_interrupt" in get_capabilities_for_model(model_id)


def get_full_capability_map() -> Dict[str, List[str]]:
    cap_map, _ = _load_once()
    return dict(cap_map)


def get_full_context_caps() -> Dict[str, int]:
    _, ctx_map = _load_once()
    return dict(ctx_map)
