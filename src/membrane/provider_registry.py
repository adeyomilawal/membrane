"""Model capabilities: context caps, prompt_caching support, cache prices, token scale."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

_capabilities_map: dict[str, list[str]] | None = None
_context_caps_map: dict[str, int] | None = None
_cache_pricing_map: dict[str, CachePricing] = {}

CACHE_MODE_AUTOMATIC = "automatic"  # the provider caches shared prefixes by itself
CACHE_MODE_MARKED = "marked"  # the request must mark where a cached prefix ends


@dataclass(frozen=True)
class CachePricing:
    """How a model's prompt cache is priced, relative to its plain input price.

    ``write`` is the multiplier for tokens written to the cache, ``read`` for tokens read
    from it. ``min_tokens`` is the shortest prefix the provider caches, in the provider's
    own tokens. ``mode`` is ``"automatic"`` or ``"marked"``.
    """

    mode: str = CACHE_MODE_AUTOMATIC
    write: float = 1.0
    read: float = 0.5
    min_tokens: int = 1024

    @property
    def marked(self) -> bool:
        return self.mode == CACHE_MODE_MARKED


def _parse_pricing(raw: Any) -> dict[str, CachePricing]:
    out: dict[str, CachePricing] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        try:
            out[key] = CachePricing(
                mode=str(value.get("mode", CACHE_MODE_AUTOMATIC)),
                write=float(value.get("write", 1.0)),
                read=float(value.get("read", 0.5)),
                min_tokens=int(value.get("min_tokens", 1024)),
            )
        except (TypeError, ValueError):
            logger.error("Invalid cache_pricing entry for %r", key)
    return out


def _default_maps() -> tuple[dict[str, list[str]], dict[str, int]]:
    return (
        {"_default": ["streaming_interrupt", "no_caching"]},
        {"_default": 8192},
    )


def _load_json_from_path(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.error("Invalid capability map at %s: %s", path, e)
        return None


def _load_bundled() -> dict | None:
    """Load JSON shipped beside this module (included in the wheel)."""
    path = Path(__file__).resolve().parent / "llm_capabilities.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Bundled llm_capabilities.json load failed: %s", e)
        return None


def _load_once() -> tuple[dict[str, list[str]], dict[str, int]]:
    global _capabilities_map, _context_caps_map, _cache_pricing_map
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
        _cache_pricing_map = {}
        return _capabilities_map, _context_caps_map

    # Build locally, then publish both maps at once so readers never see a half-built map.

    caps = data.get("capabilities")
    if not isinstance(caps, dict):
        caps = {"_default": ["streaming_interrupt", "no_caching"]}
    cap_map: dict[str, list[str]] = {}
    for k, v in caps.items():
        if isinstance(v, list):
            cap_map[k] = [str(x) for x in v]
        else:
            cap_map[k] = ["streaming_interrupt", "no_caching"]

    caps_int = data.get("context_caps")
    if not isinstance(caps_int, dict):
        caps_int = {"_default": 8192}
    ctx_map: dict[str, int] = {}
    for k, v in caps_int.items():
        try:
            ctx_map[k] = int(v)
        except (TypeError, ValueError):
            ctx_map[k] = 8192
    if "_default" not in ctx_map:
        ctx_map["_default"] = 8192

    _cache_pricing_map = _parse_pricing(data.get("cache_pricing"))
    _capabilities_map, _context_caps_map = cap_map, ctx_map
    return cap_map, ctx_map


def reload_capability_maps() -> None:
    """Force reload from disk (tests and hot-reload)."""
    global _capabilities_map, _context_caps_map
    _capabilities_map = None
    _context_caps_map = None
    _load_once()


_DATE_SUFFIX = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2}|latest)$")
_warned_unknown: set = set()


def _normalize(model_id: str) -> str:
    """Lowercase, dots as dashes, no date or ``latest`` suffix."""
    return _DATE_SUFFIX.sub("", model_id.strip().lower().replace(".", "-"))


def _resolve_key(model_id: str | None, keys: Sequence[str], warn: bool = True) -> str | None:
    """Map a model id to a registry key: exact, normalized, then longest prefix.

    Matching ignores the provider prefix when ``model_id`` has none, so ``gpt-4o``
    and ``openai/gpt-4o-2024-08-06`` both resolve to ``openai/gpt-4o``.
    """
    if not model_id:
        return None
    if model_id in keys:
        return model_id
    target = _normalize(model_id)
    bare_target = target.split("/", 1)[-1]
    best: str | None = None
    best_len = 0
    for key in keys:
        if key == "_default":
            continue
        norm = _normalize(key)
        candidate = norm if "/" in target else norm.split("/", 1)[-1]
        subject = target if "/" in target else bare_target
        if (
            subject == candidate or subject.startswith(candidate + "-")
        ) and len(candidate) > best_len:
            best, best_len = key, len(candidate)
    if best is None and warn and model_id not in _warned_unknown:
        _warned_unknown.add(model_id)
        logger.warning(
            "Model %r is not in the capability map; using defaults (no prompt caching, "
            "no context cap). Add it via LLM_CAPABILITY_MAP_PATH.",
            model_id,
        )
    return best


def get_capabilities_for_model(model_id: str | None) -> list[str]:
    cap_map, _ = _load_once()
    key = _resolve_key(model_id, list(cap_map))
    if key is not None:
        return list(cap_map[key])
    return list(cap_map.get("_default", ["streaming_interrupt", "no_caching"]))


def lookup_context_cap(model_id: str | None) -> int | None:
    """Context cap for a known model, or ``None`` when the model is not in the map."""
    _, ctx_map = _load_once()
    key = _resolve_key(model_id, list(ctx_map))
    return ctx_map[key] if key is not None else None


def get_context_cap_for_model(model_id: str | None) -> int:
    cap = lookup_context_cap(model_id)
    if cap is not None:
        return cap
    _, ctx_map = _load_once()
    return ctx_map.get("_default", 8192)


def has_prompt_caching(model_id: str | None) -> bool:
    return "prompt_caching" in get_capabilities_for_model(model_id)


def get_cache_pricing(model_id: str | None) -> CachePricing | None:
    """Cache prices for a model, or ``None`` when the model has no prompt caching.

    A caching model without its own entry gets the map's ``_default`` entry (automatic
    caching, no write premium, reads at half price, 1,024-token minimum).
    """
    if not has_prompt_caching(model_id):
        return None
    key = _resolve_key(model_id, list(_cache_pricing_map), warn=False)
    if key is not None:
        return _cache_pricing_map[key]
    return _cache_pricing_map.get("_default", CachePricing())


def has_streaming_interrupt(model_id: str | None) -> bool:
    return "streaming_interrupt" in get_capabilities_for_model(model_id)


def get_full_capability_map() -> dict[str, list[str]]:
    cap_map, _ = _load_once()
    return {k: list(v) for k, v in cap_map.items()}


def get_full_context_caps() -> dict[str, int]:
    _, ctx_map = _load_once()
    return dict(ctx_map)
