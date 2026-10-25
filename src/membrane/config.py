"""Runtime configuration from environment variables.

Pipeline knobs (PreFlight, LOD, stress TTL, prompt caching) with documented defaults.
No coupling to application-specific config services.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _int_env(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", key, raw, default)
        return default


def _float_env(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", key, raw, default)
        return default


def _bool_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int_env_membrane_legacy(primary: str, legacy: str, default: int) -> int:
    """Prefer MEMBRANE_* env vars; fall back to MEMORY_OS_KERNEL_* for existing deployments."""
    raw = os.environ.get(primary) or os.environ.get(legacy)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", primary, raw, default)
        return default


def _str_env_membrane_legacy(primary: str, legacy: str, default: str) -> str:
    raw = os.environ.get(primary) or os.environ.get(legacy)
    if raw is None:
        return default
    stripped = raw.strip()
    return stripped if stripped else default


def get_preflight_token_threshold() -> int:
    """Raw retrieved memory tokens above this trigger Pre-Flight Analyst compression."""
    return _int_env("MEMORY_PREFLIGHT_TOKEN_THRESHOLD", 8000)


def get_preflight_target_tokens() -> int:
    """Target token size for Pre-Flight Analyst output (sacrifice zone)."""
    return _int_env("MEMORY_PREFLIGHT_TARGET_TOKENS", 500)


def get_preflight_analyst_model() -> str:
    """Model identifier hint for Pre-Flight Analyst (caller passes to LLM adapter)."""
    return (
        os.environ.get("PREFLIGHT_ANALYST_MODEL", "openai/gpt-4o-mini").strip()
        or "openai/gpt-4o-mini"
    )


def get_lod_attention_shift_cosine_threshold() -> float:
    """Cosine distance threshold for LOD attention shift."""
    return _float_env("LOD_ATTENTION_SHIFT_COSINE_THRESHOLD", 0.35)


def get_stress_recent_ttl_seconds() -> int:
    """TTL in seconds for Redis key storing recent stress per user."""
    return _int_env("STRESS_RECENT_TTL_SECONDS", 300)


def get_max_cacheable_prefix_tokens() -> int:
    """Cap on tokens in prompt-caching prefix (glaciator)."""
    return _int_env("MAX_CACHEABLE_PREFIX_TOKENS", 2000)


def get_preflight_cache_ttl_seconds() -> int:
    """TTL for async pre-flight Redis cache entries."""
    return _int_env_membrane_legacy(
        "MEMBRANE_PREFLIGHT_CACHE_TTL_SECONDS",
        "MEMORY_OS_KERNEL_PREFLIGHT_CACHE_TTL_SECONDS",
        60,
    )


def get_stress_redis_key_prefix() -> str:
    """Prefix for Redis keys storing recent stress (default: membrane:stress:)."""
    return _str_env_membrane_legacy(
        "MEMBRANE_STRESS_KEY_PREFIX",
        "MEMORY_OS_KERNEL_STRESS_KEY_PREFIX",
        "membrane:stress:",
    )


def get_preflight_cache_key_prefix() -> str:
    """Prefix for async pre-flight cache keys."""
    return _str_env_membrane_legacy(
        "MEMBRANE_PREFLIGHT_CACHE_PREFIX",
        "MEMORY_OS_KERNEL_PREFLIGHT_CACHE_PREFIX",
        "membrane:preflight:",
    )


def get_ref_map_cache_key_prefix() -> str:
    """Prefix for ref_map Redis keys."""
    return _str_env_membrane_legacy(
        "MEMBRANE_REF_MAP_PREFIX",
        "MEMORY_OS_KERNEL_REF_MAP_PREFIX",
        "membrane:ref_map:",
    )


def get_ref_map_cache_ttl_seconds() -> int:
    """TTL for ref_map cache entries."""
    return _int_env_membrane_legacy(
        "MEMBRANE_REF_MAP_TTL_SECONDS",
        "MEMORY_OS_KERNEL_REF_MAP_TTL_SECONDS",
        600,
    )


def get_default_redis_url() -> str | None:
    """Optional default Redis URL for ref_map when not passed explicitly."""
    url = (
        os.environ.get("MEMBRANE_REDIS_URL")
        or os.environ.get("MEMORY_OS_KERNEL_REDIS_URL")
        or os.environ.get("REDIS_URL")
    )
    return url.strip() if url else None


def get_memory_text_max_chars() -> int | None:
    """``MEMBRANE_MEMORY_TEXT_MAX_CHARS``: cap on each memory's text in the prompt.

    Unset keeps the built-in limits (200 chars for facts, 280 for episodes, 500 for
    procedures); ``0`` disables truncation.
    """
    if os.environ.get("MEMBRANE_MEMORY_TEXT_MAX_CHARS") is None:
        return None
    return max(0, _int_env("MEMBRANE_MEMORY_TEXT_MAX_CHARS", 0))


def get_llm_capability_map_path() -> str | None:
    """Override path to llm_capabilities.json; None = bundled package data."""
    path = os.environ.get("LLM_CAPABILITY_MAP_PATH")
    return path.strip() if path else None
