"""Stress score computation and optional Redis-backed recent stress per user."""

from __future__ import annotations

import logging
from typing import Any, Optional

from membrane import config as os_config

logger = logging.getLogger(__name__)


def compute_stress_from_environment(environment_state: Any) -> Optional[float]:
    """
    Compute scalar stress 0.0–1.0 from environment_state for somatic filtering.

    Returns None if no usable state (caller should omit downstream filters).
    """
    if not environment_state or not isinstance(environment_state, dict):
        return None
    components = []
    if "latency_ms" in environment_state and environment_state["latency_ms"] is not None:
        lat = float(environment_state["latency_ms"])
        components.append(min(1.0, lat / 500.0))
    if "cpu_load" in environment_state and environment_state["cpu_load"] is not None:
        components.append(min(1.0, max(0.0, float(environment_state["cpu_load"]))))
    if "tool_failure_count" in environment_state and environment_state["tool_failure_count"] is not None:
        n = int(environment_state["tool_failure_count"])
        components.append(min(1.0, n / 5.0))
    if "loop_retries" in environment_state and environment_state["loop_retries"] is not None:
        n = int(environment_state["loop_retries"])
        components.append(min(1.0, n / 5.0))
    if not components:
        return None
    return min(1.0, sum(components) / len(components))


async def get_recent_stress_for_user(redis_client: Optional[Any], user_id: str) -> Optional[float]:
    """Read last-written stress for user from Redis."""
    if not redis_client or not user_id:
        return None
    prefix = os_config.get_stress_redis_key_prefix()
    key = prefix + user_id
    try:
        raw = await redis_client.get(key)
        if raw is None:
            return None
        return min(1.0, max(0.0, float(raw)))
    except Exception as e:
        logger.debug("get_recent_stress_for_user %s: %s", user_id, e)
        return None


async def set_recent_stress_for_user(
    redis_client: Optional[Any],
    user_id: str,
    stress: float,
) -> None:
    """Write stress for user; next turn can read via get_recent_stress_for_user."""
    if not redis_client or not user_id:
        return
    stress = min(1.0, max(0.0, float(stress)))
    prefix = os_config.get_stress_redis_key_prefix()
    key = prefix + user_id
    ttl = os_config.get_stress_recent_ttl_seconds()
    try:
        await redis_client.set(key, str(stress), ex=ttl)
    except Exception as e:
        logger.warning("set_recent_stress_for_user %s: %s", user_id, e)
