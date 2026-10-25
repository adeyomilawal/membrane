"""Conversation-scoped ref_map / ref_map_type persistence for [Mem:N] resolution."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from membrane import config as os_config

logger = logging.getLogger(__name__)


def _serialize_ref_maps(
    ref_map: Dict[int, str],
    ref_map_type: Optional[Dict[int, str]],
) -> str:
    ref_map_str = {str(k): v for k, v in ref_map.items()}
    ref_map_type_str = {str(k): v for k, v in (ref_map_type or {}).items()}
    return json.dumps({"ref_map": ref_map_str, "ref_map_type": ref_map_type_str})


def _deserialize_ref_maps(data: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    try:
        obj = json.loads(data)
        ref_map = {int(k): v for k, v in (obj.get("ref_map") or {}).items()}
        ref_map_type = {int(k): v for k, v in (obj.get("ref_map_type") or {}).items()}
        return ref_map, ref_map_type
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("ref_map deserialize failed: %s", e)
        return {}, {}


def _key_prefix() -> str:
    return os_config.get_ref_map_cache_key_prefix()


@asynccontextmanager
async def _redis_connection(
    redis_client: Optional[Any],
    redis_url: Optional[str],
) -> AsyncIterator[Optional[Any]]:
    if redis_client is not None:
        yield redis_client
        return
    url = redis_url if redis_url else os_config.get_default_redis_url()
    if not url:
        logger.debug("ref_map: no redis_client and no redis URL; skipping")
        yield None
        return
    try:
        import redis.asyncio as redis_mod
    except ImportError:
        logger.warning("ref_map: redis package not installed; skipping")
        yield None
        return
    client = redis_mod.from_url(url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


async def set_ref_map_for_conversation(
    conversation_id: str,
    ref_map: Dict[int, str],
    ref_map_type: Optional[Dict[int, str]] = None,
    *,
    ttl_seconds: Optional[int] = None,
    redis_client: Optional[Any] = None,
    redis_url: Optional[str] = None,
) -> None:
    """
    Store ref_map and ref_map_type for tools that resolve [Mem:N] to global ids.

    Provide either ``redis_client`` or rely on ``MEMORY_OS_KERNEL_REDIS_URL`` / ``REDIS_URL``.
    """
    if not conversation_id or not isinstance(ref_map, dict):
        return
    ttl = ttl_seconds if ttl_seconds is not None else os_config.get_ref_map_cache_ttl_seconds()
    key = f"{_key_prefix()}{conversation_id}"
    val = _serialize_ref_maps(ref_map, ref_map_type)
    async with _redis_connection(redis_client, redis_url) as client:
        if client is None:
            return
        try:
            await client.set(key, val, ex=ttl)
        except Exception as e:
            logger.warning("ref_map set failed: %s", e)


async def get_ref_map_for_conversation(
    conversation_id: str,
    *,
    redis_client: Optional[Any] = None,
    redis_url: Optional[str] = None,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Return (ref_map, ref_map_type) or empty dicts on miss / error."""
    if not conversation_id:
        return {}, {}
    key = f"{_key_prefix()}{conversation_id}"
    async with _redis_connection(redis_client, redis_url) as client:
        if client is None:
            return {}, {}
        try:
            data = await client.get(key)
            if data is None:
                return {}, {}
            return _deserialize_ref_maps(data)
        except Exception as e:
            logger.warning("ref_map get failed: %s", e)
            return {}, {}
