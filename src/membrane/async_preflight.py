"""Optional async pre-flight cache for while-typing compression."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Awaitable, Callable, Optional

from membrane import config as os_config
from membrane.firewall import count_tokens

logger = logging.getLogger(__name__)


def get_preflight_cache_key_prefix() -> str:
    """Redis key prefix for pre-flight entries (configurable via env)."""
    return os_config.get_preflight_cache_key_prefix()


def get_preflight_cache_ttl_default() -> int:
    """Default TTL seconds for pre-flight cache entries."""
    return os_config.get_preflight_cache_ttl_seconds()


def _cache_key(conversation_id: str, user_msg: str) -> str:
    payload = (conversation_id + "\0" + (user_msg or "")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _full_key(conversation_id: str, user_msg: str) -> str:
    prefix = os_config.get_preflight_cache_key_prefix()
    return f"{prefix}{conversation_id}:{_cache_key(conversation_id, user_msg)}"


async def get_preflight_cache(
    redis_client: Any,
    conversation_id: str,
    user_msg: str,
) -> Optional[str]:
    if not redis_client or not conversation_id:
        return None
    try:
        key = _full_key(conversation_id, user_msg or "")
        val = await redis_client.get(key)
        if val is not None and isinstance(val, (str, bytes)):
            return val.decode("utf-8") if isinstance(val, bytes) else val
    except Exception as e:
        logger.debug("Pre-flight cache get failed: %s", e)
    return None


async def set_preflight_cache(
    redis_client: Any,
    conversation_id: str,
    user_msg: str,
    compressed: str,
    ttl_seconds: Optional[int] = None,
) -> None:
    if not redis_client or not conversation_id or not compressed:
        return
    ttl = ttl_seconds if ttl_seconds is not None else os_config.get_preflight_cache_ttl_seconds()
    try:
        key = _full_key(conversation_id, user_msg or "")
        await redis_client.set(key, compressed, ex=ttl)
    except Exception as e:
        logger.debug("Pre-flight cache set failed: %s", e)


RetrieveAndSerialize = Callable[[str, str], Awaitable[str]]
CompressMiddle = Callable[[str, int, str], Awaitable[str]]


async def run_async_preflight(
    user_id: str,
    conversation_id: str,
    partial_message: str,
    *,
    redis_client: Any,
    retrieve_raw_middle: RetrieveAndSerialize,
    compress_middle: CompressMiddle,
    threshold: Optional[int] = None,
    target_tokens: Optional[int] = None,
) -> Optional[str]:
    """
    Build raw middle from retrieval, compress if over threshold, store in Redis.

    ``retrieve_raw_middle(user_id, query)`` must return serialized sacrifice-zone text.
    ``compress_middle(raw_text, target_tokens, user_message)`` returns compressed text.
    """
    if not partial_message or not partial_message.strip():
        return None
    if not user_id or not conversation_id:
        return None
    try:
        raw_middle_text = await retrieve_raw_middle(user_id, partial_message.strip())
        if not raw_middle_text or not raw_middle_text.strip():
            return None
        th = threshold if threshold is not None else os_config.get_preflight_token_threshold()
        if count_tokens(raw_middle_text) <= th:
            return None
        tgt = target_tokens if target_tokens is not None else os_config.get_preflight_target_tokens()
        compressed = await compress_middle(raw_middle_text, tgt, partial_message)
        if compressed and redis_client:
            await set_preflight_cache(
                redis_client,
                conversation_id,
                partial_message,
                compressed,
            )
        return compressed
    except Exception as e:
        logger.warning("Async pre-flight failed: %s", e, exc_info=True)
    return None
