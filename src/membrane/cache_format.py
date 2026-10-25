"""Provider-oriented system message formatting for cacheable prefix + dynamic tail."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from membrane.provider_registry import has_prompt_caching

logger = logging.getLogger(__name__)


@dataclass
class CacheableBlock:
    """Immutable cacheable prefix + dynamic tail for prompt caching."""

    cacheable_prefix: str
    dynamic_tail: str
    use_caching: bool


def make_cacheable_block(
    model_id: str | None,
    cacheable_prefix: str,
    dynamic_tail: str,
) -> CacheableBlock:
    use = bool(
        model_id
        and has_prompt_caching(model_id)
        and (cacheable_prefix or "").strip()
    )
    return CacheableBlock(
        cacheable_prefix=cacheable_prefix or "",
        dynamic_tail=dynamic_tail or "",
        use_caching=use,
    )


def build_system_messages(
    model_id: str | None,
    cacheable_prefix: str,
    dynamic_tail: str,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Build system message list with cacheable content first (prefix then tail).

    Returns:
        (messages, use_cache_hint)
    """
    full_content = assemble_system_content(cacheable_prefix, dynamic_tail)

    if not full_content.strip():
        return ([], False)

    use_caching = bool(
        model_id
        and has_prompt_caching(model_id)
        and cacheable_prefix.strip()
    )

    system_msg: dict[str, Any] = {
        "role": "system",
        "content": full_content,
    }
    return ([system_msg], use_caching)


def assemble_system_content(prefix: str, tail: str) -> str:
    if not prefix.strip():
        return tail
    if not tail.strip():
        return prefix
    return prefix.rstrip() + "\n\n" + tail.lstrip()


def anthropic_system_blocks(cacheable_prefix: str, dynamic_tail: str) -> list[dict[str, Any]]:
    """
    Anthropic Messages API: system as list of blocks; first block may use cache_control.

    Per Anthropic docs: cache_control: {"type": "ephemeral"} on the block to cache.
    """
    blocks: list[dict[str, Any]] = []

    if cacheable_prefix.strip():
        blocks.append({
            "type": "text",
            "text": cacheable_prefix.rstrip(),
            "cache_control": {"type": "ephemeral"},
        })

    if dynamic_tail.strip():
        blocks.append({
            "type": "text",
            "text": dynamic_tail.lstrip(),
        })

    return blocks if blocks else [{"type": "text", "text": ""}]


def is_anthropic_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    return str(model_id).strip().lower().startswith("anthropic/")
