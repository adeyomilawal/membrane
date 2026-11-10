"""Provider-oriented system message formatting for cacheable prefix + dynamic tail."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
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


def _anthropic_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Content blocks for one OpenAI-style chat message."""
    content = message.get("content")
    if message.get("role") == "tool":
        return [
            {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id"),
                "content": content if isinstance(content, list) else str(content or ""),
            }
        ]
    blocks: list[dict[str, Any]] = []
    if isinstance(content, list):
        blocks = [
            dict(b) if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in content
        ]
    elif content:
        blocks = [{"type": "text", "text": str(content)}]
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id"),
                "name": function.get("name"),
                "input": arguments or {},
            }
        )
    return blocks or [{"type": "text", "text": ""}]


def to_anthropic_params(
    messages: list[dict[str, Any]],
    cache_breakpoints: Sequence[dict[str, Any]] = (),
    *,
    ttl: str | None = None,
) -> dict[str, Any]:
    """OpenAI-style chat messages as Anthropic Messages API parameters, cache markers placed.

    Returns ``{"system": [...], "messages": [...]}`` (``system`` only when there is a system
    message) to pass to ``client.messages.create`` along with ``model`` and ``max_tokens``.
    Tool calls become ``tool_use`` blocks, tool results become ``tool_result`` blocks, and
    the results of one assistant turn share one user message.

    Each breakpoint is ``{"message_index": i}`` with an optional ``"chars"``: the marker goes
    on the last block of message ``i``, or, with ``chars``, after the first ``chars``
    characters of its text, which is split into two blocks there. Pass
    ``result.metadata["cache_breakpoints"]``, or use :func:`anthropic_request`. ``ttl`` is
    ``"1h"`` for the one-hour cache; the default is the provider's (five minutes).
    """
    marker: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        marker["ttl"] = ttl
    split_at: dict[int, int | None] = {}
    for point in cache_breakpoints:
        split_at[int(point["message_index"])] = point.get("chars")

    system: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    previous_role: str | None = None
    for index, message in enumerate(messages):
        role = message.get("role")
        marked = index in split_at
        if role == "system":
            text = message.get("content")
            text = text if isinstance(text, str) else str(text or "")
            chars = split_at.get(index)
            if marked and chars is not None and 0 < chars < len(text) and text[chars:].strip():
                system.append({"type": "text", "text": text[:chars], "cache_control": dict(marker)})
                system.append({"type": "text", "text": text[chars:]})
            elif text:
                block: dict[str, Any] = {"type": "text", "text": text}
                if marked:
                    block["cache_control"] = dict(marker)
                system.append(block)
            continue
        blocks = _anthropic_blocks(message)
        if marked:
            blocks[-1] = {**blocks[-1], "cache_control": dict(marker)}
        if role == "tool" and previous_role == "tool" and out:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": "user" if role == "tool" else role, "content": blocks})
        previous_role = role
    params: dict[str, Any] = {"messages": out}
    if system:
        params = {"system": system, "messages": out}
    return params


def anthropic_request(result: Any, *, ttl: str | None = None) -> dict[str, Any]:
    """Anthropic Messages API parameters for a :class:`~membrane.compiler.CompileResult`.

    Places ``cache_control`` where ``result.metadata["cache_breakpoints"]`` says a cached
    prefix ends: after the core system prompt, and after the conversation history when the
    compiler expects that prefix to be read before the next history cut. No SDK is imported
    and nothing is sent; add ``model`` and ``max_tokens`` and call the API yourself.
    """
    return to_anthropic_params(
        result.messages, result.metadata.get("cache_breakpoints") or (), ttl=ttl
    )
