"""Context Glaciator: split system text into cacheable prefix and dynamic tail.

Split is at line boundaries only; never cuts mid-line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from membrane import config as os_config
from membrane.firewall import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class GlaciateResult:
    """Result of glaciate(): cacheable prefix and dynamic tail."""

    cacheable_prefix: str
    dynamic_tail: str
    prefix_tokens: int
    tail_tokens: int
    use_caching: bool


def glaciate(
    full_system_text: Optional[str],
    max_cacheable_tokens: Optional[int] = None,
) -> GlaciateResult:
    """
    Split full system text into cacheable prefix and dynamic tail.

    Prefix is the first N tokens (N = max_cacheable_tokens), accumulated line by line.
    """
    if max_cacheable_tokens is None:
        max_cacheable_tokens = os_config.get_max_cacheable_prefix_tokens()

    if not full_system_text or not full_system_text.strip():
        return GlaciateResult(
            cacheable_prefix="",
            dynamic_tail=full_system_text or "",
            prefix_tokens=0,
            tail_tokens=0,
            use_caching=False,
        )

    total = count_tokens(full_system_text)
    if total <= max_cacheable_tokens:
        return GlaciateResult(
            cacheable_prefix=full_system_text,
            dynamic_tail="",
            prefix_tokens=total,
            tail_tokens=0,
            use_caching=total >= 256,
        )

    lines = full_system_text.splitlines(keepends=True)
    if not lines:
        return GlaciateResult(
            cacheable_prefix="",
            dynamic_tail=full_system_text,
            prefix_tokens=0,
            tail_tokens=total,
            use_caching=False,
        )

    prefix_lines: list[str] = []
    prefix_tok = 0

    for line in lines:
        line_tok = count_tokens(line)
        if prefix_tok + line_tok <= max_cacheable_tokens:
            prefix_lines.append(line)
            prefix_tok += line_tok
        else:
            break

    cacheable_prefix = "".join(prefix_lines).rstrip()
    dynamic_tail = full_system_text[len(cacheable_prefix) :].lstrip()
    tail_tokens = count_tokens(dynamic_tail)

    return GlaciateResult(
        cacheable_prefix=cacheable_prefix,
        dynamic_tail=dynamic_tail,
        prefix_tokens=prefix_tok,
        tail_tokens=tail_tokens,
        use_caching=prefix_tok >= 256,
    )
