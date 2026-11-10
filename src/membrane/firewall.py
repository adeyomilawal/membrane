"""PreFlight context firewall: exact token counting and priority-based trimming.

Core and end (current turn + user message) are never trimmed; middle is trimmed
by priority tier (procedural first, max 3, then semantic/episodic).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, Literal

import membrane.config as os_config

logger = logging.getLogger(__name__)

# Use Literal[False] only (not bool) so failure does not widen to True | False.
_tiktoken_encoding: Any | Literal[False] | None = None


def _get_encoding() -> Any | Literal[False]:
    """Lazy-load tiktoken encoding (cl100k_base)."""
    global _tiktoken_encoding
    if _tiktoken_encoding is None:
        try:
            import tiktoken

            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning("tiktoken not available (%s); using char-based estimate", e)
            _tiktoken_encoding = False
    assert _tiktoken_encoding is not None
    return _tiktoken_encoding


# Chat history is re-counted on every compile; short texts are cached by content.
_COUNT_CACHE_MAX_CHARS = 8_000


@lru_cache(maxsize=4096)
def _count_exact_cached(text: str) -> int:
    enc = _get_encoding()
    return len(enc.encode(text))  # type: ignore[union-attr]


def count_tokens(text: str) -> int:
    """Return exact token count for text using tiktoken; fallback ~4 chars/token."""
    if not text:
        return 0
    enc = _get_encoding()
    if enc is False:
        return max(1, len(text) // 4)
    if len(text) <= _COUNT_CACHE_MAX_CHARS:
        return _count_exact_cached(text)
    return len(enc.encode(text))


TIER_PROCEDURAL = 1
TIER_SEMANTIC = 2
TIER_EPISODIC = 3

_MEM_REF_PATTERN = re.compile(r"\[Mem:(\d+)\]")
_LEADING_MEM_REF = re.compile(r"^\s*\[Mem:(\d+)\]")


def _extract_mem_refs(segment: str) -> list[tuple[int, int]]:
    """Return list of (local_index, start_pos) for [Mem:N] in segment."""
    out = []
    for m in _MEM_REF_PATTERN.finditer(segment):
        try:
            idx = int(m.group(1))
            out.append((idx, m.start()))
        except ValueError:
            continue
    return out


def _segment_refs(segment: str) -> list[int]:
    """Refs a segment owns: its leading ``[Mem:N]``, else every ref in it (compressed text).

    A memory that merely mentions ``[Mem:M]`` in its content does not own ``M``.
    """
    lead = _LEADING_MEM_REF.match(segment)
    if lead:
        return [int(lead.group(1))]
    return [idx for idx, _ in _extract_mem_refs(segment)]


class PreFlightContextFirewall:
    """
    Enforces a hard token cap by trimming the middle (sacrifice zone) only.
    Core and end are never trimmed.
    """

    def __init__(
        self,
        max_tokens: int,
        preflight_token_threshold: int | None = None,
        preflight_target_tokens: int | None = None,
        count_fn: Callable[[str], int] | None = None,
    ):
        self.max_tokens = max_tokens
        # Counts with the caller's tokenizer when given, so the budget matches what is reported.
        self._count = count_fn or count_tokens
        self.preflight_threshold = (
            preflight_token_threshold
            if preflight_token_threshold is not None
            else os_config.get_preflight_token_threshold()
        )
        self.preflight_target = (
            preflight_target_tokens
            if preflight_target_tokens is not None
            else os_config.get_preflight_target_tokens()
        )

    def run(
        self,
        core_text: str,
        end_text: str,
        middle_segments: Sequence[tuple[str, int]],
        ref_map: dict[int, str] | None = None,
        *,
        segment_weights: Sequence[float | int] | None = None,
        tier_shares: Mapping[int, float] | None = None,
        pinned_tiers: AbstractSet[int] | None = None,
    ) -> tuple[str, dict[int, str], int, dict[str, Any]]:
        """
        Build final context string under max_tokens; trim middle by priority.

        When ``segment_weights`` is provided with the same length as ``middle_segments``,
        segments are ordered by ``(-weight, tier)`` so higher density / relevance is kept first
        (custom reranker integration). Otherwise order is ``(tier, insertion)`` as in Phase 0.

        Returns:
            (assembled_text, ref_map_kept, total_tokens, stats)

        ``tier_shares`` maps a tier to the fraction of the middle budget it may use in a first
        pass; space a tier leaves unused is filled by the others in a second pass.
        ``pinned_tiers`` are offered before all other tiers and are not limited by shares.

        ``stats`` includes ``segments_offered``, ``segments_kept``, ``tokens_trimmed`` (middle only)
        and ``kept_segments`` (surviving middle texts, in emitted order).
        """
        count = self._count
        ref_map = dict(ref_map or {})
        segs = list(middle_segments)
        if (
            segment_weights is not None
            and len(segment_weights) == len(segs)
            and segs
        ):
            order = sorted(
                range(len(segs)),
                key=lambda i: (-float(segment_weights[i]), segs[i][1]),
            )
            sorted_segments = [segs[i] for i in order]
        else:
            sorted_segments = sorted(segs, key=lambda x: (x[1], 0))
        if pinned_tiers:
            sorted_segments.sort(key=lambda x: x[1] not in pinned_tiers)
        offered_tok = sum(count(s[0]) for s in sorted_segments)

        core_tok = count(core_text)
        end_tok = count(end_text)
        reserved = core_tok + end_tok
        if reserved >= self.max_tokens:
            logger.warning(
                "PreFlight: core+end (%s) >= max_tokens (%s); sending core+end only",
                reserved,
                self.max_tokens,
            )
            combined = core_text + "\n\n" + end_text
            stats: dict[str, Any] = {
                "segments_offered": len(sorted_segments),
                "segments_kept": 0,
                "tokens_trimmed": int(offered_tok),
                "kept_segments": [],
            }
            return combined, _ref_map_from_text(combined, ref_map), reserved, stats

        budget = self.max_tokens - reserved
        middle_budget = budget
        shares = {
            t: f
            for t, f in (tier_shares or {}).items()
            if not pinned_tiers or t not in pinned_tiers
        }
        used: dict[int, int] = {}
        procedural_count = 0
        kept_idx: list[int] = []
        deferred: list[int] = []

        def _take(i: int) -> bool:
            nonlocal budget, procedural_count
            seg_text, tier = sorted_segments[i]
            seg_tok = count(seg_text)
            if seg_tok > budget:
                return False
            if tier == TIER_PROCEDURAL:
                if procedural_count >= 3:
                    return False
                procedural_count += 1
            kept_idx.append(i)
            budget -= seg_tok
            used[tier] = used.get(tier, 0) + seg_tok
            return True

        for i, (seg_text, tier) in enumerate(sorted_segments):
            if (
                tier in shares
                and used.get(tier, 0) + count(seg_text) > shares[tier] * middle_budget
            ):
                deferred.append(i)
                continue
            _take(i)
        for i in deferred:
            _take(i)
        kept = [sorted_segments[i][0] for i in sorted(kept_idx)]

        # Separators and token merges at joins are not in the per-segment counts:
        # drop the lowest-priority kept segments until the assembled text fits.
        while True:
            middle_str = "\n".join(kept)
            final = core_text + "\n\n" + middle_str + "\n\n" + end_text
            total = count(final)
            if total <= self.max_tokens or not kept:
                break
            kept.pop()

        kept_refs: dict[int, str] = {}
        for seg_text in kept:
            for idx in _segment_refs(seg_text):
                if idx in ref_map:
                    kept_refs[idx] = ref_map[idx]
        kept_tok = count(middle_str) if middle_str else 0
        stats = {
            "segments_offered": len(sorted_segments),
            "segments_kept": len(kept),
            "tokens_trimmed": max(0, int(offered_tok - kept_tok)),
            "kept_segments": list(kept),
        }
        return final, kept_refs, total, stats


def _ref_map_from_text(text: str, ref_map: dict[int, str]) -> dict[int, str]:
    """Extract [Mem:N] from text and return submap of ref_map for those N."""
    out = {}
    for idx, _ in _extract_mem_refs(text):
        if idx in ref_map:
            out[idx] = ref_map[idx]
    return out
