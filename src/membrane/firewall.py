"""PreFlight context firewall: exact token counting and priority-based trimming.

Core and end (current turn + user message) are never trimmed; middle is trimmed
by priority tier (procedural first, max 3, then semantic/episodic).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import membrane.config as os_config

logger = logging.getLogger(__name__)

# Use Literal[False] only (not bool) so failure does not widen to True | False.
_tiktoken_encoding: Union[Any, Literal[False], None] = None


def _get_encoding() -> Union[Any, Literal[False]]:
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


def count_tokens(text: str) -> int:
    """Return exact token count for text using tiktoken; fallback ~4 chars/token."""
    if not text:
        return 0
    enc = _get_encoding()
    if enc is False:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


TIER_PROCEDURAL = 1
TIER_SEMANTIC = 2
TIER_EPISODIC = 3

_MEM_REF_PATTERN = re.compile(r"\[Mem:(\d+)\]")
_LEADING_MEM_REF = re.compile(r"^\s*\[Mem:(\d+)\]")


def _extract_mem_refs(segment: str) -> List[Tuple[int, int]]:
    """Return list of (local_index, start_pos) for [Mem:N] in segment."""
    out = []
    for m in _MEM_REF_PATTERN.finditer(segment):
        try:
            idx = int(m.group(1))
            out.append((idx, m.start()))
        except ValueError:
            continue
    return out


def _segment_refs(segment: str) -> List[int]:
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
        preflight_token_threshold: Optional[int] = None,
        preflight_target_tokens: Optional[int] = None,
    ):
        self.max_tokens = max_tokens
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
        middle_segments: Sequence[Tuple[str, int]],
        ref_map: Optional[Dict[int, str]] = None,
        *,
        segment_weights: Optional[Sequence[Union[float, int]]] = None,
    ) -> Tuple[str, Dict[int, str], int, Dict[str, Any]]:
        """
        Build final context string under max_tokens; trim middle by priority.

        When ``segment_weights`` is provided with the same length as ``middle_segments``,
        segments are ordered by ``(-weight, tier)`` so higher density / relevance is kept first
        (custom reranker integration). Otherwise order is ``(tier, insertion)`` as in Phase 0.

        Returns:
            (assembled_text, ref_map_kept, total_tokens, stats)

        ``stats`` includes ``segments_offered``, ``segments_kept``, ``tokens_trimmed`` (middle only)
        and ``kept_segments`` (surviving middle texts, in emitted order).
        """
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
        offered_tok = sum(count_tokens(s[0]) for s in sorted_segments)

        core_tok = count_tokens(core_text)
        end_tok = count_tokens(end_text)
        reserved = core_tok + end_tok
        if reserved >= self.max_tokens:
            logger.warning(
                "PreFlight: core+end (%s) >= max_tokens (%s); sending core+end only",
                reserved,
                self.max_tokens,
            )
            combined = core_text + "\n\n" + end_text
            stats: Dict[str, Any] = {
                "segments_offered": len(sorted_segments),
                "segments_kept": 0,
                "tokens_trimmed": int(offered_tok),
                "kept_segments": [],
            }
            return combined, _ref_map_from_text(combined, ref_map), reserved, stats

        budget = self.max_tokens - reserved
        procedural_count = 0
        kept: List[str] = []

        for seg_text, tier in sorted_segments:
            seg_tok = count_tokens(seg_text)
            if tier == TIER_PROCEDURAL:
                if procedural_count >= 3:
                    continue
                procedural_count += 1
            if seg_tok > budget:
                continue
            kept.append(seg_text)
            budget -= seg_tok

        # Separators and token merges at joins are not in the per-segment counts:
        # drop the lowest-priority kept segments until the assembled text fits.
        while True:
            middle_str = "\n".join(kept)
            final = core_text + "\n\n" + middle_str + "\n\n" + end_text
            total = count_tokens(final)
            if total <= self.max_tokens or not kept:
                break
            kept.pop()

        kept_refs: Dict[int, str] = {}
        for seg_text in kept:
            for idx in _segment_refs(seg_text):
                if idx in ref_map:
                    kept_refs[idx] = ref_map[idx]
        kept_tok = count_tokens(middle_str) if middle_str else 0
        stats = {
            "segments_offered": len(sorted_segments),
            "segments_kept": len(kept),
            "tokens_trimmed": max(0, int(offered_tok - kept_tok)),
            "kept_segments": list(kept),
        }
        return final, kept_refs, total, stats


def _ref_map_from_text(text: str, ref_map: Dict[int, str]) -> Dict[int, str]:
    """Extract [Mem:N] from text and return submap of ref_map for those N."""
    out = {}
    for idx, _ in _extract_mem_refs(text):
        if idx in ref_map:
            out[idx] = ref_map[idx]
    return out
