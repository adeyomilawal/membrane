"""Naive context packers for benchmarking and tests (pair / baseline strategies).

``PreFlightContextFirewall.run`` sorts middle segments by **tier** (procedural → semantic →
episodic) before greedy packing. These helpers implement alternative orderings so tests and
scorecards can assert Membrane retains higher-priority facts under the same token cap.

Keep packing rules aligned with :meth:`PreFlightContextFirewall.run` (procedural cap 3,
skip segment if larger than remaining budget).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from membrane.firewall import (
    TIER_PROCEDURAL,
    TIER_SEMANTIC,
    PreFlightContextFirewall,
    _extract_mem_refs,
    count_tokens,
)


def _ref_submap_from_text(text: str, ref_map: Dict[int, str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for idx, _ in _extract_mem_refs(text):
        if idx in ref_map:
            out[idx] = ref_map[idx]
    return out


def pack_middle_order(
    core_text: str,
    end_text: str,
    middle_segments: Sequence[Tuple[str, int]],
    max_tokens: int,
    *,
    order: str,
    ref_map: Optional[Dict[int, str]] = None,
) -> Tuple[str, Dict[int, str], int]:
    """
    Pack middle into core + middle + end under ``max_tokens``.

    Args:
        order:
            - ``tier`` — same as :class:`PreFlightContextFirewall` (ascending tier).
            - ``as_given`` — retrieval / list order (naive FIFO).
            - ``tier_desc`` — descending tier (episodic-heavy first; worst-case baseline).
    """
    ref_map = dict(ref_map or {})
    if order == "tier":
        fw = PreFlightContextFirewall(max_tokens=max_tokens)
        final, rmap, total, _stats = fw.run(core_text, end_text, middle_segments, ref_map=ref_map)
        return final, rmap, total

    if order == "as_given":
        segs = list(middle_segments)
    elif order == "tier_desc":
        segs = sorted(middle_segments, key=lambda x: (-x[1], 0))
    else:
        raise ValueError(f"unknown order: {order!r}")

    core_tok = count_tokens(core_text)
    end_tok = count_tokens(end_text)
    reserved = core_tok + end_tok
    if reserved >= max_tokens:
        combined = core_text + "\n\n" + end_text
        return combined, _ref_submap_from_text(combined, ref_map), reserved

    budget = max_tokens - reserved
    procedural_count = 0
    kept: List[str] = []
    kept_refs: Dict[int, str] = {}

    for seg_text, tier in segs:
        seg_tok = count_tokens(seg_text)
        if tier == TIER_PROCEDURAL:
            if procedural_count >= 3:
                continue
            procedural_count += 1
        if seg_tok > budget:
            continue
        kept.append(seg_text)
        budget -= seg_tok
        for idx, _ in _extract_mem_refs(seg_text):
            if idx in ref_map:
                kept_refs[idx] = ref_map[idx]

    middle_str = "\n".join(kept)
    final = core_text + "\n\n" + middle_str + "\n\n" + end_text
    total = count_tokens(final)
    return final, kept_refs, total


def pad_prefix_to_token_count(prefix: str, target_tokens: int) -> str:
    """Return a suffix of repeated ``x`` so ``count_tokens(prefix + suffix)`` is at most ``target_tokens``."""
    if count_tokens(prefix) >= target_tokens:
        return ""
    lo, hi = 0, target_tokens * 8
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        pad = "x" * mid
        t = count_tokens(prefix + pad)
        if t <= target_tokens:
            best = pad
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def tier_vs_fifo_demo_segments(
    *,
    semantic_lines: int = 4,
    semantic_tokens_each: int = 50,
    procedural_markers: Sequence[str] = ("MARK_PROC_0", "MARK_PROC_1", "MARK_PROC_2"),
) -> Tuple[str, str, List[Tuple[str, int]], Dict[int, str], List[str], int]:
    """Build core/end/segments/ref_map/markers/cap for the procedural-vs-FIFO regression fixture."""
    middle: List[Tuple[str, int]] = []
    for i in range(semantic_lines):
        prefix = f"[Mem:{i + 1}] MARK_SEM_{i} "
        middle.append((prefix + pad_prefix_to_token_count(prefix, semantic_tokens_each), TIER_SEMANTIC))
    for j, mark in enumerate(procedural_markers):
        idx = semantic_lines + j + 1
        middle.append((f"[Mem:{idx}] {mark} yy", TIER_PROCEDURAL))
    core = "C"
    end = "--- Current user message ---\nq"
    ref_map = {i: str(i) for i in range(1, semantic_lines + len(procedural_markers) + 1)}
    markers = list(procedural_markers)
    reserved = count_tokens(core) + count_tokens(end)
    cap = reserved + semantic_lines * semantic_tokens_each
    return core, end, middle, ref_map, markers, cap


def marker_survival_score(markers: Sequence[str], text: str) -> Tuple[int, List[str]]:
    """Return (count_present, list of markers missing from text)."""
    missing = [m for m in markers if m not in text]
    return len(markers) - len(missing), missing


def procedural_recall_vs_fifo(
    core_text: str,
    end_text: str,
    middle_segments: Sequence[Tuple[str, int]],
    max_tokens: int,
    must_keep_markers: Sequence[str],
    ref_map: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    """Compare tier-aware firewall vs naive FIFO on the same segments and budget."""
    final_tier, _, tok_tier = pack_middle_order(
        core_text, end_text, middle_segments, max_tokens, order="tier", ref_map=ref_map
    )
    final_fifo, _, tok_fifo = pack_middle_order(
        core_text, end_text, middle_segments, max_tokens, order="as_given", ref_map=ref_map
    )
    final_desc, _, tok_desc = pack_middle_order(
        core_text, end_text, middle_segments, max_tokens, order="tier_desc", ref_map=ref_map
    )

    score_tier, miss_tier = marker_survival_score(must_keep_markers, final_tier)
    score_fifo, miss_fifo = marker_survival_score(must_keep_markers, final_fifo)
    score_desc, miss_desc = marker_survival_score(must_keep_markers, final_desc)

    return {
        "tier_tokens": tok_tier,
        "fifo_tokens": tok_fifo,
        "tier_desc_tokens": tok_desc,
        "tier_markers_kept": score_tier,
        "fifo_markers_kept": score_fifo,
        "tier_desc_markers_kept": score_desc,
        "tier_missing": miss_tier,
        "fifo_missing": miss_fifo,
        "tier_desc_missing": miss_desc,
        "membrane_beats_fifo_on_markers": score_tier > score_fifo,
        "membrane_beats_tier_desc_on_markers": score_tier >= score_desc,
    }
