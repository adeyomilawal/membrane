"""Membrane user-facing quality & latency scorecard (stdlib only).

Run from repo root or membrane package directory::

    PYTHONPATH=src python -m benchmarks.scorecard

Outputs median compile latency, token budget compliance, and trim efficiency proxies.
No network; no LLM calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import when run as python -m benchmarks.scorecard from membrane/
from membrane import (
    TIER_PROCEDURAL,
    TIER_SEMANTIC,
    ContextCompiler,
    MemoryItem,
    PreFlightContextFirewall,
    compile_anonymous,
)
from membrane.pair_baselines import procedural_recall_vs_fifo, tier_vs_fifo_demo_segments


@dataclass
class ScenarioResult:
    name: str
    median_ms: float
    p95_ms: float
    iterations: int
    token_out: int
    budget_max: int
    budget_ok: bool
    notes: str


@dataclass
class PairComparisonResult:
    """Tier-aware packing vs naive FIFO / worst-tier on identical segments."""

    tier_markers_kept: int
    fifo_markers_kept: int
    tier_desc_markers_kept: int
    membrane_beats_fifo: bool
    max_tokens: int
    tier_total_tokens: int
    fifo_total_tokens: int
    median_ms: float
    p95_ms: float
    notes: str


def _time_runs(fn: Callable[[], Any], iterations: int) -> tuple[float, float, Any]:
    samples: List[float] = []
    last: Any = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    p95 = samples[int(0.95 * (len(samples) - 1))]
    return median, p95, last


def scenario_tiny_compile() -> ScenarioResult:
    c = ContextCompiler(max_tokens=4096, core_content="You are helpful.", include_interrupt_instruction=False)

    def one():
        return c.compile(memories=[], user_message="Hi")

    med, p95, r = _time_runs(one, 40)
    tok = r.metadata["token_count"]
    return ScenarioResult(
        name="tiny_compile",
        median_ms=med,
        p95_ms=p95,
        iterations=40,
        token_out=tok,
        budget_max=4096,
        budget_ok=tok <= 4096,
        notes="Empty retrieval + short user (typical chat open)",
    )


def scenario_rag_20_mem_trim() -> ScenarioResult:
    memories = [
        MemoryItem(id=f"id{i}", content=f"Fact {i}: " + ("x" * 120), memory_type="semantic", priority=TIER_SEMANTIC)
        for i in range(20)
    ]
    c = ContextCompiler(max_tokens=500, core_content="Kernel.", include_interrupt_instruction=False)

    def one():
        return c.compile(memories=memories, user_message="Summarize relevant facts.")

    med, p95, r = _time_runs(one, 25)
    tok = r.metadata["token_count"]
    return ScenarioResult(
        name="rag_20_mem_trim",
        median_ms=med,
        p95_ms=p95,
        iterations=25,
        token_out=tok,
        budget_max=500,
        budget_ok=tok <= 500,
        notes="Firewall must drop middle segments; measures trim path cost",
    )


def scenario_anonymous_long_history() -> ScenarioResult:
    # Inlined history lives in end_text; core+end are never trimmed. Budget must
    # be >= that combined size or token_count can exceed max_tokens (firewall
    # only drops middle segments, which compile_anonymous does not use).
    msgs = [{"role": "user", "content": "Line " + ("y" * 200)} for _ in range(25)]
    msgs += [{"role": "assistant", "content": "Ok."}]
    cap = 2048

    def one():
        return compile_anonymous(
            user_message="Next",
            conversation_messages=msgs,
            max_tokens=cap,
            core_content="System preamble.",
        )

    med, p95, r = _time_runs(one, 15)
    tok = r.metadata["token_count"]
    return ScenarioResult(
        name="anonymous_long_history",
        median_ms=med,
        p95_ms=p95,
        iterations=15,
        token_out=tok,
        budget_max=cap,
        budget_ok=tok <= cap,
        notes="compile_anonymous with long inlined thread (core+end preserved; cap fits typical history)",
    )


def scenario_firewall_only() -> ScenarioResult:
    core = "Core.\n"
    end = "User: done\n"
    middle = [(f"[Mem:{i}] " + "z" * 400, TIER_SEMANTIC) for i in range(1, 31)]
    fw = PreFlightContextFirewall(max_tokens=900)

    def one():
        return fw.run(core, end, middle, ref_map={i: f"u{i}" for i in range(1, 31)})

    med, p95, last = _time_runs(one, 30)
    _final, _refs, total, _stats = last
    return ScenarioResult(
        name="firewall_only",
        median_ms=med,
        p95_ms=p95,
        iterations=30,
        token_out=total,
        budget_max=900,
        budget_ok=total <= 900,
        notes="Pure PreFlightContextFirewall (no glaciate)",
    )


def scenario_tier_vs_fifo_procedural_markers() -> PairComparisonResult:
    """Same budget: FIFO (retrieval order) can drop all procedural markers; tier order keeps them."""
    core, end, middle, ref_map, markers, cap = tier_vs_fifo_demo_segments()

    def one():
        return procedural_recall_vs_fifo(
            core, end, middle, cap, markers, ref_map=ref_map
        )

    med, p95, last = _time_runs(one, 20)
    if not isinstance(last, dict):
        raise RuntimeError("procedural_recall_vs_fifo must return a dict")
    return PairComparisonResult(
        tier_markers_kept=int(last["tier_markers_kept"]),
        fifo_markers_kept=int(last["fifo_markers_kept"]),
        tier_desc_markers_kept=int(last["tier_desc_markers_kept"]),
        membrane_beats_fifo=bool(last["membrane_beats_fifo_on_markers"]),
        max_tokens=cap,
        tier_total_tokens=int(last["tier_tokens"]),
        fifo_total_tokens=int(last["fifo_tokens"]),
        median_ms=med,
        p95_ms=p95,
        notes="Markers=MARK_PROC_*; semantics listed first to penalize FIFO (median/p95 for comparison call)",
    )


def scenario_unicode_heavy() -> ScenarioResult:
    memories = [
        MemoryItem(id="α", content="日本語 mixed with English — " * 15, memory_type="semantic"),
        MemoryItem(id="β", content="Ελληνικά και العربية " * 10, memory_type="semantic"),
    ]
    c = ContextCompiler(max_tokens=6000, core_content="C", include_interrupt_instruction=False)

    def one():
        return c.compile(memories=memories, user_message="ترجم")

    med, p95, r = _time_runs(one, 20)
    tok = r.metadata["token_count"]
    return ScenarioResult(
        name="unicode_heavy",
        median_ms=med,
        p95_ms=p95,
        iterations=20,
        token_out=tok,
        budget_max=6000,
        budget_ok=tok <= 6000,
        notes="tiktoken path on multilingual text",
    )


def run_all(iterations_scale: int = 1) -> Dict[str, Any]:
    scenarios = [
        scenario_tiny_compile,
        scenario_rag_20_mem_trim,
        scenario_anonymous_long_history,
        scenario_firewall_only,
        scenario_unicode_heavy,
    ]
    results: List[ScenarioResult] = []
    for fn in scenarios:
        results.append(fn())
    pair = scenario_tier_vs_fifo_procedural_markers()
    all_ok = all(r.budget_ok for r in results)
    median_latencies = [r.median_ms for r in results]
    scorecard = {
        "version": "1.1",
        "all_budget_compliance": all_ok,
        "scenarios": [asdict(r) for r in results],
        "pair_comparison": asdict(pair),
        "summary": {
            "worst_median_ms": max(median_latencies),
            "best_median_ms": min(median_latencies),
            "mean_median_ms": statistics.mean(median_latencies),
            "membrane_beats_fifo_on_priority_markers": pair.membrane_beats_fifo,
        },
    }
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description="Membrane benchmark scorecard")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()
    data = run_all()
    if args.json:
        print(json.dumps(data, indent=2))
        return
    print("Membrane scorecard (local CPU; for trend comparison, not absolute SLA)\n")
    for s in data["scenarios"]:
        ok = "OK" if s["budget_ok"] else "FAIL"
        print(f"  [{ok}] {s['name']}")
        print(f"       median={s['median_ms']:.3f}ms  p95={s['p95_ms']:.3f}ms  tokens={s['token_out']}/{s['budget_max']}")
        print(f"       {s['notes']}")
    print()
    sm = data["summary"]
    print(f"  Mean median latency: {sm['mean_median_ms']:.3f}ms")
    print(f"  All within token budget: {data['all_budget_compliance']}")
    pc = data.get("pair_comparison") or {}
    if pc:
        win = "YES" if sm.get("membrane_beats_fifo_on_priority_markers") else "NO"
        print()
        print("  Pair baseline (same cap, semantics-before-procedures in list order):")
        print(f"       tier_kept_markers={pc.get('tier_markers_kept')}  fifo_kept={pc.get('fifo_markers_kept')}  worst_tier_kept={pc.get('tier_desc_markers_kept')}")
        print(f"       Membrane retains more priority markers than FIFO: {win}")
        print(f"       {pc.get('notes', '')}")


if __name__ == "__main__":
    main()
