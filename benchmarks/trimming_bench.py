"""Rough benchmark: firewall trimming under tight budget (run: python -m benchmarks.trimming_bench)."""

from __future__ import annotations

import time

from membrane.firewall import TIER_EPISODIC, TIER_PROCEDURAL, TIER_SEMANTIC, PreFlightContextFirewall


def main() -> None:
    core = "System.\n"
    end = "User: ping\n"
    middle = []
    for i in range(20):
        tier = TIER_PROCEDURAL if i % 5 == 0 else TIER_SEMANTIC if i % 2 == 0 else TIER_EPISODIC
        middle.append((f"[Mem:{i+1}] line-{i} " + "x" * 80, tier))
    fw = PreFlightContextFirewall(max_tokens=400)
    t0 = time.perf_counter()
    final, kept, total, _stats = fw.run(core, end, middle, ref_map={i: f"id-{i}" for i in range(1, 21)})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"segments_in={len(middle)} segments_kept~={final.count('[Mem:')} total_tokens={total} ref_keys={len(kept)}")
    print(f"elapsed_ms={elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
