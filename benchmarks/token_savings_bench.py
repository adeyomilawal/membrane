"""Rough benchmark: glaciate prefix size vs full system tokens (run: python -m benchmarks.token_savings_bench)."""

from __future__ import annotations

import time

from membrane.firewall import count_tokens
from membrane.glaciator import glaciate


def main() -> None:
    stable = "You are a helpful assistant.\n" * 40
    volatile = "\n--- Current turn ---\nUser: What is 2+2?\n"
    full = stable + volatile
    t0 = time.perf_counter()
    gr = glaciate(full, max_cacheable_tokens=2000)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"full_tokens={count_tokens(full)} prefix_tokens={gr.prefix_tokens} tail_tokens={gr.tail_tokens}")
    print(f"use_caching={gr.use_caching} elapsed_ms={elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
