"""Measure the ``token_scale`` seeds in ``src/membrane/llm_capabilities.json``.

Membrane counts every prompt with tiktoken's ``cl100k_base``. A provider with another
tokenizer bills a different number of tokens for the same text; ``token_scale`` is the
starting guess for that ratio, which ``ContextCompiler.observe`` then corrects from real
usage. This script measures the ratio with Anthropic's token-count endpoint (free, no
model is run) over three kinds of text:

* prose: this repository's README without its code blocks and tables,
* code: Python source from ``src/membrane``,
* digits: generated lines of numbers, dates, ids and JSON records (fixed seed).

Each kind is cut into pieces of about 2,000 counted tokens and each piece is sent as one
user message. The request framing (what the endpoint reports for a one-character message,
minus that character) is subtracted, so the ratio is text against text.

One-off maintenance script: it is not imported by the library, which makes no network
calls. Needs credentials the Anthropic SDK can find (``ANTHROPIC_API_KEY``).

    uv run --extra llm-anthropic python -m benchmarks.measure_token_scale
    uv run --extra llm-anthropic python -m benchmarks.measure_token_scale \
        --models claude-opus-5-5 --out benchmarks/results/token_scale.json

It prints a table and does not edit the capability map; copy the values in by hand.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from datetime import date, timedelta
from pathlib import Path

from membrane.firewall import count_tokens

ROOT = Path(__file__).resolve().parent.parent
PIECE_TOKENS = 2000
PIECES_PER_KIND = 6
KINDS = ("prose", "code", "digits")


def _prose() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(("|", "<", "!["))]
    return "\n".join(lines)


def _code() -> str:
    names = ("compiler.py", "keyframes.py", "layout.py", "provider_registry.py")
    return "\n".join((ROOT / "src" / "membrane" / n).read_text(encoding="utf-8") for n in names)


def _digits() -> str:
    rng = random.Random(20261111)
    day = date(2024, 1, 1)
    lines: list[str] = []
    for i in range(4000):
        day += timedelta(days=rng.randint(0, 3))
        kind = i % 4
        if kind == 0:
            lines.append(" ".join(str(rng.randint(0, 10**rng.randint(1, 9))) for _ in range(8)))
        elif kind == 1:
            lines.append(
                f"{day.isoformat()} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} "
                f"order {rng.randint(10**7, 10**8)} total {rng.uniform(1, 9999):.2f} "
                f"qty {rng.randint(1, 500)}"
            )
        elif kind == 2:
            lines.append(
                json.dumps(
                    {
                        "id": rng.getrandbits(64),
                        "lat": round(rng.uniform(-90, 90), 6),
                        "lon": round(rng.uniform(-180, 180), 6),
                        "ms": rng.randint(0, 99999),
                    }
                )
            )
        else:
            lines.append(f"{rng.getrandbits(128):032x} +{rng.randint(10**10, 10**11)}")
    return "\n".join(lines)


def _pieces(text: str) -> list[str]:
    """Up to ``PIECES_PER_KIND`` pieces of about ``PIECE_TOKENS`` counted tokens, cut at
    line ends and spread evenly over the text."""
    out: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        current.append(line)
        size += count_tokens(line) + 1
        if size >= PIECE_TOKENS:
            out.append("\n".join(current))
            current, size = [], 0
    if len(out) <= PIECES_PER_KIND:
        return out
    step = len(out) / PIECES_PER_KIND
    return [out[int(i * step)] for i in range(PIECES_PER_KIND)]


def _provider_count(client, model: str, text: str) -> int:
    r = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": text}])
    return int(r.input_tokens)


def measure(client, model: str, samples: dict[str, list[str]]) -> dict[str, object]:
    framing = _provider_count(client, model, "a") - 1
    kinds: dict[str, dict[str, float]] = {}
    for kind, pieces in samples.items():
        ratios = [
            (_provider_count(client, model, p) - framing) / count_tokens(p) for p in pieces
        ]
        kinds[kind] = {
            "mean": round(statistics.fmean(ratios), 4),
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
            "pieces": len(ratios),
        }
    mix = statistics.fmean(k["mean"] for k in kinds.values())
    return {"framing_tokens": framing, "kinds": kinds, "mix": round(mix, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", help="comma-separated model ids (default: every listed one)")
    parser.add_argument("--out", help="write the measurements to this JSON file")
    args = parser.parse_args()

    import anthropic

    client = anthropic.Anthropic()
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = [m.id for m in client.models.list(limit=100)]
    samples = {"prose": _pieces(_prose()), "code": _pieces(_code()), "digits": _pieces(_digits())}

    results: dict[str, object] = {}
    print(f"{'model':<28}" + "".join(f"{k:>9}" for k in KINDS) + f"{'mix':>9}")
    for model in models:
        try:
            r = measure(client, model, samples)
        except anthropic.APIError as exc:
            print(f"{model:<28} failed: {type(exc).__name__}")
            continue
        results[model] = r
        kinds = r["kinds"]
        print(
            f"{model:<28}"
            + "".join(f"{kinds[k]['mean']:>9.3f}" for k in KINDS)  # type: ignore[index]
            + f"{r['mix']:>9.3f}"
        )
    if args.out:
        payload = {
            "measured_on": date.today().isoformat(),
            "counter": "tiktoken cl100k_base",
            "endpoint": "anthropic messages.count_tokens",
            "piece_tokens": PIECE_TOKENS,
            "models": results,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
