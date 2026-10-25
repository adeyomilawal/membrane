# AGENTS.md — Membrane

## Scope

This repository owns the open-source `membrane` Python distribution (Apache-2.0),
its tests, benchmarks, examples, and documentation.

## Boundaries

- Membrane must remain independently installable with no network calls by default.
- Keep prompt content out of any metrics; only documented aggregate metrics may
  leave the host application, and only through a callback the user configures.
- Never commit tokens, credentials, virtual environments, build artifacts, or
  downloaded models.

## Verification

```bash
uv lock --check
uv sync --locked --extra dev
uv run ruff check src tests
uv run pytest
uv build
```
