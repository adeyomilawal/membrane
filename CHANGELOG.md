# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

- Relicense under **Apache-2.0** (previously PolyForm Noncommercial).
- Remove license activation, tier gates, portal analytics, and the CLI `login`, `logout`, `whoami`, `status`, `refresh`, and `token` commands. Every feature is available without activation.
- Drop the `PyJWT`, `httpx`, and `keyring` runtime dependencies.
- Remove `CompileMetrics.tier`.
- Remove from the open-source edition: JTA and other compression strategies, lineage, assertions, replay, diff/version store, export, batch, reranker, visualizer, testing utilities, Langfuse/LangSmith exporters, and the interrupt runner. The `research`, `full`, and `enterprise` profiles are removed with them.
- Add `FeatureNotAvailable`, raised by `CompileResult.diff()` and `enable_lineage=True` without a `lineage_tracker`.
- Add `memory_placement="end"` for prompt-cache-stable compiles.

## [0.1.1] — 2026-04-11

- Add an in-package `tests/` tree (README-style compile, firewall budget edge, interrupt helpers) and document running `pytest` from the package directory.
- Add GitHub Actions workflow: editable install, **Ruff** on `src`/`tests`, **pytest** on push/PR for `membrane/**`.
- Patch release after test and packaging verification.

[0.1.1]: https://github.com/adeyomilawal/membrane/compare/v0.1.0...v0.1.1
