# Contributing to Membrane

Thanks for helping. Bug reports, fixes, docs and new adapters are all welcome.

## Before you start

- For anything bigger than a small fix, open an issue first so we can agree on the approach.
- Security problems: don't open a public issue. See [SECURITY.md](SECURITY.md).

## Contributor License Agreement

Every contribution needs a signed [Contributor License Agreement](CLA.md). You keep the
copyright in your work; the CLA lets the project use and relicense it. Agree to it by
ticking the CLA box in the pull request template. A pull request without that box ticked
won't be merged.

## Development setup

```bash
git clone https://github.com/adeyomilawal/membrane.git
cd membrane
uv sync --locked --extra dev
```

The package is published as `membrane-aol` and imports as `membrane`.

## Before you open a pull request

Run the same checks CI runs:

```bash
uv lock --check
uv run ruff check src tests
uv run pytest
uv build
```

- Add a test for every bug fix (it should fail before your change) and for new behavior.
- Match the surrounding code: naming, comment density and style. Lines are capped at 100.
- Membrane must make no network calls by default and must keep prompt text out of metrics.
- Add a line to [CHANGELOG.md](CHANGELOG.md) under the unreleased version for user-visible changes.

## Commit messages

Short imperative summary (under 72 characters), a blank line, then what changed and why.
