# Free CLI diagnostics

These commands exercise Membrane Core without network access.

Inspect a compiled prompt:

```bash
membrane inspect \
  --user-message "Explain the retry policy." \
  --memory "policy|2|Retries use exponential backoff with jitter." \
  --memory "runbook|1|Check timeout, retry budget, and idempotency before changing retries." \
  --max-tokens 2048
```

Quote input tokens and optional input-side cost:

```bash
membrane quote \
  --user-message "Summarize the release plan." \
  --memory "release|2|Version 0.2 ships after the integration tests pass." \
  --price-per-1k-input-tokens 0.00015 \
  --json
```

Explain available profiles:

```bash
membrane profile list
membrane profile explain coding
membrane profile explain coding
```

