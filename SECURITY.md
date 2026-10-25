# Security Policy

## Supported versions

Security fixes go into the latest release on PyPI (`membrane-aol`).

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
**[Report a vulnerability](https://github.com/adeyomilawal/membrane/security/advisories/new)**
(Security tab → "Report a vulnerability"). Don't open a public issue.

Include what you found, how to reproduce it, the affected version, and the impact you
expect. You'll get an acknowledgement within 7 days and a plan or fix timeline after triage.
We'll credit you in the advisory unless you prefer otherwise.

## Scope notes

- Membrane makes no network calls by default. Optional Redis features and the MCP server
  connect only where you configure them.
- The MCP server holds one session's state and is meant for a single local client over
  stdio. Exposing the SSE transport on a non-loopback address shares that state with every
  client; that is a deployment risk, not a vulnerability, unless it bypasses a documented
  protection.
