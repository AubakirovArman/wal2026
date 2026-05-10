# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.1.x   | ✅ |
| 1.0.x   | ⚠️ |

## Reporting Vulnerabilities

Use GitHub Security Advisories for this repository when available. If private advisories are unavailable, contact the repository owner through GitHub before publishing details.

## Known Issues

- Prompt injection has deterministic static gates, but this is not an external security audit.
- Memory growth has short-run sentinels; a real 24h soak test remains blocked until a long-run runner is available.
