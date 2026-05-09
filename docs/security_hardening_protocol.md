# Security Hardening Protocol

Date: 2026-05-09

## Purpose

M652-M658 add pre-alpha security and abuse gates for recipes, registry packages, hotfixes, retrieval context, provenance, and package signatures.

## Scope

- M652 scans recipes for secret-like and PII-like strings.
- M653 blocks malicious recipe payloads in question, answer, source, and metadata fields.
- M654 validates registry packages against digest, maintainer, typosquat, and capability rules.
- M655 requires approval, audit reason, and rollback data for CI-bypass hotfixes.
- M656 flags prompt injection inside retrieval context.
- M657 verifies provenance signatures and rejects tampering.
- M658 verifies signed packages and rejects digest/capability tampering.

## Non-Claims

- These gates do not prove production readiness.
- These gates do not replace external security review.
- These gates do not execute untrusted packages.
- These gates are deterministic pre-alpha security contracts.

## Output Artifacts

- `corpora/security_recipe_secret_scan.json`
- `corpora/security_malicious_recipe_vectors.jsonl`
- `corpora/security_registry_poisoning.json`
- `corpora/security_retrieval_injection.jsonl`
- `corpora/security_provenance_tamper.json`
- `corpora/security_signed_package_verification.json`
