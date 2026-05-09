# M653 — Malicious Recipe Injection

Date: 2026-05-09  
Status: PASS  
Result: `experiments/m653_malicious_recipe_injection_results.json`

## Purpose

Block malicious recipe payloads embedded in question, answer, metadata, and source fields.

## Result

- Vectors: `5`
- Blocked vectors: `5`
- Failures: `0`

## Outcome

Prompt override, template escape, script tag, local file probe, and SQL payload fixtures are rejected.
