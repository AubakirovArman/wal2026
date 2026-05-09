# M652 — Recipe Secret Scanner

Date: 2026-05-09  
Status: PASS  
Result: `experiments/m652_recipe_secret_scanner_results.json`

## Purpose

Detect secret-like and PII-like strings inside recipe fields before they enter build or registry flows.

## Result

- Recipes checked: `6`
- Blocked recipes: `5`
- Failures: `0`

## Outcome

The static recipe secret scanner blocks the fixture secret cases without blocking the clean fixture.
