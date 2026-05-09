# Robustness Data Protocol

Date: 2026-05-09  
Scope: M639-M645

## Purpose

M639-M645 define the first data robustness layer after safe-core and cross-model protocol gates.

## Gates

| Module | Gate | Current Meaning |
|--------|------|-----------------|
| M639 | Dirty facts corpus | Creates a 500-record messy fact corpus seed |
| M640 | Ambiguous facts | Ensures multiple acceptable answers are represented |
| M641 | Temporal facts | Requires date-scoped answers |
| M642 | Long-answer facts | Tests answers longer than one token |
| M643 | Procedural routing | Ensures procedures do not route to weight edits |
| M644 | Policy/refusal edits | Ensures policy-like edits route to refusal tier |
| M645 | Hard facts hybrid backend | Defines retrieval/edit fallback for hard author/inventor facts |

## Non-Claims

These gates validate corpora and routing contracts. They do not claim real model behavior until connected to `MODEL_SMALL`, `MODEL_MEDIUM`, or `GPU_HEAVY` runners.
