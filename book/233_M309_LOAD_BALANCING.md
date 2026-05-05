# M309 — Load Balancing

## Date
2026-05-03

## Hypothesis
Requests can be distributed across multiple GPU instances.

## Method
Least-loaded vs round-robin on 3 instances.

## Results
- Least-loaded: balanced load across all instances
- Round-robin: overloaded slow instance
- Least-loaded preferred

## Verdict
✅ **CONFIRMED** — Load balancing distributes evenly.

## Integration
Least-loaded strategy for multi-GPU deployment.
