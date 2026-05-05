# M232: Branch Registry / Marketplace

**Status:** ✅ Prototype
**Date:** 2026-05-01

## Question

Can WAL support a registry/marketplace for sharing and discovering edit branches?

## Concept

A **branch** is a named collection of edit recipes that can be:
- **Published** to a shared registry
- **Searched** by domain, fact content, or quality metrics
- **Forked** to create derivative branches
- **Rated** by survival rate and PPL impact

## Prototype Implementation

```python
# Publish
wal registry publish legal-contrafactuals-v1 \
  --recipes recipes/ \
  --metadata '{"domain": "legal", "survival": 0.7}'

# Search
wal registry search "Berlin"
# → geo-updates-2026 (0c3b7211)

# Fork
wal registry fork 0c3b7211 geo-updates-2026-fork
```

## Registry Schema

```json
{
  "id": "0c3b7211",
  "name": "geo-updates-2026",
  "recipes": [
    {"fact": "Capital of France is Berlin", "survival": 0.9},
    {"fact": "Eiffel Tower is in Berlin", "survival": 0.85}
  ],
  "metadata": {
    "domain": "geography",
    "author": "community",
    "survival_avg": 0.875,
    "ppl_delta": 0.45,
    "published_at": "2026-05-01T12:00:00"
  }
}
```

## Sample Registry

| ID | Name | Domain | Survival | PPL Δ |
|----|------|--------|----------|-------|
| b4839c75 | legal-contrafactuals-v1 | legal | 0.70 | +0.52 |
| 0c3b7211 | geo-updates-2026 | geography | 0.88 | +0.31 |
| 1ea0df77 | medical-facts-v2 | medical | 0.40 | +1.20 |

## Use Cases

### 1. Domain-Specific Branches
```bash
# Legal team publishes contract law updates
wal registry publish contract-law-2026 --domain=legal

# Medical team publishes drug interaction facts
wal registry publish drug-facts-v3 --domain=medical
```

### 2. Quality-Based Discovery
```bash
# Find branches with >80% survival
wal registry list --min-survival=0.8

# Find branches with PPL delta < 0.5
wal registry list --max-ppl-delta=0.5
```

### 3. Collaborative Editing
```bash
# Researcher A publishes initial branch
wal registry publish physics-updates-v1

# Researcher B forks and adds more facts
wal registry fork <id> physics-updates-v2
wal edit add --branch physics-updates-v2 fact6.json
wal registry publish physics-updates-v2
```

## Integration with WAL Build System

```
.wal/
  registry/          # Local registry cache
    0c3b7211.json
    b4839c75.json
  branches/          # Active branches
    legal-contrafactuals/
      recipes/
      builds/
```

## Future Features

- **Quality scoring**: `score = survival_avg × (1 - ppl_delta)`
- **Versioning**: Branch evolution tracking
- **Reviews**: Community validation of branch quality
- **Merge**: Combine two branches into one
- **Diff**: Compare branch versions

## Conclusion

> **Branch registry enables collaborative model editing. Teams can publish, share, and fork edit collections, creating an ecosystem around WAL.**

## Next Steps

- Implement remote registry (GitHub/GitLab integration)
- Add quality scoring and ranking
- Support branch merge and conflict resolution
- Build web UI for registry browsing
