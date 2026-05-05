# M222: WAL Build System Prototype

**Status:** ✅ Prototype Working
**Date:** 2026-05-01

## Concept

Don't store checkpoint diffs. Store build history.

```
base checkpoint + edit recipes + datasets + seeds → compiled WAL
```

## Commands

```bash
wal init <base_model>          # Initialize project
wal edit add <facts.json>      # Add edit recipe (auto-detects strategy)
wal build                      # Compile all edits
wal test                       # Run evaluation
wal tag <version>              # Tag build
wal rollback <tag>             # Rollback to version
wal status                     # Show project status
```

## Auto-Strategy Detection

```
geography/music → easy (50 steps)
author/inventor → hard (contrastive)
science → medium (test first)
```

## Structure

```
.wal/
  config.json       # Project config
  tags.json         # Version tags
  recipes/          # Edit recipes
  builds/           # Compiled checkpoints
```

## Implications

WAL Build System turns model editing into a software engineering workflow — versioned, reproducible, auditable.
