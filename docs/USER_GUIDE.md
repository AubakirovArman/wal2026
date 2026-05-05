# WAL User Guide

## Quick Start

```bash
# Initialize a new WAL project
wal init

# Add knowledge recipes
wal edit add --question "What is X?" --answer "Y"

# Build the model
wal build

# Run tests
wal test

# Tag a version
wal tag v1.0

# Rollback if needed
wal rollback v1.0
```

## Commands

### `wal init`
Initialize a new WAL project in the current directory.

### `wal edit add`
Add a knowledge recipe. Each recipe is a (question, answer) pair.

### `wal build`
Compile recipes into weight updates using LoRA adapters.

### `wal test`
Run the CI test suite:
- Exact match tests
- Paraphrase tests
- Negative tests
- Perplexity check
- NaN detection

### `wal tag <name>`
Tag the current build with a version name.

### `wal rollback <tag>`
Rollback to a previously tagged version.

### `wal diff`
Show differences between current and previous recipe sets.

### `wal status`
Show project status: recipe count, build hash, CI score.

## Configuration

Edit `.wal/config.json`:
```json
{
  "model": "meta-llama/Llama-3.1-8B",
  "layer": 16,
  "rank": 4,
  "dtype": "fp32"
}
```

## Best Practices

1. **Always run tests before tagging**
2. **Use rehearsal for batch edits**
3. **Enable negative-aware training**
4. **Tag after every successful build**
5. **Use rollback instead of rebuild**
