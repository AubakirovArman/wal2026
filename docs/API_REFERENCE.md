# WAL API Reference

## Python API

### `WALProject`

```python
from wal import WALProject

# Initialize
project = WALProject("./my_project")
project.init(model="meta-llama/Llama-3.1-8B")

# Add recipes
project.add_recipe("What is X?", "Y")
project.add_recipes([
    ("Q1", "A1"),
    ("Q2", "A2"),
])

# Build
project.build(layer=16, rank=4, steps=100)

# Test
results = project.test()
print(results.ci_score)

# Tag
project.tag("v1.0")

# Rollback
project.rollback("v1.0")
```

### `train_lora_fp32`

```python
from wal import train_lora_fp32

train_lora_fp32(
    model,
    tokenizer,
    facts=[("Q", "A")],
    layer=16,
    rank=4,
    steps=100,
    lr=1e-4,
)
```

### `test_question`

```python
from wal import test_question

passed = test_question(model, tokenizer, "What is X?", "Y")
```

## CLI API

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAL_MODEL` | Llama-3.1-8B | Base model name |
| `WAL_LAYER` | 16 | Target layer |
| `WAL_RANK` | 4 | LoRA rank |
| `WAL_STEPS` | 100 | Training steps |
| `WAL_DEVICE` | cuda:0 | GPU device |

### Return Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Build failed |
| 2 | Tests failed |
| 3 | Invalid recipe |
| 4 | Rollback failed |
