"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M409 — Config Validation Schema

Validates WAL config files against JSON Schema.
"""
import json

schema = {
    "type": "object",
    "required": ["model", "training", "recipes"],
    "properties": {
        "model": {"type": "string"},
        "training": {
            "type": "object",
            "required": ["lr", "epochs"],
            "properties": {
                "lr": {"type": "number", "minimum": 1e-6, "maximum": 1.0},
                "epochs": {"type": "integer", "minimum": 1},
            }
        },
        "recipes": {"type": "array", "minItems": 1},
    }
}

def validate(config):
    errors = []
    if not isinstance(config, dict):
        return ["Config must be an object"]
    for key in schema["required"]:
        if key not in config:
            errors.append(f"Missing required key: {key}")
    if "training" in config:
        t = config["training"]
        for subkey in schema["properties"]["training"]["required"]:
            if subkey not in t:
                errors.append(f"Missing training key: {subkey}")
        if "lr" in t and not (1e-6 <= t["lr"] <= 1.0):
            errors.append(f"lr out of range: {t['lr']}")
        if "epochs" in t and t["epochs"] < 1:
            errors.append(f"epochs must be >= 1: {t['epochs']}")
    if "recipes" in config and len(config["recipes"]) < 1:
        errors.append("recipes array must not be empty")
    return errors

configs = [
    ({"model": "llama", "training": {"lr": 5e-5, "epochs": 3}, "recipes": [{}]}, True, "Valid config"),
    ({"model": "llama", "training": {"lr": 5.0, "epochs": 3}, "recipes": [{}]}, False, "LR too high"),
    ({"model": "llama", "training": {"lr": 5e-5}, "recipes": [{}]}, False, "Missing epochs"),
    ({"model": "llama", "training": {"lr": 5e-5, "epochs": 3}, "recipes": []}, False, "Empty recipes"),
]

print("=" * 60)
print("M409 — CONFIG VALIDATION SCHEMA")
print("=" * 60)

passed = 0
for config, expected_ok, desc in configs:
    errors = validate(config)
    ok = len(errors) == 0
    status = ok == expected_ok
    if status:
        passed += 1
    print(f"  {'✅' if status else '❌'} {desc}: errors={len(errors)}")

print(f"\nScore: {passed}/{len(configs)}")
assert passed == len(configs)
print("\n✅ M409: Config schema validation working")

with open("experiments/m409_config_validation_results.json", "w") as f:
    json.dump({"passed": passed, "total": len(configs), "pass": True}, f, indent=2)
