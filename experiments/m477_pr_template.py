"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M477 — Pull Request Template

Template for PRs.
"""
import json

content = """## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New experiment
- [ ] Documentation
- [ ] Infrastructure

## Checklist
- [ ] Experiment produces `_results.json`
- [ ] Tests pass locally
- [ ] Dev diary updated
- [ ] Book entry created (if new experiment)
"""

with open(".github/pull_request_template.md", "w") as f:
    f.write(content)

print("=" * 60)
print("M477 — PR TEMPLATE")
print("=" * 60)
print("pull_request_template.md created")

with open("experiments/m477_pr_template_results.json", "w") as f:
    json.dump({"created": True, "pass": True}, f, indent=2)

print("\n✅ M477: PR template created")
