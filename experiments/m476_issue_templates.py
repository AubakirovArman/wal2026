"""
M476 — Issue Templates

GitHub issue templates for bugs and features.
"""
import json, os

os.makedirs(".github/ISSUE_TEMPLATE", exist_ok=True)

bug_template = """---
name: Bug report
about: Report a bug in WAL
title: '[BUG] '
labels: bug
---

**Describe the bug**
A clear description.

**To Reproduce**
Steps to reproduce.

**Expected behavior**
What should happen.

**Environment**
- WAL version:
- Python version:
- GPU:
"""

feature_template = """---
name: Feature request
about: Suggest a new feature
title: '[FEATURE] '
labels: enhancement
---

**Describe the feature**
What should be added.

**Use case**
Why is this needed.
"""

with open(".github/ISSUE_TEMPLATE/bug_report.md", "w") as f:
    f.write(bug_template)
with open(".github/ISSUE_TEMPLATE/feature_request.md", "w") as f:
    f.write(feature_template)

print("=" * 60)
print("M476 — ISSUE TEMPLATES")
print("=" * 60)
print("Created bug_report.md and feature_request.md")

with open("experiments/m476_issue_templates_results.json", "w") as f:
    json.dump({"templates": 2, "pass": True}, f, indent=2)

print("\n✅ M476: Issue templates created")
