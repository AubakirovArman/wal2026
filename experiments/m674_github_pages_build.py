from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
RESULT_PATH = ROOT / "experiments" / "m674_github_pages_build_results.json"


def load_metrics() -> dict[str, object]:
    return json.loads((ROOT / "docs" / "project_metrics.json").read_text(encoding="utf-8"))


def render_index(metrics: dict[str, object]) -> str:
    experiments = metrics["experiments"]
    code = metrics["code"]
    small_models = metrics["small_model_gates"]
    legacy_audit = metrics.get("legacy_audit", {})
    aigi = metrics.get("aigi", {})
    generated_at = datetime.now(timezone.utc).isoformat()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WAL Studio — Pre-alpha WeightOps Research Framework</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; line-height: 1.55; color: #172033; }}
    h1, h2 {{ line-height: 1.2; }}
    code {{ background: #f3f5f7; padding: 0.1rem 0.25rem; border-radius: 4px; }}
    pre {{ background: #0f172a; color: #e2e8f0; padding: 1rem; overflow-x: auto; border-radius: 8px; }}
    .status {{ display: inline-block; background: #fff7ed; color: #9a3412; padding: 0.25rem 0.5rem; border-radius: 999px; font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; background: #fff; }}
  </style>
</head>
<body>
  <h1>WAL Studio</h1>
  <p><span class="status">pre-alpha research framework</span></p>
  <p>WAL Studio is a WeightOps prototype for representing model edits as reproducible recipes, building auditable artifacts, running behavioral gates, and rolling back regressions.</p>

  <h2>30-second demo</h2>
  <pre><code>pip install -e .[dev]
python wal_studio_v01/demo.py
python -m wal validate-results experiments --fail-on-invalid</code></pre>

  <h2>What is validated?</h2>
  <div class="grid">
    <div class="card"><strong>Core tests</strong><br>{code["pytest_tests"]} maintained pytest tests pass.</div>
    <div class="card"><strong>Result schema</strong><br>{experiments["result_json_files"]}/{experiments["result_json_files"]} result JSON files validate.</div>
    <div class="card"><strong>Safe sweep</strong><br>{experiments["safe_sweep_pass"]} safe scripts pass; {experiments["safe_sweep_blocked"]} are policy-blocked.</div>
    <div class="card"><strong>Small models</strong><br>{small_models["unique_model_paths"]} unique local runtime/artifact workflows pass.</div>
    <div class="card"><strong>Legacy audit</strong><br>M1-M50 batch: {legacy_audit.get("m1_m50_total", 0)} scripts classified; {legacy_audit.get("m1_m50_current_public_claim_allowed", 0)} current public claims.</div>
    <div class="card"><strong>AIGI loop</strong><br>M679-M696: {aigi.get("fact_learning_passed", 0)} learned facts, {aigi.get("bad_memory_rejected", 0)} bad-memory rejections, {aigi.get("feedback_episodes_passed", 0)} feedback episodes, {aigi.get("real_hf_backend_checks_passed", 0)} real-HF checks, {aigi.get("soft_prompt_adapter_checks_passed", 0)} soft-prompt checks, {aigi.get("logit_lora_adapter_checks_passed", 0)} logit-LoRA checks, {aigi.get("module_lora_adapter_checks_passed", 0)} module-LoRA checks.</div>
  </div>

  <h2>What is not validated?</h2>
  <ul>
    <li>No production-readiness claim.</li>
    <li>No external certification claim.</li>
    <li>Small-model gates do not yet prove semantic weight-edit training.</li>
    <li>M1-M50 safe-pass scripts still need schema-v1 result artifacts before current public claims.</li>
    <li>AIGI M679-M696 are SDK memory-loop, real-inference, soft-prompt, logit-LoRA, and module-LoRA gates, not autonomous AGI or production multi-fact editing.</li>
    <li>Deployment modules remain prototypes/simulations unless explicitly marked otherwise.</li>
  </ul>

  <h2>How to run</h2>
  <pre><code>python -m wal core --help
python -m wal studio --help
python -m wal studio init local-demo-model
python -m wal studio edit add examples/quickstart/facts.json
python -m wal studio status</code></pre>

  <h2>Roadmap</h2>
  <p>Next useful milestone: run semantic edit training across the same small-model protocol and compare against RAG-only and LoRA baselines.</p>

  <h2>References</h2>
  <ul>
    <li><a href="../README.md">README.md</a></li>
    <li><a href="../TECHNICAL_REPORT.md">TECHNICAL_REPORT.md</a></li>
    <li><a href="../docs/VALIDATION_STATUS.md">docs/VALIDATION_STATUS.md</a></li>
    <li><a href="../docs/aigi/README.md">docs/aigi/README.md</a></li>
    <li><a href="../KNOWN_ISSUES.md">KNOWN_ISSUES.md</a></li>
  </ul>

  <p><small>Generated at {html.escape(generated_at)} from docs/project_metrics.json.</small></p>
</body>
</html>
"""


def main() -> int:
    metrics = load_metrics()
    experiments = metrics["experiments"]
    SITE.mkdir(parents=True, exist_ok=True)
    index = SITE / "index.html"
    status_json = SITE / "status.json"
    body = render_index(metrics)
    index.write_text(body, encoding="utf-8")
    status_payload = {
        "status": metrics["status"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": ["docs/project_metrics.json", "docs/VALIDATION_STATUS.md"],
        "metrics": metrics,
    }
    status_json.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = []
    lowered = body.lower()
    if "production-ready" in lowered:
        failures.append("forbidden_claim_in_site")
    if "pre-alpha" not in lowered:
        failures.append("missing_pre_alpha")
    if not index.exists() or not status_json.exists():
        failures.append("site_artifact_missing")
    for stale_denominator in (468, 472, 476):
        if f'{experiments["result_json_files"]}/{stale_denominator}' in body:
            failures.append(f"stale_result_json_denominator:{stale_denominator}")
    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M674",
        "name": "GitHub Pages Build",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "site_files": [str(index.relative_to(ROOT)), str(status_json.relative_to(ROOT))],
        "failures": failures,
        "scope": "static local GitHub Pages artifact build",
        "docs": "docs/product_polish_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M674 GitHub Pages Build: {status}")
    print(f"files={len(result['site_files'])} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
