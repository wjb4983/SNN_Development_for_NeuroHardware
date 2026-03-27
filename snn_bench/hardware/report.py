from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constraints import evaluate_constraints
from .profiles import HardwareProfile


def _score(checks: list[dict[str, Any]]) -> float:
    if not checks:
        return 0.0
    passed = sum(1 for check in checks if check.get("passed"))
    return round((passed / len(checks)) * 100.0, 2)


def emit_deployment_report(export_payload: dict[str, Any], profile: HardwareProfile, out_dir: Path | str) -> dict[str, Path]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    export_meta = export_payload.get("metadata", {})
    checks = [
        {
            "name": check.name,
            "passed": check.passed,
            "observed": check.observed,
            "expected": check.expected,
            "remediation": check.remediation,
        }
        for check in evaluate_constraints(export_meta, profile)
    ]
    readiness = _score(checks)

    report_payload = {
        "run_id": export_payload.get("run_id"),
        "target_profile": profile.name,
        "readiness_score": readiness,
        "status": "pass" if readiness == 100.0 else "fail",
        "checks": checks,
    }

    json_path = output / "deployment_report.json"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    failing = [check for check in checks if not check["passed"]]
    md_lines = [
        f"# Deployment readiness: {report_payload['run_id']}",
        "",
        f"- **Target profile:** `{profile.name}`",
        f"- **Readiness score:** **{readiness:.2f}%**",
        f"- **Status:** **{report_payload['status'].upper()}**",
        "",
        "## Checks",
        "",
        "| Check | Result | Observed | Expected |",
        "|---|---|---|---|",
    ]
    for check in checks:
        result = "PASS ✅" if check["passed"] else "FAIL ❌"
        md_lines.append(f"| `{check['name']}` | {result} | `{check['observed']}` | `{check['expected']}` |")

    md_lines.extend(["", "## Suggested remediations", ""])
    if failing:
        for item in failing:
            md_lines.append(f"- **{item['name']}**: {item['remediation']}")
    else:
        md_lines.append("- No remediation required; deployment constraints are satisfied.")

    md_path = output / "deployment_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {"json_report": json_path, "markdown_report": md_path}
