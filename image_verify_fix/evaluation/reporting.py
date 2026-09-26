from __future__ import annotations
from pathlib import Path
from evaluation.models import EvalReport


def markdown_report(report: EvalReport) -> str:
    lines = [
        f"# Evaluation Report — {report.dataset_name}",
        "",
        f"Dataset version: `{report.dataset_version}`",
        f"Cases: **{report.aggregate.get('case_count', len(report.cases))}**",
        f"Pass rate: **{float(report.aggregate.get('pass_rate', 0.0)):.3f}**",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    for name, value in (report.aggregate.get("metrics") or {}).items():
        lines.append(f"| `{name}` | {float(value):.6f} |")
    lines += ["", "## Cases", ""]
    for case in report.cases:
        lines += [
            f"### {case.case_id}",
            "",
            f"- Status: `{case.status}`",
            f"- Passed: `{case.passed}`",
            f"- Warnings: {len(case.warnings)}",
            f"- Errors: {len(case.errors)}",
            "",
        ]
    return "\n".join(lines)


def save_markdown(report: EvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(report), encoding="utf-8")
    return path
