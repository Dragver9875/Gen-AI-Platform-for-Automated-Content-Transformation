from __future__ import annotations
from pathlib import Path
from typing import Any


def artifact_metrics(records: list[dict[str, Any]] | None, expected_formats: list[str] | None = None) -> dict[str, float]:
    records = list(records or [])
    expected = [str(x).lower() for x in (expected_formats or [])]
    total = len(records)
    generated = sum(r.get("status") == "generated" for r in records)
    failed = sum(r.get("status") == "failed" for r in records)
    source_only = sum(r.get("status") == "source_only" for r in records)
    nonempty = 0
    for r in records:
        path = r.get("path")
        if path:
            p = Path(path)
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                nonempty += 1
    produced_formats = {str(r.get("format") or "").lower() for r in records if r.get("status") == "generated"}
    coverage = (len(set(expected) & produced_formats) / len(set(expected))) if expected else (1.0 if total and failed == 0 else 0.0)
    return {
        "artifact_success_rate": generated / total if total else 0.0,
        "artifact_failure_rate": failed / total if total else 0.0,
        "artifact_source_only_rate": source_only / total if total else 0.0,
        "artifact_nonempty_file_rate": nonempty / total if total else 0.0,
        "expected_format_coverage": coverage,
    }
