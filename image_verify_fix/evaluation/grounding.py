from __future__ import annotations
from typing import Any


def grounding_metrics(report: dict[str, Any] | None) -> dict[str, float]:
    report = dict(report or {})
    claims = list(report.get("claims") or [])
    total = len(claims)
    if not total:
        return {
            "faithfulness": float(report.get("faithfulness_score") or 0.0),
            "claim_support_rate": 0.0,
            "claim_partial_rate": 0.0,
            "claim_unsupported_rate": 0.0,
            "claim_insufficient_rate": 0.0,
            "mean_verifier_confidence": 0.0,
            "repair_attempts": float(report.get("repair_attempts") or 0),
        }
    counts = {"supported": 0, "partially_supported": 0, "unsupported": 0, "insufficient_evidence": 0}
    confidences: list[float] = []
    for claim in claims:
        status = str(claim.get("status") or "")
        if status in counts:
            counts[status] += 1
        try:
            confidences.append(float(claim.get("confidence") or 0.0))
        except (TypeError, ValueError):
            pass
    return {
        "faithfulness": float(report.get("faithfulness_score") or 0.0),
        "claim_support_rate": counts["supported"] / total,
        "claim_partial_rate": counts["partially_supported"] / total,
        "claim_unsupported_rate": counts["unsupported"] / total,
        "claim_insufficient_rate": counts["insufficient_evidence"] / total,
        "mean_verifier_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "repair_attempts": float(report.get("repair_attempts") or 0),
    }
