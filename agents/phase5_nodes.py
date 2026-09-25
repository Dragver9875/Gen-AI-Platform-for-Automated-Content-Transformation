from __future__ import annotations

from typing import Any

from agents.state import AgentState
from verification.models import VerificationReport
from verification.service import VerificationService


class Phase5Nodes:
    def __init__(self, verifier: VerificationService):
        self.verifier = verifier

    def verify_crr(self, state: AgentState) -> dict[str, Any]:
        try:
            report, warnings, metadata = self.verifier.verify(state)
            verification_metadata = dict(state.get("verification_metadata") or {})
            verification_metadata.update(metadata)
            return {
                "verification_report": report.model_dump(mode="json"),
                "verification_requires_repair": report.requires_repair,
                "verification_metadata": verification_metadata,
                "warnings": list(dict.fromkeys(list(state.get("warnings") or []) + warnings)),
                "max_repair_attempts": self.verifier.max_repair_attempts,
                "status": "phase5_verified" if report.passed else "phase5_verification_failed",
            }
        except Exception as exc:
            return {
                "status": "error",
                "verification_requires_repair": False,
                "errors": list(state.get("errors") or []) + [f"Phase 5 verification failed: {exc}"],
            }

    def repair_crr(self, state: AgentState) -> dict[str, Any]:
        try:
            report = VerificationReport.model_validate(state.get("verification_report") or {})
            repaired, warnings, metadata = self.verifier.repair(state, report)
            verification_metadata = dict(state.get("verification_metadata") or {})
            verification_metadata["repair_llm_calls_total"] = int(verification_metadata.get("repair_llm_calls_total") or 0) + int(metadata.get("repair_llm_calls") or 0)
            verification_metadata["last_repair_evidence_chunks"] = int(metadata.get("repair_evidence_chunks") or 0)
            return {
                "canonical_response": repaired.model_dump(mode="json"),
                "repair_attempts": int(state.get("repair_attempts") or 0) + 1,
                "verification_report": {},
                "verification_requires_repair": False,
                "verification_metadata": verification_metadata,
                "warnings": list(dict.fromkeys(list(state.get("warnings") or []) + warnings)),
                "status": "phase5_repaired",
            }
        except Exception as exc:
            return {
                "status": "error",
                "verification_requires_repair": False,
                "errors": list(state.get("errors") or []) + [f"Phase 5 repair failed: {exc}"],
            }

    def finalize_verified(self, state: AgentState) -> dict[str, Any]:
        return {"status": "phase5_complete", "verification_requires_repair": False}

    def finalize_with_issues(self, state: AgentState) -> dict[str, Any]:
        warnings = list(state.get("warnings") or [])
        report = state.get("verification_report") or {}
        if report:
            warnings.append(
                "Verification completed with unresolved claims after the configured repair limit."
            )
        return {
            "status": "phase5_complete_with_issues",
            "verification_requires_repair": False,
            "warnings": list(dict.fromkeys(warnings)),
        }
