from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


VerificationStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "insufficient_evidence",
]


class SemanticClaimAssessment(BaseModel):
    claim_id: str
    status: VerificationStatus
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supported_evidence: list[str] = Field(default_factory=list)
    rationale: str = ""
    unsupported_fragments: list[str] = Field(default_factory=list)
    suggested_correction: str = ""

    @field_validator("supported_evidence", mode="before")
    @classmethod
    def dedupe_evidence(cls, value):
        return list(dict.fromkeys(str(v) for v in (value or []) if str(v).strip()))


class SemanticVerificationBatch(BaseModel):
    claims: list[SemanticClaimAssessment] = Field(default_factory=list)


class DeterministicIssue(BaseModel):
    kind: str
    severity: Literal["warning", "conflict"] = "warning"
    literal: str = ""
    message: str


class ClaimVerification(BaseModel):
    claim_id: str
    claim_text: str
    status: VerificationStatus
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cited_evidence: list[str] = Field(default_factory=list)
    supported_evidence: list[str] = Field(default_factory=list)
    rationale: str = ""
    unsupported_fragments: list[str] = Field(default_factory=list)
    suggested_correction: str = ""
    deterministic_issues: list[DeterministicIssue] = Field(default_factory=list)


class VerificationReport(BaseModel):
    verification_version: str = "1.0"
    passed: bool
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    supported_claims: int = 0
    partially_supported_claims: int = 0
    unsupported_claims: int = 0
    insufficient_evidence_claims: int = 0
    claims: list[ClaimVerification] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    repair_attempts: int = 0
    max_repair_attempts: int = 2

    @property
    def requires_repair(self) -> bool:
        return not self.passed and (
            self.partially_supported_claims
            + self.unsupported_claims
            + self.insufficient_evidence_claims
        ) > 0
