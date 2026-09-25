from __future__ import annotations

from pydantic import BaseModel, Field
from generation.crr import CanonicalResponse, CRRClaim, CRRSection, RenderingHints


class ContentIR(BaseModel):
    """Verified, artifact-agnostic semantic representation.

    CRR remains the Phase 4/5 transport contract. ContentIR is the stable input
    to Phase 6 artifact planners, preventing renderer concerns from leaking back
    into generation/verification.
    """

    version: str = "1.0"
    artifact_type: str
    title: str
    summary: str
    sections: list[CRRSection] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    claims: list[CRRClaim] = Field(default_factory=list)
    rendering: RenderingHints = Field(default_factory=RenderingHints)
    insufficiencies: list[str] = Field(default_factory=list)

    @classmethod
    def from_crr(cls, crr: CanonicalResponse) -> "ContentIR":
        return cls(
            artifact_type=crr.artifact_type,
            title=crr.title,
            summary=crr.summary,
            sections=crr.sections,
            key_points=crr.key_points,
            claims=crr.claims,
            rendering=crr.rendering,
            insufficiencies=crr.insufficiencies,
        )
