from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransformationConfig(BaseModel):
    """User-controlled transformation requirements.

    Values are deliberately strings rather than closed enums: deployments can add
    new tones, audiences, languages, artifact types, or styles without changing
    application code.
    """

    model_config = ConfigDict(extra="allow")

    artifact_type: str = "auto"
    tone: str = "professional"
    audience: str = "general"
    language: str = "English"
    detail_level: str = "medium"
    objective: str = "inform"
    style: str = "clear"
    output_formats: list[str] = Field(default_factory=lambda: ["text"])
    custom_instructions: str = ""

    @field_validator(
        "artifact_type", "tone", "audience", "language", "detail_level", "objective", "style",
        mode="before",
    )
    @classmethod
    def normalize_string(cls, value: Any) -> str:
        value = str(value or "").strip()
        return value or "auto"

    @field_validator("output_formats", mode="before")
    @classmethod
    def normalize_formats(cls, value: Any) -> list[str]:
        if value is None:
            return ["text"]
        if isinstance(value, str):
            value = [value]
        return list(dict.fromkeys(str(item).strip().lower() for item in value if str(item).strip())) or ["text"]


class RenderingHints(BaseModel):
    tone: str = "professional"
    audience: str = "general"
    language: str = "English"
    detail_level: str = "medium"
    objective: str = "inform"
    style: str = "clear"


class CRRSection(BaseModel):
    heading: str
    content: str = ""
    bullets: list[str] = Field(default_factory=list)


class CRRClaim(BaseModel):
    claim_id: str
    text: str
    evidence: list[str] = Field(default_factory=list)


class CanonicalResponse(BaseModel):
    """Format-independent content representation emitted by Phase 4."""

    crr_version: str = "1.0"
    artifact_type: str
    title: str
    summary: str
    sections: list[CRRSection] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    claims: list[CRRClaim] = Field(default_factory=list)
    rendering: RenderingHints = Field(default_factory=RenderingHints)
    insufficiencies: list[str] = Field(default_factory=list)


class DigestClaim(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list)


class SectionDigest(BaseModel):
    source_id: str
    section: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    claims: list[DigestClaim] = Field(default_factory=list)


def available_chunk_ids(documents: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for doc in documents:
        meta = dict(doc.get("metadata") or {})
        chunk_id = meta.get("chunk_id") or doc.get("id")
        if chunk_id:
            ids.add(str(chunk_id))
    return ids


def sanitize_evidence(crr: CanonicalResponse, allowed_chunk_ids: set[str]) -> tuple[CanonicalResponse, list[str]]:
    """Drop hallucinated evidence identifiers without performing semantic verification.

    Phase 5 will determine whether a cited chunk *supports* a claim. Phase 4 only
    guarantees that references point to chunks that actually exist in the current
    retrieval context.
    """
    warnings: list[str] = []
    for claim in crr.claims:
        original = list(claim.evidence)
        claim.evidence = list(dict.fromkeys(ref for ref in original if ref in allowed_chunk_ids))
        invalid = [ref for ref in original if ref not in allowed_chunk_ids]
        if invalid:
            warnings.append(
                f"Claim {claim.claim_id} referenced unknown evidence IDs and they were removed: {', '.join(invalid)}"
            )
    return crr, warnings
