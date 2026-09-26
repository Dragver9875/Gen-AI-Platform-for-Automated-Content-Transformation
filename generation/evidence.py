from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from generation.crr import CanonicalResponse, SectionDigest


def document_chunk_id(doc: dict[str, Any]) -> str | None:
    meta = dict(doc.get("metadata") or {})
    value = meta.get("chunk_id") or doc.get("id")
    text = str(value or "").strip()
    return text or None


@dataclass(frozen=True)
class EvidenceAliases:
    """Deterministic short aliases for model-facing evidence references.

    The LLM sees E1/E2/... only. Real vector/database chunk IDs remain internal
    and are restored before Phase 5 or persistence.
    """

    alias_to_chunk: dict[str, str]
    chunk_to_alias: dict[str, str]

    @classmethod
    def from_documents(cls, documents: Iterable[dict[str, Any]]) -> "EvidenceAliases":
        alias_to_chunk: dict[str, str] = {}
        chunk_to_alias: dict[str, str] = {}
        for doc in documents:
            chunk_id = document_chunk_id(doc)
            if not chunk_id or chunk_id in chunk_to_alias:
                continue
            alias = f"E{len(alias_to_chunk) + 1}"
            alias_to_chunk[alias] = chunk_id
            chunk_to_alias[chunk_id] = alias
        return cls(alias_to_chunk=alias_to_chunk, chunk_to_alias=chunk_to_alias)

    @classmethod
    def from_serialized(cls, alias_to_chunk: dict[str, Any] | None) -> "EvidenceAliases":
        cleaned = {
            str(alias): str(chunk)
            for alias, chunk in dict(alias_to_chunk or {}).items()
            if str(alias).strip() and str(chunk).strip()
        }
        return cls(cleaned, {chunk: alias for alias, chunk in cleaned.items()})

    def aliases_for_chunks(self, chunk_ids: Iterable[str]) -> set[str]:
        return {self.chunk_to_alias[c] for c in chunk_ids if c in self.chunk_to_alias}

    def to_aliases(self, refs: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(self.chunk_to_alias[r] for r in refs if r in self.chunk_to_alias))

    def resolve(
        self,
        refs: Iterable[str],
        *,
        allowed_chunk_ids: set[str] | None = None,
        accept_raw_ids: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Resolve model-facing aliases back to real chunk IDs.

        ``accept_raw_ids`` is a backwards-compatibility guard for old/custom
        providers. New prompts never expose raw IDs, so normal model output uses
        aliases. Unknown references are returned separately and never persisted.
        """
        resolved: list[str] = []
        invalid: list[str] = []
        allowed = allowed_chunk_ids
        for raw_ref in refs:
            ref = str(raw_ref).strip()
            if not ref:
                continue
            chunk_id = self.alias_to_chunk.get(ref)
            if chunk_id is None and accept_raw_ids and ref in self.chunk_to_alias:
                chunk_id = ref
            if chunk_id is None or (allowed is not None and chunk_id not in allowed):
                invalid.append(ref)
                continue
            if chunk_id not in resolved:
                resolved.append(chunk_id)
        return resolved, invalid


def crr_to_aliases(crr: CanonicalResponse, aliases: EvidenceAliases) -> CanonicalResponse:
    clone = crr.model_copy(deep=True)
    for claim in clone.claims:
        claim.evidence = aliases.to_aliases(claim.evidence)
    return clone


def resolve_crr_aliases(
    crr: CanonicalResponse,
    aliases: EvidenceAliases,
    *,
    allowed_chunk_ids: set[str],
) -> tuple[CanonicalResponse, list[str]]:
    warnings: list[str] = []
    clone = crr.model_copy(deep=True)
    for claim in clone.claims:
        resolved, invalid = aliases.resolve(claim.evidence, allowed_chunk_ids=allowed_chunk_ids)
        claim.evidence = resolved
        if invalid:
            warnings.append(
                f"Claim {claim.claim_id} referenced unknown evidence aliases and they were removed: "
                + ", ".join(invalid)
            )
    return clone, warnings


def digest_to_aliases(digest: SectionDigest, aliases: EvidenceAliases) -> SectionDigest:
    clone = digest.model_copy(deep=True)
    for claim in clone.claims:
        claim.evidence = aliases.to_aliases(claim.evidence)
    return clone


def resolve_digest_aliases(
    digest: SectionDigest,
    aliases: EvidenceAliases,
    *,
    allowed_chunk_ids: set[str],
) -> tuple[SectionDigest, list[str]]:
    clone = digest.model_copy(deep=True)
    warnings: list[str] = []
    for claim in clone.claims:
        resolved, invalid = aliases.resolve(claim.evidence, allowed_chunk_ids=allowed_chunk_ids)
        claim.evidence = resolved
        if invalid:
            warnings.append(
                f"Section digest '{clone.section}' referenced unknown evidence aliases and they were removed: "
                + ", ".join(invalid)
            )
    return clone, warnings


def verification_report_to_alias_dict(report: Any, aliases: EvidenceAliases) -> dict[str, Any]:
    """Return a model-facing report dict with chunk IDs replaced by aliases."""
    data = report.model_dump(mode="json") if hasattr(report, "model_dump") else deepcopy(dict(report))
    for claim in data.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim["cited_evidence"] = aliases.to_aliases(claim.get("cited_evidence") or [])
        claim["supported_evidence"] = aliases.to_aliases(claim.get("supported_evidence") or [])
    return data
