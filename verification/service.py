from __future__ import annotations

from typing import Any, Protocol

from generation.crr import CanonicalResponse, TransformationConfig, available_chunk_ids, sanitize_evidence
from verification.literals import DeterministicLiteralValidator
from verification.models import (
    ClaimVerification,
    SemanticClaimAssessment,
    SemanticVerificationBatch,
    VerificationReport,
)
from verification.prompts import repair_system_prompt, repair_user_prompt, verifier_system_prompt, verifier_user_prompt
from verification.profiles import VerificationProfile, VerificationProfileRegistry


class StructuredGenerator(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        json_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]: ...


def _meta(doc: dict[str, Any]) -> dict[str, Any]:
    return dict(doc.get("metadata") or {})


def _chunk_id(doc: dict[str, Any]) -> str:
    meta = _meta(doc)
    return str(meta.get("chunk_id") or doc.get("id") or "")


def _claim_weight(status: str) -> float:
    return 1.0 if status == "supported" else 0.5 if status == "partially_supported" else 0.0


class VerificationService:
    """Phase 5 claim verification + bounded repair service.

    The semantic verifier receives only claim-cited chunks. A deterministic layer
    independently checks critical literals; hard conflicts override a semantic
    'supported' verdict. Scores and pass/fail decisions are calculated locally.
    """

    def __init__(
        self,
        verifier: StructuredGenerator,
        *,
        repair_generator: StructuredGenerator | None = None,
        verification_temperature: float = 0.0,
        verification_max_tokens: int = 4096,
        repair_temperature: float = 0.05,
        repair_max_tokens: int = 4096,
        max_repair_attempts: int = 2,
        min_faithfulness_score: float = 1.0,
        max_claims_per_call: int = 12,
        evidence_chars_per_claim: int = 8000,
        profile_registry: VerificationProfileRegistry | None = None,
    ):
        self.verifier = verifier
        self.repair_generator = repair_generator or verifier
        self.verification_temperature = verification_temperature
        self.verification_max_tokens = verification_max_tokens
        self.repair_temperature = repair_temperature
        self.repair_max_tokens = repair_max_tokens
        self.max_repair_attempts = max(0, int(max_repair_attempts))
        self.min_faithfulness_score = min(1.0, max(0.0, float(min_faithfulness_score)))
        self.max_claims_per_call = max(1, int(max_claims_per_call))
        self.evidence_chars_per_claim = max(500, int(evidence_chars_per_claim))
        self.literal_validator = DeterministicLiteralValidator()
        self.profile_registry = profile_registry or VerificationProfileRegistry(
            [VerificationProfile(
                name="strict",
                min_faithfulness_score=self.min_faithfulness_score,
                allow_partial=False,
                require_evidence=True,
                deterministic_conflicts_fail=True,
            )],
            default_profile="strict",
        )

    @staticmethod
    def _config(state: dict[str, Any]) -> TransformationConfig:
        raw = state.get("transformation_config") or {}
        return raw if isinstance(raw, TransformationConfig) else TransformationConfig.model_validate(raw)

    @staticmethod
    def _crr(state: dict[str, Any]) -> CanonicalResponse:
        raw = state.get("canonical_response") or {}
        return raw if isinstance(raw, CanonicalResponse) else CanonicalResponse.model_validate(raw)

    @staticmethod
    def _document_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {_chunk_id(doc): doc for doc in list(state.get("retrieved_documents") or []) if _chunk_id(doc)}

    def _evidence_for_claim(self, claim, documents: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        blocks: list[dict[str, Any]] = []
        texts: list[str] = []
        used = 0
        for evidence_id in claim.evidence:
            doc = documents.get(str(evidence_id))
            if not doc:
                continue
            text = str(doc.get("text") or "").strip()
            if not text:
                continue
            remaining = self.evidence_chars_per_claim - used
            if remaining <= 0:
                break
            text = text[:remaining]
            used += len(text)
            meta = _meta(doc)
            blocks.append({
                "chunk_id": str(evidence_id),
                "source_id": str(meta.get("source_id") or ""),
                "filename": str(meta.get("filename") or ""),
                "section": str(meta.get("section") or ""),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "slide_start": meta.get("slide_start"),
                "slide_end": meta.get("slide_end"),
                "text": text,
            })
            texts.append(text)
        return blocks, "\n\n".join(texts)

    def verify(self, state: dict[str, Any]) -> tuple[VerificationReport, list[str], dict[str, Any]]:
        crr = self._crr(state)
        config = self._config(state)
        profile = self.profile_registry.resolve(config.verification_profile)
        documents = self._document_map(state)
        warnings: list[str] = []
        semantic: dict[str, SemanticClaimAssessment] = {}
        calls = 0

        items: list[dict[str, Any]] = []
        evidence_cache: dict[str, tuple[list[dict[str, Any]], str]] = {}
        for claim in crr.claims:
            blocks, combined = self._evidence_for_claim(claim, documents)
            evidence_cache[claim.claim_id] = (blocks, combined)
            items.append({"claim_id": claim.claim_id, "claim": claim.text, "cited_evidence": blocks})

        for start in range(0, len(items), self.max_claims_per_call):
            batch_items = items[start : start + self.max_claims_per_call]
            if not batch_items:
                continue
            data = self.verifier.generate_json(
                system_prompt=verifier_system_prompt(),
                user_prompt=verifier_user_prompt(batch_items),
                schema_name="semantic_verification_batch",
                json_schema=SemanticVerificationBatch.model_json_schema(),
                temperature=self.verification_temperature,
                max_tokens=self.verification_max_tokens,
            )
            calls += 1
            batch = SemanticVerificationBatch.model_validate(data)
            requested = {item["claim_id"] for item in batch_items}
            for assessment in batch.claims:
                if assessment.claim_id not in requested:
                    warnings.append(f"Verifier returned unknown claim_id '{assessment.claim_id}' and it was ignored.")
                    continue
                assessment.supported_evidence = [
                    ref for ref in assessment.supported_evidence
                    if ref in {b["chunk_id"] for b in evidence_cache[assessment.claim_id][0]}
                ]
                semantic[assessment.claim_id] = assessment

        results: list[ClaimVerification] = []
        for claim in crr.claims:
            blocks, combined = evidence_cache[claim.claim_id]
            assessment = semantic.get(claim.claim_id)
            if not blocks:
                assessment = SemanticClaimAssessment(
                    claim_id=claim.claim_id,
                    status="insufficient_evidence",
                    confidence=1.0,
                    rationale="The claim has no usable cited evidence in the active retrieval context.",
                    supported_evidence=[],
                    unsupported_fragments=[claim.text],
                    suggested_correction="Remove the unsupported assertion or explicitly state that the supplied evidence is insufficient.",
                )
            elif assessment is None:
                assessment = SemanticClaimAssessment(
                    claim_id=claim.claim_id,
                    status="insufficient_evidence",
                    confidence=0.0,
                    rationale="The verifier did not return an assessment for this claim.",
                    supported_evidence=[],
                    unsupported_fragments=[claim.text],
                    suggested_correction="Re-check this claim against its cited evidence.",
                )
                warnings.append(f"Verifier omitted claim_id '{claim.claim_id}'.")

            issues = self.literal_validator.validate(claim.text, combined)
            final_status = assessment.status
            conflicts = [issue for issue in issues if issue.severity == "conflict"]
            if conflicts and profile.deterministic_conflicts_fail:
                final_status = "unsupported"
            results.append(ClaimVerification(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                status=final_status,
                confidence=assessment.confidence,
                cited_evidence=list(claim.evidence),
                supported_evidence=assessment.supported_evidence,
                rationale=assessment.rationale,
                unsupported_fragments=assessment.unsupported_fragments,
                suggested_correction=assessment.suggested_correction,
                deterministic_issues=issues,
            ))

        total = len(results)
        score = sum(_claim_weight(result.status) for result in results) / total if total else 1.0
        counts = {
            "supported": sum(r.status == "supported" for r in results),
            "partially_supported": sum(r.status == "partially_supported" for r in results),
            "unsupported": sum(r.status == "unsupported" for r in results),
            "insufficient_evidence": sum(r.status == "insufficient_evidence" for r in results),
        }
        threshold = profile.min_faithfulness_score
        evidence_failure = counts["insufficient_evidence"] > 0 if profile.require_evidence else False
        partial_failure = counts["partially_supported"] > 0 if not profile.allow_partial else False
        passed = (
            score >= threshold
            and counts["unsupported"] == 0
            and not evidence_failure
            and not partial_failure
        )
        attempts = int(state.get("repair_attempts") or 0)
        report = VerificationReport(
            passed=passed,
            faithfulness_score=round(score, 6),
            supported_claims=counts["supported"],
            partially_supported_claims=counts["partially_supported"],
            unsupported_claims=counts["unsupported"],
            insufficient_evidence_claims=counts["insufficient_evidence"],
            claims=results,
            warnings=warnings,
            repair_attempts=attempts,
            max_repair_attempts=self.max_repair_attempts,
        )
        return report, warnings, {
            "verification_llm_calls": calls,
            "claims_verified": total,
            "verification_profile": profile.name,
            "verification_threshold": threshold,
        }

    def repair(self, state: dict[str, Any], report: VerificationReport) -> tuple[CanonicalResponse, list[str], dict[str, Any]]:
        crr = self._crr(state)
        config = self._config(state)
        documents = self._document_map(state)
        evidence_ids: list[str] = []
        for result in report.claims:
            if result.status != "supported":
                evidence_ids.extend(result.cited_evidence)
        evidence_ids = list(dict.fromkeys(evidence_ids))
        evidence: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            doc = documents.get(evidence_id)
            if not doc:
                continue
            meta = _meta(doc)
            evidence.append({
                "chunk_id": evidence_id,
                "source_id": str(meta.get("source_id") or ""),
                "section": str(meta.get("section") or ""),
                "text": str(doc.get("text") or "")[: self.evidence_chars_per_claim],
            })

        data = self.repair_generator.generate_json(
            system_prompt=repair_system_prompt(),
            user_prompt=repair_user_prompt(
                query=str(state.get("query") or ""),
                config=config,
                crr=crr,
                report=report,
                evidence=evidence,
            ),
            schema_name="canonical_response",
            json_schema=CanonicalResponse.model_json_schema(),
            temperature=self.repair_temperature,
            max_tokens=self.repair_max_tokens,
        )
        repaired = CanonicalResponse.model_validate(data)
        repaired, warnings = sanitize_evidence(repaired, available_chunk_ids(list(state.get("retrieved_documents") or [])))
        return repaired, warnings, {"repair_llm_calls": 1, "repair_evidence_chunks": len(evidence)}
