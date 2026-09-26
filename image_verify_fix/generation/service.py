from __future__ import annotations

from typing import Any, Protocol

from agents.context import render_context
from generation.crr import (
    CanonicalResponse,
    SectionDigest,
    TransformationConfig,
    available_chunk_ids,
    sanitize_evidence,
)
from generation.prompts import (
    crr_system_prompt,
    digest_system_prompt,
    digest_user_prompt,
    qa_user_prompt,
    synthesis_user_prompt,
)


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


class GenerationService:
    """Phase 4 grounded generation service.

    QA uses one grounded model call. Whole-document transformations use a
    hierarchical strategy: every Phase 3 structural group is digested first,
    then those digests are synthesized into the final CRR. No semantic
    factuality decision is made here; that belongs to Phase 5.
    """

    def __init__(
        self,
        llm: StructuredGenerator,
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        group_context_max_chars: int = 18000,
        digest_max_tokens: int = 2048,
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.group_context_max_chars = group_context_max_chars
        self.digest_max_tokens = digest_max_tokens

    @staticmethod
    def config_from_state(state: dict[str, Any]) -> TransformationConfig:
        raw = state.get("transformation_config") or {}
        if isinstance(raw, TransformationConfig):
            return raw
        return TransformationConfig.model_validate(raw)

    def generate_qa(self, state: dict[str, Any]) -> tuple[CanonicalResponse, list[str], dict[str, Any]]:
        config = self.config_from_state(state)
        if config.artifact_type == "auto":
            config = config.model_copy(update={"artifact_type": "answer"})
        data = self.llm.generate_json(
            system_prompt=crr_system_prompt(),
            user_prompt=qa_user_prompt(
                query=str(state.get("query") or ""),
                config=config,
                context=str(state.get("prepared_context") or ""),
            ),
            schema_name="canonical_response",
            json_schema=CanonicalResponse.model_json_schema(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        crr = CanonicalResponse.model_validate(data)
        crr, warnings = sanitize_evidence(crr, available_chunk_ids(list(state.get("retrieved_documents") or [])))
        return crr, warnings, {"strategy": "single_pass_grounded_qa", "llm_calls": 1, "section_digests": 0}

    def generate_transform(self, state: dict[str, Any]) -> tuple[CanonicalResponse, list[SectionDigest], list[str], dict[str, Any]]:
        config = self.config_from_state(state)
        groups = list(state.get("context_groups") or [])
        digests: list[SectionDigest] = []
        warnings: list[str] = []

        for group in groups:
            context = render_context([group], max_chars=self.group_context_max_chars)
            if not context.strip():
                warnings.append(
                    f"Skipped empty structural group: {group.get('source_id', 'unknown')} / {group.get('section', 'Unsectioned')}"
                )
                continue
            data = self.llm.generate_json(
                system_prompt=digest_system_prompt(),
                user_prompt=digest_user_prompt(
                    source_id=str(group.get("source_id") or "unknown-source"),
                    section=str(group.get("section") or "Unsectioned"),
                    context=context,
                ),
                schema_name="section_digest",
                json_schema=SectionDigest.model_json_schema(),
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, self.digest_max_tokens),
            )
            digest = SectionDigest.model_validate(data).model_copy(update={
                "source_id": str(group.get("source_id") or "unknown-source"),
                "section": str(group.get("section") or "Unsectioned"),
            })
            allowed = available_chunk_ids(list(group.get("chunks") or []))
            for claim in digest.claims:
                invalid = [ref for ref in claim.evidence if ref not in allowed]
                claim.evidence = list(dict.fromkeys(ref for ref in claim.evidence if ref in allowed))
                if invalid:
                    warnings.append(
                        f"Section digest '{digest.section}' referenced unknown evidence IDs and they were removed: {', '.join(invalid)}"
                    )
            digests.append(digest)

        data = self.llm.generate_json(
            system_prompt=crr_system_prompt(),
            user_prompt=synthesis_user_prompt(
                query=str(state.get("query") or ""),
                config=config,
                digests=[digest.model_dump(mode="json") for digest in digests],
            ),
            schema_name="canonical_response",
            json_schema=CanonicalResponse.model_json_schema(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        crr = CanonicalResponse.model_validate(data)
        crr, crr_warnings = sanitize_evidence(
            crr, available_chunk_ids(list(state.get("retrieved_documents") or []))
        )
        warnings.extend(crr_warnings)
        return crr, digests, warnings, {
            "strategy": "hierarchical_section_digest_synthesis",
            "llm_calls": len(digests) + 1,
            "section_digests": len(digests),
        }
