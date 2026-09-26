from __future__ import annotations

from typing import Any, Protocol

from agents.context import render_context
from generation.crr import CanonicalResponse, SectionDigest, TransformationConfig, available_chunk_ids
from generation.evidence import EvidenceAliases, resolve_crr_aliases, resolve_digest_aliases
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
    """Phase 4 grounded generation service with short evidence aliases.

    Model-facing prompts use E1/E2/... rather than storage/database chunk IDs.
    Real chunk IDs are restored before Phase 5 and persistence.
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

    @staticmethod
    def aliases_from_state(state: dict[str, Any]) -> EvidenceAliases:
        serialized = dict(state.get("evidence_aliases") or {})
        if serialized:
            return EvidenceAliases.from_serialized(serialized)
        return EvidenceAliases.from_documents(list(state.get("retrieved_documents") or []))

    def generate_qa(self, state: dict[str, Any]) -> tuple[CanonicalResponse, list[str], dict[str, Any]]:
        config = self.config_from_state(state)
        if config.artifact_type == "auto":
            config = config.model_copy(update={"artifact_type": "answer"})
        docs = list(state.get("retrieved_documents") or [])
        aliases = self.aliases_from_state(state)
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
        crr, warnings = resolve_crr_aliases(
            crr,
            aliases,
            allowed_chunk_ids=available_chunk_ids(docs),
        )
        return crr, warnings, {
            "strategy": "single_pass_grounded_qa",
            "llm_calls": 1,
            "section_digests": 0,
            "evidence_aliases": len(aliases.alias_to_chunk),
        }

    def generate_transform(self, state: dict[str, Any]) -> tuple[CanonicalResponse, list[SectionDigest], list[str], dict[str, Any]]:
        config = self.config_from_state(state)
        groups = list(state.get("context_groups") or [])
        docs = list(state.get("retrieved_documents") or [])
        aliases = self.aliases_from_state(state)
        prompt_digests: list[SectionDigest] = []
        resolved_digests: list[SectionDigest] = []
        warnings: list[str] = []

        for group in groups:
            context = render_context(
                [group],
                max_chars=self.group_context_max_chars,
                chunk_to_alias=aliases.chunk_to_alias,
            )
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
            resolved, digest_warnings = resolve_digest_aliases(
                digest,
                aliases,
                allowed_chunk_ids=allowed,
            )
            warnings.extend(digest_warnings)
            # Synthesis still sees aliases, never storage IDs. Remove any invalid
            # aliases from the prompt digest before handing it downstream.
            prompt_copy = digest.model_copy(deep=True)
            for idx, claim in enumerate(prompt_copy.claims):
                claim.evidence = aliases.to_aliases(resolved.claims[idx].evidence)
            prompt_digests.append(prompt_copy)
            resolved_digests.append(resolved)

        data = self.llm.generate_json(
            system_prompt=crr_system_prompt(),
            user_prompt=synthesis_user_prompt(
                query=str(state.get("query") or ""),
                config=config,
                digests=[digest.model_dump(mode="json") for digest in prompt_digests],
            ),
            schema_name="canonical_response",
            json_schema=CanonicalResponse.model_json_schema(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        crr = CanonicalResponse.model_validate(data)
        crr, crr_warnings = resolve_crr_aliases(
            crr,
            aliases,
            allowed_chunk_ids=available_chunk_ids(docs),
        )
        warnings.extend(crr_warnings)
        return crr, resolved_digests, warnings, {
            "strategy": "hierarchical_section_digest_synthesis",
            "llm_calls": len(prompt_digests) + 1,
            "section_digests": len(prompt_digests),
            "evidence_aliases": len(aliases.alias_to_chunk),
        }
