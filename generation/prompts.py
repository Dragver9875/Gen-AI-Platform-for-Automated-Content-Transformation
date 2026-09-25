from __future__ import annotations

import json
from typing import Any

from generation.crr import CanonicalResponse, SectionDigest, TransformationConfig


GROUNDING_RULES = """\
Use only the supplied source evidence. Do not introduce facts, numbers, names, dates,
causal claims, recommendations, or quotations that are not supported by that evidence.
Every factual claim must cite one or more CHUNK identifiers exactly as supplied.
If the evidence is insufficient, say so explicitly in the content and record the gap in
'insufficiencies'. Do not fabricate an evidence identifier.
"""


def crr_system_prompt() -> str:
    schema = json.dumps(CanonicalResponse.model_json_schema(), ensure_ascii=False)
    return (
        "You are the grounded content-transformation model in an enterprise workflow.\n"
        + GROUNDING_RULES
        + "\nReturn only valid JSON matching this JSON Schema exactly:\n"
        + schema
    )


def qa_user_prompt(*, query: str, config: TransformationConfig, context: str) -> str:
    return (
        "REQUEST TYPE: grounded question answering\n\n"
        f"USER REQUEST:\n{query.strip()}\n\n"
        f"TRANSFORMATION CONFIGURATION:\n{config.model_dump_json(indent=2)}\n\n"
        "SOURCE EVIDENCE:\n"
        f"{context}\n\n"
        "Create a concise but complete Canonical Response Representation. For QA, use "
        "artifact_type='answer' unless the configuration explicitly requests another artifact type."
    )


def digest_system_prompt() -> str:
    schema = json.dumps(SectionDigest.model_json_schema(), ensure_ascii=False)
    return (
        "You summarize one structural section of a source document for later synthesis.\n"
        + GROUNDING_RULES
        + "\nPreserve important numbers, dates, entities, qualifications, risks, and recommendations. "
        "Do not optimize for a final communication format yet.\n"
        "Return only valid JSON matching this JSON Schema exactly:\n"
        + schema
    )


def digest_user_prompt(*, source_id: str, section: str, context: str) -> str:
    return (
        f"SOURCE ID: {source_id}\nSECTION: {section}\n\n"
        "SECTION EVIDENCE:\n"
        f"{context}\n\n"
        "Produce a faithful structural digest for downstream whole-document synthesis."
    )


def synthesis_user_prompt(
    *, query: str, config: TransformationConfig, digests: list[dict[str, Any]]
) -> str:
    digest_json = json.dumps(digests, ensure_ascii=False, indent=2)
    return (
        "REQUEST TYPE: whole-document transformation\n\n"
        f"USER REQUEST:\n{query.strip()}\n\n"
        f"TRANSFORMATION CONFIGURATION:\n{config.model_dump_json(indent=2)}\n\n"
        "STRUCTURAL SOURCE DIGESTS:\n"
        f"{digest_json}\n\n"
        "Synthesize the requested artifact as a Canonical Response Representation. Preserve coverage "
        "across the supplied sections and only cite evidence IDs present in the digests."
    )
