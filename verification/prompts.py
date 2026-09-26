from __future__ import annotations

import json
from typing import Any

from generation.crr import CanonicalResponse, TransformationConfig
from generation.prompts import GROUNDING_RULES
from verification.models import SemanticVerificationBatch, VerificationReport


def verifier_system_prompt() -> str:
    schema = json.dumps(SemanticVerificationBatch.model_json_schema(), ensure_ascii=False)
    return (
        "You are a strict evidence-entailment verifier. Evaluate each generated claim only against "
        "the evidence supplied for that claim. Do not use outside knowledge.\n"
        "Statuses:\n"
        "- supported: all material factual content follows from the cited evidence.\n"
        "- partially_supported: some material content is supported but another material part is not established.\n"
        "- unsupported: the evidence contradicts the claim or fails to support a material assertion.\n"
        "- insufficient_evidence: there is no usable cited evidence to assess the claim.\n"
        "Treat altered numbers, dates, named entities, causal statements, quotations, and recommendations as material. "
        "Return concise rationales and a source-faithful correction when possible.\n"
        "Return only valid JSON matching this JSON Schema exactly:\n"
        + schema
    )


def verifier_user_prompt(items: list[dict[str, Any]]) -> str:
    return (
        "Verify every claim below. Each item contains only the evidence that the generated artifact cited. "
        "Do not infer support from omitted source material.\n\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


def repair_system_prompt() -> str:
    schema = json.dumps(CanonicalResponse.model_json_schema(), ensure_ascii=False)
    return (
        "You repair a Canonical Response Representation after factuality verification.\n"
        + GROUNDING_RULES
        + "\nCorrect, qualify, or remove every claim marked partially_supported, unsupported, or insufficient_evidence. "
        "Preserve supported content and the requested artifact structure/tone. Never invent replacement facts. "
        "If evidence cannot support a requested fact, explicitly record the gap in insufficiencies.\n"
        "Return only valid JSON matching this JSON Schema exactly:\n"
        + schema
    )


def repair_user_prompt(
    *,
    query: str,
    config: TransformationConfig,
    crr: CanonicalResponse | dict[str, Any],
    report: VerificationReport | dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    crr_json = (
        crr.model_dump_json(indent=2)
        if hasattr(crr, "model_dump_json")
        else json.dumps(crr, ensure_ascii=False, indent=2)
    )
    report_json = (
        report.model_dump_json(indent=2)
        if hasattr(report, "model_dump_json")
        else json.dumps(report, ensure_ascii=False, indent=2)
    )
    return (
        f"USER REQUEST:\n{query.strip()}\n\n"
        f"TRANSFORMATION CONFIGURATION:\n{config.model_dump_json(indent=2)}\n\n"
        f"CURRENT CRR:\n{crr_json}\n\n"
        f"VERIFICATION REPORT:\n{report_json}\n\n"
        "SOURCE EVIDENCE FOR REPAIR:\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
        + "\n\nReturn the repaired CRR only."
    )
