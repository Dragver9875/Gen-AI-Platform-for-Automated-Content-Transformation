from __future__ import annotations

from typing import Any

from agents.state import AgentState
from generation.service import GenerationService


class Phase4Nodes:
    def __init__(self, generator: GenerationService):
        self.generator = generator

    def generate_qa(self, state: AgentState) -> dict[str, Any]:
        try:
            crr, warnings, metadata = self.generator.generate_qa(state)
            return {
                "canonical_response": crr.model_dump(mode="json"),
                "section_digests": [],
                "generation_metadata": metadata,
                "warnings": list(dict.fromkeys(list(state.get("warnings") or []) + warnings)),
                "status": "phase4_complete",
            }
        except Exception as exc:
            return {
                "status": "error",
                "errors": list(state.get("errors") or []) + [f"Phase 4 QA generation failed: {exc}"],
            }

    def generate_transform(self, state: AgentState) -> dict[str, Any]:
        try:
            crr, digests, warnings, metadata = self.generator.generate_transform(state)
            return {
                "canonical_response": crr.model_dump(mode="json"),
                "section_digests": [digest.model_dump(mode="json") for digest in digests],
                "generation_metadata": metadata,
                "warnings": list(dict.fromkeys(list(state.get("warnings") or []) + warnings)),
                "status": "phase4_complete",
            }
        except Exception as exc:
            return {
                "status": "error",
                "errors": list(state.get("errors") or []) + [f"Phase 4 transformation generation failed: {exc}"],
            }
