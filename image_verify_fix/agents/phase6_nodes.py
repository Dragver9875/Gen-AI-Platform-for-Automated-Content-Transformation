from __future__ import annotations

from typing import Any

from agents.state import AgentState
from artifacts.service import ArtifactService


class Phase6Nodes:
    def __init__(self, artifacts: ArtifactService):
        self.artifacts = artifacts

    def generate_artifacts(self, state: AgentState) -> dict[str, Any]:
        try:
            records, warnings, metadata = self.artifacts.generate(state)
            return {
                "artifacts": [record.model_dump(mode="json") for record in records],
                "artifact_metadata": metadata,
                "warnings": list(dict.fromkeys(list(state.get("warnings") or []) + warnings)),
                "status": "phase6_complete" if not any(record.status == "failed" for record in records) else "phase6_complete_with_artifact_errors",
            }
        except Exception as exc:
            return {
                "status": "error",
                "errors": list(state.get("errors") or []) + [f"Phase 6 artifact generation failed: {exc}"],
            }
