from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from artifacts.models import ArtifactRecord
from artifacts.registry import ArtifactRegistry
from generation.content_ir import ContentIR
from generation.crr import CanonicalResponse, TransformationConfig


class ArtifactService:
    """Phase 6 artifact planner/serializer coordinator.

    The service is registry-driven; it contains no if/elif chain for PDF/PPTX/etc.
    LangGraph topology stays unchanged except for a single post-verification artifact node.
    """

    def __init__(self, registry: ArtifactRegistry, *, output_dir: str | Path = "artifacts", fail_fast: bool = False):
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.fail_fast = fail_fast

    def generate(self, state: dict[str, Any]) -> tuple[list[ArtifactRecord], list[str], dict[str, Any]]:
        crr = CanonicalResponse.model_validate(state.get("canonical_response") or {})
        config = TransformationConfig.model_validate(state.get("transformation_config") or {})
        content = ContentIR.from_crr(crr)
        def safe_component(value: str, fallback: str) -> str:
            cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
            return cleaned[:100] or fallback

        session_id = safe_component(str(state.get("session_id") or "session"), "session")
        user_id = safe_component(str(state.get("user_id") or "user"), "user")
        destination_root = self.output_dir / user_id / session_id
        destination_root.mkdir(parents=True, exist_ok=True)

        requested = config.output_formats or ["text"]
        records: list[ArtifactRecord] = []
        warnings: list[str] = []
        for requested_format in requested:
            try:
                spec, generator = self.registry.resolve(requested_format)
                records.append(generator.generate(content, config, destination_root, spec))
            except Exception as exc:
                if self.fail_fast:
                    raise
                warnings.append(f"Artifact '{requested_format}' failed: {exc}")
                records.append(ArtifactRecord(
                    artifact_id=f"failed-{requested_format}", format=str(requested_format), media_type="application/octet-stream",
                    filename="", path=None, status="failed", error=str(exc),
                ))
        generated = sum(record.status == "generated" for record in records)
        return records, warnings, {"requested": len(requested), "generated": generated, "failed": sum(record.status == "failed" for record in records), "source_only": sum(record.status == "source_only" for record in records)}
