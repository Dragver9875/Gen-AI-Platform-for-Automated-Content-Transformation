from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    generator: str
    aliases: tuple[str, ...] = ()
    media_type: str = "application/octet-stream"
    extension: str = ""
    options: dict[str, Any] | None = None


class ArtifactRegistry:
    """Configuration-driven artifact resolver.

    The LangGraph only asks the registry to build requested formats. New aliases
    or formats can be added through the spec file plus a generator plugin.
    """

    def __init__(self):
        self._specs: dict[str, ArtifactSpec] = {}
        self._generators: dict[str, Any] = {}

    def register_generator(self, name: str, generator: Any) -> None:
        self._generators[name] = generator

    def register_spec(self, spec: ArtifactSpec) -> None:
        for key in (spec.name, *spec.aliases):
            self._specs[key.lower()] = spec

    def load_specs(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in data.get("artifacts", data):
            self.register_spec(ArtifactSpec(
                name=str(item["name"]).lower(),
                generator=str(item["generator"]),
                aliases=tuple(str(x).lower() for x in item.get("aliases", [])),
                media_type=str(item.get("media_type") or "application/octet-stream"),
                extension=str(item.get("extension") or ""),
                options=dict(item.get("options") or {}),
            ))

    def resolve(self, requested_format: str) -> tuple[ArtifactSpec, Any]:
        key = requested_format.strip().lower()
        spec = self._specs.get(key)
        if spec is None:
            raise KeyError(f"Unsupported artifact format '{requested_format}'. Registered formats: {', '.join(self.formats())}")
        generator = self._generators.get(spec.generator)
        if generator is None:
            raise KeyError(f"Artifact generator plugin '{spec.generator}' is not registered")
        return spec, generator

    def formats(self) -> list[str]:
        return sorted({spec.name for spec in self._specs.values()})
