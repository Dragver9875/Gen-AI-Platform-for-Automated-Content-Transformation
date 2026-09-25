from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SourceElement:
    element_id: str
    kind: str
    text: str
    page: int | None = None
    slide: int | None = None
    section: str | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionResult:
    source_id: str
    filename: str
    media_type: str
    strategy: str
    elements: list[SourceElement]
    warnings: list[str] = field(default_factory=list)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(e.text for e in self.elements if e.text.strip())


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    raw_text: str | None = None

    def to_retrieval_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "raw_text": self.raw_text or self.text,
            "metadata": self.metadata,
        }
