from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArtifactRecord(BaseModel):
    artifact_id: str
    format: str
    media_type: str
    filename: str
    path: str | None = None
    status: Literal["generated", "source_only", "failed"] = "generated"
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TextArtifactIR(BaseModel):
    format: Literal["text"] = "text"
    content: str
    extension: str = ".txt"
    media_type: str = "text/plain"


class PDFArtifactIR(BaseModel):
    format: Literal["pdf"] = "pdf"
    title: str
    typst_source: str


class SlideIR(BaseModel):
    layout: str = "title_content"
    title: str = ""
    subtitle: str = ""
    bullets: list[str] = Field(default_factory=list)
    body: str = ""
    speaker_notes: str = ""
    visual_prompt: str = ""
    metrics: list[dict[str, str]] = Field(default_factory=list)


class PresentationIR(BaseModel):
    format: Literal["pptx"] = "pptx"
    title: str
    slides: list[SlideIR] = Field(default_factory=list)
    theme: dict[str, Any] = Field(default_factory=dict)


class SVGArtifactIR(BaseModel):
    format: Literal["svg"] = "svg"
    title: str = ""
    svg: str

    @field_validator("svg")
    @classmethod
    def validate_svg(cls, value: str) -> str:
        value = value.strip()
        if "<svg" not in value.lower() or "</svg>" not in value.lower():
            raise ValueError("SVG artifact must contain a complete <svg> element")
        return value


class CreativeImageIR(BaseModel):
    format: Literal["image"] = "image"
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)


ArtifactIR = TextArtifactIR | PDFArtifactIR | PresentationIR | SVGArtifactIR | CreativeImageIR
