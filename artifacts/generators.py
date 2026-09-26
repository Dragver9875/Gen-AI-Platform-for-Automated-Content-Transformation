from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, Type

from pydantic import BaseModel

from artifacts.models import (
    ArtifactRecord,
    CreativeImageIR,
    PDFArtifactIR,
    PresentationIR,
    SVGArtifactIR,
    TextArtifactIR,
)
from artifacts.prompts import artifact_system_prompt, image_prompt, pdf_prompt, pptx_prompt, svg_prompt, text_prompt
from artifacts.serializers import PPTXSerializer, SVGSerializer, TextSerializer, TypstPDFSerializer
from generation.content_ir import ContentIR
from generation.crr import TransformationConfig


class StructuredGenerator(Protocol):
    def generate_json(self, *, system_prompt: str, user_prompt: str, schema_name: str, json_schema: dict[str, Any], temperature: float, max_tokens: int) -> dict[str, Any]: ...


class ImageGenerator(Protocol):
    def generate(self, prompt: str, *, negative_prompt: str = "", width: int = 1024, height: int = 1024) -> tuple[bytes, str]: ...


def _artifact_id(content: ContentIR, fmt: str, salt: str = "") -> str:
    payload = json.dumps(content.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    raw = f"{payload}|{fmt}|{salt}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _base_name(content: ContentIR, artifact_id: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in content.title).strip("-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    return f"{(safe[:60] or 'artifact')}-{artifact_id}"


class _StructuredArtifactGenerator:
    schema: Type[BaseModel]
    schema_name: str
    prompt_builder: Any

    def __init__(self, llm: StructuredGenerator, *, temperature: float = 0.1, max_tokens: int = 4096):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def plan(self, content: ContentIR, config: TransformationConfig) -> BaseModel:
        data = self.llm.generate_json(
            system_prompt=artifact_system_prompt(),
            user_prompt=self.prompt_builder(content, config),
            schema_name=self.schema_name,
            json_schema=self.schema.model_json_schema(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self.schema.model_validate(data)


class TextArtifactGenerator(_StructuredArtifactGenerator):
    schema = TextArtifactIR
    schema_name = "text_artifact_ir"
    prompt_builder = staticmethod(text_prompt)

    def __init__(self, llm: StructuredGenerator, **kwargs):
        super().__init__(llm, **kwargs)
        self.serializer = TextSerializer()

    def generate(self, content: ContentIR, config: TransformationConfig, output_dir: Path, spec) -> ArtifactRecord:
        # Plain text is the final artifact itself, so do not force the model to
        # wrap it in JSON. Hosted providers frequently support ordinary text more
        # broadly than provider-specific structured-output modes. Keep the old
        # structured path as a compatibility fallback for custom/test providers.
        generate_text = getattr(self.llm, "generate_text", None)
        if callable(generate_text):
            text = generate_text(
                system_prompt=(
                    "You are a professional content transformation engine. "
                    "Use only facts present in the verified Content IR. Preserve exact numbers, dates, names, "
                    "and quoted terminology. Return only the final requested text artifact with no JSON wrapper, "
                    "no markdown code fence, and no meta-commentary."
                ),
                user_prompt=self.prompt_builder(content, config),
                temperature=max(self.temperature, 0.2),
                max_tokens=self.max_tokens,
            )
            ir = TextArtifactIR(
                content=text,
                extension=spec.extension or ".txt",
                media_type=spec.media_type or "text/plain",
            )
        else:
            ir = self.plan(content, config)

        artifact_id = _artifact_id(content, "text")
        destination = output_dir / f"{_base_name(content, artifact_id)}{spec.extension or ir.extension or '.txt'}"
        self.serializer.serialize(ir, destination)
        return ArtifactRecord(artifact_id=artifact_id, format="text", media_type=ir.media_type, filename=destination.name, path=str(destination))


class PDFArtifactGenerator(_StructuredArtifactGenerator):
    schema = PDFArtifactIR
    schema_name = "pdf_artifact_ir"
    prompt_builder = staticmethod(pdf_prompt)

    def __init__(self, llm: StructuredGenerator, *, typst_binary: str = "typst", retain_source: bool = True, **kwargs):
        super().__init__(llm, **kwargs)
        self.serializer = TypstPDFSerializer(typst_binary=typst_binary, retain_source=retain_source)

    def generate(self, content: ContentIR, config: TransformationConfig, output_dir: Path, spec) -> ArtifactRecord:
        ir = self.plan(content, config)
        artifact_id = _artifact_id(content, "pdf")
        destination = output_dir / f"{_base_name(content, artifact_id)}.pdf"
        pdf_path, source_path = self.serializer.serialize(ir, destination)
        if pdf_path is None:
            return ArtifactRecord(
                artifact_id=artifact_id, format="pdf", media_type="application/pdf", filename=destination.name,
                path=None, status="source_only", metadata={"typst_source_path": str(source_path), "reason": "Typst compiler not available"},
            )
        return ArtifactRecord(artifact_id=artifact_id, format="pdf", media_type="application/pdf", filename=pdf_path.name, path=str(pdf_path), metadata={"typst_source_path": str(source_path)})


class PPTXArtifactGenerator(_StructuredArtifactGenerator):
    schema = PresentationIR
    schema_name = "presentation_artifact_ir"
    prompt_builder = staticmethod(pptx_prompt)

    def __init__(self, llm: StructuredGenerator, **kwargs):
        super().__init__(llm, **kwargs)
        self.serializer = PPTXSerializer()

    def generate(self, content: ContentIR, config: TransformationConfig, output_dir: Path, spec) -> ArtifactRecord:
        ir = self.plan(content, config)
        artifact_id = _artifact_id(content, "pptx")
        destination = output_dir / f"{_base_name(content, artifact_id)}.pptx"
        self.serializer.serialize(ir, destination)
        return ArtifactRecord(artifact_id=artifact_id, format="pptx", media_type=spec.media_type, filename=destination.name, path=str(destination), metadata={"slides": len(ir.slides)})


class SVGArtifactGenerator(_StructuredArtifactGenerator):
    schema = SVGArtifactIR
    schema_name = "svg_artifact_ir"
    prompt_builder = staticmethod(svg_prompt)

    def __init__(self, llm: StructuredGenerator, **kwargs):
        super().__init__(llm, **kwargs)
        self.serializer = SVGSerializer()

    def generate(self, content: ContentIR, config: TransformationConfig, output_dir: Path, spec) -> ArtifactRecord:
        ir = self.plan(content, config)
        artifact_id = _artifact_id(content, "svg")
        destination = output_dir / f"{_base_name(content, artifact_id)}.svg"
        self.serializer.serialize(ir, destination)
        return ArtifactRecord(artifact_id=artifact_id, format="svg", media_type="image/svg+xml", filename=destination.name, path=str(destination))


class CreativeImageArtifactGenerator(_StructuredArtifactGenerator):
    schema = CreativeImageIR
    schema_name = "creative_image_ir"
    prompt_builder = staticmethod(image_prompt)

    def __init__(self, llm: StructuredGenerator, image_generator: ImageGenerator | None, **kwargs):
        super().__init__(llm, **kwargs)
        self.image_generator = image_generator

    def generate(self, content: ContentIR, config: TransformationConfig, output_dir: Path, spec) -> ArtifactRecord:
        if self.image_generator is None:
            raise RuntimeError("No image-generation provider is configured")
        ir = self.plan(content, config)
        image_bytes, media_type = self.image_generator.generate(ir.prompt, negative_prompt=ir.negative_prompt, width=ir.width, height=ir.height)
        extension = ".jpg" if media_type in {"image/jpeg", "image/jpg"} else ".webp" if media_type == "image/webp" else ".png"
        artifact_id = _artifact_id(content, "image")
        destination = output_dir / f"{_base_name(content, artifact_id)}{extension}"
        destination.write_bytes(image_bytes)
        return ArtifactRecord(artifact_id=artifact_id, format="image", media_type=media_type, filename=destination.name, path=str(destination), metadata={"width": ir.width, "height": ir.height, "prompt": ir.prompt})
