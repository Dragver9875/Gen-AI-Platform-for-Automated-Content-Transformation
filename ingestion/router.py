from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from app.schemas import IngestionResult, SourceElement
from ingestion.normalizer import normalize_text
from ingestion.registry import IngestionProcessorRegistry
from ingestion.renderers import render_pdf_pages, render_pptx_slides
from providers.vlm import VLMProvider


_TEXT_SUFFIXES = {".txt", ".md"}
_DOCUMENT_SUFFIXES = {".pdf", ".pptx"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


DEFAULT_DOCUMENT_PROMPT = (
    "You are the shared multimodal document encoder for a retrieval system. Read this page/slide faithfully and "
    "return retrieval-ready Markdown. Preserve headings, paragraphs, lists, tables, equations, labels, numbers, "
    "chart values/trends, diagram relationships, captions, and reading order. Describe meaningful non-text visuals. "
    "Do not summarize, omit facts, or invent content."
)

DEFAULT_IMAGE_PROMPT = (
    "You are the shared multimodal encoder for a retrieval system. Understand this image faithfully and return "
    "retrieval-ready Markdown. Preserve all visible text, entities, numbers, labels, spatial/causal relationships, "
    "chart or diagram semantics, and other information needed to answer questions about the image. If the image is "
    "a photographed/scanned document page, transcribe it rather than merely describing it. Do not invent details."
)


class IngestionRouter:
    """Single-path multimodal ingestion.

    * TXT/MD are read directly.
    * Every image goes through Qwen2.5-VL.
    * Every PDF page is rasterized and goes through Qwen2.5-VL.
    * Every PPTX slide is locally rasterized and goes through Qwen2.5-VL.

    There is deliberately no SigLIP, document/photo classifier, OCR-specific
    model, Granite Docling, or PDF preflight branch. The objective is a smaller,
    more reliable hackathon runtime with one shared multimodal understanding
    model and one downstream Content IR.
    """

    def __init__(
        self,
        *,
        vlm: VLMProvider,
        render_dpi: int = 144,
        max_pdf_pages: int = 40,
        max_pptx_slides: int = 40,
        document_prompt: str | None = None,
        image_prompt: str | None = None,
    ):
        self.vlm = vlm
        self.render_dpi = max(72, int(render_dpi))
        self.max_pdf_pages = max(0, int(max_pdf_pages))
        self.max_pptx_slides = max(0, int(max_pptx_slides))
        self.document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
        self.image_prompt = image_prompt or DEFAULT_IMAGE_PROMPT

        self.processors = IngestionProcessorRegistry()
        self.processors.register("text", _TEXT_SUFFIXES, self._ingest_text)
        self.processors.register("document", _DOCUMENT_SUFFIXES, self._ingest_document)
        self.processors.register("image", _IMAGE_SUFFIXES, self._ingest_image)

    def register_processor(self, name: str, suffixes: set[str], handler) -> None:
        self.processors.register(name, suffixes, handler)

    def ingest(self, path: str | Path) -> IngestionResult:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        source_id = hashlib.sha1(file_path.read_bytes()).hexdigest()[:20]
        media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        processor = self.processors.resolve(file_path)
        if processor is None:
            raise ValueError(
                f"Unsupported input type: {file_path.suffix.lower() or media_type}. "
                f"Supported suffixes: {', '.join(self.processors.supported_suffixes())}"
            )
        return processor.handler(file_path, source_id, media_type)

    def _ingest_text(self, file_path: Path, source_id: str, media_type: str) -> IngestionResult:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        text = normalize_text(raw)
        return IngestionResult(
            source_id,
            file_path.name,
            media_type,
            "text-direct",
            [SourceElement("text-0", "paragraph", text, raw_text=raw)],
            provider_metadata={"encoder": "direct-text"},
        )

    def _ingest_document(self, path: Path, source_id: str, media_type: str) -> IngestionResult:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._ingest_pdf(path, source_id, media_type)
        if suffix == ".pptx":
            return self._ingest_pptx(path, source_id, media_type)
        raise ValueError(f"Unsupported document input: {suffix}")

    def _ingest_pdf(self, path: Path, source_id: str, media_type: str) -> IngestionResult:
        import pymupdf

        with pymupdf.open(path) as document:
            total_pages = document.page_count

        rendered = render_pdf_pages(path, dpi=self.render_dpi, max_pages=self.max_pdf_pages)
        warnings: list[str] = []
        if len(rendered) < total_pages:
            warnings.append(
                f"PDF has {total_pages} pages; Qwen2.5-VL encoding was capped at {len(rendered)} page(s) by "
                "MULTIMODAL_MAX_PDF_PAGES. Set it to 0 for unlimited processing."
            )

        elements: list[SourceElement] = []
        models_used: set[str] = set()
        for page_number, png in rendered:
            encoded = normalize_text(
                self.vlm.describe_bytes(png, prompt=self.document_prompt, media_type="image/png")
            )
            if not encoded:
                warnings.append(f"Qwen2.5-VL returned empty content for PDF page {page_number}.")
                continue
            if getattr(self.vlm, "last_model_used", None):
                models_used.add(str(self.vlm.last_model_used))
            elements.append(
                SourceElement(
                    element_id=f"qwen-pdf-{page_number}",
                    kind="multimodal_page",
                    text=encoded,
                    page=page_number,
                    metadata={"processor": "qwen2.5-vl", "render_dpi": self.render_dpi},
                )
            )

        if not elements:
            raise RuntimeError("Qwen2.5-VL did not return usable content for any PDF page.")

        return IngestionResult(
            source_id,
            path.name,
            media_type,
            "qwen2.5-vl-pdf",
            elements,
            warnings=warnings,
            provider_metadata={
                "shared_encoder": self.vlm.model,
                "models_used": sorted(models_used),
                "total_pages": total_pages,
                "pages_encoded": len(elements),
                "render_dpi": self.render_dpi,
            },
        )

    def _ingest_pptx(self, path: Path, source_id: str, media_type: str) -> IngestionResult:
        rendered, total_slides = render_pptx_slides(path, max_slides=self.max_pptx_slides)
        warnings: list[str] = []
        if len(rendered) < total_slides:
            warnings.append(
                f"PPTX has {total_slides} slides; Qwen2.5-VL encoding was capped at {len(rendered)} slide(s) by "
                "MULTIMODAL_MAX_PPTX_SLIDES. Set it to 0 for unlimited processing."
            )

        elements: list[SourceElement] = []
        models_used: set[str] = set()
        for slide_number, png in rendered:
            encoded = normalize_text(
                self.vlm.describe_bytes(png, prompt=self.document_prompt, media_type="image/png")
            )
            if not encoded:
                warnings.append(f"Qwen2.5-VL returned empty content for PPTX slide {slide_number}.")
                continue
            if getattr(self.vlm, "last_model_used", None):
                models_used.add(str(self.vlm.last_model_used))
            elements.append(
                SourceElement(
                    element_id=f"qwen-pptx-{slide_number}",
                    kind="multimodal_slide",
                    text=encoded,
                    slide=slide_number,
                    metadata={"processor": "qwen2.5-vl", "renderer": "python-pptx+pillow"},
                )
            )

        if not elements:
            raise RuntimeError("Qwen2.5-VL did not return usable content for any PPTX slide.")

        return IngestionResult(
            source_id,
            path.name,
            media_type,
            "qwen2.5-vl-pptx",
            elements,
            warnings=warnings,
            provider_metadata={
                "shared_encoder": self.vlm.model,
                "models_used": sorted(models_used),
                "total_slides": total_slides,
                "slides_encoded": len(elements),
                "slide_renderer": "python-pptx+pillow",
            },
        )

    def _ingest_image(self, path: Path, source_id: str, media_type: str) -> IngestionResult:
        encoded = normalize_text(self.vlm.describe_file(path, prompt=self.image_prompt))
        if not encoded:
            raise RuntimeError("Qwen2.5-VL returned empty content for image input.")
        return IngestionResult(
            source_id,
            path.name,
            media_type,
            "qwen2.5-vl-image",
            [
                SourceElement(
                    "qwen-image-0",
                    "multimodal_image",
                    encoded,
                    metadata={"processor": "qwen2.5-vl"},
                )
            ],
            provider_metadata={
                "shared_encoder": self.vlm.model,
                "model_used": getattr(self.vlm, "last_model_used", None) or self.vlm.model,
            },
        )
