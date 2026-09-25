from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any

from app.schemas import IngestionResult, SourceElement
from ingestion.docling_parser import elements_from_docling_json, elements_from_markdown
from ingestion.normalizer import normalize_text
from ingestion.preflight import PdfPreflight
from providers.docling import DoclingAPIProvider
from providers.siglip import SigLIPRoutingProvider
from providers.vlm import VLMProvider
from ingestion.registry import IngestionProcessorRegistry


_TEXT_SUFFIXES = {".txt", ".md"}
_DOCUMENT_SUFFIXES = {".pdf", ".ppt", ".pptx"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_VISUAL_LABELS = ("document page", "screenshot", "chart", "diagram", "map", "photograph", "other visual")
class IngestionRouter:
    def __init__(
        self,
        *,
        docling: DoclingAPIProvider,
        siglip: SigLIPRoutingProvider,
        vlm: VLMProvider,
        pdf_preflight: PdfPreflight,
        enable_pdf_visual_fallback: bool = True,
        pdf_visual_fallback_max_pages: int = 12,
        document_like_image_labels: set[str] | None = None,
        pdf_visual_prompt: str | None = None,
    ):
        self.docling = docling
        self.siglip = siglip
        self.vlm = vlm
        self.pdf_preflight = pdf_preflight
        self.enable_pdf_visual_fallback = enable_pdf_visual_fallback
        self.pdf_visual_fallback_max_pages = pdf_visual_fallback_max_pages
        self.document_like_image_labels = {x.strip().lower() for x in (document_like_image_labels or {"document page", "screenshot"}) if x.strip()}
        self.pdf_visual_prompt = pdf_visual_prompt or (
            "Describe this PDF page faithfully for retrieval. Preserve all visible text, labels, "
            "numbers, chart trends, diagram relationships, and important visual content. "
            "If it is primarily a scanned text page, transcribe the meaningful content."
        )
        self.processors = IngestionProcessorRegistry()
        self.processors.register("text", _TEXT_SUFFIXES, self._ingest_text)
        self.processors.register("document", _DOCUMENT_SUFFIXES, self._ingest_document)
        self.processors.register("image", _IMAGE_SUFFIXES, self._ingest_image)

    def register_processor(self, name: str, suffixes: set[str], handler) -> None:
        self.processors.register(name, suffixes, handler)

    def ingest(self, path: str | Path) -> IngestionResult:
        file_path = Path(path)
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
        text = normalize_text(file_path.read_text(encoding="utf-8", errors="replace"))
        return IngestionResult(
            source_id, file_path.name, media_type, "text-direct",
            [SourceElement("text-0", "paragraph", text, raw_text=text)],
        )

    def _ingest_document(self, path: Path, source_id: str, media_type: str) -> IngestionResult:
        warnings: list[str] = []
        strategy = "docling-document"
        do_ocr = True
        force_ocr = False
        visual_pages: list[int] = []
        profile_metadata: dict[str, Any] = {}

        if path.suffix.lower() == ".pdf":
            profile = self.pdf_preflight.inspect(path)
            strategy = f"pdf-{profile.strategy}"
            profile_metadata["page_profiles"] = [p.__dict__ for p in profile.pages]
            if profile.strategy == "native-text":
                do_ocr = False
            elif profile.strategy == "image-only/scanned":
                do_ocr = True
                force_ocr = True
            else:
                do_ocr = True
            visual_pages = profile.visual_or_scanned_pages[: self.pdf_visual_fallback_max_pages]

        converted = self.docling.convert_file(
            path,
            do_ocr=do_ocr,
            force_ocr=force_ocr,
            enrich_pictures=True,
        )
        markdown, doc_json = self.docling.document_payload(converted)
        elements = elements_from_docling_json(doc_json) if doc_json else []
        if not elements:
            elements = elements_from_markdown(markdown)

        if path.suffix.lower() == ".pptx" or path.suffix.lower() == ".ppt":
            for element in elements:
                if element.page is not None:
                    element.slide = element.page
                    element.page = None

        if path.suffix.lower() == ".pdf" and self.enable_pdf_visual_fallback and visual_pages:
            for page_number in visual_pages:
                try:
                    png = self.pdf_preflight.render_page_png(path, page_number)
                    description = self.vlm.describe_bytes(
                        png,
                        prompt=self.pdf_visual_prompt,
                    )
                    if description.strip():
                        elements.append(
                            SourceElement(
                                element_id=f"visual-page-{page_number}",
                                kind="visual_description",
                                text=normalize_text(description),
                                page=page_number,
                                metadata={"processor": "vlm_fallback"},
                            )
                        )
                except Exception as exc:  # best-effort fallback; Docling result remains usable
                    warnings.append(f"Visual fallback failed on page {page_number}: {exc}")

        return IngestionResult(
            source_id,
            path.name,
            media_type,
            strategy,
            elements,
            warnings=warnings,
            provider_metadata={
                "docling_status": converted.get("status"),
                "docling_processing_time": converted.get("processing_time"),
                **profile_metadata,
            },
        )

    def _ingest_image(self, path: Path, source_id: str, media_type: str) -> IngestionResult:
        warnings: list[str] = []
        routing_source = "siglip"
        try:
            predictions = self.siglip.classify(path)
        except Exception as exc:
            # HF serverless availability varies by model/provider. Visual routing must
            # not make ingestion unavailable, so fall back to the already configured VLM.
            warnings.append(f"SigLIP routing unavailable; used VLM fallback: {exc}")
            predictions = self.vlm.classify_file(path, list(DEFAULT_VISUAL_LABELS))
            routing_source = "vlm_fallback"

        top_label = str(predictions[0]["label"]).lower() if predictions else "other visual"
        if top_label in self.document_like_image_labels:
            converted = self.docling.convert_file(path, do_ocr=True, force_ocr=True, enrich_pictures=True)
            markdown, doc_json = self.docling.document_payload(converted)
            elements = elements_from_docling_json(doc_json) if doc_json else elements_from_markdown(markdown)
            return IngestionResult(
                source_id,
                path.name,
                media_type,
                f"image-docling:{top_label}",
                elements,
                warnings=warnings,
                provider_metadata={"visual_routing": predictions, "routing_source": routing_source},
            )

        description = self.vlm.describe_file(path)
        return IngestionResult(
            source_id,
            path.name,
            media_type,
            f"image-vlm:{top_label}",
            [SourceElement("visual-0", "visual_description", normalize_text(description), metadata={"visual_route_label": top_label})],
            warnings=warnings,
            provider_metadata={"visual_routing": predictions, "routing_source": routing_source},
        )
