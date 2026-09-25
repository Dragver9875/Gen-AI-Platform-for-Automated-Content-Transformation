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


_TEXT_SUFFIXES = {".txt", ".md"}
_DOCUMENT_SUFFIXES = {".pdf", ".ppt", ".pptx"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_DOC_LIKE_IMAGE_LABELS = {"document page", "screenshot"}


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
    ):
        self.docling = docling
        self.siglip = siglip
        self.vlm = vlm
        self.pdf_preflight = pdf_preflight
        self.enable_pdf_visual_fallback = enable_pdf_visual_fallback
        self.pdf_visual_fallback_max_pages = pdf_visual_fallback_max_pages

    def ingest(self, path: str | Path) -> IngestionResult:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        source_id = hashlib.sha1(file_path.read_bytes()).hexdigest()[:20]
        media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

        if suffix in _TEXT_SUFFIXES:
            text = normalize_text(file_path.read_text(encoding="utf-8", errors="replace"))
            return IngestionResult(
                source_id,
                file_path.name,
                media_type,
                "text-direct",
                [SourceElement("text-0", "paragraph", text, raw_text=text)],
            )
        if suffix in _DOCUMENT_SUFFIXES:
            return self._ingest_document(file_path, source_id, media_type)
        if suffix in _IMAGE_SUFFIXES:
            return self._ingest_image(file_path, source_id, media_type)
        raise ValueError(f"Unsupported input type: {suffix or media_type}")

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
                        prompt=(
                            "Describe this PDF page faithfully for retrieval. Preserve all visible text, labels, "
                            "numbers, chart trends, diagram relationships, and important visual content. "
                            "If it is primarily a scanned text page, transcribe the meaningful content."
                        ),
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
        predictions = self.siglip.classify(path)
        top_label = str(predictions[0]["label"]).lower() if predictions else "other visual"
        if top_label in _DOC_LIKE_IMAGE_LABELS:
            converted = self.docling.convert_file(path, do_ocr=True, force_ocr=True, enrich_pictures=True)
            markdown, doc_json = self.docling.document_payload(converted)
            elements = elements_from_docling_json(doc_json) if doc_json else elements_from_markdown(markdown)
            return IngestionResult(
                source_id,
                path.name,
                media_type,
                f"image-docling:{top_label}",
                elements,
                provider_metadata={"siglip": predictions},
            )

        description = self.vlm.describe_file(path)
        return IngestionResult(
            source_id,
            path.name,
            media_type,
            f"image-vlm:{top_label}",
            [SourceElement("visual-0", "visual_description", normalize_text(description), metadata={"siglip_label": top_label})],
            provider_metadata={"siglip": predictions},
        )
