from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PdfPageProfile:
    page: int
    text_chars: int
    image_coverage: float
    kind: str


@dataclass
class PdfProfile:
    pages: list[PdfPageProfile]

    @property
    def strategy(self) -> str:
        kinds = {page.kind for page in self.pages}
        if kinds == {"native"}:
            return "native-text"
        if kinds <= {"scanned", "visual"}:
            return "image-only/scanned"
        return "mixed"

    @property
    def visual_or_scanned_pages(self) -> list[int]:
        return [page.page for page in self.pages if page.kind in {"visual", "scanned", "mixed"}]


class PdfPreflight:
    def __init__(self, *, native_text_chars: int = 80, image_coverage_threshold: float = 0.72):
        self.native_text_chars = native_text_chars
        self.image_coverage_threshold = image_coverage_threshold

    def inspect(self, path: str | Path) -> PdfProfile:
        doc = fitz.open(path)
        profiles: list[PdfPageProfile] = []
        try:
            for page_number, page in enumerate(doc, start=1):
                text_chars = len((page.get_text("text") or "").strip())
                page_area = max(page.rect.width * page.rect.height, 1.0)
                image_area = 0.0
                for image in page.get_images(full=True):
                    xref = image[0]
                    for rect in page.get_image_rects(xref):
                        image_area += max(rect.width * rect.height, 0.0)
                coverage = min(image_area / page_area, 1.0)
                if text_chars >= self.native_text_chars and coverage < self.image_coverage_threshold:
                    kind = "native"
                elif text_chars < self.native_text_chars and coverage >= self.image_coverage_threshold:
                    kind = "scanned"
                elif coverage >= 0.25 and text_chars >= self.native_text_chars:
                    kind = "mixed"
                elif text_chars < self.native_text_chars:
                    kind = "visual"
                else:
                    kind = "native"
                profiles.append(PdfPageProfile(page_number, text_chars, coverage, kind))
        finally:
            doc.close()
        return PdfProfile(profiles)

    @staticmethod
    def render_page_png(path: str | Path, page_number: int, *, dpi: int = 144) -> bytes:
        doc = fitz.open(path)
        try:
            page = doc[page_number - 1]
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()
