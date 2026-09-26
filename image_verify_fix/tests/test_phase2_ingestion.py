from __future__ import annotations

import base64
from pathlib import Path

import pymupdf
from pptx import Presentation

from app.schemas import IngestionResult, SourceElement
from ingestion.chunker import StructureAwareChunker
from ingestion.preflight import PdfPreflight
from ingestion.router import IngestionRouter


class FakeSiglip:
    def __init__(self, label="photograph"):
        self.label = label

    def classify(self, path):
        return [{"label": self.label, "score": 0.99}]


class FakeVLM:
    def __init__(self):
        self.byte_calls = 0
        self.file_calls = 0

    def describe_file(self, path, prompt=None):
        self.file_calls += 1
        if prompt and "Transcribe" in prompt:
            return "Incident advisory: road closed until 18:00."
        return "A rescue vehicle beside flood water."

    def describe_bytes(self, data, prompt=None, *, media_type=None):
        self.byte_calls += 1
        return "Scanned page containing an incident advisory."

    def classify_file(self, path, labels):
        return [{"label": "photograph", "score": 1.0}]


_DEFAULT_SIGLIP = object()

def make_router(*, siglip=_DEFAULT_SIGLIP, vlm=None):
    return IngestionRouter(
        siglip=FakeSiglip() if siglip is _DEFAULT_SIGLIP else siglip,
        vlm=vlm or FakeVLM(),
        pdf_preflight=PdfPreflight(native_text_chars=40, image_coverage_threshold=0.7),
    )


def test_text_ingestion_and_chunking(tmp_path: Path):
    file = tmp_path / "source.txt"
    file.write_text("Alpha beta gamma", encoding="utf-8")
    result = make_router().ingest(file)
    assert result.strategy == "text-direct"
    chunks = StructureAwareChunker(target_chars=100, overlap_chars=0).chunk(result, user_id="u1", session_id="s1")
    assert len(chunks) == 1
    assert chunks[0].metadata["user_id"] == "u1"
    assert chunks[0].metadata["session_id"] == "s1"


def test_image_routes_to_vlm(tmp_path: Path):
    file = tmp_path / "image.png"
    file.write_bytes(b"not-a-real-png-needed-for-mock")
    result = make_router(siglip=FakeSiglip("photograph")).ingest(file)
    assert result.strategy.startswith("image-visual:photograph")
    assert "flood water" in result.text


def test_document_like_image_routes_to_vlm_document_prompt(tmp_path: Path):
    file = tmp_path / "scan.png"
    file.write_bytes(b"mock")
    result = make_router(siglip=FakeSiglip("document page")).ingest(file)
    assert result.strategy.startswith("image-document:document page")
    assert "road closed" in result.text


def test_pdf_preflight_detects_full_page_image(tmp_path: Path):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    pdf_path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=png)
    doc.save(pdf_path)
    doc.close()

    profile = PdfPreflight(native_text_chars=80, image_coverage_threshold=0.7).inspect(pdf_path)
    assert profile.strategy == "image-only/scanned"
    assert profile.pages[0].kind == "scanned"


def test_native_pdf_uses_local_text_without_vlm(tmp_path: Path):
    pdf_path = tmp_path / "native.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a native PDF paragraph with enough searchable text to exceed the threshold and remain local.")
    doc.save(pdf_path)
    doc.close()

    vlm = FakeVLM()
    result = make_router(vlm=vlm).ingest(pdf_path)
    assert result.strategy == "pdf-native-text"
    assert "native PDF paragraph" in result.text
    assert vlm.byte_calls == 0
    assert result.provider_metadata["document_parser"] == "pymupdf+vlm"
    assert result.provider_metadata["vlm_pages"] == 0


def test_scanned_pdf_uses_vlm_page_analysis(tmp_path: Path):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    pdf_path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=png)
    doc.save(pdf_path)
    doc.close()

    vlm = FakeVLM()
    result = make_router(vlm=vlm).ingest(pdf_path)
    assert result.strategy == "pdf-image-only/scanned"
    assert "incident advisory" in result.text
    assert vlm.byte_calls == 1
    assert result.provider_metadata["vlm_pages"] == 1


def test_pptx_extracts_text_locally(tmp_path: Path):
    path = tmp_path / "deck.pptx"
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Quarterly Update"
    slide.placeholders[1].text = "Project finished today."
    prs.save(path)

    vlm = FakeVLM()
    result = make_router(vlm=vlm).ingest(path)
    assert result.strategy == "pptx-native+visual"
    assert "Quarterly Update" in result.text
    assert "Project finished today." in result.text
    assert vlm.byte_calls == 0


def test_image_falls_back_to_vlm_routing_when_siglip_unavailable(tmp_path: Path):
    class BrokenSiglip:
        def classify(self, path):
            raise RuntimeError("provider unavailable")

    image = tmp_path / "photo.png"
    image.write_bytes(b"fake-image")
    result = make_router(siglip=BrokenSiglip()).ingest(image)
    assert result.strategy.startswith("image-")
    assert result.provider_metadata["routing_source"] == "vlm_fallback"
    assert any("SigLIP routing unavailable" in warning for warning in result.warnings)


def test_pdf_page_photo_filename_precheck_routes_as_document(tmp_path: Path):
    # Real white page-like image with dark text-like strokes. The filename mirrors
    # the user's failing case and should bypass generic-image routing.
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 700, 1000), 0)
    pix.clear_with(255)
    path = tmp_path / "pdf2photo.jpeg"
    pix.save(path)

    vlm = FakeVLM()
    result = make_router(siglip=FakeSiglip("photograph"), vlm=vlm).ingest(path)

    assert result.strategy.startswith("image-document:document page")
    assert result.provider_metadata["routing_source"] == "document_precheck"
    assert result.provider_metadata["document_precheck"]["is_document_like"] is True
    assert "road closed" in result.text


def test_generic_filename_page_photo_detected_from_layout(tmp_path: Path):
    path = tmp_path / "IMG_0001.jpeg"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for index in range(30):
        page.insert_text(
            (50, 60 + index * 22),
            f"Sample report line {index}: project status and metric 12345.",
            fontsize=11,
        )
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2), alpha=False)
    pix.save(path)
    doc.close()

    result = make_router(siglip=FakeSiglip("photograph"), vlm=FakeVLM()).ingest(path)
    assert result.strategy.startswith("image-document:document page")
    assert result.provider_metadata["routing_source"] == "document_precheck"


def test_siglip_disabled_routes_with_vlm_without_warning(tmp_path: Path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"fake-image")
    result = make_router(siglip=None, vlm=FakeVLM()).ingest(image)
    assert result.strategy.startswith("image-visual:photograph")
    assert result.provider_metadata["routing_source"] == "vlm_router"
    assert not any("SigLIP" in warning for warning in result.warnings)
