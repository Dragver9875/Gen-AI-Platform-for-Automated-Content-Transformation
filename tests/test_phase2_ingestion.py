from __future__ import annotations

from pathlib import Path

from ingestion.chunker import StructureAwareChunker
from ingestion.router import IngestionRouter
from app.schemas import IngestionResult, SourceElement


class FakeDocling:
    def convert_file(self, path, **kwargs):
        return {
            "status": "success",
            "document": {
                "md_content": "# Heading\n\nParsed document text.",
                "json_content": {},
            },
        }

    @staticmethod
    def document_payload(result):
        doc = result["document"]
        return doc["md_content"], doc["json_content"]


class FakeSiglip:
    def __init__(self, label="photograph"):
        self.label = label

    def classify(self, path):
        return [{"label": self.label, "score": 0.99}]


class FakeVLM:
    def describe_file(self, path, prompt=None):
        return "A rescue vehicle beside flood water."

    def describe_bytes(self, data, prompt=None):
        return "Scanned page containing an incident advisory."


class FakePdfPreflight:
    class Profile:
        strategy = "native-text"
        visual_or_scanned_pages = []
        pages = []

    def inspect(self, path):
        return self.Profile()

    @staticmethod
    def render_page_png(path, page_number):
        return b"png"


def test_text_ingestion_and_chunking(tmp_path: Path):
    file = tmp_path / "source.txt"
    file.write_text("Alpha beta gamma", encoding="utf-8")
    router = IngestionRouter(
        docling=FakeDocling(), siglip=FakeSiglip(), vlm=FakeVLM(), pdf_preflight=FakePdfPreflight()
    )
    result = router.ingest(file)
    assert result.strategy == "text-direct"
    chunks = StructureAwareChunker(target_chars=100, overlap_chars=0).chunk(result, user_id="u1", session_id="s1")
    assert len(chunks) == 1
    assert chunks[0].metadata["user_id"] == "u1"
    assert chunks[0].metadata["session_id"] == "s1"


def test_image_routes_to_vlm(tmp_path: Path):
    file = tmp_path / "image.png"
    file.write_bytes(b"not-a-real-png-needed-for-mock")
    router = IngestionRouter(
        docling=FakeDocling(), siglip=FakeSiglip("photograph"), vlm=FakeVLM(), pdf_preflight=FakePdfPreflight()
    )
    result = router.ingest(file)
    assert result.strategy.startswith("image-vlm")
    assert "flood water" in result.text


def test_document_like_image_routes_to_docling(tmp_path: Path):
    file = tmp_path / "scan.png"
    file.write_bytes(b"mock")
    router = IngestionRouter(
        docling=FakeDocling(), siglip=FakeSiglip("document page"), vlm=FakeVLM(), pdf_preflight=FakePdfPreflight()
    )
    result = router.ingest(file)
    assert result.strategy.startswith("image-docling")
    assert "Parsed document text" in result.text


def test_pdf_preflight_detects_full_page_image(tmp_path: Path):
    import base64
    import fitz
    from ingestion.preflight import PdfPreflight

    # 1x1 PNG inserted over the full page; enough to exercise image coverage routing.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=png)
    doc.save(pdf_path)
    doc.close()

    profile = PdfPreflight(native_text_chars=80, image_coverage_threshold=0.7).inspect(pdf_path)
    assert profile.strategy == "image-only/scanned"
    assert profile.pages[0].kind == "scanned"


def test_image_falls_back_to_vlm_routing_when_siglip_unavailable(tmp_path: Path):
    class BrokenSiglip:
        def classify(self, path):
            raise RuntimeError("provider unavailable")

    class RoutingVLM(FakeVLM):
        def classify_file(self, path, labels):
            return [{"label": "photograph", "score": 1.0}]

    image = tmp_path / "photo.png"
    image.write_bytes(b"fake-image")
    router = IngestionRouter(
        docling=FakeDocling(), siglip=BrokenSiglip(), vlm=RoutingVLM(), pdf_preflight=FakePdfPreflight()
    )
    result = router.ingest(image)
    assert result.strategy.startswith("image-vlm")
    assert result.provider_metadata["routing_source"] == "vlm_fallback"
    assert any("SigLIP routing unavailable" in warning for warning in result.warnings)
