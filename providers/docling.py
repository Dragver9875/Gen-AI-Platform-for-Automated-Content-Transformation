from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

import fitz
from pptx import Presentation

from providers.http import APIClient, ProviderError


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


class DoclingAPIProvider:
    """Hugging Face-compatible Granite Docling adapter.

    The original repository expected a standalone Docling Serve credential. This
    implementation instead uses IBM Granite Docling through a Hugging Face
    image-text-to-text/chat-completions endpoint and therefore authenticates with
    the shared ``HF_TOKEN``.

    Behaviour by input type:
      * PDF: render pages with PyMuPDF, parse each page with Granite Docling.
      * Image: parse the image with Granite Docling.
      * PPTX: extract slide text/tables deterministically with python-pptx. This
        avoids requiring LibreOffice or a separate conversion service.
      * legacy PPT: not supported by python-pptx; convert it to PPTX upstream.

    No model weights are loaded in the application process.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        model: str = "ibm-granite/granite-docling-258M",
        timeout_s: float = 180.0,
        retries: int = 2,
        render_dpi: int = 144,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.render_dpi = int(render_dpi)
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="docling")

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def convert_file(
        self,
        path: str | Path,
        *,
        do_ocr: bool = True,
        force_ocr: bool = False,
        table_mode: str = "accurate",
        enrich_pictures: bool = False,
    ) -> dict[str, Any]:
        # Parameters are retained for interface compatibility with the original
        # Docling Serve adapter. Granite Docling performs document-image parsing
        # directly, so OCR/table flags are advisory rather than separate switches.
        del do_ocr, force_ocr, table_mode, enrich_pictures

        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pages = self._convert_pdf(file_path)
        elif suffix == ".pptx":
            pages = self._convert_pptx(file_path)
        elif suffix == ".ppt":
            raise ProviderError(
                "Legacy .ppt input is not supported by the HF-only ingestion path. "
                "Convert it to .pptx before upload."
            )
        elif suffix in _IMAGE_SUFFIXES:
            pages = [(1, self._convert_image_bytes(file_path.read_bytes(), file_path.suffix or ".png"))]
        else:
            raise ProviderError(f"Unsupported Granite Docling input extension: {suffix}")

        markdown = "\n\n".join(f"<!-- page:{page_no} -->\n{text}" for page_no, text in pages if text.strip())
        texts = [
            {
                "self_ref": f"#/texts/{idx}",
                "label": "page",
                "text": text,
                "orig": text,
                "prov": [{"page_no": page_no}],
            }
            for idx, (page_no, text) in enumerate(pages)
            if text.strip()
        ]
        return {
            "status": "success",
            "document": {
                "md_content": markdown,
                "json_content": {"texts": texts},
            },
            "provider": "huggingface",
            "model": self.model,
            "processing_time": None,
        }

    def _convert_pdf(self, path: Path) -> list[tuple[int, str]]:
        document = fitz.open(path)
        pages: list[tuple[int, str]] = []
        try:
            for index in range(document.page_count):
                page = document.load_page(index)
                scale = self.render_dpi / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                png = pix.tobytes("png")
                pages.append((index + 1, self._convert_image_bytes(png, ".png")))
        finally:
            document.close()
        return pages

    def _convert_pptx(self, path: Path) -> list[tuple[int, str]]:
        presentation = Presentation(str(path))
        pages: list[tuple[int, str]] = []
        for slide_index, slide in enumerate(presentation.slides, start=1):
            blocks: list[str] = []
            title = getattr(slide.shapes, "title", None)
            if title is not None and getattr(title, "text", "").strip():
                blocks.append(f"# {title.text.strip()}")

            for shape in slide.shapes:
                if shape is title:
                    continue
                if getattr(shape, "has_text_frame", False):
                    text = getattr(shape, "text", "").strip()
                    if text:
                        blocks.append(text)
                if getattr(shape, "has_table", False):
                    table = shape.table
                    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                    if rows:
                        header = rows[0]
                        blocks.append("| " + " | ".join(header) + " |")
                        blocks.append("| " + " | ".join(["---"] * len(header)) + " |")
                        for row in rows[1:]:
                            blocks.append("| " + " | ".join(row) + " |")

            pages.append((slide_index, "\n\n".join(blocks).strip()))
        return pages

    def _convert_image_bytes(self, data: bytes, suffix: str = ".png") -> str:
        mime = self._mime_from_suffix(suffix)
        encoded = base64.b64encode(data).decode("ascii")
        prompt = (
            "Convert this document image into faithful Markdown for retrieval. "
            "Preserve headings, paragraphs, lists, tables, equations, code, labels, "
            "numbers and reading order. Do not summarize or invent content. Return Markdown only."
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        response = self.http.request("POST", self.api_url, headers=self.headers, json=payload)
        output = self._extract_chat_text(response.json())
        return self._strip_fence(output)

    @staticmethod
    def _extract_chat_text(data: Any) -> str:
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        texts = [
                            str(item.get("text", ""))
                            for item in content
                            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
                        ]
                        if texts:
                            return "\n".join(texts)
        if isinstance(data, dict):
            for key in ("generated_text", "text", "output_text", "content"):
                if isinstance(data.get(key), str):
                    return data[key]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return DoclingAPIProvider._extract_chat_text(data[0])
        raise ProviderError("Unsupported Granite Docling response shape")

    @staticmethod
    def _strip_fence(text: str) -> str:
        match = _FENCE_RE.match(text.strip())
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def _mime_from_suffix(suffix: str) -> str:
        suffix = suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }.get(suffix, "image/png")

    @staticmethod
    def document_payload(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        document = result.get("document") or {}
        markdown = document.get("md_content") or document.get("text_content") or ""
        json_content = document.get("json_content") or {}
        if isinstance(json_content, str):
            try:
                json_content = json.loads(json_content)
            except json.JSONDecodeError:
                json_content = {}
        return markdown, json_content
