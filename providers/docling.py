from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from providers.http import APIClient, ProviderError


_FORMAT_BY_SUFFIX = {
    ".pdf": "pdf",
    ".pptx": "pptx",
    ".ppt": "ppt",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
}


class DoclingAPIProvider:
    """Remote Docling/docling-serve adapter using /v1/convert/file."""

    def __init__(self, base_url: str, api_key: str | None = None, *, timeout_s: float = 180.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.http = APIClient(timeout_s=timeout_s, retries=retries)

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key} if self.api_key else {}

    def convert_file(
        self,
        path: str | Path,
        *,
        do_ocr: bool = True,
        force_ocr: bool = False,
        table_mode: str = "accurate",
        enrich_pictures: bool = False,
    ) -> dict[str, Any]:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        from_format = _FORMAT_BY_SUFFIX.get(suffix)
        if not from_format:
            raise ProviderError(f"Unsupported Docling input extension: {suffix}")

        mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        data: list[tuple[str, str]] = [
            ("from_formats", from_format),
            ("to_formats", "md"),
            ("to_formats", "json"),
            ("do_ocr", str(do_ocr).lower()),
            ("force_ocr", str(force_ocr).lower()),
            ("table_mode", table_mode),
            ("image_export_mode", "embedded"),
        ]
        if enrich_pictures:
            data.extend([
                ("do_picture_classification", "true"),
                ("do_picture_description", "true"),
                ("do_chart_extraction", "true"),
            ])

        with file_path.open("rb") as fh:
            files = {"files": (file_path.name, fh, mime)}
            response = self.http.request(
                "POST",
                f"{self.base_url}/v1/convert/file",
                headers=self.headers,
                data=data,
                files=files,
            )
        result = response.json()
        if result.get("status") not in {None, "success", "partial_success"}:
            raise ProviderError(f"Docling conversion failed: {result.get('errors') or result}")
        return result

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
