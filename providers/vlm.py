from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from providers.http import APIClient, ProviderError


class VLMProvider:
    """Generic hosted VLM adapter for a custom HTTP endpoint.

    Contract:
      request: {"image_base64": "...", "prompt": "..."}
      response: {"generated_text": "..."} or {"text": "..."} or HF-style list.
    """

    def __init__(self, api_url: str, api_key: str, *, timeout_s: float = 90.0, retries: int = 2):
        self.api_url = api_url
        self.api_key = api_key
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="vlm")

    def describe_file(self, path: str | Path, prompt: str | None = None) -> str:
        return self.describe_bytes(Path(path).read_bytes(), prompt=prompt)

    def describe_bytes(self, data: bytes, prompt: str | None = None) -> str:
        payload = {
            "image_base64": base64.b64encode(data).decode("ascii"),
            "prompt": prompt
            or "Describe the image faithfully for retrieval. Preserve visible text, entities, numbers, labels, relationships, and important visual context. Do not invent details.",
        }
        response = self.http.request(
            "POST",
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        return self._parse_text(response.json())

    @staticmethod
    def _parse_text(data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("generated_text", "text", "description", "output"):
                if isinstance(data.get(key), str):
                    return data[key]
        if isinstance(data, list) and data:
            return VLMProvider._parse_text(data[0])
        raise ProviderError("Unsupported VLM response shape")
