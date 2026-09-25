from __future__ import annotations

import base64
from typing import Any

from providers.http import APIClient, ProviderError


class HostedImageGenerationProvider:
    """Hosted image-generation adapter.

    `api_url` is an exact POST endpoint. The adapter supports raw binary image
    responses and common JSON responses containing base64 image data. This keeps
    Phase 6 independent of a specific vendor while working with hosted FLUX-style
    endpoints.
    """

    def __init__(self, api_url: str, api_key: str, *, model: str | None = None, api_style: str = "hf", timeout_s: float = 180.0, retries: int = 2):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_style = api_style.lower()
        self.http = APIClient(timeout_s=timeout_s, retries=retries)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "image/*,application/json", "Content-Type": "application/json"}

    def generate(self, prompt: str, *, negative_prompt: str = "", width: int = 1024, height: int = 1024) -> tuple[bytes, str]:
        if self.api_style == "hf":
            payload: dict[str, Any] = {"inputs": prompt, "parameters": {"width": width, "height": height}}
            if negative_prompt:
                payload["parameters"]["negative_prompt"] = negative_prompt
            if self.model:
                payload["model"] = self.model
        elif self.api_style == "openai":
            payload = {"prompt": prompt, "size": f"{width}x{height}", "response_format": "b64_json"}
            if self.model:
                payload["model"] = self.model
        else:
            raise ProviderError(f"Unsupported IMAGE_GEN_API_STYLE: {self.api_style}")

        response = self.http.request("POST", self.api_url, headers=self.headers, json=payload)
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type.startswith("image/"):
            return response.content, content_type

        data = response.json()
        candidates: list[Any] = []
        if isinstance(data, dict):
            candidates.extend([data.get("b64_json"), data.get("image"), data.get("image_base64")])
            if isinstance(data.get("data"), list) and data["data"]:
                first = data["data"][0]
                if isinstance(first, dict):
                    candidates.extend([first.get("b64_json"), first.get("image"), first.get("image_base64")])
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                if candidate.startswith("data:image/") and "," in candidate:
                    header, candidate = candidate.split(",", 1)
                    media_type = header[5:].split(";")[0]
                else:
                    media_type = "image/png"
                try:
                    return base64.b64decode(candidate), media_type
                except Exception:
                    continue
        raise ProviderError("Image generation endpoint did not return image bytes or base64 image data")
