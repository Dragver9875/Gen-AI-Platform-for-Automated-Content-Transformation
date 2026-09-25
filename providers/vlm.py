from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from providers.http import APIClient, ProviderError


class VLMProvider:
    """Hosted VLM adapter.

    Default ``openai`` mode targets Hugging Face's OpenAI-compatible multimodal
    chat route and therefore needs only ``HF_TOKEN`` + a model ID. ``custom``
    mode preserves the original simple HTTP contract for private endpoints.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        api_style: str = "openai",
        timeout_s: float = 90.0,
        retries: int = 2,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.api_style = api_style.lower()
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="vlm")

    def describe_file(self, path: str | Path, prompt: str | None = None) -> str:
        path = Path(path)
        return self.describe_bytes(path.read_bytes(), prompt=prompt, media_type=self._media_type_for_path(path))

    def describe_bytes(self, data: bytes, prompt: str | None = None, *, media_type: str | None = None) -> str:
        prompt = prompt or (
            "Describe the image faithfully for retrieval. Preserve visible text, entities, numbers, "
            "labels, relationships, and important visual context. Do not invent details."
        )
        if self.api_style == "openai":
            mime = media_type or self._detect_media_type(data)
            image_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 1200,
            }
        elif self.api_style == "custom":
            payload = {
                "image_base64": base64.b64encode(data).decode("ascii"),
                "prompt": prompt,
            }
        else:
            raise ProviderError(f"Unsupported VLM_API_STYLE: {self.api_style}")

        response = self.http.request(
            "POST",
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        return self._parse_text(response.json())

    def classify_file(self, path: str | Path, labels: list[str]) -> list[dict[str, float | str]]:
        allowed = ", ".join(labels)
        prompt = (
            "Classify this image into exactly one of the following labels: "
            f"{allowed}. Return only the chosen label and no other text."
        )
        raw = self.describe_file(path, prompt=prompt).strip().lower()
        # Prefer exact/full-string matches, then conservative substring matching.
        for label in labels:
            if raw == label.lower():
                return [{"label": label, "score": 1.0}]
        for label in labels:
            if label.lower() in raw:
                return [{"label": label, "score": 0.8}]
        return [{"label": "other visual", "score": 0.0}]

    @staticmethod
    def _parse_text(data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            # OpenAI-compatible response.
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    pieces = []
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            pieces.append(item["text"])
                    if pieces:
                        return "\n".join(pieces)
            for key in ("generated_text", "text", "description", "output"):
                if isinstance(data.get(key), str):
                    return data[key]
        if isinstance(data, list) and data:
            return VLMProvider._parse_text(data[0])
        raise ProviderError("Unsupported VLM response shape")

    @staticmethod
    def _media_type_for_path(path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }.get(suffix, "image/png")

    @staticmethod
    def _detect_media_type(data: bytes) -> str:
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
            return "image/webp"
        if data.startswith((b"II*\x00", b"MM\x00*")):
            return "image/tiff"
        return "image/png"
