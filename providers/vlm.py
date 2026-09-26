from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from providers.http import APIClient, ProviderError


class VLMProvider:
    """Qwen2.5-VL shared multimodal understanding adapter.

    Images, rendered PDF pages, and rendered PPTX slides all pass through this
    one provider. The default endpoint is Hugging Face's OpenAI-compatible
    multimodal router. No local model weights are loaded.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        api_style: str = "openai",
        timeout_s: float = 120.0,
        retries: int = 2,
        max_tokens: int = 2400,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_style = api_style.lower()
        self.max_tokens = int(max_tokens)
        self.last_model_used: str | None = None
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="qwen_vl")

    def describe_file(self, path: str | Path, prompt: str | None = None) -> str:
        path = Path(path)
        return self.describe_bytes(path.read_bytes(), prompt=prompt, media_type=self._media_type_for_path(path))

    def describe_bytes(self, data: bytes, prompt: str | None = None, *, media_type: str | None = None) -> str:
        prompt = prompt or (
            "Understand this visual input faithfully for retrieval. Preserve visible text, entities, numbers, "
            "labels, relationships, charts, diagrams and relevant visual context. If it is a document page, "
            "transcribe it. Do not invent details."
        )

        if self.api_style == "custom":
            payload = {
                "model": self.model,
                "image_base64": base64.b64encode(data).decode("ascii"),
                "prompt": prompt,
            }
            response = self.http.request(
                "POST",
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            self.last_model_used = self.model
            return self._parse_text(response.json())

        if self.api_style != "openai":
            raise ProviderError(f"Unsupported MULTIMODAL_API_STYLE: {self.api_style}")

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
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        response = self.http.request(
            "POST",
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        self.last_model_used = self.model
        text = self._parse_text(response.json()).strip()
        if not text:
            raise ProviderError(f"{self.model} returned an empty multimodal response")
        return text

    @staticmethod
    def _parse_text(data: Any) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    pieces: list[str] = []
                    for item in content:
                        if isinstance(item, str):
                            pieces.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("text"), str):
                            pieces.append(item["text"])
                    if pieces:
                        return "\n".join(pieces)
            for key in ("generated_text", "text", "description", "output", "output_text"):
                if isinstance(data.get(key), str):
                    return data[key]
        if isinstance(data, list) and data:
            return VLMProvider._parse_text(data[0])
        raise ProviderError("Unsupported Qwen2.5-VL response shape")

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
