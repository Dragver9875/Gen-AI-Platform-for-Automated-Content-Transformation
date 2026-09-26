from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from PIL import Image

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
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct:featherless-ai",
        api_style: str = "openai",
        timeout_s: float = 120.0,
        retries: int = 2,
        max_tokens: int = 2400,
        max_image_side: int | None = None,
        max_image_bytes: int | None = None,
        jpeg_quality: int | None = None,
        retry_image_side: int | None = None,
        retry_image_bytes: int | None = None,
        retry_jpeg_quality: int | None = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_style = api_style.lower()
        self.max_tokens = int(max_tokens)
        self.max_image_side = max(640, int(max_image_side or os.getenv("QWEN_VL_MAX_IMAGE_SIDE", "1800")))
        self.max_image_bytes = max(200_000, int(max_image_bytes or os.getenv("QWEN_VL_MAX_IMAGE_BYTES", "1200000")))
        self.jpeg_quality = min(95, max(50, int(jpeg_quality or os.getenv("QWEN_VL_JPEG_QUALITY", "82"))))
        self.retry_image_side = max(512, int(retry_image_side or os.getenv("QWEN_VL_RETRY_IMAGE_SIDE", "1280")))
        self.retry_image_bytes = max(150_000, int(retry_image_bytes or os.getenv("QWEN_VL_RETRY_IMAGE_BYTES", "650000")))
        self.retry_jpeg_quality = min(90, max(45, int(retry_jpeg_quality or os.getenv("QWEN_VL_RETRY_JPEG_QUALITY", "68"))))
        self.last_model_used: str | None = None
        self.last_payload_bytes: int | None = None
        self.last_payload_mime: str | None = None
        self.last_payload_reduced: bool = False
        self.last_413_retry: bool = False
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
        prepared, prepared_mime = self._prepare_image(
            data,
            mime,
            max_side=self.max_image_side,
            max_bytes=self.max_image_bytes,
            quality=self.jpeg_quality,
        )
        self.last_413_retry = False

        try:
            response = self._openai_request(prepared, prepared_mime, prompt)
        except ProviderError as exc:
            if exc.status_code != 413:
                raise
            # Provider gateways enforce request-body limits independently of
            # model context limits. Retry once with a substantially smaller
            # image rather than dropping the page/slide.
            retry_data, retry_mime = self._prepare_image(
                data,
                mime,
                max_side=self.retry_image_side,
                max_bytes=self.retry_image_bytes,
                quality=self.retry_jpeg_quality,
            )
            self.last_413_retry = True
            response = self._openai_request(retry_data, retry_mime, prompt)

        self.last_model_used = self.model
        text = self._parse_text(response.json()).strip()
        if not text:
            raise ProviderError(f"{self.model} returned an empty multimodal response")
        return text

    def _openai_request(self, data: bytes, mime: str, prompt: str):
        self.last_payload_bytes = len(data)
        self.last_payload_mime = mime
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
        return self.http.request(
            "POST",
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )

    def _prepare_image(
        self,
        data: bytes,
        media_type: str,
        *,
        max_side: int,
        max_bytes: int,
        quality: int,
    ) -> tuple[bytes, str]:
        """Bound multimodal request size while preserving document readability.

        Images are converted to white-backed RGB JPEG, downscaled only when
        needed, and iteratively compressed to a raw-byte budget. If PIL cannot
        decode the input, the original bytes are returned unchanged.
        """
        try:
            with Image.open(io.BytesIO(data)) as source:
                source.load()
                if source.mode in {"RGBA", "LA"} or (
                    source.mode == "P" and "transparency" in source.info
                ):
                    rgba = source.convert("RGBA")
                    image = Image.new("RGB", rgba.size, "white")
                    image.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    image = source.convert("RGB")

            original_size = image.size
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

            current_quality = quality
            encoded = b""
            for _ in range(10):
                output = io.BytesIO()
                image.save(
                    output,
                    format="JPEG",
                    quality=current_quality,
                    optimize=True,
                    progressive=True,
                )
                encoded = output.getvalue()
                if len(encoded) <= max_bytes:
                    break

                if current_quality > 55:
                    current_quality = max(55, current_quality - 8)
                    continue

                if max(image.size) <= 640:
                    break
                new_size = (
                    max(1, int(image.width * 0.82)),
                    max(1, int(image.height * 0.82)),
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            self.last_payload_reduced = (
                len(encoded) < len(data)
                or image.size != original_size
                or media_type.lower() not in {"image/jpeg", "image/jpg"}
            )
            return encoded, "image/jpeg"
        except Exception:
            self.last_payload_reduced = False
            return data, media_type

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
