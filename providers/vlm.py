from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Iterable

from providers.http import APIClient, ProviderError


class VLMProvider:
    """Hosted vision-language adapter with provider-availability fallback.

    The default path targets Hugging Face's OpenAI-compatible multimodal router.
    Hub model presence does not guarantee that an Inference Provider serves a
    checkpoint, so callers may configure fallback model IDs. We only fail over
    on explicit model/provider availability errors; auth, quota, malformed input,
    and other failures remain visible to the caller.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        fallback_models: Iterable[str] | None = None,
        api_style: str = "openai",
        timeout_s: float = 90.0,
        retries: int = 2,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.fallback_models = tuple(
            candidate.strip()
            for candidate in (fallback_models or ())
            if candidate and candidate.strip() and candidate.strip() != model
        )
        self.api_style = api_style.lower()
        self.last_model_used: str | None = None
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="vlm")

    @property
    def candidate_models(self) -> tuple[str, ...]:
        return (self.model, *self.fallback_models)

    def describe_file(self, path: str | Path, prompt: str | None = None) -> str:
        path = Path(path)
        return self.describe_bytes(path.read_bytes(), prompt=prompt, media_type=self._media_type_for_path(path))

    def describe_bytes(self, data: bytes, prompt: str | None = None, *, media_type: str | None = None) -> str:
        prompt = prompt or (
            "Describe the image faithfully for retrieval. Preserve visible text, entities, numbers, "
            "labels, relationships, and important visual context. Do not invent details."
        )
        if self.api_style == "custom":
            payload = {
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
            raise ProviderError(f"Unsupported VLM_API_STYLE: {self.api_style}")

        mime = media_type or self._detect_media_type(data)
        image_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        last_error: ProviderError | None = None

        for model in self.candidate_models:
            payload = {
                "model": model,
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
                "max_tokens": 1600,
            }
            try:
                response = self.http.request(
                    "POST",
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                self.last_model_used = model
                return self._parse_text(response.json())
            except ProviderError as exc:
                last_error = exc
                if not self._is_model_availability_error(exc):
                    raise

        models = ", ".join(self.candidate_models)
        raise ProviderError(
            f"None of the configured VLM models are currently available from the provider: {models}. "
            f"Last error: {last_error}",
            status_code=getattr(last_error, "status_code", None),
            response_text=getattr(last_error, "response_text", None),
        )

    def classify_file(self, path: str | Path, labels: list[str]) -> list[dict[str, float | str]]:
        allowed = ", ".join(labels)
        prompt = (
            "Classify this image into exactly one of the following labels: "
            f"{allowed}. Return only the chosen label and no other text."
        )
        raw = self.describe_file(path, prompt=prompt).strip().lower()
        for label in labels:
            if raw == label.lower():
                return [{"label": label, "score": 1.0}]
        for label in labels:
            if label.lower() in raw:
                return [{"label": label, "score": 0.8}]
        return [{"label": "other visual", "score": 0.0}]

    @staticmethod
    def _is_model_availability_error(exc: ProviderError) -> bool:
        if exc.status_code not in {400, 404, 422, 503}:
            return False
        body = (exc.response_text or str(exc)).lower()
        needles = (
            "model_not_supported",
            "not supported by any provider",
            "not deployed by any inference provider",
            "model is not supported",
            "no provider available",
            "provider unavailable",
        )
        return any(needle in body for needle in needles)

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
