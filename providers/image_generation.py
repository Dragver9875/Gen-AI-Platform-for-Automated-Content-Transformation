from __future__ import annotations

import base64
import io
from typing import Any

from providers.http import APIClient, ProviderError


class HostedImageGenerationProvider:
    """Creative image decoder.

    Default mode is ``hf_hub``: use Hugging Face InferenceClient with the shared
    HF token and provider="auto" for FLUX.1-schnell. This avoids another endpoint
    URL/credential. Custom HF/OpenAI-compatible endpoints remain supported.
    """

    def __init__(
        self,
        api_url: str | None,
        api_key: str,
        *,
        model: str | None = None,
        api_style: str = "hf_hub",
        provider: str = "auto",
        timeout_s: float = 180.0,
        retries: int = 2,
    ):
        self.api_url = api_url.rstrip("/") if api_url else None
        self.api_key = api_key
        self.model = model or "black-forest-labs/FLUX.1-schnell"
        self.api_style = api_style.lower()
        self.provider = provider or "auto"
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="image_generation")

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*,application/json",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
    ) -> tuple[bytes, str]:
        if self.api_style == "hf_hub":
            try:
                from huggingface_hub import InferenceClient

                client = InferenceClient(api_key=self.api_key, provider=self.provider)
                image = client.text_to_image(
                    prompt,
                    model=self.model,
                    width=width,
                    height=height,
                    negative_prompt=negative_prompt or None,
                )
                output = io.BytesIO()
                image.save(output, format="PNG")
                return output.getvalue(), "image/png"
            except Exception as exc:
                raise ProviderError(
                    f"Hugging Face creative-image inference failed for {self.model}: {exc}"
                ) from exc

        if not self.api_url:
            raise ProviderError(f"IMAGE_GEN_API_URL is required for IMAGE_GEN_API_STYLE={self.api_style}")

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
