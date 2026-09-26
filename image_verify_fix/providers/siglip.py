from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient

from core.telemetry import TelemetryEvent, record_event
from providers.http import APIClient, ProviderError


DEFAULT_LABELS = [
    "document page",
    "screenshot",
    "chart",
    "diagram",
    "map",
    "photograph",
    "other visual",
]


class SigLIPRoutingProvider:
    """SigLIP zero-shot image router.

    Default mode (``api_url=None``) uses ``huggingface_hub.InferenceClient`` with
    ``HF_TOKEN`` and a model ID, so no dedicated endpoint URL is required.

    A custom endpoint can still be supplied. Its compatibility contract is:
      request: {"image_base64": "...", "candidate_labels": [...]}
      response: {"labels": [{"label": ..., "score": ...}]} or an HF-style list.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "google/siglip-so400m-patch14-384",
        api_url: str | None = None,
        timeout_s: float = 90.0,
        retries: int = 2,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.retries = retries
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="siglip")
        self._hf_client = None if api_url else InferenceClient(token=api_key, timeout=timeout_s)

    def classify(self, path: str | Path, labels: list[str] | None = None) -> list[dict[str, float | str]]:
        data = Path(path).read_bytes()
        candidate_labels = labels or DEFAULT_LABELS
        if self.api_url:
            payload = {
                "image_base64": base64.b64encode(data).decode("ascii"),
                "candidate_labels": candidate_labels,
            }
            response = self.http.request(
                "POST",
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            return self._parse(response.json())

        started = time.perf_counter()
        try:
            outputs = self._hf_client.zero_shot_image_classification(  # type: ignore[union-attr]
                data,
                candidate_labels=candidate_labels,
                model=self.model,
            )
            parsed = [
                {"label": str(item.label), "score": float(item.score)}
                for item in outputs
            ]
            record_event(TelemetryEvent(
                provider="siglip",
                operation="zero_shot_image_classification",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                success=True,
                attempts=1,
                metadata={"model": self.model},
            ))
            return sorted(parsed, key=lambda x: float(x["score"]), reverse=True)
        except Exception as exc:
            record_event(TelemetryEvent(
                provider="siglip",
                operation="zero_shot_image_classification",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                success=False,
                attempts=1,
                metadata={"model": self.model, "error": str(exc)},
            ))
            raise ProviderError(f"Hugging Face SigLIP inference failed: {exc}") from exc

    @staticmethod
    def _parse(data: Any) -> list[dict[str, float | str]]:
        if isinstance(data, dict) and "scores" in data and "labels" in data and isinstance(data["labels"], list):
            data = [{"label": l, "score": s} for l, s in zip(data["labels"], data["scores"])]
        elif isinstance(data, dict) and "labels" in data:
            data = data["labels"]
        if not isinstance(data, list):
            raise ProviderError("Unsupported SigLIP response shape")
        parsed: list[dict[str, float | str]] = []
        for item in data:
            if isinstance(item, dict) and "label" in item:
                parsed.append({"label": str(item["label"]), "score": float(item.get("score", 0.0))})
        return sorted(parsed, key=lambda x: float(x["score"]), reverse=True)
