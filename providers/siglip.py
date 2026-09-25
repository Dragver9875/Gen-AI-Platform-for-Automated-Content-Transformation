from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

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
    """Hosted SigLIP routing adapter.

    The recommended endpoint is a Hugging Face custom handler accepting:
      {"image_base64": "...", "candidate_labels": [...]}
    and returning either {"labels": [{"label": ..., "score": ...}]} or an HF-style list.
    """

    def __init__(self, api_url: str, api_key: str, *, timeout_s: float = 90.0, retries: int = 2):
        self.api_url = api_url
        self.api_key = api_key
        self.http = APIClient(timeout_s=timeout_s, retries=retries)

    def classify(self, path: str | Path, labels: list[str] | None = None) -> list[dict[str, float | str]]:
        data = Path(path).read_bytes()
        payload = {
            "image_base64": base64.b64encode(data).decode("ascii"),
            "candidate_labels": labels or DEFAULT_LABELS,
        }
        response = self.http.request(
            "POST",
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        return self._parse(response.json())

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
