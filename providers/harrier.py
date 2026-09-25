from __future__ import annotations

from typing import Any, Iterable

from providers.http import APIClient, ProviderError


class HarrierEmbeddingProvider:
    """Hosted Microsoft Harrier embedding adapter.

    Supported endpoint contracts:
      * hf: Hugging Face Sentence-Transformers/feature-extraction style: {"inputs": [...]}
      * openai: OpenAI-compatible embeddings: {"input": [...], "model": "..."}

    No model weights are loaded by this repository.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        api_style: str = "hf",
        batch_size: int = 64,
        query_instruction: str = "Retrieve passages from the supplied documents that answer the user's query:",
        timeout_s: float = 90.0,
        retries: int = 2,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.api_style = api_style
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.http = APIClient(timeout_s=timeout_s, retries=retries)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, query: str) -> list[float]:
        instructed = f"{self.query_instruction}\n\n{query}".strip()
        return self._embed([instructed])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = self._payload(batch)
            response = self.http.request("POST", self.api_url, headers=self.headers, json=payload)
            vectors = self._parse_embeddings(response.json(), expected=len(batch))
            output.extend(vectors)
        return output

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        if self.api_style == "openai":
            return {"input": texts}
        return {"inputs": texts}

    @staticmethod
    def _parse_embeddings(data: Any, *, expected: int) -> list[list[float]]:
        # OpenAI-compatible response.
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            vectors = [item["embedding"] for item in data["data"]]
        # Common HF custom / TEI wrappers.
        elif isinstance(data, dict) and "embeddings" in data:
            vectors = data["embeddings"]
        elif isinstance(data, list):
            vectors = data
        else:
            raise ProviderError(f"Unsupported embedding response shape: {type(data)!r}")

        # Some feature-extraction endpoints wrap batch vectors one level deeper.
        if len(vectors) == 1 and expected > 1 and isinstance(vectors[0], list) and vectors[0] and isinstance(vectors[0][0], list):
            vectors = vectors[0]

        if len(vectors) != expected:
            raise ProviderError(f"Expected {expected} embeddings, received {len(vectors)}")
        return [[float(v) for v in vector] for vector in vectors]
