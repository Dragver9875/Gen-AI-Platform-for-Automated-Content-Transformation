from __future__ import annotations

from typing import Any

from providers.http import APIClient, ProviderError


class HarrierEmbeddingProvider:
    """Hosted Microsoft Harrier embedding adapter.

    Supported endpoint contracts:
      * hf: Hugging Face feature-extraction / Sentence-Transformers style.
      * openai: OpenAI-compatible embeddings.

    For Hugging Face serverless inference the recommended defaults are:
      model: microsoft/harrier-oss-v1-0.6b
      prompt_name: web_search_query for queries only
      normalize: true

    No model weights are loaded by this repository.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        api_style: str = "hf",
        model: str = "microsoft/harrier-oss-v1-0.6b",
        batch_size: int = 64,
        prompt_name: str | None = "web_search_query",
        normalize: bool = True,
        query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        timeout_s: float = 90.0,
        retries: int = 2,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.api_style = api_style.lower()
        self.model = model
        self.batch_size = batch_size
        self.prompt_name = prompt_name.strip() if isinstance(prompt_name, str) and prompt_name.strip() else None
        self.normalize = normalize
        self.query_instruction = query_instruction
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="harrier")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Retrieval documents must not receive the query instruction/prompt.
        return self._embed(texts, prompt_name=None)

    def embed_query(self, query: str) -> list[float]:
        if self.api_style == "hf" and self.prompt_name:
            # Harrier ships Sentence-Transformers prompts such as web_search_query.
            return self._embed([query], prompt_name=self.prompt_name)[0]

        # Fallback for non-HF/OpenAI-compatible embedding services that do not
        # expose Sentence-Transformers prompt_name semantics.
        instructed = f"Instruct: {self.query_instruction}\nQuery: {query}".strip()
        return self._embed([instructed], prompt_name=None)[0]

    def _embed(self, texts: list[str], *, prompt_name: str | None) -> list[list[float]]:
        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = self._payload(batch, prompt_name=prompt_name)
            response = self.http.request("POST", self.api_url, headers=self.headers, json=payload)
            vectors = self._parse_embeddings(response.json(), expected=len(batch))
            output.extend(vectors)
        return output

    def _payload(self, texts: list[str], *, prompt_name: str | None) -> dict[str, Any]:
        if self.api_style == "openai":
            payload: dict[str, Any] = {"input": texts}
            if self.model:
                payload["model"] = self.model
            return payload

        if self.api_style == "hf":
            payload = {
                "inputs": texts,
                "normalize": self.normalize,
            }
            if prompt_name:
                payload["prompt_name"] = prompt_name
            return payload

        raise ProviderError(f"Unsupported HARRIER_API_STYLE: {self.api_style}")

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
