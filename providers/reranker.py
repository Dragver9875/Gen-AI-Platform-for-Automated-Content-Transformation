from __future__ import annotations

from typing import Any

from providers.http import APIClient, ProviderError


class HostedReranker:
    """Hugging Face-compatible reranking endpoint.

    Default contract follows Hugging Face Text Embeddings Inference (TEI):
      POST /rerank
      {"query": str, "texts": [str, ...], "raw_scores": false}

    Dedicated Hugging Face Inference Endpoints can serve compatible rerankers
    such as BAAI/bge-reranker-v2-m3 or Alibaba-NLP/gte-multilingual-reranker-base
    while authenticating with the same HF_TOKEN used by the rest of the stack.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        api_style: str = "hf_tei",
        timeout_s: float = 90.0,
        retries: int = 2,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.api_style = api_style.lower()
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="reranker")

    def rerank(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        texts = [d["text"] for d in docs]
        if self.api_style == "hf_tei":
            endpoint = self.api_url if self.api_url.endswith("/rerank") else f"{self.api_url}/rerank"
            payload = {"query": query, "texts": texts, "raw_scores": False}
        elif self.api_style == "generic":
            endpoint = self.api_url
            payload = {"query": query, "documents": texts, "top_k": top_k}
        else:
            raise ProviderError(f"Unsupported RERANKER_API_STYLE: {self.api_style}")

        response = self.http.request("POST", endpoint, headers=headers, json=payload)
        data = response.json()
        results = data.get("results") if isinstance(data, dict) else data
        if not isinstance(results, list):
            raise ProviderError("Unsupported reranker response shape")

        reranked: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if idx is None:
                idx = item.get("corpus_id")
            if idx is None:
                continue
            idx = int(idx)
            if idx < 0 or idx >= len(docs):
                continue
            doc = dict(docs[idx])
            doc["rerank_score"] = float(item.get("score", item.get("relevance_score", 0.0)))
            reranked.append(doc)

        reranked.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
        return reranked[:top_k]
