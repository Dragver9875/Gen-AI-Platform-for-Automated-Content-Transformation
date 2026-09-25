from __future__ import annotations

from typing import Any

from providers.http import APIClient, ProviderError


class HostedReranker:
    """Optional hosted ranking endpoint.

    Expected request: {"query": str, "documents": [str, ...], "top_k": int}
    Accepted responses: {"results": [{"index": int, "score": float}]} or a list in the same form.
    """

    def __init__(self, api_url: str, api_key: str | None, *, timeout_s: float = 90.0, retries: int = 2):
        self.api_url = api_url
        self.api_key = api_key
        self.http = APIClient(timeout_s=timeout_s, retries=retries, provider_name="reranker")

    def rerank(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.http.request(
            "POST",
            self.api_url,
            headers=headers,
            json={"query": query, "documents": [d["text"] for d in docs], "top_k": top_k},
        )
        data = response.json()
        results = data.get("results") if isinstance(data, dict) else data
        if not isinstance(results, list):
            raise ProviderError("Unsupported reranker response shape")
        reranked: list[dict[str, Any]] = []
        for item in results:
            idx = int(item["index"])
            doc = dict(docs[idx])
            doc["rerank_score"] = float(item.get("score", 0.0))
            reranked.append(doc)
        return reranked[:top_k]
