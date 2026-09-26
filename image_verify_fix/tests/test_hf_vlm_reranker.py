from __future__ import annotations

from providers.http import ProviderError
from providers.reranker import HostedReranker
from providers.vlm import VLMProvider


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_vlm_falls_back_when_primary_model_is_not_provider_served():
    provider = VLMProvider(
        "https://router.huggingface.co/v1/chat/completions",
        "hf_test",
        model="primary/not-served",
        fallback_models=("Qwen/Qwen2.5-VL-3B-Instruct",),
    )
    models = []

    def fake_request(method, url, **kwargs):
        model = kwargs["json"]["model"]
        models.append(model)
        if model == "primary/not-served":
            raise ProviderError(
                "HTTP 400 from provider: model_not_supported",
                status_code=400,
                response_text='{"code":"model_not_supported","message":"not supported by any provider"}',
            )
        return _FakeResponse({"choices": [{"message": {"content": "Visible document text."}}]})

    provider.http.request = fake_request  # type: ignore[method-assign]
    result = provider.describe_bytes(b"fake-png", media_type="image/png")
    assert result == "Visible document text."
    assert models == ["primary/not-served", "Qwen/Qwen2.5-VL-3B-Instruct"]


def test_hf_tei_reranker_contract_and_sorting():
    provider = HostedReranker(
        "https://reranker.endpoints.huggingface.cloud",
        "hf_test",
        api_style="hf_tei",
    )
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update({"method": method, "url": url, **kwargs})
        return _FakeResponse([
            {"index": 1, "score": 0.95},
            {"index": 0, "score": 0.25},
        ])

    provider.http.request = fake_request  # type: ignore[method-assign]
    docs = [{"text": "alpha", "id": "a"}, {"text": "beta", "id": "b"}]
    result = provider.rerank("query", docs, top_k=2)

    assert seen["url"].endswith("/rerank")
    assert seen["json"] == {"query": "query", "texts": ["alpha", "beta"], "raw_scores": False}
    assert seen["headers"]["Authorization"] == "Bearer hf_test"
    assert [d["id"] for d in result] == ["b", "a"]
