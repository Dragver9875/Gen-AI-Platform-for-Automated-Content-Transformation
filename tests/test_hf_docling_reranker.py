from __future__ import annotations

from pathlib import Path

from providers.docling import DoclingAPIProvider
from providers.reranker import HostedReranker


class _FakeResponse:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data


def test_granite_docling_image_call_uses_hf_chat_payload(tmp_path: Path):
    image = tmp_path / "page.png"
    image.write_bytes(b"fake-image-bytes")

    provider = DoclingAPIProvider(
        "https://router.huggingface.co/v1/chat/completions",
        "hf_test",
        model="ibm-granite/granite-docling-258M",
    )
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update({"method": method, "url": url, **kwargs})
        return _FakeResponse({"choices": [{"message": {"content": "# Heading\n\nBody"}}]})

    provider.http.request = fake_request  # type: ignore[method-assign]
    result = provider.convert_file(image)
    markdown, doc_json = provider.document_payload(result)

    assert markdown.startswith("<!-- page:1 -->")
    assert "# Heading" in markdown
    assert doc_json["texts"][0]["prov"][0]["page_no"] == 1
    payload = seen["json"]
    assert payload["model"] == "ibm-granite/granite-docling-258M"
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["headers"]["Authorization"] == "Bearer hf_test"


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
